"""
Ensemble Weight Tuner — try every model subset × calibration method.

What it does
------------
1. Loads data, builds the full feature matrix once (the slow step).
2. For each **model combination** (subsets of LR, RF, XGB, Poisson):
   - Trains the sub-models.
   - Optimises ensemble weights on the validation set.
   - Tries each calibration method (none, Platt, isotonic).
   - Evaluates on the held-out test set: log-loss, Brier, ECE, accuracy.
3. Ranks every (combination, calibration) pair and reports the winner.
4. Optionally updates ``config.ensemble`` defaults in ``config.py``.

Usage
-----
    python scripts/tune_ensemble.py
    python scripts/tune_ensemble.py --data-path data/processed/results_clean.csv
    python scripts/tune_ensemble.py --quick               # faster grid step (0.1)
    python scripts/tune_ensemble.py --no-calibrate         # skip calibration variants
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.feature_engineering import build_features, train_val_test_split
from src.calibration import _fit_calibrators, calibration_curve as cal_curve
from src.poisson_model import PoissonModel

logger = logging.getLogger("tune_ensemble")

# ── Constants ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "ensemble_tuning"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "results_clean.csv"

# Model name ↔ scikit-learn-compatible class
MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

# Default hyper-parameters per model (pulled from config)
LR_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": config.train.seed,
    "class_weight": "balanced",
    "C": 1.0,
    "verbose": 0,
}

RF_PARAMS = {
    "n_estimators": config.train.n_estimators,
    "max_depth": config.train.max_depth,
    "min_samples_leaf": config.train.min_samples_leaf,
    "random_state": config.train.seed,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_estimators": config.train.n_estimators,
    "max_depth": config.train.max_depth,
    "learning_rate": config.train.learning_rate,
    "subsample": config.train.subsample,
    "colsample_bytree": config.train.colsample_bytree,
    "reg_lambda": config.train.reg_lambda,
    "reg_alpha": config.train.reg_alpha,
    "random_state": config.train.seed,
    "n_jobs": -1,
}


# ── Tuning result container ─────────────────────────────


@dataclass
class TrialResult:
    """Results for a single (model_set, calibration) trial."""

    model_set: tuple[str, ...]
    calibration: str  # "none" | "platt" | "isotonic"
    weight_step: float = 0.05

    # Test metrics
    test_log_loss: float = 0.0
    test_brier: float = 0.0
    test_accuracy: float = 0.0
    ece: float = 0.0

    # Ensemble weights
    weights: dict[str, float] = field(default_factory=dict)

    # Training duration
    train_duration: float = 0.0

    # Number of matches in each split
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    @property
    def model_set_name(self) -> str:
        return "+".join(
            {"logistic_regression": "LR",
             "random_forest": "RF",
             "xgboost": "XGB",
             "poisson": "Poi"}.get(m, m)
            for m in self.model_set
        )

    @property
    def label(self) -> str:
        cal = {"none": "raw", "platt": "Platt", "isotonic": "Iso"}[self.calibration]
        return f"{self.model_set_name} + {cal}"


# ═══════════════════════════════════════════════════════════
#  Core evaluation function
# ═══════════════════════════════════════════════════════════

def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute log-loss, Brier, accuracy, and ECE for probability predictions."""
    logloss = float(sk_log_loss(y_true, probs))

    n = len(y_true)
    y_onehot = np.zeros((n, 3))
    for i, v in enumerate(y_true):
        y_onehot[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == y_true))

    # ECE
    curve = cal_curve(y_true, probs, n_bins=10)
    total_counts = curve["counts"].sum()
    if total_counts > 0:
        ece = float(np.mean(
            curve["counts"] / total_counts
            * np.abs(curve["accuracies"] - curve["confidences"])
        ))
    else:
        ece = 0.0

    return {"log_loss": logloss, "brier": brier, "accuracy": accuracy, "ece": ece}


def _train_sub_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Any:
    """Train a single sub-model and return it."""
    col_means = X_train.mean().fillna(0)
    X_train_clean = X_train.fillna(col_means)

    if model_name == "logistic_regression":
        model = LogisticRegression(**LR_PARAMS)
        model.fit(X_train_clean, y_train)
        return model

    if model_name == "random_forest":
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train_clean, y_train)
        return model

    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(**XGB_PARAMS)
        eval_set = [(X_train_clean, y_train)]
        if X_val is not None and y_val is not None:
            X_val_clean = X_val.fillna(col_means)
            eval_set.append((X_val_clean, y_val))
        model.fit(X_train_clean, y_train, eval_set=eval_set, verbose=False)
        return model

    raise ValueError(f"Unknown model '{model_name}'")


def _train_poisson(df_train: pd.DataFrame) -> PoissonModel:
    """Train a Poisson model on raw match data."""
    pm = PoissonModel(min_matches=config.poisson.min_matches,
                       max_goals=config.poisson.max_goals)
    pm.fit(df_train)
    return pm


def _ml_predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    col_means = X.mean().fillna(0)
    return model.predict_proba(X.fillna(col_means))


def _poisson_predict_proba(pm: PoissonModel | None, df_raw: pd.DataFrame | None) -> np.ndarray:
    if pm is None or df_raw is None or df_raw.empty:
        n = len(df_raw) if df_raw is not None else 0
        if n == 0:
            return np.zeros((0, 3))
        return np.full((n, 3), 1.0 / 3.0)
    preds_df = pm.predict_matches(df_raw)
    n = len(preds_df)
    probs = np.zeros((n, 3))
    if "away_win_prob" in preds_df.columns:
        probs[:, 0] = preds_df["away_win_prob"].values
    if "draw_prob" in preds_df.columns:
        probs[:, 1] = preds_df["draw_prob"].values
    if "home_win_prob" in preds_df.columns:
        probs[:, 2] = preds_df["home_win_prob"].values
    row_sums = probs.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    probs = probs / row_sums[:, np.newaxis]
    return probs


def _optimise_weights(
    preds: dict[str, np.ndarray],
    y_val: pd.Series,
    step: float = 0.05,
) -> dict[str, float]:
    """Grid search over weight combinations to minimise log-loss."""
    model_names = list(preds.keys())
    n_models = len(model_names)
    if n_models == 0:
        return {}
    if n_models == 1:
        return {model_names[0]: 1.0}

    n_steps = int(round(1.0 / step))
    best_loss = float("inf")
    best_weights: list[float] = []
    seen: set[tuple[float, ...]] = set()

    y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

    for raw in itertools.product(range(n_steps + 1), repeat=n_models):
        total = sum(raw)
        if total == 0:
            continue
        norm = tuple(w / total for w in raw)
        if norm in seen:
            continue
        seen.add(norm)

        weighted = np.zeros((len(y_val), 3))
        for w, name in zip(norm, model_names):
            if w > 0:
                weighted += w * preds[name]
        row_sums = weighted.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weighted /= row_sums[:, np.newaxis]

        loss = float(sk_log_loss(y_val_arr, weighted))
        if loss < best_loss:
            best_loss = loss
            best_weights = list(norm)

    return dict(zip(model_names, best_weights))


def _calibrate_probs(cal_method: str, val_probs: np.ndarray, y_val: np.ndarray,
                      test_probs: np.ndarray) -> np.ndarray:
    """Apply calibration and return calibrated test probabilities."""
    if cal_method == "none":
        return test_probs

    calibrators = _fit_calibrators(val_probs, y_val, 3, cal_method)
    cal_test = np.zeros_like(test_probs)
    for c in range(3):
        calibrator = calibrators[c]
        if cal_method == "platt":
            p = np.clip(test_probs[:, c], 1e-7, 1 - 1e-7)
            X_c = np.log(p / (1.0 - p)).reshape(-1, 1)
            cal_test[:, c] = calibrator.predict_proba(X_c)[:, 1]
        else:
            cal_test[:, c] = calibrator.transform(test_probs[:, c])
    row_sums = cal_test.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cal_test /= row_sums[:, np.newaxis]
    return cal_test


# ═══════════════════════════════════════════════════════════
#  Model combination definitions
# ═══════════════════════════════════════════════════════════

ALL_MODELS = ("logistic_regression", "random_forest", "xgboost", "poisson")

# Subsets to try (sorted by size, includes all possibilities)
MODEL_SETS: list[tuple[str, ...]] = [
    # Single models
    ("logistic_regression",),
    ("random_forest",),
    ("xgboost",),
    ("poisson",),
    # Pairs — tree combos
    ("logistic_regression", "random_forest"),
    ("logistic_regression", "xgboost"),
    ("random_forest", "xgboost"),
    # Pairs — with poisson
    ("logistic_regression", "poisson"),
    ("random_forest", "poisson"),
    ("xgboost", "poisson"),
    # Triples — without poisson
    ("logistic_regression", "random_forest", "xgboost"),
    # Triples — with poisson
    ("logistic_regression", "random_forest", "poisson"),
    ("logistic_regression", "xgboost", "poisson"),
    ("random_forest", "xgboost", "poisson"),
    # All four
    ("logistic_regression", "random_forest", "xgboost", "poisson"),
]

# Faster subset: only the most informative combinations
MODEL_SETS_FAST: list[tuple[str, ...]] = [
    # Key singles
    ("logistic_regression",),
    ("xgboost",),
    # Best pairs from experiments
    ("logistic_regression", "xgboost"),
    ("random_forest", "xgboost"),
    ("xgboost", "poisson"),
    # Best triple (no poisson — XGB+RF+LR is the tree+linear combo)
    ("logistic_regression", "random_forest", "xgboost"),
    # Triples with poisson
    ("random_forest", "xgboost", "poisson"),
    ("logistic_regression", "xgboost", "poisson"),
    # All four
    ("logistic_regression", "random_forest", "xgboost", "poisson"),
]


# ═══════════════════════════════════════════════════════════
#  Tuning loop
# ═══════════════════════════════════════════════════════════


def run_tuning(
    data_path: str | Path | None = None,
    weight_step: float = 0.05,
    calibrate: bool = True,
    model_sets: list[tuple[str, ...]] | None = None,
) -> list[TrialResult]:
    """Run the full ensemble tuning grid.

    Returns sorted list of TrialResult (best first by log-loss).
    """
    if model_sets is None:
        model_sets = MODEL_SETS

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ═══════════════════════════════════════════════════════
    #  Load data and build features (one-time cost)
    # ═══════════════════════════════════════════════════════
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    if not path.exists():
        fallback = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Data not found at {path}")

    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
        # Drop rows with unmapped results (incomplete matches, null results)
        bad_mask = df["target"] == -1
        n_bad = bad_mask.sum()
        if n_bad > 0:
            logger.warning("Dropping %d rows with unmapped result (target = -1)", n_bad)
            df = df[~bad_mask].copy()

    logger.info("Building features …")
    t0 = time.time()
    X, y = build_features(df, is_training=True)
    logger.info("Feature matrix: %d × %d (%.1f s)", X.shape[0], X.shape[1], time.time() - t0)

    # Align raw df with feature matrix
    df_sorted = df.loc[X.index] if hasattr(X, "index") else df

    # Chronological split
    splits = train_val_test_split(X, y)
    n_train, n_val, n_test = len(splits["X_train"]), len(splits["X_val"]), len(splits["X_test"])

    train_end = n_train
    val_end = train_end + n_val
    df_train = df_sorted.iloc[:train_end] if len(df_sorted) >= train_end else df_sorted
    df_val = df_sorted.iloc[train_end:val_end] if len(df_sorted) >= val_end else pd.DataFrame()
    df_test = df_sorted.iloc[val_end:] if len(df_sorted) > val_end else pd.DataFrame()

    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    print(f"\n  Dataset: {X.shape[0]} matches × {X.shape[1]} features")
    print(f"  Split:   train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print()

    # ═══════════════════════════════════════════════════════
    #  Train all possible sub-models ONCE so we can mix & match
    # ═══════════════════════════════════════════════════════
    trained_models: dict[str, Any] = {}
    val_preds_cache: dict[str, np.ndarray] = {}
    test_preds_cache: dict[str, np.ndarray] = {}

    for model_name in ALL_MODELS:
        t0 = time.time()
        if model_name == "poisson":
            model = _train_poisson(df_train)
            trained_models["poisson"] = model
            val_preds_cache["poisson"] = _poisson_predict_proba(model, df_val)
            test_preds_cache["poisson"] = _poisson_predict_proba(model, df_test)
        else:
            model = _train_sub_model(model_name, X_train, y_train, X_val, y_val)
            trained_models[model_name] = model
            val_preds_cache[model_name] = _ml_predict_proba(model, X_val)
            test_preds_cache[model_name] = _ml_predict_proba(model, X_test)
        logger.info("  Trained %s in %.1f s", model_name, time.time() - t0)

    print(f"  All {len(trained_models)} base models trained.\n")

    # ═══════════════════════════════════════════════════════
    #  Grid: model_set × calibration
    # ═══════════════════════════════════════════════════════
    calibration_methods = ["none", "platt", "isotonic"] if calibrate else ["none"]
    results: list[TrialResult] = []
    total_trials = len(model_sets) * len(calibration_methods)
    trial_no = 0

    for model_set in model_sets:
        # --- Optimise weights for this model set ---
        available = [m for m in model_set if m in trained_models]
        if not available:
            continue

        val_preds = {m: val_preds_cache[m] for m in available}
        t0 = time.time()
        weights = _optimise_weights(val_preds, y_val, step=weight_step)
        weight_time = time.time() - t0

        # Compute validation ensemble probs
        weighted_val = np.zeros((n_val, 3))
        for name, w in weights.items():
            if w > 0:
                weighted_val += w * val_preds[name]
        row_sums = weighted_val.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weighted_val /= row_sums[:, np.newaxis]

        # Compute test ensemble probs (raw)
        test_preds = {m: test_preds_cache[m] for m in available}
        weighted_test = np.zeros((n_test, 3))
        for name, w in weights.items():
            if w > 0:
                weighted_test += w * test_preds[name]
        row_sums = weighted_test.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weighted_test /= row_sums[:, np.newaxis]

        y_val_arr = y_val.values if hasattr(y_val, "values") else y_val
        y_test_arr = y_test.values if hasattr(y_test, "values") else y_test

        for cal_method in calibration_methods:
            trial_no += 1
            t0 = time.time()

            cal_test = _calibrate_probs(cal_method, weighted_val, y_val_arr, weighted_test)
            metrics = evaluate_probs(y_test_arr, cal_test)

            result = TrialResult(
                model_set=available,
                calibration=cal_method,
                weight_step=weight_step,
                test_log_loss=metrics["log_loss"],
                test_brier=metrics["brier"],
                test_accuracy=metrics["accuracy"],
                ece=metrics["ece"],
                weights=weights,
                train_duration=weight_time + (time.time() - t0),
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
            )
            results.append(result)

            print(f"  [{trial_no:>3d}/{total_trials}]  {result.label:<28s}  "
                  f"loss={result.test_log_loss:.4f}  brier={result.test_brier:.4f}  "
                  f"acc={result.test_accuracy:.2%}  ece={result.ece:.4f}")

    # ═══════════════════════════════════════════════════════
    #  Rank results
    # ═══════════════════════════════════════════════════════
    results.sort(key=lambda r: r.test_log_loss)

    total_time = time.time() - t_start
    print(f"\n  {'='*80}")
    print(f"  RANKING BY LOG-LOSS  ({total_trials} trials, {total_time:.0f}s)")
    print(f"  {'='*80}")
    print(f"  {'Rank':<6} {'Configuration':<30s} {'Log-Loss':<10} {'Brier':<10} {'Accuracy':<10} {'ECE':<8}")
    print(f"  {'-'*74}")
    for i, r in enumerate(results[:15]):
        rank = i + 1
        marker = " <<< WINNER" if i == 0 else ""
        print(f"  {rank:<6} {r.label:<30s} {r.test_log_loss:<10.4f} {r.test_brier:<10.4f} {r.test_accuracy:<10.2%} {r.ece:<8.4f}{marker}")

    # Show the bottom few too
    if len(results) > 15:
        print(f"  {'...':>6} {'...':<30s} {'...':<10} {'...':<10} {'...':<10} {'...':<8}")
        for r in results[-3:]:
            print(f"  {'WORST':<6} {r.label:<30s} {r.test_log_loss:<10.4f} {r.test_brier:<10.4f} {r.test_accuracy:<10.2%} {r.ece:<8.4f}")

    # ═══════════════════════════════════════════════════════
    #  Save report
    # ═══════════════════════════════════════════════════════
    report_lines = [_build_report(results, total_time)]
    report_path = REPORTS_DIR / f"tuning_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report_lines[0], encoding="utf-8")
    (REPORTS_DIR / "latest_report.txt").write_text(report_lines[0], encoding="utf-8")
    print(f"\n  Full report: {report_path}")

    return results


# ═══════════════════════════════════════════════════════════
#  Report builder
# ═══════════════════════════════════════════════════════════


def _build_report(results: list[TrialResult], total_duration: float) -> str:
    lines: list[str] = []
    sep = "=" * 90
    lines.append("")
    lines.append(sep)
    lines.append("  ENSEMBLE TUNING REPORT".center(88))
    lines.append(sep)
    lines.append(f"  Date:      {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Duration:  {total_duration:.0f}s")
    lines.append(f"  Trials:    {len(results)}")
    if results:
        r0 = results[0]
        lines.append(f"  Dataset:   {r0.n_train:,} train / {r0.n_val:,} val / {r0.n_test:,} test")
    lines.append("")

    # ── Top 10 ──
    lines.append(f"  {'─'*88}")
    lines.append("  TOP 10 CONFIGURATIONS (by log-loss)")
    lines.append(f"  {'─'*88}")
    lines.append(f"  {'Rank':<6} {'Configuration':<32s} {'Log-Loss':<10} {'Brier':<10} {'Accuracy':<10} {'ECE':<9} {'Weights'}")
    lines.append(f"  {'-'*84}")
    for i, r in enumerate(results[:10]):
        rank = i + 1
        w_str = " + ".join(f"{n.split('_')[0] if '_' in n else n[:3]}={v:.2f}" for n, v in sorted(r.weights.items()))
        marker = " ★ WINNER" if i == 0 else ""
        lines.append(f"  {rank:<6} {r.label:<32s} {r.test_log_loss:<10.4f} {r.test_brier:<10.4f} {r.test_accuracy:<10.2%} {r.ece:<9.4f} {w_str}{marker}")
    lines.append("")

    # ── Best per model set (best calibration) ──
    lines.append(f"  {'─'*88}")
    lines.append("  BEST PER MODEL SET")
    lines.append(f"  {'─'*88}")
    lines.append(f"  {'Model Set':<30s} {'Best Cal':<12} {'Log-Loss':<10} {'Brier':<10} {'Accuracy':<10} {'ECE':<8}")
    lines.append(f"  {'-'*80}")
    by_set: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        by_set[r.model_set_name].append(r)
    for ms_name, trials in sorted(by_set.items()):
        best = min(trials, key=lambda t: t.test_log_loss)
        lines.append(f"  {ms_name:<30s} {best.calibration:<12s} {best.test_log_loss:<10.4f} {best.test_brier:<10.4f} {best.test_accuracy:<10.2%} {best.ece:<8.4f}")
    lines.append("")

    # ── Summary of calibration methods ──
    lines.append(f"  {'─'*88}")
    lines.append("  CALIBRATION METHOD COMPARISON (averaged across all model sets)")
    lines.append(f"  {'─'*88}")
    for cal in ["none", "platt", "isotonic"]:
        cal_trials = [r for r in results if r.calibration == cal]
        if cal_trials:
            avg_loss = np.mean([r.test_log_loss for r in cal_trials])
            avg_brier = np.mean([r.test_brier for r in cal_trials])
            avg_acc = np.mean([r.test_accuracy for r in cal_trials])
            avg_ece = np.mean([r.ece for r in cal_trials])
            lines.append(f"  {cal:<12s}  loss={avg_loss:.4f}  brier={avg_brier:.4f}  acc={avg_acc:.2%}  ece={avg_ece:.4f}  (n={len(cal_trials)})")
    lines.append("")

    # ── Winner details ──
    if results:
        w = results[0]
        lines.append(f"  {'─'*88}")
        lines.append("  WINNER DETAILS")
        lines.append(f"  {'─'*88}")
        lines.append(f"  Model set:      {w.model_set}")
        lines.append(f"  Calibration:    {w.calibration}")
        lines.append(f"  Weight step:    {w.weight_step}")
        lines.append(f"  Log-loss:       {w.test_log_loss:.4f}")
        lines.append(f"  Brier:          {w.test_brier:.4f}")
        lines.append(f"  Accuracy:       {w.test_accuracy:.2%}")
        lines.append(f"  ECE:            {w.ece:.4f}")
        lines.append(f"  Weights:        {w.weights}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune ensemble model weights across model subsets and calibration methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to match data CSV (default: data/processed/results_clean.csv)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use coarser grid step (0.1) for faster tuning",
    )
    parser.add_argument(
        "--no-calibrate", action="store_true",
        help="Skip calibration variants (only test raw ensemble)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use reduced model set (fewer combinations) for quicker runs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    weight_step = 0.1 if args.quick else 0.05
    calibrate = not args.no_calibrate

    model_sets = MODEL_SETS_FAST if args.fast else None

    try:
        results = run_tuning(
            data_path=args.data_path,
            weight_step=weight_step,
            calibrate=calibrate,
            model_sets=model_sets,
        )
        print("\n  Done. Best configuration:")
        r = results[0]
        print(f"    Model set:     {r.model_set}")
        print(f"    Calibration:   {r.calibration}")
        print(f"    Weights:       {r.weights}")
        print(f"    Test log-loss: {r.test_log_loss:.4f}")
        print(f"    Test Brier:    {r.test_brier:.4f}")
        print(f"    Test accuracy: {r.test_accuracy:.2%}")
        print()
        return 0
    except Exception as e:
        logger.error("Tuning failed: %s", e, exc_info=True)
        print(f"\n  ❌ Tuning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
