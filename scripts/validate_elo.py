#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Elo Rating System — Validation & Hyperparameter Tuning                   ║
║                                                                           ║
║  Tunes Elo parameters (k, home_advantage, regress_to_mean, draw_k) via    ║
║  grid search on a validation split, then compares the best Elo model      ║
║  against a logistic regression baseline.                                  ║
║                                                                           ║
║  Key differences from Poisson/DC validation:                              ║
║  - Elo is an online/sequential system, not batch-fitted                   ║
║  - Hyperparameter grid search is the primary focus                        ║
║  - Baseline: LR using Elo difference as the sole feature                  ║
║                                                                           ║
║  Outputs:                                                                 ║
║  - models/elo_model.joblib                                               ║
║  - reports/elo_validation_{timestamp}.json                               ║
║  - reports/elo_vs_baseline_{timestamp}.json                              ║
║                                                                           ║
║  Usage:                                                                   ║
║      python scripts/validate_elo.py                                       ║
║      python scripts/validate_elo.py --data data/raw/results.csv           ║
║      python scripts/validate_elo.py --quiet                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from config import config
from src.elo import EloSystem

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

_DEFAULT_TRAIN_END = "2022-01-01"      # train: before 2022
_DEFAULT_VAL_END = "2023-01-01"         # val: 2022-2023
_DEFAULT_TEST_END = "2026-12-31"        # test: 2023-2026

_REQUIRED_COLS = ["date", "home_team", "away_team", "result", "home_goals", "away_goals"]

_MODEL_DIR = PROJECT_ROOT / "models"
_REPORT_DIR = PROJECT_ROOT / "reports"

# Hyperparameter grid
PARAM_GRID = {
    "k": [16, 24, 32, 40, 48],
    "home_advantage": [50, 75, 100, 125],
    "regress_to_mean": [True, False],
    "draw_k": [0.15, 0.20, 0.25, 0.30],
}

# Parameter names for human-readable output
_PARAM_LABELS = {
    "k": "K-factor",
    "home_advantage": "Home Adv (Elo pts)",
    "regress_to_mean": "Season Regression",
    "draw_k": "Draw Prob Constant",
}


# ═══════════════════════════════════════════════════════════
#  Data loading & splitting
# ═══════════════════════════════════════════════════════════


def load_data(path: str | Path, min_date: str = "2010-01-01") -> pd.DataFrame:
    """Load match data from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_dates = df["date"].copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    if df["date"].isna().sum() > len(df) * 0.5:
        logger.warning("ISO format failed -- retrying with dayfirst=True")
        df["date"] = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")

    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(subset=["date"])

    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)].copy()

    df = df[df["result"].notna() & df["result"].isin(["H", "D", "A"])].copy()

    logger.info("Loaded %d matches from %s to %s", len(df),
                df["date"].min().strftime("%Y-%m-%d"),
                df["date"].max().strftime("%Y-%m-%d"))
    return df


def chronological_split(
    df: pd.DataFrame,
    train_end: str = _DEFAULT_TRAIN_END,
    val_end: str = _DEFAULT_VAL_END,
    test_end: str = _DEFAULT_TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train / val / test."""
    t_train = pd.Timestamp(train_end)
    t_val = pd.Timestamp(val_end)
    t_test = pd.Timestamp(test_end)

    train = df[df["date"] < t_train].copy()
    val = df[(df["date"] >= t_train) & (df["date"] < t_val)].copy()
    test = df[(df["date"] >= t_val) & (df["date"] <= t_test)].copy()

    logger.info("Split: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def get_targets(df: pd.DataFrame) -> np.ndarray:
    mapping = {"A": 0, "D": 1, "H": 2}
    result = df["result"].map(mapping).fillna(-1).values.astype(int)
    n_bad = int((result == -1).sum())
    if n_bad > 0:
        logger.warning("get_targets: %d unmapped values", n_bad)
    return result


# ═══════════════════════════════════════════════════════════
#  Metrics (shared with other validations)
# ═══════════════════════════════════════════════════════════


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, name: str = "Model"):
    from sklearn.metrics import log_loss as sk_ll
    pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred == y_true))
    ll = float(sk_ll(y_true, probs))
    y_oh = np.zeros((len(y_true), 3))
    for i, v in enumerate(y_true):
        if 0 <= v <= 2:
            y_oh[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))
    return {"accuracy": round(accuracy, 4), "log_loss": round(ll, 4),
            "brier_score": round(brier, 4), "n": int(len(y_true))}


def compute_btts_metrics(hg: np.ndarray, ag: np.ndarray, pred_probs: np.ndarray):
    actual = ((hg > 0) & (ag > 0)).astype(float)
    pred = (pred_probs > 0.5).astype(float)
    return {"btts_accuracy": round(float(np.mean(pred == actual)), 4),
            "btts_brier": round(float(np.mean((pred_probs - actual) ** 2)), 4)}


def compute_ou_metrics(hg: np.ndarray, ag: np.ndarray, pred_probs: np.ndarray, threshold: float = 2.5):
    actual = ((hg + ag) > threshold).astype(float)
    pred = (pred_probs > 0.5).astype(float)
    k = f"over_under_{threshold:.1f}".replace(".", "_")
    return {f"{k}_accuracy": round(float(np.mean(pred == actual)), 4),
            f"{k}_brier": round(float(np.mean((pred_probs - actual) ** 2)), 4)}


def evaluate_model(elo: EloSystem, df_test: pd.DataFrame, name: str = "Elo"):
    """Evaluate Elo on a test set: walk through predicting then updating."""
    hg = df_test["home_goals"].values.astype(float)
    ag = df_test["away_goals"].values.astype(float)
    y_true = get_targets(df_test)

    # Make predictions (read-only, no rating update)
    preds = elo.predict_matches(df_test)
    probs = np.column_stack([preds["away_win_prob"].values,
                              preds["draw_prob"].values,
                              preds["home_win_prob"].values])
    btts = preds["btts_prob"].values
    ou = preds["over_2_5_prob"].values

    metrics = compute_metrics(y_true, probs, name)
    btts_m = compute_btts_metrics(hg, ag, btts)
    ou_m = compute_ou_metrics(hg, ag, ou)
    return {**metrics, **btts_m, **ou_m}


# ═══════════════════════════════════════════════════════════
#  Hyperparameter tuning (grid search)
# ═══════════════════════════════════════════════════════════


def _walk_elo(elo: EloSystem, df: pd.DataFrame) -> None:
    """Walk Elo through matches sequentially (update ratings in-place)."""
    elo.process_matches(
        df,
        home_col="home_team",
        away_col="away_team",
        result_col="result",
        home_goals_col="home_goals",
        away_goals_col="away_goals",
        season_col="season",
    )


def grid_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    quiet: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Grid search over Elo hyperparameters.

    Evaluates each parameter combo by:
    1. Walking through training data to build ratings
    2. Predicting on validation data (log-loss as score)

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data (chronologically before validation).
    df_val : pd.DataFrame
        Validation data (used for scoring each combo).
    quiet : bool
        Suppress per-combo logging.

    Returns
    -------
    tuple[dict, dict, list]
        ``(best_params, best_metrics, all_results)``
    """
    from sklearn.metrics import log_loss as sk_ll

    param_keys = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    n_combos = int(np.prod([len(v) for v in param_values]))
    y_val = get_targets(df_val)

    best_score = float("inf")
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    all_results: list[dict[str, Any]] = []

    combo_idx = 0
    for combo in itertools.product(*param_values):
        combo_idx += 1
        params = dict(zip(param_keys, combo))

        # Build Elo with these params
        elo = EloSystem(
            k=params["k"],
            home_advantage=params["home_advantage"],
            regress_to_mean=params["regress_to_mean"],
            draw_k=params["draw_k"],
            initial_rating=1500,
            use_goal_margin=True,
            regress_factor=1.0 / 3.0,
        )

        # Walk through training data
        _walk_elo(elo, df_train)

        # Predict on validation (read-only)
        preds = elo.predict_matches(df_val)
        probs = np.column_stack([preds["away_win_prob"].values,
                                  preds["draw_prob"].values,
                                  preds["home_win_prob"].values])

        # Score with log-loss
        ll = sk_ll(y_val, probs)

        result = {
            "params": params,
            "log_loss": round(ll, 4),
            "params_display": {
                "k": params["k"],
                "home_adv": params["home_advantage"],
                "regress": params["regress_to_mean"],
                "draw_k": params["draw_k"],
            },
        }
        all_results.append(result)

        if not quiet:
            logger.info(
                "  [%d/%d] K=%d  H=%d  Reg=%s  dK=%.2f  →  val_LL=%.4f",
                combo_idx, n_combos,
                params["k"], params["home_advantage"],
                "Y" if params["regress_to_mean"] else "N",
                params["draw_k"], ll,
            )

        if ll < best_score:
            best_score = ll
            best_params = params
            # Compute full metrics for best
            best_metrics = evaluate_model(elo, df_val, f"Elo (best so far)")

    # Sort results by log-loss
    all_results.sort(key=lambda r: r["log_loss"])

    if not quiet:
        print(f"\n  Grid search complete: {n_combos} combos evaluated")
        print(f"  Best params: {best_params}")
        print(f"  Best val log-loss: {best_score:.4f}")

    return best_params, best_metrics, all_results


# ═══════════════════════════════════════════════════════════
#  Baseline model (LR with Elo difference-only feature)
# ═══════════════════════════════════════════════════════════


def train_baseline_lr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Train a logistic regression using Elo difference as the sole feature.

    This isolates the predictive value of Elo difference alone,
    providing a fair comparison against the full Elo model's probabilities.
    """
    from sklearn.linear_model import LogisticRegression

    # Train Elo on training data to get Elo_Difference features
    elo_train = EloSystem(k=32, home_advantage=100, regress_to_mean=True)
    df_train_feats = elo_train.process_matches(df_train)

    # Get Elo difference for test matches (using trained Elo, read-only)
    test_preds = elo_train.predict_matches(df_test)
    elo_diff_test = test_preds["Elo_Difference"].values.reshape(-1, 1)

    # Last training Elo difference
    if "Elo_Difference" in df_train_feats.columns:
        elo_diff_train = df_train_feats["Elo_Difference"].values.reshape(-1, 1)
    else:
        # Fallback: process again
        _elo = EloSystem(k=32, home_advantage=100, regress_to_mean=True)
        _elo.process_matches(df_train)
        elo_diff_train = _elo.predict_matches(df_train)["Elo_Difference"].values.reshape(-1, 1)

    y_train = get_targets(df_train)
    y_test = get_targets(df_test)

    lr = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42,
                            C=1.0, class_weight="balanced")
    lr.fit(elo_diff_train, y_train)
    test_probs = lr.predict_proba(elo_diff_test)

    metrics = compute_metrics(y_test, test_probs, "LR (Elo-Diff-Only)")
    return lr, test_probs, metrics


# ═══════════════════════════════════════════════════════════
#  Main validation pipeline
# ═══════════════════════════════════════════════════════════


def run_validation(
    data_path: str | Path,
    train_end: str = _DEFAULT_TRAIN_END,
    val_end: str = _DEFAULT_VAL_END,
    test_end: str = _DEFAULT_TEST_END,
    quiet: bool = False,
    skip_grid_search: bool = False,
) -> dict[str, Any]:
    """Run the full Elo model validation pipeline.

    Parameters
    ----------
    data_path : str | Path
        Path to match data CSV.
    train_end, val_end, test_end : str
        Chronological split dates.
    quiet : bool
        Suppress verbose output.
    skip_grid_search : bool
        Skip grid search, use default parameters (for quick testing).

    Returns
    -------
    dict[str, Any]
        Full validation report.
    """
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "data_path": str(data_path),
        "split": {"train_end": train_end, "val_end": val_end, "test_end": test_end},
        "param_grid": {k: list(v) for k, v in PARAM_GRID.items()},
        "model_type": "EloSystem",
        "baseline_type": "LogisticRegression (Elo Diff only)",
    }

    print("\n" + "=" * 70)
    print("  ELO RATING SYSTEM — VALIDATION & HYPERPARAMETER TUNING")
    print("=" * 70)

    # ── [1/8] Load data ─────────────────────────────────
    print(f"\n[1/8] Loading data from {data_path}...")
    df = load_data(data_path)
    report["total_matches"] = len(df)
    report["date_range"] = {
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
    }

    # ── [2/8] Split chronologically (train / val / test) ─
    print(f"\n[2/8] Splitting chronologically...")
    print(f"       Train: < {train_end}")
    print(f"       Val:   {train_end} to {val_end}")
    print(f"       Test:  {val_end} to {test_end}")
    df_train, df_val, df_test = chronological_split(df, train_end, val_end, test_end)
    report["train_size"] = len(df_train)
    report["val_size"] = len(df_val)
    report["test_size"] = len(df_test)

    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        raise ValueError(f"Empty split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ── [3/8] Hyperparameter grid search ────────────────
    print(f"\n[3/8] Grid search over {len(list(itertools.product(*PARAM_GRID.values())))} combos...")
    print(f"       Params: {', '.join(f'{v}={k}' for k, v in _PARAM_LABELS.items())}")

    if skip_grid_search:
        best_params = {"k": 32, "home_advantage": 100, "regress_to_mean": True, "draw_k": 0.25}
        all_results = [{"params": best_params, "log_loss": 999.0}]
        print("  SKIPPING grid search -- using default params")
    else:
        best_params, best_val_metrics, all_results = grid_search(df_train, df_val, quiet=quiet)

    report["param_search"] = {
        "n_combos": len(all_results),
        "best_params": best_params,
        "best_val_log_loss": all_results[0]["log_loss"] if all_results else None,
        "top_5": all_results[:5] if len(all_results) >= 5 else all_results,
    }

    print(f"\n  Best params from grid search:")
    for p, v in best_params.items():
        print(f"    {_PARAM_LABELS.get(p, p):<30s} {v}")

    # ── [4/8] Train final Elo model with best params ────
    print(f"\n[4/8] Training final Elo model with best params on train+val ({len(df_train) + len(df_val)} matches)...")
    df_train_full = pd.concat([df_train, df_val], ignore_index=True)
    df_train_full.sort_values(["date", "home_team"], inplace=True)
    df_train_full.reset_index(drop=True, inplace=True)

    elo_final = EloSystem(
        k=best_params["k"],
        home_advantage=best_params["home_advantage"],
        regress_to_mean=best_params["regress_to_mean"],
        draw_k=best_params["draw_k"],
        initial_rating=1500,
        use_goal_margin=True,
        regress_factor=1.0 / 3.0,
    )
    _walk_elo(elo_final, df_train_full)

    n_teams = len(elo_final.ratings)
    print(f"  Ratings built for {n_teams} teams")

    report["final_model"] = {
        "params": best_params,
        "n_teams": n_teams,
        "training_size": len(df_train_full),
    }

    # ── [5/8] Evaluate on test set ──────────────────────
    print(f"\n[5/8] Evaluating on {len(df_test)} test matches...")
    elo_metrics = evaluate_model(elo_final, df_test, "Elo (final)")
    report["elo_metrics"] = elo_metrics

    print(f"  Elo test metrics:")
    for m in ["accuracy", "log_loss", "brier_score", "btts_accuracy", "over_under_2_5_accuracy"]:
        print(f"    {m:<25s} {elo_metrics.get(m, 'N/A')}")

    # ── [6/8] Train baseline LR ─────────────────────────
    print(f"\n[6/8] Training Logistic Regression baseline (Elo difference only)...")
    lr_model, lr_probs, lr_metrics = train_baseline_lr(df_train_full, df_test)
    report["baseline_metrics"] = lr_metrics

    # ── [7/8] Comparison ──────────────────────────────
    print(f"\n[7/8] Comparing Elo vs LR baseline...")
    comparison = {}
    for metric in ["accuracy", "log_loss", "brier_score"]:
        if metric in elo_metrics and metric in lr_metrics:
            ev = elo_metrics[metric]
            bv = lr_metrics[metric]
            diff = ev - bv
            better = "elo" if (
                (metric == "accuracy" and diff > 0)
                or (metric in ["log_loss", "brier_score"] and diff < 0)
            ) else "baseline" if diff != 0 else "tie"
            comparison[metric] = {
                "elo": ev,
                "baseline": bv,
                "difference": round(diff, 4),
                "better": better,
            }
    report["comparison"] = comparison

    # Print comparison table
    print(f"  {'Metric':<22s} {'Elo':<12s} {'LR Baseline':<12s} {'Better':<10s}")
    print(f"  {'-'*56}")
    for m in ["accuracy", "log_loss", "brier_score"]:
        if m in comparison:
            c = comparison[m]
            print(f"  {m:<22s} {c['elo']:<12.4f} {c['baseline']:<12.4f} {c['better']:<10s}")

    # ── [8/8] Save outputs ─────────────────────────────
    print(f"\n[8/8] Saving outputs...")

    # Save model
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "elo_model.joblib"
    joblib.dump(elo_final, model_path)
    report["model_path"] = str(model_path)
    print(f"  Model: {model_path}")

    # Verify model loads and can predict
    loaded = joblib.load(model_path)
    assert isinstance(loaded, EloSystem), "Loaded model must be an EloSystem!"
    n_teams_loaded = len(loaded.ratings)
    preds_check = loaded.predict_matches(df_test.iloc[:5])
    assert "home_win_prob" in preds_check.columns, "Loaded model must produce predictions!"
    print(f"  [OK] Model loads correctly: {n_teams_loaded} team ratings, predictions verified")

    # Save validation report
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    val_report_path = _REPORT_DIR / f"elo_validation_{timestamp}.json"
    with open(val_report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {val_report_path}")

    # Save comparison report
    comp_report_path = _REPORT_DIR / f"elo_vs_baseline_{timestamp}.json"
    comp_report = {
        "timestamp": timestamp,
        "model": "EloSystem",
        "best_params": best_params,
        "baseline": "LogisticRegression",
        "comparison": comparison,
        "elo_metrics": elo_metrics,
        "baseline_metrics": lr_metrics,
        "param_search_summary": {
            "n_combos": len(all_results),
            "top_5_params": [
                {"params": r["params"], "val_log_loss": r["log_loss"]}
                for r in all_results[:5]
            ],
        },
    }
    with open(comp_report_path, "w") as f:
        json.dump(comp_report, f, indent=2, default=str)
    print(f"  Comparison: {comp_report_path}")

    # ── Summary ────────────────────────────────────────
    duration = time.time() - start_time
    report["duration_seconds"] = round(duration, 2)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY".center(68))
    print("=" * 70)
    print(f"  Duration:         {duration:.2f}s")
    print(f"  Training matches: {len(df_train_full)} ({len(df_train)} train + {len(df_val)} val)")
    print(f"  Test matches:     {len(df_test)}")
    print(f"  Teams rated:      {n_teams}")
    print(f"  Grid combos:      {len(all_results)}")

    print(f"\n  Best parameters:")
    for p, v in best_params.items():
        print(f"    {_PARAM_LABELS.get(p, p):<30s} {v}")

    print(f"\n  {'Metric':<22s} {'Elo':<12s} {'LR Baseline':<12s} {'Better':<10s}")
    print(f"  {'-'*56}")
    for m in ["accuracy", "log_loss", "brier_score"]:
        if m in comparison:
            c = comparison[m]
            print(f"  {m:<22s} {c['elo']:<12.4f} {c['baseline']:<12.4f} {c['better']:<10s}")

    print(f"\n  {'Metric':<22s} {'Elo Value':<12s}")
    print(f"  {'-'*34}")
    for m in ["btts_accuracy", "btts_brier", "over_under_2_5_accuracy", "over_under_2_5_brier"]:
        if m in elo_metrics:
            print(f"  {m:<22s} {elo_metrics[m]:<12.4f}")

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE".center(68))
    print("=" * 70)

    return report


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Elo rating system")
    parser.add_argument("--data", default=str(config.paths.raw / config.data_collection.output_file))
    parser.add_argument("--train-end", default=_DEFAULT_TRAIN_END)
    parser.add_argument("--val-end", default=_DEFAULT_VAL_END)
    parser.add_argument("--test-end", default=_DEFAULT_TEST_END)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--skip-grid", action="store_true", help="Skip grid search (use defaults)")
    args = parser.parse_args()

    try:
        run_validation(
            data_path=args.data,
            train_end=args.train_end,
            val_end=args.val_end,
            test_end=args.test_end,
            quiet=args.quiet,
            skip_grid_search=args.skip_grid,
        )
        return 0
    except Exception as e:
        logger.error("Validation failed: %s", e, exc_info=True)
        print(f"\n[FAIL] Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
