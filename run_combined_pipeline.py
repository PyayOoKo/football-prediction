"""
Combined Prediction Pipeline — Train Ensemble, Calibrate, Evaluate, and Backtest.

End-to-end pipeline that brings together ALL signals into a single run:

1. **Load data** — from processed CSV or raw + preprocess
2. **Feature engineering** — Elo, Poisson, Dixon-Coles, xG, odds, rolling stats (5/10/20),
   H2H, league positions, competition importance, player info
3. **Chronological split** — train/val/test (no leakage)
4. **Train individual models** — Logistic Regression, Random Forest, XGBoost
5. **Ensemble** — weight-optimised averaging over LR + RF + XGB + Poisson
6. **Calibration** — Platt scaling on top of ensemble probabilities
7. **Evaluation** — log loss, Brier score, accuracy, ECE, calibration curve
8. **Backtest** — Kelly-criterion value betting simulation
9. **Report** — unified metrics summary + charts

Usage:
    python run_combined_pipeline.py
    python run_combined_pipeline.py --data-path data/raw/worldcup_all.csv
    python run_combined_pipeline.py --skip-backtest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure matplotlib backend early
if matplotlib.get_backend() in ("", None):
    matplotlib.use("Agg")

from config import config
from src.feature_engineering import build_features, train_val_test_split
from src.ensemble import EnsembleModel
from src.calibration import CalibratedModel, calibration_report, calibration_curve as cal_curve
from src.backtesting import run_backtest
from src.time_series_cv import time_series_train_val_test_split

logger = logging.getLogger("combined_pipeline")

# ── Constants ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "combined_pipeline"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "results_clean.csv"

LABELS = ["Away Win", "Draw", "Home Win"]
RESULT_TO_TARGET = {"H": 2, "D": 1, "A": 0}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════════════════════
#  Pipeline State
# ═══════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Aggregated results from the full pipeline run."""

    # Feature info
    n_matches: int = 0
    n_features: int = 0
    n_teams: int = 0

    # Split sizes
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    # Individual model metrics (validation)
    individual_val_loss: dict[str, float] = field(default_factory=dict)

    # Ensemble metrics (validation)
    ensemble_val_loss: float = 0.0
    ensemble_weights: dict[str, float] = field(default_factory=dict)

    # Calibrated ensemble metrics (test)
    calibrated_test_loss: float = 0.0
    raw_test_loss: float = 0.0
    calibrated_brier: float = 0.0
    raw_brier: float = 0.0
    ece: float = 0.0
    test_accuracy: float = 0.0
    brier_improvement: float = 0.0
    log_loss_improvement: float = 0.0

    # Calibration method
    calibration_method: str = "platt"

    # Backtest metrics
    total_bets: int = 0
    roi_pct: float = 0.0
    yield_pct: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit: float = 0.0

    # Paths
    report_path: str = ""
    chart_dir: str = ""

    # Duration
    duration_seconds: float = 0.0

    # Models (stored for potential reuse)
    raw_ensemble: Any = None
    calibrated_ensemble: Any = None
    backtest_engine: Any = None


# ═══════════════════════════════════════════════════════════
#  Pipeline Steps
# ═══════════════════════════════════════════════════════════


def step_load_data(data_path: str | Path | None) -> pd.DataFrame:
    """Load and prepare match data."""
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    if not path.exists():
        # Try raw data fallback
        fallback = PROJECT_ROOT / "data" / "raw" / "results.csv"
        if fallback.exists():
            logger.info("Processed data not found — preprocessing raw data...")
            from src.preprocessing import run_preprocessing
            run_preprocessing(input_path=str(fallback), save=True)
            path = DEFAULT_DATA_PATH
        else:
            raise FileNotFoundError(
                f"Data not found at {path} or {fallback}. "
                "Run data collection first."
            )

    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, low_memory=False)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")

    # Add target column if missing
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map(RESULT_TO_TARGET).fillna(-1).astype("int8")

    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def step_build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full feature matrix with all available signals."""
    logger.info("Building features (Elo, Poisson, DC, xG, odds, rolling 5/10/20, H2H, pos, imp)...")
    t0 = time.time()
    X, y = build_features(df, is_training=True)
    elapsed = time.time() - t0
    logger.info("Feature matrix: %d rows × %d columns (%.1f s)", X.shape[0], X.shape[1], elapsed)
    return X, y


def step_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Calibrated Ensemble",
) -> str:
    """Generate a calibration curve plot and save it.

    Returns the file path.
    """
    curve = cal_curve(y_true, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Plot the calibration curve
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.plot(
        curve["confidences"],
        curve["accuracies"],
        "o-",
        color="#2196F3",
        linewidth=2,
        markersize=8,
        label=f"{model_name}",
    )

    # Add bin counts as annotations
    for i in range(len(curve["bin_centers"])):
        if curve["counts"][i] > 0:
            ax.annotate(
                f"n={int(curve['counts'][i])}",
                (curve["confidences"][i], curve["accuracies"][i]),
                textcoords="offset points",
                xytext=(0, 12),
                fontsize=7,
                ha="center",
                alpha=0.7,
            )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(f"Calibration Curve — {model_name}", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    path = REPORTS_DIR / "calibration_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def step_feature_importance_plot(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
) -> str | None:
    """Generate a horizontal bar chart of top-N feature importances.

    Works with XGBoost, Random Forest, or Logistic Regression coefficients.
    Returns the file path or None if model doesn't expose importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return None

    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="#2196F3",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([str(feature_names[i])[:50] for i in indices[::-1]], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold", fontsize=13)
    ax.invert_yaxis()

    path = REPORTS_DIR / "feature_importance.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ═══════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════


def run_pipeline(
    data_path: str | Path | None = None,
    skip_backtest: bool = False,
    verbose: bool = True,
    calibration_method: str = "platt",
) -> PipelineResult:
    """Run the full combined prediction pipeline.

    Parameters
    ----------
    data_path : str | Path, optional
        Path to the match data CSV.
    skip_backtest : bool
        If True, skip the value betting backtest.
    verbose : bool
        Print progress to console.
    calibration_method : str
        ``\"platt\"`` (sigmoid) or ``\"isotonic\"``.

    Returns
    -------
    PipelineResult
        All computed metrics and models.
    """
    result = PipelineResult()
    t_start = time.time()

    # ── Setup ────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.chart_dir = str(REPORTS_DIR)

    if verbose:
        print()
        print("=" * 90)
        print("  COMBINED PREDICTION PIPELINE".center(88))
        print("=" * 90)
        print(f"  Calibration: {calibration_method}")
        print(f"  Data:        {data_path or DEFAULT_DATA_PATH}")
        print()

    # ═══════════════════════════════════════════════════════
    #  STEP 1 — Load Data
    # ═══════════════════════════════════════════════════════
    logger.info("─" * 50)
    logger.info("STEP 1: Load data")
    logger.info("─" * 50)
    df = step_load_data(data_path)
    result.n_matches = len(df)

    # Count teams
    all_teams = set()
    if "home_team" in df.columns:
        all_teams |= set(df["home_team"].dropna().unique())
    if "away_team" in df.columns:
        all_teams |= set(df["away_team"].dropna().unique())
    result.n_teams = len(all_teams)

    if verbose:
        print(f"  [{df.shape[0]:,} matches] [{result.n_teams} teams] [{df.shape[1]} columns]")

    # ═══════════════════════════════════════════════════════
    #  STEP 2 — Build Features
    # ═══════════════════════════════════════════════════════
    logger.info("─" * 50)
    logger.info("STEP 2: Build features (all signals)")
    logger.info("─" * 50)

    X, y = step_build_features(df)
    result.n_features = X.shape[1]

    # Keep the raw df with indices aligned for calibration step
    # (build_features sorts and resets index, but we need the original
    #  chronological order preserved in X and y)
    df_sorted = df.loc[X.index] if hasattr(X, "index") else df

    if verbose:
        print(f"  [{X.shape[0]:,} rows] [{X.shape[1]} features]")
        dist = y.value_counts(normalize=True)
        print(f"  Target dist:  H={dist.get(2, 0):.1%}  D={dist.get(1, 0):.1%}  A={dist.get(0, 0):.1%}")

    # ═══════════════════════════════════════════════════════
    #  STEP 3 — Chronological Split
    # ═══════════════════════════════════════════════════════
    logger.info("─" * 50)
    logger.info("STEP 3: Chronological split (70/15/15)")
    logger.info("─" * 50)

    splits = train_val_test_split(X, y)
    result.n_train = len(splits["X_train"])
    result.n_val = len(splits["X_val"])
    result.n_test = len(splits["X_test"])

    if verbose:
        print(f"  Train: {result.n_train:,}  |  Val: {result.n_val:,}  |  Test: {result.n_test:,}")

    # Also keep raw data splits for Poisson model
    train_end = result.n_train
    val_end = train_end + result.n_val

    df_train = df_sorted.iloc[:train_end] if len(df_sorted) >= train_end else df_sorted
    df_val = df_sorted.iloc[train_end:val_end] if len(df_sorted) >= val_end else pd.DataFrame()
    df_test = df_sorted.iloc[val_end:] if len(df_sorted) > val_end else pd.DataFrame()

    # ═══════════════════════════════════════════════════════
    #  STEP 4 — Train Ensemble
    # ═══════════════════════════════════════════════════════
    logger.info("─" * 50)
    logger.info("STEP 4: Train Ensemble (LR + RF + XGB + Poisson)")
    logger.info("─" * 50)

    ensemble = EnsembleModel()
    fit_report = ensemble.fit(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        df_train=df_train, df_val=df_val,
    )

    result.ensemble_weights = fit_report["weights"]
    result.ensemble_val_loss = fit_report["val_log_loss"]
    result.individual_val_loss = {
        k: v for k, v in fit_report["individual_log_losses"].items()
        if k.endswith("_val")
    }
    result.raw_ensemble = ensemble

    if verbose:
        print(f"  Ensemble val log-loss: {result.ensemble_val_loss:.4f}")
        print(f"  Weights: {result.ensemble_weights}")
        print()

    # ═══════════════════════════════════════════════════════
    #  STEP 5 — Calibrate Ensemble Probabilities
    # ═══════════════════════════════════════════════════════
    logger.info("─" * 50)
    logger.info(f"STEP 5: Calibrate ensemble ({calibration_method})")
    logger.info("─" * 50)

    result.calibration_method = calibration_method

    # Pre-compute raw ensemble probabilities with df_raw for Poisson model
    raw_test_probs = ensemble.predict_proba(splits["X_test"], df_test)

    # Get raw ensemble predictions on validation set for calibration (include Poisson)
    if df_val is not None and not df_val.empty:
        val_probs = ensemble.predict_proba(splits["X_val"], df_val)
    else:
        val_probs = ensemble.predict_proba(splits["X_val"])

    # Fit calibrators on the pre-computed ensemble probabilities
    # (bypasses CalibratedModel.fit() to avoid double-fitting the base model)
    from src.calibration import _fit_calibrators

    calibrators = _fit_calibrators(val_probs, splits["y_val"].values, 3, calibration_method)

    # Apply calibrated ensemble
    cal_test_probs = np.zeros_like(raw_test_probs)
    for c in range(3):
        calibrator = calibrators[c]
        if calibration_method == "platt":
            p = np.clip(raw_test_probs[:, c], 1e-7, 1 - 1e-7)
            X_c = np.log(p / (1.0 - p)).reshape(-1, 1)
            cal_test_probs[:, c] = calibrator.predict_proba(X_c)[:, 1]
        else:
            cal_test_probs[:, c] = calibrator.transform(raw_test_probs[:, c])
    # Renormalise
    row_sums = cal_test_probs.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    cal_test_probs = cal_test_probs / row_sums[:, np.newaxis]

    # Store for possible reuse
    result.calibrated_ensemble = ensemble  # not wrapped, but calibration is applied

    # ── Evaluate calibration ────────────────────────────
    from sklearn.metrics import log_loss as sk_log_loss

    # Raw ensemble metrics
    result.raw_test_loss = float(sk_log_loss(splits["y_test"], raw_test_probs))

    # One-hot encode for Brier
    n_test = len(splits["y_test"])
    y_onehot = np.zeros((n_test, 3))
    for i, v in enumerate(splits["y_test"]):
        y_onehot[i, int(v)] = 1

    result.raw_brier = float(np.mean(np.sum((raw_test_probs - y_onehot) ** 2, axis=1)))

    # Calibrated metrics
    result.calibrated_test_loss = float(sk_log_loss(splits["y_test"], cal_test_probs))
    result.calibrated_brier = float(np.mean(np.sum((cal_test_probs - y_onehot) ** 2, axis=1)))

    result.log_loss_improvement = result.raw_test_loss - result.calibrated_test_loss
    result.brier_improvement = result.raw_brier - result.calibrated_brier

    # Accuracy
    preds = np.argmax(cal_test_probs, axis=1)
    result.test_accuracy = float(np.mean(preds == splits["y_test"].values))

    # Expected Calibration Error (ECE)
    curve = cal_curve(
        splits["y_test"].values,
        cal_test_probs,
        n_bins=10,
    )
    total_counts = curve["counts"].sum()
    if total_counts > 0:
        result.ece = float(np.mean(
            curve["counts"] / total_counts
            * np.abs(curve["accuracies"] - curve["confidences"])
        ))
    else:
        result.ece = 0.0

    # Generate calibration curve plot
    cal_curve_path = step_calibration_curve(
        splits["y_test"].values,
        cal_test_probs,
        model_name="Calibrated Ensemble",
    )

    # Generate feature importance plot (fallback chain: xgboost → rf → lr)
    imp_model = None
    for name in ("xgboost", "random_forest", "logistic_regression"):
        if name in ensemble.models:
            imp_model = ensemble.models[name]
            break
    if imp_model is not None:
        imp_path = step_feature_importance_plot(
            imp_model,
            list(X.columns),
            top_n=20,
        )

    if verbose:
        print(f"  Raw ensemble test log-loss:       {result.raw_test_loss:.4f}")
        print(f"  Calibrated test log-loss:         {result.calibrated_test_loss:.4f}")
        print(f"  Log-loss improvement:             {result.log_loss_improvement:+.4f}")
        print(f"  Raw Brier:                        {result.raw_brier:.4f}")
        print(f"  Calibrated Brier:                 {result.calibrated_brier:.4f}")
        print(f"  Brier improvement:                {result.brier_improvement:+.4f}")
        print(f"  Test accuracy:                    {result.test_accuracy:.2%}")
        print(f"  ECE (Expected Calibration Error): {result.ece:.4f}")
        print()

    # Print calibration report
    cal_report = calibration_report(
        splits["y_test"].values,
        cal_test_probs,
        model_name=f"Ensemble ({calibration_method})",
    )
    if verbose:
        print(cal_report)
        print()

    # ═══════════════════════════════════════════════════════
    #  STEP 6 — Backtest (optional)
    # ═══════════════════════════════════════════════════════
    if not skip_backtest:
        logger.info("─" * 50)
        logger.info("STEP 6: Value betting backtest")
        logger.info("─" * 50)

        # Extract odds for the test set
        if df_test is not None and not df_test.empty:
            # Try to find odds columns
            odds_candidates = [
                ("BbAvA", "BbAvD", "BbAvH"),
                ("B365A", "B365D", "B365H"),
                ("BWA", "BWD", "BWH"),
            ]
            odds_cols = None
            for cols in odds_candidates:
                if all(c in df_test.columns for c in cols):
                    odds_cols = cols
                    break

            if odds_cols:
                odds_df = df_test[list(odds_cols) + ["home_team", "away_team"]].copy()
                odds_df["home_team"] = odds_df["home_team"].fillna("Unknown")
                odds_df["away_team"] = odds_df["away_team"].fillna("Unknown")
            else:
                odds_df = None
                if verbose:
                    print("  No odds columns found — skipping backtest")
        else:
            odds_df = None

        if odds_df is not None:
            bt_result = run_backtest(
                model=calibrated_ensemble,
                X_test=splits["X_test"],
                y_test=splits["y_test"],
                odds_df=odds_df,
                odds_cols=odds_cols or ("BbAvA", "BbAvD", "BbAvH"),
                team_cols=("home_team", "away_team"),
                initial_bankroll=config.value_betting.bankroll,
                kelly_fraction=config.value_betting.kelly_fraction,
                min_ev=config.value_betting.min_ev,
                output_dir=str(REPORTS_DIR / "backtest_charts"),
                print_report=verbose,
                show_charts=False,
            )

            result.total_bets = bt_result["metrics"].total_bets
            result.roi_pct = bt_result["metrics"].roi_pct
            result.yield_pct = bt_result["metrics"].yield_pct
            result.win_rate_pct = bt_result["metrics"].win_rate_pct
            result.max_drawdown_pct = bt_result["metrics"].max_drawdown_pct
            result.profit = bt_result["metrics"].total_profit
            result.backtest_engine = bt_result["engine"]

    # ═══════════════════════════════════════════════════════
    #  STEP 7 — Save Report
    # ═══════════════════════════════════════════════════════
    result.duration_seconds = time.time() - t_start

    report_text = _build_report(result)
    report_path = REPORTS_DIR / f"pipeline_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    result.report_path = str(report_path)

    if verbose:
        print(report_text)

    # Also save to the standard reports dir
    (REPORTS_DIR / "latest_report.txt").write_text(report_text, encoding="utf-8")

    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE — %.1f s", result.duration_seconds)
    logger.info("Report: %s", result.report_path)
    logger.info("Charts: %s", result.chart_dir)
    logger.info("=" * 50)

    return result


# ═══════════════════════════════════════════════════════════
#  Report Builder
# ═══════════════════════════════════════════════════════════


def _build_report(r: PipelineResult) -> str:
    """Build a formatted text report from pipeline results."""
    lines: list[str] = []
    sep = "=" * 90

    lines.append("")
    lines.append(sep)
    lines.append("  COMBINED PREDICTION PIPELINE — RESULTS".center(88))
    lines.append(sep)
    lines.append(f"")
    lines.append(f"  Date:            {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Duration:        {r.duration_seconds:.1f}s")
    lines.append(f"  Calibration:     {r.calibration_method}")
    lines.append("")

    # ── Dataset Info ────────────────────────────────────
    lines.append(f"  {'─' * 88}")
    lines.append("  DATASET")
    lines.append(f"  {'─' * 88}")
    lines.append(f"  Matches:         {r.n_matches:,}")
    lines.append(f"  Teams:           {r.n_teams}")
    lines.append(f"  Features:        {r.n_features}")
    lines.append(f"  Train:           {r.n_train:,}")
    lines.append(f"  Validation:      {r.n_val:,}")
    lines.append(f"  Test:            {r.n_test:,}")
    lines.append("")

    # ── Ensemble Weights ────────────────────────────────
    lines.append(f"  {'─' * 88}")
    lines.append("  ENSEMBLE WEIGHTS")
    lines.append(f"  {'─' * 88}")
    for name, w in sorted(r.ensemble_weights.items()):
        lines.append(f"  {name:<24s}  {w:.3f}")
    lines.append(f"  {'─' * 88}")
    lines.append(f"  Ensemble val log-loss:  {r.ensemble_val_loss:.4f}")
    lines.append("")

    # ── Individual Model Performance ────────────────────
    if r.individual_val_loss:
        lines.append(f"  {'─' * 88}")
        lines.append("  INDIVIDUAL MODEL LOG-LOSS (validation)")
        lines.append(f"  {'─' * 88}")
        for name, loss in sorted(r.individual_val_loss.items()):
            marker = " ← BEST" if abs(loss - min(r.individual_val_loss.values())) < 1e-6 else ""
            lines.append(f"  {name:<28s}  {loss:.4f}{marker}")
        lines.append("")

    # ── Calibration Results ─────────────────────────────
    lines.append(f"  {'─' * 88}")
    lines.append("  CALIBRATION EVALUATION (test set)")
    lines.append(f"  {'─' * 88}")
    lines.append(f"  {'Metric':<32s} {'Raw Ensemble':>14s} {'Calibrated':>14s} {'Δ':>12s}")
    lines.append(f"  {'Log-loss':<32s} {r.raw_test_loss:>14.4f} {r.calibrated_test_loss:>14.4f} {r.log_loss_improvement:>+12.4f}")
    lines.append(f"  {'Brier score':<32s} {r.raw_brier:>14.4f} {r.calibrated_brier:>14.4f} {r.brier_improvement:>+12.4f}")
    lines.append(f"  {'Test accuracy':<32s} {'':>14s} {r.test_accuracy:>14.2%}")
    lines.append(f"  {'ECE (Expected Cal Error)':<32s} {'':>14s} {r.ece:>14.4f}")
    lines.append("")

    # ── Backtest Results ────────────────────────────────
    if r.total_bets > 0:
        lines.append(f"  {'─' * 88}")
        lines.append("  BACKTEST RESULTS")
        lines.append(f"  {'─' * 88}")
        lines.append(f"  {'Total bets':<30s} {r.total_bets:>10d}")
        lines.append(f"  {'Win rate':<30s} {r.win_rate_pct:>9.1f}%")
        lines.append(f"  {'ROI':<30s} {r.roi_pct:>+9.2f}%")
        lines.append(f"  {'Yield':<30s} {r.yield_pct:>+9.2f}%")
        lines.append(f"  {'Profit/Loss':<30s} £{r.profit:>+9.2f}")
        lines.append(f"  {'Max drawdown':<30s} {r.max_drawdown_pct:>9.1f}%")
        lines.append(f"  {'Kelly fraction':<30s} {config.value_betting.kelly_fraction:.0%}")
        lines.append("")

    # ── Charts
    lines.append(f"  {'─' * 88}")
    lines.append("  OUTPUT")
    lines.append(f"  {'─' * 88}")
    lines.append(f"  Report:    {r.report_path}")
    lines.append(f"  Charts:    {r.chart_dir}")
    lines.append("")

    # ── Interpretation ─────────────────────────────────
    lines.append(f"  {'─' * 88}")
    lines.append("  INTERPRETATION")
    lines.append(f"  {'─' * 88}")

    if r.calibrated_test_loss < r.raw_test_loss:
        lines.append(
            f"  ✅ Calibration ({r.calibration_method}) improved log-loss by "
            f"{r.log_loss_improvement:.4f} and Brier by {r.brier_improvement:.4f}."
        )
    else:
        lines.append(
            f"  ⚠ Calibration ({r.calibration_method}) did NOT improve log-loss "
            f"(Δ={r.log_loss_improvement:+.4f}). Raw ensemble may already be well-calibrated."
        )

    # Baseline comparison
    baseline_home = 0.45  # typical home win rate
    if r.test_accuracy > baseline_home:
        lines.append(
            f"  ✅ Model accuracy ({r.test_accuracy:.1%}) beats naive home-win baseline "
            f"({baseline_home:.0%}) by {r.test_accuracy - baseline_home:+.1%}."
        )

    if r.ece < 0.05:
        lines.append(f"  ✅ Excellent calibration (ECE={r.ece:.4f}, target < 0.05).")
    elif r.ece < 0.10:
        lines.append(f"  ✓ Good calibration (ECE={r.ece:.4f}).")
    else:
        lines.append(f"  ⚠ Poor calibration (ECE={r.ece:.4f}, target < 0.05).")

    if r.total_bets > 0:
        if r.roi_pct > 0:
            lines.append(f"  ✅ Profitable backtest with {r.roi_pct:+.1f}% ROI.")
        else:
            lines.append(f"  ⚠ Backtest showed losses ({r.roi_pct:+.1f}% ROI).")

    lines.append("")
    lines.append(sep)
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined Prediction Pipeline — Train Ensemble, Calibrate, Evaluate, Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to match data CSV (default: data/processed/results_clean.csv)",
    )
    parser.add_argument(
        "--calibration", type=str, default="platt", choices=["platt", "isotonic"],
        help="Calibration method (default: platt)",
    )
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Skip the value betting backtest",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose console output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        result = run_pipeline(
            data_path=args.data_path,
            skip_backtest=args.skip_backtest,
            verbose=not args.quiet,
            calibration_method=args.calibration,
        )
        return 0
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        print(f"\n  ❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
