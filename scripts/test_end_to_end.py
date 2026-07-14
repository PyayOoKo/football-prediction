"""
End-to-End Pipeline Validation Script.

Validates the entire prediction pipeline:
  1. Load data
  2. Inline data-quality checks
  3. Build features (with built-in leakage prevention)
  4. Inline feature-matrix validation
  5. Chronological train/val/test split (no shuffle)
  6. Train a logistic-regression baseline
  7. Save the trained model
  8. Evaluate on held-out test set
  9. Backtest value-betting simulation (synthetic odds if real odds absent)

Usage::

    python scripts/test_end_to_end.py

Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_e2e")

# ── Project root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Suppress verbose third-party logging
for name in ("matplotlib", "PIL", "urllib3", "fsspec", "sklearn", "xgboost"):
    logging.getLogger(name).setLevel(logging.WARNING)

# Force non-interactive Matplotlib backend
import matplotlib
matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════

def main() -> int:
    status = 0
    timings: dict[str, float] = {}

    print("\n" + "=" * 70)
    print("  END-TO-END PIPELINE VALIDATION")
    print("=" * 70)

    # ── Step 1: Load data ─────────────────────────────────
    print("\n[1/9] Loading data ...")
    t0 = time.time()
    data_path = ROOT / "data" / "processed" / "results_clean.csv"
    if not data_path.exists():
        log.error("Data file not found: %s. Run preprocessing first.", data_path)
        return 1

    df = pd.read_csv(data_path, low_memory=False)
    timings["1_load"] = time.time() - t0
    print(f"      Loaded {len(df)} rows x {len(df.columns)} cols  ({timings['1_load']:.1f}s)")

    # ── Step 2: Data quality checks ───────────────────────
    print("\n[2/9] Running data quality checks ...")
    t0 = time.time()
    errors = _check_data_quality(df)
    timings["2_quality"] = time.time() - t0
    if errors:
        for e in errors:
            print(f"      FAIL: {e}")
        return 1
    print(f"      All checks passed  ({timings['2_quality']:.2f}s)")

    # ── Step 3: Build features ────────────────────────────
    print("\n[3/9] Building features (leakage-safe) ...")
    t0 = time.time()
    try:
        from src.feature_engineering import build_features
        X, y = build_features(df, is_training=True)
    except Exception as exc:
        log.error("Feature engineering failed: %s", exc)
        return 1
    timings["3_features"] = time.time() - t0
    print(f"      Feature matrix: {X.shape}, target: {y.shape}  ({timings['3_features']:.1f}s)")

    # ── Step 4: Feature validation ────────────────────────
    print("\n[4/9] Validating feature matrix ...")
    t0 = time.time()
    errors = _check_features(X, y)
    timings["4_validate"] = time.time() - t0
    if errors:
        for e in errors:
            print(f"      FAIL: {e}")
        return 1
    print(f"      All feature checks passed  ({timings['4_validate']:.2f}s)")

    # ── Step 5: Chronological split ───────────────────────
    print("\n[5/9] Chronological train/val/test split ...")
    t0 = time.time()
    try:
        from src.time_series_cv import time_series_train_val_test_split
        splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))
    except Exception as exc:
        log.error("Split failed: %s", exc)
        return 1
    timings["5_split"] = time.time() - t0
    for k in ("X_train", "X_val", "X_test"):
        print(f"      {k}: {splits[k].shape[0]} rows")
    print(f"      Split complete  ({timings['5_split']:.2f}s)")

    # ── Step 6: Train baseline ────────────────────────────
    print("\n[6/9] Training baseline model (logistic regression) ...")
    t0 = time.time()
    try:
        from config import config
        from src.train import train_model

        original_type = config.train.model_type
        config.train.model_type = "logistic_regression"
        model, history = train_model(
            splits["X_train"],
            splits["y_train"],
            splits["X_val"],
            splits["y_val"],
        )
        config.train.model_type = original_type
    except Exception as exc:
        log.error("Training failed: %s", exc)
        return 1
    timings["6_train"] = time.time() - t0
    # Print best metrics
    if history.get("val_loss"):
        print(f"      Train log-loss: {history['train_loss'][-1]:.4f}")
        print(f"      Val   log-loss: {history['val_loss'][-1]:.4f}")
        if history.get("val_accuracy"):
            print(f"      Val   accuracy: {history['val_accuracy'][-1]:.2%}")
    print(f"      Training complete  ({timings['6_train']:.1f}s)")

    # Capture training column means for test-set imputation
    col_means = splits["X_train"].mean().fillna(0)

    # ── Step 7: Save model ────────────────────────────────
    print("\n[7/9] Saving model ...")
    t0 = time.time()
    try:
        from src.train import save_model

        model_path = save_model(model, "baseline_logistic_regression")
    except Exception as exc:
        log.error("Save failed: %s", exc)
        return 1
    timings["7_save"] = time.time() - t0
    print(f"      Saved to {model_path}  ({timings['7_save']:.1f}s)")

    # ── Step 8: Evaluate on test set ──────────────────────
    print("\n[8/9] Evaluating on test set ...")
    t0 = time.time()
    try:
        from config import config
        from src.evaluate import evaluate_model

        config.eval.plot_confusion_matrix = False
        config.eval.plot_roc_curve = False
        config.eval.plot_feature_importance = False

        X_test_clean = splits["X_test"].fillna(col_means)
        metrics = evaluate_model(model, X_test_clean, splits["y_test"])
    except Exception as exc:
        log.error("Evaluation failed: %s", exc)
        return 1
    timings["8_eval"] = time.time() - t0
    # Print key metrics
    for key in ("accuracy", "precision", "recall", "f1", "log_loss", "roc_auc"):
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"      {key}: {val:.4f}")
            else:
                print(f"      {key}: {val}")
    print(f"      Evaluation complete  ({timings['8_eval']:.1f}s)")

    # ── Step 9: Backtest ──────────────────────────────────
    print("\n[9/9] Running value-betting backtest ...")
    t0 = time.time()
    try:
        from src.backtesting import run_backtest

        backtest_result = run_backtest(
            model,
            X_test_clean,
            splits["y_test"],
            odds_df=None,
            initial_bankroll=1000.0,
            kelly_fraction=0.25,
            min_ev=0.0,
            output_dir=str(ROOT / "reports" / "backtest"),
            print_report=False,
            show_charts=False,
        )
    except Exception as exc:
        log.error("Backtest failed: %s", exc)
        return 1
    timings["9_backtest"] = time.time() - t0
    bt = backtest_result["metrics"]
    print(f"      Bets placed: {bt.total_bets}")
    print(f"      Total staked: £{bt.total_staked:.2f}")
    print(f"      Total profit: £{bt.total_profit:+.2f}")
    print(f"      ROI: {bt.roi_pct:+.2f}%")
    print(f"      Yield: {bt.yield_pct:+.2f}%")
    print(f"      Win rate: {bt.win_rate_pct:.1f}%")
    print(f"      Max drawdown: {bt.max_drawdown_pct:.1f}%")
    print(f"      Backtest complete  ({timings['9_backtest']:.1f}s)")

    # ── Summary ───────────────────────────────────────────
    total_time = sum(timings.values())
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for step, dur in timings.items():
        print(f"  {step:<20s}: {dur:>7.2f}s")
    print(f"  {'─' * 28}")
    print(f"  {'total':<20s}: {total_time:>7.2f}s")
    print(f"\n  Model:       logistic_regression")
    print(f"  Test acc:    {metrics.get('accuracy', float('nan')):.2%}")
    print(f"  Test logloss:{metrics.get('log_loss', float('nan')):.4f}")
    print(f"  Test samples:{len(splits['y_test'])}")
    print(f"  Backtest:    {bt.total_bets} bets, £{bt.total_profit:+.2f} P&L")
    print(f"  Status:      {'PASS' if status == 0 else 'FAIL'}")
    print("=" * 70)
    return status


# ═══════════════════════════════════════════════════════════
#  Validation helpers
# ═══════════════════════════════════════════════════════════


def _check_data_quality(df: pd.DataFrame) -> list[str]:
    """Run essential data-quality assertions. Returns a list of error messages."""
    errors: list[str] = []

    if len(df) == 0:
        errors.append("DataFrame is empty")

    required = {"date", "home_team", "away_team", "result", "target"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    if "target" in df.columns:
        invalid = df[~df["target"].isin([-1, 0, 1, 2])]
        if len(invalid) > 0:
            errors.append(f"{len(invalid)} rows have invalid target values (not -1,0,1,2)")

        unknown = (df["target"] == -1).sum()
        if unknown > 0:
            errors.append(f"{unknown} rows have unknown target (-1) — drop these before training")

    if "date" in df.columns:
        try:
            parsed = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            if parsed.isna().any():
                errors.append(f"{parsed.isna().sum()} unparseable dates")
        except Exception:
            errors.append("Failed to parse date column")

    if "home_team" in df.columns and "away_team" in df.columns:
        overlapping = df[df["home_team"] == df["away_team"]]
        if len(overlapping) > 0:
            errors.append(f"{len(overlapping)} rows have home_team == away_team")

    return errors


def _check_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Assert feature-matrix invariants. Returns a list of error messages."""
    errors: list[str] = []

    if len(X) == 0:
        errors.append("Feature matrix is empty")

    if len(X) != len(y):
        errors.append(f"X ({len(X)}) and y ({len(y)}) length mismatch")

    # Must be numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        errors.append(f"Non-numeric feature columns: {non_numeric}")

    # Target/result leakage
    leak_cols = [c for c in X.columns if c in ("result", "target", "home_goals", "away_goals")]
    if leak_cols:
        errors.append(f"Leakage columns still in feature matrix: {leak_cols}")

    # y should be integer-ish
    if not np.issubdtype(y.dtype, np.integer):
        errors.append(f"Target dtype is {y.dtype}, expected integer")

    # y values in range
    if len(y) > 0:
        vmin, vmax = y.min(), y.max()
        if vmin < 0 or vmax > 2:
            errors.append(f"Target values outside [0, 2]: min={vmin}, max={vmax}")

    return errors


if __name__ == "__main__":
    sys.exit(main())
