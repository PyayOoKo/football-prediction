#!/usr/bin/env python3
"""
Tune all Phase 4 ML models with time-series cross-validation.

Loads data once, splits chronologically, then tunes each model:
  - XGBoost, LightGBM, Random Forest  (5-fold CV, 50 random iterations)
  - Neural Network: skipped (no tune_hyperparameters support)

Saves:
  - Best params:  reports/hyperparameter_tuning_{timestamp}.json
  - Tuned models: models/{model_type}_tuned.joblib  (retrained with best params)

Usage:
    python scripts/tune_all_models.py
    python scripts/tune_all_models.py --quiet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from src.data_loader import load_results
from src.feature_engineering import build_features
from src.time_series_cv import time_series_train_val_test_split
from src.train import train_model, save_model
from src.evaluate import evaluate_model
from src.hyperparameter_tuning import tune_hyperparameters

logger = logging.getLogger(__name__)

_TUNE_MODELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
}

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune all Phase 4 ML models with time-series CV"
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds (default 5)")
    parser.add_argument("--n-iter", type=int, default=50,
                        help="Number of random search iterations (default 50)")
    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 75)
    print("  HYPER-PARAMETER TUNING — ALL PHASE 4 MODELS")
    print(f"  CV folds: {args.n_folds}  |  Random iters: {args.n_iter}")
    print("=" * 75)

    # ── 1. Load & prepare data ──────────────────────────
    print("\n[1/4] Loading and preparing data...")
    df = load_results(low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df = df[df["target"] >= 0].copy()
    print(f"  Loaded {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")

    print("\n[2/4] Building feature matrix...")
    X, y = build_features(df, is_training=True)
    print(f"  Feature matrix: {X.shape}")

    print("\n[3/4] Splitting chronologically (60/20/20)...")
    splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))
    for k in ("X_train", "X_val", "X_test"):
        print(f"  {k}: {splits[k].shape[0]} rows")

    # ── 2. Tune each model ──────────────────────────────
    print(f"\n[4/4] Tuning {len(_TUNE_MODELS)} models...")
    tuning_results: dict[str, dict] = {}
    retrained_models: dict[str, dict] = {}

    for model_type, display_name in _TUNE_MODELS.items():
        print(f"\n  {'─' * 55}")
        print(f"  Model: {display_name}")
        print(f"  {'─' * 55}")

        t_mod = time.time()
        try:
            config.train.model_type = model_type
            result = tune_hyperparameters(
                splits["X_train"], splits["y_train"],
                model_type=model_type,
                n_folds=args.n_folds,
                n_iter=args.n_iter,
                verbose=not args.quiet,
            )
            elapsed = time.time() - t_mod

            tuning_results[model_type] = {
                "best_params": result["best_params"],
                "cv_log_loss": round(result["cv_log_loss"], 4),
                "n_folds": args.n_folds,
                "n_iter": args.n_iter,
                "elapsed_seconds": round(elapsed, 2),
            }

            # Apply best params to config
            for k, v in result["best_params"].items():
                if hasattr(config.train, k):
                    setattr(config.train, k, v)

            # Retrain with best params
            print(f"  Retraining with best params...")
            model, history = train_model(
                splits["X_train"], splits["y_train"],
                splits["X_val"], splits["y_val"],
            )

            # Evaluate on test set
            config.eval.plot_confusion_matrix = False
            config.eval.plot_roc_curve = False
            config.eval.plot_feature_importance = False
            X_test_clean = splits["X_test"].fillna(splits["X_train"].mean().fillna(0))
            metrics = evaluate_model(model, X_test_clean, splits["y_test"])

            # Save retrained model
            tuned_path = save_model(model, f"{model_type}_tuned_model")
            print(f"  Tuned model saved: {tuned_path}")

            retrained_models[model_type] = {
                "model_path": str(tuned_path),
                "test_accuracy": round(metrics.get("accuracy", 0), 4),
                "test_log_loss": round(metrics.get("log_loss", 0), 4),
                "test_brier_score": round(metrics.get("brier_score", 0), 4),
                "val_log_loss": round(history.get("val_loss", [0])[-1], 4),
                "training_seconds": round(elapsed, 2),
            }

            print(f"  Test accuracy: {metrics.get('accuracy', '?'):.4f}  "
                  f"log-loss: {metrics.get('log_loss', '?'):.4f}  "
                  f"brier: {metrics.get('brier_score', '?'):.4f}")

        except Exception as e:
            logger.error("%s failed: %s", display_name, e, exc_info=True)
            print(f"  [FAIL] {e}")
            tuning_results[model_type] = {"error": str(e)}

    # ── 3. Save tuning report ───────────────────────────
    print(f"\n  {'=' * 55}")
    print(f"  SAVING REPORT")
    print(f"  {'=' * 55}")

    report = {
        "timestamp": timestamp,
        "n_folds": args.n_folds,
        "n_iter": args.n_iter,
        "dataset_size": len(df),
        "n_features": X.shape[1],
        "train_size": len(splits["X_train"]),
        "val_size": len(splits["X_val"]),
        "test_size": len(splits["X_test"]),
        "best_params": {mt: r["best_params"] for mt, r in tuning_results.items()
                        if "best_params" in r},
        "cv_log_loss": {mt: r["cv_log_loss"] for mt, r in tuning_results.items()
                        if "cv_log_loss" in r},
        "retrained_metrics": retrained_models,
        "total_duration_seconds": round(time.time() - t0, 2),
    }

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"hyperparameter_tuning_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")

    # ── 4. Summary table ───────────────────────────────
    duration = time.time() - t0
    print(f"\n  {'=' * 55}")
    print(f"  TUNING SUMMARY")
    print(f"  {'=' * 55}")
    print(f"  {'Model':<18s} {'CV LogLoss':<12s} {'Test Acc':<10s} {'Test Brier':<12s} {'Time':<8s}")
    print(f"  {'-' * 55}")
    for model_type, display_name in _TUNE_MODELS.items():
        if model_type in tuning_results and "cv_log_loss" in tuning_results[model_type]:
            tr = tuning_results[model_type]
            rm = retrained_models.get(model_type, {})
            cv = f"{tr['cv_log_loss']:.4f}"
            acc = f"{rm.get('test_accuracy', 0):.4f}"
            brier = f"{rm.get('test_brier_score', 0):.4f}"
            elapsed = f"{tr['elapsed_seconds']:.0f}s"
            print(f"  {display_name:<18s} {cv:<12s} {acc:<10s} {brier:<12s} {elapsed:<8s}")
        else:
            print(f"  {display_name:<18s} {'FAILED':<12s}")

    print(f"\n  Total: {duration:.1f}s")
    print(f"  Status: PASS")
    print(f"  {'=' * 55}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
