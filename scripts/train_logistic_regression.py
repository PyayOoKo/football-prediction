#!/usr/bin/env python3
"""
Train and validate Logistic Regression classifier using the full feature engineering pipeline.

Usage:
    python scripts/train_logistic_regression.py
    python scripts/train_logistic_regression.py --quiet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Logistic Regression model")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyper-parameter tuning before final training")
    parser.add_argument("--tune-folds", type=int, default=5,
                        help="Number of CV folds for tuning (default 5)")
    parser.add_argument("--tune-iter", type=int, default=50,
                        help="Number of random search iterations (default 50)")
    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    t0 = time.time()

    total_steps = 6 if args.tune else 5
    print("\n" + "=" * 70)
    print("  TRAINING: Logistic Regression")
    print("=" * 70)

    step = 0
    step += 1
    print(f"\n[{step}/{total_steps}] Loading data...")
    df = load_results(low_memory=False)
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df = df[df["target"] >= 0].copy()
    print(f"  Loaded {len(df)} rows")

    step += 1
    print(f"\n[{step}/{total_steps}] Building features...")
    X, y = build_features(df, is_training=True)
    print(f"  Feature matrix: {X.shape}")

    step += 1
    print(f"\n[{step}/{total_steps}] Splitting chronologically...")
    splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))
    for k in ("X_train", "X_val", "X_test"):
        print(f"  {k}: {splits[k].shape[0]} rows")

    # 4. Optional hyper-parameter tuning
    if args.tune:
        step += 1
        print(f"\n[{step}/{total_steps}] Tuning hyper-parameters...")
        result = tune_hyperparameters(
            splits["X_train"], splits["y_train"],
            model_type="logistic_regression", n_folds=args.tune_folds,
            n_iter=args.tune_iter, verbose=not args.quiet,
        )
        print(f"  Best params: {result['best_params']}")
        for k, v in result["best_params"].items():
            if hasattr(config.train, k):
                setattr(config.train, k, v)

    step += 1
    print(f"\n[{step}/{total_steps}] Training Logistic Regression...")
    config.train.model_type = "logistic_regression"
    model, history = train_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )
    if history.get("val_loss"):
        print(f"  Train log-loss: {history['train_loss'][-1]:.4f}")
        print(f"  Val   log-loss: {history['val_loss'][-1]:.4f}")
    print(f"  Complete ({time.time() - t0:.1f}s)")

    step += 1
    print(f"\n[{step}/{total_steps}] Evaluating and saving...")
    config.eval.plot_confusion_matrix = False
    config.eval.plot_roc_curve = False
    config.eval.plot_feature_importance = False
    metrics = evaluate_model(model, splits["X_test"], splits["y_test"])
    for key in ("accuracy", "log_loss", "f1"):
        if key in metrics:
            print(f"  Test {key}: {metrics[key]:.4f}")

    model_path = save_model(model, "logistic_regression_model")
    print(f"  Model saved: {model_path}")

    duration = time.time() - t0
    if args.tune:
        print(f"  (tuned with 5-fold time-series CV)")
    print(f"\n  Total: {duration:.1f}s  |  Test acc: {metrics.get('accuracy', 'N/A'):.2%}")
    print(f"  Status: PASS\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
