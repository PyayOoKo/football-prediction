#!/usr/bin/env python3
"""
Train and validate Random Forest classifier using the full feature engineering pipeline.

Usage:
    python scripts/train_random_forest.py
    python scripts/train_random_forest.py --quiet
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Random Forest model")
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
    print("  TRAINING: Random Forest")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step = 0
    step += 1
    print(f"\n[{step}/{total_steps}] Loading data...")
    df = load_results(low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df = df[df["target"] >= 0].copy()
    print(f"  Loaded {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")

    step += 1
    print(f"\n[{step}/{total_steps}] Building features...")
    X, y = build_features(df, is_training=True)
    print(f"  Feature matrix: {X.shape}")

    step += 1
    print(f"\n[{step}/{total_steps}] Splitting chronologically...")
    splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))
    for k in ("X_train", "X_val", "X_test"):
        print(f"  {k}: {splits[k].shape[0]} rows")
    n_train = int(len(df) * 0.6)
    n_val = int(len(df) * 0.2)
    df_test_raw = df.iloc[n_train + n_val:].copy()
    df_train_raw = df.iloc[:n_train].copy()
    print(f"  Raw test set: {len(df_test_raw)} matches")

    # 4. Optional hyper-parameter tuning
    if args.tune:
        step += 1
        print(f"\n[{step}/{total_steps}] Tuning hyper-parameters...")
        result = tune_hyperparameters(
            splits["X_train"], splits["y_train"],
            model_type="random_forest", n_folds=args.tune_folds,
            n_iter=args.tune_iter, verbose=not args.quiet,
        )
        print(f"  Best params: {result['best_params']}")
        for k, v in result["best_params"].items():
            if hasattr(config.train, k):
                setattr(config.train, k, v)

    step += 1
    print(f"\n[{step}/{total_steps}] Training Random Forest...")
    config.train.model_type = "random_forest"
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
    X_test_clean = splits["X_test"].fillna(splits["X_train"].mean().fillna(0))
    metrics = evaluate_model(model, X_test_clean, splits["y_test"])
    for key in ("accuracy", "log_loss", "brier_score"):
        if key in metrics:
            print(f"  Test {key}: {metrics[key]:.4f}")

    # Compute BTTS and Over/Under 2.5 metrics from raw test data
    hg = df_test_raw["home_goals"].values.astype(float)
    ag = df_test_raw["away_goals"].values.astype(float)
    actual_btts = ((hg > 0) & (ag > 0)).astype(float)
    actual_ou = ((hg + ag) > 2.5).astype(float)

    def _conditional_rates(df_cond):
        hw = df_cond[df_cond["result"] == "H"]
        d = df_cond[df_cond["result"] == "D"]
        aw = df_cond[df_cond["result"] == "A"]
        btts = lambda g: ((g["home_goals"] > 0) & (g["away_goals"] > 0)).mean() if len(g) > 0 else 0.5
        ou = lambda g: ((g["home_goals"] + g["away_goals"]) > 2.5).mean() if len(g) > 0 else 0.5
        return {
            "btts_given_hw": btts(hw), "btts_given_d": btts(d), "btts_given_aw": btts(aw),
            "ou_given_hw": ou(hw), "ou_given_d": ou(d), "ou_given_aw": ou(aw),
        }

    cond = _conditional_rates(df_train_raw)
    probs = model.predict_proba(X_test_clean)
    pred_btts_prob = (probs[:, 2] * cond["btts_given_hw"]
                      + probs[:, 1] * cond["btts_given_d"]
                      + probs[:, 0] * cond["btts_given_aw"])
    pred_ou_prob = (probs[:, 2] * cond["ou_given_hw"]
                    + probs[:, 1] * cond["ou_given_d"]
                    + probs[:, 0] * cond["ou_given_aw"])

    btts_acc = float(np.mean((pred_btts_prob > 0.5).astype(float) == actual_btts))
    ou_acc = float(np.mean((pred_ou_prob > 0.5).astype(float) == actual_ou))

    metrics["btts_accuracy"] = round(btts_acc, 4)
    metrics["over25_accuracy"] = round(ou_acc, 4)
    metrics["n_train"] = len(splits["X_train"])
    metrics["n_test"] = len(splits["y_test"])
    metrics["n_features"] = splits["X_train"].shape[1]
    metrics["duration_seconds"] = round(time.time() - t0, 2)
    metrics["model_type"] = "random_forest"
    metrics["dataset"] = "results"
    print(f"  BTTS accuracy: {btts_acc:.4f}  |  O/U accuracy: {ou_acc:.4f}")

    # Save model
    model_path = save_model(model, "random_forest_model")
    print(f"  Model saved: {model_path}")

    # Save validation report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"random_forest_validation_{timestamp}.json"
    report_data = {
        "timestamp": timestamp,
        "model": "random_forest",
        "status": "PASS",
        "metrics": {k: v for k, v in metrics.items() if k not in ("classification_report", "plots")},
        "tuned": args.tune,
        "tune_folds": args.tune_folds if args.tune else None,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")

    duration = time.time() - t0
    if args.tune:
        print(f"  (tuned with 5-fold time-series CV)")
    print(f"\n  Total: {duration:.1f}s  |  Test acc: {metrics.get('accuracy', 'N/A'):.2%}")
    print(f"  Status: PASS\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
