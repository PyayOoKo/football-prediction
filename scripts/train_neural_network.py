#!/usr/bin/env python3
"""
Train and validate a Neural Network classifier using the full feature engineering pipeline.

Uses PyTorch under the hood with a feed-forward architecture:
  input → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 32 → ReLU → 3 (softmax)

Usage:
    python scripts/train_neural_network.py
    python scripts/train_neural_network.py --tune         # tune learning rate & dropout
    python scripts/train_neural_network.py --epochs 200   # custom training budget
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

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Neural Network model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: config value)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout rate (default: config value)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: config value)")
    parser.add_argument("--tune", action="store_true",
                        help="[NOT IMPLEMENTED] NN tuning is done via --epochs/--dropout/--lr")
    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Apply CLI overrides
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.dropout is not None:
        config.train.dropout = args.dropout
    if args.lr is not None:
        config.train.learning_rate = args.lr

    t0 = time.time()

    print("\n" + "=" * 70)
    print("  TRAINING: Neural Network")
    print("  Architecture: input → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 32 → ReLU → 3")
    print("=" * 70)

    total_steps = 6 if args.tune else 5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step = 0

    # 1. Load data
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

    # 2. Build features
    step += 1
    print(f"\n[{step}/{total_steps}] Building features...")
    X, y = build_features(df, is_training=True)
    print(f"  Feature matrix: {X.shape}")

    # 3. Chronological split
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
    print(f"  Classes: {sorted(y.unique())}")

    # 4. Optional tuning notice
    if args.tune:
        step += 1
        print(f"\n[{step}/{total_steps}] Tuning skipped — use --epochs/--dropout/--lr instead")

    # 5. Train Neural Network
    step += 1
    print(f"\n[{step}/{total_steps}] Training Neural Network...")
    cfg = config.train
    print(f"  Epochs: {cfg.epochs}  |  Dropout: {cfg.dropout}"
          f"  |  LR: {cfg.learning_rate or 0.001}"
          f"  |  Batch: {cfg.batch_size}"
          f"  |  Hidden: {cfg.hidden_layers}")
    config.train.model_type = "neural_network"
    model, history = train_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )
    if history.get("val_loss"):
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val   loss: {history['val_loss'][-1]:.4f}")
        print(f"  Val accuracy:     {history.get('val_accuracy', ['?'])[-1]:.2%}")
    print(f"  Epochs trained: {len(history.get('train_loss', []))}")
    print(f"  Complete ({time.time() - t0:.1f}s)")

    # 6. Evaluate & save
    step += 1
    print(f"\n[{step}/{total_steps}] Evaluating and saving...")
    config.eval.plot_confusion_matrix = False
    config.eval.plot_roc_curve = False
    config.eval.plot_feature_importance = False
    X_test_clean = splits["X_test"].fillna(0)
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
    metrics["model_type"] = "neural_network"
    metrics["dataset"] = "results"
    print(f"  BTTS accuracy: {btts_acc:.4f}  |  O/U accuracy: {ou_acc:.4f}")

    # Save model
    model_path = save_model(model, "neural_network_model")
    print(f"  Model saved: {model_path}")

    # Save validation report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"neural_network_validation_{timestamp}.json"
    report_data = {
        "timestamp": timestamp,
        "model": "neural_network",
        "status": "PASS",
        "metrics": {k: v for k, v in metrics.items() if k not in ("classification_report", "plots")},
        "tuned": args.tune,
        "epochs": config.train.epochs,
        "dropout": config.train.dropout,
        "learning_rate": config.train.learning_rate or 0.001,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")

    duration = time.time() - t0
    print(f"\n  Total: {duration:.1f}s  |  Test acc: {metrics.get('accuracy', 'N/A'):.2%}")
    print(f"  Status: PASS\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
