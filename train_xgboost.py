"""
XGBoost Prediction Model — with hyper-parameter tuning, evaluation, and model saving.

Usage:
    python train_xgboost.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from src.feature_engineering import build_features

config.train.model_type = "xgboost"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_xgboost")


def main() -> None:
    print("=" * 72)
    print("  FOOTBALL PREDICTION — XGBOOST WITH HYPER-PARAMETER TUNING")
    print("=" * 72)

    # ── 1. Load data ─────────────────────────────────────
    data_path = config.paths.processed / "results_clean.csv"
    if not data_path.exists():
        print(f"\n  ✗ Preprocessed data not found at {data_path}")
        print("    Run:  from src.preprocessing import run_preprocessing")
        print("          report = run_preprocessing()")
        sys.exit(1)

    print(f"\n  Loading preprocessed data ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  ✓ {len(df):,} rows × {len(df.columns)} columns")

    # ── 2. Build features ────────────────────────────────
    print("\n  Building features (rolling stats, H2H, league position) ...")
    X, y = build_features(df, is_training=True)
    print(f"  ✓ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  ✓ Target: {dist}")

    # ── 3. Split chronologically ─────────────────────────
    from src.feature_engineering import train_val_test_split

    print("\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    print(f"  ✓ Train: {len(splits['X_train']):,}  |  "
          f"Val: {len(splits['X_val']):,}  |  "
          f"Test: {len(splits['X_test']):,}")

    # ── 4. Hyper-parameter tuning ────────────────────────
    from src.train import tune_hyperparameters

    print("\n  Tuning hyper-parameters (RandomizedSearchCV, 5-fold CV, 80 iters) ...")
    best_params = tune_hyperparameters(
        splits["X_train"], splits["y_train"],
        n_folds=5, n_iter=80,
    )
    print(f"  ✓ Best params: {best_params}")

    # Apply best params to config so _build_model uses them
    for key, val in best_params.items():
        if hasattr(config.train, key):
            setattr(config.train, key, val)

    # ── 5. Train final model ─────────────────────────────
    from src.train import train_model

    print("\n  Training final XGBoost with best params ...")
    model, history = train_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )
    print(f"  ✓ Training log-loss:   {history['train_loss'][0]:.4f}")
    print(f"  ✓ Validation log-loss: {history.get('val_loss', ['N/A'])[0]}")
    print(f"  ✓ Validation accuracy: {history.get('val_accuracy', ['N/A'])[0]}")

    # ── 6. Evaluate on test set ──────────────────────────
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X_test = splits["X_test"]
    y_test = splits["y_test"]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Away Win", "Draw", "Home Win"],
        digits=3,
    )

    # ── 7. Print results ─────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  RESULTS")
    print(f"{'=' * 72}")

    # ── Accuracy ─────────────────────────────────────────
    print(f"\n{'ACCURACY':-^72}")
    print(f"\n    Test Accuracy:           {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"    Correct:                 {int(accuracy * len(y_test)):,} / {len(y_test):,}")
    print(f"    Errors:                  {int((1 - accuracy) * len(y_test)):,} / {len(y_test):,}")

    baseline_home = (y_test == 2).mean()
    baseline_away = (y_test == 0).mean()
    baseline_draw = (y_test == 1).mean()
    best_baseline = max(baseline_home, baseline_away, baseline_draw)
    improvement = (accuracy - best_baseline) * 100
    print(f"\n    Naive baselines:")
    print(f"      Always Home: {baseline_home * 100:.1f}%")
    print(f"      Always Draw: {baseline_draw * 100:.1f}%")
    print(f"      Always Away: {baseline_away * 100:.1f}%")
    print(f"      XGBoost beats best baseline by {improvement:+.1f} pp")

    # ── Confusion Matrix ─────────────────────────────────
    print(f"\n{'CONFUSION MATRIX':-^72}")
    labels = ["Away Win", "Draw", "Home Win"]
    print(f"\n    {'':>12} {'Away Win':>10} {'Draw':>10} {'Home Win':>10}")
    print(f"    {'-' * 42}")
    for i, label in enumerate(labels):
        row = f"  {label:>10}"
        for j in range(3):
            row += f"{cm[i, j]:>10}"
        print(row)
    print(f"\n    Diagonal = correct predictions")
    for i, label in enumerate(labels):
        correct = cm[i, i]
        total = cm[i].sum()
        print(f"    • {label:>10}: {correct:>4}/{total:<4}  ({correct/total*100:.0f}%)" if total else "")

    # ── Classification Report ────────────────────────────
    print(f"\n{'CLASSIFICATION REPORT':-^72}")
    print(f"\n{report}")

    # ── Feature Importance ───────────────────────────────
    print(f"{'FEATURE IMPORTANCE (XGBoost gain)':-^72}")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        print(f"\n    {'Rank':<6} {'Feature':<38} {'Importance':>12}")
        print(f"    {'-' * 56}")
        for rank, idx in enumerate(indices, 1):
            print(f"    {rank:<6} {X.columns[idx]:<38} {importances[idx]:>12.4f}")
        print(f"\n    Top feature is {X.columns[indices[0]]} (importance={importances[indices[0]]:.4f})")

    # ── Save model ───────────────────────────────────────
    from src.train import save_model

    print(f"\n{'SAVING MODEL':-^72}")
    model_path = save_model(model, "xgboost_model.joblib")
    print(f"  ✓ Model saved to: {model_path}")
    print(f"\n    To load later:  from src.train import load_model")
    print(f"                    model = load_model('xgboost_model.joblib')")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")
    print(f"\n    Model:       XGBoost")
    print(f"    Features:    {X.shape[1]}")
    print(f"    Test size:   {len(y_test)} matches")
    print(f"    Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"    Best params: {best_params}")
    print(f"\n    Key observations:")
    print(f"    • XGBoost handles non-linear relationships and feature interactions")
    print(f"    • Handles NaN natively (no imputation needed)")
    print(f"    • Built-in feature importance via gain / cover / frequency")
    print(f"    • Improvement over Logistic Regression expected due to:")
    print(f"      - Captures interactions between features (e.g. H2H × league position)")
    print(f"      - Robust to outliers and skewed distributions")
    print(f"      - Built-in regularisation (L1/L2) prevents overfitting")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
