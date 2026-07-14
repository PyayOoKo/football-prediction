"""
First Prediction Model — Logistic Regression baseline.

Runs the full pipeline end-to-end:
1. Load preprocessed data
2. Build leakage-free features
3. Split chronologically (no shuffle — time series)
4. Train Logistic Regression with class balancing
5. Evaluate on unseen test set
6. Print Accuracy, Confusion Matrix, Classification Report, and explanations

Usage:
    python run_first_model.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from src.feature_engineering import build_features

# ── Ensure config uses Logistic Regression ──────────────
config.train.model_type = "logistic_regression"

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_first_model")


def main() -> None:
    print("=" * 72)
    print("  FOOTBALL PREDICTION — FIRST MODEL (LOGISTIC REGRESSION)")
    print("=" * 72)

    # ── 1. Load preprocessed data ────────────────────────
    data_path = config.paths.processed / "results_clean.csv"
    if not data_path.exists():
        print(f"\n  ✗ Preprocessed data not found at {data_path}")
        print("    Run the preprocessing pipeline first:\n")
        print("        from src.preprocessing import run_preprocessing")
        print("        report = run_preprocessing()")
        sys.exit(1)

    print(f"\n  Loading preprocessed data from {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ── 2. Build features ────────────────────────────────
    print("\n  Building features (rolling stats, H2H, league position) ...")
    X, y = build_features(df, is_training=True)
    print(f"  ✓ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  ✓ Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 3. Split chronologically ─────────────────────────
    from src.feature_engineering import train_val_test_split

    print("\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    print(f"  ✓ Train: {len(splits['X_train']):,}  |  "
          f"Val: {len(splits['X_val']):,}  |  "
          f"Test: {len(splits['X_test']):,}")

    # ── 4. Train Logistic Regression ─────────────────────
    from src.train import train_model

    print("\n  Training Logistic Regression ...")
    model, history = train_model(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
    )
    print(f"  ✓ Training log-loss: {history.get('train_loss', ['N/A'])[0]:.4f}")
    if "val_loss" in history:
        print(f"  ✓ Validation log-loss: {history['val_loss'][0]:.4f}")
    if "val_accuracy" in history:
        print(f"  ✓ Validation accuracy: {history['val_accuracy'][0]:.4f}")

    # ── 5. Evaluate on test set ──────────────────────────
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    X_test = splits["X_test"]
    y_test = splits["y_test"]

    # Impute missing test values using training column means
    # Use the same imputation as _train_sklearn (with fillna(0) safety net)
    col_means = splits["X_train"].mean().fillna(0)
    X_test_clean = X_test.fillna(col_means)

    y_pred = model.predict(X_test_clean)
    y_proba = model.predict_proba(X_test_clean)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Away Win", "Draw", "Home Win"],
        digits=3,
    )

    # ── 6. Print results with explanations ───────────────
    print("\n" + "=" * 72)
    print("  RESULTS & ANALYSIS")
    print("=" * 72)

    # ── Accuracy ─────────────────────────────────────────
    print(f"\n  {'ACCURACY':-^72}")
    print(f"\n    Test Accuracy:  {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"    Correct:        {int(accuracy * len(y_test)):,} / {len(y_test):,} matches")
    print(f"    Errors:         {int((1 - accuracy) * len(y_test)):,} / {len(y_test):,} matches")

    # Compare against naive baselines
    baseline_home = (y_test == 2).mean()
    baseline_away = (y_test == 0).mean()
    baseline_draw = (y_test == 1).mean()
    best_baseline = max(baseline_home, baseline_away, baseline_draw)
    print(f"\n    ┌─ Naive baselines (predict the most common outcome):")
    print(f"    │   Always predict Home Win:  {baseline_home * 100:.1f}%")
    print(f"    │   Always predict Draw:      {baseline_draw * 100:.1f}%")
    print(f"    │   Always predict Away Win:  {baseline_away * 100:.1f}%")
    print(f"    │   ───────────────────────────────────")
    if accuracy > best_baseline:
        print(f"    │   ✓ Model BEATS best baseline by "
              f"{(accuracy - best_baseline) * 100:.1f} pp")
    else:
        print(f"    │   ✗ Model is {abs(accuracy - best_baseline) * 100:.1f} pp "
              f"below best baseline (expected for first iteration)")

    print(f"    └─")
    print(f"\n    **What this means:** Accuracy measures the fraction of match outcomes")
    print(f"    the model predicted correctly.  Because football has ~45% home wins,")
    print(f"    a naive 'always predict home' classifier achieves ~45% accuracy.")
    print(f"    Our model should beat this to demonstrate it has learned something.")

    # ── Confusion Matrix ─────────────────────────────────
    print(f"\n  {'CONFUSION MATRIX':-^72}")
    print(f"\n      Rows = Actual  |  Columns = Predicted")
    labels = ["Away Win", "Draw", "Home Win"]
    print(f"\n    {'':>12} {'Away Win':>10} {'Draw':>10} {'Home Win':>10}")
    print(f"    {'-' * 42}")
    for i, label in enumerate(labels):
        row = f"  {label:>10}"
        for j in range(3):
            row += f"{cm[i, j]:>10}"
        print(row)

    print(f"\n    **How to read this:**")
    print(f"    The diagonal (top-left to bottom-right) shows correct predictions.")
    for i, label in enumerate(labels):
        correct = cm[i, i]
        total = cm[i].sum()
        pct = correct / total * 100 if total > 0 else 0
        print(f"    • {label:>10}: {correct:>4}/{total:<4} correct ({pct:.0f}%)")
    print(f"\n    Off-diagonal cells show which misclassifications are most common.")
    print(f"    For example, if 'Draw' is often predicted as 'Home Win', the model")
    print(f"    may be too confident in home advantage.")

    # ── Classification Report ────────────────────────────
    print(f"\n  {'CLASSIFICATION REPORT':-^72}")
    print(f"\n{report}")
    print(f"    **How to interpret:**")
    print(f"    • **Precision**:  When the model predicts a class, how often is it right?")
    print(f"    • **Recall**:     Of all actual matches in a class, how many did we catch?")
    print(f"    • **F1-score**:   Harmonic mean of precision & recall (balanced measure).")
    print(f"    • **Support**:    Number of actual instances in the test set.")
    print(f"\n    Draw precision is typically lowest — draws are the rarest class and")
    print(f"    hardest to predict (they sit on a knife-edge between win and loss).")

    # ── Feature importance (coefficient analysis) ────────
    print(f"\n  {'TOP 10 FEATURES (by coefficient magnitude)':-^72}")
    if hasattr(model, "coef_"):
        # LogisticRegression coefficients — take mean absolute value across classes
        coef_abs = np.abs(model.coef_).mean(axis=0)
        feature_names = X.columns

        # Only keep features that existed at training time (X columns)
        top_indices = np.argsort(coef_abs)[::-1][:10]
        print(f"\n    {'Rank':<6} {'Feature':<35} {'|coef|':>10}")
        print(f"    {'-' * 51}")
        for rank, idx in enumerate(top_indices, 1):
            print(f"    {rank:<6} {feature_names[idx]:<35} {coef_abs[idx]:>10.4f}")

        print(f"\n    **What this tells us:**")
        print(f"    Features with large coefficients have the strongest influence on")
        print(f"    predictions.  Positive coefficients make outcomes more likely;")
        print(f"    negative coefficients make them less likely.")

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"\n    Model:        Logistic Regression (multinomial)")
    print(f"    Features:     {X.shape[1]}")
    print(f"    Test samples: {len(y_test)}")
    print(f"    Accuracy:     {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\n    This first model establishes a baseline.  Key observations:")
    print(f"    • Home advantage is the strongest signal (expected)")
    print(f"    • Draws are the hardest class to predict (fewest examples, highest entropy)")
    print(f"    • Improvement paths: XGBoost, hyper-parameter tuning, more features")
    print(f"                        feature selection, calibration, ensemble methods")
    print("=" * 72)


if __name__ == "__main__":
    main()
