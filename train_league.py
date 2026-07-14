"""
train_league.py — Train XGBoost on league data (no heavy hyperparameter tuning).

Usage:
    python train_league.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import config

# ── Configure for league data ──────────────────────────
config.train.model_type = "xgboost"
config.features.include_h2h = True
config.features.include_league_position = True
config.elo.home_advantage = 100
config.elo.k = 32
config.elo.regress_to_mean = True
config.odds.compute_consensus = True

# Good default params (tuned on large football datasets - reduced for speed)
config.train.n_estimators = 200
config.train.max_depth = 5
config.train.learning_rate = 0.05
config.train.subsample = 0.7
config.train.colsample_bytree = 0.7
config.train.reg_lambda = 2.0
config.train.reg_alpha = 0.1

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "results_clean.csv"


def main() -> int:
    t0 = time.time()

    print("=" * 72)
    print("  LEAGUE DATA — XGBoost TRAINING")
    print("=" * 72)

    if not DATA_PATH.exists():
        print(f"\n  [X] Data not found at {DATA_PATH}")
        print("    Run:  python collect_leagues.py")
        return 1

    # ── 1. Load data ─────────────────────────────────────
    print(f"\n  Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  [*] {len(df):,} rows x {len(df.columns)} cols")

    # ── 2. Build features ────────────────────────────────
    from src.feature_engineering import build_features

    print(f"\n  Building features (Elo, Poisson, rolling stats, odds, xG, H2H, league pos) ...")
    X, y = build_features(df, is_training=True)
    print(f"  [*] Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  [*] Target distribution: {dist}")

    # ── 3. Split chronologically ─────────────────────────
    from src.feature_engineering import train_val_test_split

    print(f"\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    print(f"  [*] Train: {len(splits['X_train']):,}  |  "
          f"Val: {len(splits['X_val']):,}  |  "
          f"Test: {len(splits['X_test']):,}")

    # ── 4. Train XGBoost (no tuning — using good defaults) ──
    import xgboost as xgb

    print(f"\n  Training XGBoost (n_estimators={config.train.n_estimators}, "
          f"max_depth={config.train.max_depth}, lr={config.train.learning_rate}) ...")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=config.train.n_estimators,
        max_depth=config.train.max_depth,
        learning_rate=config.train.learning_rate,
        subsample=config.train.subsample,
        colsample_bytree=config.train.colsample_bytree,
        reg_lambda=config.train.reg_lambda,
        reg_alpha=config.train.reg_alpha,
        random_state=config.train.seed,
        n_jobs=-1,
        early_stopping_rounds=15,
    )

    model.fit(
        splits["X_train"], splits["y_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=True,
    )

    # ── 5. Evaluate ─────────────────────────────────────
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    for name, X_set, y_set in [
        ("Train", splits["X_train"], splits["y_train"]),
        ("Val", splits["X_val"], splits["y_val"]),
        ("Test", splits["X_test"], splits["y_test"]),
    ]:
        probs = model.predict_proba(X_set)
        preds = model.predict(X_set)
        acc = accuracy_score(y_set, preds)
        print(f"\n  {name} accuracy: {acc:.4f} ({acc*100:.1f}%)")

    y_pred = model.predict(splits["X_test"])
    y_test = splits["y_test"]
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 72}")
    print(f"  TEST SET RESULTS")
    print(f"{'=' * 72}")
    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Correct: {int(accuracy * len(y_test)):,} / {len(y_test):,}")

    baseline_home = (y_test == 2).mean()
    print(f"  Baseline (always Home): {baseline_home*100:.1f}%")
    print(f"  Model beats baseline by: {(accuracy - baseline_home)*100:+.1f}pp")

    print(f"\n  Confusion matrix:")
    print(f"    {'':>12} {'Away Win':>10} {'Draw':>10} {'Home Win':>10}")
    print(f"    {'-' * 42}")
    for i, label in enumerate(["Away Win", "Draw", "Home Win"]):
        row = f"  {label:>10}"
        for j in range(3):
            row += f"{cm[i, j]:>10}"
        print(row)

    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred,
          target_names=["Away Win", "Draw", "Home Win"], digits=3))

    # Feature importance
    print(f"\n{'TOP 20 FEATURES':-^72}")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        print(f"\n    {'Rank':<6} {'Feature':<42} {'Importance':>12}")
        print(f"    {'-' * 60}")
        for rank, idx in enumerate(indices, 1):
            print(f"    {rank:<6} {X.columns[idx]:<42} {importances[idx]:>12.4f}")

    # ── 6. Save model ───────────────────────────────────
    from src.train import save_model

    model_path = save_model(model, "league_xgboost.joblib")
    print(f"\n  [*] Model saved: {model_path}")

    # ── Summary ─────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  DONE — {elapsed:.1f}s")
    print(f"  Training data: {X.shape[0]:,} matches x {X.shape[1]} features")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  vs baseline:   {baseline_home:.1%} (always Home)")
    print(f"{'=' * 72}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
