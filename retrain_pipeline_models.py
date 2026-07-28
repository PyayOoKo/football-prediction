"""
retrain_pipeline_models.py — Train tree models using the exact same
feature pipeline as run_pipeline.py's predict step.

Fixes the feature mismatch where models trained on 158-feature subset
can't predict on the pipeline's 220-feature dataset.

Usage:
    python retrain_pipeline_models.py
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain_pipeline_models")

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    t_start = time.time()

    print()
    print("=" * 72)
    print("  RETRAIN PIPELINE MODELS — XGBoost + LightGBM + CatBoost")
    print("  Using the exact feature pipeline as run_pipeline.py predict step")
    print("=" * 72)

    # ── 1. Load preprocessed data (same as pipeline) ───────
    from src.config import config
    from src.feature_engineering import build_features, train_val_test_split

    data_path = config.paths.processed / "results_clean.csv"
    if not data_path.exists():
        logger.error("Data not found at %s", data_path)
        return 1

    logger.info("Loading %s ...", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    logger.info("  %d rows x %d cols", len(df), len(df.columns))

    # ── 2. Build features (exactly as the pipeline does) ───
    logger.info("Building features (pipeline-style, is_training=True)...")
    X, y = build_features(df, is_training=True)
    logger.info("  Feature matrix: %d rows x %d cols", X.shape[0], X.shape[1])
    dist = dict(zip(*np.unique(y, return_counts=True)))
    logger.info("  Target distribution: %s", dist)

    # Drop _row_id if present (leaks row order)
    if "_row_id" in X.columns:
        X = X.drop(columns=["_row_id"])
        logger.info("  Dropped _row_id → %d cols", X.shape[1])

    # ── 3. Split chronologically (same as pipeline) ────────
    logger.info("Splitting chronologically (70/15/15)...")
    splits = train_val_test_split(X, y)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    logger.info("  Train: %d | Val: %d | Test: %d",
                len(X_train), len(X_val), len(X_test))

    y_test_np = y_test.values.astype(int)

    # ── 4. Train models ─────────────────────────────────────
    import joblib
    models_dir = config.paths.models
    models_dir.mkdir(exist_ok=True)

    results = []

    # ── XGBoost ──────────────────────────────────────────
    print("\n" + "-" * 60)
    logger.info("Training XGBoost ...")
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=5,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_brier = float(np.mean(np.sum((xgb_probs - np.eye(3)[y_test_np]) ** 2, axis=1)))
    xgb_acc = float(np.mean(np.argmax(xgb_probs, axis=1) == y_test_np))
    joblib.dump(xgb_model, models_dir / "xgboost_model.joblib")
    logger.info("  XGBoost: brier=%.5f, acc=%.2f%% → saved", xgb_brier, xgb_acc * 100)
    results.append(("XGBoost", xgb_brier, xgb_acc))

    # ── LightGBM ─────────────────────────────────────────
    print("\n" + "-" * 60)
    logger.info("Training LightGBM ...")
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    lgb_probs = lgb_model.predict_proba(X_test)
    lgb_brier = float(np.mean(np.sum((lgb_probs - np.eye(3)[y_test_np]) ** 2, axis=1)))
    lgb_acc = float(np.mean(np.argmax(lgb_probs, axis=1) == y_test_np))
    joblib.dump(lgb_model, models_dir / "lightgbm_model.joblib")
    logger.info("  LightGBM: brier=%.5f, acc=%.2f%% → saved", lgb_brier, lgb_acc * 100)
    results.append(("LightGBM", lgb_brier, lgb_acc))

    # ── CatBoost ─────────────────────────────────────────
    print("\n" + "-" * 60)
    logger.info("Training CatBoost ...")
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
        thread_count=-1,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cat_probs = cat_model.predict_proba(X_test)
    cat_brier = float(np.mean(np.sum((cat_probs - np.eye(3)[y_test_np]) ** 2, axis=1)))
    cat_acc = float(np.mean(np.argmax(cat_probs, axis=1) == y_test_np))
    joblib.dump(cat_model, models_dir / "catboost_model.joblib")
    logger.info("  CatBoost: brier=%.5f, acc=%.2f%% → saved", cat_brier, cat_acc * 100)
    results.append(("CatBoost", cat_brier, cat_acc))

    # ── Summary ─────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("  RESULTS — Pipeline Feature Set ({:,} rows, {} features)".format(
        X.shape[0], X.shape[1]))
    print("=" * 72)
    print(f"\n  {'Model':<14} {'Brier':>10} {'Accuracy':>12}  {'Saved As'}")
    print(f"  {'-' * 62}")
    for name, brier, acc in results:
        fname = f"{name.lower()}_model.joblib"
        print(f"  {name:<14} {brier:>10.5f} {acc:>10.2%}  {fname}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Models in: {models_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
