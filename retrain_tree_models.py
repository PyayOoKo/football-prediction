"""retrain_tree_models.py — Retrain XGBoost/LightGBM/CatBoost from scratch
on the enriched 127-column dataset with populated bookmaker odds features.

Skips the expensive Dixon-Coles MLE fit (15-20 min) since tree models
benefit most from Elo, odds, rolling, and xG features anyway.
Use this for 1X2 tree model retraining; O/U and BTTS use DC-only.

Usage:
    python retrain_tree_models.py
"""

from __future__ import annotations

import logging
import time
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain_tree_models")


def main() -> int:
    t_start = time.time()

    print()
    print("=" * 72)
    print("  RETRAIN TREE MODELS — XGBoost + LightGBM + CatBoost")
    print("  Using enriched 127-col data with populated odds features")
    print("  DC fit SKIPPED (saves ~15 min; not critical for tree models)")
    print("=" * 72)

    # ── 1. Disable DC (the 15-20 min bottleneck) ─────────
    from config import config
    dc_was_enabled = config.dixon_coles.enabled
    if dc_was_enabled:
        config.dixon_coles.enabled = False
        logger.info("Dixon-Coles disabled for fast feature building")

    # ── 2. Load preprocessed data ─────────────────────────
    from src.feature_engineering import build_features, train_val_test_split

    data_path = config.paths.processed / "results_clean.csv"
    if not data_path.exists():
        logger.error("Data not found at %s", data_path)
        return 1

    logger.info("Loading enriched data from %s ...", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    logger.info("  %d rows x %d cols", len(df), len(df.columns))

    # ── 3. Build features (NO Dixon-Coles → 15-20 min saved) ──
    logger.info("Building features (NO DC — fast path) ...")
    X, y = build_features(df, is_training=True, use_cache=False)
    logger.info("  Feature matrix: %d rows x %d cols", X.shape[0], X.shape[1])

    # Drop _row_id if present
    if "_row_id" in X.columns:
        X = X.drop(columns=["_row_id"])
        logger.info("  Dropped _row_id → %d cols", X.shape[1])

    # ── 4. Split chronologically ──────────────────────────
    logger.info("Splitting chronologically (70/15/15)...")
    splits = train_val_test_split(X, y)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    logger.info("  Train: %d | Val: %d | Test: %d",
                len(X_train), len(X_val), len(X_test))

    y_test_np = y_test.values.astype(int)

    # ── 5. Train models ─━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

    # ── 6. Show top features from XGBoost ─────────────────
    if hasattr(xgb_model, "feature_importances_"):
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        print("\n" + "-" * 60)
        logger.info("Top 15 features (XGBoost gain):")
        for rank, idx in enumerate(indices, 1):
            print(f"  {rank:>2}. {X.columns[idx]:<40} {importances[idx]:.4f}")

    # ── 7. Summary ─────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("  RESULTS — Enriched data ({:,} rows, {} features)".format(
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

    # ── 8. Re-enable DC if we disabled it ─────────────────
    if dc_was_enabled:
        config.dixon_coles.enabled = True
        logger.info("Dixon-Coles re-enabled (original setting restored)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
