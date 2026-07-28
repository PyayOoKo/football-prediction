"""Train CatBoost model and save to models/catboost_model.joblib.

Trains on the general results_clean.csv data so the model works with
the same feature pipeline used by ThreeModelBlend and the weight optimiser.
"""

from __future__ import annotations

import logging
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from config import config
from src.feature_engineering import build_features, train_val_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_catboost")


def main() -> None:
    print("=" * 72)
    print("  FOOTBALL PREDICTION — CATBOOST")
    print("=" * 72)

    data_path = config.paths.processed / "results_clean.csv"
    if not data_path.exists():
        print(f"\n  Preprocessed data not found at {data_path}")
        sys.exit(1)

    print(f"\n  Loading preprocessed data ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    print("\n  Building features ...")
    X, y = build_features(df, is_training=True)
    print(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  Target: {dist}")

    print("\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    print(f"  Train: {len(splits['X_train']):,}  |  "
          f"Val: {len(splits['X_val']):,}  |  "
          f"Test: {len(splits['X_test']):,}")

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]

    print("\n  Training CatBoost ...")
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        early_stopping_rounds=30,
        verbose=100,
        random_seed=42,
        thread_count=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    out_path = config.paths.models / "catboost_model.joblib"
    joblib.dump(model, out_path)
    print(f"\n  Model saved to {out_path}")

    val_preds = model.predict_proba(X_val)
    acc = (np.argmax(val_preds, axis=1) == y_val.values).mean()
    print(f"  Validation accuracy: {acc:.2%}")

    print(f"  Features in model: {len(model.feature_names_)}")
    print("\n  Done!")


if __name__ == "__main__":
    main()
