"""
debug_elo.py — Debug Elo baseline alignment with feature matrix.

Verifies that the chronological split between build_features() output
and the raw DataFrame match up correctly for Elo computation.

Usage:
    python debug_elo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("debug_elo")

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    print("=" * 72)
    print("  ELO DEBUG — Verify chronological split alignment")
    print("=" * 72)

    data_path = PROJECT_ROOT / "data" / "processed" / "results_clean.csv"
    if not data_path.exists():
        logger.error("Data not found: %s", data_path)
        return 1

    # ── 1. Load raw data ────────────────────────────────
    print("\n-- 1. Loading raw data -----------------------")
    df_raw = pd.read_csv(data_path, low_memory=False)
    print(f"  Raw rows: {len(df_raw)}")
    print(f"  Columns with '_row_id': {'_row_id' in df_raw.columns}")
    if "_row_id" in df_raw.columns:
        print(f"  _row_id range: {df_raw['_row_id'].min()} to {df_raw['_row_id'].max()}")
        print(f"  _row_id dtype: {df_raw['_row_id'].dtype}")

    print(f"  Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")
    print(f"  Date dtype: {df_raw['date'].dtype}")

    # ── 2. Build features (this does the sorting) ───────
    print("\n-- 2. Building features -----------------------")
    from src.feature_engineering import build_features

    X, y = build_features(df_raw, is_training=True)
    print(f"  Feature matrix: {X.shape[0]} rows x {X.shape[1]} cols")
    print(f"  Has _row_id: {'_row_id' in X.columns}")

    if "_row_id" in X.columns:
        print(f"  _row_id range in X: {X['_row_id'].min()} to {X['_row_id'].max()}")
        print(f"  _row_id dtype in X: {X['_row_id'].dtype}")

    # ── 3. Chronological split ─────────────────────────
    print("\n-- 3. Splitting -------------------------------")
    from src.feature_engineering import train_val_test_split

    splits = train_val_test_split(X, y)
    n_train = len(splits["X_train"])
    n_val = len(splits["X_val"])
    n_test = len(splits["X_test"])
    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
    print(f"  Total: {n_train + n_val + n_test:,} (raw: {len(df_raw)})")
    print(f"  Match: {'YES' if n_train + n_val + n_test == len(df_raw) else 'NO — MISMATCH!'}")

    X_test = splits["X_test"]

    # ── 4. Verify sorting matches ──────────────────────
    print("\n-- 4. Verifying sort order ---------------------")

    # build_features sorts by [date, home_team] after converting date to datetime
    # Let's replicate that sorting on the raw data
    df_sorted = df_raw.copy()
    df_sorted["date_dt"] = pd.to_datetime(df_sorted["date"], errors="coerce")
    df_sorted = df_sorted.sort_values(["date_dt", "home_team"]).reset_index(drop=True)

    print(f"  Sorted df has {len(df_sorted)} rows")

    # Check if the _row_id sequence after sorting matches X's _row_id sequence
    if "_row_id" in X.columns and "_row_id" in df_raw.columns:
        df_sorted_with_id = df_sorted.copy()
        df_sorted_with_id["_row_id_original"] = df_raw["_row_id"].iloc[
            df_sorted.index if not df_sorted.index.equals(pd.RangeIndex(len(df_sorted)))
            else range(len(df_sorted))
        ]

        # build_features sorts its COPY of df, then resets index
        # The _row_id column in X is preserved from the sorted df
        x_row_ids = X["_row_id"].values[:100]  # first 100
        df_row_ids_after_sort = df_sorted["_row_id"].values[:100]  # _row_id from original df_raw

        # Actually, _row_id was in df_raw before sorting. build_features
        # sorts df (which has _row_id in it), so the _row_id in X should
        # match the _row_id column in df_sorted (which was carried through the sort)
        id_match = (X["_row_id"].values == df_sorted["_row_id"].values).all()
        print(f"  _row_id sequence matches between X and df_sorted: {id_match}")

        if not id_match:
            print("  MISMATCH! First 10 _row_ids from X:", X["_row_id"].values[:10])
            print("  First 10 _row_ids from df_sorted:", df_sorted["_row_id"].values[:10])

    # ── 5. Verify test set alignment ────────────────────
    print("\n-- 5. Test set alignment ------------------------")

    if "_row_id" in X.columns:
        test_row_ids = set(X_test["_row_id"].values)

        if "_row_id" in df_raw.columns:
            df_sorted_has_id = "_row_id" in df_sorted.columns
            if df_sorted_has_id:
                # The test rows in df_sorted should be the LAST n_test rows
                test_in_sorted = set(df_sorted.iloc[n_train + n_val:]["_row_id"].values)
                overlap = len(test_row_ids & test_in_sorted)
                print(f"  Test row IDs in X_test: {len(test_row_ids)}")
                print(f"  Test row IDs in last {n_test} of df_sorted: {len(test_in_sorted)}")
                print(f"  Overlap: {overlap}")
                print(f"  Match: {'YES' if overlap == n_test else 'NO — MISMATCH!'}")

                if overlap < n_test:
                    print(f"  Missing: {len(test_row_ids - test_in_sorted)} IDs not in expected slice")
                    print(f"  Extra: {len(test_in_sorted - test_row_ids)} IDs in slice not in X_test")

    # ── 6. Simple Elo test ──────────────────────────────
    print("\n-- 6. Elo baseline test -------------------------")
    from src.elo import EloSystem

    # METHOD 1: Sort raw df by [date, home_team], take chronological slices
    elo = EloSystem(k=20, home_advantage=100, initial_rating=1500)

    df_sorted_train = df_sorted.iloc[:n_train + n_val].copy()
    df_sorted_test = df_sorted.iloc[n_train + n_val:].copy()

    print(f"  Elo train: {len(df_sorted_train)} matches")
    print(f"  Elo test:  {len(df_sorted_test)} matches")
    print(f"  Train date range: {df_sorted_train['date'].min()} to {df_sorted_train['date'].max()}")
    print(f"  Test date range: {df_sorted_test['date'].min()} to {df_sorted_test['date'].max()}")

    # Process Elo
    elo.process_matches(df_sorted_train)

    # Predict test matches
    test_probs = []
    for _, row in df_sorted_test.iterrows():
        try:
            proba = elo.predict_proba(pd.DataFrame([{"home_team": row["home_team"], "away_team": row["away_team"]}]))[0]
            test_probs.append(proba)
        except Exception:
            test_probs.append(np.array([0.33, 0.34, 0.33]))

    test_probs = np.array(test_probs)

    # Compare with actual outcomes
    y_test_actual = df_sorted_test["result"].map({"A": 0, "D": 1, "H": 2}).values
    preds = np.argmax(test_probs, axis=1)

    # Brier
    y_oh = np.zeros_like(test_probs)
    for i, v in enumerate(y_test_actual):
        if 0 <= int(v) <= 2:
            y_oh[i, int(v)] = 1
    brier = float(np.mean(np.sum((test_probs - y_oh) ** 2, axis=1)))

    acc = (preds == y_test_actual).mean()
    print(f"\n  METHOD 1 (chronological slice of sorted df):")
    print(f"    Brier: {brier:.5f}")
    print(f"    Accuracy: {acc:.2%}")
    print(f"    n_test: {len(y_test_actual)}")

    # Naive baseline
    home_pct = (y_test_actual == 2).mean()
    draw_pct = (y_test_actual == 1).mean()
    away_pct = (y_test_actual == 0).mean()
    always_home = np.array([[0.0, 0.0, 1.0]] * len(y_test_actual))
    y_oh_home = np.zeros_like(always_home)
    for i, v in enumerate(y_test_actual):
        if 0 <= int(v) <= 2:
            y_oh_home[i, int(v)] = 1
    brier_home = float(np.mean(np.sum((always_home - y_oh_home) ** 2, axis=1)))
    print(f"\n  Naive baselines on this test set:")
    print(f"    Home win rate: {home_pct:.1%}")
    print(f"    Draw rate: {draw_pct:.1%}")
    print(f"    Away win rate: {away_pct:.1%}")
    print(f"    Always-Home Brier: {brier_home:.5f}, Acc: {home_pct:.1%}")

    # ── 7. Compare with tree model predictions ──────────
    print("\n-- 7. Cross-check with XGBoost ------------------")
    import joblib

    xgb_path = PROJECT_ROOT / "models" / "xgboost_model.joblib"
    if xgb_path.exists():
        xgb = joblib.load(xgb_path)
        # Use X_test from the feature matrix for tree predictions
        X_test_features = X_test.drop(columns=["_row_id"], errors="ignore")
        xgb_probs = xgb.predict_proba(X_test_features)
        y_test_from_X = y.iloc[len(X) - n_test:]  # last n_test rows of y
        y_test_from_X_np = y_test_from_X.values.astype(int)

        y_oh_xgb = np.zeros_like(xgb_probs)
        for i, v in enumerate(y_test_from_X_np):
            if 0 <= int(v) <= 2:
                y_oh_xgb[i, int(v)] = 1
        xgb_brier = float(np.mean(np.sum((xgb_probs - y_oh_xgb) ** 2, axis=1)))
        xgb_acc = (np.argmax(xgb_probs, axis=1) == y_test_from_X_np).mean()

        print(f"  XGBoost on X_test:")
        print(f"    Brier: {xgb_brier:.5f}")
        print(f"    Accuracy: {xgb_acc:.2%}")
        print(f"    n_test (from X): {len(y_test_from_X_np)}")
    else:
        print("  XGBoost model not found — skipping cross-check")

    # ── 8. Verify that Elo predictions make sense ──────
    print("\n-- 8. Elo prediction sanity check ---------------")
    print(f"  Elo prediction distribution:")
    print(f"    Home win probs: mean={test_probs[:, 2].mean():.4f}, "
          f"min={test_probs[:, 2].min():.4f}, max={test_probs[:, 2].max():.4f}")
    print(f"    Draw probs: mean={test_probs[:, 1].mean():.4f}")
    print(f"    Away win probs: mean={test_probs[:, 0].mean():.4f}")

    # Check first few predictions
    print(f"\n  First 5 Elo predictions vs actual:")
    for i in range(min(5, len(test_probs))):
        home = df_sorted_test.iloc[i]["home_team"]
        away = df_sorted_test.iloc[i]["away_team"]
        actual = df_sorted_test.iloc[i]["result"]
        pred_home = test_probs[i, 2]
        pred_draw = test_probs[i, 1]
        pred_away = test_probs[i, 0]
        pred_label = "H" if pred_home > pred_draw and pred_home > pred_away else \
                     "D" if pred_draw > pred_away else "A"
        print(f"    {home} vs {away}: Elo={pred_home:.3f}/{pred_draw:.3f}/{pred_away:.3f} "
              f"(pred={pred_label}, actual={actual})")

    print("\n" + "=" * 72)
    print("  DEBUG COMPLETE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
