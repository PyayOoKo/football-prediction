"""
Train & Predict — World Cup Edition (Multi-Tournament).

Trains an XGBoost model on combined historical World Cup match data
(2018 + 2022 + 2026 completed matches) and generates predictions for
the 2026 knockout rounds.

Usage:
    python train_worldcup.py
    python train_worldcup.py --model lr       # Logistic Regression baseline
    python train_worldcup.py --skip-train      # Use existing saved model
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Must configure BEFORE importing src modules that read config
from config import config

# ── Configure for World Cup data ─────────────────────────
# Disable features that don't apply to international tournament data
config.features.include_h2h = True
config.features.include_league_position = False
config.odds.compute_consensus = False
config.odds.warn_missing = False
config.player_info.enabled = True
config.player_features.enabled = True
config.player_features.rolling_windows = (5, 10)
config.xg.warn_missing = False
# xG data is now available from StatsBomb for 2018 & 2022 World Cups
# Player data loaded from data/external/players.csv (collect_player_data.py)
config.xg.compute_xpts = True
config.elo.regress_to_mean = True
# Neutral venue — small home advantage for Elo training signal
# Host nations (Russia 2018, Qatar 2022) had real home advantage
# Swap-and-average at prediction time neutralises this bias
config.elo.home_advantage = 50  # neutral venue base (host gets +50 in Elo)  # neutral venue base (host gets +50 in Elo)  # neutral venue base (host gets +50 in Elo)  # neutral venue base (host gets +50 in Elo)
# Manual Elo penalty — user expressed skepticism about Morocco's recent form
config.elo.adjustments = {"Morocco": 50}

# Enable Dixon-Coles features (small WC dataset = fast MLE)
config.dixon_coles.enabled = True
config.dixon_coles.refit_every = 100  # more responsive for WC data

# Improved configuration for 658-match dataset with xAG + enhanced Elo
config.train.n_estimators = 300  # more data can support more trees
config.train.max_depth = 5      # deeper trees for more complex patterns
config.train.min_samples_leaf = 3
config.train.learning_rate = 0.05
config.train.subsample = 0.8
config.train.reg_lambda = 3.0
config.train.reg_alpha = 0.5
config.train.cv_folds = 5

# ── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_worldcup")


# ═══════════════════════════════════════════════════════════
#  Config-derived paths
# ═══════════════════════════════════════════════════════════

# These used to be hardcoded constants. They are now centralised in
# config.worldcup so every script that needs them can use the same
# single source of truth. Change them in one place (config.py or
# at runtime via config.worldcup.* = ...) and everything follows.

WORLDCUP_CSV = Path(config.worldcup.data_path)
PREDICTIONS_DIR = Path(config.worldcup.predictions_dir)
MODEL_SAVE_NAME = config.worldcup.model_save_name
LABEL_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}
RESULT_TO_TARGET = {"H": 2, "D": 1, "A": 0}


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and predict on 2026 World Cup data",
    )
    parser.add_argument(
        "--model", default="lgb", choices=["lgb", "xgb", "lr", "rf"],
        help="Model type (default: lgb — LightGBM, best backtest performer)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training -- use existing saved model",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=3,
        help="Number of CV folds (default: 3, small data)",
    )
    return parser.parse_args(argv)


def _add_target_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'target' column from 'result' (H=2, D=1, A=0, NaN=-1)."""
    df = df.copy()
    df["target"] = df["result"].map(RESULT_TO_TARGET).fillna(-1).astype("int8")
    return df


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    # Map model type
    model_type_map = {"lgb": "lightgbm", "xgb": "xgboost", "lr": "logistic_regression", "rf": "random_forest"}
    config.train.model_type = model_type_map[args.model]

    print("=" * 72)
    print(f"  WORLD CUP - TRAIN & PREDICT (Multi-Tournament)")
    print(f"  Model: {config.train.model_type.upper()}")
    print(f"  Data:  {WORLDCUP_CSV}")
    print("=" * 72)

    # ── 1. Load data ─────────────────────────────────────
    if not WORLDCUP_CSV.exists():
        print(f"\n  [X] World Cup data not found at {WORLDCUP_CSV}")
        print("    Run:  from src.data_collection import collect_worldcup")
        print("          collect_worldcup()")
        return 1

    print(f"\n  Loading {WORLDCUP_CSV} ...")
    df = pd.read_csv(WORLDCUP_CSV, low_memory=False, parse_dates=["date"])
    print(f"  [*] {len(df)} rows x {len(df.columns)} cols")

    # Add the 'target' column that build_features() expects
    df = _add_target_col(df)

    # Split completed vs upcoming
    completed_mask = df["result"].notna()
    df_completed = df[completed_mask].copy()
    df_upcoming = df[~completed_mask].copy()

    print(f"  [*] Completed: {len(df_completed)} matches")
    print(f"  [*] Upcoming:  {len(df_upcoming)} matches")
    _t_load = time.time()
    print(f"  [TIME] Load data: {_t_load - t0:.1f}s")

    # ── 2. Build features on completed data ──────────────
    _t_feat = time.time()
    print(f"\n  Building features ...")
    from src.feature_engineering import build_features

    X, y = build_features(df_completed, is_training=True)
    print(f"  [*] Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  [TIME] Build features (train): {time.time() - _t_feat:.1f}s")

    if X.shape[0] < 10:
        print(f"  [X] Too few matches with valid features ({X.shape[0]}). Can't train.")
        return 1

    # ── 3. Split chronologically ─────────────────────────
    _t_split = time.time()
    from src.feature_engineering import train_val_test_split

    print(f"\n  Splitting chronologically (70/15/15) ...")
    splits = train_val_test_split(X, y)
    print(f"  [*] Train: {len(splits['X_train'])}  |  "
          f"Val: {len(splits['X_val'])}  |  "
          f"Test: {len(splits['X_test'])}")
    print(f"  [TIME] Split: {time.time() - _t_split:.1f}s")

    # ── 4. Hyper-parameter tuning (lightweight for small data) ──
    if not args.skip_train:
        _t_tune = time.time()
        from src.train import tune_hyperparameters

        print(f"\n  Tuning hyper-parameters ({args.cv_folds}-fold time-series CV) ...")
        try:
            best_params = tune_hyperparameters(
                splits["X_train"], splits["y_train"],
                n_folds=args.cv_folds, n_iter=20, verbose=False,
            )
            print(f"  [*] Best params: {best_params}")
            print(f"  [TIME] Hyper-parameter tuning: {time.time() - _t_tune:.1f}s")
            for key, val in best_params.items():
                if hasattr(config.train, key):
                    setattr(config.train, key, val)
        except Exception as exc:
            print(f"  [W] Tuning failed: {exc} - using defaults")

    # ── 5. Train final model ─────────────────────────────
    if args.skip_train:
        _t_load_model = time.time()
        print(f"\n  Loading existing model ...")
        from src.train import load_model
        model = load_model(MODEL_SAVE_NAME)
        print(f"  [*] Model loaded")
        print(f"  [TIME] Load model: {time.time() - _t_load_model:.1f}s")
    else:
        _t_train = time.time()
        from src.train import train_model

        print(f"\n  Training {config.train.model_type} ...")
        model, history = train_model(
            splits["X_train"], splits["y_train"],
            splits["X_val"], splits["y_val"],
        )
        train_loss = history.get("train_loss", [None])[0]
        val_loss = history.get("val_loss", [None])[0]
        val_acc = history.get("val_accuracy", [None])[0]
        if train_loss is not None:
            print(f"  [*] Train log-loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  [*] Val log-loss:   {val_loss:.4f}")
        if val_acc is not None:
            print(f"  [*] Val accuracy:   {val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"  [TIME] Train model: {time.time() - _t_train:.1f}s")

        # Save
        _t_save = time.time()
        from src.train import save_model
        save_model(model, MODEL_SAVE_NAME)
        print(f"  [*] Model saved: models/{MODEL_SAVE_NAME}")
        print(f"  [TIME] Save model: {time.time() - _t_save:.1f}s")

    # ── 6. Evaluate on test set ──────────────────────────
    _t_eval = time.time()
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X_test = splits["X_test"]
    y_test = splits["y_test"]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 72}")
    print(f"  TEST SET RESULTS")
    print(f"{'=' * 72}")
    print(f"\n  Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Correct: {int(accuracy * len(y_test))} / {len(y_test)}")

    baseline_home = (y_test == 2).mean()
    print(f"  Naive baseline (always Home): {baseline_home*100:.1f}%")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"    {'':>12} {'Away Win':>10} {'Draw':>10} {'Home Win':>10}")
    print(f"    {'-' * 42}")
    for i, label in enumerate(["Away Win", "Draw", "Home Win"]):
        row = f"  {label:>10}"
        for j in range(3):
            row += f"{cm[i, j]:>10}"
        print(row)

    print(f"\n  Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Away Win", "Draw", "Home Win"],
        digits=3,
    ))            # Feature importance
    print(f"\n{'FEATURE IMPORTANCE':-^72}")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        print(f"\n    {'Rank':<6} {'Feature':<38} {'Importance':>12}")
        print(f"    {'-' * 56}")
        for rank, idx in enumerate(indices, 1):
            print(f"    {rank:<6} {X.columns[idx]:<38} {importances[idx]:>12.4f}")
    print(f"  [TIME] Evaluate: {time.time() - _t_eval:.1f}s")

    # ── 7. Predict upcoming matches ─────────────────────
    _t_predict = time.time()
    print(f"\n{'=' * 72}")
    print(f"  PREDICTING UPCOMING MATCHES")
    print(f"{'=' * 72}")

    # Import placeholder check from the source module
    from src.data_collection.sources.worldcup import is_placeholder_team

    # Filter out upcoming matches with placeholder teams
    predict_mask = df_upcoming.apply(
        lambda r: not is_placeholder_team(r["home_team"])
                  and not is_placeholder_team(r["away_team"]),
        axis=1,
    )
    df_predictable = df_upcoming[predict_mask].copy()
    df_placeholder = df_upcoming[~predict_mask].copy()

    print(f"\n  Predictable matches:  {len(df_predictable)} (both teams known)")
    print(f"  Placeholder matches:  {len(df_placeholder)} (opponent TBD)")

    if len(df_predictable) > 0:
        # ── Neutral venue prediction via swap-and-average ──
        # World Cup knockout matches are at neutral venues, so "home" is
        # just the first-listed team — meaningless. To eliminate positional
        # bias, we predict each match twice (once with each team as "home")
        # and average the probabilities.
        #
        # build_features() sorts by ["date", "home_team"], so the row order
        # differs between original and swapped calls. We use _pred_id to
        # track and re-align rows after prediction.

        # Create swapped copy with _pred_id tracking
        n_pred = len(df_predictable)
        df_predictable["_pred_id"] = np.arange(n_pred)
        df_predictable["_swap_group"] = 0  # 0 = original order
        df_swapped = df_predictable.copy()
        df_swapped["home_team"] = df_predictable["away_team"].values
        df_swapped["away_team"] = df_predictable["home_team"].values
        df_swapped["home_goals"] = np.nan
        df_swapped["away_goals"] = np.nan
        df_swapped["result"] = np.nan
        df_swapped["target"] = -1
        df_swapped["_swap_group"] = 1  # 1 = swapped order

        # Build features ONCE on completed + both orderings
        _t_feat_pred = time.time()
        df_combined = pd.concat([df_completed, df_predictable, df_swapped], ignore_index=True)
        X_all, _ = build_features(df_combined, is_training=True)
        print(f"  [TIME] Build features (predict, combined orig+swapped): {time.time() - _t_feat_pred:.1f}s")

        # Split back into original and swapped order
        pred_start = len(df_completed)
        X_pred_all = X_all.iloc[pred_start:].copy()
        swap_group = X_pred_all.pop("_swap_group").values

        orig_mask = swap_group == 0
        X_orig_pred = X_pred_all[orig_mask].copy()
        X_swp_pred = X_pred_all[~orig_mask].copy()
        pred_order_orig = X_orig_pred.pop("_pred_id").values
        pred_order_swp = X_swp_pred.pop("_pred_id").values

        if len(X_orig_pred) == 0:
            print(f"\n  [W] No feature rows generated for predictable matches.")
        else:
            _t_pred_model = time.time()
            probs_orig = model.predict_proba(X_orig_pred)  # [away, draw, home]
            probs_swp = model.predict_proba(X_swp_pred)    # [away, draw, home]

            # Re-align both probability arrays by _pred_id before averaging
            orig_sort = np.argsort(pred_order_orig)
            swp_sort = np.argsort(pred_order_swp)
            probs_orig = probs_orig[orig_sort]
            probs_swp = probs_swp[swp_sort]

            # Average: orig home_win = P(A wins), swp away_win = P(A wins under swapped naming)
            avg_home = (probs_orig[:, 2] + probs_swp[:, 0]) / 2  # P(original home team wins)
            avg_draw = (probs_orig[:, 1] + probs_swp[:, 1]) / 2  # P(draw)
            avg_away = (probs_orig[:, 0] + probs_swp[:, 2]) / 2  # P(original away team wins)

            print(f"  [TIME] Model predict (swap-average + combine): {time.time() - _t_pred_model:.1f}s")

            probs = np.column_stack([avg_away, avg_draw, avg_home])
            preds = np.argmax(probs, axis=1)
            confidences = probs.max(axis=1)

            # Build output
            cols = ["date", "home_team", "away_team", "round", "ground"]
            output = df_predictable[[c for c in cols if c in df_predictable.columns]].copy()
            output["prediction"] = [LABEL_MAP[p] for p in preds]
            output["home_win_prob"] = avg_home
            output["draw_prob"] = avg_draw
            output["away_win_prob"] = avg_away
            output["confidence"] = confidences

            print(f"\n  {'=' * 72}")
            print(              f"  KNOCKOUT ROUND PREDICTIONS (neutral venue — averaged both orderings)")
            print(f"  {'=' * 72}")
            print(f"  {'Date':<14} {'Team 1':<22} {'Team 2':<22} "
                  f"{'Prediction':<16} {'Conf':<8}")
            print(f"  {'-' * 80}")
            for _, r in output.iterrows():
                wt = r["home_win_prob"]
                dt = r["draw_prob"]
                lt = r["away_win_prob"]
                if max(wt, dt, lt) == dt:
                    outcome = "Draw"
                else:
                    favorite = r["home_team"] if wt >= lt else r["away_team"]
                    outcome = f"{favorite} wins"
                print(f"  {str(r['date'])[:10]:<14} {r['home_team']:<22} "
                      f"{r['away_team']:<22} {outcome:<16} "
                      f"{r['confidence']:.0%}")

            # Print detailed probabilities table
            print(f"\n  Detailed probabilities:")
            print(f"  {'Date':<14} {'Match':<46} {'Team 1':<12} {'Draw':<12} {'Team 2':<12}")
            print(f"  {'-' * 82}")
            for _, r in output.iterrows():
                match = f"{r['home_team']} vs {r['away_team']}"
                print(f"  {str(r['date'])[:10]:<14} {match:<46} "
                      f"{r['home_win_prob']*100:>5.1f}%   "
                      f"{r['draw_prob']*100:>5.1f}%   "
                      f"{r['away_win_prob']*100:>5.1f}%")

            # Save to CSV
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = PREDICTIONS_DIR / config.worldcup.predictions_file
            output.to_csv(out_path, index=False)
            print(f"\n  [*] Predictions saved to {out_path}")

            # ── Poisson goal predictions (from historical goal rates) ──
            _t_poisson = time.time()
            # Use the PoissonModel directly on completed data, then predict
            # each upcoming match to avoid NaN propagation from upcoming rows.
            from src.poisson_model import PoissonModel

            poisson = PoissonModel(min_matches=0, max_goals=8)
            poisson.add_poisson_features(df_completed.copy())

            exp_home_goals: list[float] = []
            exp_away_goals: list[float] = []

            for _, row in df_predictable.iterrows():
                h = row["home_team"]
                a = row["away_team"]
                lam_h, lam_a = poisson.expected_goals(h, a)
                exp_home_goals.append(round(lam_h, 3))
                exp_away_goals.append(round(lam_a, 3))

            output["expected_home_goals"] = exp_home_goals
            output["expected_away_goals"] = exp_away_goals
            print(f"  [TIME] Poisson goal predictions: {time.time() - _t_poisson:.1f}s")

            # Most likely scoreline
            def _score_str(h: float, a: float) -> str:
                if pd.isna(h) or pd.isna(a):
                    return "?"
                return f"{int(round(h))}-{int(round(a))}"

            output["predicted_score"] = [
                _score_str(h, a)
                for h, a in zip(exp_home_goals, exp_away_goals)
            ]

            # ── Print POISSON GOAL PREDICTIONS ──
            print(f"\n  {'=' * 72}")
            print(f"  POISSON GOAL PREDICTIONS (from historical goal rates)")
            print(f"  {'=' * 72}")
            print(f"  {'Date':<14} {'Match':<46} {'Exp Goals':<22} {'Score':<10} {'O/U 2.5':<10}")
            print(f"  {'-' * 92}")
            for _, r in output.iterrows():
                match = f"{r['home_team']} vs {r['away_team']}"
                lam_h = r["expected_home_goals"]
                lam_a = r["expected_away_goals"]
                total_goals = lam_h + lam_a
                over_under = "OVER" if total_goals > 2.5 else "UNDER"
                score = r["predicted_score"]
                goals_str = f"{r['home_team']}: {lam_h:.2f} / {r['away_team']}: {lam_a:.2f}"
                print(f"  {str(r['date'])[:10]:<14} {match:<46} "
                      f"{goals_str:<22} "
                      f"{score:<10} {over_under}")

            # Summary stats
            home_preds = int((preds == 2).sum())
            draw_preds = int((preds == 1).sum())
            away_preds = int((preds == 0).sum())
            total_preds = len(preds)
            print(f"  \n  Prediction summary:")
            print(f"    Team 1 wins: {home_preds} ({home_preds/total_preds*100:.0f}%)")
            print(f"    Draws:       {draw_preds} ({draw_preds/total_preds*100:.0f}%)")
            print(f"    Team 2 wins: {away_preds} ({away_preds/total_preds*100:.0f}%)")
            print(f"    Avg confidence: {confidences.mean():.1%}")

    if len(df_placeholder) > 0:
        print(f"\n  [W] {len(df_placeholder)} matches have TBD opponents:")
        for _, r in df_placeholder.iterrows():
            h = r["home_team"]
            a = r["away_team"]
            home = h if is_placeholder_team(h) else h
            away = a if is_placeholder_team(a) else a
            print(f"    {str(r['date'])[:10]} | {home:<22} vs {away:<22} | {r['round']}")
        print(f"  -> Re-run once opponents are known and data is updated")
    print(f"  [TIME] Predict step total: {time.time() - _t_predict:.1f}s")

    # ── Summary ──────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  DONE - {elapsed:.1f}s")
    print(f"  Model:  {config.train.model_type}")
    print(f"  Trained on: {X.shape[0]} matches x {X.shape[1]} features")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Predictions: {len(df_predictable)} matches -> "
          f"{PREDICTIONS_DIR / config.worldcup.predictions_file}")
    print(f"{'=' * 72}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
