"""
train_market_models.py — Train market-specific tree models for O/U 2.5 and BTTS.

Trains XGBoost, LightGBM, and CatBoost separately for each binary market,
so they learn DIRECTLY on O/U 2.5 and BTTS targets rather than deriving
them from 1X2 predictions.

Saves models to ``models/per_league/{league}/``:
  - xgboost_ou.joblib / lightgbm_ou.joblib / catboost_ou.joblib
  - xgboost_btts.joblib / lightgbm_btts.joblib / catboost_btts.joblib

Usage:
    python scripts/train_market_models.py --leagues F1
    python scripts/train_market_models.py --leagues F1 --over-only
    python scripts/train_market_models.py --leagues F1 --btts-only
    python scripts/train_market_models.py --leagues E0 F1 D1 I1 SP1
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_market_models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODELS_DIR = PROJECT_ROOT / "models" / "per_league"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure project root is on sys.path for config imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # remaining 15% is test

# Minimum matches to train trees
MIN_MATCHES = 500

# Tree params (same as analyse_tree_models.py)
TREE_N_ESTIMATORS = 500
TREE_MAX_DEPTH = 6
TREE_LEARNING_RATE = 0.05
TREE_EARLY_STOP = 30


# ── Data Loading ───────────────────────────────────────

def load_league_data(league: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               season,
               home_odds, draw_odds, away_odds,
               home_xg, away_xg,
               home_shots, away_shots, home_shots_target, away_shots_target,
               home_corners, away_corners, home_fouls, away_fouls,
               home_yellow, away_yellow, home_red, away_red
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# ── Chronological Split ────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


# ── Feature Preparation ────────────────────────────────

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for build_features pipeline."""
    df = df.copy()

    # Target for 1X2 (needed by pipeline even if we don't use it)
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"A": 0, "D": 1, "H": 2})

    # Season
    if "season" not in df.columns:
        df["season"] = df["date"].dt.year.astype(str)
    else:
        mask = df["season"].isna() | (df["season"].astype(str).str.strip() == "")
        if mask.any():
            df.loc[mask, "season"] = df.loc[mask, "date"].dt.year.astype(str)

    if "league" not in df.columns:
        df["league"] = "UNKNOWN"

    # Numeric conversion
    num_cols = [
        "home_goals", "away_goals", "home_odds", "draw_odds", "away_odds",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_yellow", "away_yellow", "home_red", "away_red",
    ]
    for col in num_cols:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


# ── Binary Targets ─────────────────────────────────────

def make_over_target(df: pd.DataFrame) -> pd.Series:
    """Return binary target: 1 = Over 2.5, 0 = Under 2.5."""
    return ((df["home_goals"] + df["away_goals"]) > 2.5).astype(int)


def make_btts_target(df: pd.DataFrame) -> pd.Series:
    """Return binary target: 1 = Both Teams Scored, 0 = No BTTS."""
    return ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)


# ── Model Training ─────────────────────────────────────

def _train_binary_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> Any | None:
    """Train a binary classifier (XGBoost / LightGBM / CatBoost) for O/U or BTTS."""
    t0 = time.time()

    try:
        if model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=TREE_N_ESTIMATORS,
                max_depth=TREE_MAX_DEPTH,
                learning_rate=TREE_LEARNING_RATE,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                gamma=0.1,
                min_child_weight=5,
                early_stopping_rounds=TREE_EARLY_STOP,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weight,
                verbose=False,
            )

        elif model_type == "lightgbm":
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                objective="binary",
                metric="binary_logloss",
                n_estimators=TREE_N_ESTIMATORS,
                max_depth=TREE_MAX_DEPTH,
                learning_rate=TREE_LEARNING_RATE,
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
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(TREE_EARLY_STOP), lgb.log_evaluation(0)],
                sample_weight=sample_weight,
            )

        elif model_type == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                iterations=TREE_N_ESTIMATORS,
                depth=TREE_MAX_DEPTH,
                learning_rate=TREE_LEARNING_RATE,
                loss_function="Logloss",
                eval_metric="Logloss",
                early_stopping_rounds=TREE_EARLY_STOP,
                verbose=0,
                random_seed=42,
                thread_count=-1,
            )
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                sample_weight=sample_weight,
                use_best_model=True,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        elapsed = time.time() - t0

        # Evaluate
        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        train_brier = float(np.mean((train_probs - y_train) ** 2))
        val_brier = float(np.mean((val_probs - y_val) ** 2))
        val_acc = float(np.mean((val_probs > 0.5).astype(float) == y_val))
        from sklearn.metrics import log_loss as sk_ll
        val_ll = float(sk_ll(y_val, val_probs))

        logger.info(
            "    %s: val_brier=%.4f, val_acc=%.1f%%, val_ll=%.4f (%.1fs)",
            model_type, val_brier, val_acc * 100, val_ll, elapsed,
        )
        return model

    except Exception as exc:
        logger.warning("    %s training failed: %s", model_type, exc)
        return None


def compute_sample_weights(
    df: pd.DataFrame,
    date_col: str = "date",
    halflife_days: float = 730.0,
) -> np.ndarray | None:
    if date_col not in df.columns or halflife_days <= 0:
        return None
    dates = pd.to_datetime(df[date_col])
    ref = dates.max() + pd.Timedelta(days=1)
    days_ago = (ref - dates).dt.days.values.astype(float)
    days_ago = np.maximum(days_ago, 0)
    weights = np.exp(-np.log(2) * days_ago / halflife_days)
    return weights


# ── Save Model ─────────────────────────────────────────

def save_model(model: Any, league: str, market: str, model_type: str):
    import joblib
    model_dir = MODELS_DIR / league
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_type}_{market}.joblib"
    path = model_dir / filename
    joblib.dump(model, path)
    logger.info("    Saved %s", path)


# ── Main ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train market-specific tree models")
    parser.add_argument("--leagues", nargs="+", default=["F1"],
                        help="League codes to train (default: F1)")
    parser.add_argument("--over-only", action="store_true",
                        help="Train O/U models only (skip BTTS)")
    parser.add_argument("--btts-only", action="store_true",
                        help="Train BTTS models only (skip O/U)")
    parser.add_argument("--no-sample-weights", action="store_true",
                        help="Disable time-decay sample weights")
    parser.add_argument("--k", type=float, default=730.0,
                        help="Sample weight halflife in days (default: 730)")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  MARKET-SPECIFIC TREE MODEL TRAINING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    train_over = not args.btts_only
    train_btts = not args.over_only

    all_results: list[dict[str, Any]] = []

    for league in args.leagues:
        print()
        print(f"  {'─' * 60}")
        print(f"  LEAGUE: {league}")
        print(f"  {'─' * 60}")

        # ── 1. Load & split data ──
        df = load_league_data(league)
        if len(df) < MIN_MATCHES:
            logger.warning("  Only %d matches (need %d) — skipping", len(df), MIN_MATCHES)
            continue

        train_df, val_df, test_df = chronological_split(df)
        logger.info("  Data: %d train / %d val / %d test", len(train_df), len(val_df), len(test_df))

        # ── 2. Build features ──
        logger.info("  Building features...")
        train_prep = _prepare_df(train_df)
        val_prep = _prepare_df(val_df)
        test_prep = _prepare_df(test_df)
        combined = pd.concat([train_prep, val_prep, test_prep], ignore_index=True)

        from config import config as cfg
        _orig_enabled = {
            "weather": cfg.weather.enabled,
            "referee": cfg.referee.enabled,
            "extended": cfg.extended_features.enabled,
        }
        cfg.weather.enabled = False
        cfg.referee.enabled = False
        cfg.extended_features.enabled = False

        X_full = None
        try:
            from src.feature_engineering import build_features
            # Build features with the FULL combined dataset, then slice
            X_full, y_1x2_full = build_features(combined, is_training=True, use_cache=False)
        except Exception as exc:
            logger.warning("  Feature engineering failed: %s", exc)
            cfg.weather.enabled = _orig_enabled["weather"]
            cfg.referee.enabled = _orig_enabled["referee"]
            cfg.extended_features.enabled = _orig_enabled["extended"]
            continue
        finally:
            cfg.weather.enabled = _orig_enabled["weather"]
            cfg.referee.enabled = _orig_enabled["referee"]
            cfg.extended_features.enabled = _orig_enabled["extended"]

        if X_full is None or len(X_full) == 0:
            logger.warning("  No features generated — skipping")
            continue

        # Drop _row_id if present (leaks row ordering — time leakage risk)
        if "_row_id" in X_full.columns:
            X_full = X_full.drop(columns=["_row_id"])

        # Split feature matrix
        n_train = len(train_prep)
        n_val = len(val_prep)
        X_train_f = X_full.iloc[:n_train].copy()
        X_val_f = X_full.iloc[n_train:n_train + n_val].copy()
        X_test_f = X_full.iloc[n_train + n_val:].copy()

        logger.info("  Feature matrix: %d cols", X_full.shape[1])

        # Drop date (datetime not supported by sklearn)
        date_col = "date" if "date" in X_train_f.columns else None
        if date_col:
            X_train_f = X_train_f.drop(columns=[date_col])
            X_val_f = X_val_f.drop(columns=[date_col])
            X_test_f = X_test_f.drop(columns=[date_col])

        # ── 3. Compute sample weights ──
        sample_weights = None
        if not args.no_sample_weights:
            sw = compute_sample_weights(train_prep, halflife_days=args.k)
            if sw is not None:
                sample_weights = sw
                logger.info("  Sample weights: mean=%.3f, min=%.3f, max=%.3f (halflife=%.0f d)",
                             float(np.mean(sw)), float(np.min(sw)),
                             float(np.max(sw)), args.k)

        # ── 4. Train market models ──
        league_results: dict[str, Any] = {
            "league": league,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": X_test_f.shape[0],
        }

        tree_types = ["xgboost", "lightgbm", "catboost"]

        # ── Over 2.5 ──
        if train_over:
            print()
            print(f"  ── [MARKET] Over 2.5 ──")
            y_over_train = make_over_target(train_prep).values
            y_over_val = make_over_target(val_prep).values
            y_over_test = make_over_target(test_prep).values

            over_rate = float(y_over_train.mean())
            logger.info("  Over 2.5 rate: train=%.1f%%, val=%.1f%%, test=%.1f%%",
                         over_rate * 100, float(y_over_val.mean()) * 100,
                         float(y_over_test.mean()) * 100)

            for mt in tree_types:
                logger.info("  Training %s (OU)...", mt)
                model = _train_binary_model(
                    mt, X_train_f, y_over_train, X_val_f, y_over_val,
                    sample_weight=sample_weights,
                )
                if model is not None:
                    save_model(model, league, "ou", mt)
                    # Evaluate on test set
                    test_probs = model.predict_proba(X_test_f)[:, 1]
                    test_brier = float(np.mean((test_probs - y_over_test) ** 2))
                    test_acc = float(np.mean((test_probs > 0.5).astype(float) == y_over_test))
                    from sklearn.metrics import log_loss as sk_ll
                    test_ll = float(sk_ll(y_over_test, test_probs))
                    logger.info("    Test: brier=%.4f, acc=%.1f%%, ll=%.4f",
                                 test_brier, test_acc * 100, test_ll)
                    league_results[f"{mt}_ou_brier"] = round(test_brier, 4)
                    league_results[f"{mt}_ou_acc"] = round(test_acc, 4)
                else:
                    league_results[f"{mt}_ou_brier"] = None

        # ── BTTS ──
        if train_btts:
            print()
            print(f"  ── [MARKET] BTTS ──")
            y_btts_train = make_btts_target(train_prep).values
            y_btts_val = make_btts_target(val_prep).values
            y_btts_test = make_btts_target(test_prep).values

            btts_rate = float(y_btts_train.mean())
            logger.info("  BTTS rate: train=%.1f%%, val=%.1f%%, test=%.1f%%",
                         btts_rate * 100, float(y_btts_val.mean()) * 100,
                         float(y_btts_test.mean()) * 100)

            for mt in tree_types:
                logger.info("  Training %s (BTTS)...", mt)
                model = _train_binary_model(
                    mt, X_train_f, y_btts_train, X_val_f, y_btts_val,
                    sample_weight=sample_weights,
                )
                if model is not None:
                    save_model(model, league, "btts", mt)
                    # Evaluate on test set
                    test_probs = model.predict_proba(X_test_f)[:, 1]
                    test_brier = float(np.mean((test_probs - y_btts_test) ** 2))
                    test_acc = float(np.mean((test_probs > 0.5).astype(float) == y_btts_test))
                    from sklearn.metrics import log_loss as sk_ll
                    test_ll = float(sk_ll(y_btts_test, test_probs))
                    logger.info("    Test: brier=%.4f, acc=%.1f%%, ll=%.4f",
                                 test_brier, test_acc * 100, test_ll)
                    league_results[f"{mt}_btts_brier"] = round(test_brier, 4)
                    league_results[f"{mt}_btts_acc"] = round(test_acc, 4)
                else:
                    league_results[f"{mt}_btts_brier"] = None

        all_results.append(league_results)

    # ── Summary ──
    if all_results:
        print()
        print("=" * 80)
        print("  CROSS-LEAGUE MARKET-SPECIFIC MODEL COMPARISON")
        print("=" * 80)
        print()

        # Table header
        markets = ["ou", "btts"] if train_over and train_btts \
                  else ["ou"] if train_over else ["btts"]
        for market in markets:
            print(f"  [{market.upper()}]")
            print(f"  {'League':<8} {'XGB Brier':>10} {'XGB Acc':>8} "
                  f"{'LGB Brier':>10} {'LGB Acc':>8} "
                  f"{'Cat Brier':>10} {'Cat Acc':>8}")
            print(f"  {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")
            for r in all_results:
                league = r["league"]
                xb_b = r.get(f"xgboost_{market}_brier", "—")
                xb_a = r.get(f"xgboost_{market}_acc", "—")
                lb_b = r.get(f"lightgbm_{market}_brier", "—")
                lb_a = r.get(f"lightgbm_{market}_acc", "—")
                cb_b = r.get(f"catboost_{market}_brier", "—")
                cb_a = r.get(f"catboost_{market}_acc", "—")
                xb_b_s = f"{xb_b:.4f}" if isinstance(xb_b, float) else str(xb_b)
                xb_a_s = f"{xb_a*100:.0f}%" if isinstance(xb_a, float) else str(xb_a)
                lb_b_s = f"{lb_b:.4f}" if isinstance(lb_b, float) else str(lb_b)
                lb_a_s = f"{lb_a*100:.0f}%" if isinstance(lb_a, float) else str(lb_a)
                cb_b_s = f"{cb_b:.4f}" if isinstance(cb_b, float) else str(cb_b)
                cb_a_s = f"{cb_a*100:.0f}%" if isinstance(cb_a, float) else str(cb_a)
                print(f"  {league:<8} {xb_b_s:>10} {xb_a_s:>8} "
                      f"{lb_b_s:>10} {lb_a_s:>8} "
                      f"{cb_b_s:>10} {cb_a_s:>8}")
            print()

        # Save JSON report
        report_path = REPORTS_DIR / "market_models_comparison.json"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "n_estimators": TREE_N_ESTIMATORS,
                    "max_depth": TREE_MAX_DEPTH,
                    "learning_rate": TREE_LEARNING_RATE,
                    "early_stopping_rounds": TREE_EARLY_STOP,
                    "sample_weight_halflife_days": args.k if not args.no_sample_weights else 0,
                },
                "results": all_results,
            }, f, indent=2)
        logger.info("Report saved to %s", report_path)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
