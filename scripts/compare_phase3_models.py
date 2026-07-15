#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 3 Model Comparison — Poisson vs Dixon-Coles vs Elo vs Baseline     ║
║                                                                           ║
║  Trains all 4 models on identical training data and evaluates them on     ║
║  the same test set.  Generates a leaderboard CSV identifying the best     ║
║  model for each market (1X2, BTTS, Over/Under).                          ║
║                                                                           ║
║  Outputs:                                                                 ║
║  - reports/phase3_leaderboard_{timestamp}.csv                             ║
║  - reports/phase3_comparison_{timestamp}.json                             ║
║                                                                           ║
║  Usage:                                                                   ║
║      python scripts/compare_phase3_models.py                              ║
║      python scripts/compare_phase3_models.py --data data/raw/results.csv  ║
║      python scripts/compare_phase3_models.py --quiet                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from config import config
from src.poisson_model import PoissonModel
from src.dixon_coles import DixonColesModel
from src.elo import EloSystem

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

_DEFAULT_TRAIN_END = "2023-01-01"
_DEFAULT_TEST_END = "2026-12-31"
_REQUIRED_COLS = ["date", "home_team", "away_team", "result", "home_goals", "away_goals"]
_REPORT_DIR = PROJECT_ROOT / "reports"

# Hyperparameters (from individual validations)
ELO_BEST_PARAMS = {"k": 40, "home_advantage": 50, "regress_to_mean": False, "draw_k": 0.20}
DC_PARAMS = {"decay_halflife_days": 1460, "use_importance": True, "max_goals_table": 8,
             "regress_prior": True, "prior_strength": 0.01}
POISSON_PARAMS = {"min_matches": 0, "max_goals": 8}

# Markets
MARKETS = ["1X2", "BTTS", "Over/Under 2.5"]
MARKET_METRICS = {
    "1X2": ["accuracy", "log_loss", "brier_score"],
    "BTTS": ["btts_accuracy", "btts_brier"],
    "Over/Under 2.5": ["over_under_2_5_accuracy", "over_under_2_5_brier"],
}


# ═══════════════════════════════════════════════════════════
#  Data loading & splitting
# ═══════════════════════════════════════════════════════════


def load_data(path: str | Path, min_date: str | None = "2010-01-01") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_dates = df["date"].copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    if df["date"].isna().sum() > len(df) * 0.5:
        df["date"] = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")

    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(subset=["date"])

    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)].copy()

    # Only completed matches with known results
    df = df[df["result"].notna() & df["result"].isin(["H", "D", "A"])].copy()

    logger.info("Loaded %d matches from %s to %s", len(df),
                df["date"].min().strftime("%Y-%m-%d"),
                df["date"].max().strftime("%Y-%m-%d"))
    return df


def chronological_split(df: pd.DataFrame, train_end: str, test_end: str):
    tc = pd.Timestamp(train_end)
    tec = pd.Timestamp(test_end)
    train = df[df["date"] < tc].copy()
    test = df[(df["date"] >= tc) & (df["date"] <= tec)].copy()
    logger.info("Split: train=%d, test=%d", len(train), len(test))
    return train, test


def get_targets(df: pd.DataFrame) -> np.ndarray:
    mapping = {"A": 0, "D": 1, "H": 2}
    result = df["result"].map(mapping).fillna(-1).values.astype(int)
    return result


def check_odds_available(df: pd.DataFrame) -> list[str]:
    """Check if betting odds columns are present for CLV computation."""
    odds_keywords = ["odd", "b365", "bw", "bb", "max", "av",
                     "psh", "psd", "psa", "whh", "whd", "wha",
                     "vch", "vcd", "vca", "lbh", "lbd", "lba",
                     "sbh", "sbd", "sba"]
    found = []
    for col in df.columns:
        col_lower = col.lower().strip()
        for kw in odds_keywords:
            if col_lower.startswith(kw):
                found.append(col)
                break
    return found


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════


def compute_metrics_triple(y_true: np.ndarray, probs: np.ndarray, name: str = "Model"):
    """Compute 1X2 metrics: accuracy, log_loss, brier_score."""
    from sklearn.metrics import log_loss as sk_ll
    pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred == y_true))
    ll = float(sk_ll(y_true, probs))
    y_oh = np.zeros((len(y_true), 3))
    for i, v in enumerate(y_true):
        if 0 <= v <= 2:
            y_oh[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))
    return {"accuracy": round(accuracy, 4), "log_loss": round(ll, 4),
            "brier_score": round(brier, 4)}


def compute_btts_metrics(hg: np.ndarray, ag: np.ndarray, pred_probs: np.ndarray):
    actual = ((hg > 0) & (ag > 0)).astype(float)
    pred = (pred_probs > 0.5).astype(float)
    return {"btts_accuracy": round(float(np.mean(pred == actual)), 4),
            "btts_brier": round(float(np.mean((pred_probs - actual) ** 2)), 4)}


def compute_ou_metrics(hg: np.ndarray, ag: np.ndarray, pred_probs: np.ndarray, threshold: float = 2.5):
    actual = ((hg + ag) > threshold).astype(float)
    pred = (pred_probs > 0.5).astype(float)
    k = f"over_under_{threshold:.1f}".replace(".", "_")
    return {f"{k}_accuracy": round(float(np.mean(pred == actual)), 4),
            f"{k}_brier": round(float(np.mean((pred_probs - actual) ** 2)), 4)}


def evaluate_model_predictions(
    model: Any,
    df_test: pd.DataFrame,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Run full evaluation: 1X2, BTTS, O/U 2.5.

    Works with any model that has ``predict_matches()`` returning
    columns: away_win_prob, draw_prob, home_win_prob, btts_prob, over_2_5_prob.
    """
    hg = df_test["home_goals"].values.astype(float)
    ag = df_test["away_goals"].values.astype(float)
    y_true = get_targets(df_test)

    preds = model.predict_matches(df_test)
    probs = np.column_stack([preds["away_win_prob"].values,
                              preds["draw_prob"].values,
                              preds["home_win_prob"].values])

    metrics_1x2 = compute_metrics_triple(y_true, probs, model_name)
    btts_m = compute_btts_metrics(hg, ag, preds["btts_prob"].values)
    ou_m = compute_ou_metrics(hg, ag, preds["over_2_5_prob"].values)

    return {**metrics_1x2, **btts_m, **ou_m, "n_test": len(df_test)}


def compute_clv_metrics(
    df_test: pd.DataFrame,
    preds_df: pd.DataFrame,
    home_odds_col: str = "BbAvH",
    draw_odds_col: str = "BbAvD",
    away_odds_col: str = "BbAvA",
) -> dict[str, Any]:
    """Compute Closing Line Value if odds columns exist.

    CLV = (model_prob - market_prob) × direction_sign
    Positive CLV means the model identified value.

    Returns empty dict if odds columns not found.
    """
    if not all(c in df_test.columns for c in [home_odds_col, draw_odds_col, away_odds_col]):
        return {}

    try:
        home_odds = df_test[home_odds_col].values.astype(float)
        draw_odds = df_test[draw_odds_col].values.astype(float)
        away_odds = df_test[away_odds_col].values.astype(float)

        # Remove vig (overround) — simple proportional normalization
        market_probs = np.column_stack([
            1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds
        ])
        total_vig = market_probs.sum(axis=1)
        market_probs = market_probs / total_vig[:, np.newaxis]

        model_probs = np.column_stack([
            preds_df["away_win_prob"].values,
            preds_df["draw_prob"].values,
            preds_df["home_win_prob"].values,
        ])

        # CLV = model_prob - market_prob, averaged
        clv = model_probs - market_probs
        mean_clv = float(np.mean(clv))
        mean_abs_clv = float(np.mean(np.abs(clv)))

        return {
            "mean_clv": round(mean_clv, 4),
            "mean_abs_clv": round(mean_abs_clv, 4),
            "n_with_odds": len(df_test),
        }
    except Exception as e:
        logger.warning("CLV computation failed: %s", e)
        return {}


# ═══════════════════════════════════════════════════════════
#  Baseline model (LR with rolling features)
# ═══════════════════════════════════════════════════════════


def _compute_simple_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-free per-team rolling averages (goals scored/conceded)."""
    df = df.copy()
    n = len(df)
    home_goals_avg = np.zeros(n)
    home_conceded_avg = np.zeros(n)
    away_goals_avg = np.zeros(n)
    away_conceded_avg = np.zeros(n)
    team_stats: dict[str, list[float]] = {}

    for idx in range(n):
        home = df["home_team"].iloc[idx]
        away = df["away_team"].iloc[idx]
        hg = float(df["home_goals"].iloc[idx] or 0)
        ag = float(df["away_goals"].iloc[idx] or 0)

        def _get_avg(team: str, stat: str = "scored") -> float:
            s = team_stats.get(team)
            if s is None or s[2] == 0:
                return 0.0
            total = s[0] if stat == "scored" else s[1]
            return total / s[2]

        home_goals_avg[idx] = _get_avg(home, "scored")
        home_conceded_avg[idx] = _get_avg(home, "conceded")
        away_goals_avg[idx] = _get_avg(away, "scored")
        away_conceded_avg[idx] = _get_avg(away, "conceded")

        for team, scored, conceded in ((home, hg, ag), (away, ag, hg)):
            s = team_stats.get(team)
            if s is None:
                team_stats[team] = [scored, conceded, 1.0]
            else:
                s[0] += scored
                s[1] += conceded
                s[2] += 1.0

    total_scored = sum(s[0] for s in team_stats.values())
    total_matches = sum(s[2] for s in team_stats.values())
    global_avg = total_scored / total_matches if total_matches > 0 else 1.0

    result = pd.DataFrame({
        "home_goals_avg": home_goals_avg, "home_conceded_avg": home_conceded_avg,
        "away_goals_avg": away_goals_avg, "away_conceded_avg": away_conceded_avg,
    }, index=df.index)

    unseen = (result == 0.0).all(axis=1)
    for col in result.columns:
        result.loc[unseen, col] = global_avg

    return result


def train_baseline_lr(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict[str, Any]:
    """Train LR baseline with rolling features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss as sk_ll

    combined = pd.concat([df_train, df_test], ignore_index=True)
    feat_df = _compute_simple_rolling_features(combined)

    n_train = len(df_train)
    train_feats = feat_df.iloc[:n_train].fillna(0).values
    test_feats = feat_df.iloc[n_train:].fillna(0).values

    y_train = get_targets(df_train)
    y_test = get_targets(df_test)

    lr = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42,
                            C=1.0, class_weight="balanced")
    lr.fit(train_feats, y_train)
    test_probs = lr.predict_proba(test_feats)

    pred_labels = np.argmax(test_probs, axis=1)
    accuracy = float(np.mean(pred_labels == y_test))
    ll = float(sk_ll(y_test, test_probs))
    y_oh = np.zeros((len(y_test), 3))
    for i, v in enumerate(y_test):
        if 0 <= v <= 2:
            y_oh[i, int(v)] = 1
    brier = float(np.mean(np.sum((test_probs - y_oh) ** 2, axis=1)))

    # Wrap in dict with predict_matches() interface for unified evaluation
    class LRPredictWrapper:
        def __init__(self, lr_model, featurizer_df, n_train_):
            self._lr = lr_model
            self._feat = featurizer_df
            self._offset = n_train_

        def predict_matches(self, df):
            feats = self._feat.iloc[self._offset:self._offset + len(df)].fillna(0).values
            probs = self._lr.predict_proba(feats)
            return pd.DataFrame({
                "away_win_prob": probs[:, 0],
                "draw_prob": probs[:, 1],
                "home_win_prob": probs[:, 2],
                "btts_prob": pd.Series([0.5] * len(df)),  # LR doesn't have native BTTS
                "over_2_5_prob": pd.Series([0.5] * len(df)),  # LR doesn't have native O/U
            })

    wrapper = LRPredictWrapper(lr, feat_df, n_train)

    # Save the baseline LR model to disk for production use
    model_save_path = PROJECT_ROOT / "models" / "baseline_logistic_regression.joblib"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, model_save_path)

    return {
        "model": wrapper,
        "model_path": str(model_save_path),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "n_test": len(df_test),
        },
    }


# ═══════════════════════════════════════════════════════════
#  Model trainers
# ═══════════════════════════════════════════════════════════


def train_poisson(df_train: pd.DataFrame) -> PoissonModel:
    logger.info("Training Poisson model on %d matches...", len(df_train))
    poisson = PoissonModel(**POISSON_PARAMS)
    poisson.add_poisson_features(df_train)
    logger.info("  Poisson fitted: μ_home=%.3f, μ_away=%.3f, %d teams",
                poisson.league_avg_home, poisson.league_avg_away, len(poisson.team_strengths))
    return poisson


def train_dixon_coles(df_train: pd.DataFrame) -> DixonColesModel:
    logger.info("Training Dixon-Coles on %d matches...", len(df_train))
    dc = DixonColesModel(**DC_PARAMS)
    dc.fit(df_train, verbose=False)
    logger.info("  DC fitted: γ=%.3f, ρ=%.3f, %d teams, converged=%s",
                dc.home_advantage, dc.rho, len(dc.team_attack), dc._optimise_success)
    return dc


def train_elo(df_train: pd.DataFrame) -> EloSystem:
    logger.info("Training Elo system on %d matches...", len(df_train))
    elo = EloSystem(**ELO_BEST_PARAMS)
    elo.process_matches(df_train, home_col="home_team", away_col="away_team",
                         result_col="result", home_goals_col="home_goals",
                         away_goals_col="away_goals", season_col="season")
    logger.info("  Elo fitted: %d teams, params=%s", len(elo.ratings), ELO_BEST_PARAMS)
    return elo


# ═══════════════════════════════════════════════════════════
#  Leaderboard builder
# ═══════════════════════════════════════════════════════════


def build_leaderboard(all_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create a ranked leaderboard DataFrame using rank-based scoring.

    For each metric, models are ranked (best=highest rank), then scores
    are summed across all metrics. This avoids scale-dependence issues.

    Markets weighted: 1X2 (50%%), BTTS (25%%), Over/Under (25%%).
    """
    rows: list[dict[str, Any]] = []

    for model_name, metrics in all_metrics.items():
        row = {"Model": model_name, "n_test": metrics.get("n_test", "N/A")}
        for m in ["accuracy", "log_loss", "brier_score"]:
            row[f"1X2_{m}"] = metrics.get(m, None)
        for m in ["btts_accuracy", "btts_brier"]:
            row[f"BTTS_{m}"] = metrics.get(m, None)
        for m in ["over_under_2_5_accuracy", "over_under_2_5_brier"]:
            row[f"OU_{m}"] = metrics.get(m, None)
        for m in ["mean_clv", "mean_abs_clv"]:
            row[f"CLV_{m}"] = metrics.get(m, None)
        rows.append(row)

    df_lb = pd.DataFrame(rows)

    # Define market groups: (metrics, directions (1=higher better), weight)
    # Note: BTTS and O/U are only compared for models that can predict them
    # (Baseline LR returns placeholder 0.5 so is excluded from those markets)
    market_groups = {
        "1X2": (["accuracy", "log_loss", "brier_score"], [1, -1, -1], 0.50),
        "BTTS": (["btts_accuracy", "btts_brier"], [1, -1], 0.25),
        "OU": (["over_under_2_5_accuracy", "over_under_2_5_brier"], [1, -1], 0.25),
    }

    model_names = list(all_metrics.keys())
    combined_scores: dict[str, float] = {m: 0.0 for m in model_names}
    total_weights: dict[str, float] = {m: 0.0 for m in model_names}

    for market, (met_list, dirs, weight) in market_groups.items():
        # Determine which models can compete in this market
        # (all metrics present and not a placeholder 0.5 for all predictions)
        eligible = {}
        for mn in model_names:
            m = all_metrics.get(mn, {})
            if all(met in m and m[met] is not None for met in met_list):
                # Check it's not a placeholder — skip if all BTTS/O/U metrics = 0.5
                # (which happens when LR returns 0.5 defaults)
                if market in ("BTTS", "OU") and mn == "Baseline (LR)":
                    continue  # LR has no native BTTS/O/U predictions
                eligible[mn] = {met: m[met] for met in met_list}

        if len(eligible) < 2:
            continue  # need at least 2 models for ranking

        # Compute rank-based scores within this market
        for met, direction in zip(met_list, dirs):
            vals = [(mn, eligible[mn][met]) for mn in eligible]
            # Sort by value (ascending for lower-better, descending for higher-better)
            sorted_vals = sorted(vals, key=lambda x: x[1], reverse=(direction == 1))

            # Assign points: N-1 for first, N-2 for second, ..., 0 for last
            n_models = len(sorted_vals)
            for rank_idx, (mn, _) in enumerate(sorted_vals):
                points = n_models - 1 - rank_idx
                combined_scores[mn] += points * weight
                total_weights[mn] += weight

    # Normalize by total weight to get 0-1 range
    for mn in model_names:
        if total_weights[mn] > 0:
            avg = combined_scores[mn] / total_weights[mn]
            # Normalize to 0-1: max possible score per metric is (n-1) points
            max_possible = len(model_names) - 1
            final_score = avg / max_possible if max_possible > 0 else 0.0
            combined_scores[mn] = round(final_score, 4)
        else:
            combined_scores[mn] = 0.0

    leaderboard = df_lb.copy()
    leaderboard["Composite_Score"] = leaderboard["Model"].map(combined_scores)
    leaderboard.sort_values(["Composite_Score", "Model"], ascending=[False, True], inplace=True)
    leaderboard.reset_index(drop=True, inplace=True)
    leaderboard.index = leaderboard.index + 1
    leaderboard.index.name = "Rank"

    return leaderboard


def find_best_per_market(all_metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Identify the best model for each market and metric."""
    best: dict[str, dict[str, str]] = {}

    for market, metrics_list in MARKET_METRICS.items():
        market_best: dict[str, str] = {}
        for metric in metrics_list:
            best_model = None
            best_val = None
            lower_better = metric in ["log_loss", "brier_score", "btts_brier",
                                      "over_under_2_5_brier"]

            for model_name, m in all_metrics.items():
                val = m.get(metric)
                if val is not None:
                    if best_val is None:
                        best_val = val
                        best_model = model_name
                    elif lower_better and val < best_val:
                        best_val = val
                        best_model = model_name
                    elif not lower_better and val > best_val:
                        best_val = val
                        best_model = model_name

            if best_model:
                market_best[metric] = best_model
        best[market] = market_best

    return best


# ═══════════════════════════════════════════════════════════
#  Main comparison pipeline
# ═══════════════════════════════════════════════════════════


def run_comparison(
    data_path: str | Path,
    train_end: str = _DEFAULT_TRAIN_END,
    test_end: str = _DEFAULT_TEST_END,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the full Phase 3 model comparison.

    Parameters
    ----------
    data_path : str | Path
        Path to match data CSV.
    train_end, test_end : str
        Chronological split dates.
    quiet : bool
        Suppress verbose output.

    Returns
    -------
    dict[str, Any]
        Full comparison report with all metrics, leaderboard, per-market best.
    """
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "data_path": str(data_path),
        "split": {"train_end": train_end, "test_end": test_end},
        "models": {},
        "leaderboard": {},
        "best_per_market": {},
    }

    print("\n" + "=" * 75)
    print("  PHASE 3 — MODEL COMPARISON")
    print("  Poisson  |  Dixon-Coles  |  Elo  |  Baseline (LR)")
    print("=" * 75)

    # ── [1/4] Load & split data ────────────────────────────
    print(f"\n[1/4] Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"  Total matches: {len(df)}")

    # Check for odds columns
    odds_found = check_odds_available(df)
    has_odds = len(odds_found) > 0
    if has_odds:
        print(f"  Odds columns found: {len(odds_found)} — CLV will be computed")
        report["odds_columns"] = odds_found
    else:
        print(f"  No betting odds columns found — CLV skipped")
        report["odds_columns"] = []

    print(f"\n[2/4] Splitting chronologically...")
    print(f"       Train: < {train_end}")
    print(f"       Test:  {train_end} to {test_end}")
    df_train, df_test = chronological_split(df, train_end, test_end)
    report["train_size"] = len(df_train)
    report["test_size"] = len(df_test)

    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError(f"Empty split: train={len(df_train)}, test={len(df_test)}")

    # ── [3/4] Train all models ─────────────────────────────
    print(f"\n[3/4] Training all 4 models on {len(df_train)} matches...")
    models: dict[str, Any] = {}
    all_metrics: dict[str, dict[str, Any]] = {}

    # Poisson
    print(f"\n  {'>'*5} Poisson Model (expanding-window goals)")
    try:
        poisson = train_poisson(df_train)
        models["Poisson"] = poisson
    except Exception as e:
        logger.error("Poisson training failed: %s", e)
        print(f"  [FAIL] Poisson: {e}")

    # Dixon-Coles
    print(f"\n  {'>'*5} Dixon-Coles MLE (tau correction, recency weighting)")
    try:
        dc = train_dixon_coles(df_train)
        models["Dixon-Coles"] = dc
    except Exception as e:
        logger.error("DC training failed: %s", e)
        print(f"  [FAIL] Dixon-Coles: {e}")

    # Elo
    print(f"\n  {'>'*5} Elo Rating System (K=40, HomeAdv=50, draw_k=0.20)")
    try:
        elo = train_elo(df_train)
        models["Elo"] = elo
    except Exception as e:
        logger.error("Elo training failed: %s", e)
        print(f"  [FAIL] Elo: {e}")

    # Baseline
    print(f"\n  {'>'*5} Logistic Regression (rolling goals averages)")
    try:
        baseline_result = train_baseline_lr(df_train, df_test)
        models["Baseline (LR)"] = baseline_result["model"]
        all_metrics["Baseline (LR)"] = baseline_result["metrics"]
        print(f"  LR baseline: acc={baseline_result['metrics']['accuracy']:.4f}")
    except Exception as e:
        logger.error("Baseline training failed: %s", e)
        print(f"  [FAIL] Baseline: {e}")

    # ── [4/4] Evaluate & compare ────────────────────────────
    print(f"\n[4/4] Evaluating on {len(df_test)} test matches...")

    for model_name, model in models.items():
        if model_name == "Baseline (LR)":
            continue  # already evaluated
        try:
            metrics = evaluate_model_predictions(model, df_test, model_name)
            all_metrics[model_name] = metrics
            print(f"  {model_name:<20s} acc={metrics['accuracy']:.4f} "
                  f"ll={metrics['log_loss']:.4f} brier={metrics['brier_score']:.4f}")
        except Exception as e:
            logger.error("%s evaluation failed: %s", model_name, e)
            print(f"  [FAIL] {model_name}: {e}")

    # CLV (if odds available)
    if has_odds:
        print(f"\n  Computing CLV for models with odds...")
        for model_name, model in models.items():
            try:
                preds = model.predict_matches(df_test)
                clv = compute_clv_metrics(df_test, preds)
                if clv:
                    all_metrics[model_name].update(clv)
                    print(f"  {model_name:<20s} mean_CLV={clv.get('mean_clv', 'N/A')}")
            except Exception as e:
                logger.warning("CLV failed for %s: %s", model_name, e)

    report["all_metrics"] = all_metrics

    # Build leaderboard
    print(f"\n  Building leaderboard...")
    leaderboard_df = build_leaderboard(all_metrics)
    report["leaderboard"] = leaderboard_df.to_dict()

    # Best per market
    best = find_best_per_market(all_metrics)
    report["best_per_market"] = best

    # ── Print results ──────────────────────────────────────
    duration = time.time() - start_time
    report["duration_seconds"] = round(duration, 2)

    print("\n" + "=" * 75)
    print("  LEADERBOARD".center(73))
    print("=" * 75)
    print(f"  Duration: {duration:.2f}s  |  Test set: {len(df_test)} matches")
    print()

    # Print leaderboard table
    lb_print = leaderboard_df.copy()
    display_cols = ["Model", "Composite_Score", "1X2_accuracy", "1X2_log_loss",
                    "1X2_brier_score", "BTTS_accuracy", "OU_over_under_2_5_accuracy"]
    display_cols = [c for c in display_cols if c in lb_print.columns]
    pd.set_option("display.max_colwidth", 20)
    pd.set_option("display.width", 120)

    # Format for display
    print(lb_print[display_cols].to_string())
    print()

    # Best per market
    print(f"  {'='*55}")
    print(f"  {'BEST MODEL PER MARKET':^55s}")
    print(f"  {'='*55}")
    for market, market_best in best.items():
        unique_best = list(set(market_best.values()))
        models_str = ", ".join(unique_best)
        print(f"  {market:<25s} {models_str}")
        for metric, model_name in market_best.items():
            val = all_metrics.get(model_name, {}).get(metric, "?")
            print(f"    {metric:<30s} {model_name:<20s} ({val})")
        print()

    # ── Save outputs ───────────────────────────────────────
    print(f"  Saving outputs...")
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save leaderboard CSV
    lb_path = _REPORT_DIR / f"phase3_leaderboard_{timestamp}.csv"
    leaderboard_df.to_csv(lb_path, index_label="Rank")
    print(f"  Leaderboard CSV: {lb_path}")
    report["leaderboard_path"] = str(lb_path)

    # Save comparison report JSON
    comp_path = _REPORT_DIR / f"phase3_comparison_{timestamp}.json"
    comparison_report = {
        "timestamp": timestamp,
        "split": report["split"],
        "train_size": report["train_size"],
        "test_size": report["test_size"],
        "duration_seconds": report["duration_seconds"],
        "has_odds": has_odds,
        "metrics": all_metrics,
        "best_per_market": {m: dict(v) for m, v in best.items()},
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }
    with open(comp_path, "w") as f:
        json.dump(comparison_report, f, indent=2, default=str)
    print(f"  Comparison JSON: {comp_path}")

    # Summary header
    print("\n" + "=" * 75)
    print("  COMPARISON COMPLETE".center(73))
    print("=" * 75)
    print(f"  Top model: {leaderboard_df.iloc[0]['Model']} "
          f"(score={leaderboard_df.iloc[0]['Composite_Score']:.4f})")
    print()

    return report


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 model comparison")
    parser.add_argument("--data", default=str(config.paths.raw / config.data_collection.output_file))
    parser.add_argument("--train-end", default=_DEFAULT_TRAIN_END)
    parser.add_argument("--test-end", default=_DEFAULT_TEST_END)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    try:
        run_comparison(
            data_path=args.data,
            train_end=args.train_end,
            test_end=args.test_end,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        logger.error("Comparison failed: %s", e, exc_info=True)
        print(f"\n[FAIL] Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
