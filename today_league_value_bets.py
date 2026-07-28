"""
today_league_value_bets.py — Live value bets for league-trained models.

For each league, loads per-league models (Dixon-Coles + Elo), fetches
upcoming matches from the DB, gets live odds via The Odds API, and
computes Kelly-sized value bets using market-proven models.

Market Strategy (backtest-proven, see LEAGUE_STRATEGY dict):
- E0 (EPL):     1X2 ❌ (-37%) | O/U 2.5 ✅ (+11.67% RF) | BTTS ✅ (+13.51%)
- D1 (Bundesliga): 1X2 ❌ (-29%) | O/U 2.5 ✅ (+17.12% RF) | BTTS ✅
- F1 (Ligue 1): 1X2 ❌ | O/U 2.5 ✅ (+19.32% RF) | BTTS ✅
- Other leagues: Full market coverage (1X2 + O/U + BTTS)

Models per market:
- 1X2: DC + Elo blend (calibrated, only for profitable leagues)
- O/U 2.5: Random Forest model (107 features, rolling team stats + odds)
- BTTS: Implied-from-markets model (market odds -> BTTS prob, Brier=0.244)

Usage
-----
    python today_league_value_bets.py
    python today_league_value_bets.py --leagues E0 D1 F1 --ou --btts
    python today_league_value_bets.py --leagues NOR SWE --bankroll 5000 --kelly 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("today_league_value_bets")

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODELS_DIR = PROJECT_ROOT / "models" / "per_league"
REPORTS_DIR = PROJECT_ROOT / "reports" / "value_bets"

# ── Global models for binary markets ────────────────────
# These models are proven to beat the market in backtests:
# O/U RF: +10.58% ROI | BTTS Implied: Brier=0.244
GLOBAL_MODELS_DIR = PROJECT_ROOT / "models"
BTTS_IMPLIED_MODEL_PATH = GLOBAL_MODELS_DIR / "btts_implied_from_markets.joblib"

# O/U RF model — trained on 107 features (rolling stats + odds + league + H2H)
OU_RF_MODEL_PATH = GLOBAL_MODELS_DIR / "over_under_random_forest_20260725_222912.joblib"

# Rolling windows for team stats (must match preprocess_over_under.py)
_ROLLING_WINDOWS = (5, 10, 20)
_MIN_ROLLING_MATCHES = 2

# Post-match leaky features that need defaults for upcoming matches
_LEAKY_FEATURES = {
    "home_xg", "away_xg", "home_shots", "away_shots",
    "home_shots_target", "away_shots_target", "home_corners", "away_corners",
}

INITIAL_BANKROLL = 10_000.0
DEFAULT_MIN_EV = 0.05
DEFAULT_KELLY_FRAC = 0.25
MAX_ODDS = 30.0

LEAGUE_NAMES = {
    "E0":  "England Premier League",
    "SP1": "Spain La Liga",
    "D1":  "Germany Bundesliga",
    "I1":  "Italy Serie A",
    "F1":  "France Ligue 1",
    "NOR": "Norway Eliteserien",
    "SWE": "Sweden Allsvenskan",
    "IRL": "Ireland Premier Division",
    "SE1": "Sweden Superettan",
    "FI":  "Finland Veikkausliiga",
}

# ── Per-League Betting Strategy ───────────────────────────
# Controls which markets are checked per league, based on
# backtest results. Top 5 leagues have efficient 1X2 markets
# (negative ROI), so we skip them and focus on Over 2.5 + BTTS.
# Keys: '1x2', 'over_under', 'btts'
# Values: True/False (enable or disable the market)
LEAGUE_STRATEGY: dict[str, dict[str, bool]] = {
    # EPL: 1X2 -37.1% ❌ | Over 2.5 +11.67% ✅ | BTTS +13.51% ✅
    "E0":  {"1x2": False, "over_under": True, "btts": True},
    # Bundesliga: 1X2 -29.0% ❌ | Over 2.5 +17.12% ✅ | BTTS +13.51% ✅
    "D1":  {"1x2": False, "over_under": True, "btts": True},
    # Ligue 1: 1X2 unprofitable ❌ | Over 2.5 +19.32% ✅ | BTTS +13.51% ✅
    "F1":  {"1x2": False, "over_under": True, "btts": True},
    # All other leagues: full market coverage
}

# Default strategy for leagues not explicitly configured
_DEFAULT_LEAGUE_STRATEGY = {"1x2": True, "over_under": True, "btts": True}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live league value bets")
    p.add_argument("--leagues", nargs="+", default=["E0", "D1", "F1"],
                        help="League codes (default: E0, D1, F1 — profitable top 5 leagues)")
    p.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL, help=f"Bankroll (default: {INITIAL_BANKROLL})")
    p.add_argument("--kelly", type=float, default=DEFAULT_KELLY_FRAC, help=f"Kelly fraction (default: {DEFAULT_KELLY_FRAC})")
    p.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV, help=f"Min EV threshold (default: {DEFAULT_MIN_EV})")
    p.add_argument("--max-odds", type=float, default=MAX_ODDS, help=f"Max decimal odds (default: {MAX_ODDS})")
    p.add_argument("--calibrate", choices=["none", "hybrid", "platt", "isotonic"], default="hybrid",
                        help="Probability calibration method (default: hybrid)")
    p.add_argument("--no-calibrate", action="store_true", help="Disable calibration")
    p.add_argument("--ou", action="store_true", help="Check over/under (totals) value bets too")
    p.add_argument("--btts", action="store_true", help="Check BTTS value bets using market-implied BTTS odds from 1X2+O/U markets")
    p.add_argument("--no-save", action="store_true", help="Skip saving report")
    p.add_argument("--quiet", action="store_true", help="Minimal output")
    return p.parse_args(argv)


def normalize_team(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.replace(".", "").replace("-", " ").replace("'", "")
    name = " ".join(name.split())
    return name


def fuzzy_match(db_name: str, api_names: list[str], threshold: float = 0.6) -> str | None:
    db_norm = normalize_team(db_name)
    best, best_score = None, 0
    for api in api_names:
        api_norm = normalize_team(api)
        score = SequenceMatcher(None, db_norm, api_norm).ratio()
        if score > best_score:
            best, best_score = api, score
    return best if best_score >= threshold else None


def load_team_name_map(league: str) -> dict[str, str]:
    path = MODELS_DIR / league / "team_name_map.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_league_models(league: str) -> dict[str, Any] | None:
    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    if not dc_path.exists() or not elo_path.exists():
        return None
    return {
        "dc": joblib.load(dc_path),
        "elo": joblib.load(elo_path),
    }


def get_upcoming_matches(league: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team
        FROM matches
        WHERE league = ? AND result IS NULL
        ORDER BY date ASC
    """
    df = __import__("pandas").read_sql_query(query, conn, params=(league,))
    conn.close()
    return df.to_dict("records")


def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1 else 0.0


def kelly_stake(prob: float, odds: float, fraction: float = 0.25) -> float:
    if odds <= 1 or prob <= 0:
        return 0.0
    full_kelly = (prob * odds - 1.0) / (odds - 1.0)
    return max(0.0, full_kelly * fraction)


# ═══════════════════════════════════════════════════════════
#  BTTS Implied Model — market BTTS odds derived from 1X2+O/U
# ═══════════════════════════════════════════════════════════
# Trained by scripts/derive_btts_implied.py (Random Forest, Brier=0.244).
# Takes market odds as input → estimates what the market "thinks" about BTTS.
# ═══════════════════════════════════════════════════════════


def load_btts_implied_model() -> tuple[Any, list[str]] | None:
    """Load the BTTS implied-from-markets model.

    Returns (model, feature_cols) or None if unavailable.
    """
    if not BTTS_IMPLIED_MODEL_PATH.exists():
        logger.info("  BTTS implied model not found at %s", BTTS_IMPLIED_MODEL_PATH)
        return None
    try:
        data = joblib.load(BTTS_IMPLIED_MODEL_PATH)
        model = data["model"]
        feature_cols = data["feature_cols"]

        # Extract best model's Brier score safely
        metrics = data.get("metrics", {})
        best_name = metrics.get("best_model", "RandomForest")
        best_brier = 0.244  # default fallback for Brier
        for m in metrics.get("all_models", []):
            if m.get("model") == best_name:
                best_brier = m.get("brier", 0.244)
                break

        logger.info("  Loaded BTTS implied model: %s (Brier=%.4f, generated: %s)",
                     best_name, best_brier, data.get("generated", "?"))
        return model, feature_cols
    except Exception as exc:
        logger.warning("  Failed to load BTTS implied model: %s", exc)
        return None


def predict_market_btts_prob(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    over25_odds: float,
    under25_odds: float,
    model: Any,
    feature_cols: list[str],
    league_btts_rolling: float = 0.50,
    year: int = 2026,
    month: int = 7,
) -> float:
    """Estimate what the market thinks about BTTS probability.

    Uses the trained RF model that learned BTTS patterns from
    31,837 historical matches by correlating 1X2 + O/U odds
    with actual BTTS outcomes.

    Returns: float between 0 and 1 (market-implied BTTS probability)
    """
    # Implied probabilities from odds
    home_imp = 1.0 / home_odds
    draw_imp = 1.0 / draw_odds
    away_imp = 1.0 / away_odds
    over25_imp = 1.0 / over25_odds
    under25_imp = 1.0 / under25_odds

    margin_1x2 = home_imp + draw_imp + away_imp
    margin_ou = over25_imp + under25_imp

    home_prob = home_imp / margin_1x2
    draw_prob = draw_imp / margin_1x2
    away_prob = away_imp / margin_1x2
    over25_prob = over25_imp / margin_ou
    under25_prob = under25_imp / margin_ou

    favorite_imp = min(home_imp, away_imp)
    underdog_imp = max(home_imp, away_imp)
    favorite_dominance = underdog_imp / max(favorite_imp, 0.001)
    ou_ratio = over25_prob / max(under25_prob, 0.001)

    # Build feature vector matching training order
    feature_data = {
        "home_imp": home_imp,
        "draw_imp": draw_imp,
        "away_imp": away_imp,
        "over25_imp": over25_imp,
        "under25_imp": under25_imp,
        "margin_1x2": margin_1x2,
        "margin_ou": margin_ou,
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "over25_prob": over25_prob,
        "favorite_imp": favorite_imp,
        "underdog_imp": underdog_imp,
        "favorite_dominance": favorite_dominance,
        "ou_ratio": ou_ratio,
        "league_btts_rolling": league_btts_rolling,
        "year": float(year),
        "month": float(month),
    }

    X = np.array([[feature_data[c] for c in feature_cols]], dtype=np.float32)
    # Handle NaN
    X = np.nan_to_num(X, nan=0.5)

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    return float(model.predict(X)[0])



# ═══════════════════════════════════════════════════════════
#  O/U RF Model — 107 features from rolling team stats + odds
# ═══════════════════════════════════════════════════════════
# Uses the global Random Forest model (Brier=0.2411, +10.58% ROI)
# Feature engineering mirrors scripts/preprocess_over_under.py
# ═══════════════════════════════════════════════════════════


def load_ou_rf_model() -> Any | None:
    """Load the global O/U Random Forest model."""
    if not OU_RF_MODEL_PATH.exists():
        logger.info("  O/U RF model not found at %s", OU_RF_MODEL_PATH)
        return None
    try:
        model = joblib.load(OU_RF_MODEL_PATH)
        logger.info("  Loaded O/U RF model (%d features)", model.n_features_in_)
        return model
    except Exception as exc:
        logger.warning("  Failed to load O/U RF model: %s", exc)
        return None


def _db_to_ou_dataframe(conn: sqlite3.Connection, leagues: list[str]) -> pd.DataFrame:
    """Load match data from DB into the format needed for O/U feature engineering.

    Returns DataFrame with columns matching preprocess_over_under.py input.
    """
    placeholders = ",".join("?" for _ in leagues)
    query = f"""
        SELECT date, league, season, home_team, away_team,
               home_goals, away_goals,
               home_odds, draw_odds, away_odds,
               over25_odds, under25_odds,
               home_xg, away_xg,
               home_shots, away_shots,
               home_shots_target, away_shots_target,
               home_corners, away_corners,
               home_fouls, away_fouls,
               home_yellow, away_yellow,
               home_red, away_red,
               result
        FROM matches
        WHERE league IN ({placeholders})
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=leagues)

    # Compute derived columns
    df["date"] = pd.to_datetime(df["date"])
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)
    df["over35"] = (df["total_goals"] > 3.5).astype(int)
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    # Ensure numeric types for rolling stats
    for col in ["home_goals", "away_goals", "total_goals",
                "home_xg", "away_xg", "home_shots", "away_shots",
                "home_shots_target", "away_shots_target", "home_corners", "away_corners"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add proxy match_id (row index for merging)
    df["match_id"] = df.index.values
    df["season"] = df["season"].fillna(df["date"].dt.year.astype(str))

    return df


def _compute_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Un-pivot match data into per-team per-game format (2 rows per match)."""
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        hg = row["home_goals"]
        ag = row["away_goals"]
        total = hg + ag if pd.notna(hg) and pd.notna(ag) else float("nan")

        records.append({
            "match_id": row["match_id"],
            "team": row["home_team"],
            "date": row["date"],
            "league": row["league"],
            "opponent": row["away_team"],
            "is_home": 1,
            "goals_scored": hg,
            "goals_conceded": ag,
            "total_goals": total,
            "over25": 1 if pd.notna(total) and total > 2.5 else (float("nan") if pd.isna(total) else 0),
            "over35": 1 if pd.notna(total) and total > 3.5 else (float("nan") if pd.isna(total) else 0),
            "btts": 1 if pd.notna(hg) and pd.notna(ag) and hg > 0 and ag > 0 else (float("nan") if pd.isna(hg) or pd.isna(ag) else 0),
            "xg_for": row.get("home_xg", float("nan")),
            "xg_against": row.get("away_xg", float("nan")),
            "shots_for": row.get("home_shots", float("nan")),
            "shots_against": row.get("away_shots", float("nan")),
        })
        records.append({
            "match_id": row["match_id"],
            "team": row["away_team"],
            "date": row["date"],
            "league": row["league"],
            "opponent": row["home_team"],
            "is_home": 0,
            "goals_scored": ag,
            "goals_conceded": hg,
            "total_goals": total,
            "over25": 1 if pd.notna(total) and total > 2.5 else (float("nan") if pd.isna(total) else 0),
            "over35": 1 if pd.notna(total) and total > 3.5 else (float("nan") if pd.isna(total) else 0),
            "btts": 1 if pd.notna(hg) and pd.notna(ag) and hg > 0 and ag > 0 else (float("nan") if pd.isna(hg) or pd.isna(ag) else 0),
            "xg_for": row.get("away_xg", float("nan")),
            "xg_against": row.get("home_xg", float("nan")),
            "shots_for": row.get("away_shots", float("nan")),
            "shots_against": row.get("home_shots", float("nan")),
        })

    game_log = pd.DataFrame(records)
    game_log.sort_values(["team", "date"], inplace=True)
    game_log.reset_index(drop=True, inplace=True)
    return game_log


def _add_rolling_team_features(game_log: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free rolling averages per team."""
    rolling_stats = [
        "goals_scored", "goals_conceded", "total_goals",
        "over25", "over35", "btts",
        "xg_for", "xg_against",
        "shots_for", "shots_against",
    ]

    result_dfs = []
    for team, grp in game_log.groupby("team"):
        grp = grp.sort_values("date").copy()
        for stat in rolling_stats:
            if grp[stat].isna().all():
                continue
            for w in _ROLLING_WINDOWS:
                col = f"rolling_{stat}_{w}"
                grp[col] = grp[stat].rolling(w, min_periods=_MIN_ROLLING_MATCHES).mean().shift(1)
        # Cumulative averages
        for stat in rolling_stats:
            if grp[stat].isna().all():
                continue
            col = f"cumavg_{stat}"
            grp[col] = grp[stat].expanding().mean().shift(1)
        result_dfs.append(grp)

    return pd.concat(result_dfs, ignore_index=True)


def _merge_rolling_features(df: pd.DataFrame, game_log: pd.DataFrame) -> pd.DataFrame:
    """Merge team rolling features back onto match-level DataFrame."""
    base_cols = {
        "match_id", "team", "date", "league", "opponent", "is_home",
        "goals_scored", "goals_conceded", "total_goals",
        "over25", "over35", "btts",
        "xg_for", "xg_against", "shots_for", "shots_against",
        "pts",
    }
    feat_cols = sorted([c for c in game_log.columns if c not in base_cols])
    if not feat_cols:
        return df

    home = game_log[game_log["is_home"] == 1][["match_id"] + feat_cols].copy()
    home.columns = ["match_id"] + [f"h_{c}" for c in feat_cols]

    away = game_log[game_log["is_home"] == 0][["match_id"] + feat_cols].copy()
    away.columns = ["match_id"] + [f"a_{c}" for c in feat_cols]

    df = df.merge(home, on="match_id", how="left")
    df = df.merge(away, on="match_id", how="left")

    # Goal-scoring differences
    for w in _ROLLING_WINDOWS:
        h_s = f"h_rolling_goals_scored_{w}"
        a_c = f"a_rolling_goals_conceded_{w}"
        a_s = f"a_rolling_goals_scored_{w}"
        h_c = f"h_rolling_goals_conceded_{w}"
        h_t = f"h_rolling_total_goals_{w}"
        a_t = f"a_rolling_total_goals_{w}"
        if h_s in df.columns and a_c in df.columns:
            df[f"diff_att_def_{w}"] = df[h_s] - df[a_c]
        if a_s in df.columns and h_c in df.columns:
            df[f"diff_def_att_{w}"] = df[a_s] - df[h_c]
        if h_t in df.columns and a_t in df.columns:
            df[f"expected_total_goals_{w}"] = (df[h_t] + df[a_t]) / 2

    return df


def _add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head rolling features."""
    h2h_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        h2h_records.append({
            "pair": tuple(sorted([row["home_team"], row["away_team"]])),
            "match_id": row["match_id"],
            "date": row["date"],
            "total_goals": row["total_goals"] if pd.notna(row["total_goals"]) else 0,
            "over25": row["over_2_5"] if pd.notna(row["over_2_5"]) else 0,
        })
    h2h = pd.DataFrame(h2h_records)
    h2h.sort_values(["pair", "date"], inplace=True)

    h2h["h2h_total_goals_last_5"] = (
        h2h.groupby("pair")["total_goals"].rolling(5, min_periods=1).mean().shift(1).values
    )
    h2h["h2h_over25_rate_last_5"] = (
        h2h.groupby("pair")["over25"].rolling(5, min_periods=1).mean().shift(1).values
    )

    df = df.merge(h2h[["match_id", "h2h_total_goals_last_5", "h2h_over25_rate_last_5"]],
                  on="match_id", how="left")
    df["h2h_total_goals_last_5"] = df["h2h_total_goals_last_5"].fillna(df["total_goals"].mean())
    df["h2h_over25_rate_last_5"] = df["h2h_over25_rate_last_5"].fillna(0.45)
    return df


def _add_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add league-average features."""
    df = df.sort_values(["league", "date"]).copy()

    df["league_avg_total_goals"] = (
        df.groupby("league")["total_goals"].expanding().mean().shift(1).values
    )
    df["league_over25_rate"] = (
        df.groupby("league")["over_2_5"].expanding().mean().shift(1).values
    )
    df["league_points_per_game"] = (
        df.groupby(["league", "season"])["total_goals"].expanding().mean().shift(1).values
    )

    df["league_avg_total_goals"] = df["league_avg_total_goals"].fillna(2.5)
    df["league_over25_rate"] = df["league_over25_rate"].fillna(0.45)
    df["league_points_per_game"] = df["league_points_per_game"].fillna(1.5)
    return df


def _get_feature_cols() -> list[str]:
    """Return the exact 107 feature column names in sorted order.

    These are the columns the RF model was trained on —
    rolling stats + odds + league + H2H, excluding id/target columns.
    """
    id_cols = {
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "total_goals", "result",
        "btts",
    }
    target_cols = {"over_2_5", "over35"}

    # All rolling columns identified from the training data
    roll_cols = []
    for prefix in ["h_cumavg_", "h_rolling_", "a_cumavg_", "a_rolling_"]:
        for stat in ["btts", "goals_conceded", "goals_scored", "over25", "over35",
                      "shots_against", "shots_for", "total_goals",
                      "xg_against", "xg_for"]:
            if "cumavg" in prefix:
                roll_cols.append(f"{prefix}{stat}")
            else:
                for w in [5, 10, 20]:
                    roll_cols.append(f"{prefix}{stat}_{w}")

    other_cols = [
        "away_corners", "away_odds", "away_shots", "away_shots_target", "away_xg",
        "diff_att_def_10", "diff_att_def_20", "diff_att_def_5",
        "diff_def_att_10", "diff_def_att_20", "diff_def_att_5",
        "draw_odds",
        "expected_total_goals_10", "expected_total_goals_20", "expected_total_goals_5",
        "h2h_over25_rate_last_5", "h2h_total_goals_last_5",
        "home_corners", "home_odds", "home_shots", "home_shots_target", "home_xg",
        "league_avg_total_goals", "league_over25_rate", "league_points_per_game",
        "over25_odds", "under25_odds",
    ]

    all_cols = set(roll_cols + other_cols)
    # Remove any that might overlap with id/target
    all_cols -= id_cols
    all_cols -= target_cols

    return sorted(all_cols)


def build_ou_features_and_predict(
    conn: sqlite3.Connection,
    league: str,
    upcoming_matches: list[dict[str, Any]],
    rf_model: Any,
    api_odds_map: dict[tuple[str, str], dict[str, Any]],
    name_map: dict[str, str],
) -> dict[tuple[str, str], float]:
    """Build the 107-feature O/U matrix for upcoming matches and predict.

    Args:
        conn: DB connection
        league: League code (e.g. "SE1")
        upcoming_matches: List of dicts with home_team, away_team, date
        rf_model: The loaded RandomForestClassifier (107 features)
        api_odds_map: Dict mapping (api_home, api_away) -> odds dict
        name_map: Team name mapping (DB name -> API name)

    Returns:
        Dict mapping (db_home, db_away) -> over_2_5_prob from RF model
    """
    # 1. Load historical + upcoming data from DB
    df = _db_to_ou_dataframe(conn, [league])

    # 2. Identify upcoming match rows (no result / null goals)
    upcoming_idx = []
    for i, row in df.iterrows():
        if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
            for um in upcoming_matches:
                if um["home_team"] == row["home_team"] and um["away_team"] == row["away_team"]:
                    upcoming_idx.append(i)
                    break

    if not upcoming_idx:
        logger.info("  No upcoming matches found in DB for feature building")
        return {}

    logger.info("  Building O/U features for %d upcoming matches using full pipeline...", len(upcoming_idx))

    # 3. Build the game log and compute rolling features
    game_log = _compute_team_game_log(df)
    game_log = _add_rolling_team_features(game_log)
    df = _merge_rolling_features(df, game_log)

    # 4. Add H2H and league features
    df = _add_h2h_features(df)
    df = _add_league_features(df)

    # 5. Get the exact 107 feature columns
    feature_cols = _get_feature_cols()
    logger.info("  Feature matrix: %d columns", len(feature_cols))

    # 6. Fill odds features from API data for upcoming matches
    # Build reverse lookup: (db_home, db_away) -> api odds
    odds_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for (api_h, api_a), od in api_odds_map.items():
        # Find the DB names for this API pair
        for db_um in upcoming_matches:
            db_h = db_um["home_team"]
            db_a = db_um["away_team"]
            api_db_h = name_map.get(db_h, db_h)
            api_db_a = name_map.get(db_a, db_a)
            if (api_h.lower() == api_db_h.lower() and api_a.lower() == api_db_a.lower()) or \
               (api_a.lower() == api_db_h.lower() and api_h.lower() == api_db_a.lower()):
                odds_lookup[(db_h, db_a)] = od

    # Set odds for upcoming matches
    for i in upcoming_idx:
        h = df.at[i, "home_team"]
        a = df.at[i, "away_team"]
        key = (h, a)
        od = odds_lookup.get(key)
        if od:
            df.at[i, "home_odds"] = od.get("home_odds", float("nan"))
            df.at[i, "draw_odds"] = od.get("draw_odds", float("nan"))
            df.at[i, "away_odds"] = od.get("away_odds", float("nan"))
            if "over_odds" in od:
                df.at[i, "over25_odds"] = od["over_odds"]
                df.at[i, "under25_odds"] = od.get("under_odds", float("nan"))

    # 7. Impute NaNs and build feature matrix
    X_list = []
    upcoming_keys = []
    for i in upcoming_idx:
        h = df.at[i, "home_team"]
        a = df.at[i, "away_team"]

        row_data = {}
        for col in feature_cols:
            val = df.at[i, col]
            if pd.isna(val):
                # For leaky features, use 0 (unknown = neutral)
                if col in _LEAKY_FEATURES:
                    val = 0.0
                elif col in {"home_odds", "draw_odds", "away_odds", "over25_odds", "under25_odds"}:
                    val = 2.0  # default even-ish odds
                elif "rolling" in col:
                    val = 0.5  # neutral rolling rate
                elif "cumavg" in col:
                    val = 0.5
                elif "diff_" in col:
                    val = 0.0
                elif "expected_total" in col:
                    val = 2.5
                elif "h2h_" in col:
                    val = 0.45
                elif "league_" in col:
                    val = 2.5 if "total_goals" in col else (0.45 if "over25" in col else 1.5)
                else:
                    val = 0.0
            row_data[col] = float(val)
        X_list.append(row_data)
        upcoming_keys.append((h, a))

    X = np.array([[r[c] for c in feature_cols] for r in X_list], dtype=np.float32)

    # 8. Predict using the RF model
    if hasattr(rf_model, "predict_proba"):
        probs = rf_model.predict_proba(X)[:, 1]
    else:
        probs = rf_model.predict(X).astype(float)

    # 9. Return lookup
    result = {}
    for (h, a), prob in zip(upcoming_keys, probs):
        result[(h, a)] = float(prob)

    logger.info("  RF O/U predictions: %d matches (avg prob=%.1f%%)",
                 len(result), sum(result.values()) / len(result) * 100 if result else 0)
    return result


def main(argv=None):
    args = parse_args(argv)

    if args.no_calibrate:
        args.calibrate = "none"

    # BTTS needs O/U odds from the API to derive market-implied BTTS probability
    if args.btts and not args.ou:
        logger.info("  BTTS needs O/U odds — auto-enabling --ou to fetch totals markets")
        args.ou = True

    # Remove redundant import inside main() — joblib is already imported at module level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Load .env if it exists
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    from src.odds_api import OddsAPIClient, LEAGUE_TO_SPORT_KEY

    odds_client = OddsAPIClient()

    if not odds_client.api_key:
        logger.error("THE_ODDS_API_KEY not set. Get a free key at https://the-odds-api.com/")
        return 1

    # Load the global O/U RF model once (used for all leagues)
    rf_model = load_ou_rf_model()
    if rf_model is not None:
        logger.info("  Global O/U RF model loaded (%d features) — will use for O/U predictions", rf_model.n_features_in_)

    all_results: list[dict[str, Any]] = []

    for league in args.leagues:
        league_name = LEAGUE_NAMES.get(league, league)
        cal_label = f" [calibrated: {args.calibrate}]" if args.calibrate != "none" else ""
        logger.info("=" * 65)
        logger.info("  %s - %s%s", league, league_name, cal_label)
        logger.info("=" * 65)

        sport_key = LEAGUE_TO_SPORT_KEY.get(league)
        if not sport_key:
            logger.warning("  No sport key for league %s — skipping", league)
            continue

        models = load_league_models(league)
        if models is None:
            logger.warning("  No trained models found — run train_league_models.py --leagues %s first", league)
            continue

        dc = models["dc"]
        elo = models["elo"]

        # Load calibrators (1X2 + OU)
        calibrator = None
        ou_calibrator = None
        if args.calibrate != "none":
            cal_path = MODELS_DIR / league / f"blend_calibrator_{args.calibrate}.joblib"
            if not cal_path.exists():
                cal_path = MODELS_DIR / league / "blend_calibrator.joblib"
            if cal_path.exists():
                calibrator = joblib.load(cal_path)
                logger.info("  Loaded calibrator: %s", args.calibrate)
            # Load OU calibrator
            ou_cal_path = MODELS_DIR / league / f"blend_calibrator_{args.calibrate}_over_under.joblib"
            if not ou_cal_path.exists():
                ou_cal_path = MODELS_DIR / league / "blend_calibrator_over_under.joblib"
            if ou_cal_path.exists():
                ou_calibrator = joblib.load(ou_cal_path)
                logger.info("  Loaded OU calibrator: %s", args.calibrate)

        name_map = load_team_name_map(league)
        api_team_names = list(name_map.values())

        upcoming = get_upcoming_matches(league)
        if not upcoming:
            logger.info("  No upcoming matches found")
            # Save empty result
            all_results.append({"league": league, "bets": [], "metrics": {"n_bets": 0}})
            continue

        logger.info("  Upcoming matches: %d", len(upcoming))

        # Build team pairs for odds API lookup
        api_pairs = []
        db_to_api: dict[int, str] = {}
        for i, m in enumerate(upcoming):
            h = name_map.get(m["home_team"]) or fuzzy_match(m["home_team"], api_team_names)
            a = name_map.get(m["away_team"]) or fuzzy_match(m["away_team"], api_team_names)
            if h and a:
                api_pairs.append((h, a))
                db_to_api[i] = f"{h} vs {a}"

        if not api_pairs:
            logger.warning("  No team name mappings available for current upcoming matches")
            all_results.append({"league": league, "bets": [], "metrics": {"n_bets": 0}})
            continue

        # Fetch live odds (combine h2h + totals if --ou)
        markets = "h2h,totals" if args.ou else "h2h"
        logger.info("  Fetching live odds via The Odds API (sport=%s, markets=%s)...", sport_key, markets)
        matches = odds_client.get_upcoming_odds(sport_key=sport_key, markets=markets)

        # Build lookup
        lookup: dict[tuple[str, str], Any] = {}
        for match in matches:
            lookup[(match.home_team.lower(), match.away_team.lower())] = match

        # Build H2H + OU results
        live_odds: dict[tuple[str, str], dict[str, Any]] = {}
        ou_odds_map: dict[tuple[str, str], dict[str, Any]] = {}
        for h, a in api_pairs:
            key = (h.lower(), a.lower())
            swp = (a.lower(), h.lower())
            odds = lookup.get(key) or lookup.get(swp)
            if odds is None:
                continue
            is_swapped = key not in lookup and swp in lookup
            if is_swapped:
                live_odds[(h, a)] = {
                    "home_odds": odds.away_odds,
                    "draw_odds": odds.draw_odds,
                    "away_odds": odds.home_odds,
                    "bookmaker": odds.bookmaker,
                    "match_date": odds.match_date,
                }
            else:
                live_odds[(h, a)] = {
                    "home_odds": odds.home_odds,
                    "draw_odds": odds.draw_odds,
                    "away_odds": odds.away_odds,
                    "bookmaker": odds.bookmaker,
                    "match_date": odds.match_date,
                }
            if args.ou and odds.over_odds > 0:
                ou_odds_map[(h, a)] = {
                    "over_odds": odds.over_odds,
                    "under_odds": odds.under_odds,
                    "totals_point": odds.totals_point,
                }

        logger.info("  Got odds for %d/%d matches", len(live_odds), len(api_pairs))
        if args.ou:
            logger.info("  With over/under odds: %d/%d", len(ou_odds_map), len(live_odds))

        if not live_odds:
            logger.warning("  No odds returned from API")
            all_results.append({"league": league, "bets": [], "metrics": {"n_bets": 0}})
            continue

        # Build reverse map: (api_home, api_away) -> (db_home, db_away)
        reverse_map: dict[tuple[str, str], tuple[str, str]] = {}
        for i, m in enumerate(upcoming):
            h = name_map.get(m["home_team"]) or fuzzy_match(m["home_team"], api_team_names)
            a = name_map.get(m["away_team"]) or fuzzy_match(m["away_team"], api_team_names)
            if h and a:
                key = (h, a)
                # Odds API might swap home/away
                if key in live_odds:
                    reverse_map[key] = (m["home_team"], m["away_team"])
                elif (a, h) in live_odds:
                    reverse_map[(a, h)] = (m["home_team"], m["away_team"])

        # Load BTTS implied model if requested
        btts_implied = None
        if args.btts:
            btts_implied = load_btts_implied_model()
            if btts_implied is not None:
                logger.info("  BTTS implied model loaded (Brier=0.244)")
            else:
                logger.info("  BTTS implied model not found — run scripts/derive_btts_implied.py first")

        # Get O/U RF model predictions for upcoming matches (if model loaded)
        ou_rf_preds: dict[tuple[str, str], float] = {}
        if rf_model is not None and live_odds:
            try:
                # Merge live h2h odds with O/U odds for the RF feature builder
                # (ou_odds_map only has over/under, live_odds has h2h)
                combined_api_odds: dict[tuple[str, str], dict[str, Any]] = {}
                for key, od in live_odds.items():
                    combined_api_odds[key] = dict(od)
                    if key in ou_odds_map:
                        combined_api_odds[key].update(ou_odds_map[key])

                conn = sqlite3.connect(str(DB_PATH))
                ou_rf_preds = build_ou_features_and_predict(
                    conn, league, upcoming, rf_model, combined_api_odds, name_map,
                )
                conn.close()
                if ou_rf_preds:
                    logger.info("  Using RF model for O/U predictions (%d matches)", len(ou_rf_preds))
                else:
                    logger.info("  RF model couldn't compute features — falling back to DC-only for O/U")
            except Exception as fe:
                logger.debug("  Feature building failed: %s — using DC-only for O/U", fe)

        # ── Per-league market strategy ──
        # Controls which markets are checked (see LEAGUE_STRATEGY dict).
        # Top 5 leagues (E0, D1, F1) skip 1X2 (proven -29% to -37% ROI).
        league_strat = LEAGUE_STRATEGY.get(league, _DEFAULT_LEAGUE_STRATEGY)
        active_markets = [k for k, v in league_strat.items() if v]
        logger.info("  Strategy: markets=%s", active_markets)

        # Compute value bets
        bankroll = args.bankroll
        bets: list[dict[str, Any]] = []

        for (api_h, api_a), odds_data in live_odds.items():
            db_pair = reverse_map.get((api_h, api_a)) or reverse_map.get((api_a, api_h))
            if not db_pair:
                continue
            db_home, db_away = db_pair

            # Re-check orientation
            if (api_h, api_a) in live_odds and (api_h, api_a) in reverse_map:
                od = {
                    "home_odds": odds_data["home_odds"],
                    "draw_odds": odds_data["draw_odds"],
                    "away_odds": odds_data["away_odds"],
                }
            elif (api_a, api_h) in reverse_map:
                od = {
                    "home_odds": odds_data["away_odds"],
                    "draw_odds": odds_data["draw_odds"],
                    "away_odds": odds_data["home_odds"],
                }
            else:
                continue

            odds_h = float(od["home_odds"])
            odds_d = float(od["draw_odds"])
            odds_a = float(od["away_odds"])

            if odds_h > args.max_odds or odds_d > args.max_odds or odds_a > args.max_odds:
                continue

            # Get model predictions
            try:
                dc_pred = dc.predict(db_home, db_away)
                dc_probs = np.array([dc_pred.away_win_prob, dc_pred.draw_prob, dc_pred.home_win_prob])

                R_home = elo.get_rating(db_home)
                R_away = elo.get_rating(db_away)
                E_home = elo.expected_score(R_home, R_away)
                elo_away, elo_draw, elo_home = elo._expected_to_probs(E_home)
                elo_probs = np.array([elo_away, elo_draw, elo_home])

                blend_probs = (dc_probs + elo_probs) / 2.0

                # Apply calibration if available
                if calibrator is not None:
                    blend_probs = calibrator.transform(blend_probs.reshape(1, -1))[0]
            except Exception as e:
                logger.debug("  Model error for %s vs %s: %s", db_home, db_away, e)
                continue

                # Find match date from DB
            match_date = ""
            for m in upcoming:
                if m["home_team"] == db_home and m["away_team"] == db_away:
                    match_date = m["date"]
                    break

            # ── 1X2 value check ── (skipped for leagues where proven unprofitable)
            if league_strat["1x2"]:
                outcomes = [
                    ("H", 2, odds_h, blend_probs[2]),
                    ("D", 1, odds_d, blend_probs[1]),
                    ("A", 0, odds_a, blend_probs[0]),
                ]

                for outcome_label, outcome_idx, odds, model_prob in outcomes:
                    if odds <= 1 or model_prob <= 0:
                        continue

                    implied = implied_prob(odds)
                    ev = model_prob / implied - 1.0

                    if ev < args.min_ev:
                        continue

                    stake_pct = kelly_stake(model_prob, odds, args.kelly)
                    if stake_pct <= 0:
                        continue

                    stake_amount = bankroll * stake_pct

                    bets.append({
                        "date": match_date,
                        "home": db_home,
                        "away": db_away,
                        "outcome": outcome_label,
                        "odds": round(odds, 2),
                        "model_prob": round(model_prob, 4),
                        "implied_prob": round(implied, 4),
                        "ev": round(ev, 4),
                        "stake_pct": round(stake_pct, 4),
                        "stake": round(stake_amount, 2),
                        "profit": 0.0,
                    })

            # ── Over/Under 2.5 value check ── (RF model primary, backed by +11.67% EPL ROI)
            if args.ou and league_strat["over_under"]:
                # Use RF model prediction if available, else DC-only
                ou_over_prob = dc_pred.over_2_5_prob
                ou_prob_source = "DC"
                key = (db_home, db_away)
                if key in ou_rf_preds:
                    ou_over_prob = ou_rf_preds[key]
                    ou_prob_source = "RF"

                if ou_over_prob <= 0:
                    continue

                ou_under_prob = 1.0 - ou_over_prob
                ou_probs = np.array([ou_under_prob, ou_over_prob])
                ou_data = ou_odds_map.get((api_h, api_a))
                if ou_data:
                    over_odds = ou_data["over_odds"]
                    under_odds = ou_data["under_odds"]
                    tp = ou_data["totals_point"]
                    for label, odds, model_prob in [("O" + str(tp), over_odds, ou_probs[1]), ("U" + str(tp), under_odds, ou_probs[0])]:
                        if odds <= 1 or model_prob <= 0:
                            continue
                        impl = implied_prob(odds)
                        e = model_prob / impl - 1.0
                        if e < args.min_ev:
                            continue
                        sp = kelly_stake(model_prob, odds, args.kelly)
                        if sp <= 0:
                            continue
                        bets.append({
                            "date": match_date,
                            "home": db_home,
                            "away": db_away,
                            "outcome": label,
                            "odds": round(odds, 2),
                            "model_prob": round(model_prob, 4),
                            "implied_prob": round(impl, 4),
                            "ev": round(e, 4),
                            "stake_pct": round(sp, 4),
                            "stake": round(bankroll * sp, 2),
                            "profit": 0.0,
                        })

            # ── BTTS value check ── (implied model, +13.51% ROI across all leagues)
            if args.btts and btts_implied is not None and league_strat["btts"]:
                try:
                    # Our model's BTTS probability (DC-only, proven best for binary markets)
                    our_btts_prob = dc_pred.btts_prob
                    if our_btts_prob > 0:
                        # Check if we have all the odds needed for the implied model
                        ou_data = ou_odds_map.get((api_h, api_a))
                        if ou_data and odds_h > 0 and odds_d > 0 and odds_a > 0:
                            over_odds = ou_data["over_odds"]
                            under_odds = ou_data["under_odds"]
                            tp = ou_data["totals_point"]

                            if over_odds > 1 and under_odds > 1:
                                # Estimate league BTTS rolling rate from per-league data
                                league_btts_rate = 0.50  # default
                                try:
                                    conn = sqlite3.connect(str(DB_PATH))
                                    cur = conn.execute(
                                        "SELECT AVG(CASE WHEN home_goals > 0 AND away_goals > 0 THEN 1.0 ELSE 0.0 END) "
                                        "FROM matches WHERE league=? AND home_goals IS NOT NULL "
                                        "ORDER BY date DESC LIMIT 100",
                                        (league,)
                                    )
                                    row = cur.fetchone()
                                    if row and row[0] is not None:
                                        league_btts_rate = float(row[0])
                                    conn.close()
                                except Exception:
                                    pass

                                # Extract year/month from match date for the BTTS model
                                match_dt = None
                                match_year, match_month = 2026, 7
                                if match_date:
                                    try:
                                        from datetime import datetime as _dt
                                        match_dt = _dt.strptime(str(match_date)[:10], "%Y-%m-%d")
                                        match_year = match_dt.year
                                        match_month = match_dt.month
                                    except Exception:
                                        pass

                                # Market-implied BTTS probability from 1X2+O/U odds
                                btts_implied_model, btts_feature_cols = btts_implied
                                market_btts_prob = predict_market_btts_prob(
                                    odds_h, odds_d, odds_a,
                                    over_odds, under_odds,
                                    btts_implied_model, btts_feature_cols,
                                    league_btts_rolling=league_btts_rate,
                                    year=match_year, month=match_month,
                                )

                                # Derive odds from market probability (assume 5% margin)
                                margin = 0.05
                                btts_yes_odds = 1.0 / (market_btts_prob * (1 + margin))
                                btts_no_odds = 1.0 / ((1 - market_btts_prob) * (1 + margin))

                                # Check both directions: BET YES or BET NO
                                for label, odds, model_prob, market_prob in [
                                    ("BYes", btts_yes_odds, our_btts_prob, market_btts_prob),
                                    ("BNo", btts_no_odds, 1 - our_btts_prob, 1 - market_btts_prob),
                                ]:
                                    if odds <= 1 or model_prob <= 0:
                                        continue
                                    if odds > args.max_odds:
                                        continue
                                    impl = 1.0 / odds  # market implied prob including vig
                                    e = model_prob / max(impl, 0.001) - 1.0
                                    if e < args.min_ev:
                                        continue
                                    sp = kelly_stake(model_prob, odds, args.kelly)
                                    if sp <= 0:
                                        continue
                                    bets.append({
                                        "date": match_date,
                                        "home": db_home,
                                        "away": db_away,
                                        "outcome": label,
                                        "odds": round(odds, 2),
                                        "model_prob": round(model_prob, 4),
                                        "implied_prob": round(impl, 4),  # 1/odds for consistency
                                        "ev": round(e, 4),
                                        "stake_pct": round(sp, 4),
                                        "stake": round(bankroll * sp, 2),
                                        "profit": 0.0,
                                        "btts_method": "derived_from_1x2_ou",
                                    })
                except Exception as btts_err:
                    logger.debug("  BTTS value check error for %s vs %s: %s", db_home, db_away, btts_err)

        # Display results
        if not bets:
            logger.info("  No value bets found")
            all_results.append({"league": league, "bets": [], "metrics": {"n_bets": 0}})
            continue

        bets.sort(key=lambda b: b["ev"], reverse=True)

        logger.info("")
        logger.info("  VALUE BETS FOUND: %d", len(bets))
        logger.info("  %s", "-" * 80)
        logger.info("  %-22s %-22s %-5s %-6s %-7s %-7s %-7s %-6s",
                     "Home", "Away", "Pick", "Odds", "Prob", "Fair", "EV", "Stake")
        logger.info("  %s", "-" * 80)
        for b in bets:
            label = {"H": "Home", "D": "Draw", "A": "Away"}.get(b["outcome"], b["outcome"])
            logger.info("  %-22s %-22s %-5s %-6.2f %-6.1f%% %-6.1f%% %-6.1f%%  GBP %-6.1f",
                         b["home"][:20], b["away"][:20], label, b["odds"],
                         b["model_prob"] * 100, b["implied_prob"] * 100, b["ev"] * 100, b["stake"])

        # Metrics
        metrics = {
            "n_bets": len(bets),
            "total_staked": round(sum(b["stake"] for b in bets), 2),
            "avg_odds": round(float(np.mean([b["odds"] for b in bets])), 2),
            "avg_ev": round(float(np.mean([b["ev"] for b in bets])), 4),
            "avg_stake_pct": round(float(np.mean([b["stake_pct"] for b in bets])), 4),
        }

        result_data = {
            "league": league,
            "league_name": league_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bankroll": args.bankroll,
            "kelly_frac": args.kelly,
            "min_ev": args.min_ev,
            "metrics": metrics,
            "bets": bets,
        }
        all_results.append(result_data)

        # Save per-league report
        if not args.no_save:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            report_path = REPORTS_DIR / f"value_bets_{league}_{ts}.json"
            with open(report_path, "w") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            logger.info("  Report saved: %s", report_path)

    # Summary
    total_bets = sum(len(r.get("bets", [])) for r in all_results)
    logger.info("")
    logger.info("=" * 65)
    logger.info("  SUMMARY")
    logger.info("=" * 65)
    logger.info("  Leagues: %s", ", ".join(args.leagues))
    logger.info("  Total value bets: %d", total_bets)
    logger.info("  Bankroll: GBP %.2f", args.bankroll)
    logger.info("  Kelly fraction: %.0f%%", args.kelly * 100)
    logger.info("  Min EV: %.0f%%", args.min_ev * 100)
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
