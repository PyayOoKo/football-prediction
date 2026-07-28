"""
preprocess_btts.py — Preprocess match data for BTTS (Both Teams to Score) model training.

Loads matches.csv, engineers features (scored/conceded rates, clean sheets,
attack/defence strength, H2H BTTS rates, rest days, derby flags, xG),
splits chronologically, and saves to Parquet.

Output:
    data/processed/btts_data_{timestamp}.parquet
    config/btts_features_{timestamp}.json

Usage:
    python scripts/preprocess_btts.py
    python scripts/preprocess_btts.py --train-start 2016 --train-end 2022
"""

from __future__ import annotations

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_btts")

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "league_all.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_CONFIG_DIR = PROJECT_ROOT / "config"

# Rolling windows for team stats
WINDOWS = (5, 10, 20)

# Min matches needed for rolling features
MIN_ROLLING_MATCHES = 2

# Year boundaries for train/test split
# Using a wider date range to capture more historical data up to the present.
# Train: 2014-2024 (11 years)
# Test:  2025-2026 (most recent matches — current date is July 2026)
TRAIN_START_YEAR = 2014
TRAIN_END_YEAR = 2024
TEST_START_YEAR = 2025


# ═══════════════════════════════════════════════════════════
#  1. Load & prepare data
# ═══════════════════════════════════════════════════════════


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise raw league_all.csv columns to the expected format.

    ``league_all.csv`` (from football-data.co.uk) has raw columns like
    ``home_team``, ``away_team``, ``FTHG``/``FTAG``, but lacks computed
    fields like ``total_goals``, ``btts``, ``over_2_5``, and ``match_id``.
    This function adds those computed fields so the rest of the pipeline
    works identically regardless of whether the input is ``matches.csv``
    (which already has them) or ``league_all.csv`` (raw source).

    Also handles column name variations from different data sources.
    """
    df = df.copy()

    # ── Column name normalisation ────────────────────────
    # football-data.co.uk uses FTHG/FTAG for goals; some sources use
    # home_goals/away_goals directly. Map both to the standard names.
    col_map = {
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_target",
        "AST": "away_shots_target",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellow",
        "AY": "away_yellow",
        "HR": "home_red",
        "AR": "away_red",
        "Div": "league",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # ── Generate match_id if not present ─────────────────
    if "match_id" not in df.columns:
        df["match_id"] = (
            df["league"].astype(str)
            + "_" + pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            + "_" + df["home_team"].astype(str).str.replace(" ", "_")
            + "_" + df["away_team"].astype(str).str.replace(" ", "_")
        )

    # ── Parse dates ───────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])

    # ── Compute derived fields ────────────────────────────
    hg = pd.to_numeric(df["home_goals"], errors="coerce")
    ag = pd.to_numeric(df["away_goals"], errors="coerce")

    if "total_goals" not in df.columns:
        df["total_goals"] = hg + ag

    if "btts" not in df.columns:
        df["btts"] = ((hg > 0) & (ag > 0)).astype(int)

    if "over_2_5" not in df.columns:
        df["over_2_5"] = (hg + ag > 2.5).astype(int)

    # Ensure result is standardised (H/A/D, not 1/2/X or lower case)
    if "result" in df.columns:
        df["result"] = (
            df["result"]
            .astype(str)
            .str.upper()
            .str.replace("1", "H", regex=False)
            .str.replace("2", "A", regex=False)
            .str.replace("X", "D", regex=False)
        )

    # Drop rows with missing goals (non-started or abandoned matches)
    before = len(df)
    df = df.dropna(subset=["home_goals", "away_goals"])
    after = len(df)
    if before - after > 0:
        logger.info("Dropped %d rows with missing goals (abandoned/future matches)", before - after)

    # Sort for chronological feature engineering
    df.sort_values(["league", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Standardised: %d rows, %d cols | btts=%.1f%% over25=%.1f%%",
        len(df), len(df.columns),
        df["btts"].mean() * 100 if "btts" in df.columns else 0,
        df["over_2_5"].mean() * 100 if "over_2_5" in df.columns else 0,
    )
    return df


def load_data(path: Path) -> pd.DataFrame:
    """Load matches CSV and prepare for BTTS feature engineering."""
    logger.info("Loading data from %s ...", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Check for optional data sources
    for fname in ["team_stats.csv", "xg_data.csv", "odds.csv", "weather.csv"]:
        fpath = PROJECT_ROOT / "data" / fname
        if fpath.exists():
            logger.info("  Optional data found: %s (%.1f MB)", fname, fpath.stat().st_size / 1024 / 1024)
        else:
            logger.info("  Optional data not found: %s — will compute from matches.csv only", fname)

    # Standardise raw columns into the pipeline's expected format
    df = _standardise_columns(df)

    # btts target is now guaranteed to exist (computed by _standardise_columns)
    assert "btts" in df.columns, "btts column must exist after standardisation!"
    logger.info(
        "Target distribution — btts_yes: %.1f%%, btts_no: %.1f%%",
        df["btts"].mean() * 100, (1 - df["btts"].mean()) * 100,
    )
    return df


# ═══════════════════════════════════════════════════════════
#  2. Team rolling features (BTTS-specific)
# ═══════════════════════════════════════════════════════════


def _compute_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Un-pivot match data into per-team per-game format.

    For each match produces 2 rows (home team, away team) with:
    - scored (team scored ≥1 goal)
    - conceded (team conceded ≥1 goal)
    - clean_sheet (team conceded 0 goals)
    - btts_contrib (the match ended BTTS Yes)
    - goals_scored, goals_conceded
    """
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        hg = row["home_goals"]
        ag = row["away_goals"]
        btts = row["btts"]

        records.append({
            "match_id": row["match_id"],
            "team": row["home_team"],
            "date": row["date"],
            "league": row["league"],
            "opponent": row["away_team"],
            "is_home": 1,
            "scored": 1 if hg > 0 else 0,
            "conceded": 1 if ag > 0 else 0,
            "clean_sheet": 1 if ag == 0 else 0,
            "btts_match": btts,
            "goals_scored": hg,
            "goals_conceded": ag,
            "total_goals": hg + ag,
            "xg_for": row.get("home_xg", np.nan),
            "xg_against": row.get("away_xg", np.nan),
            "pts": 3 if hg > ag else (1 if hg == ag else 0),
        })
        records.append({
            "match_id": row["match_id"],
            "team": row["away_team"],
            "date": row["date"],
            "league": row["league"],
            "opponent": row["home_team"],
            "is_home": 0,
            "scored": 1 if ag > 0 else 0,
            "conceded": 1 if hg > 0 else 0,
            "clean_sheet": 1 if hg == 0 else 0,
            "btts_match": btts,
            "goals_scored": ag,
            "goals_conceded": hg,
            "total_goals": hg + ag,
            "xg_for": row.get("away_xg", np.nan),
            "xg_against": row.get("home_xg", np.nan),
            "pts": 3 if ag > hg else (1 if ag == hg else 0),
        })

    game_log = pd.DataFrame(records)
    game_log.sort_values(["team", "date"], inplace=True)
    game_log.reset_index(drop=True, inplace=True)
    logger.info("Game log: %d rows (%d matches x 2 teams)", len(game_log), len(game_log) // 2)
    return game_log


def _add_rolling_team_features(game_log: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free rolling averages per team for BTTS-relevant stats.

    For each team, computes rolling rates over windows 5/10/20 for:
    - scored% (how often the team scores)
    - conceded% (how often the team concedes)
    - clean_sheet% (how often the team keeps a clean sheet)
    - btts_match% (how often the team's matches end BTTS Yes)
    - goals_scored, goals_conceded, total_goals
    - xG for/against
    """
    btts_stats = [
        "scored", "conceded", "clean_sheet", "btts_match",
        "goals_scored", "goals_conceded", "total_goals",
        "xg_for", "xg_against",
    ]

    result_dfs = []

    for team, grp in game_log.groupby("team"):
        grp = grp.sort_values("date").copy()

        for stat in btts_stats:
            if grp[stat].isna().all():
                continue
            for w in WINDOWS:
                col_name = f"rolling_{stat}_{w}"
                grp[col_name] = (
                    grp[stat]
                    .rolling(w, min_periods=MIN_ROLLING_MATCHES)
                    .mean()
                    .shift(1)  # Leakage prevention
                )

        # Cumulative averages (all-time team tendency)
        for stat in btts_stats:
            if grp[stat].isna().all():
                continue
            col_name = f"cumavg_{stat}"
            grp[col_name] = grp[stat].expanding().mean().shift(1)

        result_dfs.append(grp)

    return pd.concat(result_dfs, ignore_index=True)


def _merge_rolling_features(df: pd.DataFrame, game_log: pd.DataFrame) -> pd.DataFrame:
    """Merge team rolling features onto match-level DataFrame.

    Generates:
    - h_rolling_scored_5, a_rolling_scored_5, etc. (home/away team rates)
    - h_cumavg_scored, a_cumavg_scored, etc.
    - attack_strength (home goals scored vs away goals conceded differential)
    - defence_strength (home clean_sheet vs away scored differential)
    """
    base_cols = {
        "match_id", "team", "date", "league", "opponent", "is_home",
        "scored", "conceded", "clean_sheet", "btts_match", "both_scored",
        "goals_scored", "goals_conceded", "total_goals",
        "xg_for", "xg_against", "pts",
    }
    feat_cols = sorted([c for c in game_log.columns if c not in base_cols])

    if not feat_cols:
        logger.warning("No rolling feature columns generated!")
        return df

    # Home team features
    home = game_log[game_log["is_home"] == 1][["match_id"] + feat_cols].copy()
    home.sort_values("match_id", inplace=True)
    home.columns = ["match_id"] + [f"h_{c}" for c in feat_cols]

    # Away team features
    away = game_log[game_log["is_home"] == 0][["match_id"] + feat_cols].copy()
    away.sort_values("match_id", inplace=True)
    away.columns = ["match_id"] + [f"a_{c}" for c in feat_cols]

    df = df.merge(home, on="match_id", how="left")
    df = df.merge(away, on="match_id", how="left")

    # Create BTTS-specific derived features
    for w in WINDOWS:
        # Attack strength = home scored% - away conceded% (higher = more BTTS likely)
        h_scored = f"h_rolling_scored_{w}"
        a_conceded = f"a_rolling_conceded_{w}"
        if h_scored in df.columns and a_conceded in df.columns:
            df[f"home_attack_{w}"] = df[h_scored]
            df[f"away_defence_{w}"] = df[a_conceded]
            df[f"attack_defence_{w}"] = df[h_scored] + df[a_conceded]

        # Defence strength = home clean_sheet% - away scored%
        h_cs = f"h_rolling_clean_sheet_{w}"
        a_scored = f"a_rolling_scored_{w}"
        if h_cs in df.columns and a_scored in df.columns:
            df[f"home_defence_{w}"] = df[h_cs]
            df[f"away_attack_{w}"] = df[a_scored]
            df[f"defence_attack_{w}"] = df[h_cs] + df[a_scored]

        # BTTS propensity = how often both teams score in these teams' matches
        h_btts = f"h_rolling_btts_match_{w}"
        a_btts = f"a_rolling_btts_match_{w}"
        if h_btts in df.columns and a_btts in df.columns:
            df[f"btts_propensity_{w}"] = (df[h_btts] + df[a_btts]) / 2

    logger.info("Merged %d rolling feature columns", len(feat_cols) * 2 + 3 * 3)
    return df


# ═══════════════════════════════════════════════════════════
#  3. Head-to-head BTTS features
# ═══════════════════════════════════════════════════════════


def _add_h2h_btts_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add H2H rolling features specific to BTTS.

    For each team-pair, computes:
    - Last 5 meetings: BTTS rate
    - Last 5 meetings: home team scored rate
    - Last 5 meetings: away team scored rate
    """
    h2h_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        h2h_records.append({
            "pair": tuple(sorted([row["home_team"], row["away_team"]])),
            "match_id": row["match_id"],
            "date": row["date"],
            "btts": row["btts"],
            "home_scored": 1 if row["home_goals"] > 0 else 0,
            "away_scored": 1 if row["away_goals"] > 0 else 0,
        })
    h2h = pd.DataFrame(h2h_records)
    h2h.sort_values(["pair", "date"], inplace=True)

    h2h["h2h_btts_rate_last_5"] = (
        h2h.groupby("pair")["btts"]
        .rolling(5, min_periods=1)
        .mean()
        .shift(1)
        .values
    )
    h2h["h2h_home_scored_rate"] = (
        h2h.groupby("pair")["home_scored"]
        .rolling(5, min_periods=1)
        .mean()
        .shift(1)
        .values
    )
    h2h["h2h_away_scored_rate"] = (
        h2h.groupby("pair")["away_scored"]
        .rolling(5, min_periods=1)
        .mean()
        .shift(1)
        .values
    )

    h2h_cols = h2h[["match_id", "h2h_btts_rate_last_5", "h2h_home_scored_rate", "h2h_away_scored_rate"]]
    df = df.merge(h2h_cols, on="match_id", how="left")

    # Fill NaN for first-time matchups with league averages
    df["h2h_btts_rate_last_5"] = df["h2h_btts_rate_last_5"].fillna(df["btts"].mean())
    df["h2h_home_scored_rate"] = df["h2h_home_scored_rate"].fillna(df["btts"].mean())
    df["h2h_away_scored_rate"] = df["h2h_away_scored_rate"].fillna(df["btts"].mean())

    logger.info("Added H2H BTTS features — %d unique team-pairs", len(h2h["pair"].unique()))
    return df


# ═══════════════════════════════════════════════════════════
#  4. League features
# ═══════════════════════════════════════════════════════════


def _add_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add league-average features for BTTS normalisation."""
    df = df.sort_values(["league", "date"]).copy()

    df["league_btts_rate"] = (
        df.groupby("league")["btts"]
        .expanding()
        .mean()
        .shift(1)
        .values
    )
    df["league_avg_goals"] = (
        df.groupby("league")["total_goals"]
        .expanding()
        .mean()
        .shift(1)
        .values
    )

    df["league_btts_rate"] = df["league_btts_rate"].fillna(0.45)
    df["league_avg_goals"] = df["league_avg_goals"].fillna(2.5)

    logger.info("Added league features for %d leagues", df["league"].nunique())
    return df


# ═══════════════════════════════════════════════════════════
#  5. Match context features
# ═══════════════════════════════════════════════════════════


def _add_match_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest days and derby flag features.

    - home_rest_days: days since home team's last match
    - away_rest_days: days since away team's last match
    - is_derby: both teams from same city / known derby rivalry
      (approximated as teams sharing a city prefix in their name)
    """
    # Build a team-level date log to compute rest days
    team_dates: dict[str, pd.Timestamp] = {}

    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        dt = row["date"]
        ht = row["home_team"]
        at = row["away_team"]

        if ht in team_dates:
            home_rest.append((dt - team_dates[ht]).days)
        else:
            home_rest.append(None)
        team_dates[ht] = dt

        if at in team_dates:
            away_rest.append((dt - team_dates[at]).days)
        else:
            away_rest.append(None)
        team_dates[at] = dt

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest

    # Fill NaN first-match values with median (typically ~7 days)
    median_rest = df["home_rest_days"].median()
    df["home_rest_days"] = df["home_rest_days"].fillna(median_rest)
    df["away_rest_days"] = df["away_rest_days"].fillna(median_rest)

    # Cap extreme rest values (off-season gaps)
    for col in ["home_rest_days", "away_rest_days"]:
        df[col] = df[col].clip(lower=1, upper=60)

    # Derby flag: approximate by checking if teams share the first word
    # of their name (e.g. "Manchester United" vs "Manchester City")
    # Also exact name match check (some teams have city-only names)
    def _is_derby(home: str, away: str) -> int:
        h_parts = home.lower().split()
        a_parts = away.lower().split()
        # Same city prefix
        if h_parts and a_parts and h_parts[0] == a_parts[0] and home.lower() != away.lower():
            return 1
        # Known derby pairs can be extended here
        return 0

    df["is_derby"] = df.apply(lambda r: _is_derby(r["home_team"], r["away_team"]), axis=1)
    derby_count = df["is_derby"].sum()
    logger.info("Added match context: rest_days + derby (%d derbies detected)", derby_count)

    return df


# ═══════════════════════════════════════════════════════════
#  6. Clean & select features
# ═══════════════════════════════════════════════════════════


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, outliers."""
    df = df[df["date"].dt.year >= 2010].copy()

    before = len(df)

    # Remove extreme outliers (scored goals > 10 — very rare)
    df = df[(df["home_goals"] <= 10) & (df["away_goals"] <= 10)]

    # Remove duplicate match_id rows
    df = df.drop_duplicates(subset=["match_id"])

    after = len(df)
    logger.info("Cleaned: %d -> %d rows (removed %d)", before, after, before - after)
    return df


def select_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select BTTS-relevant features."""
    target_cols = ["btts"]

    id_cols = [
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "total_goals", "result",
        "over_2_5", "over35",
    ]

    feature_cols = sorted([
        c for c in df.columns
        if c not in target_cols
        and c not in id_cols
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
    ])

    # Remove fully-NaN columns
    initial = len(feature_cols)
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]
    if len(feature_cols) < initial:
        logger.info("Dropped %d fully-NaN columns", initial - len(feature_cols))

    # Impute remaining NaNs — cumavg fallback for rolling features
    for col in feature_cols:
        na_count = df[col].isna().sum()
        if na_count == 0:
            continue

        cumavg_col = None
        if col.startswith("h_rolling_"):
            stat_part = col.replace("h_rolling_", "").rsplit("_", 1)[0]
            cumavg_col = f"h_cumavg_{stat_part}"
            if cumavg_col not in df.columns:
                cumavg_col = None
        elif col.startswith("a_rolling_"):
            stat_part = col.replace("a_rolling_", "").rsplit("_", 1)[0]
            cumavg_col = f"a_cumavg_{stat_part}"
            if cumavg_col not in df.columns:
                cumavg_col = None

        if cumavg_col and cumavg_col in df.columns:
            df[col] = df[col].fillna(df[cumavg_col])
            still_na = df[col].isna().sum()
            if still_na > 0:
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                logger.debug("  %s: %d obs still NaN after cumavg fill, using mean=%.4f",
                             col, still_na, fill_val)
        else:
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            if na_count > len(df) * 0.1:
                logger.debug("  %s: imputed %d/%d NaNs with mean=%.4f",
                             col, na_count, len(df), fill_val)

    logger.info(
        "Final feature set: %d columns, target: btts",
        len(feature_cols),
    )
    return df, feature_cols


# ═══════════════════════════════════════════════════════════
#  7. Train/test split
# ═══════════════════════════════════════════════════════════


def split_by_date(
    df: pd.DataFrame,
    train_start: int = TRAIN_START_YEAR,
    train_end: int = TRAIN_END_YEAR,
    test_start: int = TEST_START_YEAR,
    test_end: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Chronological split: train -> val (last 20%) -> test.

    Test set covers ``test_start`` to ``test_end`` (defaults to max year in data).
    """
    if test_end is None:
        test_end = df["date"].dt.year.max()
    test_mask = (df["date"].dt.year >= test_start) & (df["date"].dt.year <= test_end)
    train_mask = (df["date"].dt.year >= train_start) & (df["date"].dt.year <= train_end)

    train_val = df[train_mask].copy()
    test = df[test_mask].copy()

    train_val = train_val.sort_values("date")
    split_idx = int(len(train_val) * 0.8)
    train = train_val.iloc[:split_idx].copy()
    val = train_val.iloc[split_idx:].copy()

    logger.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(train), len(train) / len(df) * 100,
        len(val), len(val) / len(df) * 100,
        len(test), len(test) / len(df) * 100,
    )
    return {"train": train, "val": val, "test": test}


# ═══════════════════════════════════════════════════════════
#  8. Save
# ═══════════════════════════════════════════════════════════


def save_processed(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_cols: list[str],
    output_path: Path,
    feature_config_path: Path,
):
    """Save processed data and feature descriptions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    df_full.to_parquet(output_path, index=False)
    file_size = output_path.stat().st_size
    logger.info("Saved processed data: %s (%.1f MB)", output_path, file_size / 1024 / 1024)

    # Build feature descriptions
    feature_descriptions = {}
    for col in target_cols + feature_cols:
        if col == "btts":
            feature_descriptions[col] = "Target: Both Teams to Score (1=Yes, 0=No)"
        elif col.startswith("h_rolling_scored_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team scored%% rate — rolling avg last {w} matches"
        elif col.startswith("a_rolling_scored_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team scored%% rate — rolling avg last {w} matches"
        elif col.startswith("h_rolling_conceded_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team conceded%% rate — rolling avg last {w} matches"
        elif col.startswith("a_rolling_conceded_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team conceded%% rate — rolling avg last {w} matches"
        elif col.startswith("h_rolling_clean_sheet_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team clean sheet%% rate — rolling avg last {w} matches"
        elif col.startswith("a_rolling_clean_sheet_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team clean sheet%% rate — rolling avg last {w} matches"
        elif col.startswith("h_rolling_btts_match_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team's matches ending BTTS%% — rolling avg last {w}"
        elif col.startswith("a_rolling_btts_match_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team's matches ending BTTS%% — rolling avg last {w}"
        elif col.startswith("h_rolling_goals_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team {col.replace('h_rolling_', '').replace(f'_{w}', '')} avg — last {w}"
        elif col.startswith("a_rolling_goals_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team {col.replace('a_rolling_', '').replace(f'_{w}', '')} avg — last {w}"
        elif col.startswith("h_rolling_xg_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team {col.replace('h_rolling_', '').replace(f'_{w}', '')} avg — last {w}"
        elif col.startswith("a_rolling_xg_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team {col.replace('a_rolling_', '').replace(f'_{w}', '')} avg — last {w}"
        elif col.startswith("h_cumavg_"):
            feature_descriptions[col] = f"Home team {col.replace('h_cumavg_', '')} — cumulative average"
        elif col.startswith("a_cumavg_"):
            feature_descriptions[col] = f"Away team {col.replace('a_cumavg_', '')} — cumulative average"
        elif col.startswith("home_attack_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team scored%% rate (attack) — window {w}"
        elif col.startswith("away_attack_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team scored%% rate (attack) — window {w}"
        elif col.startswith("home_defence_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team clean sheet%% rate (defence) — window {w}"
        elif col.startswith("away_defence_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team clean sheet%% rate (defence) — window {w}"
        elif col.startswith("attack_defence_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home scored%% + Away conceded%% (BTTS likelihood) — window {w}"
        elif col.startswith("defence_attack_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home clean sheet%% + Away scored%% — window {w}"
        elif col.startswith("btts_propensity_"):
            w = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Avg BTTS rate across both teams — window {w}"
        elif col.startswith("h2h_"):
            feature_descriptions[col] = f"Head-to-head {col.replace('h2h_', '').replace('_', ' ')}"
        elif col.startswith("league_"):
            feature_descriptions[col] = f"League-wide {col.replace('league_', '').replace('_', ' ')}"
        else:
            feature_descriptions[col] = col.replace("_", " ").title()

    # Dynamic test_end: use the later of TEST_START_YEAR and max year in data
    _test_end = max(TEST_START_YEAR, int(df_full["date"].dt.year.max()))
    config = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(INPUT_PATH),
        "output": str(output_path),
        "n_matches_total": len(df_full),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "split": {
            "train": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR} (last 20%% -> val)",
            "test": f"{TEST_START_YEAR}-{_test_end}",
        },
        "targets": target_cols,
        "n_features": len(feature_cols),
        "features": feature_descriptions,
    }

    with open(feature_config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved feature config: %s", feature_config_path)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for BTTS model")
    parser.add_argument("--input", default=str(INPUT_PATH), help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--train-start", type=int, default=TRAIN_START_YEAR)
    parser.add_argument("--train-end", type=int, default=TRAIN_END_YEAR)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or OUTPUT_DIR / f"btts_data_{timestamp}.parquet")
    feature_config_path = FEATURE_CONFIG_DIR / f"btts_features_{timestamp}.json"

    print("=" * 65)
    print("  BTTS DATA PREPROCESSING")
    print("=" * 65)

    total_start = time.time()

    # Step 1: Load
    print("\n--- Step 1: Loading data ---")
    df = load_data(Path(args.input))
    print(f"  {len(df)} matches ({df['date'].min().year}-{df['date'].max().year})")
    print(f"  BTTS distribution: Yes={df['btts'].sum():,} ({df['btts'].mean()*100:.1f}%), "
          f"No={(1-df['btts']).sum():,} ({(1-df['btts'].mean())*100:.1f}%)")

    # Step 2: Team rolling features
    print("\n--- Step 2: Computing team rolling features (scored/conceded/CS rates) ---")
    game_log = _compute_team_game_log(df)
    game_log = _add_rolling_team_features(game_log)
    df = _merge_rolling_features(df, game_log)
    rolling_count = len([c for c in df.columns if c.startswith(("h_", "a_"))])
    print(f"  {rolling_count} rolling feature columns")

    # Step 3: H2H BTTS features
    print("\n--- Step 3: Computing H2H BTTS features ---")
    df = _add_h2h_btts_features(df)

    # Step 4: League features
    print("\n--- Step 4: Computing league features ---")
    df = _add_league_features(df)

    # Step 5: Match context
    print("\n--- Step 5: Computing match context (rest days, derby) ---")
    df = _add_match_context(df)

    # Step 6: Clean
    print("\n--- Step 6: Cleaning data ---")
    df = clean_features(df)

    # Step 7: Select features
    print("\n--- Step 7: Selecting features ---")
    df, feature_cols = select_feature_columns(df)
    target_cols = ["btts"]
    print(f"  {len(feature_cols)} features selected")

    # Step 8: Split
    print("\n--- Step 8: Train/val/test split (chronological) ---")
    splits = split_by_date(df)
    print(f"  Train: {len(splits['train']):>6}  ({TRAIN_START_YEAR}-{TRAIN_END_YEAR})")
    print(f"  Val:   {len(splits['val']):>6}  (last 20% of train)")
    _disp_test_end = max(TEST_START_YEAR, int(df["date"].dt.year.max()))
    print(f"  Test:  {len(splits['test']):>6}  ({TEST_START_YEAR}-{_disp_test_end})")

    # Step 9: Save
    print("\n--- Step 9: Saving processed data ---")
    save_processed(splits, feature_cols, target_cols, output_path, feature_config_path)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 65)
    print(f"  [OK] BTTS PREPROCESSING COMPLETE ({total_elapsed:.1f}s)")
    print("=" * 65)
    print(f"  Output: {output_path}")
    print(f"  Config: {feature_config_path}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: btts")
    print()


if __name__ == "__main__":
    main()
