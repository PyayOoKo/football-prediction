"""
preprocess_over_under.py — Preprocess match data for Over/Under model training.

Loads matches.csv, engineers features (rolling team stats, H2H, league averages),
splits chronologically, and saves the processed feature matrix to Parquet.

Output:
    data/processed/over_under_data_{timestamp}.parquet
    config/over_under_features.json

Usage:
    python scripts/preprocess_over_under.py
    python scripts/preprocess_over_under.py --train-start 2016 --train-end 2022
    python scripts/preprocess_over_under.py --output data/processed/ou_data.parquet
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
logger = logging.getLogger("preprocess_over_under")

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "league_all.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_CONFIG_DIR = PROJECT_ROOT / "config"

# Rolling windows for team stats
WINDOWS = (5, 10, 20)

# Min matches needed for rolling features (shorter = more data, but noisier)
MIN_ROLLING_MATCHES = 2

# Year boundaries for train/test split
# Using a wider date range to capture more historical data up to the present.
# Train: 2014-2024 (11 years, including recent COVID-era and post-COVID football)
# Test:  2025-2026 (most recent completed + current season — current date is July 2026)
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
    """Load matches CSV and prepare for feature engineering."""
    logger.info("Loading data from %s ...", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Check for optional data sources
    for fname in ["team_stats.csv", "xg_data.csv", "odds.csv", "weather.csv"]:
        fpath = PROJECT_ROOT / "data" / fname
        if fpath.exists():
            logger.info("  Optional data found: %s (%.1f MB)", fname, fpath.stat().st_size / 1024 / 1024)
        else:
            logger.info("  Optional data not found: %s — will compute features from matches.csv only", fname)

    # Standardise raw columns into the pipeline's expected format
    df = _standardise_columns(df)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["league", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create target: over35
    df["over35"] = (df["total_goals"] > 3.5).astype(int)

    logger.info(
        "Target distribution — over25: %.1f%%, over35: %.1f%%",
        df["over_2_5"].mean() * 100, df["over35"].mean() * 100,
    )
    return df


# ═══════════════════════════════════════════════════════════
#  2. Team rolling features
# ═══════════════════════════════════════════════════════════


def _compute_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Un-pivot match data into per-team per-game format (2 rows per match)."""
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        hg = row["home_goals"]
        ag = row["away_goals"]
        total = hg + ag

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
            "over25": 1 if total > 2.5 else 0,
            "over35": 1 if total > 3.5 else 0,
            "btts": row["btts"],
            "xg_for": row.get("home_xg", np.nan),
            "xg_against": row.get("away_xg", np.nan),
            "shots_for": row.get("home_shots", np.nan),
            "shots_against": row.get("away_shots", np.nan),
            "shots_target_for": row.get("home_shots_target", np.nan),
            "pts": 3 if hg > ag else (1 if hg == ag else 0),
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
            "over25": 1 if total > 2.5 else 0,
            "over35": 1 if total > 3.5 else 0,
            "btts": row["btts"],
            "xg_for": row.get("away_xg", np.nan),
            "xg_against": row.get("home_xg", np.nan),
            "shots_for": row.get("away_shots", np.nan),
            "shots_against": row.get("home_shots", np.nan),
            "shots_target_for": row.get("away_shots_target", np.nan),
            "pts": 3 if ag > hg else (1 if ag == hg else 0),
        })

    game_log = pd.DataFrame(records)
    game_log.sort_values(["team", "date"], inplace=True)
    game_log.reset_index(drop=True, inplace=True)
    logger.info("Game log: %d rows (%d matches × 2 teams)", len(game_log), len(game_log) // 2)
    return game_log


def _add_rolling_team_features(game_log: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free rolling averages per team.

    Each stat is computed over windows (5, 10, 20) and shifted by 1 match
    so the current match's data is never used to predict itself.
    """
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
            # Check if there's any non-null data for this stat
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

        # Cumulative averages (all-time form)
        for stat in rolling_stats:
            if grp[stat].isna().all():
                continue
            col_name = f"cumavg_{stat}"
            grp[col_name] = grp[stat].expanding().mean().shift(1)

        result_dfs.append(grp)

    return pd.concat(result_dfs, ignore_index=True)


def _merge_rolling_features(df: pd.DataFrame, game_log: pd.DataFrame) -> pd.DataFrame:
    """Merge team rolling features back onto the match-level DataFrame.

    For each match, we capture:
    - Home team's rolling stats (prefixed h_)
    - Away team's rolling stats (prefixed a_)
    - Difference features (home - away)
    """
    # Identify rolling feature columns (everything in game_log that's not base metadata)
    base_cols = {"match_id", "team", "date", "league", "opponent", "is_home",
                 "goals_scored", "goals_conceded", "total_goals",
                 "over25", "over35", "btts",
                 "xg_for", "xg_against", "shots_for", "shots_against",
                 "shots_target_for", "pts"}
    feat_cols = sorted([
        c for c in game_log.columns if c not in base_cols
    ])

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

    # Merge onto match-level df
    df = df.merge(home, on="match_id", how="left")
    df = df.merge(away, on="match_id", how="left")

    # Goal-scoring difference: home_attack - away_defence
    for w in WINDOWS:
        h_scored = f"h_rolling_goals_scored_{w}"
        a_conceded = f"a_rolling_goals_conceded_{w}"
        if h_scored in df.columns and a_conceded in df.columns:
            df[f"diff_att_def_{w}"] = df[h_scored] - df[a_conceded]

        a_scored = f"a_rolling_goals_scored_{w}"
        h_conceded = f"h_rolling_goals_conceded_{w}"
        if a_scored in df.columns and h_conceded in df.columns:
            df[f"diff_def_att_{w}"] = df[a_scored] - df[h_conceded]

        # Total goals expectation
        h_total = f"h_rolling_total_goals_{w}"
        a_total = f"a_rolling_total_goals_{w}"
        if h_total in df.columns and a_total in df.columns:
            df[f"expected_total_goals_{w}"] = (df[h_total] + df[a_total]) / 2

    logger.info("Merged %d rolling feature columns (h_* + a_* + diff_*)", len(feat_cols) * 2 + 3 * len(WINDOWS))
    return df


# ═══════════════════════════════════════════════════════════
#  3. Head-to-head features
# ═══════════════════════════════════════════════════════════


def _add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head rolling features for each pair of teams.

    For each match, computes the last N H2H meetings' total goals and over25 rate.
    """
    # Build sorted H2H game log
    h2h_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        h2h_records.append({
            "pair": tuple(sorted([row["home_team"], row["away_team"]])),
            "match_id": row["match_id"],
            "date": row["date"],
            "total_goals": row["total_goals"],
            "over25": row["over_2_5"],
        })
    h2h = pd.DataFrame(h2h_records)
    h2h.sort_values(["pair", "date"], inplace=True)

    # Rolling H2H features
    h2h["h2h_total_goals_last_5"] = (
        h2h.groupby("pair")["total_goals"]
        .rolling(5, min_periods=1)
        .mean()
        .shift(1)
        .values
    )
    h2h["h2h_over25_rate_last_5"] = (
        h2h.groupby("pair")["over25"]
        .rolling(5, min_periods=1)
        .mean()
        .shift(1)
        .values
    )

    # Merge back
    h2h_cols = h2h[["match_id", "h2h_total_goals_last_5", "h2h_over25_rate_last_5"]]
    df = df.merge(h2h_cols, on="match_id", how="left")

    # Fill NaN for first-time matchups
    df["h2h_total_goals_last_5"] = df["h2h_total_goals_last_5"].fillna(df["total_goals"].mean())
    df["h2h_over25_rate_last_5"] = df["h2h_over25_rate_last_5"].fillna(df["over_2_5"].mean())

    logger.info("Added H2H features — %d unique team-pairs", len(h2h["pair"].unique()))
    return df


# ═══════════════════════════════════════════════════════════
#  4. League features
# ═══════════════════════════════════════════════════════════


def _add_league_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add league-average features to normalise team stats.

    For each league, computes:
    - Rolling average total goals per match and over25 rate
    - Per-team rolling league position (rank by points within league)
    """
    df = df.sort_values(["league", "date"]).copy()

    # League-wide rolling averages
    df["league_avg_total_goals"] = (
        df.groupby("league")["total_goals"]
        .expanding()
        .mean()
        .shift(1)
        .values
    )
    df["league_over25_rate"] = (
        df.groupby("league")["over_2_5"]
        .expanding()
        .mean()
        .shift(1)
        .values
    )

    # League position: approximate by rolling points per game within league/season
    # We compute a simple "points_per_game" rank within each league+season
    df["league_points_per_game"] = (
        df.groupby(["league", "season"])["total_goals"]
        .expanding()
        .mean()
        .shift(1)
        .values
    )

    # Fill first match of each league with sensible defaults
    df["league_avg_total_goals"] = df["league_avg_total_goals"].fillna(2.5)
    df["league_over25_rate"] = df["league_over25_rate"].fillna(0.45)
    df["league_points_per_game"] = df["league_points_per_game"].fillna(1.5)

    logger.info("Added league features for %d leagues", df["league"].nunique())
    return df


# ═══════════════════════════════════════════════════════════
#  5. Clean & select features
# ═══════════════════════════════════════════════════════════


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, remove outliers, fix types."""
    # Focus on the period with good data (2010+)
    df = df[df["date"].dt.year >= 2010].copy()

    before = len(df)

    # Remove extreme outliers (scored goals > 10)
    df = df[(df["home_goals"] <= 10) & (df["away_goals"] <= 10)]

    # Remove duplicate match_id rows (shouldn't exist, but safe)
    df = df.drop_duplicates(subset=["match_id"])

    after = len(df)
    logger.info("Cleaned: %d → %d rows (removed %d)", before, after, before - after)

    return df


def select_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select O/U-relevant features and return (X, y) plus feature list."""
    # Target columns for O/U prediction
    target_cols = ["over_2_5", "over35"]

    # Base identifiers (not features)
    id_cols = [
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "total_goals", "result",
        "btts",
    ]

    # Feature columns: rolling stats + H2H + league averages + xG
    feature_cols = sorted([
        c for c in df.columns
        if c not in target_cols
        and c not in id_cols
        and c != "over_2_5"  # Already in target
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
    ])

    # Remove columns that are entirely NaN
    initial = len(feature_cols)
    feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]
    if len(feature_cols) < initial:
        logger.info("Dropped %d fully-NaN columns", initial - len(feature_cols))

    # Impute remaining NaNs — use rolling's own cumavg prefix as fallback
    for col in feature_cols:
        na_count = df[col].isna().sum()
        if na_count == 0:
            continue

        # For rolling features, prefer the corresponding cumavg as fallback
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
            # Fill rolling NaN with the cumavg value (team's own history)
            df[col] = df[col].fillna(df[cumavg_col])
            # If still NaN (team truly has no history), fill with global mean
            still_na = df[col].isna().sum()
            if still_na > 0:
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                logger.debug("  %s: %d obs still NaN after cumavg fill, using global mean=%.4f",
                             col, still_na, fill_val)
        else:
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            if na_count > len(df) * 0.1:
                logger.debug("  %s: imputed %d/%d NaNs with mean=%.4f",
                             col, na_count, len(df), fill_val)

    logger.info(
        "Final feature set: %d columns, targets: %s",
        len(feature_cols), target_cols,
    )
    return df, feature_cols


# ═══════════════════════════════════════════════════════════
#  6. Train/test split
# ═══════════════════════════════════════════════════════════


def split_by_date(
    df: pd.DataFrame,
    train_start: int = TRAIN_START_YEAR,
    train_end: int = TRAIN_END_YEAR,
    test_start: int = TEST_START_YEAR,
    test_end: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Chronological split: train → validation (last 20% of train) → test.

    Test set covers ``test_start`` to ``test_end`` (defaults to max year in data).
    """
    if test_end is None:
        test_end = df["date"].dt.year.max()
    test_mask = (df["date"].dt.year >= test_start) & (df["date"].dt.year <= test_end)
    train_mask = (df["date"].dt.year >= train_start) & (df["date"].dt.year <= train_end)

    train_val = df[train_mask].copy()
    test = df[test_mask].copy()

    # Validation: last 20% of train_val by date
    train_val = train_val.sort_values("date")
    split_idx = int(len(train_val) * 0.8)
    train = train_val.iloc[:split_idx].copy()
    val = train_val.iloc[split_idx:].copy()

    logger.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%) — date range: %s to %s",
        len(train), len(train) / len(df) * 100,
        len(val), len(val) / len(df) * 100,
        len(test), len(test) / len(df) * 100,
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )

    return {"train": train, "val": val, "test": test}


# ═══════════════════════════════════════════════════════════
#  7. Save
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

    # Save as Parquet
    df_full = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    df_full.to_parquet(output_path, index=False)
    file_size = output_path.stat().st_size
    logger.info("Saved processed data: %s (%.1f MB)", output_path, file_size / 1024 / 1024)

    # Save feature descriptions
    feature_descriptions = {}
    for col in target_cols + feature_cols:
        prefix = col.split("_")[0] if "_" in col else col
        if col.startswith("h_rolling_"):
            stat = col.replace("h_rolling_", "").rsplit("_", 1)[0]
            window = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Home team {stat} — rolling avg last {window} matches"
        elif col.startswith("a_rolling_"):
            stat = col.replace("a_rolling_", "").rsplit("_", 1)[0]
            window = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Away team {stat} — rolling avg last {window} matches"
        elif col.startswith("h_cumavg_"):
            stat = col.replace("h_cumavg_", "")
            feature_descriptions[col] = f"Home team {stat} — cumulative average"
        elif col.startswith("a_cumavg_"):
            stat = col.replace("a_cumavg_", "")
            feature_descriptions[col] = f"Away team {stat} — cumulative average"
        elif col.startswith("diff_"):
            desc = col.replace("diff_", "").rsplit("_", 1)[0]
            window = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Difference: {desc} — window {window}"
        elif col.startswith("expected_total_"):
            window = col.rsplit("_", 1)[-1]
            feature_descriptions[col] = f"Expected total goals (home+away avg) — window {window}"
        elif col.startswith("h2h_"):
            feature_descriptions[col] = f"Head-to-head {col.replace('h2h_', '').replace('_', ' ')}"
        elif col.startswith("league_"):
            feature_descriptions[col] = f"League-wide {col.replace('league_', '').replace('_', ' ')}"
        elif col in target_cols:
            feature_descriptions[col] = f"Target: {col}"
        else:
            feature_descriptions[col] = col

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
            "train": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR} (last 20% → val)",
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

    parser = argparse.ArgumentParser(description="Preprocess data for Over/Under model")
    parser.add_argument("--input", default=str(INPUT_PATH), help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--train-start", type=int, default=TRAIN_START_YEAR)
    parser.add_argument("--train-end", type=int, default=TRAIN_END_YEAR)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or OUTPUT_DIR / f"over_under_data_{timestamp}.parquet")
    feature_config_path = FEATURE_CONFIG_DIR / f"over_under_features_{timestamp}.json"

    print("=" * 65)
    print("  OVER/UNDER DATA PREPROCESSING")
    print("=" * 65)

    total_start = time.time()

    # Step 1: Load
    print("\n--- Step 1: Loading data ---")
    df = load_data(Path(args.input))
    print(f"  {len(df)} matches ({df['date'].min().year}-{df['date'].max().year})")

    # Step 2: Compute team game log
    print("\n--- Step 2: Computing team rolling features ---")
    game_log = _compute_team_game_log(df)
    game_log = _add_rolling_team_features(game_log)
    df = _merge_rolling_features(df, game_log)
    rolling_count = len([c for c in df.columns if c.startswith(("h_", "a_"))])
    print(f"  {rolling_count} rolling feature columns")

    # Step 3: Head-to-head features
    print("\n--- Step 3: Computing H2H features ---")
    df = _add_h2h_features(df)

    # Step 4: League features
    print("\n--- Step 4: Computing league features ---")
    df = _add_league_features(df)

    # Step 5: Clean
    print("\n--- Step 5: Cleaning data ---")
    df = clean_features(df)

    # Step 6: Select features
    print("\n--- Step 6: Selecting features ---")
    df, feature_cols = select_feature_columns(df)
    target_cols = ["over_2_5", "over35"]
    print(f"  {len(feature_cols)} features selected")

    # Step 7: Split
    print("\n--- Step 7: Train/val/test split (chronological) ---")
    splits = split_by_date(df)
    # Dynamic test_end for display
    _disp_test_end = max(TEST_START_YEAR, int(df["date"].dt.year.max()))
    print(f"  Train: {len(splits['train']):>6}  ({TRAIN_START_YEAR}-{TRAIN_END_YEAR})")
    print(f"  Val:   {len(splits['val']):>6}  (last 20% of train)")
    print(f"  Test:  {len(splits['test']):>6}  ({TEST_START_YEAR}-{_disp_test_end})")

    # Step 8: Save
    print("\n--- Step 8: Saving processed data ---")
    save_processed(splits, feature_cols, target_cols, output_path, feature_config_path)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 65)
    print(f"  [OK] PREPROCESSING COMPLETE ({total_elapsed:.1f}s)")
    print("=" * 65)
    print(f"  Output: {output_path}")
    print(f"  Config: {feature_config_path}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: over_2_5, over35")
    print()


if __name__ == "__main__":
    main()
