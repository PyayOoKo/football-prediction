"""
collect_team_stats.py — Export team rolling statistics for O/U & BTTS models.

Computes per-team rolling averages for:
  - Goals scored / conceded (last 5, 10, 20 matches)
  - Total goals, Over 2.5%, BTTS%
  - Shots, shots on target
  - xG for / against
  - Clean sheet % (defensive strength)
  - Scored % (offensive consistency)

Output:
    data/team_stats.csv — One row per match, per team (2 rows per match)

Usage:
    python scripts/collect_team_stats.py
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_team_stats")

DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
OUTPUT_DIR = PROJECT_ROOT / "data"

# Minimum matches a team must have played to compute rolling stats
MIN_MATCHES = 2


def load_matches() -> pd.DataFrame:
    """Load all historical matches from DB with stats columns."""
    conn = sqlite3.connect(str(DB_PATH))

    query = """
        SELECT
            match_id, date, league, season,
            home_team, away_team,
            home_goals, away_goals, result,
            home_shots, away_shots,
            home_shots_target, away_shots_target,
            home_xg, away_xg,
            home_corners, away_corners,
            home_yellow, away_yellow,
            home_red, away_red,
            home_fouls, away_fouls,
            home_odds, draw_odds, away_odds,
            over25_odds, under25_odds
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info("Loaded %d matches from database", len(df))
    return df


def compute_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-match, per-team rolling statistics.

    Returns a DataFrame with two rows per match (home team + away team).
    Rolling stats are shifted by 1 to prevent leakage.
    """
    records: list[dict] = []

    for _, row in df.iterrows():
        hg = row["home_goals"]
        ag = row["away_goals"]
        total_goals = hg + ag

        # Derive targets
        over_2_5 = 1 if total_goals > 2.5 else 0
        btts = 1 if hg > 0 and ag > 0 else 0
        home_clean_sheet = 1 if ag == 0 else 0
        home_scored = 1 if hg > 0 else 0
        away_clean_sheet = 1 if hg == 0 else 0
        away_scored = 1 if ag > 0 else 0

        home_result = (
            3 if hg > ag else 1 if hg == ag else 0
        )
        away_result = (
            3 if ag > hg else 1 if ag == hg else 0
        )

        # Home team row
        records.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "league": row["league"],
            "season": row["season"],
            "team": row["home_team"],
            "opponent": row["away_team"],
            "is_home": 1,
            "goals_scored": hg,
            "goals_conceded": ag,
            "total_goals": total_goals,
            "over_2_5": over_2_5,
            "btts": btts,
            "clean_sheet": home_clean_sheet,
            "scored": home_scored,
            "points": home_result,
            "shots_for": row["home_shots"] if pd.notna(row["home_shots"]) else np.nan,
            "shots_against": row["away_shots"] if pd.notna(row["away_shots"]) else np.nan,
            "shots_target_for": row["home_shots_target"] if pd.notna(row["home_shots_target"]) else np.nan,
            "shots_target_against": row["away_shots_target"] if pd.notna(row["away_shots_target"]) else np.nan,
            "xg_for": row["home_xg"] if pd.notna(row["home_xg"]) else np.nan,
            "xg_against": row["away_xg"] if pd.notna(row["away_xg"]) else np.nan,
            "corners_for": row["home_corners"] if pd.notna(row["home_corners"]) else np.nan,
            "fouls_for": row["home_fouls"] if pd.notna(row["home_fouls"]) else np.nan,
            "yellow_for": row["home_yellow"] if pd.notna(row["home_yellow"]) else np.nan,
        })

        # Away team row
        records.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "league": row["league"],
            "season": row["season"],
            "team": row["away_team"],
            "opponent": row["home_team"],
            "is_home": 0,
            "goals_scored": ag,
            "goals_conceded": hg,
            "total_goals": total_goals,
            "over_2_5": over_2_5,
            "btts": btts,
            "clean_sheet": away_clean_sheet,
            "scored": away_scored,
            "points": away_result,
            "shots_for": row["away_shots"] if pd.notna(row["away_shots"]) else np.nan,
            "shots_against": row["home_shots"] if pd.notna(row["home_shots"]) else np.nan,
            "shots_target_for": row["away_shots_target"] if pd.notna(row["away_shots_target"]) else np.nan,
            "shots_target_against": row["home_shots_target"] if pd.notna(row["home_shots_target"]) else np.nan,
            "xg_for": row["away_xg"] if pd.notna(row["away_xg"]) else np.nan,
            "xg_against": row["home_xg"] if pd.notna(row["home_xg"]) else np.nan,
            "corners_for": row["away_corners"] if pd.notna(row["away_corners"]) else np.nan,
            "fouls_for": row["away_fouls"] if pd.notna(row["away_fouls"]) else np.nan,
            "yellow_for": row["away_yellow"] if pd.notna(row["away_yellow"]) else np.nan,
        })

    team_df = pd.DataFrame(records)
    team_df["date"] = pd.to_datetime(team_df["date"])
    return team_df


def add_rolling_averages(
    team_df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20)
) -> pd.DataFrame:
    """Add rolling averages for all numeric stats per team.

    Each window generates a shifted (leakage-free) average for each stat.
    """
    rolling_cols = [
        "goals_scored", "goals_conceded", "total_goals",
        "over_2_5", "btts", "clean_sheet", "scored", "points",
        "shots_for", "shots_against",
        "xg_for", "xg_against",
    ]

    result_dfs = []

    for team, grp in team_df.groupby("team"):
        grp = grp.sort_values("date").copy()

        for col in rolling_cols:
            for w in windows:
                col_name = f"rolling_{col}_{w}"
                grp[col_name] = (
                    grp[col]
                    .rolling(w, min_periods=1)
                    .mean()
                    .shift(1)  # ← leakage prevention
                )

        # Also compute cumulative averages (all-time for the team)
        for col in rolling_cols:
            col_name = f"cumavg_{col}"
            grp[col_name] = grp[col].expanding().mean().shift(1)

        result_dfs.append(grp)

    return pd.concat(result_dfs, ignore_index=True)


def export_to_csv(
    team_df: pd.DataFrame, output_path: Path
) -> dict[str, int]:
    """Export team stats to CSV with a clean column set."""
    # Define output columns (rolling stats are all prefixed with rolling_ or cumavg_)
    base_cols = [
        "match_id", "date", "league", "season", "team", "opponent", "is_home",
        "goals_scored", "goals_conceded", "total_goals", "over_2_5", "btts",
        "clean_sheet", "scored", "points",
        "shots_for", "shots_against",
        "xg_for", "xg_against",
    ]
    rolling_cols = sorted(
        [c for c in team_df.columns if c.startswith("rolling_") or c.startswith("cumavg_")]
    )
    all_cols = base_cols + rolling_cols

    # Filter to only columns that actually exist
    output_cols = [c for c in all_cols if c in team_df.columns]

    team_df[output_cols].to_csv(output_path, index=False, encoding="utf-8")

    stats = {
        "rows": len(team_df),
        "matches": len(team_df) // 2,
        "teams": team_df["team"].nunique(),
        "leagues": team_df["league"].nunique(),
        "rolling_features": len(rolling_cols),
        "output_file": str(output_path),
    }
    return stats


def main():
    print("=" * 60)
    print("  TEAM STATS DATA COLLECTION")
    print("=" * 60)

    # Step 1: Load data
    print("\n--- Step 1: Loading match data ---")
    df = load_matches()
    print(f"  Loaded {len(df)} matches")

    # Step 2: Compute per-team per-match stats
    print("\n--- Step 2: Computing team stats ---")
    team_df = compute_team_stats(df)
    print(f"  Created {len(team_df)} team-rows ({len(team_df)//2} matches)")

    # Step 3: Add rolling averages
    print("\n--- Step 3: Adding rolling averages (windows 5, 10, 20) ---")
    team_df = add_rolling_averages(team_df, windows=(5, 10, 20))
    print(f"  Features: {len([c for c in team_df.columns if c.startswith('rolling_')])} rolling + "
          f"{len([c for c in team_df.columns if c.startswith('cumavg_')])} cumulative")

    # Step 4: Export to CSV
    print("\n--- Step 4: Exporting to CSV ---")
    output_path = OUTPUT_DIR / "team_stats.csv"
    stats = export_to_csv(team_df, output_path)
    print(f"  Rows: {stats['rows']:,} ({stats['matches']:,} matches)")
    print(f"  Teams: {stats['teams']:,}")
    print(f"  Leagues: {stats['leagues']}")
    print(f"  Rolling features: {stats['rolling_features']}")
    print(f"  Output: {stats['output_file']}")

    # League-by-league breakdown
    print("\n--- Per-League Breakdown ---")
    for league, grp in team_df.groupby("league"):
        n_matches = grp["match_id"].nunique()
        n_teams = grp["team"].nunique()
        has_xg = grp["xg_for"].notna().mean() * 100
        has_shots = grp["shots_for"].notna().mean() * 100
        avg_o25 = grp["over_2_5"].mean() * 100
        avg_btts = grp["btts"].mean() * 100
        avg_goals = grp["total_goals"].mean()
        print(f"  {league:>4}: {n_matches:>5} matches, {n_teams:>2} teams | "
              f"xg={has_xg:>4.0f}% shots={has_shots:>4.0f}% | "
              f"O2.5={avg_o25:>4.1f}% BTTS={avg_btts:>4.1f}% AvgG={avg_goals:.2f}")

    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  ✅ data/team_stats.csv — {stats['rows']:,} rows, {stats['rolling_features']} features")
    print("  ℹ️  Data is leakage-free (rolling stats are shifted by 1 match)")
    print()


if __name__ == "__main__":
    main()
