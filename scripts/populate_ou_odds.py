"""
populate_ou_odds.py — Backfill Over/Under 2.5 odds from football-data.co.uk.

Reads the original CSV files from football-data.co.uk for each league,
extracts the BbAv>2.5 and BbAv<2.5 closing odds, and UPDATEs the existing
matches in the DB with those values.

This is a one-time migration for existing data. New data collected after
the pipeline changes (sqlite.py + football_data.py) will automatically
include O/U odds columns.

Usage:
    python scripts/populate_ou_odds.py
    python scripts/populate_ou_odds.py --leagues E0 SE1
    python scripts/populate_ou_odds.py --max-seasons 3
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("populate_ou_odds")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"

# Known leagues from the config
LEAGUE_CODES = ["E0", "E1", "E2", "E3", "EC", "SC0", "D1", "D2", "I1", "I2",
                "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1",
                "SE1", "SE2", "NO1", "NO2", "FI1", "FI2",
                "SWE", "NOR", "FIN", "DEN", "JPN", "MEX", "USA", "BRA", "ARG",
                "AUT", "POL", "SUI", "IRL"]

BASE_URL = "https://www.football-data.co.uk"
MMZ_PATH = "mmz4281"
NEW_CSV_URL = "https://www.football-data.co.uk/new/{league}.csv"

OU_COLUMNS = ["bbav>2.5", "bbav<2.5"]
# Fallback columns if BbAv not available
FALLBACK_OU = ["avg>2.5", "avg<2.5"]
FALLBACK2_OU = ["b365>2.5", "b365<2.5"]


def _download_season_csv(season: str, league: str) -> pd.DataFrame | None:
    """Download a single season CSV, returning lowercase columns."""
    url = f"{BASE_URL}/{MMZ_PATH}/{season}/{league}.csv"
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (compatible; FootballDataCollector/1.0)",
        })
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text), na_values=["", "NA", "N/A"])
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as exc:
        logger.debug("Failed to download %s: %s", url, exc)
        return None


def _download_current_csv(league: str) -> pd.DataFrame | None:
    """Download the current in-progress season CSV."""
    url = NEW_CSV_URL.format(league=league)
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (compatible; FootballDataCollector/1.0)",
        })
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text), na_values=["", "NA", "N/A"])
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as exc:
        logger.debug("Failed to download current %s: %s", league, exc)
        return None


def _find_ou_cols(df: pd.DataFrame) -> tuple[str, str] | None:
    """Find over/under 2.5 columns in the DataFrame.
    
    Returns (over_col, under_col) or None if not found.
    """
    for over_col, under_col in [OU_COLUMNS, FALLBACK_OU, FALLBACK2_OU]:
        if over_col in df.columns and under_col in df.columns:
            return over_col, under_col
    return None


def populate_league(league: str, max_seasons: int = 10) -> dict:
    """Populate O/U odds for a single league by re-downloading CSV data.
    
    Returns stats dict with rows_updated, total_matches, etc.
    """
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Get existing matches for this league that have 1X2 odds but no O/U odds
    c.execute("""
        SELECT match_id, date, home_team, away_team
        FROM matches
        WHERE league = ? AND home_odds IS NOT NULL
          AND (over25_odds IS NULL OR under25_odds IS NULL)
        ORDER BY date ASC
    """, (league,))
    existing = c.fetchall()

    if not existing:
        conn.close()
        logger.info("League %s: No matches need O/U odds (all already populated)", league)
        return {"league": league, "rows_updated": 0, "total_matches": 0, "seasons": 0}

    logger.info("League %s: %d matches need O/U odds — downloading CSV data...", league, len(existing))

    # Build lookup from existing matches
    match_lookup = {}
    for match_id, date_str, home, away in existing:
        match_lookup[(date_str, home.lower().strip(), away.lower().strip())] = match_id

    # Generate season codes
    today = pd.Timestamp.now()
    current_season_start_year = today.year if today.month >= 8 else today.year - 1
    seasons = []
    for i in range(max_seasons):
        sy = current_season_start_year - i
        ey = sy + 1
        seasons.append(f"{str(sy)[2:]}{str(ey)[2:]}")

    rows_updated = 0
    total_downloaded = 0
    seasons_with_data = 0

    # Download archive seasons
    for season in seasons:
        df = _download_season_csv(season, league)
        if df is None or df.empty:
            continue
        seasons_with_data += 1

        ou_cols = _find_ou_cols(df)
        if ou_cols is None:
            logger.debug("  Season %s: No O/U columns found", season)
            continue

        over_col, under_col = ou_cols
        updated = _update_from_dataframe(df, conn, match_lookup, over_col, under_col, league)
        rows_updated += updated
        total_downloaded += len(df)

    # Download current season
    df_current = _download_current_csv(league)
    if df_current is not None and not df_current.empty:
        ou_cols = _find_ou_cols(df_current)
        if ou_cols is not None:
            over_col, under_col = ou_cols
            updated = _update_from_dataframe(df_current, conn, match_lookup, over_col, under_col, league)
            rows_updated += updated
            total_downloaded += len(df_current)

    conn.close()

    logger.info("League %s: %d/%d matches updated (from %d CSV rows, %d seasons)",
                league, rows_updated, len(existing), total_downloaded, seasons_with_data)

    return {
        "league": league,
        "rows_updated": rows_updated,
        "total_matches": len(existing),
        "seasons": seasons_with_data,
    }


def _update_from_dataframe(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    match_lookup: dict[tuple[str, str, str], int],
    over_col: str,
    under_col: str,
    league: str,
) -> int:
    """Match DataFrame rows to match_lookup and UPDATE O/U odds."""
    updated = 0
    c = conn.cursor()

    for _, row in df.iterrows():
        # Parse date
        date_raw = row.get("date")
        if pd.isna(date_raw):
            continue
        try:
            date_val = pd.to_datetime(date_raw, dayfirst=True, errors="coerce")
            if pd.isna(date_val):
                continue
            date_str = date_val.strftime("%Y-%m-%d")
        except Exception:
            continue

        # Find home/away team columns
        home = str(row.get("hometeam", row.get("home_team", ""))).lower().strip()
        away = str(row.get("awayteam", row.get("away_team", ""))).lower().strip()
        if not home or not away:
            continue

        # Look up match_id
        match_id = match_lookup.get((date_str, home, away))
        if match_id is None:
            continue

        # Get O/U odds
        over_val = row.get(over_col)
        under_val = row.get(under_col)
        if pd.isna(over_val) or pd.isna(under_val):
            continue

        try:
            over_odds = float(over_val)
            under_odds = float(under_val)
            if over_odds <= 1.0 or under_odds <= 1.0:
                continue
        except (ValueError, TypeError):
            continue

        # UPDATE existing match
        c.execute(
            "UPDATE matches SET over25_odds = ?, under25_odds = ? WHERE match_id = ?",
            (over_odds, under_odds, match_id),
        )
        updated += 1

    conn.commit()
    return updated


def main():
    parser = argparse.ArgumentParser(description="Populate O/U 2.5 odds for existing matches")
    parser.add_argument("--leagues", nargs="+", help="League codes to process (default: all)")
    parser.add_argument("--max-seasons", type=int, default=10, help="Max seasons to check (default: 10)")
    args = parser.parse_args()

    # First, add columns to DB if they don't exist
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    existing_cols = {r[1] for r in c.execute("PRAGMA table_info(matches)").fetchall()}
    for col in ["over25_odds", "under25_odds"]:
        if col not in existing_cols:
            logger.info("Adding column %s to matches table...", col)
            c.execute(f"ALTER TABLE matches ADD COLUMN {col} REAL")
    conn.commit()
    conn.close()

    leagues = args.leagues or LEAGUE_CODES

    print()
    print("=" * 65)
    print("  BACKFILL OVER/UNDER 2.5 ODDS")
    print("=" * 65)
    print(f"  Leagues: {', '.join(leagues)}")
    print(f"  Max seasons: {args.max_seasons}")
    print()

    total_updated = 0
    results = []

    for league in leagues:
        start = time.time()
        result = populate_league(league, max_seasons=args.max_seasons)
        elapsed = time.time() - start
        result["elapsed"] = round(elapsed, 1)
        results.append(result)
        total_updated += result["rows_updated"]

        status = "OK" if result["rows_updated"] > 0 else "SKIP"
        seasons = result.get("seasons", 0)
        print(f"  [{status}] {league}: {result['rows_updated']:>4} rows updated "
              f"({seasons} seasons, {result['elapsed']:.1f}s)")

    print()
    print("=" * 65)
    print(f"  TOTAL: {total_updated} rows updated across {len(results)} leagues")
    print("=" * 65)


if __name__ == "__main__":
    main()
