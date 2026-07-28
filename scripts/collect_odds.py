"""
collect_odds.py — Collect Over/Under 2.5 and BTTS odds for all leagues.

Steps:
1. Backfill O/U 2.5 odds from football-data.co.uk CSV archives
2. Backfill BTTS odds (BbAv>2.5/BbAv<2.5 equivalent structure)
3. Export all odds to structured CSV for model training

Output:
    data/odds.csv — All available betting odds per match

Usage:
    python scripts/collect_odds.py                          # All leagues
    python scripts/collect_odds.py --leagues E0 F1          # Specific leagues
    python scripts/collect_odds.py --max-seasons 5          # Last 5 seasons only
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_odds")

DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
OUTPUT_DIR = PROJECT_ROOT / "data"

# All known league codes from football-data.co.uk
LEAGUE_CODES = [
    "E0", "E1", "E2", "E3", "EC", "SC0",
    "D1", "D2", "I1", "I2",
    "SP1", "SP2",
    "F1", "F2",
    "N1", "B1", "P1", "T1",
    "SE1", "SE2",
    "NO1", "NO2",
    "FI1", "FI2",
    "SWE", "NOR", "FIN", "DEN",
    "JPN", "MEX", "USA", "BRA", "ARG",
    "AUT", "POL", "SUI", "IRL",
]

BASE_URL = "https://www.football-data.co.uk"
MMZ_PATH = "mmz4281"
NEW_CSV_URL = "https://www.football-data.co.uk/new/{league}.csv"

# O/U 2.5 columns in priority order (best to fallback)
OU_COLUMN_PAIRS = [
    ("bbav>2.5", "bbav<2.5"),      # Best: BetBrain average closing
    ("avg>2.5", "avg<2.5"),        # Average across all bookmakers
    ("b365>2.5", "b365<2.5"),      # Bet365 (most common)
    ("bf>2.5", "bf<2.5"),          # Betfair
    ("p>2.5", "p<2.5"),            # Pinnacle
    ("wh>2.5", "wh<2.5"),          # William Hill
    ("vc>2.5", "vc<2.5"),          # VC Bet
    ("iwc>2.5", "iwc<2.5"),        # IWC
    ("ps>2.5", "ps<2.5"),          # Paddy Power
    ("sb>2.5", "sb<2.5"),          # Sportingbet
    ("sj>2.5", "sj<2.5"),          # Stan James
    ("lb>2.5", "lb<2.5"),          # Ladbrokes
]

# BTTS columns — football-data.co.uk has single-column "yes" odds for BTTS.
# The implied "no" odds will be derived (1 / (1 - 1/yes_odds)) for model training.
BTTS_COLUMNS = [
    "bbavbtts",    # BetBrain average (best quality)
    "b365btts",    # Bet365
    "btts",        # Generic
    "avbtts",      # Average
    "pintbtts",    # Pinnacle
]


def ensure_db_columns():
    """Ensure required columns exist in the matches table."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    existing = {r[1] for r in c.execute("PRAGMA table_info(matches)").fetchall()}

    needed = {
        "over25_odds": "REAL",
        "under25_odds": "REAL",
        "btts_yes_odds": "REAL",
        "btts_no_odds": "REAL",
    }
    for col, dtype in needed.items():
        if col not in existing:
            logger.info("Adding column %s to matches table...", col)
            c.execute(f"ALTER TABLE matches ADD COLUMN {col} {dtype}")

    conn.commit()
    conn.close()


def download_csv(url: str, timeout: int = 30) -> pd.DataFrame | None:
    """Download a CSV and return with lowercase columns."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
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


def find_ou_cols(df: pd.DataFrame) -> tuple[str, str] | None:
    """Find the best available O/U 2.5 column pair."""
    for over_col, under_col in OU_COLUMN_PAIRS:
        if over_col in df.columns and under_col in df.columns:
            return over_col, under_col
    return None


def find_btts_col(df: pd.DataFrame) -> str | None:
    """Find the best available single BTTS "yes" column."""
    for col in BTTS_COLUMNS:
        if col in df.columns:
            return col
    return None


def parse_date(date_val) -> str | None:
    """Try to parse a date value from CSV to YYYY-MM-DD format."""
    try:
        date_val = pd.to_datetime(date_val, dayfirst=True, errors="coerce")
        if pd.notna(date_val):
            return date_val.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def backfill_league(league: str, max_seasons: int = 10) -> dict:
    """Backfill O/U and BTTS odds for a single league."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Get matches that need odds
    c.execute("""
        SELECT match_id, date, home_team, away_team
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
        ORDER BY date ASC
    """, (league,))
    all_matches = c.fetchall()

    if not all_matches:
        conn.close()
        return {"league": league, "ou_updated": 0, "btts_updated": 0,
                "total_matches": 0, "seasons": 0}

    # Build lookup: (date, home_lower, away_lower) -> match_id
    match_lookup = {}
    for mid, dt, ht, at in all_matches:
        match_lookup[(dt, ht.lower().strip(), at.lower().strip())] = mid

    # Generate season codes
    today = pd.Timestamp.now()
    start_year = today.year if today.month >= 8 else today.year - 1
    seasons = []
    for i in range(max_seasons):
        sy = start_year - i
        ey = sy + 1
        seasons.append(f"{str(sy)[2:]}{str(ey)[2:]}")

    ou_updated = 0
    btts_updated = 0
    seasons_with_data = 0

    def _update_from_df(df: pd.DataFrame) -> tuple[int, int]:
        """Update DB from a single CSV DataFrame. Returns (ou_updated, btts_updated)."""
        nonlocal seasons_with_data
        seasons_with_data += 1
        ou = 0
        btts = 0

        ou_cols = find_ou_cols(df)
        btts_col = find_btts_col(df)

        for _, row in df.iterrows():
            date_str = parse_date(row.get("date"))
            if not date_str:
                continue

            home = str(row.get("hometeam", row.get("home_team", ""))).lower().strip()
            away = str(row.get("awayteam", row.get("away_team", ""))).lower().strip()
            if not home or not away:
                continue

            match_id = match_lookup.get((date_str, home, away))
            if match_id is None:
                continue

            # O/U odds
            if ou_cols:
                over_val = row.get(ou_cols[0])
                under_val = row.get(ou_cols[1])
                if pd.notna(over_val) and pd.notna(under_val):
                    try:
                        over_odds = float(over_val)
                        under_odds = float(under_val)
                        if over_odds > 1.0 and under_odds > 1.0:
                            c.execute(
                                "UPDATE matches SET over25_odds = ?, under25_odds = ? "
                                "WHERE match_id = ? AND (over25_odds IS NULL OR under25_odds IS NULL)",
                                (over_odds, under_odds, match_id),
                            )
                            if c.rowcount > 0:
                                ou += 1
                    except (ValueError, TypeError):
                        pass

            # BTTS odds — single "yes" column, derive "no" from inverse probability
            if btts_col:
                yes_val = row.get(btts_col)
                if pd.notna(yes_val):
                    try:
                        yes_odds = float(yes_val)
                        if yes_odds > 1.0:
                            # Derive "no" odds: no_odds = 1 / (1 - 1/yes_odds)
                            imp_prob = 1.0 / yes_odds
                            if imp_prob < 1.0:
                                no_odds = 1.0 / (1.0 - imp_prob)
                                if no_odds > 1.0:
                                    c.execute(
                                        "UPDATE matches SET btts_yes_odds = ?, btts_no_odds = ? "
                                        "WHERE match_id = ? AND btts_yes_odds IS NULL",
                                        (yes_odds, no_odds, match_id),
                                    )
                                    if c.rowcount > 0:
                                        btts += 1
                    except (ValueError, TypeError):
                        pass

        conn.commit()
        return ou, btts

    # Download archive seasons
    for season in seasons:
        url = f"{BASE_URL}/{MMZ_PATH}/{season}/{league}.csv"
        df = download_csv(url)
        if df is None or df.empty:
            continue
        o, b = _update_from_df(df)
        ou_updated += o
        btts_updated += b

    # Download current (in-progress) season
    df_current = download_csv(NEW_CSV_URL.format(league=league))
    if df_current is not None and not df_current.empty:
        o, b = _update_from_df(df_current)
        ou_updated += o
        btts_updated += b

    conn.close()

    logger.info(
        "League %s: O/U=%d, BTTS=%d updated (%d seasons, %d total matches)",
        league, ou_updated, btts_updated, seasons_with_data, len(all_matches),
    )
    return {
        "league": league,
        "ou_updated": ou_updated,
        "btts_updated": btts_updated,
        "total_matches": len(all_matches),
        "seasons": seasons_with_data,
    }


def export_odds_csv():
    """Export all available odds from DB to a structured CSV."""
    conn = sqlite3.connect(str(DB_PATH))

    # Check which columns exist — some may not be added yet
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
    odds_cols = ["home_odds", "draw_odds", "away_odds",
                 "over25_odds", "under25_odds"]
    if "btts_yes_odds" in existing_cols:
        odds_cols.extend(["btts_yes_odds", "btts_no_odds"])

    cols_sql = ",\n            ".join(odds_cols)

    query = f"""
        SELECT
            match_id, date, league, season,
            home_team, away_team,
            home_goals, away_goals,
            {cols_sql}
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Derive targets
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    # Add target implied probabilities (safe divide)
    for col_in, col_out in [
        ("over25_odds", "implied_ou_over"),
        ("under25_odds", "implied_ou_under"),
        ("btts_yes_odds", "implied_btts_yes"),
        ("btts_no_odds", "implied_btts_no"),
    ]:
        if col_in in df.columns:
            df[col_out] = 1.0 / df[col_in]
        else:
            df[col_out] = None

    output_path = OUTPUT_DIR / "odds.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    # Stats
    has_ou = df["over25_odds"].notna().sum()
    has_btts = df["btts_yes_odds"].notna().sum()
    total = len(df)

    stats = {
        "total_matches": total,
        "with_ou_odds": has_ou,
        "with_btts_odds": has_btts,
        "ou_pct": round(has_ou / total * 100, 1) if total else 0,
        "btts_pct": round(has_btts / total * 100, 1) if total else 0,
        "output_file": str(output_path),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect O/U and BTTS odds")
    parser.add_argument("--leagues", nargs="+", help="League codes (default: all)")
    parser.add_argument("--max-seasons", type=int, default=10, help="Max seasons (default: 10)")
    parser.add_argument("--export-only", action="store_true",
                        help="Skip backfill, just export current odds to CSV")
    args = parser.parse_args()

    print("=" * 65)
    print("  ODDS DATA COLLECTION — Over/Under & BTTS")
    print("=" * 65)

    leagues = args.leagues or LEAGUE_CODES

    if not args.export_only:
        # Step 1: Ensure DB columns exist
        print("\n--- Step 1: Ensuring DB columns ---")
        ensure_db_columns()
        print("  ✅ Columns ensured")

        # Step 2: Backfill odds from CSV archives
        print(f"\n--- Step 2: Backfilling odds for {len(leagues)} leagues ---")
        total_ou = 0
        total_btts = 0

        for league in leagues:
            start = time.time()
            result = backfill_league(league, max_seasons=args.max_seasons)
            elapsed = time.time() - start
            total_ou += result["ou_updated"]
            total_btts += result["btts_updated"]

            status = "✅" if result["ou_updated"] > 0 or result["btts_updated"] > 0 else "⏭️"
            print(f"  [{status}] {league}: O/U={result['ou_updated']:>4}, "
                  f"BTTS={result['btts_updated']:>3} ({result['seasons']} seasons, {elapsed:.1f}s)")

        print(f"\n  Total: O/U={total_ou}, BTTS={total_btts} rows updated")

    # Step 3: Export to CSV
    print("\n--- Final Step: Exporting odds to CSV ---")
    stats = export_odds_csv()
    print(f"  Matches: {stats['total_matches']:,}")
    print(f"  With O/U 2.5 odds: {stats['with_ou_odds']:,} ({stats['ou_pct']}%)")
    print(f"  With BTTS odds: {stats['with_btts_odds']:,} ({stats['btts_pct']}%)")
    print(f"  Output: {stats['output_file']}")

    # League breakdown
    print("\n--- Per-League Odds Coverage ---")
    conn = sqlite3.connect(str(DB_PATH))
    for league in leagues:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN over25_odds IS NOT NULL THEN 1 ELSE 0 END) as has_ou,
                SUM(CASE WHEN btts_yes_odds IS NOT NULL THEN 1 ELSE 0 END) as has_btts
            FROM matches WHERE league = ? AND home_goals IS NOT NULL
        """, (league,)).fetchone()
        if row and row[0] > 0:
            ou_pct = row[1] * 100.0 / row[0]
            btts_pct = row[2] * 100.0 / row[0]
            print(f"  {league:>4}: {row[0]:>6} matches | O/U={row[1]:>6} ({ou_pct:>4.1f}%) "
                  f"| BTTS={row[2]:>6} ({btts_pct:>4.1f}%)")
    conn.close()

    print("\n" + "=" * 65)
    print("  COLLECTION COMPLETE")
    print("=" * 65)
    print(f"  ✅ data/odds.csv — {stats['with_ou_odds']:,} O/U odds, {stats['with_btts_odds']:,} BTTS odds")
    print()


if __name__ == "__main__":
    main()
