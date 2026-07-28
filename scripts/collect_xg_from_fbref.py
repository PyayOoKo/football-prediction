"""
Collect xG data from FBref for leagues without detailed match stats.

Usage
-----
    python scripts/collect_xg_from_fbref.py

Collects xG data by parsing FBref Scores & Fixtures HTML pages saved
locally by the user (manual save mode). Updates existing rows in the DB
with home_xg/away_xg values.

Steps
-----
1. Check data/fbref_pages/ for saved HTML files
2. Parse each file with FBrefParser (extracts home_xg/away_xg)
3. For each match with xG data, UPDATE the DB row matching by
   (league, date, home_team, away_team)
4. Report counts
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_data.collectors.fbref import FBrefParser
from football_data.database import Database
from football_data.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_xg")

XG_LEAGUES = {"SE1", "NO2", "FI2", "FI3", "IRL", "D2", "P1"}


async def update_xg_in_db(
    db: Database, matches: list[dict], league: str
) -> int:
    updated = 0
    for m in matches:
        hxg = m.get("home_xg")
        axg = m.get("away_xg")
        if hxg is None and axg is None:
            continue
        sql = """
            UPDATE matches
            SET home_xg = ?, away_xg = ?
            WHERE league = ?
              AND date = ?
              AND home_team = ?
              AND away_team = ?
              AND (home_xg IS NULL OR away_xg IS NULL OR
                   home_xg != ? OR away_xg != ?)
        """
        cur = db._conn.execute(
            sql,
            (
                hxg, axg,
                m.get("league", league),
                m.get("date"),
                m.get("home_team"),
                m.get("away_team"),
                hxg, axg,
            ),
        )
        if cur.rowcount > 0:
            updated += cur.rowcount
    return updated


async def main():
    parser = FBrefParser()
    saved_dir = parser.saved_dir

    files = list(saved_dir.glob("*.html"))
    if not files:
        print(FBrefParser.print_instructions())
        print(f"\nNo HTML files found in {saved_dir}")
        print("Save FBref Scores & Fixtures pages and re-run this script.")
        return

    db = Database()
    await db.connect()

    total_updated = 0
    total_with_xg = 0
    total_matches = 0

    for fpath in sorted(files):
        league = parser._guess_league(fpath.name)
        if league == "unknown" or league not in XG_LEAGUES:
            logger.info("Skipping %s (league=%s, not in target set)", fpath.name, league)
            continue

        matches = parser.parse_saved_page(fpath)
        with_xg = sum(1 for m in matches if m.get("home_xg") is not None or m.get("away_xg") is not None)
        logger.info(
            "%s: %d matches, %d with xG data",
            fpath.name, len(matches), with_xg,
        )

        updated = await update_xg_in_db(db, matches, league)
        db._conn.commit()

        total_updated += updated
        total_with_xg += with_xg
        total_matches += len(matches)

    print()
    print(f"Files parsed:        {len(files)}")
    print(f"Total matches seen:  {total_matches}")
    print(f"Matches with xG:     {total_with_xg}")
    print(f"DB rows updated:     {total_updated}")

    await db.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
