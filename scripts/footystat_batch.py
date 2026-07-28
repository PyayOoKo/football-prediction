"""
Generate FootyStats match URLs for SE1 matches missing xG data,
then batch-parse saved HTML pages.

Usage
-----
    # Step 1: Generate URLs to open in browser
    python scripts/footystat_batch.py --list-urls

    # Step 2: After saving pages, parse them all
    python scripts/footystat_batch.py --parse
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("footystat_batch")

DB_PATH = Path("data/football_data.db")
SAVED_DIR = Path("data/scrapers/footystat")

FOOTYSTAT_SLUGS = {
    "Falkenberg": "falkenbergs-ff",
    "GIF Sundsvall": "gif-sundsvall",
    "Helsingborgs IF": "helsingborgs-if",
    "IK Brage": "ik-brage",
    "IK Oddevold": "ik-oddevold",
    "Landskrona BoIS": "landskrona-bois",
    "Ljungskile Sk": "ljungskile-sk",
    "Norrby": "norrby-if",
    "Norrköping": "ifk-norrkoping",
    "Nordic United": "united-ik-nordic",
    "Sandvikens If": "sandvikens-if",
    "Varbergs BoIS": "varbergs-bois-fc",
    "Värnamo": "ifk-varnamo",
    "Örebro": "orebro-sk",
    "Öster": "osters-if",
    "Östersund": "ostersunds-fk",
    "Utsikten": "utsiktens-bk",
    "Trelleborgs FF": "trelleborgs-ff",
    "Brage": "ik-brage",
    "Landskrona": "landskrona-bois",
    "Ljungskile": "ljungskile-sk",
    "Helsingborg": "helsingborgs-if",
    "Värnamo": "ifk-varnamo",
    "Sandviken": "sandvikens-if",
    "Östersunds FK": "ostersunds-fk",
}


def team_slug(name: str) -> str:
    name_clean = name.strip().replace("�", "o").replace("�", "a").replace("�", "a")
    return FOOTYSTAT_SLUGS.get(name_clean, name_clean.lower().replace(" ", "-"))


def list_missing():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("""
        SELECT DISTINCT date, home_team, away_team
        FROM matches
        WHERE league = 'SE1'
          AND home_goals IS NOT NULL
          AND date >= '2024-01-01'
          AND (home_shots IS NULL)
        ORDER BY date DESC
    """).fetchall()
    conn.close()

    print(f"Matches without xG/stats: {len(rows)}\n")

    urls = []
    for date, home, away in rows:
        slug_a = team_slug(home)
        slug_b = team_slug(away)
        url = f"https://footystats.org/sweden/{slug_a}-vs-{slug_b}-h2h-stats"
        urls.append((date, home, away, url))

    for date, home, away, url in urls[:30]:
        print(f"  {date}  {home:25s} vs {away:25s}")
        print(f"         {url}")
        print()
    if len(urls) > 30:
        print(f"  ... and {len(urls) - 30} more")
    print(f"\nTotal: {len(urls)} matches need data")
    print("\nOpen each URL in your browser, save as 'Webpage, HTML only'")
    print(f"to: {SAVED_DIR.resolve()}")
    print("Then run: python scripts/footystat_batch.py --parse")


def parse_all():
    from scripts.parse_footystat_match import parse_match_html, update_db
    files = sorted(SAVED_DIR.glob("*.html"))
    if not files:
        print(f"No HTML files found in {SAVED_DIR}")
        return

    ok = 0
    fail = 0
    for fpath in files:
        if fpath.name == "Superettan xG (Expected Goals) - Sweden _ FootyStats.html":
            continue
        logger.info("Parsing %s", fpath.name)
        match = parse_match_html(str(fpath))
        if not match:
            logger.warning("  Failed to parse")
            fail += 1
            continue
        if update_db(match):
            ok += 1
            logger.info("  Updated OK")
        else:
            logger.warning("  Match not found in DB")
            fail += 1

    print(f"\nDone: {ok} updated, {fail} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-urls", action="store_true", help="List matches needing xG")
    parser.add_argument("--parse", action="store_true", help="Parse saved HTML files")
    args = parser.parse_args()

    if args.list_urls:
        list_missing()
    elif args.parse:
        parse_all()
    else:
        print("Specify --list-urls or --parse")


if __name__ == "__main__":
    main()
