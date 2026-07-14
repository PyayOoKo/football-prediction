"""
collect_lineups.py — Scrape starting XI lineups from Transfermarkt match reports.

Match ID Discovery Strategy
--------------------------
Brute-force scanning of the Transfermarkt ID ranges (e.g. 3.7M+ attempts for
2022 alone) is infeasible. Instead, we use **team match history pages**:

    1. For each World Cup year (2002–2026), get all participating teams.
    2. For each team, scrape ``transfermarkt.com/{team}/alle_spiele/verein/{id}``
       — this single page lists ALL matches that team played that season, with
       links to match report pages containing the starting XI.
    3. Collect unique match IDs, verify they are World Cup matches (via page
       title), and scrape their starting XIs.

Caching
-------
Discovered match IDs are cached to ``data/external/known_match_ids.csv`` so
subsequent runs only process new matches.

Output
------
    data/external/lineups.csv — Columns: team, date, player_name
    data/external/known_match_ids.csv — Columns: match_id, year, date, home_team, away_team, scraped

Usage
-----
    python collect_lineups.py                            # All World Cup years
    python collect_lineups.py --year 2022                # Single tournament
    python collect_lineups.py --test                     # Quick test: 1 team, 1 year
    python collect_lineups.py --dry-run                  # Preview without saving
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import config
from src.data_collection.sources.transfermarkt import TEAM_TO_TM_ID
from src.data_collection.sources.transfermarkt_lineups import (
    scrape_match_lineup,
    scrape_team_matches,
    session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_lineups")

# ── Paths ───────────────────────────────────────────────

DATA_EXTERNAL = config.paths.external
WC_CSV = config.paths.raw / "worldcup_all.csv"
LINEUPS_CSV = DATA_EXTERNAL / "lineups.csv"
CACHE_CSV = DATA_EXTERNAL / "known_match_ids.csv"
PLAYERS_CSV = DATA_EXTERNAL / "players.csv"

# ── Request delay ──────────────────────────────────────
REQUEST_DELAY = 1.0  # seconds between requests (be polite)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape starting XI lineups from Transfermarkt",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Tournament year (default: all World Cup years 2002–2026)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test: scrape just Brazil+Serbia 2022 and stop",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be scraped without actually scraping lineups",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-scrape all lineups even if already cached",
    )
    return parser.parse_args(argv)


# ═══════════════════════════════════════════════════════════
#  Core logic
# ═══════════════════════════════════════════════════════════


def _is_placeholder(name: str) -> bool:
    """Check if a team name is a knockout bracket placeholder (W89, L101, …)."""
    return bool(re.match(r"^[WRLP]\d+$", name))


def get_teams_by_year(df: pd.DataFrame) -> dict[int, list[str]]:
    """Extract unique (non-placeholder) teams per tournament year."""
    teams_by_year: dict[int, list[str]] = {}
    for year in sorted(df["season"].unique()):
        year_df = df[df["season"] == year]
        teams = set(year_df["home_team"].unique()) | set(year_df["away_team"].unique())
        # Remove bracket placeholders (W89, L101, etc.)
        teams = {t for t in teams if not _is_placeholder(t)}
        teams_by_year[int(year)] = sorted(teams)
    return teams_by_year


def load_cache() -> pd.DataFrame:
    """Load previously discovered match IDs from cache."""
    if CACHE_CSV.exists():
        try:
            df = pd.read_csv(CACHE_CSV, dtype={"match_id": "int64", "year": "int64"})
            logger.info("Loaded %d cached match IDs from %s", len(df), CACHE_CSV)
            return df
        except Exception as exc:
            logger.warning("Failed to load cache: %s — starting fresh", exc)
    return pd.DataFrame(columns=["match_id", "year", "date", "home_team", "away_team", "scraped"])


def save_cache(df: pd.DataFrame) -> None:
    """Save discovered match IDs to cache."""
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["match_id"], keep="last")
    df = df.sort_values(["year", "match_id"]).reset_index(drop=True)
    df.to_csv(CACHE_CSV, index=False)
    logger.info("Saved %d match IDs to %s", len(df), CACHE_CSV)


def load_existing_lineups() -> set[int]:
    """Return the set of match_ids whose lineups have already been scraped."""
    if LINEUPS_CSV.exists():
        return set()
    # We track this via the cache's 'scraped' column, not the lineups file
    return set()


def _is_world_cup_html(html: str) -> bool:
    """Check if a Transfermarkt HTML page is a World Cup match by reading the title."""
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1)
        if "World Cup" in title or "WM " in title or "Weltmeisterschaft" in title:
            return True
    return False


def discover_match_ids(
    year: int,
    teams: list[str],
    existing_cache: pd.DataFrame,
    sess: requests.Session,
    delay: float = REQUEST_DELAY,
) -> pd.DataFrame:
    """Discover Transfermarkt match IDs for a World Cup year.

    For each team, scrapes their match history page and extracts links to
    match report pages.  Returns a DataFrame of newly discovered match IDs
    (not yet in the cache).
    """
    already_known = set(existing_cache["match_id"].unique()) if not existing_cache.empty else set()
    discovered: list[dict[str, Any]] = []

    for i, team in enumerate(teams):
        if team not in TEAM_TO_TM_ID:
            logger.debug("  [%d/%d] %s: no TM ID — skipping", i + 1, len(teams), team)
            continue

        logger.info("  [%d/%d] %s (%d) …", i + 1, len(teams), team, year)

        try:
            matches_df = scrape_team_matches(team, season=year, sess=sess)
        except Exception as exc:
            logger.debug("    [W] Failed to scrape %s history: %s", team, exc)
            continue

        if matches_df.empty:
            logger.debug("    -> no matches found")
            continue

        new_from_team = 0
        for _, row in matches_df.iterrows():
            mid = row.get("match_id")
            if mid is None or mid in already_known:
                continue

            # Record the match — we'll verify it's a World Cup match later
            discovered.append({
                "match_id": mid,
                "year": year,
                "date": str(row.get("date", "")),
                "home_team": str(row.get("home_team", "")),
                "away_team": str(row.get("away_team", "")),
                "scraped": False,
            })
            already_known.add(mid)
            new_from_team += 1

        logger.info("    -> %d new match IDs (from this team)", new_from_team)

        if i < len(teams) - 1:
            time.sleep(delay)

    if discovered:
        df_new = pd.DataFrame(discovered)
        logger.info("  Total new match IDs discovered for %d: %d", year, len(df_new))
        return df_new

    return pd.DataFrame(columns=["match_id", "year", "date", "home_team", "away_team", "scraped"])


def verify_and_scrape(
    cache_df: pd.DataFrame,
    sess: requests.Session,
    dry_run: bool = False,
    force: bool = False,
    delay: float = REQUEST_DELAY,
) -> pd.DataFrame:
    """For unscraped match IDs, verify they're World Cup matches and scrape lineups."""
    unscraped = cache_df[cache_df["scraped"] != True].copy() if not force else cache_df.copy()
    if unscraped.empty:
        logger.info("All cached matches already scraped")
        return cache_df

    logger.info("Checking %d unscraped match IDs for World Cup status …", len(unscraped))

    # Sort by year then match_id for a logical order
    unscraped = unscraped.sort_values(["year", "match_id"])

    all_lineup_dfs: list[pd.DataFrame] = []
    verified_ids: set[int] = set()
    non_wc_ids: set[int] = set()

    for i, (idx, row) in enumerate(unscraped.iterrows()):
        mid = int(row["match_id"])
        year = int(row["year"])

        url = f"https://www.transfermarkt.com/index/spielbericht/{mid}"
        logger.info("  [%d/%d] Match %d …", i + 1, len(unscraped), mid)

        if dry_run:
            logger.info("    [DRY RUN] Would scrape lineups for match %d", mid)
            cache_df.at[idx, "scraped"] = True
            continue

        # Fetch the page ONCE — share between verification and scraping
        try:
            resp = sess.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("    [W] Failed to fetch %s: %s", url, exc)
            continue

        html = resp.text

        # Verify it's a World Cup match from the already-fetched HTML
        if not force and mid not in verified_ids:
            if not _is_world_cup_html(html):
                logger.debug("    -> not a World Cup match — marking as scraped to skip future checks")
                cache_df.at[idx, "scraped"] = True
                non_wc_ids.add(mid)
                if i < len(unscraped) - 1:
                    time.sleep(delay / 2)
                continue
            verified_ids.add(mid)

        # Scrape lineups from the SAME HTML (no second fetch)
        try:
            lineup_df = scrape_match_lineup(url, sess=sess, html=html)
            if lineup_df.empty:
                logger.info("    -> no lineups found (might be an invalid match page)")
                cache_df.at[idx, "scraped"] = True
            else:
                n_players = len(lineup_df)
                teams_found = lineup_df["team"].unique()
                logger.info("    -> %d players (%s)", n_players, ", ".join(teams_found))
                all_lineup_dfs.append(lineup_df)
                cache_df.at[idx, "scraped"] = True
        except Exception as exc:
            logger.warning("    [W] Failed to scrape match %d: %s", mid, exc)

        if i < len(unscraped) - 1:
            time.sleep(delay)

    # Save combined lineups
    if all_lineup_dfs:
        combined = pd.concat(all_lineup_dfs, ignore_index=True)
        # Deduplicate just in case
        combined = combined.drop_duplicates(subset=["team", "date", "player_name"], keep="first")
        DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
        combined.to_csv(LINEUPS_CSV, index=False)
        logger.info(
            "Saved %d lineup records (%d matches) to %s",
            len(combined), len(all_lineup_dfs), LINEUPS_CSV,
        )

    logger.info(
        "Results: %d World Cup matches scraped, %d non-WC skipped",
        len(verified_ids - non_wc_ids),
        len(non_wc_ids),
    )

    return cache_df


def run_test_mode(sess: requests.Session) -> None:
    """Quick integration test: scrape Brazil & Serbia 2022."""
    logger.info("TEST MODE: scraping Brazil and Serbia 2022 match history …")

    for team, year in [("Brazil", 2022), ("Serbia", 2022)]:
        df = scrape_team_matches(team, year, sess=sess)
        print(f"\n  {team} {year}: {len(df)} matches found")
        if not df.empty:
            print(f"  Columns: {list(df.columns)}")
            for _, row in df.head(5).iterrows():
                print(f"    {row.get('date','?')} | {row.get('home_team','?')} vs {row.get('away_team','?')} | ID={row.get('match_id','?')}")

    # Known match: Brazil vs Serbia 2022
    url = "https://www.transfermarkt.com/brazil_serbia/index/spielbericht/3788846"
    print(f"\n  Testing lineup scrape: {url}")
    lineup_df = scrape_match_lineup(url, sess=sess)
    if not lineup_df.empty:
        print(f"  Lineup: {len(lineup_df)} players")
        for team in sorted(lineup_df["team"].unique()):
            players = lineup_df[lineup_df["team"] == team]["player_name"].tolist()
            print(f"    {team}: {', '.join(players)}")
    else:
        print("  No lineup data found")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ── Load World Cup data ──────────────────────────────
    if not WC_CSV.exists():
        logger.error("World Cup data not found at %s", WC_CSV)
        logger.error("Run: python refresh_worldcup.py --skip-train")
        return 1

    df = pd.read_csv(WC_CSV, parse_dates=["date"])
    teams_by_year = get_teams_by_year(df)

    if args.year:
        if args.year not in teams_by_year:
            logger.error("No data for year %d", args.year)
            return 1
        teams_by_year = {args.year: teams_by_year[args.year]}

    logger.info(
        "World Cup data loaded: %d tournaments (%s)",
        len(teams_by_year),
        ", ".join(str(y) for y in sorted(teams_by_year.keys())),
    )

    # ── Create session ───────────────────────────────────
    sess = session()

    # ── Test mode ────────────────────────────────────────
    if args.test:
        run_test_mode(sess)
        return 0

    # ── Load cache ───────────────────────────────────────
    cache_df = load_cache()

    # ── Discover match IDs per year ──────────────────────
    logger.info("─" * 50)
    logger.info("STEP 1: Discover match IDs")
    logger.info("─" * 50)

    all_new: list[pd.DataFrame] = []
    for year in sorted(teams_by_year.keys()):
        teams = teams_by_year[year]
        logger.info("Discovering match IDs for %d %s …", year, teams)

        new_df = discover_match_ids(year, teams, cache_df, sess)
        if not new_df.empty:
            all_new.append(new_df)

    if all_new:
        cache_df = pd.concat([cache_df] + all_new, ignore_index=True)
        save_cache(cache_df)
    else:
        logger.info("No new match IDs discovered")

    # ── Verify & scrape lineups ──────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 2: Verify & scrape lineups")
    logger.info("─" * 50)

    cache_df = verify_and_scrape(cache_df, sess, dry_run=args.dry_run, force=args.force)
    save_cache(cache_df)

    # ── Summary ──────────────────────────────────────────
    n_total = len(cache_df)
    n_scraped = cache_df["scraped"].sum() if "scraped" in cache_df.columns else 0

    logger.info("─" * 50)
    logger.info("SUMMARY")
    logger.info("─" * 50)
    logger.info("  Total match IDs in cache:  %d", n_total)
    logger.info("  Successfully scraped:      %d", int(n_scraped))
    logger.info("  Remaining (non-WC/failed): %d", n_total - int(n_scraped))

    if LINEUPS_CSV.exists():
        lineups = pd.read_csv(LINEUPS_CSV)
        n_teams = lineups["team"].nunique() if "team" in lineups.columns else 0
        n_matches = lineups.groupby(["team", "date"]).ngroups if not lineups.empty else 0
        logger.info("  Lineup CSV:                 %d records, %d teams, ~%d matches",
                     len(lineups), n_teams, n_matches)

    return 0


if __name__ == "__main__":
    sys.exit(main())
