"""
collect_understat_xg.py — Fetch real xG data from Understat for top 5 European leagues.

Replaces estimated xG (from shots-on-target conversion) with real xG from Understat's
per-shot model, which is the gold standard for freely available xG data.

How it works
------------
1. For each top-5 league (EPL, La Liga, Bundesliga, Serie A, Ligue 1):
   a. Fetch Understat's league page (contains per-match xG for all matches in a season)
   b. Parse match-level home_xg and away_xg values
   c. Normalise team names using TeamNormalizer (Understat ↔ DB name matching)
   d. UPDATE the football_data.db matches table with real xG values

2. Incremental: tracks which (league, season) pairs have been collected.
   Skips already-processed seasons unless --force is used.

3. Reports: prints a summary of matches updated, skipped, and any match errors.

Usage
-----
    python scripts/collect_understat_xg.py                         # All 5 leagues, 5 seasons each
    python scripts/collect_understat_xg.py --leagues E0 SP1         # Specific leagues only
    python scripts/collect_understat_xg.py --seasons 3              # Only last 3 seasons
    python scripts/collect_understat_xg.py --force                  # Re-fetch already-collected seasons
    python scripts/collect_understat_xg.py --dry-run                # Show what would be done without writing

League Mapping
--------------
    our code      Understat code
    ──────────    ──────────────
    E0            EPL            (Premier League)
    SP1           La_liga        (La Liga)
    D1            Bundesliga     (Bundesliga)
    I1            Serie_A        (Serie A)
    F1            Ligue_1        (Ligue 1)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Ensure project root is on sys.path ───────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect_understat_xg")

# ── Paths ────────────────────────────────────────────────
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
STATE_FILE = PROJECT_ROOT / "data" / "scrapers" / "understat" / "xg_collected.json"

# ── League mapping: our DB code → Understat league code ──
LEAGUE_MAP: dict[str, str] = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}
UNDERSTAT_TO_DB = {v: k for k, v in LEAGUE_MAP.items()}

# Pretty names for logging
LEAGUE_DISPLAY: dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# Recent seasons to collect (starting year for Understat)
# Understat uses the season start year: 2024 = 2024-2025 season
RECENT_SEASONS = [2026, 2025, 2024, 2023, 2022, 2021, 2020]


# ═══════════════════════════════════════════════════════════
#  State management (incremental collection tracking)
# ═══════════════════════════════════════════════════════════


def load_state() -> dict[str, list[int]]:
    """Load the set of already-collected (league, year) pairs."""
    state_path = Path(STATE_FILE)
    if state_path.exists():
        try:
            with open(state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load state file: %s", exc)
    return {"collected": {}}


def save_state(state: dict[str, Any]) -> None:
    """Save collection state."""
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("State saved to %s", STATE_FILE)


def is_collected(state: dict[str, Any], league: str, year: int) -> bool:
    """Check if a (league, year) pair has been collected."""
    collected = state.get("collected", {})
    years = collected.get(league, [])
    return year in years


def mark_collected(state: dict[str, Any], league: str, year: int) -> None:
    """Mark a (league, year) pair as collected."""
    if "collected" not in state:
        state["collected"] = {}
    if league not in state["collected"]:
        state["collected"][league] = []
    if year not in state["collected"][league]:
        state["collected"][league].append(year)
        state["collected"][league].sort()


# ═══════════════════════════════════════════════════════════
#  Team name matching
# ═══════════════════════════════════════════════════════════


def build_team_map(league: str, conn: sqlite3.Connection) -> dict[str, str]:
    """Build a mapping from Understat team names → DB team names.

    Uses the TeamNormalizer to canonicalise both Understat and DB names,
    then matches by canonical form. Falls back to direct case-insensitive
    comparison if normalizer can't resolve.

    Parameters
    ----------
    league : str
        DB league code (E0, SP1, etc.).
    conn : sqlite3.Connection
        Open DB connection.

    Returns
    -------
    dict[str, str]
        Understat team name → DB team name.
    """
    from src.team_normalizer import TeamNormalizer

    normalizer = TeamNormalizer()

    # Load all DB team names for this league
    cursor = conn.execute(
        """
        SELECT DISTINCT home_team FROM matches WHERE league = ?
        UNION
        SELECT DISTINCT away_team FROM matches WHERE league = ?
        """,
        (league, league),
    )
    db_teams = sorted({r[0] for r in cursor.fetchall()})

    # Canonicalise DB names
    db_canonical: dict[str, str] = {}  # canonical → db_name
    for db_name in db_teams:
        result = normalizer.resolve(db_name)
        canonical = result.canonical
        # Only use if resolved with decent confidence
        if result.confidence >= 0.6:
            db_canonical.setdefault(canonical.lower(), db_name)
        else:
            db_canonical.setdefault(db_name.lower(), db_name)

    return db_canonical


def match_team(
    understat_name: str, db_canonical: dict[str, str]
) -> tuple[str | None, str]:
    """Match an Understat team name to a DB team name.

    Returns
    -------
    tuple[str | None, str]
        (db_name, method) — db_name is None if unmatched.
        method describes how the match was made.
    """
    from src.team_normalizer import TeamNormalizer

    normalizer = TeamNormalizer()
    result = normalizer.resolve(understat_name)
    canonical = result.canonical.lower()

    # Direct canonical match
    if canonical in db_canonical:
        return db_canonical[canonical], f"canonical({result.method})"

    # Fuzzy match on canonical if needed
    # Try case-insensitive direct match
    for db_key, db_name in db_canonical.items():
        if canonical == db_key.lower():
            return db_name, "case_insensitive"

    # Fallback: return None
    return None, "unmatched"


# ═══════════════════════════════════════════════════════════
#  Data collection & DB update
# ═══════════════════════════════════════════════════════════


async def collect_league_xg(
    league: str,
    years: list[int],
    conn: sqlite3.Connection,
    state: dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Collect real xG data from Understat for a single league.

    Parameters
    ----------
    league : str
        DB league code (E0, SP1, etc.).
    years : list[int]
        Season start years to collect.
    conn : sqlite3.Connection
        Open DB connection.
    state : dict
        Collection state for incremental tracking.
    force : bool
        Re-collect already-collected seasons.
    dry_run : bool
        Don't write to DB.

    Returns
    -------
    dict[str, Any]
        Stats: matches_updated, already_had_xg, skipped, unmatched, errors.
    """
    understat_code = LEAGUE_MAP[league]
    league_display = LEAGUE_DISPLAY.get(league, league)
    print()
    print(f"{'=' * 60}")
    print(f"  {league_display} ({league}) - Understat: {understat_code}")
    print(f"{'=' * 60}")

    from src.data_collection.sources.understat.importer import UnderstatImporter

    importer = UnderstatImporter()

    stats = {
        "matches_updated": 0,
        "already_had_xg": 0,
        "skipped_no_match": 0,
        "unmatched_teams": set(),
        "errors": [],
        "total_matches": 0,
    }

    # Build team name map once per league
    db_canonical = build_team_map(league, conn)
    logger.info("Built team map: %d DB names for %s", len(db_canonical), league)

    try:
        for year in years:
            if not force and is_collected(state, league, year):
                logger.info("  [SKIP] %s/%d already collected", understat_code, year)
                continue

            # Fetch Understat match-level xG data for this league/season
            try:
                teams, matches = await importer.get_league_xg(understat_code, year)
            except Exception as exc:
                msg = f"Failed to fetch {understat_code}/{year}: {exc}"
                logger.warning("  [ERR] %s", msg)
                stats["errors"].append(msg)
                continue

            if not matches:
                logger.info("  [SKIP] %s/%d: no match data returned", understat_code, year)
                continue

            logger.info(
                "  [FETCH] %s/%d: %d teams, %d matches",
                understat_code, year, len(teams), len(matches),
            )
            stats["total_matches"] += len(matches)

            # Pre-fetch existing xG for all matches in this season (batch query)
            # Use generous date range to catch all matches in the season
            season_start = f"{year}-06-01"
            season_end = f"{year + 1}-08-31"
            cursor = conn.execute(
                "SELECT date, home_team, away_team, home_xg, away_xg FROM matches "
                "WHERE league = ? AND date >= ? AND date <= ?",
                (league, season_start, season_end),
            )
            db_matches: dict[tuple[str, str, str], tuple[float | None, float | None]] = {}
            for r in cursor.fetchall():
                key = (r["date"], r["home_team"], r["away_team"])
                db_matches[key] = (r["home_xg"], r["away_xg"])

            season_updated = 0
            season_already = 0
            season_no_match = 0
            season_batch: list[tuple[float, float, str, str, str]] = []

            for match in matches:
                understat_home = match.home_team
                understat_away = match.away_team
                h_xg = match.home_xg
                a_xg = match.away_xg
                date_str = match.date

                # Match teams via canonical normalisation
                db_home, _ = match_team(understat_home, db_canonical)
                db_away, _ = match_team(understat_away, db_canonical)

                if db_home is None or db_away is None:
                    if db_home is None:
                        stats["unmatched_teams"].add(understat_home)
                    if db_away is None:
                        stats["unmatched_teams"].add(understat_away)
                    season_no_match += 1
                    continue

                if dry_run:
                    season_updated += 1
                    continue

                # Check if this match already has real xG
                db_key = (date_str, db_home, db_away)
                existing = db_matches.get(db_key)
                if existing is not None:
                    existing_h, existing_a = existing
                    if existing_h is not None and existing_h != 0 and existing_a is not None and existing_a != 0:
                        season_already += 1
                        continue

                # Queue for update
                season_batch.append((h_xg, a_xg, date_str, db_home, db_away))

            # Batch-update all matches for this season in a single commit
            if season_batch:
                conn.executemany(
                    "UPDATE matches SET home_xg = ?, away_xg = ? "
                    "WHERE league = ? AND date = ? AND home_team = ? AND away_team = ?",
                    [(h, a, league, d, home, away) for h, a, d, home, away in season_batch],
                )

            conn.commit()
            season_updated = len(season_batch)

            # Log season summary
            logger.info(
                "  [DONE] %s/%d: %d updated, %d already had xG, %d no match",
                understat_code, year, season_updated, season_already, season_no_match,
            )

            stats["matches_updated"] += season_updated
            stats["already_had_xg"] += season_already
            stats["skipped_no_match"] += season_no_match

            if not dry_run:
                mark_collected(state, league, year)
                save_state(state)

            await asyncio.sleep(0.5)  # Be polite

    finally:
        await importer.close()

    return stats


# ═══════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════


def print_summary(all_stats: dict[str, dict[str, Any]], dry_run: bool = False):
    """Print a comprehensive summary of the collection run."""
    print()
    print("=" * 65)
    print("  UNDERSTAT xG COLLECTION SUMMARY")
    print("=" * 65)

    grand_total_updated = 0
    grand_total_already = 0
    grand_total_skipped = 0
    grand_total_matches = 0
    all_unmatched: set[str] = set()

    for league, stats in sorted(all_stats.items()):
        display = LEAGUE_DISPLAY.get(league, league)
        updated = stats["matches_updated"]
        already = stats["already_had_xg"]
        skipped = stats["skipped_no_match"]
        total = stats["total_matches"]
        errors = len(stats["errors"])
        unmatched = stats["unmatched_teams"]

        grand_total_updated += updated
        grand_total_already += already
        grand_total_skipped += skipped
        grand_total_matches += total
        all_unmatched.update(unmatched)

        status = "OK" if errors == 0 else "ERR"
        print(
            f"  [{status}] {league:6s} {display:25s} "
            f"Updated: {updated:>4}  Already: {already:>4}  "
            f"NoMatch: {skipped:>3}  Errors: {errors}"
        )

    print()
    print(f"  {'-' * 55}")
    print(f"  TOTAL         {'':25s} Updated: {grand_total_updated:>4}  "
          f"Already: {grand_total_already:>4}  "
          f"NoMatch: {grand_total_skipped:>3}")
    print(f"  Total matches processed: {grand_total_matches}")

    if dry_run:
        print()
        print("  [DRY RUN] No data was written to the database")

    if all_unmatched:
        print()
        print(f"  Unmatched teams ({len(all_unmatched)}):")
        for team in sorted(all_unmatched):
            print(f"    - {team}")
        print()
        print("  These Understat team names could not be matched to DB names.")
        print("  Check if they need to be added to the team normalizer registry.")

    # Print per-league details
    for league, stats in sorted(all_stats.items()):
        if stats["errors"]:
            print()
            print(f"  Errors for {league}:")
            for err in stats["errors"]:
                print(f"    - {err}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


async def main():
    parser = argparse.ArgumentParser(
        description="Fetch real xG data from Understat for top 5 European leagues"
    )
    parser.add_argument(
        "--leagues", nargs="+",
        choices=list(LEAGUE_MAP.keys()),
        help="Leagues to process (default: all top 5)",
    )
    parser.add_argument(
        "--seasons", type=int, default=5,
        help="Number of recent seasons to collect (default: 5)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-collect already-collected seasons",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing to DB",
    )
    parser.add_argument(
        "--reset-state", action="store_true",
        help="Reset all collection state before starting",
    )
    args = parser.parse_args()

    leagues = args.leagues or list(LEAGUE_MAP.keys())
    seasons = RECENT_SEASONS[:max(args.seasons, 1)]

    print()
    print("=" * 65)
    print("  COLLECT REAL xG FROM UNDERSTAT")
    print("=" * 65)
    print(f"  Leagues: {', '.join(leagues)}")
    season_str = f"{seasons[0]} to {seasons[-1]} ({len(seasons)} seasons)"
    print(f"  Seasons: {season_str}")
    print(f"  Force:   {'YES' if args.force else 'No (incremental)'}")
    print(f"  Mode:    {'DRY RUN' if args.dry_run else 'LIVE'}")

    # Load state
    state = load_state()
    if args.reset_state:
        state = {"collected": {}}
        logger.info("State reset to empty")
        save_state(state)

    # Connect to DB
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    all_stats: dict[str, dict[str, Any]] = {}

    for league in leagues:
        stats = await collect_league_xg(
            league, seasons, conn, state,
            force=args.force, dry_run=args.dry_run,
        )
        all_stats[league] = stats

    conn.close()
    save_state(state)

    print_summary(all_stats, dry_run=args.dry_run)

    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
