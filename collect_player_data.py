"""collect_player_data.py — Scrape squad data from Transfermarkt for all World Cup teams.

Downloads per-player information (age, market value, position, injury status)
for every national team in the World Cup dataset, saves to ``data/external/players.csv``,
ready for ``src.player_info.add_player_features()``.

Usage:
    python collect_player_data.py                           # Scrape all World Cup teams
    python collect_player_data.py --teams Brazil,England    # Specific teams only
    python collect_player_data.py --list-teams              # List available teams
    python collect_player_data.py --dry-run                 # Preview only (no save)
    python collect_player_data.py --test                    # Quick test with Brazil only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from config import config
from src.data_collection.sources.transfermarkt import (
    TEAM_TO_TM_ID,
    scrape_squads,
    get_supported_teams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_player_data")

PLAYERS_CSV = config.paths.external / "players.csv"
"""Output path for the players DataFrame."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape squad player data from Transfermarkt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--teams", type=str, default=None,
        help="Comma-separated team names (default: all World Cup teams)",
    )
    parser.add_argument(
        "--list-teams", action="store_true",
        help="List all available teams with Transfermarkt IDs and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scrape but don't save to CSV",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test: scrape only Brazil (no save)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between requests (default 1.5, be polite)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ── List teams mode ──
    if args.list_teams:
        print("\\n  Available teams with Transfermarkt IDs:")
        print(f"  {'Team':<30} {'TM ID':<8}")
        print(f"  {'-' * 38}")
        for team, tm_id in sorted(TEAM_TO_TM_ID.items(), key=lambda x: x[0].lower()):
            print(f"  {team:<30} {tm_id:<8}")
        print(f"\\n  Total: {len(TEAM_TO_TM_ID)} teams (with aliases)")
        return 0

    # ── Determine teams to scrape ──
    if args.test:
        team_names = ["Brazil"]
        args.dry_run = True
        logger.info("TEST MODE — scraping Brazil only")
    elif args.teams:
        team_names = [t.strip() for t in args.teams.split(",")]
        logger.info("Custom teams: %s", ", ".join(team_names))
    else:
        # Get all unique teams from the World Cup data
        wc_csv = Path("data/raw/worldcup_all.csv")
        if wc_csv.exists():
            df = pd.read_csv(wc_csv)
            teams_in_data: set[str] = set()
            for col in ["home_team", "away_team"]:
                for name in df[col].dropna().unique():
                    name = str(name).strip()
                    if not any(name.startswith(p) for p in ["W", "R", "Q", "P", "L"]):
                        teams_in_data.add(name)
            # Filter to only teams we have Transfermarkt IDs for
            team_names = sorted(
                t for t in teams_in_data if t in TEAM_TO_TM_ID
            )
            logger.info(
                "Found %d teams in World Cup data with Transfermarkt IDs",
                len(team_names),
            )
        else:
            team_names = sorted(
                t for t in get_supported_teams()
                if t not in ("Bosnia & Herzegovina", "Ivory Coast")  # aliases only
            )
            logger.info(
                "No World Cup data found — scraping all supported teams (%d)",
                len(team_names),
            )

    logger.info("Scraping %d teams from Transfermarkt...", len(team_names))

    # ── Scrape ──
    t0 = time.time()
    players_df = scrape_squads(
        team_names=team_names,
        delay=args.delay,
        save_path=None if args.dry_run else str(PLAYERS_CSV),
    )
    elapsed = time.time() - t0

    # ── Report ──
    print(f"\\n{'=' * 60}")
    print(f"  PLAYER DATA COLLECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Teams scraped:  {len(team_names)}")
    print(f"  Players total:  {len(players_df)}")
    print(f"  Duration:       {elapsed:.1f}s")

    if len(players_df) > 0:
        print(f"\\n  Per-team stats:")
        print(f"  {'Team':<25} {'Players':<9} {'Avg Age':<9} {'Squad Value (€m)':<17} {'Injured':<9}")
        print(f"  {'-' * 68}")
        for (team, grp) in players_df.groupby("team"):
            avg_age = grp["age"].mean()
            squad_value = grp["market_value"].sum()
            injured = grp["injured"].sum()
            print(f"  {team:<25} {len(grp):<9} {avg_age:<9.1f} {squad_value:<17.1f} {int(injured):<9}")

    if args.dry_run:
        print(f"\\n  [DRY RUN] Not saved. Run without --dry-run to persist.")
    else:
        print(f"\\n  [OK] Saved to {PLAYERS_CSV}")
        print(f"  To use in pipeline:")
        print(f"    1. python refresh_worldcup.py")
        print(f"    2. Or run: python train_worldcup.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
