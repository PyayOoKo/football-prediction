"""
Collect closing odds from all available sources and store in the database.

Usage:
    python scripts/collect_closing_odds.py

Options:
    --source    Source(s) to collect from (default: all)
                Options: football_data, oddsportal, betexplorer
    --start     Start date (ISO format, e.g. 2024-01-01)
    --end       End date (ISO format, e.g. 2024-12-31)
    --leagues   League code(s) to collect (comma-separated, default: E0)
    --dry-run   Print what would be collected without storing

Examples:
    # Collect all sources for all available data
    python scripts/collect_closing_odds.py

    # Collect only Football-Data for a specific date range
    python scripts/collect_closing_odds.py --source football_data \\
        --start 2024-01-01 --end 2024-12-31

    # Dry run to see what would be collected
    python scripts/collect_closing_odds.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("collect_closing_odds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect closing odds from multiple sources",
    )
    parser.add_argument(
        "--source", nargs="+",
        default=["football_data", "oddsportal", "betexplorer"],
        help="Source(s) to collect from (default: all)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (ISO format, e.g. 2024-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (ISO format, e.g. 2024-12-31)",
    )
    parser.add_argument(
        "--leagues", type=str, default=None,
        help="League code(s) (comma-separated, default: E0 for football-data)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be collected without storing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse dates
    start_date = date.fromisoformat(args.start) if args.start else None
    end_date = date.fromisoformat(args.end) if args.end else None
    leagues = args.leagues.split(",") if args.leagues else None

    from src.data_collection.closing_odds import (
        ClosingOddsOrchestrator,
        FootballDataClosingOddsCollector,
        OddsPortalClosingOddsCollector,
        BetExplorerClosingOddsCollector,
    )

    if args.dry_run:
        logger.info("DRY RUN — will not store anything")
        for source_name in args.source:
            if source_name == "football_data":
                collector = FootballDataClosingOddsCollector()
            elif source_name == "oddsportal":
                collector = OddsPortalClosingOddsCollector()
            elif source_name == "betexplorer":
                collector = BetExplorerClosingOddsCollector()
            else:
                logger.warning("Unknown source: %s", source_name)
                continue

            records = collector.collect(
                start_date=start_date,
                end_date=end_date,
                leagues=leagues,
            )
            logger.info(
                "  %s: would collect %d records",
                source_name, len(records),
            )
            if records:
                # Show sample
                for r in records[:3]:
                    logger.info(
                        "    %s | %s vs %s | 1=%s X=%s 2=%s",
                        r.match_date, r.home_team, r.away_team,
                        r.odds_home, r.odds_draw, r.odds_away,
                    )
        return

    # Full collection
    orchestrator = ClosingOddsOrchestrator()
    logger.info("Starting closing odds collection...")
    logger.info("  Sources: %s", ", ".join(args.source))
    logger.info("  Date range: %s to %s", start_date or "any", end_date or "any")
    logger.info("  Leagues: %s", leagues or "all")

    results = orchestrator.collect_all(
        sources=args.source,
        start_date=start_date,
        end_date=end_date,
        leagues=leagues,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("  CLOSING ODDS COLLECTION RESULTS")
    print(f"{'=' * 60}")
    for source, count in sorted(results.items()):
        status = "OK" if count > 0 else "EMPTY"
        print(f"  [{status}] {source:<25s} {count:>6d} records")
    total = sum(results.values())
    print(f"  {'─' * 40}")
    print(f"  TOTAL{'':<25s} {total:>6d} records")
    print(f"{'=' * 60}")
    print()


if __name__ == "__main__":
    main()
