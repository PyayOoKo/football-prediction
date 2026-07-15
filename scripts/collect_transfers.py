"""
Collect transfer history data for World Cup and league teams.

Usage:
    python scripts/collect_transfers.py
    python scripts/collect_transfers.py --teams Brazil,England,Germany
    python scripts/collect_transfers.py --max-windows 5 --output data/external/transfers.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect transfer data")
    parser.add_argument("--teams", type=str, default=None,
                        help="Comma-separated team names (default: all supported)")
    parser.add_argument("--max-windows", type=int, default=5,
                        help="Max transfer windows per team (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Delay between requests in seconds (default: 1.5)")
    args = parser.parse_args()

    from src.data_collection.sources.transfermarkt import TEAM_TO_TM_ID
    from src.data_collection.sources.transfers import scrape_transfers

    # Determine teams to scrape
    if args.teams:
        team_names = [t.strip() for t in args.teams.split(",")]
    else:
        # Use all teams with TM IDs (deduplicated)
        seen: set[str] = set()
        team_names = []
        for name in sorted(TEAM_TO_TM_ID):
            if name not in seen:
                seen.add(name)
                team_names.append(name)

    output = args.output or str(config.paths.external / config.transfer_collector.output_file)
    logger.info("Collecting transfers for %d teams (max %d windows each)...",
                len(team_names), args.max_windows)

    df = scrape_transfers(
        team_names=team_names,
        max_windows=args.max_windows,
        delay=args.delay,
        save_path=output,
    )

    logger.info("Done — %d rows collected", len(df))
    logger.info("Output: %s", output)


if __name__ == "__main__":
    main()
