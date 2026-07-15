"""
Collect referee statistics from FBref for competitions.

Usage:
    python scripts/collect_referee_stats.py
    python scripts/collect_referee_stats.py --competition 9 --season 2024-2025
    python scripts/collect_referee_stats.py --all-leagues
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

# Default competition map for --all-leagues
ALL_LEAGUES = {
    "9": "Premier League",
    "12": "La Liga",
    "11": "Serie A",
    "20": "Bundesliga",
    "13": "Ligue 1",
    "19": "Eredivisie",
    "32": "Primeira Liga",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect referee statistics")
    parser.add_argument("--competition", type=str, default="9",
                        help="FBref competition ID (default: 9 = Premier League)")
    parser.add_argument("--season", type=str, default="2024-2025",
                        help="Season slug (default: 2024-2025)")
    parser.add_argument("--all-leagues", action="store_true",
                        help="Scrape all major European leagues")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay between requests in seconds (default: 2.0)")
    args = parser.parse_args()

    from src.data_collection.sources.referee_stats import scrape_referees

    output = args.output or str(config.paths.external / config.referee_collector.output_file)
    out_dir = Path(output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_leagues:
        all_dfs = []
        for comp_id, league_name in ALL_LEAGUES.items():
            logger.info("Collecting referees for %s (%s)...", league_name, comp_id)
            try:
                df = scrape_referees(
                    competition_id=comp_id,
                    season=args.season,
                    delay=args.delay,
                )
                if not df.empty:
                    all_dfs.append(df)
                    logger.info("  -> %d referees", len(df))
            except Exception as exc:
                logger.warning("  -> Failed: %s", exc)

        if all_dfs:
            import pandas as pd
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(output, index=False)
            logger.info("Combined %d total rows → %s", len(combined), output)
        else:
            logger.warning("No referee data collected for any league")
    else:
        logger.info("Collecting referees for competition %s, season %s...",
                    args.competition, args.season)
        df = scrape_referees(
            competition_id=args.competition,
            season=args.season,
            delay=args.delay,
            save_path=output,
        )
        logger.info("Done — %d rows collected", len(df))
        logger.info("Output: %s", output)


if __name__ == "__main__":
    main()
