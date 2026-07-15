"""
Collect historical weather data for match dates/locations.

Usage:
    python scripts/collect_weather.py
    python scripts/collect_weather.py --matches data/raw/worldcup_all.csv
    python scripts/collect_weather.py --api-key YOUR_KEY --output data/external/weather.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect weather data for matches")
    parser.add_argument("--matches", type=str, default=None,
                        help="CSV with match data (must have date and home_team columns)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenWeatherMap API key (default: OPENWEATHER_API_KEY env var)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data/external/weather.csv)")
    parser.add_argument("--coordinates", type=str, default=None,
                        help="Custom team coordinates CSV (team,latitude,longitude)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip cached weather responses")
    args = parser.parse_args()

    from src.data_collection.sources.weather_api import collect_weather, get_team_coordinates

    # Load match data
    if args.matches:
        matches_path = Path(args.matches)
        if not matches_path.exists():
            logger.error("Match file not found: %s", matches_path)
            sys.exit(1)
        matches_df = pd.read_csv(matches_path)
    else:
        # Try common match files
        candidates = [
            config.paths.raw / "worldcup_all.csv",
            config.paths.raw / "results.csv",
            config.paths.raw / "worldcup_2026.csv",
        ]
        matches_df = pd.DataFrame()
        for c in candidates:
            if c.exists():
                matches_df = pd.read_csv(c)
                logger.info("Loaded %d matches from %s", len(matches_df), c)
                break
        if matches_df.empty:
            logger.error("No match data found. Provide a CSV with --matches")
            sys.exit(1)

    required_cols = {"date", "home_team"}
    if not required_cols.issubset(matches_df.columns):
        logger.error("Match CSV must have columns: %s", required_cols)
        sys.exit(1)

    # Load custom coordinates if provided
    lat_lon_map = None
    if args.coordinates:
        coords_path = Path(args.coordinates)
        if coords_path.exists():
            coords_df = pd.read_csv(coords_path)
            lat_lon_map = {
                row["team"]: (row["latitude"], row["longitude"])
                for _, row in coords_df.iterrows()
            }
            logger.info("Loaded %d team coordinates from %s", len(lat_lon_map), coords_path)
        else:
            logger.warning("Coordinates file not found: %s — using defaults", coords_path)

    output = args.output or str(config.paths.external / config.weather_collector.output_file)
    logger.info("Collecting weather for %d matches...", len(matches_df))

    df = collect_weather(
        matches_df=matches_df,
        lat_lon_map=lat_lon_map,
        api_key=args.api_key,
        use_cache=not args.no_cache,
        output_path=output,
    )

    logger.info("Done — %d rows collected", len(df))
    logger.info("Output: %s", output)


if __name__ == "__main__":
    main()
