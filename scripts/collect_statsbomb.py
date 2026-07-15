"""
Collect StatsBomb open data for matches, events, and lineups.

Usage:
    python scripts/collect_statsbomb.py
    python scripts/collect_statsbomb.py --list-competitions
    python scripts/collect_statsbomb.py --competition "World Cup 2022"
    python scripts/collect_statsbomb.py --match 3869685 --output data/scrapers/statsbomb
"""

from __future__ import annotations

import argparse
import logging
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
    parser = argparse.ArgumentParser(description="Collect StatsBomb open data")
    parser.add_argument("--list-competitions", action="store_true",
                        help="List available competitions and exit")
    parser.add_argument("--competition", type=str, default=None,
                        help="Competition name (e.g. 'World Cup 2022')")
    parser.add_argument("--match", type=int, default=None,
                        help="Single match ID to fetch")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/scrapers/statsbomb)")
    parser.add_argument("--format", type=str, default="csv",
                        choices=["csv", "parquet"],
                        help="Output format (default: csv)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip cached responses")
    args = parser.parse_args()

    from src.data_collection.sources.statsbomb_open import (
        list_competitions,
        list_matches,
        get_match_events,
        get_match_lineups,
        matches_to_dataframe,
        events_to_dataframe,
        lineups_to_dataframe,
        shots_to_dataframe,
    )

    output_dir = Path(args.output or config.statsbomb_collector.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.list_competitions:
        comps = list_competitions(use_cache=not args.no_cache)
        print(f"\n{'Competition':<40} {'ID':<6} {'Season':<12} {'Matches':<8}")
        print(f"{'-'*66}")
        for c in comps:
            cname = c.get("competition_name", "")
            sname = c.get("season_name", "")
            comp_id = c.get("competition_id", "")
            season_id = c.get("season_id", "")
            match_count = c.get("match_updated") or c.get("match_available", "?")
            print(f"{cname} {sname:<12} {comp_id:<6} {season_id:<12} {match_count:<8}")
        return

    if args.match:
        # Fetch single match
        match_id = args.match
        logger.info("Fetching match %d...", match_id)

        events = get_match_events(match_id, use_cache=not args.no_cache)
        events_df = events_to_dataframe(events)
        if not events_df.empty:
            out_path = output_dir / f"events_{match_id}.{args.format}"
            if args.format == "csv":
                events_df.to_csv(out_path, index=False)
            else:
                events_df.to_parquet(out_path, index=False)
            logger.info("Saved %d events → %s", len(events_df), out_path)

        lineups_df = lineups_to_dataframe(match_id, use_cache=not args.no_cache)
        if not lineups_df.empty:
            out_path = output_dir / f"lineups_{match_id}.{args.format}"
            if args.format == "csv":
                lineups_df.to_csv(out_path, index=False)
            else:
                lineups_df.to_parquet(out_path, index=False)
            logger.info("Saved %d lineups → %s", len(lineups_df), out_path)

        shots_df = shots_to_dataframe([match_id], use_cache=not args.no_cache)
        if not shots_df.empty:
            out_path = output_dir / f"shots_{match_id}.{args.format}"
            if args.format == "csv":
                shots_df.to_csv(out_path, index=False)
            else:
                shots_df.to_parquet(out_path, index=False)
            logger.info("Saved %d shots → %s", len(shots_df), out_path)

        return

    if args.competition:
        # Fetch all matches for a competition
        logger.info("Fetching matches for %s...", args.competition)
        matches = list_matches(
            competition_name=args.competition,
            use_cache=not args.no_cache,
        )
        logger.info("Found %d matches", len(matches))

        matches_df = matches_to_dataframe(matches)
        if not matches_df.empty:
            out_path = output_dir / f"matches_{args.competition.replace(' ', '_')}.{args.format}"
            if args.format == "csv":
                matches_df.to_csv(out_path, index=False)
            else:
                matches_df.to_parquet(out_path, index=False)
            logger.info("Saved %d matches → %s", len(matches_df), out_path)

        # Optionally fetch all events
        match_ids = [m.match_id for m in matches]
        logger.info("Fetching events for %d matches...", len(match_ids))
        shot_dfs = []
        for mid in match_ids[:5]:  # Limit to 5 matches for demo
            events = get_match_events(mid, use_cache=not args.no_cache)
            shot_row = shots_to_dataframe([mid], use_cache=not args.no_cache)
            if not shot_row.empty:
                shot_dfs.append(shot_row)

        if shot_dfs:
            shots_combined = pd.concat(shot_dfs, ignore_index=True)
            shots_path = output_dir / f"shots_{args.competition.replace(' ', '_')}.{args.format}"
            if args.format == "csv":
                shots_combined.to_csv(shots_path, index=False)
            else:
                shots_combined.to_parquet(shots_path, index=False)
            logger.info("Saved %d shots → %s", len(shots_combined), shots_path)

        logger.info("Done — data saved to %s", output_dir)
        return

    # No arguments — print help
    parser.print_help()


if __name__ == "__main__":
    main()
