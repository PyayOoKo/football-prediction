"""
collect_leagues.py — Download Top 5 European League data for prediction.

Downloads 10 recent seasons from football-data.co.uk for each of the
Top 5 European leagues, saves to data/raw/league_all.csv, and optionally
runs preprocessing and training.

Usage:
    python collect_leagues.py                         # Download only
    python collect_leagues.py --train                  # Download + train
    python collect_leagues.py --skip-download          # Use existing data
    python collect_leagues.py --seasons 5              # Just 5 seasons per league
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.data_collection.sources.football_data_co_uk import (
    download_season,
    _download_current,
    _generate_season_codes as gen_seasons,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_leagues")


TOP5_LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

PROJECT_ROOT = Path(__file__).resolve().parent


def download_league(code: str, seasons: list[str]) -> pd.DataFrame:
    """Download all seasons for a single league code."""
    parts: list[pd.DataFrame] = []

    for season in seasons:
        try:
            df = download_season(season, code)
            parts.append(df)
            logger.info("  %s %s: %d rows", code, season, len(df))
        except Exception as exc:
            logger.warning("  %s %s: skipped (%s)", code, season, exc)

    try:
        df = _download_current(code)
        if len(df) > 0:
            parts.append(df)
            logger.info("  %s current: %d rows", code, len(df))
    except Exception as exc:
        logger.warning("  %s current: skipped (%s)", code, exc)

    if parts:
        combined = pd.concat(parts, ignore_index=True)
        logger.info("  → %s: %d total rows", code, len(combined))
        return combined
    return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Top 5 European League data",
    )
    parser.add_argument("--train", action="store_true", help="Also run training")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing")
    parser.add_argument("--seasons", type=int, default=10, help="Seasons per league (default 10)")
    args = parser.parse_args()

    t0 = time.time()
    output_path = PROJECT_ROOT / "data" / "raw" / "league_all.csv"

    # ── Download ─────────────────────────────────────────
    if not args.skip_download:
        print()
        print("=" * 60)
        print("  TOP 5 LEAGUE DATA COLLECTION")
        print("=" * 60)

        seasons = gen_seasons(args.seasons)
        print(f"\n  Seasons: {seasons[0]} to {seasons[-1]} ({len(seasons)} seasons)\n")

        all_dfs: list[pd.DataFrame] = []
        for code, name in TOP5_LEAGUES.items():
            print(f"  {name} ({code}) ...")
            df = download_league(code, seasons)
            if len(df) > 0:
                all_dfs.append(df)

        if not all_dfs:
            print("\n  ✗ No data downloaded. Check network connectivity.")
            return 1

        combined = pd.concat(all_dfs, ignore_index=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

        print(f"\n  {'=' * 60}")
        print(f"  DOWNLOAD COMPLETE")
        print(f"  {'=' * 60}")
        print(f"    Total rows:  {len(combined):,}")
        print(f"    Leagues:")
        league_counts = combined["league"].value_counts()
        for code, name in TOP5_LEAGUES.items():
            count = league_counts.get(code, 0)
            print(f"      {name:<20} {count:>6,} rows")
        print(f"    Date range:  {combined['date'].min():%Y-%m-%d} to {combined['date'].max():%Y-%m-%d}")
        print(f"    Saved to:    {output_path}")
        print(f"    Duration:    {time.time() - t0:.1f}s")
        print()
    else:
        print(f"\n  Skipping download. Using existing: {output_path}")

    # ── Preprocess & Train ───────────────────────────────
    if args.train:
        print(f"\n  {'=' * 60}")
        print(f"  TRAINING ON LEAGUE DATA")
        print(f"  {'=' * 60}")

        from config import config

        config.data_collection.leagues = tuple(TOP5_LEAGUES.keys())
        config.data_collection.output_file = "league_all.csv"
        config.preprocessing.output_file = "results_clean.csv"  # matches train_xgboost.py expectation
        config.features.include_h2h = True
        config.features.include_league_position = True
        config.elo.home_advantage = 100
        config.elo.k = 32

        print("\n  Preprocessing ...")
        from src.preprocessing import run_preprocessing

        pp_report = run_preprocessing(input_path=output_path)
        print(f"    {pp_report['total_rows']:,} rows, {pp_report['total_columns']} cols")

        print("\n  Training XGBoost ...")
        from train_xgboost import main as train_main

        train_main()

    print(f"\n  Done. Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
