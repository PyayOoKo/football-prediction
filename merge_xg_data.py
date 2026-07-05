"""
merge_xg_data.py — Merge StatsBomb xG data into the combined World Cup dataset
and run the full training + prediction pipeline with goal forecasts.

Usage:
    python merge_xg_data.py                           # Full run
    python merge_xg_data.py --skip-xg                 # Skip xG merge, just add goal preds
    python merge_xg_data.py --dry-run                 # Preview only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_xg_data")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
COMBINED_CSV = DATA_DIR / "worldcup_all.csv"
XG_CSV = DATA_DIR / "worldcup_xg.csv"


def merge_xg(dry_run: bool = False) -> pd.DataFrame | None:
    """Merge StatsBomb xG data into the combined World Cup dataset.

    Adds home_xg and away_xg columns to the combined CSV.
    2018 and 2022 matches get real xG values.
    2026 matches (no xG data yet) get NaN (will be zero-filled by feature engineering).
    """
    logger.info("Loading combined World Cup data …")
    df = pd.read_csv(COMBINED_CSV, low_memory=False)
    logger.info("  %d rows loaded", len(df))

    if not XG_CSV.exists():
        logger.error("xG data not found. Run collect_worldcup_xg.py first.")
        return None

    xg_df = pd.read_csv(XG_CSV)
    logger.info("Loading xG data: %d matches", len(xg_df))

    # Normalise dates to date-only for matching
    df["date"] = pd.to_datetime(df["date"]).dt.date
    xg_df["match_date"] = pd.to_datetime(xg_df["match_date"]).dt.date

    # Merge: match on season + home_team + away_team + date
    merged = df.merge(
        xg_df[["season", "match_date", "home_team", "away_team", "home_xg", "away_xg"]],
        how="left",
        left_on=["season", "date", "home_team", "away_team"],
        right_on=["season", "match_date", "home_team", "away_team"],
    )

    # Drop the duplicate date column
    merged.drop(columns=["match_date"], inplace=True)

    xg_found = merged["home_xg"].notna().sum()
    logger.info("  xG matched for %d / %d completed matches", xg_found, merged["result"].notna().sum())

    # For 2026 matches and unmatched matches, xG stays NaN
    # The xg_features module will zero-fill these automatically

    if not dry_run:
        merged.to_csv(COMBINED_CSV, index=False)
        logger.info("  Saved to %s", COMBINED_CSV)
    else:
        logger.info("  [DRY RUN] Not saved")

    return merged


def run_training(dry_run: bool = False) -> bool:
    """Run the training script (which now includes xG features and goal preds)."""
    train_script = PROJECT_ROOT / "train_worldcup.py"
    python_exe = _get_python_exe()

    cmd = [python_exe, "-u", str(train_script)]
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        for line in result.stdout.splitlines():
            logger.info("  %s", line)
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning("  [stderr] %s", line)

        if result.returncode != 0:
            logger.error("Training failed with exit code %d", result.returncode)
            return False

        logger.info("Training completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Training timed out")
        return False
    except Exception as e:
        logger.error("Training failed: %s", e)
        return False


def _get_python_exe() -> str:
    candidates: list[str] = [
        r"C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe",
        "python",
        "python3",
    ]
    import os, sys as _sys
    if _sys.executable and os.path.exists(_sys.executable):
        candidates.insert(0, _sys.executable)
    for exe in candidates:
        try:
            r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return exe
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "python"


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge xG data and run predictions")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-xg", action="store_true", help="Skip xG merge, just add goal preds")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("  xG MERGE + GOAL PREDICTIONS")
    print("=" * 60)

    # Step 1: Merge xG (unless skipped)
    if not args.skip_xg:
        df = merge_xg(dry_run=args.dry_run)
        if df is None:
            return 1
    else:
        logger.info("Skipping xG merge")

    # Step 2: Run training (which now outputs Poisson goal predictions)
    if not args.dry_run:
        success = run_training(dry_run=False)
        if not success:
            return 1

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
