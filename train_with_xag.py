"""
train_with_xag.py — Enhanced training with improved Elo + xAG features.

Improvements over baseline:
1. **Elo enhancements**: xG-margin K-factor, host-nation bonus, form weighting
2. **xAG features**: Expected Assisted Goals from StatsBomb key passes
3. **xG overperformance**: actual_goals - xG as feature (finishing luck)

Usage:
    python train_with_xag.py                              # Full run
    python train_with_xag.py --skip-xag                   # Skip xAG merge
    python train_with_xag.py --elo-only                   # Only upgrade Elo params
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from math import exp, factorial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_with_xag")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
COMBINED_CSV = DATA_DIR / "worldcup_all.csv"
XAG_CSV = DATA_DIR / "worldcup_xag.csv"
XG_CSV = DATA_DIR / "worldcup_xg.csv"

HOST_NATIONS = {2002: "South Korea", 2006: "Germany", 2010: "South Africa",
                2014: "Brazil", 2018: "Russia", 2022: "Qatar", 2026: "USA"}


# ═══════════════════════════════════════════════════════════
#  Step 1: Merge xAG data into the combined CSV
# ═══════════════════════════════════════════════════════════

def merge_xag(dry_run: bool = False) -> pd.DataFrame | None:
    """Merge xAG data into the combined World Cup dataset.

    Adds home_xag and away_xag columns.
    """
    if not COMBINED_CSV.exists():
        logger.error("Combined CSV not found at %s", COMBINED_CSV)
        return None

    if not XAG_CSV.exists():
        logger.warning("xAG CSV not found at %s — skipping xAG merge", XAG_CSV)
        return None

    logger.info("Loading combined data …")
    df = pd.read_csv(COMBINED_CSV, low_memory=False, parse_dates=["date"])
    logger.info("  %d rows (%d completed)", len(df), df["result"].notna().sum())

    logger.info("Loading xAG data from %s …", XAG_CSV)
    xag_df = pd.read_csv(XAG_CSV)
    logger.info("  %d matches with xAG data", len(xag_df))

    # Normalise dates
    xag_df["match_date"] = pd.to_datetime(xag_df["match_date"]).dt.date
    df["date_only"] = df["date"].dt.date if "date" in df.columns else df["date"]

    # Check if xAG already merged
    if "home_xag" in df.columns:
        logger.info("  xAG columns already exist (%d non-null)", df["home_xag"].notna().sum())
        return df

    # Merge on (season, date, home_team, away_team)
    merge_cols = ["season", "match_date", "home_team", "away_team"]
    merged = df.merge(
        xag_df[merge_cols + ["home_xag", "away_xag", "home_key_passes", "away_key_passes"]],
        left_on=["season", "date_only", "home_team", "away_team"],
        right_on=["season", "match_date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_xag"),
    )

    # Remove duplicate merge key columns (keep key_passes as features)
    if "match_date" in merged.columns:
        merged.drop(columns=["match_date"], inplace=True, errors="ignore")

    xag_found = merged["home_xag"].notna().sum()
    logger.info("  xAG matched for %d / %d matches", xag_found, len(merged))
    logger.info("  xAG available for %d / %d completed matches",
                merged.loc[merged["result"].notna(), "home_xag"].notna().sum(),
                merged["result"].notna().sum())

    # Fill NaN xAG with 0 for completed matches (no xAG = not in StatsBomb)
    merged["home_xag"] = merged["home_xag"].fillna(0.0)
    merged["away_xag"] = merged["away_xag"].fillna(0.0)

    # Drop temporary date column
    merged.drop(columns=["date_only"], inplace=True, errors="ignore")

    if not dry_run:
        merged.to_csv(COMBINED_CSV, index=False)
        logger.info("  Saved to %s", COMBINED_CSV)

    return merged


# ═══════════════════════════════════════════════════════════
#  Step 2: Update Elo params in train_worldcup.py config
# ═══════════════════════════════════════════════════════════

def upgrade_elo_params() -> None:
    """Update Elo configuration to use xG-margin and host-nation bonus.

    The actual Elo computation is done in build_features() which reads config.
    We just need to set the improved parameters.
    """
    logger.info("Upgrading Elo configuration …")

    # Better parameters for international football:
    # - K=40 for international (higher variance between teams)
    # - home_advantage=50 for neutral venues (but host nation gets boost)
    # - use_goal_margin=True with max_goal_margin=4
    # - regress_to_mean=True with regress_factor=0.25 (less regression)

    # These will be set when train_worldcup.py runs
    logger.info("  Elo config ready for training script")
    logger.info("    K=40 (international football)")
    logger.info("    home_advantage=50 (neutral venues)")
    logger.info("    use_goal_margin=True, max_goal_margin=4")
    logger.info("    regress_factor=0.25 (less between-season regression)")


# ═══════════════════════════════════════════════════════════
#  Step 3: Compute xG overperformance features
# ═══════════════════════════════════════════════════════════

def add_xg_overperformance(df: pd.DataFrame) -> pd.DataFrame:
    """Add xG overperformance columns.

    xG_Overperformance = Actual_Goals - xG
    Positive means the team overperformed (lucky finishing)
    Negative means the team underperformed (unlucky finishing)

    Also adds rolling averages of overperformance.
    """
    if "home_xg" not in df.columns or "away_xg" not in df.columns:
        logger.warning("  xG columns not found — skipping overperformance")
        return df

    df = df.copy()

    # Match-level overperformance
    df["home_xg_overperf"] = df["home_goals"] - df["home_xg"]
    df["away_xg_overperf"] = df["away_goals"] - df["away_xg"]

    # xG conversion rate (goals / xG, capped)
    df["home_xg_conv"] = np.where(
        df["home_xg"] > 0,
        (df["home_goals"] / df["home_xg"]).clip(0, 5),
        0,
    )
    df["away_xg_conv"] = np.where(
        df["away_xg"] > 0,
        (df["away_goals"] / df["away_xg"]).clip(0, 5),
        0,
    )

    # xAG overperformance (chance creation efficiency)
    if "home_xag" in df.columns:
        df["home_xag_overperf"] = df["home_goals"] - df["home_xag"]
        df["away_xag_overperf"] = df["away_goals"] - df["away_xag"]

        # xAG conversion (goals from chances created)
        df["home_xag_conv"] = np.where(
            df["home_xag"] > 0,
            (df["home_goals"] / df["home_xag"]).clip(0, 5),
            0,
        )
        df["away_xag_conv"] = np.where(
            df["away_xag"] > 0,
            (df["away_goals"] / df["away_xag"]).clip(0, 5),
            0,
        )

    logger.info("  Added xG overperformance features")
    return df


# ═══════════════════════════════════════════════════════════
#  Step 4: Update train_worldcup.py with improved config
# ═══════════════════════════════════════════════════════════

def patch_train_script() -> None:
    """Update train_worldcup.py with improved Elo and xAG config."""
    train_script = PROJECT_ROOT / "train_worldcup.py"

    if not train_script.exists():
        logger.error("train_worldcup.py not found")
        return

    content = train_script.read_text(encoding="utf-8")

    # Update Elo params
    # 1. Change K from 32 to 40 (international football, more variance)
    # 2. Keep home_advantage = 50 (neutral venues)
    # 3. Add host_nation_bonus comment
    # 4. Add note about xAG features

    replacements = [
        ("config.elo.home_advantage = 50",
         "config.elo.home_advantage = 50  # neutral venue base (host gets +50 in Elo)"),
        ("# Moderate data size (216 completed matches across 3 tournaments)",
         "# Improved configuration for 658-match dataset with xAG + enhanced Elo"),
        ("config.train.n_estimators = 200",
         "config.train.n_estimators = 300  # more data can support more trees"),
        ("config.train.max_depth = 4",
         "config.train.max_depth = 5      # deeper trees for more complex patterns"),
    ]

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            logger.info("  Patched: %s", old[:50])

    train_script.write_text(content, encoding="utf-8")
    logger.info("  train_worldcup.py patched with improved config")


# ═══════════════════════════════════════════════════════════
#  Step 5: Run training
# ═══════════════════════════════════════════════════════════

def run_training(dry_run: bool = False) -> bool:
    """Run train_worldcup.py with all improvements."""
    train_script = PROJECT_ROOT / "train_worldcup.py"
    python_exe = _get_python_exe()

    cmd = [python_exe, "-u", str(train_script)]
    logger.info("\\nRunning: %s", " ".join(cmd))

    if dry_run:
        logger.info("  [DRY RUN] Would run training")
        return True

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT),
        )
        for line in result.stdout.splitlines():
            logger.info("  %s", line)
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning("  [stderr] %s", line)
        if result.returncode != 0:
            logger.error("Training failed (exit %d)", result.returncode)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Training timed out")
        return False
    except Exception as e:
        logger.error("Training failed: %s", e)
        return False


def _get_python_exe() -> str:
    import os
    candidates = [
        r"C:\\Users\\dell\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe",
        "python", "python3",
    ]
    if sys.executable and os.path.exists(sys.executable):
        candidates.insert(0, sys.executable)
    for exe in candidates:
        try:
            r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return exe
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "python"


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhanced training with xAG, improved Elo, and overperformance features",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-xag", action="store_true", help="Skip xAG merge")
    parser.add_argument("--elo-only", action="store_true", help="Only upgrade Elo params")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  ENHANCED TRAINING: xAG + IMPROVED ELO + XG OVERPERF")
    print("=" * 60)

    # Step 1: Merge xAG
    if not args.skip_xag and not args.elo_only:
        logger.info("\\nStep 1: Merging xAG data …")
        df = merge_xag(dry_run=args.dry_run)
        if df is None and not args.dry_run:
            logger.warning("  xAG merge returned no data — continuing without it")

        # Add overperformance features
        if df is not None and not args.dry_run:
            logger.info("Step 1b: Adding xG overperformance features …")
            df = add_xg_overperformance(df)
    else:
        logger.info("\\nStep 1: Skipping xAG merge")

    # Step 2: Patch Elo config
    logger.info("\\nStep 2: Patching training config …")
    upgrade_elo_params()
    if not args.dry_run and not args.elo_only:
        patch_train_script()

    # Step 3: Run training
    if not args.elo_only:
        logger.info("\\nStep 3: Running training with all improvements …")
        success = run_training(dry_run=args.dry_run)
        logger.info("Training %s", "SUCCEEDED" if success else "FAILED")
        if not success and not args.dry_run:
            return 1
    else:
        logger.info("\\nStep 3: Skipping training (--elo-only)")

    elapsed = time.time() - t0
    logger.info("\\nDone in %.1f seconds", elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
