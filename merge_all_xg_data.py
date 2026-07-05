"""
merge_all_xg_data.py — Merge StatsBomb xG + xAG data into the combined
World Cup dataset and run the full training + prediction pipeline.

Usage:
    python merge_all_xg_data.py                         # Full run
    python merge_all_xg_data.py --skip-train            # Merge only, skip training
    python merge_all_xg_data.py --dry-run               # Preview only
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
logger = logging.getLogger("merge_all_xg_data")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
COMBINED_CSV = DATA_DIR / "worldcup_all.csv"
XG_CSV = DATA_DIR / "worldcup_xg.csv"
XAG_CSV = DATA_DIR / "worldcup_xag.csv"


def merge_xg_data(
    df: pd.DataFrame,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Merge StatsBomb xG data into the combined World Cup dataset.

    Adds home_xg and away_xg columns. 2018/2022 matches get real xG values.
    2026 matches get NaN (zero-filled automatically by xg_features module).
    """
    if not XG_CSV.exists():
        logger.error("xG data not found at %s", XG_CSV)
        return df

    xg_df = pd.read_csv(XG_CSV)
    logger.info("  Loading xG data: %d matches", len(xg_df))

    # Normalise dates to date-only for matching
    df["date"] = pd.to_datetime(df["date"]).dt.date
    xg_df["match_date"] = pd.to_datetime(xg_df["match_date"]).dt.date

    # Merge on season + home_team + away_team + date
    merged = df.merge(
        xg_df[["season", "match_date", "home_team", "away_team", "home_xg", "away_xg"]],
        how="left",
        left_on=["season", "date", "home_team", "away_team"],
        right_on=["season", "match_date", "home_team", "away_team"],
    )
    merged.drop(columns=["match_date"], inplace=True)

    xg_found = merged["home_xg"].notna().sum()
    completed = merged["result"].notna().sum()
    logger.info(
        "  xG matched for %d / %d completed matches (%.1f%%)",
        xg_found, completed, (xg_found / completed * 100) if completed else 0,
    )

    # Show xG覆盖率 by tournament
    for season in sorted(merged["season"].unique()):
        sub = merged[merged["season"] == season]
        sub_comp = sub["result"].notna().sum()
        sub_xg = sub["home_xg"].notna().sum()
        pct = sub_xg / sub_comp * 100 if sub_comp else 0
        logger.info("    %d World Cup: %d/%d with xG (%.0f%%)", season, sub_xg, sub_comp, pct)

    return merged


def merge_xag_data(
    df: pd.DataFrame,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Merge StatsBomb xAG data into the combined World Cup dataset.

    Adds home_xag and away_xag columns. Only World Cup matches are matched
    (xAG data includes Euro, Copa America, AFCON — filtered out by merge key).
    """
    if not XAG_CSV.exists():
        logger.error("xAG data not found at %s", XAG_CSV)
        return df

    xag_df = pd.read_csv(XAG_CSV)
    logger.info("  Loading xAG data: %d matches", len(xag_df))

    # Filter xAG to only World Cup entries for cleaner merge
    wc_xag = xag_df[xag_df["competition"].str.contains("World Cup", na=False)].copy()
    logger.info("  World Cup xAG entries: %d", len(wc_xag))

    wc_xag["match_date"] = pd.to_datetime(wc_xag["match_date"]).dt.date

    merged = df.merge(
        wc_xag[["season", "match_date", "home_team", "away_team", "home_xag", "away_xag",
                 "home_key_passes", "away_key_passes"]],
        how="left",
        left_on=["season", "date", "home_team", "away_team"],
        right_on=["season", "match_date", "home_team", "away_team"],
    )
    merged.drop(columns=["match_date"], inplace=True)

    xag_found = merged["home_xag"].notna().sum()
    completed = merged["result"].notna().sum()
    logger.info(
        "  xAG matched for %d / %d completed matches (%.1f%%)",
        xag_found, completed, (xag_found / completed * 100) if completed else 0,
    )

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge xG + xAG data and run training pipeline",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-train", action="store_true", help="Merge only, skip training")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  xG + xAG DATA INTEGRATION PIPELINE")
    print("=" * 60)

    # ── 1. Load the combined dataset ──
    print("\n  Loading combined World Cup data ...")
    if not COMBINED_CSV.exists():
        logger.error("Combined dataset not found at %s. Run refresh_worldcup.py first.", COMBINED_CSV)
        return 1

    df = pd.read_csv(COMBINED_CSV, low_memory=False)
    print(f"  [OK] {len(df)} rows loaded")

    # ── 2. Merge xG data ──
    print("\n  Merging xG data ...")
    df = merge_xg_data(df, dry_run=args.dry_run)

    # ── 3. Merge xAG data ──
    print("\n  Merging xAG data ...")
    df = merge_xag_data(df, dry_run=args.dry_run)

    # ── 4. Save enriched dataset ──
    print("\n  Saving enriched dataset ...")
    # Convert date back to string for CSV
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Make backup of original
    if not args.dry_run:
        backup_path = COMBINED_CSV.with_suffix(".bak.csv")
        if not backup_path.exists():
            import shutil
            shutil.copy2(COMBINED_CSV, backup_path)
            print(f"  Backup saved: {backup_path}")

        df.to_csv(COMBINED_CSV, index=False)
        print(f"  [OK] Enriched dataset saved to {COMBINED_CSV}")
    else:
        print("  [DRY RUN] Not saved to disk")

    # ── Summary stats ──
    print("\n" + "=" * 60)
    print("  INTEGRATION SUMMARY")
    print("=" * 60)
    has_xg = "home_xg" in df.columns
    has_xag = "home_xag" in df.columns
    xg_count = df["home_xg"].notna().sum() if has_xg else 0
    xag_count = df["home_xag"].notna().sum() if has_xag else 0
    total_completed = df["result"].notna().sum()
    print(f"  Total matches:          {len(df)}")
    print(f"  Completed matches:      {total_completed}")
    print(f"  With xG data:           {xg_count} ({xg_count/total_completed*100:.0f}%)")
    print(f"  With xAG data:          {xag_count} ({xag_count/total_completed*100:.0f}%)")
    print(f"  New columns added:      home_xg, away_xg, home_xag, away_xag, home_key_passes, away_key_passes")
    elapsed = time.time() - t0
    print(f"  Duration:               {elapsed:.1f}s")

    # ── 5. Retrain & predict ──
    if not args.skip_train and not args.dry_run:
        print("\n" + "=" * 60)
        print("  RETRAINING MODEL WITH xG/xAG DATA")
        print("=" * 60)

        train_script = PROJECT_ROOT / "train_worldcup.py"
        python_exe = _get_python_exe()

        cmd = [python_exe, "-u", str(train_script)]
        print(f"  Running: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=str(PROJECT_ROOT),
            )

            # Print stdout (filter to key lines)
            for line in result.stdout.splitlines():
                line_stripped = line.strip()
                if line_stripped:
                    print(f"  {line_stripped}")

            if result.stderr:
                for line in result.stderr.splitlines():
                    if line.strip():
                        print(f"  [stderr] {line.strip()}")

            if result.returncode != 0:
                logger.error("Training failed with exit code %d", result.returncode)
                return 1

            logger.info("Training & prediction completed successfully")
        except subprocess.TimeoutExpired:
            logger.error("Training timed out (10 min)")
            return 1
        except Exception as e:
            logger.error("Training failed: %s", e)
            return 1

    elapsed_total = time.time() - t0
    print(f"\n  Pipeline complete in {elapsed_total:.1f}s")
    return 0


def _get_python_exe() -> str:
    candidates: list[str] = [
        r"C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe",
        "python", "python3",
    ]
    import os
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


if __name__ == "__main__":
    sys.exit(main())
