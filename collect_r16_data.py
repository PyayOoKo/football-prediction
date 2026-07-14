"""
collect_r16_data.py — Collect past 5 years of international match data with xG
for the 16 teams that reached the 2026 World Cup Round of 16, merge with
existing World Cup data, and re-run predictions.

Sources:
  - StatsBomb Open Data (UEFA Euro 2020/2024, Copa America 2024, AFCON 2023)
  - Existing combined World Cup data (7 tournaments, 2002-2026)

Team coverage:
  Euro 2020/2024  -> France, England, Portugal, Spain, Belgium, Switzerland, Norway
  Copa America    -> Argentina, Brazil, Colombia, Paraguay, Mexico, Canada
  AFCON 2023      -> Morocco, Egypt

Usage:
    python collect_r16_data.py                          # Full run
    python collect_r16_data.py --dry-run                # Preview only
    python collect_r16_data.py --skip-merge             # Skip merge, just train
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_r16_data")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
COMBINED_CSV = DATA_DIR / "worldcup_all.csv"
XG_CSV = DATA_DIR / "worldcup_xg.csv"

# StatsBomb international competitions with xG for R16 teams
# Format: (competition_id, season_id, label, year)
INTERNATIONAL_COMPETITIONS = [
    (55, 43, "UEFA Euro 2020", 2021),
    (55, 282, "UEFA Euro 2024", 2024),
    (223, 282, "Copa America 2024", 2024),
    (1267, 107, "African Cup of Nations 2023", 2023),
]

# Team name mapping: StatsBomb -> openfootball convention
TEAM_NAME_MAP: dict[str, str] = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "DR Congo": "DR Congo",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Czech Republic": "Czech Republic",  # Same
}


# ═══════════════════════════════════════════════════════════
#  Step 1: Download StatsBomb data
# ═══════════════════════════════════════════════════════════


def download_international_matches(dry_run: bool = False) -> pd.DataFrame:
    """Download international matches with xG for all R16-relevant competitions."""
    from statsbombpy import sb

    all_records: list[dict[str, Any]] = []

    for comp_id, season_id, label, year in INTERNATIONAL_COMPETITIONS:
        logger.info("Downloading %s (%d/%d) …", label, comp_id, season_id)
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            logger.info("  %d matches found", len(matches))
        except Exception as e:
            logger.warning("  SKIPPED - %s", e)
            continue

        for _, match in matches.iterrows():
            match_id = match["match_id"]
            home_team = _normalise_name(match["home_team"])
            away_team = _normalise_name(match["away_team"])
            match_date = match["match_date"]

            # Get events for xG
            try:
                events = sb.events(match_id=match_id)
            except Exception:
                # No events for this match (rare)
                home_xg, away_xg = 0.0, 0.0
            else:
                shots = events[events["type"] == "Shot"].copy()
                home_xg = float(
                    shots.loc[shots["team"] == match["home_team"], "shot_statsbomb_xg"].sum()
                )
                away_xg = float(
                    shots.loc[shots["team"] == match["away_team"], "shot_statsbomb_xg"].sum()
                )

            all_records.append({
                "season": year,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_xg": round(home_xg, 3),
                "away_xg": round(away_xg, 3),
                "home_score": int(match["home_score"]),
                "away_score": int(match["away_score"]),
                "competition": label,
                "source": "StatsBomb",
            })

    df = pd.DataFrame(all_records)
    df.sort_values(["season", "match_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "\n  Collected %d matches across %d competitions",
        len(df), len(INTERNATIONAL_COMPETITIONS),
    )

    if not dry_run:
        out_path = DATA_DIR / "r16_international_data.csv"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("  Saved to %s", out_path)

    return df


def _normalise_name(name: str) -> str:
    """Map StatsBomb team names to openfootball conventions."""
    return TEAM_NAME_MAP.get(name, name)


# ═══════════════════════════════════════════════════════════
#  Step 2: Merge with existing data
# ═══════════════════════════════════════════════════════════


def merge_datasets(
    intl_df: pd.DataFrame,
    dry_run: bool = False,
) -> pd.DataFrame | None:
    """Merge the new international data with the existing World Cup combined CSV."""
    if not COMBINED_CSV.exists():
        logger.error("Combined CSV not found at %s", COMBINED_CSV)
        return None

    logger.info("\nLoading existing combined data …")
    existing = pd.read_csv(COMBINED_CSV, low_memory=False)
    logger.info("  %d rows loaded (completed: %d)",
                 len(existing), existing["result"].notna().sum())

    # Check if xG is already merged
    has_xg = "home_xg" in existing.columns
    if has_xg:
        logger.info("  xG columns already present (%d non-null)", existing["home_xg"].notna().sum())

    # Convert the international data to match the existing schema
    logger.info("Converting %d international matches to schema …", len(intl_df))

    new_rows: list[dict[str, Any]] = []
    for _, r in intl_df.iterrows():
        hg = r["home_score"]
        ag = r["away_score"]
        if hg > ag:
            result = "H"
        elif hg < ag:
            result = "A"
        else:
            result = "D"

        new_rows.append({
            "season": r["season"],
            "date": r["match_date"],
            "league": "INTL",
            "round": r.get("competition", "International"),
            "group": "",
            "ground": "",
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "result": result,
            "home_goals": hg,
            "away_goals": ag,
            "home_goals_ht": None,
            "away_goals_ht": None,
            "home_xg": r["home_xg"],
            "away_xg": r["away_xg"],
            "source": r["source"],
            "downloaded_at": datetime.now().isoformat(),
        })

    new_df = pd.DataFrame(new_rows)

    # Remove duplicates: check if any of these matches already exist in the combined data
    existing_keys = set(
        zip(existing["season"], existing["date"].astype(str),
            existing["home_team"], existing["away_team"])
    )
    new_df["_key"] = list(
        zip(new_df["season"], new_df["date"].astype(str),
            new_df["home_team"], new_df["away_team"])
    )
    dupe_mask = new_df["_key"].isin(existing_keys)
    new_deduplicated = new_df[~dupe_mask].drop(columns=["_key"])
    dupes_removed = dupe_mask.sum()

    if dupes_removed > 0:
        logger.info("  Removed %d duplicates already in existing data", dupes_removed)

    # Combine: existing + new non-duplicate rows
    all_cols = list(existing.columns)
    combined = pd.concat([existing, new_deduplicated[all_cols]], ignore_index=True)
    combined.sort_values(["date", "home_team"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    completed = combined["result"].notna().sum()
    upcoming = combined["result"].isna().sum()
    logger.info(
        "\n  Merged dataset: %d total (%d completed, %d upcoming)",
        len(combined), completed, upcoming,
    )

    # Stats by competition
    if "league" in combined.columns:
        for league in ["INTL", "WC"]:
            count = (combined["league"] == league).sum()
            logger.info("    %s: %d matches", league, count)

    if not dry_run:
        combined.to_csv(COMBINED_CSV, index=False)
        logger.info("  Saved to %s", COMBINED_CSV)
    else:
        logger.info("  [DRY RUN] Not saved")

    return combined


# ═══════════════════════════════════════════════════════════
#  Step 3: Run training
# ═══════════════════════════════════════════════════════════


def run_training(dry_run: bool = False) -> bool:
    """Run train_worldcup.py with the enriched dataset."""
    train_script = PROJECT_ROOT / "train_worldcup.py"
    python_exe = _get_python_exe()

    cmd = [python_exe, "-u", str(train_script)]
    logger.info("\nRunning: %s", " ".join(cmd))

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
        r"C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe",
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
        description="Collect R16 international data with xG and re-run predictions",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge, just train")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("  COLLECTING R16 INTERNATIONAL DATA WITH xG")
    print("=" * 60)

    # Step 1: Download StatsBomb data
    if not args.skip_merge:
        intl_df = download_international_matches(dry_run=args.dry_run)
        if len(intl_df) == 0:
            logger.error("No international data collected. Aborting.")
            return 1

        # Step 2: Merge
        combined = merge_datasets(intl_df, dry_run=args.dry_run)
        if combined is None:
            return 1
    else:
        logger.info("Skipping download/merge (--skip-merge)")

    # Step 3: Train & Predict
    if not args.dry_run:
        success = run_training(dry_run=False)
        logger.info("Training %s", "SUCCEEDED" if success else "FAILED")
        if not success:
            return 1

    elapsed = time.time() - t0
    logger.info("\nDone in %.1f seconds", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
