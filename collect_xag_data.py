"""
collect_xag_data.py — Extract xAG (expected assisted goals) from StatsBomb.

xAG measures the quality of chances created by a team. For each key pass
(pass_shot_assist = True), we take the xG of the resulting shot and sum it
as the "expected assisted goals" for the assisting team.

This creates team-level xAG values for each match, parallel to xG.

Usage:
    python collect_xag_data.py                    # Extract & save
    python collect_xag_data.py --skip-wc           # Skip World Cup, just international
    python collect_xag_data.py --summary           # Print stats
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_xag")

# StatsBomb competition IDs to process
# Format: (competition_id, season_id, label, year)
COMPETITIONS = [
    # World Cups
    (43, 3, "FIFA World Cup 2018", 2018),
    (43, 106, "FIFA World Cup 2022", 2022),
    # International tournaments (already collected for xG)
    (55, 43, "UEFA Euro 2020", 2021),
    (55, 282, "UEFA Euro 2024", 2024),
    (223, 282, "Copa America 2024", 2024),
    (1267, 107, "African Cup of Nations 2023", 2023),
]

OUTPUT_PATH = Path("data/raw/worldcup_xag.csv")

TEAM_NAME_MAP: dict[str, str] = {
    "United States": "USA",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "DR Congo",
    "Czech Republic": "Czech Republic",
}


def extract_xag_data(dry_run: bool = False) -> pd.DataFrame | None:
    """Extract match-level xAG from StatsBomb.

    For each match, computes total xAG for home and away teams by:
    1. Finding all passes with pass_shot_assist=True
    2. For each such pass, finding the resulting shot and its xG
    3. Summing the xG of those shots per team
    """
    from statsbombpy import sb

    all_matches: list[dict[str, Any]] = []

    for comp_id, season_id, label, year in COMPETITIONS:
        logger.info("Processing %s (%d/%d) …", label, comp_id, season_id)
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as e:
            logger.warning("  SKIPPED - %s", e)
            continue

        logger.info("  %d matches found", len(matches))

        for _, match in matches.iterrows():
            match_id = match["match_id"]
            home_team = _normalise_name(match["home_team"])
            away_team = _normalise_name(match["away_team"])

            try:
                events = sb.events(match_id=match_id)
            except Exception:
                continue

            # For each shot, find the key pass that preceded it via shot_key_pass_id
            shots = events[events["type"] == "Shot"].copy()
            home_xag = 0.0
            away_xag = 0.0
            home_key_passes = 0
            away_key_passes = 0

            for _, shot in shots.iterrows():
                shot_team = shot.get("team", "")
                xg = float(shot.get("shot_statsbomb_xg", 0) or 0)
                key_pass_id = shot.get("shot_key_pass_id")

                if pd.notna(key_pass_id) and str(key_pass_id) != "":
                    # This shot came from a key pass
                    if shot_team == match["home_team"]:
                        home_xag += xg
                        home_key_passes += 1
                    elif shot_team == match["away_team"]:
                        away_xag += xg
                        away_key_passes += 1

            all_matches.append({
                "season": year,
                "match_date": match["match_date"],
                "home_team": home_team,
                "away_team": away_team,
                "home_xag": round(home_xag, 3),
                "away_xag": round(away_xag, 3),
                "home_key_passes": home_key_passes,
                "away_key_passes": away_key_passes,
                "home_score": int(match["home_score"]),
                "away_score": int(match["away_score"]),
                "competition": label,
                "source": "StatsBomb",
            })

    if not all_matches:
        logger.error("No xAG data extracted. Aborting.")
        return None

    df = pd.DataFrame(all_matches)
    df.sort_values(["season", "match_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("")
    logger.info("=== xAG DATA SUMMARY ===")
    logger.info("  Total matches with xAG: %d", len(df))
    for comp in sorted(df["competition"].unique()):
        subset = df[df["competition"] == comp]
        logger.info("  %s: %d matches, avg xAG: home=%.3f away=%.3f",
                    comp, len(subset),
                    subset["home_xag"].mean(), subset["away_xag"].mean())

    if not dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info("  Saved to %s", OUTPUT_PATH)

    return df


def _normalise_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract xAG data from StatsBomb",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--summary", action="store_true", help="Print stats")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  COLLECTING xAG DATA FROM STATSBOMB")
    print("=" * 60)

    df = extract_xag_data(dry_run=args.dry_run)
    if df is None:
        return 1

    if args.summary and df is not None:
        print(f"\n  Top 10 team xAG performances:")
        for _, r in df.nlargest(10, "home_xag")[["match_date", "home_team", "away_team", "home_xag"]].iterrows():
            print(f"    {r['match_date']} | {r['home_team']:20s} vs {r['away_team']:20s} | xAG: {r['home_xag']:.2f}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
