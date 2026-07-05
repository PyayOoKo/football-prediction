"""
collect_worldcup_xg.py — Extract xG data from StatsBomb for 2018/2022 World Cups.

StatsBomb provides free event-level data including per-shot xG for the
2018 and 2022 FIFA World Cups.  This script aggregates shot-level xG
into match-level home_xg / away_xg values and saves them as a CSV that
can be merged into the training dataset.

Data source: https://github.com/statsbomb/open-data
Licence: MIT (attribution required — cite StatsBomb)

Usage:
    python collect_worldcup_xg.py                   # Extract & save
    python collect_worldcup_xg.py --dry-run          # Preview only
    python collect_worldcup_xg.py --summary          # Print detailed stats
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
logger = logging.getLogger("collect_worldcup_xg")

# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

# StatsBomb competition IDs
WC_COMP_ID = 43
SEASONS = {2018: 3, 2022: 106}

OUTPUT_PATH = Path("data/raw/worldcup_xg.csv")

# ── Team name mapping: StatsBomb → openfootball ─────────
# Names that differ between the two sources
TEAM_NAME_MAP: dict[str, str] = {
    "United States": "USA",
    "South Korea": "South Korea",  # Same in both
    "Russia": "Russia",            # Same in both
    "Iran": "Iran",                # Same in both
    "Korea Republic": "South Korea",
    "Ivory Coast": "Côte d'Ivoire",
}

# ═══════════════════════════════════════════════════════════
#  Main extraction
# ═══════════════════════════════════════════════════════════


def extract_xg_data(dry_run: bool = False) -> pd.DataFrame | None:
    """Extract match-level xG for 2018 and 2022 World Cups from StatsBomb.

    Returns a DataFrame with columns:
        season, match_date, home_team, away_team, home_xg, away_xg,
        home_score, away_score, source
    """
    from statsbombpy import sb

    all_matches: list[dict[str, Any]] = []

    for year, season_id in SEASONS.items():
        logger.info("Fetching %d World Cup matches from StatsBomb …", year)
        matches = sb.matches(competition_id=WC_COMP_ID, season_id=season_id)
        logger.info("  %d matches found", len(matches))

        for _, match in matches.iterrows():
            match_id = match["match_id"]
            home_team = _normalise_name(match["home_team"])
            away_team = _normalise_name(match["away_team"])

            # Get events for this match to compute xG
            try:
                events = sb.events(match_id=match_id)
            except Exception as e:
                logger.warning("  Skipping match %d: events error - %s", match_id, e)
                continue

            # Filter shot events and sum xG by team
            shots = events[events["type"] == "Shot"].copy()
            home_xg = shots.loc[shots["team"] == match["home_team"], "shot_statsbomb_xg"].sum()
            away_xg = shots.loc[shots["team"] == match["away_team"], "shot_statsbomb_xg"].sum()

            all_matches.append({
                "season": year,
                "match_date": match["match_date"],
                "home_team": home_team,
                "away_team": away_team,
                "home_xg": round(float(home_xg), 3),
                "away_xg": round(float(away_xg), 3),
                "home_score": int(match["home_score"]),
                "away_score": int(match["away_score"]),
                "source": "StatsBomb",
            })

    if not all_matches:
        logger.error("No xG data extracted. Aborting.")
        return None

    df = pd.DataFrame(all_matches)
    df.sort_values(["season", "match_date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("")
    logger.info("=== xG DATA SUMMARY ===")
    logger.info("  Total matches with xG: %d", len(df))
    logger.info("  2018: %d matches", len(df[df["season"] == 2018]))
    logger.info("  2022: %d matches", len(df[df["season"] == 2022]))
    logger.info("  Avg home_xG: %.3f", df["home_xg"].mean())
    logger.info("  Avg away_xG: %.3f", df["away_xg"].mean())
    logger.info("  Avg match xG total: %.3f", (df["home_xg"] + df["away_xg"]).mean())

    if not dry_run:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info("  Saved to %s", OUTPUT_PATH)
    else:
        logger.info("  [DRY RUN] Not saved to disk")

    return df


def _normalise_name(name: str) -> str:
    """Map a StatsBomb team name to the openfootball convention."""
    return TEAM_NAME_MAP.get(name, name)


def _print_summary(df: pd.DataFrame) -> None:
    """Print a detailed summary of the extracted xG data."""
    print(f"\n{'─' * 60}")
    print("  DETAILED xG SUMMARY")
    print(f"{'─' * 60}")

    for year in sorted(df["season"].unique()):
        subset = df[df["season"] == year]
        print(f"\n  {year} World Cup:")
        print(f"    Matches: {len(subset)}")
        print(f"    Avg home_xG: {subset['home_xg'].mean():.3f}")
        print(f"    Avg away_xG: {subset['away_xg'].mean():.3f}")

        # Top 5 highest xG matches
        top5 = subset.nlargest(5, "home_xg")[["match_date", "home_team", "away_team", "home_xg", "away_xg"]]
        print(f"    Highest home_xG matches:")
        for _, r in top5.iterrows():
            print(f"      {r['match_date']} | {r['home_team']:20s} vs {r['away_team']:20s} | xG: {r['home_xg']:.2f}-{r['away_xg']:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract World Cup xG data from StatsBomb",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--summary", action="store_true", help="Print detailed stats")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  COLLECTING WORLD CUP xG DATA FROM STATSBOMB")
    print("=" * 60)

    df = extract_xg_data(dry_run=args.dry_run)
    if df is None:
        return 1

    if args.summary:
        _print_summary(df)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
