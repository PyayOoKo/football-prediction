"""
Download, combine, and save historical World Cup data (2002 – 2026).

Collects match data from openfootball/worldcup.json for multiple tournaments,
converts each to the project's standard schema, and saves a single CSV
suitable for training the prediction model on.

Usage:
    python collect_all_worldcups.py              # Download & combine all
    python collect_all_worldcups.py --no-save    # Preview only
    python collect_all_worldcups.py --summary    # Print summary stats
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_all_worldcups")

# ── Sources ─────────────────────────────────────────────

TOURNAMENTS: list[dict[str, Any]] = [
    {"year": 2002, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2002/worldcup.json", "league": "WC"},
    {"year": 2006, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2006/worldcup.json", "league": "WC"},
    {"year": 2010, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2010/worldcup.json", "league": "WC"},
    {"year": 2014, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json", "league": "WC"},
    {"year": 2018, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json", "league": "WC"},
    {"year": 2022, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json", "league": "WC"},
    {"year": 2026, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json", "league": "WC"},
]

# ── Helpers ─────────────────────────────────────────────


def download_tournament(
    url: str,
    season: int,
    league: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download a single World Cup tournament from openfootball JSON.

    Parameters
    ----------
    url : str
        URL of the openfootball worldcup.json file.
    season : int
        The season/year identifier to store in the ``season`` column.
    league : str
        League code (e.g. ``"WC"``).
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Tournament matches in the project's standard schema.
    """
    logger.info("Downloading %d World Cup data from %s", season, url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    data: dict[str, Any] = resp.json()
    raw_matches: list[dict[str, Any]] = data.get("matches", [])

    if not raw_matches:
        raise ValueError(f"No matches found in {url}")

    rows: list[dict[str, Any]] = []
    for m in raw_matches:
        rows.append(_convert_match(m, season, league))

    df = pd.DataFrame(rows)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["source"] = f"openfootball/worldcup.json/{season}"
    df["downloaded_at"] = datetime.now().isoformat()

    completed = df["result"].notna().sum()
    logger.info(
        "  -> %d matches (%d completed, %d upcoming)",
        len(df), completed, len(df) - completed,
    )
    return df


def _convert_match(
    m: dict[str, Any],
    season: int,
    league: str,
) -> dict[str, Any]:
    """Convert a single openfootball match dict to the project schema."""
    team1 = m.get("team1", "")
    team2 = m.get("team2", "")
    score = m.get("score") or {}
    ft = score.get("ft") if isinstance(score, dict) else None
    ht = score.get("ht") if isinstance(score, dict) else None

    result: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    home_goals_ht: int | None = None
    away_goals_ht: int | None = None

    if isinstance(ft, (list, tuple)) and len(ft) >= 2:
        try:
            home_goals = int(ft[0])
            away_goals = int(ft[1])
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"
        except (TypeError, ValueError):
            pass

    if isinstance(ht, (list, tuple)) and len(ht) >= 2:
        try:
            home_goals_ht = int(ht[0])
            away_goals_ht = int(ht[1])
        except (TypeError, ValueError):
            pass

    return {
        "season": season,
        "date": m.get("date"),
        "league": league,
        "round": m.get("round"),
        "group": m.get("group"),
        "ground": m.get("ground"),
        "home_team": team1,
        "away_team": team2,
        "result": result,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_goals_ht": home_goals_ht,
        "away_goals_ht": away_goals_ht,
    }


# ── Main ────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and combine historical World Cup data",
    )
    parser.add_argument("--no-save", action="store_true", help="Preview only")
    parser.add_argument("--summary", action="store_true", help="Print summary stats")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  COLLECTING HISTORICAL WORLD CUP DATA")
    print("=" * 60)

    all_dfs: list[pd.DataFrame] = []
    total_matches = 0

    for t in TOURNAMENTS:
        try:
            df = download_tournament(t["url"], t["year"], t["league"])
            all_dfs.append(df)
            total_matches += len(df)
            print(f"  ✓ {t['year']}: {len(df)} matches loaded")
        except Exception as e:
            print(f"  ✗ {t['year']}: FAILED - {e}")

    if not all_dfs:
        print("\n  [X] No data downloaded. Aborting.")
        return 1

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values(["date", "home_team"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Standard output columns
    csv_cols = [
        "season", "date", "league", "round", "group", "ground",
        "home_team", "away_team",
        "result", "home_goals", "away_goals",
        "home_goals_ht", "away_goals_ht",
        "source", "downloaded_at",
    ]
    csv_cols = [c for c in csv_cols if c in combined.columns]
    combined_csv = combined[csv_cols].copy()

    # ── Summary ─────────────────────────────────────────
    completed = combined_csv["result"].notna().sum()
    upcoming = combined_csv["result"].isna().sum()

    print(f"\n{'=' * 60}")
    print(f"  COMBINED DATASET")
    print(f"  {'Total matches:':<20} {len(combined_csv)}")
    print(f"  {'Completed:':<20} {completed}")
    print(f"  {'Upcoming:':<20} {upcoming}")
    years_str = ", ".join(str(t["year"]) for t in TOURNAMENTS)
    print(f"  {'Tournaments:':<20} {len(all_dfs)} ({years_str})")
    print()

    by_year = combined_csv.groupby("season").agg(
        total=("result", "count"),
        completed=("result", lambda x: x.notna().sum()),
    )
    print(f"  {'Season':>8} {'Total':>8} {'Completed':>12}")
    print(f"  {'-' * 28}")
    for season, row in by_year.iterrows():
        print(f"  {season:>8} {row['total']:>8} {row['completed']:>12}")

    # ── Save ────────────────────────────────────────────
    output_path = Path("data/raw/worldcup_all.csv")

    if not args.no_save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_csv.to_csv(output_path, index=False)
        print(f"\n  ✓ Saved to {output_path}")
    else:
        print(f"\n  [--no-save] Not written to disk.")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")

    if args.summary:
        _print_summary(combined_csv)

    return 0


def _print_summary(df: pd.DataFrame) -> None:
    """Print a detailed summary of the combined dataset."""
    print(f"\n{'─' * 60}")
    print("  DETAILED SUMMARY")
    print(f"{'─' * 60}")

    # Teams per tournament
    for season in sorted(df["season"].unique()):
        subset = df[df["season"] == season]
        teams = set(subset["home_team"].unique()) | set(subset["away_team"].unique())
        # Filter out placeholder teams
        real_teams = {t for t in teams if not (t.startswith(("W", "R", "Q", "P")) and t[1:].isdigit())}
        print(f"\n  {season}: {len(real_teams)} teams, {len(subset)} matches")

    # Rounds distribution
    print(f"\n  Rounds across all seasons:")
    for rnd, cnt in df["round"].value_counts().head(15).items():
        print(f"    {rnd:<25} {cnt}")

    # Sample completed matches
    print(f"\n  Sample completed matches:")
    sample = df[df["result"].notna()].head(5)
    for _, r in sample.iterrows():
        print(f"    {r['date']} | {r['home_team']:20s} {int(r['home_goals'])}-{int(r['away_goals'])} {r['away_team']:20s}")


if __name__ == "__main__":
    sys.exit(main())
