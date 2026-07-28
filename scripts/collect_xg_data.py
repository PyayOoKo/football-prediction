"""
collect_xg_data.py — Comprehensive xG data collection for O/U & BTTS models.

Strategy:
1. Top 5 leagues (E0, SP1, D1, I1, F1) — Understat real xG (2016-2025)
2. Secondary leagues (SE1) — SofaScore xG (already collected)
3. All other leagues — DC-estimated xG from historical goals
4. Export everything to structured CSV

Output:
    data/xg_data.csv — All matches with xG data (real or estimated)

Usage:
    python scripts/collect_xg_data.py                          # Full collection
    python scripts/collect_xg_data.py --understat-only          # Only Understat
    python scripts/collect_xg_data.py --estimate-only           # Only DC estimation
    python scripts/collect_xg_data.py --export-only             # Only export to CSV
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_xg_data")

DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
OUTPUT_DIR = PROJECT_ROOT / "data"

# Leagues that have Understat xG data
UNDERSTAT_LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]

# Leagues with SofaScore xG (already collected)
SOFASCORE_LEAGUES = ["SE1"]

# Leagues that need DC-estimated xG (no real xG source)
LEAGUES_FOR_DC_ESTIMATE = [
    "E0", "SP1", "D1", "I1", "F1",  # Top 5 — fill missing early seasons
    "SE1", "NO2", "FI2",             # Nordic second tiers
    "NOR", "SWE", "FI", "DN1",       # Nordic first tiers
    "IRL", "POL", "AUT", "SUI",      # Other leagues
]


# ═══════════════════════════════════════════════════════════
#  1. Understat collection (top 5 leagues)
# ═══════════════════════════════════════════════════════════


def run_understat_collection() -> dict[str, int]:
    """Run the existing Understat xG collection script for top 5 leagues."""
    script = PROJECT_ROOT / "scripts" / "collect_understat_xg.py"
    if not script.exists():
        logger.warning("Understat script not found at %s", script)
        return {"matches_updated": 0}

    logger.info("Running Understat xG collection for top 5 leagues...")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        logger.warning("Understat collection had issues:\n%s", result.stderr[:500])

    # Parse output for stats
    updated = 0
    for line in result.stdout.split("\n"):
        if "updated" in line.lower():
            # Try to extract a number
            import re
            nums = re.findall(r"\d+", line)
            if nums:
                updated = max(updated, int(nums[-1]))

    logger.info("Understat collection output:\n%s", result.stdout[:500])
    return {"matches_updated": updated}


# ═══════════════════════════════════════════════════════════
#  2. DC-estimated xG (all leagues)
# ═══════════════════════════════════════════════════════════


def estimate_xg_for_league(league: str) -> dict[str, Any]:  # type: ignore[misc]
    """Use Dixon-Coles to estimate xG for a league's missing matches.

    Only fills rows where home_xg IS NULL (preserving real xG).
    """
    from src.dixon_coles import DixonColesModel

    conn = sqlite3.connect(str(DB_PATH))

    # Load all matches for the league
    df = pd.read_sql_query(
        """
        SELECT match_id, date, home_team, away_team, home_goals, away_goals,
               home_xg, away_xg
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
        """,
        conn,
        params=(league,),
    )

    if df.empty:
        conn.close()
        return {"league": league, "total": 0, "estimated": 0, "had_real_xg": 0}

    # Count what already has real xG
    has_real_xg = (df["home_xg"].notna() | df["away_xg"].notna()).sum()
    needs_estimate = (df["home_xg"].isna() | df["away_xg"].isna()).sum()

    if needs_estimate == 0:
        conn.close()
        return {"league": league, "total": len(df), "estimated": 0, "had_real_xg": has_real_xg}

    logger.info("  %s: %d/%d matches need xG estimation", league, needs_estimate, len(df))

    # Fit Dixon-Coles on ALL matches (uses goals, not existing xG)
    dc = DixonColesModel(
        decay_halflife_days=1460.0,
        use_importance=True,
    )
    dc.fit(df)

    # Predict expected goals for each match
    home_xg_vals = []
    away_xg_vals = []
    for _, row in df.iterrows():
        lam, mu = dc.expected_goals(row["home_team"], row["away_team"])
        home_xg_vals.append(lam)
        away_xg_vals.append(mu)

    df["dc_home_xg"] = home_xg_vals
    df["dc_away_xg"] = away_xg_vals

    # Update only rows that DON'T have real xG
    updated = 0
    c = conn.cursor()
    for _, row in df.iterrows():
        if pd.isna(row["home_xg"]) or pd.isna(row["away_xg"]):
            c.execute(
                """
                UPDATE matches
                SET home_xg = ?, away_xg = ?
                WHERE match_id = ? AND (home_xg IS NULL OR away_xg IS NULL)
                """,
                (float(row["dc_home_xg"]), float(row["dc_away_xg"]), row["match_id"]),
            )
            updated += c.rowcount

    conn.commit()
    conn.close()

    logger.info("  %s: Estimated xG for %d matches", league, updated)
    return {
        "league": league,
        "total": len(df),
        "estimated": updated,
        "had_real_xg": has_real_xg,
    }


# ═══════════════════════════════════════════════════════════
#  3. Export to CSV
# ═══════════════════════════════════════════════════════════


def export_xg_csv() -> dict[str, Any]:
    """Export all matches with xG data to structured CSV."""
    conn = sqlite3.connect(str(DB_PATH))

    # Get matches with xG (real or estimated), alongside other useful columns
    query = """
        SELECT
            match_id, date, league, season,
            home_team, away_team,
            home_goals, away_goals, result,
            home_shots, away_shots,
            home_shots_target, away_shots_target,
            home_xg, away_xg,
            home_corners, away_corners,
            home_yellow, away_yellow, home_red, away_red
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Derive targets
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)

    # Flags: real xG vs estimated
    df["has_real_xg"] = (df["home_xg"].notna() & (df["home_xg"] > 0)).astype(int)
    df["has_shots"] = df["home_shots"].notna().astype(int)

    # xG difference (proxy for match dominance)
    df["xg_diff"] = df["home_xg"] - df["away_xg"]
    df["total_xg"] = df["home_xg"] + df["away_xg"]

    output_path = OUTPUT_DIR / "xg_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    # Stats
    stats = {
        "total_matches": len(df),
        "with_real_xg": int(df["has_real_xg"].sum()),
        "with_real_xg_pct": round(df["has_real_xg"].mean() * 100, 1),
        "with_shots": int(df["has_shots"].sum()),
        "with_shots_pct": round(df["has_shots"].mean() * 100, 1),
        "output_file": str(output_path),
    }
    return stats


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Comprehensive xG data collection")
    parser.add_argument("--understat-only", action="store_true",
                        help="Only run Understat collection")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only run DC estimation")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export current xG to CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("  XG DATA COLLECTION")
    print("=" * 60)

    # Phase 1: Understat
    if not args.estimate_only and not args.export_only:
        print("\n--- Phase 1: Understat collection (top 5 leagues) ---")
        understat_stats = run_understat_collection()
        print(f"  Understat: {understat_stats['matches_updated']} matches updated")

    # Phase 2: DC estimation
    if not args.understat_only and not args.export_only:
        print(f"\n--- Phase 2: DC-estimated xG ({len(LEAGUES_FOR_DC_ESTIMATE)} leagues) ---")
        total_estimated = 0

        for league in LEAGUES_FOR_DC_ESTIMATE:
            result = estimate_xg_for_league(league)
            total_estimated += result["estimated"]
            print(f"  [{league}] {result['estimated']:>4} estimated / "
                  f"{result['had_real_xg']:>4} real / {result['total']:>5} total")

        print(f"\n  Total: {total_estimated} matches with DC-estimated xG")

    # Phase 3: Export
    if not args.understat_only and not args.estimate_only:
        print("\n--- Phase 3: Exporting to CSV ---")
        stats = export_xg_csv()
        print(f"  Matches: {stats['total_matches']:,}")
        print(f"  With real xG: {stats['with_real_xg']:,} ({stats['with_real_xg_pct']}%)")
        print(f"  With shots data: {stats['with_shots']:,} ({stats['with_shots_pct']}%)")
        print(f"  Output: {stats['output_file']}")
    else:
        # Still export but from existing data
        print("\n--- Exporting to CSV ---")
        stats = export_xg_csv()
        print(f"  Matches: {stats['total_matches']:,}")
        print(f"  With real xG: {stats['with_real_xg']:,} ({stats['with_real_xg_pct']}%)")
        print(f"  Output: {stats['output_file']}")

    # Summary
    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  ✅ data/xg_data.csv — xG data for all leagues")
    print(f"  ├─ Top 5 leagues: Understat real xG (2016-2025)")
    print(f"  ├─ SE1: SofaScore real xG (already collected)")
    print(f"  └─ All others: DC-estimated xG from historical goals")
    print()


if __name__ == "__main__":
    main()
