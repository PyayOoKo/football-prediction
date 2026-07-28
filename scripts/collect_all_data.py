"""
collect_all_data.py — Master orchestrator for collecting all data needed for O/U & BTTS models.

Runs all collection scripts in dependency order:
1. collect_matches.py        — Base match results from DB + SofaScore
2. collect_odds.py           — Backfill O/U & BTTS odds from CSV archives
3. collect_xg_data.py        — Understat + DC-estimated xG
4. collect_team_stats.py     — Rolling team statistics

Usage:
    python scripts/collect_all_data.py                          # Full run
    python scripts/collect_all_data.py --skip-odds              # Skip odds backfill
    python scripts/collect_all_data.py --skip-matches           # Skip match export
    python scripts/collect_all_data.py --quick                  # Minimal: matches + team stats only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_all_data")

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PYTHON = sys.executable


def run_script(name: str, args: list[str] | None = None) -> dict:
    """Run a collection script and return timing + status."""
    script_path = SCRIPTS_DIR / name
    if not script_path.exists():
        return {"script": name, "status": "SKIPPED", "reason": "File not found"}

    cmd = [PYTHON, str(script_path)]
    if args:
        cmd.extend(args)

    logger.info("Running %s ...", name)
    start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    elapsed = time.time() - start
    success = result.returncode == 0

    # Log the last few output lines for context
    output_lines = result.stdout.strip().split("\n")[-10:]
    if output_lines:
        for line in output_lines:
            if line.strip():
                logger.info("  %s", line.strip())

    if result.stderr:
        for line in result.stderr.strip().split("\n")[-5:]:
            if line.strip():
                logger.warning("  [stderr] %s", line.strip())

    return {
        "script": name,
        "status": "✅ OK" if success else "❌ FAILED",
        "duration": round(elapsed, 1),
        "returncode": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description="Master data collection orchestrator")
    parser.add_argument("--skip-matches", action="store_true", help="Skip match export")
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds backfill")
    parser.add_argument("--skip-xg", action="store_true", help="Skip xG collection")
    parser.add_argument("--skip-team-stats", action="store_true", help="Skip team stats")
    parser.add_argument("--quick", action="store_true",
                        help="Minimal: only matches + team stats (skip odds + xG)")
    args = parser.parse_args()

    results: list[dict] = []
    total_start = time.time()

    print()
    print("=" * 65)
    print("  MASTER DATA COLLECTION — O/U & BTTS Models")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── 1. Match results ─────────────────────────────
    if not args.skip_matches:
        print("\n" + "-" * 65)
        print("  STEP 1/4: Match Results")
        print("-" * 65)
        results.append(run_script("collect_matches.py"))
    else:
        print("\n  ⏭️  STEP 1/4: Match Results — SKIPPED")

    # ── 2. Odds backfill ─────────────────────────────
    if not args.skip_odds and not args.quick:
        print("\n" + "-" * 65)
        print("  STEP 2/4: Odds (O/U & BTTS)")
        print("-" * 65)
        results.append(run_script("collect_odds.py"))
    else:
        print("\n  ⏭️  STEP 2/4: Odds — SKIPPED")

    # ── 3. xG data ───────────────────────────────────
    if not args.skip_xg and not args.quick:
        print("\n" + "-" * 65)
        print("  STEP 3/4: xG Data")
        print("-" * 65)
        results.append(run_script("collect_xg_data.py", ["--export-only"]))
    else:
        print("\n  ⏭️  STEP 3/4: xG Data — SKIPPED")

    # ── 4. Team stats ────────────────────────────────
    if not args.skip_team_stats:
        print("\n" + "-" * 65)
        print("  STEP 4/4: Team Statistics")
        print("-" * 65)
        results.append(run_script("collect_team_stats.py"))
    else:
        print("\n  ⏭️  STEP 4/4: Team Stats — SKIPPED")

    # ── Summary ──────────────────────────────────────
    total_elapsed = time.time() - total_start
    print()
    print("=" * 65)
    print("  COLLECTION SUMMARY")
    print("=" * 65)
    for r in results:
        print(f"  {r['status']}  {r['script']:35s} ({r['duration']:.1f}s)")
    print(f"\n  Total time: {total_elapsed:.1f}s")
    print()

    # Check output files exist
    data_dir = PROJECT_ROOT / "data"
    expected_files = [
        "matches.csv",
        "odds.csv",
        "xg_data.csv",
        "team_stats.csv",
    ]
    print("  Output files:")
    for fname in expected_files:
        fpath = data_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            size_str = f"{size / 1024:.0f} KB" if size > 1024 else f"{size} B"
            print(f"    ✅ data/{fname:25s} ({size_str})")
        else:
            skipped = any(
                (fname.startswith("matches") and args.skip_matches)
                or (fname == "odds.csv" and (args.skip_odds or args.quick))
                or (fname == "xg_data.csv" and (args.skip_xg or args.quick))
                or (fname == "team_stats.csv" and args.skip_team_stats)
            )
            if skipped:
                print(f"    ⏭️  data/{fname:25s} (skipped)")
            else:
                print(f"    ❌ data/{fname:25s} (missing)")
    print()

    return 0 if all(r.get("returncode", 0) == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
