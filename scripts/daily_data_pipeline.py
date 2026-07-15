#!/usr/bin/env python3
"""
Daily Data Pipeline — Fetches new match data from all configured sources.

Steps:
1. Fetch data from football-data.co.uk (top 5 leagues)
2. Fetch data from openfootball (World Cup / international)
3. Fetch xG data from Understat
4. Fetch club/player data from Transfermarkt (incremental)
5. Clean & normalize all data
6. Store in database / CSV
7. Log success/failure + metrics

Scheduler:
    python -m src.scheduler.cli run --tasks daily_data_pipeline
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.monitoring import Monitor

logger = logging.getLogger("daily_data_pipeline")

# ── Sources configuration ─────────────────────────────
SOURCES = {
    "worldcup": {"module": "collect_all_worldcups", "func": "main"},
    "leagues": {"module": "collect_leagues", "func": "main"},
    "player_data": {"module": "collect_player_data", "func": "main"},
    "xg_data": {"module": "collect_worldcup_xg", "func": "main"},
}

# ── Data paths ────────────────────────────────────────
RAW_DIR = _project_root / "data" / "raw"
PROCESSED_DIR = _project_root / "data" / "processed"


def fetch_source(source_name: str, source_cfg: dict) -> dict:
    """Fetch data from a single source.

    Returns a dict with:
        success: bool
        rows: int (rows imported)
        duration: float (seconds)
        error: str | None
        output: str | None (output file path)
    """
    result = {"success": False, "rows": 0, "duration": 0.0, "error": None, "output": None}
    start = time.perf_counter()

    try:
        mod = __import__(source_cfg["module"], fromlist=[source_cfg["func"]])
        func = getattr(mod, source_cfg["func"])
        return_code = func() if callable(func) else None

        elapsed = time.perf_counter() - start
        result["duration"] = elapsed
        result["success"] = True

        # Count output rows if CSV was created
        if source_name == "worldcup":
            csv_path = RAW_DIR / "worldcup_all.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                result["rows"] = len(df)
                result["output"] = str(csv_path)
        elif source_name == "leagues":
            league_files = list(RAW_DIR.glob("*.csv"))
            total_rows = 0
            for lf in league_files:
                if lf.stat().st_size > 0:
                    df = pd.read_csv(lf)
                    total_rows += len(df)
            result["rows"] = total_rows
            result["output"] = str(league_files[0]) if league_files else None
        elif source_name == "player_data":
            csv_path = _project_root / "data" / "external" / "players.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                result["rows"] = len(df)
                result["output"] = str(csv_path)

        logger.info("Fetched '%s': %d rows in %.1fs", source_name, result["rows"], elapsed)

    except Exception as exc:
        elapsed = time.perf_counter() - start
        result["duration"] = elapsed
        result["error"] = str(exc)
        logger.error("Failed to fetch '%s': %s", source_name, exc)

    return result


def run_data_pipeline(monitor: Monitor | None = None) -> dict:
    """Run the full daily data pipeline across all sources.

    Returns a report dict with per-source results and overall summary.
    """
    logger.info("=" * 60)
    logger.info("STARTING DAILY DATA PIPELINE")
    logger.info("=" * 60)

    pipeline_start = time.perf_counter()
    results = {}
    total_rows = 0
    total_errors = 0

    # Step 1: Fetch World Cup data
    results["worldcup"] = fetch_source("worldcup", SOURCES["worldcup"])
    if results["worldcup"]["success"]:
        total_rows += results["worldcup"]["rows"]

    # Step 2: Fetch league data
    results["leagues"] = fetch_source("leagues", SOURCES["leagues"])
    if results["leagues"]["success"]:
        total_rows += results["leagues"]["rows"]

    # Step 3: Fetch player data
    results["player_data"] = fetch_source("player_data", SOURCES["player_data"])
    if results["player_data"]["success"]:
        total_rows += results["player_data"]["rows"]

    # Step 4: Fetch xG data
    results["xg_data"] = fetch_source("xg_data", SOURCES["xg_data"])
    if results["xg_data"]["success"]:
        total_rows += results["xg_data"]["rows"]

    # Step 5: Clean and merge raw data
    clean_result = _clean_and_merge()
    results["clean"] = clean_result

    # Count errors
    for src_name, src_result in results.items():
        if not src_result.get("success", False) and src_name != "clean":
            total_errors += 1

    pipeline_elapsed = time.perf_counter() - pipeline_start

    # Build report
    report = {
        "pipeline": "daily_data_pipeline",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(pipeline_elapsed, 2),
        "total_rows": total_rows,
        "sources_succeeded": sum(1 for r in results.values() if r.get("success")),
        "sources_failed": total_errors,
        "per_source": {k: {_k: _v for _k, _v in v.items() if _k != "func"} for k, v in results.items()},
    }

    # Record monitoring metrics
    if monitor:
        for src_name, src_result in results.items():
            monitor.record_etl(
                pipeline=f"daily_data_pipeline/{src_name}",
                duration_seconds=src_result.get("duration", 0.0),
                rows_imported=src_result.get("rows", 0),
                success=src_result.get("success", False),
                error_message=src_result.get("error", ""),
            )

    logger.info("Daily data pipeline complete: %.1fs, %d rows, %d errors",
                pipeline_elapsed, total_rows, total_errors)
    return report


def _clean_and_merge() -> dict:
    """Clean and merge raw data files into processed output."""
    result = {"success": False, "rows": 0, "duration": 0.0, "error": None}
    start = time.perf_counter()

    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Load all CSV files from raw directory
        all_files = list(RAW_DIR.glob("*.csv"))
        if not all_files:
            logger.warning("No raw CSV files found to clean")
            result["success"] = True
            result["duration"] = time.perf_counter() - start
            return result

        dfs = []
        for f in all_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df["source_file"] = f.name
                dfs.append(df)
            except Exception as exc:
                logger.warning("Could not read %s: %s", f.name, exc)

        if not dfs:
            logger.warning("No data frames loaded from raw files")
            result["success"] = True
            result["duration"] = time.perf_counter() - start
            return result

        combined = pd.concat(dfs, ignore_index=True)

        # Normalize columns
        combined.columns = [c.strip().lower().replace(" ", "_") for c in combined.columns]

        # Remove full duplicates
        before = len(combined)
        dedup_cols = [c for c in ["date", "home_team", "away_team"] if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        after = len(combined)

        # Save processed
        output_path = PROCESSED_DIR / "results_clean.csv"
        combined.to_csv(output_path, index=False)

        elapsed = time.perf_counter() - start
        result["success"] = True
        result["rows"] = after
        result["duration"] = elapsed
        result["output"] = str(output_path)

        logger.info("Cleaned data: %d -> %d rows (removed %d duplicates), saved to %s",
                    before, after, before - after, output_path)

    except Exception as exc:
        elapsed = time.perf_counter() - start
        result["error"] = str(exc)
        result["duration"] = elapsed
        logger.error("Clean/merge failed: %s", exc)

    return result


def main() -> int:
    """CLI entry point for the daily data pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Daily data pipeline")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--sources", default="all", help="Comma-separated sources to fetch")
    parser.add_argument("--skip-clean", action="store_true", help="Skip cleaning step")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    monitor = Monitor()
    report = run_data_pipeline(monitor)

    if not args.quiet:
        print(f"\n{'=' * 50}")
        print(f"  DAILY DATA PIPELINE REPORT")
        print(f"{'=' * 50}")
        print(f"  Duration:  {report['duration_seconds']:.1f}s")
        print(f"  Total rows: {report['total_rows']}")
        print(f"  Sources:    {report['sources_succeeded']} succeeded, {report['sources_failed']} failed")
        for name, src in report["per_source"].items():
            status = "✅" if src.get("success") else "❌"
            rows = src.get("rows", 0)
            dur = src.get("duration", 0.0)
            print(f"    {status} {name}: {rows} rows in {dur:.1f}s")
        print(f"{'=' * 50}\n")

    return 0 if report["sources_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
