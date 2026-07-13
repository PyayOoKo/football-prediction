"""
CLI — command-line interface for the Data Quality system.

Commands
--------
::

    # Generate a complete data quality dashboard
    python -m src.data_quality.cli generate --source "2025 EPL"

    # Coverage analysis only (no dashboard)
    python -m src.data_quality.cli coverage --file data/raw/results.csv

    # Serve the latest dashboard as a simple HTTP page
    python -m src.data_quality.cli serve

    # Show latest data quality snapshot
    python -m src.data_quality.cli status
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import sys
import webbrowser
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_quality.coverage import CoverageAnalyzer
from src.data_quality.dashboard import DataQualityDashboard
from src.monitoring.store import MonitoringStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data_quality.cli",
        description="Data Quality Dashboard — monitor and report on data quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # ── generate ─────────────────────────────────────
    gen = subparsers.add_parser(
        "generate",
        help="Generate a complete data quality dashboard",
        description="Analyze dataset, build snapshot, and generate HTML/JSON/CSV.",
    )
    gen.add_argument("--source", "-s", default="dataset",
                     help="Source name for the dashboard")
    gen.add_argument("--file", "-f", default="",
                     help="Path to dataset CSV (default: data/processed/results_clean.csv)")
    gen.add_argument("--prev", default="",
                     help="Path to previous dataset version for drift detection")
    gen.add_argument("--output", "-o", default="reports/data_quality",
                     help="Output directory")
    gen.add_argument("--days", type=int, default=30,
                     help="Lookback days for trend charts")
    gen.add_argument("--open", action="store_true",
                     help="Open the dashboard in browser after generation")

    # ── coverage ─────────────────────────────────────
    cov = subparsers.add_parser(
        "coverage",
        help="Analyze data coverage only",
        description="Compute odds/xG/league/season coverage percentages.",
    )
    cov.add_argument("--file", "-f", required=True,
                     help="Path to dataset CSV or Parquet")

    # ── serve ────────────────────────────────────────
    serve = subparsers.add_parser(
        "serve",
        help="Serve the latest dashboard as an HTTP page",
    )
    serve.add_argument("--port", type=int, default=8502,
                       help="Port to serve on")
    serve.add_argument("--dir", default="reports/data_quality",
                       help="Dashboard output directory")
    serve.add_argument("--open", action="store_true",
                       help="Open browser automatically")

    # ── status ───────────────────────────────────────
    subparsers.add_parser(
        "status",
        help="Show the latest data quality snapshot",
        description="Display a summary of the last recorded data quality metrics.",
    )

    return parser


def cmd_generate(args: argparse.Namespace) -> int:
    """Execute the ``generate`` command."""
    # Determine data file path
    data_file = args.file or str(PROJECT_ROOT / "data" / "processed" / "results_clean.csv")
    data_path = Path(data_file)

    df = None
    if data_path.exists():
        try:
            df = pd.read_csv(data_path, low_memory=False)
            print(f"  📊 Loaded {len(df):,} rows from {data_path}")
        except Exception as exc:
            print(f"  ⚠ Failed to load data: {exc}")

    df_previous = None
    if args.prev:
        prev_path = Path(args.prev)
        if prev_path.exists():
            try:
                df_previous = pd.read_csv(prev_path, low_memory=False)
                print(f"  📊 Loaded {len(df_previous):,} rows from previous data")
            except Exception as exc:
                print(f"  ⚠ Failed to load previous data: {exc}")

    dq = DataQualityDashboard(
        df=df,
        source_name=args.source,
        output_dir=args.output,
        monitor_store=MonitoringStore(),
        df_previous=df_previous,
    )

    print(f"  🔍 Building data quality snapshot ...")
    results = dq.generate(days=args.days)

    for fmt, path_str in results.items():
        if path_str and not fmt.endswith("_error"):
            print(f"  ✅ {fmt.upper():6s}: {path_str}")

    for fmt, error in [(k.replace("_error", ""), v)
                        for k, v in results.items() if k.endswith("_error")]:
        print(f"  ❌ {fmt}: {error}")

    if args.open and results.get("html"):
        webbrowser.open(f"file://{Path(results['html']).resolve()}")

    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """Execute the ``coverage`` command."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return 1

    try:
        if file_path.suffix.lower() in (".csv",):
            metrics = CoverageAnalyzer.from_csv(file_path)
        elif file_path.suffix.lower() in (".parquet", ".pq"):
            metrics = CoverageAnalyzer.from_parquet(file_path)
        else:
            print(f"  ✗ Unsupported format: {file_path.suffix}")
            return 1
    except Exception as exc:
        print(f"  ✗ Coverage analysis failed: {exc}")
        return 1

    print(f"\n  📊 Coverage Analysis: {file_path}")
    print(f"  {'─' * 48}")
    print(f"  Odds Coverage:    {metrics.odds_coverage_pct:>6.1f}%")
    print(f"  xG Coverage:      {metrics.xg_coverage_pct:>6.1f}%")
    print(f"  League Coverage:  {metrics.league_coverage_pct:>6.1f}% ({len(metrics.league_coverage)} leagues)")
    print(f"  Season Coverage:  {metrics.season_count} seasons")
    print(f"  Schema:           {metrics.n_columns_actual}/{metrics.n_columns_expected} cols")
    if metrics.columns_missing:
        print(f"  Missing columns:  {', '.join(metrics.columns_missing[:8])}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Execute the ``serve`` command — start a simple HTTP server."""
    dashboard_dir = Path(args.dir)
    if not dashboard_dir.exists():
        print(f"  ✗ Dashboard directory not found: {dashboard_dir}")
        print(f"    Run 'python -m src.data_quality.cli generate' first.")
        return 1

    html_path = dashboard_dir / "data_quality.html"
    if not html_path.exists():
        print(f"  ✗ Dashboard file not found: {html_path}")
        print(f"    Run 'python -m src.data_quality.cli generate' first.")
        return 1

    port = args.port
    host = "127.0.0.1"
    url = f"http://{host}:{port}/data_quality.html"

    print(f"  🌐 Serving Data Quality Dashboard")
    print(f"     URL:  {url}")
    print(f"     Dir:  {dashboard_dir.resolve()}")
    print(f"     Press Ctrl+C to stop")

    if args.open:
        webbrowser.open(url)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logger.debug(format, *args)

    server = http.server.HTTPServer((host, port), QuietHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Execute the ``status`` command."""
    store = MonitoringStore()
    snap = store.get_latest()

    print(f"\n  📊 Latest Data Quality Snapshot")
    print(f"  {'─' * 48}")

    if snap.etl:
        e = snap.etl
        print(f"  📥 Pipeline:     {e.pipeline}")
        print(f"     Duration:     {e.duration_seconds:.1f}s")
        print(f"     Rows:         {e.rows_imported:,}")
        print(f"     Dups:         {e.duplicate_pct:.2f}%")
        print(f"     Success:      {'✅' if e.success else '❌'}")

    if snap.data_quality:
        dq = snap.data_quality
        print(f"  ✅ Data Quality: {dq.source}")
        print(f"     Rows:         {dq.n_rows:,}")
        print(f"     Null rate:    {dq.null_pct:.2f}%")
        print(f"     Dup rate:     {dq.duplicate_pct:.2f}%")

    if snap.system:
        s = snap.system
        print(f"  💻 System:       CPU {s.cpu_percent:.1f}% / Mem {s.memory_percent:.1f}%")
        print(f"     DB Size:      {s.db_size_mb:.2f} MB")

    if snap.cache:
        c = snap.cache
        print(f"  🎯 Cache:        Hit rate {c.hit_rate:.1%}")

    print(f"  {'─' * 48}")
    print(f"  Generated: {snap.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "generate": cmd_generate,
        "coverage": cmd_coverage,
        "serve": cmd_serve,
        "status": cmd_status,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        return 1
    except Exception as exc:
        print(f"  ✗ Error: {exc}")
        logger.exception("Command failed: %s", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
