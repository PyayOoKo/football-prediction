"""
CLI — command-line interface for the data profiling system.

Commands
--------
::

    # Profile a CSV or Parquet file
    python -m src.data_profiling create-report data/raw/results.csv --source my_dataset

    # List all profiling reports
    python -m src.data_profiling list-reports

    # Compare two reports
    python -m src.data_profiling compare --prev report_1.json --curr report_2.json

    # Run auto-profiling on latest data and generate all outputs
    python -m src.data_profiling auto --source my_dataset

    # Generate dashboard without re-profiling (from existing JSON)
    python -m src.data_profiling dashboard reports/profiling/report.json

All commands write output to ``reports/profiling/`` by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_profiling import DataProfiler, DataDriftDetector
from src.data_profiling.profiler import ProfilingReport

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/profiling")


def _ensure_reports_dir() -> Path:
    """Create reports/profiling directory if needed."""
    p = REPORTS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_report(path: str) -> ProfilingReport:
    """Load a ProfilingReport from a JSON file on disk."""
    from src.data_profiling.profiler import ProfileSection

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = ProfilingReport(source_name=data.get("source_name", "unknown"))

    # Populate metadata
    if "timestamp" in data:
        from datetime import datetime
        try:
            report.timestamp = datetime.fromisoformat(data["timestamp"])
        except (ValueError, TypeError, KeyError):
            pass
    report.n_rows = data.get("n_rows", 0)
    report.n_columns = data.get("n_columns", 0)
    report.duration_seconds = data.get("duration_seconds", 0.0)

    # Populate sections
    sections_data = data.get("sections", {})
    section_names = [
        "missing_values", "duplicate_records", "column_summary",
        "result_distribution", "goal_distribution", "odds_distribution",
        "league_distribution", "season_distribution", "team_distribution",
        "home_advantage", "outliers", "schema_validation", "type_validation",
        "data_drift",
    ]
    for name in section_names:
        d = sections_data.get(name, {})
        section = ProfileSection(
            name=d.get("name", name),
            data=d.get("data", {}),
            chart_type=d.get("chart_type", "table"),
        )
        setattr(report, name, section)

    return report


def cmd_create_report(args: argparse.Namespace) -> int:
    """Profile a CSV/Parquet file and generate all report formats."""
    _ensure_reports_dir()

    # Load data
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return 1

    logger.info("Loading data from %s", filepath)
    try:
        if filepath.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, low_memory=False)
    except pd.errors.EmptyDataError:
        print(f"⚠ Empty file: {filepath}")
        return 0

    source_name = args.source or filepath.stem
    logger.info("Loaded %d rows, %d columns — profiling...", len(df), len(df.columns))

    # Profile
    profiler = DataProfiler(
        odds_column_patterns=args.odds_patterns,
        outlier_std_threshold=args.outlier_std,
    )
    report = profiler.profile(df, source_name=source_name)

    # Print summary
    print(report.summary_text())

    # Save JSON
    json_path = REPORTS_DIR / f"{source_name}.json"
    report.to_json(str(json_path))
    print(f"  ✅ JSON  → {json_path}")

    # Save CSV
    csv_path = REPORTS_DIR / f"{source_name}.csv"
    report.to_csv(str(csv_path))
    print(f"  ✅ CSV   → {csv_path}")

    # Save HTML
    html_path = REPORTS_DIR / f"{source_name}.html"
    report.to_html(str(html_path))
    print(f"  ✅ HTML  → {html_path}")

    return 0


def cmd_list_reports(args: argparse.Namespace) -> int:
    """List all profiling reports in the reports directory."""
    reports_dir = _ensure_reports_dir()
    json_files = sorted(reports_dir.glob("*.json"))

    if not json_files:
        print("📊 No profiling reports found.")
        return 0

    print(f"\n📊 Profiling Reports ({len(json_files)} found)\n")
    print(f"{'File':<40s} {'Rows':<10s} {'Source':<25s} {'Date':<20s}")
    print("-" * 95)
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            src = data.get("source_name", "?")
            rows = str(data.get("n_rows", "?"))
            ts = data.get("timestamp", "?")[:19] if data.get("timestamp") else "?"
            print(f"{jf.name:<40s} {rows:<10s} {src:<25s} {ts:<20s}")
        except (json.JSONDecodeError, OSError):
            print(f"{jf.name:<40s} {'ERROR':<10s} {'':<25s} {'':<20s}")

    print()
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two profiling reports and detect data drift."""
    _ensure_reports_dir()

    prev_path = args.prev
    curr_path = args.curr

    if not Path(prev_path).exists():
        print(f"❌ Previous report not found: {prev_path}")
        return 1
    if not Path(curr_path).exists():
        print(f"❌ Current report not found: {curr_path}")
        return 1

    prev_report = _load_report(prev_path)
    curr_report = _load_report(curr_path)

    detector = DataDriftDetector(
        row_count_threshold=args.row_threshold,
        null_pct_threshold=args.null_threshold,
        metric_threshold=args.metric_threshold,
    )
    drift = detector.detect(curr_report, prev_report)

    print()
    print(drift.summary_text())
    print()

    # Save drift HTML if requested
    if args.output:
        drift_html = _drift_to_html(drift)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(drift_html)
        print(f"  ✅ Drift report → {args.output}")

    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Auto-profile: find latest CSV data, profile it, compare with previous."""
    _ensure_reports_dir()

    # Find latest CSV data
    source_name = args.source or "latest"
    data_dirs = [
        Path("data/raw"),
        Path("data/processed"),
        Path("data"),
    ]

    found_file: Path | None = None
    for d in data_dirs:
        if d.exists():
            csvs = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if csvs:
                found_file = csvs[0]
                break

    if found_file is None:
        print("❌ No CSV data files found in data/ directories.")
        return 1

    print(f"📂 Using latest data file: {found_file}")

    # Profile
    df = pd.read_csv(found_file, low_memory=False)
    profiler = DataProfiler()
    report = profiler.profile(df, source_name=source_name)

    # Save
    report.to_json(str(REPORTS_DIR / f"{source_name}.json"))
    report.to_csv(str(REPORTS_DIR / f"{source_name}.csv"))
    report.to_html(str(REPORTS_DIR / f"{source_name}.html"))

    print(f"  ✅ Reports saved to {REPORTS_DIR}/")
    print(report.summary_text())

    # Compare with previous report if it exists
    prev_path = REPORTS_DIR / f"{source_name}.previous.json"
    if prev_path.exists():
        print("\n📊 Comparing with previous profile...")
        prev_report = _load_report(str(prev_path))
        detector = DataDriftDetector()
        drift = detector.detect(report, prev_report)
        print(drift.summary_text())

    # Save current as previous for next comparison
    import shutil
    current_path = REPORTS_DIR / f"{source_name}.json"
    if current_path.exists():
        shutil.copy2(current_path, prev_path)

    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Generate HTML dashboard from an existing JSON profiling report."""
    json_path = Path(args.report_json)
    if not json_path.exists():
        print(f"❌ Report not found: {json_path}")
        return 1

    report = _load_report(str(json_path))

    html_path = json_path.with_suffix(".html")
    report.to_html(str(html_path))
    print(f"  ✅ Dashboard → {html_path}")

    return 0


def _drift_to_html(drift: Any) -> str:
    """Generate a simple HTML page for drift results."""
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Data Drift Report</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; background: #0e1117; color: #e0e0e0; padding: 24px; }",
        "h1 { color: #4fc3f7; }",
        ".metric { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px; padding: 12px; margin: 8px 0; }",
        ".info { border-left: 4px solid #3b82f6; }",
        ".warning { border-left: 4px solid #f59e0b; }",
        ".critical { border-left: 4px solid #ef4444; }",
        ".name { font-weight: 600; }",
        ".delta { color: #8b8fa3; font-size: 0.9em; }",
        "</style></head><body>",
        f"<h1>📊 Data Drift: {drift.previous_source} → {drift.current_source}</h1>",
        f"<p>{drift.n_warnings} signal(s) detected</p>",
    ]
    for m in drift.metrics:
        sev_class = m.severity
        lines.append(
            f'<div class="metric {sev_class}">'
            f'<div class="name">{m.name}</div>'
            f'<div class="delta">{m.previous_value} → {m.current_value} ({m.delta:+.1%})</div>'
            f'<div>{m.description}</div>'
            f'</div>'
        )
    lines.append("</body></html>")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="📊 Football Data Profiling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── create-report ─────────────────────────────────
    p_create = sub.add_parser("create-report", help="Profile a CSV/Parquet file")
    p_create.add_argument("filepath", help="Path to CSV or Parquet file")
    p_create.add_argument("--source", help="Dataset source name (default: filename stem)")
    p_create.add_argument("--odds-patterns", nargs="*", default=None,
                          help="Column patterns for odds detection")
    p_create.add_argument("--outlier-std", type=float, default=3.0,
                          help="Z-score threshold for outliers (default: 3.0)")

    # ── list-reports ──────────────────────────────────
    sub.add_parser("list-reports", help="List all profiling reports")

    # ── compare ───────────────────────────────────────
    p_compare = sub.add_parser("compare", help="Compare two profiling reports")
    p_compare.add_argument("--prev", required=True, help="Previous report JSON path")
    p_compare.add_argument("--curr", required=True, help="Current report JSON path")
    p_compare.add_argument("--output", "-o", help="Output path for drift HTML report")
    p_compare.add_argument("--row-threshold", type=float, default=0.05, help="Row count drift threshold")
    p_compare.add_argument("--null-threshold", type=float, default=5.0, help="Null %% drift threshold (pp)")
    p_compare.add_argument("--metric-threshold", type=float, default=0.10, help="Metric drift threshold")

    # ── auto ──────────────────────────────────────────
    p_auto = sub.add_parser("auto", help="Auto-profile latest data and compare")
    p_auto.add_argument("--source", default="latest", help="Source name for the report")

    # ── dashboard ─────────────────────────────────────
    p_dash = sub.add_parser("dashboard", help="Generate HTML dashboard from JSON report")
    p_dash.add_argument("report_json", help="Path to profiling JSON report")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "create-report": cmd_create_report,
        "list-reports": cmd_list_reports,
        "compare": cmd_compare,
        "auto": cmd_auto,
        "dashboard": cmd_dashboard,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
