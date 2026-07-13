"""
CLI entry point for the monitoring framework.

Usage
-----
::

    # Quick summary
    python -m src.monitoring summary
    python -m src.monitoring summary --days 7

    # Generate all reports
    python -m src.monitoring report
    python -m src.monitoring report --days 90 --output reports/monitoring

    # Open interactive dashboard
    python -m src.monitoring dashboard

    # Collect system metrics (one-shot)
    python -m src.monitoring collect

    # Cleanup old metrics
    python -m src.monitoring cleanup --retention 30

    # Storage statistics
    python -m src.monitoring stats

    # Serve dashboard on localhost (Python HTTP server)
    python -m src.monitoring serve --port 8080
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path


from src.monitoring.monitor import Monitor

logger = logging.getLogger(__name__)


def _create_monitor(args: argparse.Namespace) -> Monitor:
    """Create a Monitor instance from CLI args."""
    return Monitor(
        db_path=getattr(args, "db", "data/monitoring/monitor.db"),
        output_dir=getattr(args, "output", "reports/monitoring"),
        data_dir=getattr(args, "data_dir", "data"),
        retention_days=getattr(args, "retention", 90),
    )


def cmd_summary(args: argparse.Namespace) -> None:
    """Print the daily summary to stdout."""
    monitor = _create_monitor(args)
    days = args.days or 30
    text = monitor.daily_report.generate(
        label=f"Monitoring Summary — Last {days}d "
              f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
    )
    print(text)


def cmd_report(args: argparse.Namespace) -> None:
    """Generate all report formats."""
    monitor = _create_monitor(args)
    days = args.days or 30
    results = monitor.generate_reports(days=days)

    print(f"\n  📊 Reports generated ({days}d view)")
    print(f"  {'─' * 45}")
    if "html" in results:
        print(f"  ✅ HTML:    {results['html']}")
    if "json" in results:
        print(f"  ✅ JSON:    {results['json']}")
    if "csv" in results:
        for p in results["csv"]:
            print(f"  ✅ CSV:     {p}")
    if "summary" in results:
        print(f"  ✅ Summary: {results['summary']}")

    for key in ["html_error", "json_error", "csv_error", "summary_error"]:
        if key in results:
            print(f"  ⚠️  {key}: {results[key]}")

    print()


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Generate the HTML dashboard and open it in a browser."""
    monitor = _create_monitor(args)
    days = args.days or 30
    path = monitor.html_report.generate(days=days)
    print(f"  📊 Dashboard: {path}")
    try:
        webbrowser.open(str(path.absolute()))
        print("  🌐 Opened in browser")
    except Exception:
        print(f"  Open manually: file://{path.absolute()}")


def cmd_collect(args: argparse.Namespace) -> None:
    """Collect system metrics (one-shot)."""
    monitor = _create_monitor(args)
    row_id = monitor.record_system()
    print(f"  ✅ System metric recorded (id={row_id})")

    # Optionally collect cache
    if args.cache:
        row_id = monitor.record_cache(
            hits=args.cache_hits or 0,
            misses=args.cache_misses or 0,
            hit_rate=args.cache_rate or 0.0,
            entries=args.cache_entries or 0,
            size_bytes=args.cache_size or 0,
        )
        print(f"  ✅ Cache metric recorded (id={row_id})")


def cmd_cleanup(args: argparse.Namespace) -> None:
    """Remove metrics older than retention_days."""
    monitor = _create_monitor(args)
    retention = args.retention or 90
    deleted = monitor.cleanup(retention_days=retention)
    total = sum(deleted.values())
    print(f"  🧹 Cleanup: removed {total} rows older than {retention} days")
    for table, count in deleted.items():
        if count > 0:
            print(f"     {table}: {count} rows")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print storage statistics."""
    monitor = _create_monitor(args)
    stats = monitor.store.get_stats()
    print(f"\n  📊 Monitoring Store Statistics")
    print(f"  {'─' * 45}")
    print(f"  Database:    {stats['db_path']}")
    print(f"  Retention:   {stats['retention_days']} days")
    print()
    for table, info in stats["tables"].items():
        print(f"  {table}")
        print(f"     Count:   {info['count']}")
        print(f"     Oldest:  {info['oldest'] or 'N/A'}")
        print(f"     Newest:  {info['newest'] or 'N/A'}")
    print()


def cmd_serve(args: argparse.Namespace) -> None:
    """Generate the dashboard and start a local HTTP server."""
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    monitor = _create_monitor(args)
    days = args.days or 30
    monitor.html_report.generate(days=days)

    port = args.port or 8080
    output_dir = Path(args.output or "reports/monitoring")

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=str(output_dir), **kw)

    server = HTTPServer(("localhost", port), _Handler)
    print(f"  🌐 Monitoring dashboard at http://localhost:{port}/dashboard.html")
    print(f"  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped")
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Football Prediction — Monitoring Framework",
    )
    parser.add_argument("--db", default="data/monitoring/monitor.db",
                        help="Monitoring database path")
    parser.add_argument("--output", "-o", default="reports/monitoring",
                        help="Report output directory")
    parser.add_argument("--data-dir", default="data",
                        help="Data directory for disk checks")
    parser.add_argument("--retention", type=int, default=90,
                        help="Retention days (for cleanup)")

    sub = parser.add_subparsers(dest="command", required=True)

    # summary
    p_summary = sub.add_parser("summary", help="Print daily summary")
    p_summary.add_argument("--days", type=int, default=7,
                           help="Lookback days for trends")

    # report
    p_report = sub.add_parser("report", help="Generate all reports")
    p_report.add_argument("--days", type=int, default=30)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Open HTML dashboard")
    p_dash.add_argument("--days", type=int, default=30)

    # collect
    p_collect = sub.add_parser("collect", help="Collect system metrics")
    p_collect.add_argument("--cache", action="store_true",
                           help="Also record cache metrics")
    p_collect.add_argument("--cache-hits", type=int, default=0)
    p_collect.add_argument("--cache-misses", type=int, default=0)
    p_collect.add_argument("--cache-rate", type=float, default=0.0)
    p_collect.add_argument("--cache-entries", type=int, default=0)
    p_collect.add_argument("--cache-size", type=int, default=0)

    # cleanup
    p_clean = sub.add_parser("cleanup", help="Purge old metrics")
    p_clean.add_argument("--retention", type=int, default=90)

    # stats
    sub.add_parser("stats", help="Storage statistics")

    # serve
    p_serve = sub.add_parser("serve", help="Serve dashboard via HTTP")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--days", type=int, default=30)

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    commands = {
        "summary": cmd_summary,
        "report": cmd_report,
        "dashboard": cmd_dashboard,
        "collect": cmd_collect,
        "cleanup": cmd_cleanup,
        "stats": cmd_stats,
        "serve": cmd_serve,
    }

    try:
        commands[args.command](args)
        return 0
    except Exception as exc:
        logger.exception("Command '%s' failed", args.command)
        print(f"  ❌ Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
