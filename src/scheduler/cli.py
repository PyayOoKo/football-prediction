"""
CLI — command-line interface for the scheduler system.

Supports manual execution, scheduled installation (Windows/cron),
and status queries.

Usage
-----
::

    # Run all tasks
    python -m src.scheduler.cli run

    # Run specific tasks
    python -m src.scheduler.cli run --tasks download_fixtures,backup_database

    # Run with selected output
    python -m src.scheduler.cli run --quiet --no-report

    # Install scheduled task
    python -m src.scheduler.cli install --platform windows
    python -m src.scheduler.cli install --platform cron

    # Generate install scripts
    python -m src.scheduler.cli generate --platform windows --output setup.bat

    # Check status of last run
    python -m src.scheduler.cli status

    # List available tasks
    python -m src.scheduler.cli list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scheduler.engine import TaskEngine
from src.scheduler.models import ScheduleConfig

logger = logging.getLogger(__name__)

TASK_NAMES = [
    "download_fixtures",
    "validate_data",
    "clean_data",
    "update_database",
    "backup_database",
    "generate_logs",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.scheduler.cli",
        description="Football Prediction — Automated Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ── Run ────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run", help="Run pipeline tasks",
        description="Execute one or more pipeline tasks.",
    )
    run_parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task names to run (default: all enabled)",
    )
    run_parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output (for cron/scheduled runs)",
    )
    run_parser.add_argument(
        "--no-report", action="store_true",
        help="Skip writing report to disk",
    )
    run_parser.add_argument(
        "--abort", action="store_true", default=True,
        help="Abort pipeline on task failure (default: True)",
    )

    # ── Install ─────────────────────────────────────────
    install_parser = subparsers.add_parser(
        "install", help="Install scheduled task",
        description="Install the pipeline as a scheduled task (Windows Task Scheduler or cron).",
    )
    install_parser.add_argument(
        "--platform", choices=["windows", "cron"], default="windows",
        help="Target platform (default: windows)",
    )
    install_parser.add_argument(
        "--schedule", default="daily",
        help="Schedule frequency: daily, hourly, weekly (default: daily)",
    )
    install_parser.add_argument(
        "--time", default="08:00",
        help="Schedule time (HH:MM) for daily tasks (default: 08:00)",
    )
    install_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be installed without actually installing",
    )

    # ── Generate ────────────────────────────────────────
    gen_parser = subparsers.add_parser(
        "generate", help="Generate install scripts",
        description="Generate .bat or crontab files without installing.",
    )
    gen_parser.add_argument(
        "--platform", choices=["windows", "cron"], default="windows",
        help="Target platform",
    )
    gen_parser.add_argument(
        "--output", default="",
        help="Output file path (default: auto-named)",
    )

    # ── Status ─────────────────────────────────────────
    subparsers.add_parser(
        "status", help="Show scheduler status",
        description="Check last run status and scheduled tasks.",
    )

    # ── List ───────────────────────────────────────────
    subparsers.add_parser(
        "list", help="List available tasks",
        description="Show all available pipeline tasks with descriptions.",
    )

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the ``run`` command."""
    cfg = ScheduleConfig.default()
    cfg.abort_on_failure = args.abort

    engine = TaskEngine(config=cfg)

    task_names = None
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    report = engine.run_all(task_names=task_names)

    # Save report
    if not args.no_report:
        report_dir = Path(cfg.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    # Output
    if not args.quiet:
        print(format_report(report))

    return 0 if report.success else 1


def cmd_install(args: argparse.Namespace) -> int:
    """Execute the ``install`` command."""
    if args.platform == "windows":
        from src.scheduler.windows_scheduler import (
            WindowsScheduleTask,
            WindowsScheduler,
        )

        scheduler = WindowsScheduler()
        if args.dry_run:
            task = WindowsScheduleTask(
                schedule_type=args.schedule,
                schedule_time=args.time,
            )
            print(scheduler.create_task_bat(task))
        else:
            task = WindowsScheduleTask(
                schedule_type=args.schedule,
                schedule_time=args.time,
            )
            success = scheduler.install_task(task)
            return 0 if success else 1

    elif args.platform == "cron":
        from src.scheduler.cron_scheduler import CronScheduler

        scheduler = CronScheduler()
        if args.dry_run:
            print(scheduler.generate_crontab())
        else:
            success = scheduler.install_crontab()
            return 0 if success else 1

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Execute the ``generate`` command."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d")

    if args.platform == "windows":
        from src.scheduler.windows_scheduler import (
            WindowsScheduleTask,
            WindowsScheduler,
        )

        scheduler = WindowsScheduler()
        output = args.output or f"setup_football_pipeline_{timestamp}.bat"

        scheduler.save_task_bat(output)
        print(f"Generated: {output}")

    elif args.platform == "cron":
        from src.scheduler.cron_scheduler import CronScheduler

        scheduler = CronScheduler()
        output = args.output or f"crontab_{timestamp}.txt"
        scheduler.save_crontab(output)
        print(f"Generated: {output}")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Execute the ``status`` command."""
    report_dir = Path(ScheduleConfig.default().report_dir)

    # Find most recent run report
    if report_dir.exists():
        reports = sorted(report_dir.glob("run_*.json"), reverse=True)
        if reports:
            latest = reports[0]
            with open(latest) as f:
                data = json.load(f)
            print(format_report_dict(data))
        else:
            print("  No run reports found.")
            print(f"  Run a pipeline first: python -m src.scheduler.cli run")
    else:
        print(f"  Report directory not found: {report_dir}")

    # Show scheduled tasks on Windows
    if sys.platform == "win32":
        from src.scheduler.windows_scheduler import WindowsScheduler

        scheduler = WindowsScheduler()
        tasks = scheduler.list_tasks()
        if tasks:
            print(f"\n  Windows Scheduled Tasks:")
            for t in tasks:
                print(f"    {t['name']}: {t['status']} (next: {t['next_run']})")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Execute the ``list`` command."""
    print(f"  {'Task Name':<25} {'Enabled':<10} {'Retries':<10} {'Depends On'}")
    print(f"  {'─' * 65}")
    cfg = ScheduleConfig.default()
    for task in cfg.tasks:
        deps = ", ".join(task.dependencies) if task.dependencies else "—"
        print(f"  {task.name:<25} {'Yes' if task.enabled else 'No':<10} "
              f"{task.retry_count:<10} {deps}")
    print(f"\n  Pipeline: {cfg.pipeline_name}")
    print(f"  Tasks: {len(cfg.tasks)} total")
    return 0


# ── Formatting ───────────────────────────────────────────


def format_report(report: Any) -> str:
    """Format a RunReport as a human-readable string."""
    return format_report_dict(report.to_dict() if hasattr(report, "to_dict") else report)


def format_report_dict(data: dict[str, Any]) -> str:
    """Format a report dict as a human-readable string."""
    lines = [
        f"  Pipeline:  {data.get('pipeline_name', '?')}",
        f"  Started:   {data.get('started_at', '?')}",
        f"  Duration:  {data.get('duration_seconds', 0):.1f}s",
        f"  Results:   {data.get('succeeded', 0)} succeeded, "
        f"{data.get('failed', 0)} failed, "
        f"{data.get('skipped', 0)} skipped",
    ]

    if data.get("errors"):
        lines.append(f"  Errors:")
        for e in data["errors"][:5]:
            lines.append(f"    - {e[:120]}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. Defaults to sys.argv[1:].

    Returns
    -------
    int
        Exit code: 0 for success, 1 for failure.
    """
    # Configure logging
    from src.config.logging import configure_logging
    configure_logging(level="INFO")

    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {
        "run": cmd_run,
        "install": cmd_install,
        "generate": cmd_generate,
        "status": cmd_status,
        "list": cmd_list,
    }

    if args.command is None:
        parser.print_help()
        return 1

    handler = commands.get(args.command)
    if not handler:
        print(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        return 1
    except Exception as exc:
        logger.exception("Command failed: %s", args.command)
        print(f"  Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
