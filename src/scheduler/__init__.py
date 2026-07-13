"""
Automated Scheduler — orchestrates recurring football data tasks.

Manages 6 core tasks with support for Windows Task Scheduler,
cron (Unix), CLI commands, and manual execution.

Tasks
-----
1. **download_fixtures** — fetch new match data from configured sources
2. **update_database** — ingest parsed data into PostgreSQL
3. **validate_data** — run validation checks on imported data
4. **clean_data** — deduplicate, archive old raw files, purge stale checkpoints
5. **generate_logs** — rotate logs, archive old reports, write summary
6. **backup_database** — pg_dump or SQLite backup with retention

Execution modes
---------------
- Sequential (default) — run tasks one at a time, abort on failure
- Parallel (optional) — run independent tasks concurrently
- Manual — run specific tasks via CLI
- Scheduled — install as Windows Task Scheduler or cron job

Usage
-----
::

    # CLI
    python -m src.scheduler.cli run                          # Run all tasks
    python -m src.scheduler.cli run --tasks download,backup  # Selected tasks
    python -m src.scheduler.cli install --platform windows   # Install scheduler
    python -m src.scheduler.cli status                       # Check last run

    # Programmatic
    from src.scheduler import TaskEngine
    engine = TaskEngine()
    report = engine.run_all()
"""

from __future__ import annotations

from src.scheduler.cli import main as cli_main
from src.scheduler.engine import TaskEngine
from src.scheduler.models import ScheduleConfig, Task, TaskResult, TaskStatus
from src.scheduler.tasks import (
    backup_database,
    clean_data,
    download_fixtures,
    generate_logs,
    update_database,
    validate_data,
)

__all__ = [
    "TaskEngine",
    "Task",
    "TaskResult",
    "TaskStatus",
    "ScheduleConfig",
    "download_fixtures",
    "update_database",
    "validate_data",
    "clean_data",
    "generate_logs",
    "backup_database",
    "cli_main",
]
