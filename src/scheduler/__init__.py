"""
Automated Scheduler — orchestrates recurring football data tasks.

Manages 10 core tasks with support for Windows Task Scheduler,
cron (Unix), CLI commands, and manual execution.

Tasks
-----
1. **download_fixtures** — fetch new match data from configured sources
2. **update_database** — ingest parsed data into PostgreSQL
3. **validate_data** — run validation checks on imported data
4. **clean_data** — deduplicate, archive old raw files, purge stale checkpoints
5. **generate_logs** — rotate logs, archive old reports, write summary
6. **backup_database** — pg_dump or SQLite backup with retention
7. **daily_data_pipeline** — fetch new data from all sources, clean & merge
8. **daily_feature_computation** — compute all features via Feature Store
9. **daily_model_retraining** — retrain models, validate, save best
10. **daily_predictions** — load fixtures, generate predictions

Execution modes
---------------
- Sequential (default) — run tasks one at a time, abort on failure
- Parallel (optional) — run independent tasks concurrently
- Manual — run specific tasks via CLI
- Scheduled — install as Windows Task Scheduler or cron job

Notification system
-------------------
The ``Notifier`` class sends pipeline status alerts via console, email,
Slack webhook, and/or file log. Configured via environment variables.

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

    # Notifications
    from src.scheduler.notifications import Notifier
    notifier = Notifier()
    notifier.send_pipeline_report(report)
"""

from __future__ import annotations

from src.scheduler.cli import main as cli_main
from src.scheduler.engine import TaskEngine
from src.scheduler.models import ScheduleConfig, Task, TaskResult, TaskStatus
from src.scheduler.notifications import Notifier, NotificationConfig
from src.scheduler.tasks import (
    backup_database,
    clean_data,
    daily_data_pipeline,
    daily_feature_computation,
    daily_model_retraining,
    daily_predictions,
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
    "Notifier",
    "NotificationConfig",
    "download_fixtures",
    "update_database",
    "validate_data",
    "clean_data",
    "generate_logs",
    "backup_database",
    "daily_data_pipeline",
    "daily_feature_computation",
    "daily_model_retraining",
    "daily_predictions",
    "cli_main",
]
