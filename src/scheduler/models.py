"""
Data models for the automated scheduler system.

Defines Task, TaskResult, TaskStatus, and ScheduleConfig used
by all scheduler components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Status of a single task execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Definition of a schedulable task.

    Attributes
    ----------
    name : str
        Unique task name (e.g. ``download_fixtures``).
    description : str
        Human-readable description.
    enabled : bool
        Whether the task is enabled by default (default True).
    timeout_seconds : int
        Max execution time before the task is killed (default 600).
    retry_count : int
        Number of retries on failure (default 1).
    dependencies : list[str]
        Task names that must complete before this one starts.
    parallel_group : str or None
        If set, tasks with the same group name can run in parallel.
    """

    name: str
    description: str = ""
    enabled: bool = True
    timeout_seconds: int = 600
    retry_count: int = 1
    dependencies: list[str] = field(default_factory=list)
    parallel_group: str | None = None


@dataclass
class TaskResult:
    """Result of executing a single task.

    Attributes
    ----------
    task_name : str
        Name of the task that was executed.
    status : TaskStatus
        Outcome of the execution.
    started_at : datetime or None
        When execution began.
    completed_at : datetime or None
        When execution finished.
    duration_seconds : float
        Wall-clock execution time.
    output : str
        Captured stdout/stderr or summary message.
    error : str or None
        Error message if the task failed.
    records_processed : int
        Number of records affected (rows, files, etc.).
    warnings : list[str]
        Non-fatal issues encountered during execution.
    """

    task_name: str = ""
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    output: str = ""
    error: str | None = None
    records_processed: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == TaskStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "output": self.output[:500],
            "error": self.error[:500] if self.error else None,
            "records_processed": self.records_processed,
            "warnings": self.warnings[:10],
        }


@dataclass
class RunReport:
    """Complete report of a scheduler run.

    Attributes
    ----------
    pipeline_name : str
        Name of the pipeline that ran.
    started_at : datetime
        When the pipeline started.
    completed_at : datetime or None
        When the pipeline finished.
    duration_seconds : float
        Total execution time.
    task_results : dict[str, TaskResult]
        Results indexed by task name.
    total_tasks : int
        Number of tasks in the pipeline.
    succeeded : int
        Number of tasks that completed successfully.
    failed : int
        Number of tasks that failed.
    skipped : int
        Number of tasks that were skipped.
    errors : list[str]
        All error messages collected across tasks.
    """

    pipeline_name: str = "default"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    total_tasks: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_tasks": self.total_tasks,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors[:20],
        }


@dataclass
class ScheduleConfig:
    """Configuration for the task scheduler.

    Attributes
    ----------
    pipeline_name : str
        Name for this pipeline instance.
    tasks : list[Task]
        All tasks in the pipeline.
    parallel_groups : bool
        Whether to run tasks in the same parallel group concurrently (default False).
    abort_on_failure : bool
        Whether to stop the pipeline if a task fails (default True).
    log_dir : str
        Directory for pipeline logs (default ``logs/scheduler``).
    report_dir : str
        Directory for run reports (default ``reports/scheduler``).
    backup_dir : str
        Directory for database backups (default ``data/backups``).
    backup_retention_days : int
        Number of days to keep backups (default 7).
    max_log_age_days : int
        Maximum age of log files before archiving (default 30).
    """

    pipeline_name: str = "football_pipeline"
    tasks: list[Task] = field(default_factory=list)
    parallel_groups: bool = False
    abort_on_failure: bool = True
    log_dir: str = "logs/scheduler"
    report_dir: str = "reports/scheduler"
    backup_dir: str = "data/backups"
    backup_retention_days: int = 7
    max_log_age_days: int = 30

    @classmethod
    def default(cls) -> ScheduleConfig:
        """Create the default pipeline configuration with all 6 tasks."""
        return cls(
            tasks=[
                Task(
                    name="download_fixtures",
                    description="Download new match data from configured sources",
                    timeout_seconds=300,
                    retry_count=2,
                ),
                Task(
                    name="validate_data",
                    description="Validate integrity and schema of downloaded data",
                    timeout_seconds=120,
                    dependencies=["download_fixtures"],
                ),
                Task(
                    name="clean_data",
                    description="Deduplicate, normalise, and archive raw data",
                    timeout_seconds=180,
                    dependencies=["validate_data"],
                ),
                Task(
                    name="update_database",
                    description="Ingest cleaned data into the database",
                    timeout_seconds=600,
                    dependencies=["clean_data"],
                    retry_count=2,
                ),
                Task(
                    name="backup_database",
                    description="Create a database backup with retention",
                    timeout_seconds=300,
                    parallel_group="maintenance",
                ),
                Task(
                    name="generate_logs",
                    description="Rotate logs, archive reports, write summary",
                    timeout_seconds=60,
                    parallel_group="maintenance",
                    dependencies=["backup_database"],
                ),
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "tasks": [t.name for t in self.tasks],
            "parallel_groups": self.parallel_groups,
            "abort_on_failure": self.abort_on_failure,
            "backup_retention_days": self.backup_retention_days,
        }
