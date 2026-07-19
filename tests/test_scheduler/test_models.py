"""
Unit tests for scheduler data models.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.scheduler.models import (
    RunReport,
    ScheduleConfig,
    Task,
    TaskResult,
    TaskStatus,
)


class TestTaskStatus:
    def test_values(self) -> None:
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestTask:
    def test_minimal(self) -> None:
        task = Task(name="test_task")
        assert task.name == "test_task"
        assert task.enabled is True
        assert task.timeout_seconds == 600
        assert task.retry_count == 1

    def test_with_deps(self) -> None:
        task = Task(
            name="task_b",
            dependencies=["task_a"],
            parallel_group="group1",
        )
        assert task.dependencies == ["task_a"]
        assert task.parallel_group == "group1"


class TestTaskResult:
    def test_pending(self) -> None:
        result = TaskResult(task_name="test")
        assert result.status == TaskStatus.PENDING
        assert result.success is False

    def test_successful(self) -> None:
        result = TaskResult(
            task_name="test",
            status=TaskStatus.SUCCESS,
            output="All good",
            records_processed=100,
        )
        assert result.success is True
        assert result.records_processed == 100

    def test_to_dict(self) -> None:
        result = TaskResult(
            task_name="test",
            status=TaskStatus.FAILED,
            error="Something broke",
            records_processed=0,
        )
        d = result.to_dict()
        assert d["task_name"] == "test"
        assert d["status"] == "failed"
        assert "broke" in d["error"]


class TestRunReport:
    def test_success_empty(self) -> None:
        report = RunReport()
        assert report.success is True
        assert report.failed == 0

    def test_with_failures(self) -> None:
        report = RunReport(
            failed=2,
            errors=["Task A failed", "Task B failed"],
        )
        assert report.success is False
        assert len(report.errors) == 2

    def test_to_dict(self) -> None:
        now = datetime.now(timezone.utc)
        report = RunReport(
            pipeline_name="test_pipeline",
            started_at=now,
            completed_at=now,
            duration_seconds=10.5,
            total_tasks=3,
            succeeded=2,
            failed=1,
            skipped=0,
            errors=["Task B failed"],
        )
        d = report.to_dict()
        assert d["pipeline_name"] == "test_pipeline"
        assert d["duration_seconds"] == 10.5
        assert d["succeeded"] == 2


class TestScheduleConfig:
    def test_default_config(self) -> None:
        cfg = ScheduleConfig.default()
        assert len(cfg.tasks) == 10
        assert cfg.pipeline_name == "football_pipeline"

    def test_default_task_names(self) -> None:
        cfg = ScheduleConfig.default()
        names = [t.name for t in cfg.tasks]
        assert "download_fixtures" in names
        assert "update_database" in names
        assert "validate_data" in names
        assert "clean_data" in names
        assert "backup_database" in names
        assert "generate_logs" in names

    def test_task_dependencies(self) -> None:
        cfg = ScheduleConfig.default()
        task_map = {t.name: t for t in cfg.tasks}

        # validate_data depends on download_fixtures
        assert "download_fixtures" in task_map["validate_data"].dependencies

        # update_database depends on clean_data
        assert "clean_data" in task_map["update_database"].dependencies

    def test_to_dict(self) -> None:
        cfg = ScheduleConfig.default()
        d = cfg.to_dict()
        assert d["pipeline_name"] == "football_pipeline"
        assert len(d["tasks"]) == 10
