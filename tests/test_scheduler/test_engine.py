"""
Unit tests for TaskEngine — orchestration, dependency resolution, retry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scheduler.engine import TaskEngine
from src.scheduler.models import (
    ScheduleConfig,
    Task,
    TaskResult,
    TaskStatus,
)


class TestTaskEngineRegistration:
    def test_init_with_builtin_tasks(self) -> None:
        """Engine has all 6 built-in tasks registered."""
        engine = TaskEngine()
        assert "download_fixtures" in engine._task_map
        assert "validate_data" in engine._task_map
        assert "clean_data" in engine._task_map
        assert "update_database" in engine._task_map
        assert "backup_database" in engine._task_map
        assert "generate_logs" in engine._task_map

    def test_register_custom_task(self) -> None:
        """Custom tasks can be registered and will be called."""
        engine = TaskEngine()

        def my_task(cfg):
            return TaskResult(
                task_name="my_task",
                status=TaskStatus.SUCCESS,
                output="Custom task ran",
            )

        engine.register("my_task", my_task)
        assert "my_task" in engine._task_map


class TestTaskEngineExecution:
    def test_execute_registered_task(self) -> None:
        """Running a registered task executes its function."""
        calls = []

        def test_task(cfg):
            calls.append("executed")
            return TaskResult(
                task_name="test_task",
                status=TaskStatus.SUCCESS,
                output="Ran successfully",
            )

        cfg = ScheduleConfig(tasks=[
            Task(name="test_task", enabled=True),
        ])
        engine = TaskEngine(config=cfg)
        engine.register("test_task", test_task)

        report = engine.run_all(task_names=["test_task"])
        assert len(calls) == 1
        assert report.succeeded == 1

    def test_unregistered_task_fails(self) -> None:
        """Running an unregistered task returns FAILED status."""
        cfg = ScheduleConfig(tasks=[
            Task(name="no_such_task", enabled=True),
        ])
        engine = TaskEngine(config=cfg)
        # Don't register the task

        report = engine.run_all(task_names=["no_such_task"])
        assert report.failed == 1
        assert "No handler registered" in report.errors[0]

    def test_dependency_skip(self) -> None:
        """If a dependency fails, dependents are skipped."""
        def failing_task(cfg):
            return TaskResult(
                task_name="task_a",
                status=TaskStatus.FAILED,
                error="Task A failure",
            )

        skipped = []

        def dependent_task(cfg):
            skipped.append("should not run")
            return TaskResult(task_name="task_b", status=TaskStatus.SUCCESS)

        cfg = ScheduleConfig(
            tasks=[
                Task(name="task_a", enabled=True),
                Task(name="task_b", dependencies=["task_a"], enabled=True),
            ],
            abort_on_failure=False,
        )
        engine = TaskEngine(config=cfg)
        engine.register("task_a", failing_task)
        engine.register("task_b", dependent_task)

        report = engine.run_all()
        assert report.failed == 1
        assert report.skipped == 1
        assert skipped == []  # dependent was never called

    def test_retry_on_failure(self) -> None:
        """Task with retry_count > 1 is retried on failure."""
        attempt_count = [0]

        def flaky_task(cfg):
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                return TaskResult(
                    task_name="flaky",
                    status=TaskStatus.FAILED,
                    error=f"Attempt {attempt_count[0]} failed",
                )
            return TaskResult(
                task_name="flaky",
                status=TaskStatus.SUCCESS,
                output=f"Succeeded on attempt {attempt_count[0]}",
            )

        cfg = ScheduleConfig(tasks=[
            Task(name="flaky", enabled=True, retry_count=3),
        ])
        engine = TaskEngine(config=cfg)
        engine.register("flaky", flaky_task)

        report = engine.run_all(task_names=["flaky"])
        assert report.succeeded == 1
        assert attempt_count[0] == 2  # Failed once, succeeded once

    def test_abort_on_failure(self) -> None:
        """Pipeline aborts after a task fails when abort_on_failure=True."""
        calls = []

        def task_a(cfg):
            calls.append("a")
            return TaskResult(task_name="a", status=TaskStatus.FAILED, error="fail")

        def task_b(cfg):
            calls.append("b")
            return TaskResult(task_name="b", status=TaskStatus.SUCCESS)

        cfg = ScheduleConfig(
            tasks=[
                Task(name="task_a", enabled=True),
                Task(name="task_b", enabled=True),
            ],
            abort_on_failure=True,
        )
        engine = TaskEngine(config=cfg)
        engine.register("task_a", task_a)
        engine.register("task_b", task_b)

        report = engine.run_all()
        assert report.failed == 1
        assert report.succeeded == 0  # task_b never ran
        assert calls == ["a"]  # Only task_a was called


class TestDependencyResolution:
    def test_topological_sort(self) -> None:
        """Tasks are ordered by dependencies."""
        cfg = ScheduleConfig(tasks=[
            Task(name="download", enabled=True),
            Task(name="validate", dependencies=["download"], enabled=True),
            Task(name="store", dependencies=["validate"], enabled=True),
        ])
        ordered = TaskEngine(config=cfg)._resolve_order(cfg.tasks)
        names = [t.name for t in ordered]
        # download must come before validate before store
        assert names.index("download") < names.index("validate")
        assert names.index("validate") < names.index("store")

    def test_no_dependency_order_preserved(self) -> None:
        """Tasks without dependencies keep their original order."""
        cfg = ScheduleConfig(tasks=[
            Task(name="a", enabled=True),
            Task(name="b", enabled=True),
            Task(name="c", enabled=True),
        ])
        ordered = TaskEngine(config=cfg)._resolve_order(cfg.tasks)
        assert [t.name for t in ordered] == ["a", "b", "c"]
