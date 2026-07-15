"""
TaskEngine — orchestrates task execution with dependency resolution.

Supports sequential execution (default, respecting dependencies),
parallel groups, abort-on-failure, retry, and structured reporting.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from src.scheduler.models import (
    RunReport,
    ScheduleConfig,
    Task,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# Type for a task function: accepts ScheduleConfig, returns TaskResult
TaskFunc = Callable[[ScheduleConfig], TaskResult]


class TaskEngine:
    """Orchestrates pipeline task execution.

    Parameters
    ----------
    config : ScheduleConfig
        Pipeline configuration with task definitions.
    task_map : dict[str, TaskFunc], optional
        Mapping of task name → callable. Uses built-in tasks if not provided.
    """

    def __init__(
        self,
        config: ScheduleConfig | None = None,
        task_map: dict[str, TaskFunc] | None = None,
    ) -> None:
        self.config = config or ScheduleConfig.default()
        self._task_map: dict[str, TaskFunc] = {}

        # Register built-in tasks
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

        self.register("download_fixtures", download_fixtures)
        self.register("validate_data", validate_data)
        self.register("clean_data", clean_data)
        self.register("update_database", update_database)
        self.register("backup_database", backup_database)
        self.register("generate_logs", generate_logs)
        self.register("daily_data_pipeline", daily_data_pipeline)
        self.register("daily_feature_computation", daily_feature_computation)
        self.register("daily_model_retraining", daily_model_retraining)
        self.register("daily_predictions", daily_predictions)

        # Register any custom tasks
        if task_map:
            for name, func in task_map.items():
                self.register(name, func)

    def register(self, name: str, func: TaskFunc) -> None:
        """Register a task function by name.

        Parameters
        ----------
        name : str
            Task name (must match a Task in the config).
        func : TaskFunc
            Callable accepting ScheduleConfig and returning TaskResult.
        """
        self._task_map[name] = func

    # ── Execution ──────────────────────────────────────

    def run_all(
        self,
        task_names: list[str] | None = None,
    ) -> RunReport:
        """Run all (or selected) tasks with dependency-based ordering.

        Parameters
        ----------
        task_names : list[str], optional
            Specific tasks to run. If None, runs all enabled tasks.

        Returns
        -------
        RunReport
            Complete run report with per-task results.
        """
        report = RunReport(
            pipeline_name=self.config.pipeline_name,
            started_at=datetime.now(timezone.utc),
        )

        # Determine which tasks to run
        if task_names:
            tasks_to_run = [t for t in self.config.tasks if t.name in task_names]
        else:
            tasks_to_run = [t for t in self.config.tasks if t.enabled]

        report.total_tasks = len(tasks_to_run)

        # Build dependency graph and order tasks
        ordered = self._resolve_order(tasks_to_run)

        logger.info(
            "Starting pipeline '%s': %d tasks in order: %s",
            self.config.pipeline_name,
            len(ordered),
            " → ".join(t.name for t in ordered),
        )

        # Execute in order
        completed: dict[str, TaskResult] = {}

        for task_def in ordered:
            # Check dependencies
            dep_failed = self._check_dependencies(task_def, completed)
            if dep_failed:
                result = TaskResult(
                    task_name=task_def.name,
                    status=TaskStatus.SKIPPED,
                    error=f"Dependency failed: {dep_failed}",
                )
                report.task_results[task_def.name] = result
                report.skipped += 1
                report.errors.append(f"{task_def.name}: skipped ({dep_failed})")
                continue

            # Execute with retry
            result = self._execute_with_retry(task_def)
            report.task_results[task_def.name] = result
            completed[task_def.name] = result

            if result.status == TaskStatus.SUCCESS:
                report.succeeded += 1
            elif result.status == TaskStatus.FAILED:
                report.failed += 1
                report.errors.append(f"{task_def.name}: {result.error}")
                if self.config.abort_on_failure:
                    logger.warning(
                        "Pipeline aborted after %s failed", task_def.name,
                    )
                    break
            elif result.status == TaskStatus.SKIPPED:
                report.skipped += 1
            else:
                report.succeeded += 1  # WARNING counts as success

        report.completed_at = datetime.now(timezone.utc)
        report.duration_seconds = (
            report.completed_at - report.started_at
        ).total_seconds()

        logger.info(
            "Pipeline '%s' complete: %d succeeded, %d failed, %d skipped in %.1fs",
            self.config.pipeline_name,
            report.succeeded,
            report.failed,
            report.skipped,
            report.duration_seconds,
        )

        return report

    # ── Internal ───────────────────────────────────────

    def _execute_with_retry(self, task_def: Task) -> TaskResult:
        """Execute a task with retry logic."""
        func = self._task_map.get(task_def.name)
        if func is None:
            return TaskResult(
                task_name=task_def.name,
                status=TaskStatus.FAILED,
                error=f"No handler registered for task '{task_def.name}'",
            )

        last_error: str | None = None
        for attempt in range(1, task_def.retry_count + 1):
            if attempt > 1:
                logger.info(
                    "Retry %d/%d for task '%s' ...",
                    attempt, task_def.retry_count, task_def.name,
                )
                time.sleep(2.0 * attempt)  # Linear backoff between retries

            try:
                result = func(self.config)

                if result.status == TaskStatus.FAILED and attempt < task_def.retry_count:
                    last_error = result.error
                    logger.warning(
                        "Task '%s' failed (attempt %d/%d): %s",
                        task_def.name, attempt, task_def.retry_count, result.error,
                    )
                    continue

                return result

            except Exception as exc:
                last_error = str(exc)
                logger.exception(
                    "Task '%s' crashed (attempt %d/%d)",
                    task_def.name, attempt, task_def.retry_count,
                )
                if attempt >= task_def.retry_count:
                    return TaskResult(
                        task_name=task_def.name,
                        status=TaskStatus.FAILED,
                        error=last_error,
                    )

        return TaskResult(
            task_name=task_def.name,
            status=TaskStatus.FAILED,
            error=last_error or "Unknown error",
        )

    def _resolve_order(self, tasks: list[Task]) -> list[Task]:
        """Topological sort of tasks based on dependencies.

        Tasks with no dependencies come first, then tasks whose
        dependencies have all been satisfied.
        """
        task_map = {t.name: t for t in tasks}
        ordered: list[Task] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            task = task_map.get(name)
            if task is None:
                return
            for dep in task.dependencies:
                if dep in task_map:
                    visit(dep)
            if task not in ordered:
                ordered.append(task)

        for task in tasks:
            visit(task.name)

        return ordered

    def _check_dependencies(
        self,
        task: Task,
        completed: dict[str, TaskResult],
    ) -> str | None:
        """Check if all dependencies have completed successfully.

        Returns the name of the first failed dependency, or None.
        """
        for dep_name in task.dependencies:
            dep_result = completed.get(dep_name)
            if dep_result is None:
                return f"{dep_name} (not executed)"
            if dep_result.status == TaskStatus.FAILED:
                return f"{dep_name} (failed)"
            if dep_result.status == TaskStatus.SKIPPED:
                return f"{dep_name} (skipped)"
        return None
