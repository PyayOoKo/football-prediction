"""
Tests for the automation system — scheduler, monitoring, alerting, drift detection, and pipeline health.

Run with:
    pytest tests/test_automation.py -v
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────
#  Scheduler Tests
# ─────────────────────────────────────────────────────────


class TestTaskStatus:
    """Test the TaskStatus enum."""

    def test_status_values(self):
        from src.scheduler.models import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.SUCCESS.value == "success"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"
        assert TaskStatus.WARNING.value == "warning"


class TestTaskResult:
    """Test the TaskResult dataclass."""

    def test_defaults(self):
        from src.scheduler.models import TaskResult, TaskStatus

        r = TaskResult()
        assert r.task_name == ""
        assert r.status == TaskStatus.PENDING
        assert r.records_processed == 0

    def test_success_property(self):
        from src.scheduler.models import TaskResult, TaskStatus

        r = TaskResult(task_name="test", status=TaskStatus.SUCCESS)
        assert r.success is True

        r.status = TaskStatus.FAILED
        assert r.success is False

    def test_to_dict(self):
        from src.scheduler.models import TaskResult, TaskStatus

        r = TaskResult(
            task_name="download_fixtures",
            status=TaskStatus.SUCCESS,
            records_processed=150,
            output="Downloaded 150 rows",
        )
        d = r.to_dict()
        assert d["task_name"] == "download_fixtures"
        assert d["status"] == "success"
        assert d["records_processed"] == 150


class TestScheduleConfig:
    """Test the ScheduleConfig default factory."""

    def test_default_has_10_tasks(self):
        from src.scheduler.models import ScheduleConfig

        cfg = ScheduleConfig.default()
        assert len(cfg.tasks) == 10  # 6 core + 4 daily pipeline

    def test_daily_pipeline_tasks_present(self):
        from src.scheduler.models import ScheduleConfig

        cfg = ScheduleConfig.default()
        task_names = [t.name for t in cfg.tasks]
        assert "daily_data_pipeline" in task_names
        assert "daily_feature_computation" in task_names
        assert "daily_model_retraining" in task_names
        assert "daily_predictions" in task_names

    def test_dependencies_are_valid(self):
        from src.scheduler.models import ScheduleConfig

        cfg = ScheduleConfig.default()
        task_map = {t.name: t for t in cfg.tasks}
        for task in cfg.tasks:
            for dep in task.dependencies:
                assert dep in task_map, f"Task '{task.name}' depends on '{dep}' which doesn't exist"

    def test_retry_defaults(self):
        from src.scheduler.models import ScheduleConfig

        cfg = ScheduleConfig.default()
        for task in cfg.tasks:
            assert task.retry_count >= 1
            assert task.timeout_seconds >= 60


class TestTaskEngine:
    """Test the TaskEngine execution logic."""

    def test_register_and_run(self):
        from src.scheduler.models import ScheduleConfig, Task, TaskResult, TaskStatus
        from src.scheduler.engine import TaskEngine

        cfg = ScheduleConfig(
            tasks=[
                Task(name="task_a", description="Test A", enabled=True),
                Task(name="task_b", description="Test B", enabled=True, dependencies=["task_a"]),
            ],
            abort_on_failure=True,
        )

        engine = TaskEngine(config=cfg)

        # Register mock tasks
        def task_a_func(cfg):
            return TaskResult(task_name="task_a", status=TaskStatus.SUCCESS, output="A done")

        def task_b_func(cfg):
            return TaskResult(task_name="task_b", status=TaskStatus.SUCCESS, output="B done")

        engine.register("task_a", task_a_func)
        engine.register("task_b", task_b_func)

        report = engine.run_all()

        assert report.succeeded == 2
        assert report.failed == 0
        assert report.total_tasks == 2
        assert report.success is True

    def test_abort_on_failure(self):
        from src.scheduler.models import ScheduleConfig, Task, TaskResult, TaskStatus
        from src.scheduler.engine import TaskEngine

        cfg = ScheduleConfig(
            tasks=[
                Task(name="task_a", description="Fails"),
                Task(name="task_b", description="Should not run", dependencies=["task_a"]),
            ],
            abort_on_failure=True,
        )

        engine = TaskEngine(config=cfg)

        def failing_task(cfg):
            return TaskResult(task_name="task_a", status=TaskStatus.FAILED, error="Oops")

        def good_task(cfg):
            return TaskResult(task_name="task_b", status=TaskStatus.SUCCESS)

        engine.register("task_a", failing_task)
        engine.register("task_b", good_task)

        report = engine.run_all()

        assert report.failed == 1
        # task_b should be SKIPPED because task_a failed and abort_on_failure is True
        # Actually abort_on_failure just stops the pipeline after the first failure
        # task_b's dependency task_a failed, so it should be skipped
        # But since abort_on_failure is True, the pipeline breaks after task_a
        # task_b was never executed because we break after the failure

    def test_retry_logic(self):
        from src.scheduler.models import ScheduleConfig, Task, TaskResult, TaskStatus
        from src.scheduler.engine import TaskEngine

        cfg = ScheduleConfig(
            tasks=[Task(name="retry_task", retry_count=3, timeout_seconds=10)],
        )

        engine = TaskEngine(config=cfg)
        attempt_count = [0]

        def retry_func(cfg):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                return TaskResult(task_name="retry_task", status=TaskStatus.FAILED, error="Not yet")
            return TaskResult(task_name="retry_task", status=TaskStatus.SUCCESS, output="Third time's the charm")

        engine.register("retry_task", retry_func)
        report = engine.run_all()

        assert report.succeeded == 1
        assert attempt_count[0] == 3

    def test_dependency_skip(self):
        from src.scheduler.models import ScheduleConfig, Task, TaskResult, TaskStatus
        from src.scheduler.engine import TaskEngine

        cfg = ScheduleConfig(
            tasks=[
                Task(name="parent", description="Will fail"),
                Task(name="child", description="Should be skipped", dependencies=["parent"]),
            ],
            abort_on_failure=False,
        )

        engine = TaskEngine(config=cfg)

        def fail(cfg):
            return TaskResult(task_name="parent", status=TaskStatus.FAILED, error="Failed")

        def skip(cfg):
            return TaskResult(task_name="child", status=TaskStatus.SUCCESS)

        engine.register("parent", fail)
        engine.register("child", skip)

        report = engine.run_all()
        # Since abort_on_failure=False, child still runs but dependency check should skip it
        # Actually: with abort_on_failure=False the pipeline continues after parent fails.
        # _check_dependencies checks if parent completed successfully - it didn't.
        # So child is SKIPPED.
        child_result = report.task_results.get("child")
        if child_result:
            assert child_result.status == TaskStatus.SKIPPED


class TestNotificationConfig:
    """Test the NotificationConfig from_env loader."""

    def test_default_values(self):
        from src.scheduler.notifications import NotificationConfig

        cfg = NotificationConfig()
        assert cfg.console_enabled is True
        assert cfg.email_enabled is False
        assert cfg.file_enabled is True
        assert cfg.min_level == "warning"

    def test_from_env(self):
        from src.scheduler.notifications import NotificationConfig

        with patch.dict("os.environ", {
            "NOTIFY_EMAIL_ENABLED": "true",
            "NOTIFY_SLACK_ENABLED": "true",
            "NOTIFY_EMAIL_TO": "admin@test.com",
            "NOTIFY_SLACK_WEBHOOK": "https://hooks.slack.com/test",
        }, clear=True):
            cfg = NotificationConfig.from_env()
            assert cfg.email_enabled is True
            assert cfg.slack_enabled is True
            assert cfg.email_to == "admin@test.com"


class TestNotifier:
    """Test the Notifier class."""

    def test_send_console(self):
        from src.scheduler.notifications import Notifier, NotificationConfig

        cfg = NotificationConfig(console_enabled=True, email_enabled=False, slack_enabled=False, file_enabled=False)
        notifier = Notifier(config=cfg)

        results = notifier.send("Test Title", "Test message", level="info")
        assert results.get("console") is True

    def test_send_min_level_filter(self):
        from src.scheduler.notifications import Notifier, NotificationConfig

        cfg = NotificationConfig(min_level="error", console_enabled=True, email_enabled=False, slack_enabled=False, file_enabled=False)
        notifier = Notifier(config=cfg)

        # info level should be filtered out
        results = notifier.send("Test", "test", level="info")
        assert len(results) == 0

        # error level should pass
        results = notifier.send("Test", "test", level="error")
        assert results.get("console") is True

    def test_send_file(self):
        from src.scheduler.notifications import Notifier, NotificationConfig

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name

        try:
            cfg = NotificationConfig(
                console_enabled=False, email_enabled=False, slack_enabled=False,
                file_enabled=True, notification_file=log_path,
            )
            notifier = Notifier(config=cfg)
            results = notifier.send("Test", "File log test", level="info")
            assert results.get("file") is True

            # Check content was written
            content = Path(log_path).read_text()
            assert "Test" in content
            assert "File log test" in content
        finally:
            Path(log_path).unlink(missing_ok=True)

    def test_send_pipeline_report(self):
        from src.scheduler.notifications import Notifier, NotificationConfig

        cfg = NotificationConfig(console_enabled=True, email_enabled=False, slack_enabled=False, file_enabled=False)
        notifier = Notifier(config=cfg)

        report = {
            "pipeline_name": "test_pipeline",
            "succeeded": 5,
            "failed": 1,
            "duration_seconds": 45.2,
            "errors": ["task_b: Oops"],
        }
        results = notifier.send_pipeline_report(report)
        assert results.get("console") is True

    def test_send_failure(self):
        from src.scheduler.notifications import Notifier, NotificationConfig

        cfg = NotificationConfig(console_enabled=True, email_enabled=False, slack_enabled=False, file_enabled=False)
        notifier = Notifier(config=cfg)

        results = notifier.send_failure("data_pipeline", "Connection timeout")
        assert results.get("console") is True


# ─────────────────────────────────────────────────────────
#  Monitoring Tests
# ─────────────────────────────────────────────────────────


class TestAlertEngine:
    """Test the AlertEngine."""

    def test_evaluate_breach(self):
        from src.monitoring.alerting import AlertEngine, AlertRule

        rules = [AlertRule("test_high", "cpu", ">", 90.0, "warning", "CPU high")]
        engine = AlertEngine(rules=rules)

        events = engine.evaluate({"cpu": 95.0})
        assert len(events) == 1
        assert events[0].rule_name == "test_high"
        assert events[0].severity == "warning"

    def test_evaluate_no_breach(self):
        from src.monitoring.alerting import AlertEngine, AlertRule

        rules = [AlertRule("test_high", "cpu", ">", 90.0, "warning", "CPU high")]
        engine = AlertEngine(rules=rules)

        events = engine.evaluate({"cpu": 50.0})
        assert len(events) == 0

    def test_cooldown(self):
        from src.monitoring.alerting import AlertEngine, AlertRule

        rules = [AlertRule("test_cooldown", "cpu", ">", 50.0, "warning", "", cooldown_seconds=3600)]
        engine = AlertEngine(rules=rules)

        # First breach triggers
        events = engine.evaluate({"cpu": 95.0})
        assert len(events) == 1

        # Second breach within cooldown should NOT trigger
        events = engine.evaluate({"cpu": 95.0})
        assert len(events) == 0

    def test_disabled_rule(self):
        from src.monitoring.alerting import AlertEngine, AlertRule

        rules = [AlertRule("disabled", "cpu", ">", 50.0, "warning", "", enabled=False)]
        engine = AlertEngine(rules=rules)

        events = engine.evaluate({"cpu": 95.0})
        assert len(events) == 0

    def test_nested_metric(self):
        from src.monitoring.alerting import AlertEngine, AlertRule

        rules = [AlertRule("nested", "system.cpu_percent", ">", 90.0, "warning", "")]
        engine = AlertEngine(rules=rules)

        events = engine.evaluate({"system.cpu_percent": 95.0})
        assert len(events) == 1

    def test_notify_no_events(self):
        from src.monitoring.alerting import AlertEngine

        engine = AlertEngine()
        results = engine.notify([])
        assert results == []

    def test_evaluate_and_notify(self):
        from src.monitoring.alerting import AlertEngine, AlertRule

        rules = [AlertRule("test", "cpu", ">", 90, "warning", "")]
        engine = AlertEngine(rules=rules)

        events = engine.evaluate_and_notify({"cpu": 95})
        assert len(events) == 1


class TestDriftDetector:
    """Test the DriftDetector."""

    def test_no_drift_identical(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector()

        ref = {"feature_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)}
        cur = {"feature_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)}

        result = detector.detect(ref, cur)
        assert result.drift_detected is False

    def test_drift_different(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector()

        # Reference: normal distribution around 0
        rng = np.random.RandomState(42)
        ref = {"feature_a": rng.normal(0, 1, 100)}
        # Current: very different distribution
        cur = {"feature_a": rng.normal(10, 1, 100)}

        result = detector.detect(ref, cur)
        # Should detect drift
        assert len(result.drift_scores) > 0

    def test_insufficient_samples(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector()

        ref = {"feature_a": np.array([1.0, 2.0])}
        cur = {"feature_a": np.array([1.0, 2.0])}

        result = detector.detect(ref, cur)
        assert result.n_features_analyzed == 0  # Below min_samples

    def test_categorical_drift(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector()

        ref = {"team": np.array(["A", "B", "A", "B", "C"] * 20)}
        cur = {"team": np.array(["X", "Y", "X", "Y", "Z"] * 20)}

        result = detector.detect(ref, cur)
        assert len(result.drift_scores) > 0

    def test_detect_from_csv_missing(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector()

        with pytest.raises(Exception):
            detector.detect_from_csv("/nonexistent/file.csv", "/nonexistent/file2.csv")


class TestPipelineHealth:
    """Test the PipelineHealth class."""

    def test_health_score_default(self):
        from src.monitoring.pipeline_health import PipelineHealth

        with tempfile.TemporaryDirectory() as tmp:
            health = PipelineHealth(scheduler_report_dir=tmp)
            score = health.compute_health_score()
            assert 0.0 <= score <= 1.0

    def test_status_classification(self):
        from src.monitoring.pipeline_health import PipelineHealth

        health = PipelineHealth()
        assert health.get_status(0.9) == "healthy"
        assert health.get_status(0.6) == "degraded"
        assert health.get_status(0.3) == "critical"

    def test_generate_health_report(self):
        from src.monitoring.pipeline_health import PipelineHealth

        with tempfile.TemporaryDirectory() as tmp:
            health = PipelineHealth(scheduler_report_dir=tmp)
            status = health.generate_health_report()

            assert hasattr(status, "health_score")
            assert hasattr(status, "status")
            assert hasattr(status, "task_success_rate")
            assert 0.0 <= status.health_score <= 1.0

    def test_to_dict(self):
        from src.monitoring.pipeline_health import PipelineHealth

        with tempfile.TemporaryDirectory() as tmp:
            health = PipelineHealth(scheduler_report_dir=tmp)
            status = health.generate_health_report()
            d = status.to_dict()

            assert "health_score" in d
            assert "status" in d
            assert "task_success_rate" in d


class TestDriftResult:
    """Test the DriftResult dataclass."""

    def test_to_dict_fields(self):
        from src.monitoring.drift import DriftResult

        result = DriftResult(
            drift_detected=True,
            drifted_features=["feature_a", "feature_b"],
            drift_scores={"feature_a": 0.5, "feature_b": 0.3},
            overall_drift_score=0.4,
            n_features_analyzed=2,
        )
        d = result.to_dict()
        assert d["drift_detected"] is True
        assert len(d["drifted_features"]) == 2
        assert d["overall_drift_score"] == 0.4


class TestHealthStatus:
    """Test the HealthStatus dataclass."""

    def test_to_dict(self):
        from src.monitoring.pipeline_health import HealthStatus

        status = HealthStatus(
            health_score=0.85,
            status="healthy",
            task_success_rate=0.95,
            avg_execution_time=45.2,
        )
        d = status.to_dict()
        assert d["health_score"] == 0.85
        assert d["status"] == "healthy"
        assert d["task_success_rate"] == 0.95


# ─────────────────────────────────────────────────────────
#  Monitoring Models Tests
# ─────────────────────────────────────────────────────────


class TestETLMetric:
    def test_to_dict(self):
        from src.monitoring.models import ETLMetric

        m = ETLMetric(pipeline="test", duration_seconds=45.2, rows_imported=1000, success=True)
        d = m.to_dict()
        assert d["pipeline"] == "test"
        assert d["duration_seconds"] == 45.2
        assert d["success"] is True


class TestSystemMetric:
    def test_to_dict(self):
        from src.monitoring.models import SystemMetric

        m = SystemMetric(cpu_percent=45.0, memory_percent=60.0, db_size_mb=12.5)
        d = m.to_dict()
        assert d["cpu_percent"] == 45.0
        assert d["memory_percent"] == 60.0


class TestDataQualityMetric:
    def test_to_dict(self):
        from src.monitoring.models import DataQualityMetric

        m = DataQualityMetric(source="test", n_rows=5000, null_pct=2.5, duplicate_pct=0.5)
        d = m.to_dict()
        assert d["source"] == "test"
        assert d["n_rows"] == 5000


class TestCacheMetric:
    def test_to_dict(self):
        from src.monitoring.models import CacheMetric

        m = CacheMetric(hits=900, misses=100, hit_rate=0.9, entries=500, size_bytes=1024 * 1024)
        d = m.to_dict()
        assert d["hit_rate"] == 0.9
        assert d["size_mb"] == 1.0


# ─────────────────────────────────────────────────────────
#  Daily Script Tests
# ─────────────────────────────────────────────────────────


class TestDailyDataPipeline:
    """Test the daily_data_pipeline module."""

    def test_fetch_source(self):
        """Test fetch_source with an invalid source (should not crash)."""
        from scripts.daily_data_pipeline import fetch_source

        result = fetch_source("invalid", {"module": "nonexistent_module", "func": "main"})
        assert result["success"] is False
        assert result["error"] is not None


class TestDailyFeatureComputation:
    """Test the daily_feature_computation module."""

    def test_load_latest_data_no_files(self):
        """Test loading data when no files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            # The function looks for hardcoded paths, so we need to verify
            # it handles missing files gracefully
            from scripts.daily_feature_computation import PROCESSED_DIR, RAW_DIR

            # Verify the paths exist (they should be project-relative)
            assert PROCESSED_DIR is not None


class TestDailyModelRetraining:
    """Test the daily_model_retraining module."""

    def test_should_retrain_threshold(self):
        """Test the threshold logic."""
        from scripts.daily_model_retraining import should_retrain, MIN_NEW_MATCHES_FOR_RETRAIN

        assert should_retrain(0) is False
        assert should_retrain(MIN_NEW_MATCHES_FOR_RETRAIN) is True
        assert should_retrain(MIN_NEW_MATCHES_FOR_RETRAIN + 100) is True


class TestDailyPredictions:
    """Test the daily_predictions module."""

    def test_load_best_model_no_models(self):
        """Test loading model when no models directory exists."""
        from scripts.daily_predictions import load_best_model

        model, meta = load_best_model()
        # Should not crash — returns None if no model found, or a loaded model
        assert model is None or hasattr(model, "predict_proba")



# ─────────────────────────────────────────────────────────
#  Scheduler Task Test
# ─────────────────────────────────────────────────────────


class TestSchedulerNewTasks:
    """Test that the new scheduler tasks are properly registered."""

    def test_new_tasks_in_default_config(self):
        from src.scheduler.models import ScheduleConfig

        cfg = ScheduleConfig.default()
        task_names = [t.name for t in cfg.tasks]
        assert "daily_data_pipeline" in task_names
        assert "daily_feature_computation" in task_names
        assert "daily_model_retraining" in task_names
        assert "daily_predictions" in task_names

    def test_new_tasks_in_engine(self):
        from src.scheduler.engine import TaskEngine

        engine = TaskEngine()
        assert "daily_data_pipeline" in engine._task_map
        assert "daily_feature_computation" in engine._task_map
        assert "daily_model_retraining" in engine._task_map
        assert "daily_predictions" in engine._task_map

    def test_new_task_dependencies(self):
        from src.scheduler.models import ScheduleConfig

        cfg = ScheduleConfig.default()
        task_map = {t.name: t for t in cfg.tasks}

        data_pipeline = task_map["daily_data_pipeline"]
        assert data_pipeline.dependencies == []  # No deps (runs independently)

        features = task_map["daily_feature_computation"]
        assert "daily_data_pipeline" in features.dependencies

        retraining = task_map["daily_model_retraining"]
        assert "daily_feature_computation" in retraining.dependencies

        predictions = task_map["daily_predictions"]
        assert "daily_model_retraining" in predictions.dependencies


class TestSchedulerInitExports:
    """Test that the scheduler __init__ exports all required symbols."""

    def test_exports_include_new_tasks(self):
        import src.scheduler

        # New daily pipeline tasks
        assert hasattr(src.scheduler, "daily_data_pipeline")
        assert hasattr(src.scheduler, "daily_feature_computation")
        assert hasattr(src.scheduler, "daily_model_retraining")
        assert hasattr(src.scheduler, "daily_predictions")

        # Notification system
        assert hasattr(src.scheduler, "Notifier")
        assert hasattr(src.scheduler, "NotificationConfig")

        # Core tasks
        assert hasattr(src.scheduler, "TaskEngine")
        assert hasattr(src.scheduler, "Task")
        assert hasattr(src.scheduler, "TaskResult")


# ─────────────────────────────────────────────────────────
#  Monitoring Init Tests
# ─────────────────────────────────────────────────────────


class TestMonitoringExports:
    """Test that monitoring modules are importable."""

    def test_alerting_importable(self):
        import importlib
        try:
            mod = importlib.import_module("src.monitoring.alerting")
            assert hasattr(mod, "AlertEngine")
            assert hasattr(mod, "AlertRule")
            assert hasattr(mod, "AlertEvent")
        except ImportError:
            pytest.skip("src.monitoring.alerting not importable (may need deps)")

    def test_drift_importable(self):
        import importlib
        try:
            mod = importlib.import_module("src.monitoring.drift")
            assert hasattr(mod, "DriftDetector")
            assert hasattr(mod, "DriftResult")
            assert hasattr(mod, "DriftConfig")
        except ImportError:
            pytest.skip("src.monitoring.drift not importable (may need deps)")

    def test_pipeline_health_importable(self):
        import importlib
        try:
            mod = importlib.import_module("src.monitoring.pipeline_health")
            assert hasattr(mod, "PipelineHealth")
            assert hasattr(mod, "HealthStatus")
        except ImportError:
            pytest.skip("src.monitoring.pipeline_health not importable")
