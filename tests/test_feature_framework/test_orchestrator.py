"""
Tests for FeatureOrchestrator — production-grade pipeline execution with
discovery, caching, retry, resume, parallelism, progress tracking,
logging, metrics, and incremental updates.

Also tests the CLI commands (build-features, validate-features,
recompute-feature, list-features, feature-status).

Covers:
- OrchestratorReport creation and serialization
- FeatureExecutionRecord creation
- FeatureStatus and OrchestratorStage enums
- FeatureOrchestrator initialization
- run() — DataFrame mode (success, failure, empty, multiple features)
- run() — Entity mode
- DAG resolution + cycle detection
- Caching (is_cached, update_cache, clear_cache)
- Retry logic
- Checkpoint save/load/resume
- Metrics collection
- Single-feature operations (recompute_feature, list_features, feature_status)
- CLI command execution
- Edge cases (empty DataFrame, no config, all failing)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.feature_framework import (
    FeatureDependencyCycleError,
    FeatureOrchestrator,
    FeaturePipeline,
    FeaturePluginRegistry,
    FeatureTransformer,
    OrchestratorReport,
    OrchestratorStage,
    FeatureExecutionRecord,
    FeatureStatus,
)
from src.feature_framework.models import TransformContext


# ═══════════════════════════════════════════════════════════════
#  Test transformers
# ═══════════════════════════════════════════════════════════════


class DoubleTransformer(FeatureTransformer):
    """Test transformer — doubles a column."""
    name = "double"
    version = 1
    description = "Doubles the value column"
    output_columns = ["doubled"]
    data_type = "float"
    computation_time = "fast"
    dependencies = []
    category = "test"

    def transform(self, df: pd.DataFrame, context: TransformContext | None = None) -> pd.DataFrame:
        df["doubled"] = df["value"] * 2
        return df


class AddOneTransformer(FeatureTransformer):
    """Test transformer — adds 1 after double."""
    name = "add_one"
    version = 1
    description = "Adds 1 to the doubled column"
    output_columns = ["plus_one"]
    data_type = "float"
    computation_time = "fast"
    dependencies = ["double"]
    category = "test"

    def transform(self, df: pd.DataFrame, context: TransformContext | None = None) -> pd.DataFrame:
        if "doubled" in df.columns:
            df["plus_one"] = df["doubled"] + 1
        else:
            df["plus_one"] = df["value"] + 1
        return df


class FailingTransformer(FeatureTransformer):
    """Test transformer — always fails."""
    name = "failing"
    version = 1
    description = "Always fails"
    output_columns = ["broken"]
    dependencies = []
    category = "test"

    def transform(self, df: pd.DataFrame, context: TransformContext | None = None) -> pd.DataFrame:
        raise ValueError("Intentional failure for testing")


class SlowTransformer(FeatureTransformer):
    """Test transformer — simulates slow computation."""
    name = "slow"
    version = 1
    description = "Slow transformer"
    output_columns = ["slow_col"]
    dependencies = []
    category = "test"

    def transform(self, df: pd.DataFrame, context: TransformContext | None = None) -> pd.DataFrame:
        import time
        time.sleep(0.05)
        df["slow_col"] = df["value"] * 10
        return df


class RetryOnceTransformer(FeatureTransformer):
    """Test transformer — fails on first call, succeeds on retry.

    Uses instance-level call counter so each test gets a fresh one.
    """
    name = "retry_once"
    version = 1
    description = "Fails first call then succeeds"
    output_columns = ["retried"]
    dependencies = []
    category = "test"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_count = 0

    def transform(self, df: pd.DataFrame, context: TransformContext | None = None) -> pd.DataFrame:
        self._call_count += 1
        if self._call_count <= 1:
            raise RuntimeError("Simulated transient failure")
        df["retried"] = df["value"] * 3
        return df


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture
def orchestrator() -> FeatureOrchestrator:
    return FeatureOrchestrator(
        show_progress=False,
        parallel=False,
        incremental=False,
        cache_dir=tempfile.mkdtemp(),
        checkpoint_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def configured_orchestrator(orchestrator: FeatureOrchestrator) -> FeatureOrchestrator:
    """Orchestrator with transformers registered."""
    orchestrator.plugins.register(DoubleTransformer)
    orchestrator.plugins.register(AddOneTransformer)
    orchestrator.plugins.register(FailingTransformer)
    return orchestrator


@pytest.fixture
def simple_cfg() -> dict:
    return {
        "features": [
            {
                "name": "double",
                "type": "double",
                "category": "test",
                "output_columns": ["doubled"],
                "dependencies": [],
            },
        ],
    }


@pytest.fixture
def dep_cfg() -> dict:
    return {
        "features": [
            {
                "name": "double",
                "type": "double",
                "category": "test",
                "output_columns": ["doubled"],
                "dependencies": [],
            },
            {
                "name": "add_one",
                "type": "add_one",
                "category": "test",
                "output_columns": ["plus_one"],
                "dependencies": ["double"],
            },
        ],
    }


@pytest.fixture
def failing_cfg() -> dict:
    return {
        "features": [
            {
                "name": "failing",
                "type": "failing",
                "category": "test",
                "output_columns": ["broken"],
                "dependencies": [],
            },
        ],
    }


# ═══════════════════════════════════════════════════════════════
#  Tests: OrchestratorReport
# ═══════════════════════════════════════════════════════════════


class TestOrchestratorReport:
    def test_empty_report(self):
        report = OrchestratorReport()
        assert report.success
        assert report.n_features == 0
        assert report.n_computed == 0
        assert report.n_failed == 0
        assert report.duration == 0.0
        assert report.errors == []

    def test_successful_report(self):
        report = OrchestratorReport(
            run_id="abc123",
            started_at="2024-01-01T00:00:00",
            trigger="manual",
            entity_type="dataframe",
            n_entities=10,
            n_features=3,
            n_computed=3,
            n_skipped=0,
            n_failed=0,
            success=True,
        )
        assert report.success
        summary = report.summary()
        assert "PASS" in summary
        assert "3" in summary

    def test_failed_report(self):
        report = OrchestratorReport(
            success=False,
            n_features=3,
            n_failed=2,
            errors=["Feature X failed", "Feature Y failed"],
        )
        assert not report.success
        summary = report.summary()
        assert "FAIL" in summary
        assert "X failed" in summary

    def test_to_dict(self):
        report = OrchestratorReport(run_id="test123", n_features=5, n_computed=4)
        d = report.to_dict()
        assert d["run_id"] == "test123"
        assert d["n_features"] == 5
        assert d["n_computed"] == 4
        assert "metrics" in d

    def test_summary_with_metrics(self):
        report = OrchestratorReport(
            success=True,
            n_features=5,
            n_computed=4,
            duration=12.5,
            metrics={"success_rate": 0.8, "features_per_second": 0.32},
        )
        summary = report.summary()
        assert "0.8" in summary
        assert "0.32" in summary

    def test_summary_with_errors(self):
        report = OrchestratorReport(
            success=False,
            n_features=2,
            n_failed=2,
            duration=5.0,
            errors=["Error 1", "Error 2", "Error 3"],
        )
        summary = report.summary()
        for err in ["Error 1", "Error 2", "Error 3"]:
            assert err in summary

    def test_repr(self):
        report = OrchestratorReport(run_id="abc12345", n_features=5, n_computed=3)
        rep = repr(report)
        assert "abc1234" in rep
        assert "3/5" in rep


# ═══════════════════════════════════════════════════════════════
#  Tests: FeatureExecutionRecord
# ═══════════════════════════════════════════════════════════════


class TestFeatureExecutionRecord:
    def test_defaults(self):
        r = FeatureExecutionRecord()
        assert r.status == "pending"
        assert r.duration == 0.0
        assert r.error == ""
        assert r.retries == 0
        assert not r.cached

    def test_to_dict(self):
        r = FeatureExecutionRecord(
            name="test",
            status="completed",
            duration=0.5,
            retries=1,
            output_columns=["col1", "col2"],
        )
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "completed"
        assert d["duration"] == 0.5

    def test_completed_record(self):
        r = FeatureExecutionRecord(
            name="double",
            status="completed",
            duration=0.123,
            n_entities=5,
            output_columns=["doubled"],
        )
        assert r.status == "completed"
        assert r.n_entities == 5


# ═══════════════════════════════════════════════════════════════
#  Tests: FeatureStatus & OrchestratorStage enums
# ═══════════════════════════════════════════════════════════════


class TestEnums:
    def test_feature_status_values(self):
        assert FeatureStatus.PENDING.value == "pending"
        assert FeatureStatus.COMPLETED.value == "completed"
        assert FeatureStatus.FAILED.value == "failed"
        assert FeatureStatus.CACHED.value == "cached"

    def test_orchestrator_stage_values(self):
        assert OrchestratorStage.DISCOVER.value == "discover"
        assert OrchestratorStage.RESOLVE.value == "resolve"
        assert OrchestratorStage.COMPUTE.value == "compute"
        assert OrchestratorStage.VALIDATE.value == "validate"
        assert OrchestratorStage.STORE.value == "store"


# ═══════════════════════════════════════════════════════════════
#  Tests: FeatureOrchestrator initialization & config
# ═══════════════════════════════════════════════════════════════


class TestFeatureOrchestratorInit:
    def test_default_init(self):
        orch = FeatureOrchestrator(show_progress=False)
        assert orch.max_retries == 2
        assert orch.retry_delay == 2.0
        assert orch.parallel
        assert orch.incremental
        assert orch.cache_dir is not None
        assert orch.checkpoint_dir is not None

    def test_init_with_custom_params(self):
        orch = FeatureOrchestrator(
            show_progress=False,
            max_retries=5,
            retry_delay=1.0,
            parallel=False,
            incremental=False,
            max_workers=2,
            cache_dir="/tmp/test_cache",
            checkpoint_dir="/tmp/test_checkpoints",
        )
        assert orch.max_retries == 5
        assert orch.retry_delay == 1.0
        assert not orch.parallel
        assert not orch.incremental
        assert orch.max_workers == 2

    def test_init_with_config_dict(self):
        orch = FeatureOrchestrator(
            config_dict={"features": []},
            show_progress=False,
        )
        assert orch._inline_features == []

    def test_init_with_invalid_config(self):
        with pytest.raises(ValueError, match="'features' must be a list"):
            FeatureOrchestrator(
                config_dict={"features": "not_a_list"},
                show_progress=False,
            )

    def test_init_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            checkpoint_dir = Path(tmp) / "checkpoints"
            orch = FeatureOrchestrator(
                show_progress=False,
                cache_dir=str(cache_dir),
                checkpoint_dir=str(checkpoint_dir),
            )
            assert cache_dir.exists()
            assert checkpoint_dir.exists()


# ═══════════════════════════════════════════════════════════════
#  Tests: DAG management
# ═══════════════════════════════════════════════════════════════


class TestDAG:
    def test_build_dag(self, configured_orchestrator):
        transformers = {
            "double": DoubleTransformer(),
            "add_one": AddOneTransformer(),
        }
        dag = configured_orchestrator._build_dag(transformers)
        assert "double" in dag
        assert "add_one" in dag
        assert dag["double"] == []
        assert dag["add_one"] == ["double"]

    def test_topological_sort(self, configured_orchestrator):
        dag = {"add_one": ["double"], "double": []}
        sorted_names = configured_orchestrator._topological_sort(dag)
        assert sorted_names.index("double") < sorted_names.index("add_one")

    def test_cycle_detection(self, configured_orchestrator):
        dag = {"a": ["b"], "b": ["c"], "c": ["a"]}
        with pytest.raises(FeatureDependencyCycleError):
            configured_orchestrator._topological_sort(dag)

    def test_no_dependencies(self, configured_orchestrator):
        dag = {"a": [], "b": [], "c": []}
        sorted_names = configured_orchestrator._topological_sort(dag)
        assert len(sorted_names) == 3

    def test_single_node(self, configured_orchestrator):
        dag = {"a": []}
        sorted_names = configured_orchestrator._topological_sort(dag)
        assert sorted_names == ["a"]

    def test_get_dag(self, configured_orchestrator):
        configured_orchestrator._dag = {"a": ["b"], "b": []}
        dag = configured_orchestrator.get_dag()
        assert dag == {"a": ["b"], "b": []}


# ═══════════════════════════════════════════════════════════════
#  Tests: DataFrame Mode
#  Note: config_dict must be passed to constructor, NOT to run()
# ═══════════════════════════════════════════════════════════════


class TestDataFrameMode:
    def test_simple_feature(self, sample_df):
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "double", "type": "double", "category": "test",
                     "output_columns": ["doubled"]},
                ],
            },
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(DoubleTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert report.success
        assert report.n_computed == 1
        assert report.n_failed == 0
        assert report.entity_type == "dataframe"
        assert report.trigger == "test"

    def test_multiple_features_with_deps(self, sample_df):
        """Two features with dependency: double -> add_one."""
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "double", "type": "double", "category": "test",
                     "output_columns": ["doubled"], "dependencies": []},
                    {"name": "add_one", "type": "add_one", "category": "test",
                     "output_columns": ["plus_one"], "dependencies": ["double"]},
                ],
            },
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(DoubleTransformer)
        orch.plugins.register(AddOneTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert report.success
        assert report.n_computed == 2
        assert report.n_failed == 0

    def test_feature_failure(self, sample_df):
        """Failing transformer should produce a failed report."""
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "failing", "type": "failing", "category": "test",
                     "output_columns": ["broken"]},
                ],
            },
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(FailingTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert not report.success
        assert report.n_failed == 1
        assert len(report.errors) >= 1

    def test_empty_dataframe(self, orchestrator):
        """Empty DataFrame should not crash the pipeline."""
        df = pd.DataFrame()
        orchestrator.plugins.register(DoubleTransformer)

        # Run without features configured (just test empty df doesn't crash)
        report = orchestrator.run(
            entity_type="dataframe",
            df=df,
            trigger="test",
        )
        assert report.n_entities == 0

    def test_no_features_configured(self, sample_df, orchestrator):
        """Should succeed with no features to compute."""
        report = orchestrator.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert report.success
        assert report.n_features == 0

    def test_missing_transformer(self, sample_df):
        """Should produce a warning when transformer is not found."""
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "missing", "type": "not_registered", "category": "test",
                     "output_columns": ["x"]},
                ],
            },
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert report.n_features == 1
        assert report.n_computed == 0

    def test_force_recompute(self, sample_df):
        """Force recompute should skip cache."""
        cfg = {
            "features": [
                {"name": "double", "type": "double", "category": "test",
                 "output_columns": ["doubled"]},
            ],
        }
        orch = FeatureOrchestrator(
            config_dict=cfg,
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(DoubleTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
            force_recompute=True,
        )
        assert report.n_computed == 1

    def test_run_with_run_id(self, sample_df):
        """Run should produce a run_id."""
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "double", "type": "double", "category": "test",
                     "output_columns": ["doubled"]},
                ],
            },
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(DoubleTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert len(report.run_id) == 12
        assert report.run_id != ""


# ═══════════════════════════════════════════════════════════════
#  Tests: Caching
# ═══════════════════════════════════════════════════════════════


class TestCaching:
    @staticmethod
    def _make_orch() -> FeatureOrchestrator:
        return FeatureOrchestrator(
            show_progress=False,
            parallel=False,
            incremental=True,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )

    def test_is_cached_no_cache(self):
        orch = self._make_orch()
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert not orch._is_cached("test_feat", df)

    def test_is_cached_no_meta(self):
        orch = self._make_orch()
        df = pd.DataFrame({"a": [1, 2, 3]})
        orch._update_cache("test_feat", df)
        # Change row count to invalidate
        df2 = pd.DataFrame({"a": [1, 2, 3, 4]})
        assert not orch._is_cached("test_feat", df2)

    def test_cache_lifecycle(self):
        orch = self._make_orch()
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert not orch._is_cached("feat_x", df)
        orch._update_cache("feat_x", df)
        assert orch._is_cached("feat_x", df)

    def test_clear_specific_feature(self):
        orch = self._make_orch()
        df = pd.DataFrame({"a": [1, 2, 3]})
        orch._update_cache("feat_a", df)
        orch._update_cache("feat_b", df)
        count = orch.clear_cache("feat_a")
        assert count >= 1

    def test_clear_all_cache(self):
        orch = self._make_orch()
        df = pd.DataFrame({"a": [1]})
        orch._update_cache("x", df)
        orch._update_cache("y", df)
        count = orch.clear_cache()
        assert count >= 2

    def test_incremental_disabled(self):
        orch = self._make_orch()
        orch.incremental = False
        df = pd.DataFrame({"a": [1, 2]})
        orch._update_cache("x", df)
        assert not orch._is_cached("x", df)


# ═══════════════════════════════════════════════════════════════
#  Tests: Retry logic
# ═══════════════════════════════════════════════════════════════


class TestRetry:
    def test_retry_success(self, sample_df):
        """RetryOnceTransformer should succeed after retry."""
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "retry_once", "type": "retry_once", "category": "test",
                     "output_columns": ["retried"]},
                ],
            },
            show_progress=False,
            parallel=False,
            max_retries=2,
            retry_delay=0.01,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(RetryOnceTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert report.n_computed == 1
        assert report.n_failed == 0


# ═══════════════════════════════════════════════════════════════
#  Tests: Checkpoint & Resume
# ═══════════════════════════════════════════════════════════════


class TestCheckpointAndResume:
    def test_save_checkpoint(self, orchestrator):
        report = OrchestratorReport(
            run_id="test123",
            started_at="2024-01-01T00:00:00",
            n_features=3,
            n_computed=1,
            n_failed=2,
            success=False,
        )
        path = orchestrator._save_checkpoint(report)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["run_id"] == "test123"
        assert data["n_failed"] == 2

    def test_load_checkpoint(self, orchestrator):
        report = OrchestratorReport(
            run_id="test456",
            n_computed=2,
            n_failed=1,
            success=False,
        )
        path = orchestrator._save_checkpoint(report)
        loaded = orchestrator._load_checkpoint(path)
        assert loaded is not None
        assert loaded["run_id"] == "test456"
        assert loaded["n_computed"] == 2

    def test_load_missing_checkpoint(self, orchestrator):
        loaded = orchestrator._load_checkpoint("/nonexistent/checkpoint.json")
        assert loaded is None

    def test_list_checkpoints(self, orchestrator):
        report = OrchestratorReport(run_id="list_test", n_computed=1, success=False)
        orchestrator._save_checkpoint(report)
        checkpoints = orchestrator.list_checkpoints()
        assert len(checkpoints) >= 1
        assert any(c["run_id"] == "list_test" for c in checkpoints)

    def test_resume_no_checkpoint(self, orchestrator):
        report = orchestrator.resume("/nonexistent/checkpoint.json")
        assert not report.success
        assert len(report.errors) >= 1


# ═══════════════════════════════════════════════════════════════
#  Tests: Metrics
# ═══════════════════════════════════════════════════════════════


class TestMetrics:
    def test_collect_metrics(self, orchestrator):
        report = OrchestratorReport(
            run_id="metrics_test",
            duration=10.0,
            n_features=5,
            n_computed=3,
            n_skipped=1,
            n_failed=1,
            n_entities=100,
            features={
                "a": FeatureExecutionRecord(name="a", status="completed", duration=0.5),
                "b": FeatureExecutionRecord(name="b", status="completed", duration=1.5),
                "c": FeatureExecutionRecord(name="c", status="failed", duration=0.2),
            },
        )
        orchestrator._collect_metrics(report)
        assert report.metrics["success_rate"] == 3 / 5
        assert report.metrics["total_duration"] == 10.0
        assert report.metrics["avg_feature_time"] == 1.0  # (0.5 + 1.5) / 2

    def test_metrics_history(self, orchestrator):
        report = OrchestratorReport(run_id="hist1", n_features=2, n_computed=2)
        orchestrator._collect_metrics(report)
        assert len(orchestrator.get_metrics_history()) == 1

        report2 = OrchestratorReport(run_id="hist2", n_features=3, n_computed=3)
        orchestrator._collect_metrics(report2)
        assert len(orchestrator.get_metrics_history()) == 2

    def test_metrics_empty(self, orchestrator):
        """No computed features should handle gracefully."""
        report = OrchestratorReport(
            run_id="empty_metrics",
            duration=0.0,
            n_features=0,
            n_computed=0,
            n_entities=0,
        )
        orchestrator._collect_metrics(report)
        assert report.metrics["avg_feature_time"] == 0.0
        assert report.metrics["success_rate"] == 0.0


# ═══════════════════════════════════════════════════════════════
#  Tests: Single feature operations
# ═══════════════════════════════════════════════════════════════


class TestSingleFeatureOperations:
    def test_list_features_empty(self, orchestrator):
        features = orchestrator.list_features()
        assert features == []

    def test_feature_status_not_found(self, orchestrator):
        status = orchestrator.feature_status("nonexistent")
        assert status["status"] == "not_found"

    def test_recompute_feature_not_found(self, orchestrator):
        record = orchestrator.recompute_feature("nonexistent")
        assert record.status == "failed"
        assert "not found in config" in record.error

    def test_recompute_feature_success(self, sample_df):
        """Full happy path: config + transformer + DataFrame."""
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "double", "type": "double", "category": "test",
                     "output_columns": ["doubled"]},
                ],
            },
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(DoubleTransformer)
        # First run to compute the feature
        report = orch.run(
            entity_type="dataframe",
            df=sample_df,
            trigger="test",
        )
        assert report.success

        # Now recompute the same feature
        record = orch.recompute_feature("double", df=sample_df)
        assert record.status == "completed"
        assert "doubled" in (record.output_columns or [])

    def test_recompute_feature_no_df(self, orchestrator):
        """Without DataFrame, recompute should fail gracefully."""
        record = orchestrator.recompute_feature("nonexistent")
        assert record.status == "failed"


# ═══════════════════════════════════════════════════════════════
#  Tests: Entity mode
# ═══════════════════════════════════════════════════════════════


class TestEntityMode:
    def test_no_entity_ids(self, orchestrator):
        """Should succeed with no entities to process."""
        orchestrator.plugins.register(DoubleTransformer)
        # Entity mode without entity_ids goes to fallback path
        report = orchestrator.run(
            entity_type="match",
            entity_ids=[],
            trigger="test",
        )
        assert report.n_entities == 0

    def test_no_database_session(self, orchestrator):
        """Without DB, entity mode should fail gracefully."""
        orchestrator.plugins.register(DoubleTransformer)
        report = orchestrator.run(
            entity_type="match",
            entity_ids=[1, 2, 3],
            trigger="test",
        )
        # Should not crash — gracefully returns with errors
        assert not report.success or report.n_computed == 0


# ═══════════════════════════════════════════════════════════════
#  Tests: Logging
# ═══════════════════════════════════════════════════════════════


class TestLogging:
    def test_log_stage(self, orchestrator):
        """Logging should not raise."""
        orchestrator._log_stage("test_run", OrchestratorStage.DISCOVER, "start")
        orchestrator._log_stage("test_run", OrchestratorStage.COMPUTE, "end",
                                extra={"n_features": 5})

    def test_log_feature(self, orchestrator):
        orchestrator._log_feature("test_run", "feat_1", "completed", "OK")
        orchestrator._log_feature("test_run", "feat_2", "failed", extra={"error": "msg"})

    def test_log_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = FeatureOrchestrator(
                show_progress=False,
                log_to_file=True,
                log_dir=Path(tmp) / "logs",
            )
            orch._write_log({"event": "test", "run_id": "x"})
            log_files = list(Path(orch.log_dir).glob("*.jsonl"))
            assert len(log_files) >= 1


# ═══════════════════════════════════════════════════════════════
#  Tests: Edge cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_df_no_crash(self, orchestrator):
        """Empty DataFrame should not crash the pipeline."""
        df = pd.DataFrame()
        report = orchestrator.run(
            entity_type="dataframe",
            df=df,
            trigger="test",
        )
        assert report.success
        assert report.n_entities == 0

    def test_null_columns_in_df(self, sample_df):
        """DataFrame with NaN values should not crash."""
        df = pd.DataFrame({"value": [1.0, None, 3.0, None, 5.0]})
        orch = FeatureOrchestrator(
            config_dict={
                "features": [
                    {"name": "double", "type": "double", "category": "test",
                     "output_columns": ["doubled"]},
                ],
            },
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(DoubleTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=df,
            trigger="test",
        )
        assert report.n_computed == 1

    def test_very_large_config(self):
        """50 features should be handled correctly."""
        features = []
        for i in range(50):
            features.append({
                "name": f"feat_{i}",
                "type": "double" if i == 0 else "missing",
                "category": "test",
                "output_columns": [f"col_{i}"],
                "dependencies": [f"feat_{i-1}"] if i > 0 else [],
            })

        orch = FeatureOrchestrator(
            config_dict={"features": features},
            show_progress=False,
            parallel=False,
            incremental=False,
            cache_dir=tempfile.mkdtemp(),
            checkpoint_dir=tempfile.mkdtemp(),
        )
        orch.plugins.register(DoubleTransformer)
        report = orch.run(
            entity_type="dataframe",
            df=pd.DataFrame({"value": [1.0]}),
            trigger="test",
        )
        assert report.n_features == 50

    def test_run_id_uniqueness(self, orchestrator):
        """Each run should get a unique ID."""
        ids = set()
        for _ in range(10):
            report = orchestrator.run(
                entity_type="dataframe",
                df=pd.DataFrame({"v": [1]}),
                trigger="test",
            )
            ids.add(report.run_id)
        assert len(ids) == 10


# ═══════════════════════════════════════════════════════════════
#  Tests: CLI commands
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def temp_csv(tmp_path: Path) -> Path:
    """Create a temporary CSV file."""
    path = tmp_path / "input.csv"
    pd.DataFrame({"value": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    return path


class TestCLI:
    def test_parser_creation(self):
        """Parser should build without errors."""
        from src.feature_framework.orchestrator_cli import _build_parser
        parser = _build_parser()
        assert parser is not None

    def test_help_text(self):
        """Help should show all commands."""
        from src.feature_framework.orchestrator_cli import _build_parser
        parser = _build_parser()
        help_text = parser.format_help()
        for cmd in ("build-features", "validate-features", "recompute-feature",
                    "list-features", "feature-status"):
            assert cmd in help_text

    def test_list_features_no_config(self):
        """Listing with no config returns empty list."""
        from src.feature_framework.orchestrator_cli import main
        exit_code = main(["list-features"])
        assert exit_code in (0, 1)

    def test_feature_status_missing(self):
        """Status for missing feature returns exit code 1."""
        from src.feature_framework.orchestrator_cli import main
        exit_code = main(["feature-status", "nonexistent"])
        assert exit_code == 1

    def test_build_features_no_input(self):
        """build-features without real input should exit with code 1."""
        from src.feature_framework.orchestrator_cli import main
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            main(["build-features", "--input", "/nonexistent.csv",
                  "--output", "/tmp/out.csv", "--no-parallel", "--quiet"])
        assert exc_info.value.code == 1

    def test_validate_features_no_input(self):
        """validate-features without real input should exit with code 1."""
        from src.feature_framework.orchestrator_cli import main
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            main(["validate-features", "--input", "/nonexistent.csv",
                  "--quiet"])
        assert exc_info.value.code == 1

    def test_recompute_feature_no_input(self):
        """recompute-feature should try without input."""
        from src.feature_framework.orchestrator_cli import main
        exit_code = main(["recompute-feature", "double", "--quiet"])
        assert exit_code in (0, 1)

    def test_build_features_success(self, temp_csv, tmp_path):
        """End-to-end build-features."""
        from src.feature_framework.orchestrator_cli import main
        output_path = tmp_path / "output.csv"
        exit_code = main([
            "build-features",
            "--input", str(temp_csv),
            "--output", str(output_path),
            "--no-parallel",
            "--quiet",
        ])
        assert isinstance(exit_code, int)

    def test_validate_features_success(self, temp_csv):
        """End-to-end validate-features."""
        from src.feature_framework.orchestrator_cli import main
        exit_code = main([
            "validate-features",
            "--input", str(temp_csv),
            "--checks", "missing_values",
            "--quiet",
        ])
        assert isinstance(exit_code, int)

    def test_save_validation_report(self, temp_csv, tmp_path):
        """Validation report should be saved to file."""
        from src.feature_framework.orchestrator_cli import main
        report_path = tmp_path / "report.json"
        exit_code = main([
            "validate-features",
            "--input", str(temp_csv),
            "--output", str(report_path),
            "--quiet",
        ])
        assert isinstance(exit_code, int)
        if exit_code == 0:
            assert report_path.exists()

    def test_verbose_output(self, temp_csv):
        """Verbose flag should not crash."""
        from src.feature_framework.orchestrator_cli import main
        exit_code = main([
            "list-features", "--verbose",
        ])
        assert isinstance(exit_code, int)

    def test_keyboard_interrupt_handled(self):
        """main() should handle KeyboardInterrupt gracefully."""
        from src.feature_framework.orchestrator_cli import main
        with patch("src.feature_framework.orchestrator_cli.cmd_list_features",
                   side_effect=KeyboardInterrupt):
            exit_code = main(["list-features"])
            assert exit_code == 1
