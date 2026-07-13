"""
Tests for the Feature Engineering Framework.

Covers:
- FeatureTransformer base class lifecycle
- FeatureConfig loading and validation
- FeaturePluginRegistry registration
- FeaturePipeline DAG construction
- PipelineReport formatting
- ParallelComputer execution
- Custom exceptions
- Duplicate name detection
- Dependency cycle detection
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.feature_framework import (
    FeatureConfig,
    FeaturePipeline,
    FeaturePluginRegistry,
    FeatureTransformer,
    PipelineReport,
    FeatureDependencyCycleError,
    FeatureConfigError,
    FeatureNotFoundError,
    FeatureComputationError,
)
from src.feature_framework.models import (
    ComputationResult,
    FeatureMetadata,
    TransformContext,
)
from src.feature_framework.parallel import ParallelComputer, make_thread_pool


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


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
    """Test transformer — adds 1 to a column."""
    name = "add_one"
    version = 1
    description = "Adds 1 to the value column"
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


class ValidationTransformer(FeatureTransformer):
    """Test transformer — validates output."""
    name = "validation"
    version = 1
    description = "Tests validation"
    output_columns = ["validated_col"]
    data_type = "float"
    dependencies = []
    category = "test"

    def transform(self, df: pd.DataFrame, context: TransformContext | None = None) -> pd.DataFrame:
        df["validated_col"] = df["value"] * 1.5
        return df


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})


yaml = pytest.importorskip("yaml", reason="PyYAML required for YAML config tests")


@pytest.fixture
def sample_yaml(tmp_path: Path) -> Path:
    config = {
        "version": "1.0",
        "pipeline": {"default_entity_type": "match", "show_progress": False},
        "features": [
            {
                "name": "double",
                "version": 1,
                "type": "double",
                "category": "test",
                "data_type": "float",
                "output_columns": ["doubled"],
                "dependencies": [],
            },
            {
                "name": "add_one",
                "version": 1,
                "type": "add_one",
                "category": "test",
                "data_type": "float",
                "output_columns": ["plus_one"],
                "dependencies": ["double"],
            },
        ],
    }
    path = tmp_path / "features.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# ═══════════════════════════════════════════════════════════
#  Tests: FeatureTransformer
# ═══════════════════════════════════════════════════════════


class TestFeatureTransformer:
    def test_metadata(self):
        t = DoubleTransformer()
        meta = t.metadata
        assert meta.name == "double"
        assert meta.version == 1
        assert meta.data_type == "float"
        assert meta.computation_time == "fast"
        assert isinstance(meta, FeatureMetadata)

    def test_transform(self, sample_df):
        t = DoubleTransformer()
        result = t.transform(sample_df.copy())
        assert "doubled" in result.columns
        assert result["doubled"].tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_validate_output_passes(self, sample_df):
        t = DoubleTransformer()
        result = t.transform(sample_df.copy())
        errors = t.validate_output(result)
        assert errors == []

    def test_validate_output_missing_column(self):
        t = DoubleTransformer()
        df = pd.DataFrame({"wrong": [1, 2, 3]})
        errors = t.validate_output(df)
        assert len(errors) == 1
        assert "Missing output column: doubled" in errors[0]

    def test_validate_output_wrong_type(self):
        t = ValidationTransformer()
        t.data_type = "int"
        df = pd.DataFrame({"validated_col": [1.5, 2.5]})
        errors = t.validate_output(df)
        assert len(errors) > 0

    def test_validate_input_default(self):
        t = DoubleTransformer()
        errors = t.validate_input(pd.DataFrame())
        assert errors == []

    def test_lifecycle_init(self):
        t = DoubleTransformer()
        assert t._initialized is False
        t.init()
        assert t._initialized is True

    def test_to_dict(self):
        t = DoubleTransformer()
        d = t.to_dict()
        assert d["name"] == "double"
        assert d["version"] == 1
        assert d["data_type"] == "float"

    def test_repr(self):
        t = DoubleTransformer()
        assert repr(t) == "<FeatureTransformer 'double' v1>"


# ═══════════════════════════════════════════════════════════
#  Tests: FeatureConfig
# ═══════════════════════════════════════════════════════════


class TestFeatureConfig:
    def test_load_yaml(self, sample_yaml):
        cfg = FeatureConfig(sample_yaml)
        assert len(cfg.features) == 2
        assert cfg.feature_names == ["double", "add_one"]

    def test_load_json(self, tmp_path):
        config = {
            "features": [
                {"name": "test_feat", "type": "test", "category": "test"},
            ]
        }
        path = tmp_path / "features.json"
        with open(path, "w") as f:
            json.dump(config, f)
        cfg = FeatureConfig(path)
        assert len(cfg.features) == 1
        assert cfg.features[0]["name"] == "test_feat"

    def test_missing_file(self):
        with pytest.raises(FeatureConfigError, match="not found"):
            FeatureConfig("/nonexistent/path.yaml")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(FeatureConfigError, match="Invalid JSON"):
            FeatureConfig(path)

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("test")
        with pytest.raises(FeatureConfigError, match="Unsupported file format"):
            FeatureConfig(path)

    def test_duplicate_feature_names(self, tmp_path):
        config = {
            "features": [
                {"name": "dup", "type": "test", "category": "test"},
                {"name": "dup", "type": "test", "category": "test"},
            ]
        }
        path = tmp_path / "dup.yaml"
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(FeatureConfigError, match="Duplicate feature name"):
            FeatureConfig(path)

    def test_missing_required_fields(self, tmp_path):
        config = {"features": [{"name": "only_name"}]}
        path = tmp_path / "missing.yaml"
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(FeatureConfigError, match="Missing required field"):
            FeatureConfig(path)

    def test_invalid_data_type(self, tmp_path):
        config = {
            "features": [
                {"name": "bad_type", "type": "test", "category": "test",
                 "data_type": "complex128"},
            ]
        }
        path = tmp_path / "bad_type.yaml"
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f)
        with pytest.raises(FeatureConfigError, match="Invalid data_type"):
            FeatureConfig(path)

    def test_pipeline_config(self, sample_yaml):
        cfg = FeatureConfig(sample_yaml)
        assert cfg.pipeline_config["default_entity_type"] == "match"
        assert cfg.pipeline_config["show_progress"] is False

    def test_get_by_type(self, sample_yaml):
        cfg = FeatureConfig(sample_yaml)
        doubles = cfg.get_by_type("double")
        assert len(doubles) == 1
        assert doubles[0]["name"] == "double"

    def test_get_by_category(self, sample_yaml):
        cfg = FeatureConfig(sample_yaml)
        tests = cfg.get_by_category("test")
        assert len(tests) == 2

    def test_get_existing(self, sample_yaml):
        cfg = FeatureConfig(sample_yaml)
        feat = cfg.get("double")
        assert feat is not None
        assert feat["name"] == "double"

    def test_get_missing(self, sample_yaml):
        cfg = FeatureConfig(sample_yaml)
        assert cfg.get("nonexistent") is None

    def test_get_enabled(self, tmp_path):
        config = {
            "features": [
                {"name": "enabled", "type": "test", "category": "test"},
                {"name": "disabled", "type": "test", "category": "test", "enabled": False},
            ]
        }
        path = tmp_path / "enabled.yaml"
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f)
        cfg = FeatureConfig(path)
        assert len(cfg.get_enabled()) == 1


# ═══════════════════════════════════════════════════════════
#  Tests: FeaturePluginRegistry
# ═══════════════════════════════════════════════════════════


class TestFeaturePluginRegistry:
    def test_register_class(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        registry.register(DoubleTransformer)
        assert "double" in registry.list_types()
        assert len(registry) == 1

    def test_get_instance(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        registry.register(DoubleTransformer)
        instance = registry.get("double")
        assert instance is not None
        assert isinstance(instance, DoubleTransformer)
        # Same instance from cache
        assert registry.get("double") is instance

    def test_get_or_create_with_params(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        registry.register(DoubleTransformer)
        instance = registry.get_or_create("double", custom_param=42)
        assert instance is not None
        assert instance.params.get("custom_param") == 42

    def test_get_missing(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        assert registry.get("nonexistent") is None

    def test_unregister(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        registry.register(DoubleTransformer)
        assert registry.unregister("double") is True
        assert registry.get("double") is None

    def test_unregister_missing(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        assert registry.unregister("nonexistent") is False

    def test_register_non_transformer(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        with pytest.raises(TypeError):
            registry.register(int)  # type: ignore

    def test_registry_wraps_multiple(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        registry.register(DoubleTransformer)
        registry.register(AddOneTransformer)
        assert len(registry) == 2

    def test_list_instances(self):
        registry = FeaturePluginRegistry(auto_discover=False)
        registry.register(DoubleTransformer)
        registry.get("double")
        instances = registry.list_instances()
        assert "double" in instances


# ═══════════════════════════════════════════════════════════
#  Tests: FeaturePipeline
# ═══════════════════════════════════════════════════════════


class TestFeaturePipeline:
    def test_dataframe_mode(self):
        pipeline = FeaturePipeline(config_dict={
            "features": [
                {"name": "double", "type": "double", "category": "test",
                 "output_columns": ["doubled"]},
            ]
        }, show_progress=False)
        pipeline.plugins.register(DoubleTransformer)

        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        report = pipeline.run(entity_type="dataframe", df=df, trigger="test")
        assert report.success
        assert report.n_computed == 1
        assert report.n_failed == 0

    def test_dataframe_mode_with_simple_pipeline(self, sample_df):
        pipeline = FeaturePipeline(
            plugin_registry=FeaturePluginRegistry(auto_discover=False),
            show_progress=False,
        )
        pipeline.plugins.register(DoubleTransformer)

        report = pipeline.run(
            entity_type="dataframe",
            df=sample_df.copy(),
            trigger="test",
        )
        # No features configured in pipeline without config
        assert report.n_features == 0

    def test_dataframe_mode_multiple_features(self, sample_df):
        pipeline = FeaturePipeline(config_dict={
            "features": [
                {"name": "double", "type": "double", "category": "test",
                 "output_columns": ["doubled"], "dependencies": []},
                {"name": "add_one", "type": "add_one", "category": "test",
                 "output_columns": ["plus_one"], "dependencies": ["double"]},
            ]
        }, show_progress=False)
        pipeline.plugins.register(DoubleTransformer)
        pipeline.plugins.register(AddOneTransformer)

        report = pipeline.run(entity_type="dataframe", df=sample_df.copy(), trigger="test")
        assert report.success
        assert report.n_computed == 2
        assert report.n_failed == 0

    def test_dataframe_mode_failure(self, sample_df):
        pipeline = FeaturePipeline(config_dict={
            "features": [
                {"name": "failing", "type": "failing", "category": "test",
                 "output_columns": ["broken"], "dependencies": []},
            ]
        }, show_progress=False)
        pipeline.plugins.register(FailingTransformer)

        report = pipeline.run(entity_type="dataframe", df=sample_df.copy(), trigger="test")
        assert not report.success
        assert report.n_failed == 1

    def test_empty_config(self):
        pipeline = FeaturePipeline(show_progress=False)
        report = pipeline.run(entity_type="dataframe", trigger="test")
        assert report.success
        assert report.n_features == 0

    def test_yaml_config(self, sample_yaml, sample_df):
        pipeline = FeaturePipeline(config_path=sample_yaml, show_progress=False)
        pipeline.plugins.register(DoubleTransformer)
        pipeline.plugins.register(AddOneTransformer)

        report = pipeline.run(entity_type="dataframe", df=sample_df.copy(), trigger="test")
        assert report.success
        assert report.n_computed == 2

    def test_dag_building(self):
        pipeline = FeaturePipeline(show_progress=False)
        pipeline._transformers = {
            "add_one": AddOneTransformer(),
            "double": DoubleTransformer(),
        }
        dag = pipeline._build_dag(pipeline._transformers)
        assert "double" in dag
        assert "add_one" in dag
        assert dag["add_one"] == ["double"]
        assert dag["double"] == []

    def test_topological_sort(self):
        pipeline = FeaturePipeline(show_progress=False)
        dag = {"add_one": ["double"], "double": []}
        sorted_names = pipeline._topological_sort(dag)
        # double must come before add_one
        assert sorted_names.index("double") < sorted_names.index("add_one")

    def test_cycle_detection(self):
        pipeline = FeaturePipeline(show_progress=False)
        dag = {"a": ["b"], "b": ["c"], "c": ["a"]}
        with pytest.raises(FeatureDependencyCycleError):
            pipeline._topological_sort(dag)

    def test_register_transformer_class(self):
        pipeline = FeaturePipeline(show_progress=False)
        pipeline.register_transformer_class(DoubleTransformer)
        assert "double" in pipeline.plugins.list_types()

    def test_register_transformer_instance(self):
        pipeline = FeaturePipeline(show_progress=False)
        t = DoubleTransformer()
        pipeline.register_transformer(t)
        assert pipeline._transformers["double"] is t

    def test_print_dag(self, capsys):
        pipeline = FeaturePipeline(show_progress=False)
        pipeline.print_dag()  # No DAG yet
        captured = capsys.readouterr()
        assert "No DAG built yet" in captured.out

        pipeline._dag = {"feat1": [], "feat2": ["feat1"]}
        pipeline.print_dag()
        captured = capsys.readouterr()
        assert "FEATURE DEPENDENCY DAG" in captured.out


# ═══════════════════════════════════════════════════════════
#  Tests: PipelineReport
# ═══════════════════════════════════════════════════════════


class TestPipelineReport:
    def test_empty_report(self):
        report = PipelineReport()
        assert report.success
        assert report.n_features == 0
        assert report.total_duration == 0.0

    def test_to_dict(self):
        report = PipelineReport(
            success=True,
            n_features=5,
            n_computed=4,
            n_skipped=1,
            n_failed=0,
            trigger="manual",
        )
        d = report.to_dict()
        assert d["success"] is True
        assert d["n_features"] == 5
        assert d["trigger"] == "manual"

    def test_failed_report_to_dict(self):
        report = PipelineReport(
            success=False,
            n_features=3,
            n_failed=2,
            errors=["Feature X failed", "Feature Y failed"],
        )
        d = report.to_dict()
        assert d["success"] is False
        assert len(d["errors"]) == 2

    def test_print_summary(self, capsys):
        report = PipelineReport(
            success=True,
            n_features=3,
            n_computed=3,
            n_entities=100,
            total_duration=12.5,
            trigger="scheduled",
        )
        report.print_summary()
        captured = capsys.readouterr()
        assert "FEATURE PIPELINE REPORT" in captured.out
        assert "SUCCESS" in captured.out


# ═══════════════════════════════════════════════════════════
#  Tests: ComputationResult
# ═══════════════════════════════════════════════════════════


class TestComputationResult:
    def test_success_result(self):
        r = ComputationResult(
            feature_name="test",
            entity_id=42,
            entity_type="match",
            value=0.75,
            success=True,
            duration_seconds=0.1,
        )
        d = r.to_dict()
        assert d["feature_name"] == "test"
        assert d["value"] == 0.75

    def test_failed_result(self):
        r = ComputationResult(
            feature_name="test",
            entity_id=99,
            entity_type="team",
            success=False,
            error="Something broke",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "Something broke"

    def test_default_values(self):
        r = ComputationResult(feature_name="x", entity_id=1)
        assert r.success is True
        assert r.value is None
        assert r.duration_seconds == 0.0


# ═══════════════════════════════════════════════════════════
#  Tests: TransformContext
# ═══════════════════════════════════════════════════════════


class TestTransformContext:
    def test_defaults(self):
        ctx = TransformContext()
        assert ctx.entity_type == "match"
        assert ctx.trigger == "manual"
        assert ctx.entity_ids == []

    def test_custom(self):
        ctx = TransformContext(
            entity_type="team",
            entity_ids=[1, 2, 3],
            trigger="scheduled",
        )
        assert ctx.entity_type == "team"
        assert len(ctx.entity_ids) == 3

    def test_elapsed(self):
        ctx = TransformContext()
        elapsed = ctx.elapsed_seconds()
        assert elapsed >= 0.0


# ═══════════════════════════════════════════════════════════
#  Tests: ParallelComputer
# ═══════════════════════════════════════════════════════════


class TestParallelComputer:
    def test_empty_ids(self):
        def compute(eid: int) -> ComputationResult:
            return ComputationResult(feature_name="test", entity_id=eid)
        pc = ParallelComputer(compute, show_progress=False)
        results = pc.run([])
        assert results == []

    def test_single_entity(self):
        def compute(eid: int) -> ComputationResult:
            return ComputationResult(feature_name="test", entity_id=eid, value=float(eid))

        pc = ParallelComputer(compute, show_progress=False)
        results = pc.run([42])
        assert len(results) == 1
        assert results[0].entity_id == 42
        assert results[0].value == 42.0

    def test_multiple_entities(self):
        def compute(eid: int) -> ComputationResult:
            return ComputationResult(feature_name="test", entity_id=eid, value=float(eid * 2))

        pc = ParallelComputer(compute, show_progress=False)
        results = pc.run([1, 2, 3, 4, 5])
        assert len(results) == 5
        assert results[0].value == 2.0
        assert results[4].value == 10.0

    def test_entity_order_preserved(self):
        """Results should maintain input order."""
        def compute(eid: int) -> ComputationResult:
            return ComputationResult(feature_name="test", entity_id=eid)

        pc = ParallelComputer(compute, show_progress=False)
        ids = [5, 3, 1, 4, 2]
        results = pc.run(ids)
        result_ids = [r.entity_id for r in results]
        assert result_ids == ids

    def test_handles_failures(self):
        def compute(eid: int) -> ComputationResult:
            if eid == 3:
                raise ValueError("Intentional failure")
            return ComputationResult(feature_name="test", entity_id=eid, value=float(eid))

        pc = ParallelComputer(compute, show_progress=False)
        results = pc.run([1, 2, 3, 4, 5])
        assert len(results) == 5
        failures = [r for r in results if not r.success]
        assert len(failures) == 1
        assert failures[0].entity_id == 3

    def test_batch_processing(self):
        def compute(eid: int) -> ComputationResult:
            return ComputationResult(feature_name="test", entity_id=eid, value=float(eid))

        pc = ParallelComputer(compute, show_progress=False)
        results = pc.run_batch([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], batch_size=3)
        assert len(results) == 10

    def test_make_thread_pool(self):
        pool = make_thread_pool(max_workers=2)
        assert pool._max_workers == 2
        pool.shutdown()


# ═══════════════════════════════════════════════════════════
#  Tests: Exceptions
# ═══════════════════════════════════════════════════════════


class TestExceptions:
    def test_feature_computation_error(self):
        exc = FeatureComputationError("test_feat", 42, "Something broke")
        assert "test_feat" in str(exc)
        assert "42" in str(exc)
        assert exc.feature_name == "test_feat"
        assert exc.entity_id == 42

    def test_feature_not_found_error(self):
        exc = FeatureNotFoundError("missing_feat")
        assert "missing_feat" in str(exc)
        exc2 = FeatureNotFoundError("missing_feat", version=2)
        assert "v2" in str(exc2)

    def test_feature_dependency_cycle_error(self):
        exc = FeatureDependencyCycleError(["a", "b", "c"])
        assert "Circular dependency" in str(exc)
        assert "a" in str(exc)

    def test_feature_config_error(self):
        exc = FeatureConfigError("Bad config", "/path/to/config.yaml")
        assert "Bad config" in str(exc)
        assert "/path/to/config.yaml" in str(exc)
