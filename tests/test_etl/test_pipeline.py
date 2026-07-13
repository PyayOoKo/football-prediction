"""
Tests for the ETL Pipeline orchestrator — ETLPipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.etl.pipeline import ETLPipeline
from src.etl.models import (
    ETLConfig,
    PipelineStage,
    StageResult,
    StageStatus,
)
from src.etl.extract import BaseExtractor
from src.etl.store import FileStore


# ═══════════════════════════════════════════════════════════
#  Helper: mock extractor
# ═══════════════════════════════════════════════════════════

class MockExtractor(BaseExtractor):
    """A simple extractor that returns predefined data."""

    def __init__(self, data=None, fail=False):
        super().__init__(name="MockExtractor")
        self._data = data or [{"id": 1}]
        self._fail = fail

    def _extract(self, **kwargs):
        if self._fail:
            raise RuntimeError("Extraction failed")
        return list(self._data)


# ═══════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════

class TestETLPipeline:
    def test_init_defaults(self, tmp_path: Path) -> None:
        pipeline = ETLPipeline(
            name="test_pipeline",
            source="test_source",
            extractor=MockExtractor(),
            store=FileStore(output_dir=tmp_path),
        )
        assert pipeline.name == "test_pipeline"
        assert pipeline.source == "test_source"
        assert pipeline.tracker is None  # checkpoint=False by default
        assert pipeline.config is not None

    def test_successful_run(self, tmp_path: Path) -> None:
        pipeline = ETLPipeline(
            name="test_pipeline",
            source="test_source",
            extractor=MockExtractor([{"id": 1}, {"id": 2}]),
            store=FileStore(output_dir=tmp_path, filename="test.csv"),
        )
        result = pipeline.run()

        assert result.success is True
        assert result.overall_status == StageStatus.SUCCESS
        assert PipelineStage.EXTRACT in result.stages
        assert PipelineStage.VALIDATE in result.stages
        assert PipelineStage.CLEAN in result.stages
        assert PipelineStage.NORMALIZE in result.stages
        assert PipelineStage.TRANSFORM in result.stages
        assert PipelineStage.STORE in result.stages
        assert result.pipeline_name == "test_pipeline"
        assert result.source == "test_source"

    def test_empty_data_flow(self, tmp_path: Path) -> None:
        pipeline = ETLPipeline(
            name="empty_test",
            source="empty",
            extractor=MockExtractor([]),
            store=FileStore(output_dir=tmp_path),
        )
        result = pipeline.run()

        # Pipeline runs to completion with empty data
        # SUCCESS is acceptable since no stage failed
        assert result.overall_status in (StageStatus.SUCCESS, StageStatus.WARNING)

    def test_extraction_failure_aborts(self, tmp_path: Path) -> None:
        pipeline = ETLPipeline(
            name="fail_test",
            source="fail",
            extractor=MockExtractor(fail=True),
            store=FileStore(output_dir=tmp_path),
        )
        result = pipeline.run()

        assert result.success is False
        assert result.overall_status == StageStatus.FAILED

    def test_with_checkpoint(self, tmp_path: Path) -> None:
        """Checkpoint mode creates a tracker."""
        pipeline = ETLPipeline(
            name="checkpoint_test",
            source="test",
            extractor=MockExtractor([{"id": 1}]),
            store=FileStore(output_dir=tmp_path),
            checkpoint=True,
        )
        assert pipeline.tracker is not None
        result = pipeline.run()
        assert result.success is True
        assert result.checkpoint_id is not None

    def test_with_custom_config(self, tmp_path: Path) -> None:
        config = ETLConfig(
            name="custom",
            source="custom",
            batch_size=500,
            parallel=True,
            max_workers=2,
        )
        pipeline = ETLPipeline(
            name="custom",
            source="custom",
            extractor=MockExtractor([{"x": 1}]),
            store=FileStore(output_dir=tmp_path),
            config=config,
        )
        assert pipeline.config.batch_size == 500
        assert pipeline.config.parallel is True

    def test_validate_strict_parameter(self, tmp_path: Path) -> None:
        """When validate_strict is passed via config, ensure validation is loaded."""
        from src.etl.validate import DataValidator

        pipeline = ETLPipeline(
            name="strict_test",
            source="test",
            extractor=MockExtractor([{"id": 1}]),
            store=FileStore(output_dir=tmp_path),
        )
        assert isinstance(pipeline.validator, DataValidator)

    def test_run_report_structure(self, tmp_path: Path) -> None:
        pipeline = ETLPipeline(
            name="report_test",
            source="test",
            extractor=MockExtractor([{"a": 1}, {"b": 2}]),
            store=FileStore(output_dir=tmp_path),
        )
        result = pipeline.run()

        d = result.to_dict()
        assert d["pipeline_name"] == "report_test"
        assert d["source"] == "test"
        assert "stages" in d
        assert len(d["stages"]) == 6
        assert "overall_status" in d
        assert "total_duration_seconds" in d
        assert d["total_duration_seconds"] > 0

    def test_tracker_cleanup_on_success(self, tmp_path: Path) -> None:
        """On successful completion, checkpoint file is deleted."""
        pipeline = ETLPipeline(
            name="cleanup_test",
            source="test",
            extractor=MockExtractor([{"id": 1}]),
            store=FileStore(output_dir=tmp_path),
            checkpoint=True,
        )
        result = pipeline.run()

        assert result.success is True
        assert pipeline.tracker is not None
        # Checkpoint should be deleted (no leftover files)
        assert pipeline.tracker.list_jobs() == []

    def test_progress_bar_created(self, tmp_path: Path) -> None:
        pipeline = ETLPipeline(
            name="progress_test",
            source="test",
            extractor=MockExtractor([{"id": 1}]),
            store=FileStore(output_dir=tmp_path),
        )
        assert pipeline.progress is not None
        assert pipeline.progress.enabled is True
