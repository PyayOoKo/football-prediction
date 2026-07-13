"""
Tests for ETL data models — StageResult, ETLResult, ValidationReport, ETLConfig.
"""

from __future__ import annotations

from src.etl.models import (
    ETLConfig,
    ETLResult,
    PipelineStage,
    StageResult,
    StageStatus,
    ValidationReport,
    ValidationRuleResult,
)


class TestStageResult:
    def test_default_status(self) -> None:
        r = StageResult(stage=PipelineStage.EXTRACT)
        assert r.status == StageStatus.PENDING
        assert r.stage == PipelineStage.EXTRACT

    def test_custom_values(self) -> None:
        r = StageResult(
            stage=PipelineStage.CLEAN,
            status=StageStatus.SUCCESS,
            records_in=100,
            records_out=95,
            errors=["5 duplicate rows removed"],
        )
        assert r.status == StageStatus.SUCCESS
        assert r.records_in == 100
        assert r.records_out == 95


class TestETLResult:
    def test_success_property(self) -> None:
        r = ETLResult(overall_status=StageStatus.SUCCESS)
        assert r.success is True

        r = ETLResult(overall_status=StageStatus.FAILED)
        assert r.success is False

    def test_to_dict(self) -> None:
        r = ETLResult(
            pipeline_name="test_pipeline",
            source="test_source",
            overall_status=StageStatus.SUCCESS,
            total_records=100,
        )
        d = r.to_dict()
        assert d["pipeline_name"] == "test_pipeline"
        assert d["overall_status"] == "success"
        assert d["total_records"] == 100


class TestValidationReport:
    def test_default(self) -> None:
        report = ValidationReport()
        assert report.is_valid is True
        assert report.total_checks == 0

    def test_with_failures(self) -> None:
        report = ValidationReport(
            total_checks=3,
            passed=1,
            warnings=1,
            failures=1,
            rules=[
                ValidationRuleResult("not_empty", passed=True),
                ValidationRuleResult("unique_rows", passed=False, failure_count=2),
            ],
        )
        assert report.is_valid is False
        assert report.failures == 1

    def test_to_dict(self) -> None:
        d = ValidationReport(failures=2).to_dict()
        assert d["is_valid"] is False
        assert d["failures"] == 2


class TestETLConfig:
    def test_defaults(self) -> None:
        cfg = ETLConfig(name="test")
        assert cfg.name == "test"
        assert cfg.batch_size == 1000
        assert cfg.parallel is False

    def test_from_dict(self) -> None:
        cfg = ETLConfig.from_dict({
            "name": "import_matches",
            "source": "csv",
            "batch_size": 500,
            "stages": {
                "extract": {"filepath": "data.csv"},
                "clean": {"fill_strategy": "drop"},
            },
        })
        assert cfg.name == "import_matches"
        assert cfg.batch_size == 500
        assert PipelineStage.EXTRACT in cfg.stages
        assert PipelineStage.CLEAN in cfg.stages
