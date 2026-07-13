"""
Tests for the ETL progress reporter — ProgressReporter.
"""

from __future__ import annotations

import pytest

from src.etl.progress import ProgressReporter
from src.etl.models import PipelineStage, StageStatus


class TestProgressReporter:
    def test_init_defaults(self) -> None:
        reporter = ProgressReporter()
        assert reporter.enabled is True
        assert reporter.desc == "ETL Pipeline"
        assert reporter._bar is None

    def test_init_disabled(self) -> None:
        reporter = ProgressReporter(enabled=False)
        assert reporter.enabled is False

    def test_start_pipeline(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(total_stages=6)
        assert reporter._bar is not None
        assert reporter._bar.total == 6

    def test_start_pipeline_disabled(self) -> None:
        reporter = ProgressReporter(enabled=False)
        reporter.start_pipeline(total_stages=6)
        assert reporter._bar is None  # no bar when disabled

    def test_start_stage(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(total_stages=6)
        reporter.start_stage(PipelineStage.EXTRACT, record_count=100)
        assert reporter._current_stage == "extract"
        assert PipelineStage.EXTRACT.value in reporter._stage_times

    def test_start_stage_disabled_no_crash(self) -> None:
        reporter = ProgressReporter(enabled=False)
        reporter.start_stage(PipelineStage.EXTRACT)

    def test_update(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(total_stages=6)
        reporter.update(1)  # Should not crash

    def test_finish_stage(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(total_stages=6)
        reporter.start_stage(PipelineStage.EXTRACT, record_count=100)
        reporter.finish_stage(PipelineStage.EXTRACT, StageStatus.SUCCESS, records=100)

        assert reporter._current_stage == ""  # reset after finish

    def test_finish_stage_no_start(self) -> None:
        """Finishing a stage without starting it should not crash."""
        reporter = ProgressReporter()
        reporter.finish_stage(PipelineStage.EXTRACT, StageStatus.SUCCESS)

    def test_finish_pipeline(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(total_stages=6)
        reporter.finish_pipeline(StageStatus.SUCCESS, total_duration=10.5, total_records=500)
        assert reporter._bar is None  # bar closed

    def test_finish_pipeline_no_bar(self) -> None:
        reporter = ProgressReporter()
        reporter.finish_pipeline(StageStatus.SUCCESS, 1.0, 10)  # Should not crash

    def test_stage_progress_logging(self) -> None:
        reporter = ProgressReporter()
        # Should log at 25%, 50%, 75%, 100%
        reporter.stage_progress(PipelineStage.EXTRACT, 50, 200)  # Should not crash
        reporter.stage_progress(PipelineStage.EXTRACT, 100, 200)  # Should not crash

    def test_stage_progress_disabled(self) -> None:
        reporter = ProgressReporter(enabled=False)
        reporter.stage_progress(PipelineStage.EXTRACT, 50, 100)  # Should not crash

    def test_stage_progress_no_total(self) -> None:
        reporter = ProgressReporter()
        reporter.stage_progress(PipelineStage.EXTRACT, 5, 0)  # Should not crash

    def test_stage_context_success(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(total_stages=1)

        with reporter.stage_context(PipelineStage.EXTRACT, record_count=100):
            pass  # Stage runs successfully

        assert reporter._current_stage == ""

    def test_stage_context_failure(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(total_stages=1)

        class TestError(Exception):
            pass

        with pytest.raises(TestError):
            with reporter.stage_context(PipelineStage.EXTRACT):
                raise TestError("Boom")

    def test_custom_desc(self) -> None:
        reporter = ProgressReporter(desc="World Cup Pipeline")
        assert reporter.desc == "World Cup Pipeline"

    def test_twice_start_pipeline_no_crash(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(6)
        reporter.start_pipeline(3)  # Second call replaces bar
        assert reporter._bar is not None

    def test_update_with_kwargs(self) -> None:
        reporter = ProgressReporter()
        reporter.start_pipeline(3)
        reporter.update(1, stage="extract", status="running")  # Should not crash
