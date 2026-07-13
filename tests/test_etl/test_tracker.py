"""
Tests for the ETL job tracker — JobState, JobTracker checkpoint/resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.etl.tracker import JobState, JobTracker
from src.etl.models import PipelineStage


# ═══════════════════════════════════════════════════════════
#  JobState
# ═══════════════════════════════════════════════════════════

class TestJobState:
    def test_initial_state(self) -> None:
        state = JobState(job_id="test_001")
        assert state.job_id == "test_001"
        assert state.completed_stages == set()
        assert state.failed_stage is None
        assert state.is_complete is True

    def test_mark_done(self) -> None:
        state = JobState(job_id="test")
        state.mark_done(PipelineStage.EXTRACT)
        assert PipelineStage.EXTRACT in state.completed_stages
        assert state.is_complete is True

    def test_mark_failed(self) -> None:
        state = JobState(job_id="test")
        state.mark_failed(PipelineStage.CLEAN)
        assert state.failed_stage == PipelineStage.CLEAN
        assert state.is_complete is False

    def test_resume_from_first_incomplete(self) -> None:
        state = JobState(job_id="test")
        state.mark_done(PipelineStage.EXTRACT)
        state.mark_done(PipelineStage.VALIDATE)
        resume = state.resume_from
        assert resume == PipelineStage.CLEAN

    def test_resume_from_when_all_done(self) -> None:
        state = JobState(job_id="test")
        for stage in PipelineStage:
            state.mark_done(stage)
        assert state.resume_from is None

    def test_resume_from_starts_at_extract(self) -> None:
        state = JobState(job_id="test")
        assert state.resume_from == PipelineStage.EXTRACT

    def test_to_dict(self) -> None:
        state = JobState(job_id="j1", pipeline_name="pipeline_a")
        state.mark_done(PipelineStage.EXTRACT)
        state.records_processed = 500

        d = state.to_dict()
        assert d["job_id"] == "j1"
        assert d["pipeline_name"] == "pipeline_a"
        assert d["completed_stages"] == ["extract"]
        assert d["records_processed"] == 500
        assert d["failed_stage"] is None

    def test_from_dict(self) -> None:
        d = {
            "job_id": "j2",
            "pipeline_name": "pipeline_b",
            "completed_stages": ["extract", "clean"],
            "failed_stage": "store",
            "records_processed": 100,
            "started_at": "2024-01-01T00:00:00",
            "extra": {"note": "test"},
        }
        state = JobState.from_dict(d)
        assert state.job_id == "j2"
        assert PipelineStage.EXTRACT in state.completed_stages
        assert PipelineStage.CLEAN in state.completed_stages
        assert state.failed_stage == PipelineStage.STORE
        assert state.records_processed == 100
        assert state.extra["note"] == "test"

    def test_from_dict_no_failed(self) -> None:
        d = {"job_id": "j3", "completed_stages": []}
        state = JobState.from_dict(d)
        assert state.failed_stage is None
        assert state.pipeline_name == ""


# ═══════════════════════════════════════════════════════════
#  JobTracker
# ═══════════════════════════════════════════════════════════

class TestJobTracker:
    def test_create_job(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        state = tracker.create_job("test_pipeline")

        assert state.pipeline_name == "test_pipeline"
        assert state.job_id.startswith("test_pipeline_")
        # Checkpoint file was created
        assert list(tmp_path.glob("*.json"))

    def test_create_job_with_custom_id(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        state = tracker.create_job("pipe", job_id="custom_id")
        assert state.job_id == "custom_id"
        assert (tmp_path / "custom_id.json").exists()

    def test_resume_job_exists(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        original = tracker.create_job("test_pipeline", job_id="resume_test")
        original.mark_done(PipelineStage.EXTRACT)
        tracker.save_checkpoint(original)

        loaded = tracker.resume_job("resume_test")
        assert loaded is not None
        assert loaded.job_id == "resume_test"
        assert PipelineStage.EXTRACT in loaded.completed_stages

    def test_resume_job_not_found(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        loaded = tracker.resume_job("nonexistent")
        assert loaded is None

    def test_resume_corrupted_checkpoint(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        with open(tmp_path / "corrupt.json", "w") as f:
            f.write("not valid json")

        loaded = tracker.resume_job("corrupt")
        assert loaded is None

    def test_mark_stage_done_saves(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        state = tracker.create_job("test", job_id="stage_done")

        tracker.mark_stage_done(state, PipelineStage.EXTRACT)
        assert PipelineStage.EXTRACT in state.completed_stages

        # Verify persisted
        with open(tmp_path / "stage_done.json") as f:
            data = json.load(f)
        assert "extract" in data["completed_stages"]

    def test_mark_stage_failed_saves(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        state = tracker.create_job("test", job_id="stage_fail")

        tracker.mark_stage_failed(state, PipelineStage.STORE)
        assert state.failed_stage == PipelineStage.STORE

        with open(tmp_path / "stage_fail.json") as f:
            data = json.load(f)
        assert data["failed_stage"] == "store"

    def test_delete_checkpoint(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        tracker.create_job("test", job_id="to_delete")
        assert (tmp_path / "to_delete.json").exists()

        tracker.delete_checkpoint("to_delete")
        assert not (tmp_path / "to_delete.json").exists()

    def test_delete_nonexistent_no_error(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        tracker.delete_checkpoint("does_not_exist")  # Should not raise

    def test_list_jobs(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        tracker.create_job("pipe_a", job_id="job_a")
        tracker.create_job("pipe_b", job_id="job_b")

        jobs = tracker.list_jobs()
        assert "job_a" in jobs
        assert "job_b" in jobs

    def test_list_jobs_empty(self, tmp_path: Path) -> None:
        tracker = JobTracker(checkpoint_dir=tmp_path)
        assert tracker.list_jobs() == []

    def test_checkpoint_dir_created(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "checkpoints" / "sub"
        tracker = JobTracker(checkpoint_dir=new_dir)
        assert new_dir.exists()
