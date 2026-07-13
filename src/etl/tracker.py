"""
Job tracker — enables checkpoint/resume for failed ETL runs.

When a pipeline fails partway through, the tracker records which
stages completed so the job can be resumed from the failure point
rather than restarting from scratch.

State is persisted to a JSON file in the ``checkpoints/`` directory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.etl.models import PipelineStage, StageStatus

logger = logging.getLogger(__name__)


class JobState:
    """Mutable state for a single ETL job.

    Attributes
    ----------
    job_id : str
        Unique job identifier.
    pipeline_name : str
        Name of the pipeline being run.
    completed_stages : set[PipelineStage]
        Stages that completed successfully.
    failed_stage : PipelineStage | None
        The stage that failed (if any).
    records_processed : int
        Total records processed so far.
    started_at : str
        ISO timestamp of job start.
    extra : dict
        Arbitrary extra state for subclasses.
    """

    def __init__(
        self,
        job_id: str,
        pipeline_name: str = "",
    ) -> None:
        self.job_id = job_id
        self.pipeline_name = pipeline_name
        self.completed_stages: set[PipelineStage] = set()
        self.failed_stage: PipelineStage | None = None
        self.records_processed: int = 0
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.extra: dict[str, Any] = {}

    @property
    def is_complete(self) -> bool:
        return self.failed_stage is None

    @property
    def resume_from(self) -> PipelineStage | None:
        """Return the stage to resume from.

        Returns the first incomplete stage, or None if all done.
        """
        for stage in PipelineStage:
            if stage not in self.completed_stages:
                return stage
        return None

    def mark_done(self, stage: PipelineStage) -> None:
        self.completed_stages.add(stage)

    def mark_failed(self, stage: PipelineStage) -> None:
        self.failed_stage = stage

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pipeline_name": self.pipeline_name,
            "completed_stages": [s.value for s in self.completed_stages],
            "failed_stage": self.failed_stage.value if self.failed_stage else None,
            "records_processed": self.records_processed,
            "started_at": self.started_at,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobState:
        state = cls(
            job_id=d["job_id"],
            pipeline_name=d.get("pipeline_name", ""),
        )
        state.completed_stages = {
            PipelineStage(s) for s in d.get("completed_stages", [])
        }
        fs = d.get("failed_stage")
        state.failed_stage = PipelineStage(fs) if fs else None
        state.records_processed = d.get("records_processed", 0)
        state.started_at = d.get("started_at", state.started_at)
        state.extra = d.get("extra", {})
        return state


class JobTracker:
    """Persistent job tracker for checkpoint/resume.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory to store checkpoint JSON files (default ``checkpoints/``).
    """

    def __init__(self, checkpoint_dir: str | Path = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, pipeline_name: str, job_id: str | None = None) -> JobState:
        """Create a new job and persist its initial state.

        Parameters
        ----------
        pipeline_name : str
            Pipeline name.
        job_id : str, optional
            Custom job ID. Auto-generated if not provided.

        Returns
        -------
        JobState
        """
        if job_id is None:
            job_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        state = JobState(job_id=job_id, pipeline_name=pipeline_name)
        self._save(state)
        logger.info("Job created: %s", job_id)
        return state

    def resume_job(self, job_id: str) -> JobState | None:
        """Resume a previously saved job.

        Parameters
        ----------
        job_id : str
            Job ID to resume.

        Returns
        -------
        JobState | None
            The loaded job state, or None if not found.
        """
        path = self._path(job_id)
        if not path.exists():
            logger.warning("Checkpoint not found: %s", path)
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            state = JobState.from_dict(data)
            logger.info(
                "Resumed job %s (completed: %s, failed: %s)",
                job_id,
                [s.value for s in state.completed_stages],
                state.failed_stage.value if state.failed_stage else None,
            )
            return state
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load checkpoint %s: %s", path, exc)
            return None

    def save_checkpoint(self, state: JobState) -> None:
        """Persist the current job state."""
        self._save(state)
        logger.debug("Checkpoint saved: %s", state.job_id)

    def mark_stage_done(self, state: JobState, stage: PipelineStage) -> None:
        """Mark a stage as completed and persist."""
        state.mark_done(stage)
        self.save_checkpoint(state)
        logger.info("Stage %s completed for job %s", stage.value, state.job_id)

    def mark_stage_failed(self, state: JobState, stage: PipelineStage) -> None:
        """Mark a stage as failed and persist."""
        state.mark_failed(stage)
        self.save_checkpoint(state)
        logger.warning("Stage %s failed for job %s", stage.value, state.job_id)

    def delete_checkpoint(self, job_id: str) -> None:
        """Remove a checkpoint file (e.g. after successful completion)."""
        path = self._path(job_id)
        if path.exists():
            path.unlink()
            logger.info("Checkpoint deleted: %s", job_id)

    def list_jobs(self) -> list[str]:
        """List all tracked job IDs."""
        return sorted(
            [p.stem for p in self.checkpoint_dir.glob("*.json")]
        )

    def _path(self, job_id: str) -> Path:
        return self.checkpoint_dir / f"{job_id}.json"

    def _save(self, state: JobState) -> None:
        path = self._path(state.job_id)
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
