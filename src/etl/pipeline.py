"""
ETL Pipeline — orchestrates Extract → Validate → Clean → Normalize → Transform → Store.

This is the top-level orchestrator that composes the six pipeline
stages into a single configurable execution flow. Every scrapter
or data source gets its own pipeline instance wired with the
specific extractor, transforms, and store it needs.

Usage
-----
::

    from src.etl import ETLPipeline
    from src.etl.extract import CSVExtractor
    from src.etl.clean import DataCleaner
    from src.etl.normalize import DataNormalizer
    from src.etl.store import DatabaseStore
    from src.database.models import Match

    pipeline = ETLPipeline(
        name="import_matches",
        source="football-data-co-uk",
        extractor=CSVExtractor("data/raw/results.csv"),
        cleaner=DataCleaner(fill_strategy="drop"),
        normalizer=DataNormalizer(
            team_name_columns=["home_team", "away_team"],
            date_columns=["date"],
        ),
        store=DatabaseStore(Match, unique_columns=["match_id"]),
        checkpoint=True,
    )

    result = pipeline.run()
    print(result.status)   # StageStatus.SUCCESS
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.etl.extract import BaseExtractor
from src.etl.validate import DataValidator
from src.etl.clean import DataCleaner
from src.etl.normalize import DataNormalizer
from src.etl.transform import DataTransformer
from src.etl.store import DataStore
from src.etl.tracker import JobTracker
from src.etl.progress import ProgressReporter
from src.etl.models import (
    ETLConfig,
    ETLResult,
    PipelineStage,
    StageResult,
    StageStatus,
)

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Composable ETL pipeline orchestrator.

    Parameters
    ----------
    name : str
        Pipeline name (used in logs and checkpoints).
    source : str
        Data source identifier.
    extractor : BaseExtractor
        Stage 1: data extraction.
    validator : DataValidator, optional
        Stage 2: validation (default: DataValidator() with basic checks).
    cleaner : DataCleaner, optional
        Stage 3: cleaning (default: DataCleaner() with defaults).
    normalizer : DataNormalizer, optional
        Stage 4: normalisation (default: DataNormalizer()).
    transformer : DataTransformer, optional
        Stage 5: transformation (default: DataTransformer() with no ops).
    store : DataStore
        Stage 6: persistence (REQUIRED).
    config : ETLConfig, optional
        Override default pipeline configuration.
    checkpoint : bool
        Enable checkpoint/resume (default False).
    parallel : bool
        Enable parallel processing (default False).
    max_workers : int
        Thread pool size for parallel stages (default 4).
    """

    def __init__(
        self,
        name: str,
        source: str,
        extractor: BaseExtractor,
        store: DataStore,
        validator: DataValidator | None = None,
        cleaner: DataCleaner | None = None,
        normalizer: DataNormalizer | None = None,
        transformer: DataTransformer | None = None,
        config: ETLConfig | None = None,
        checkpoint: bool = False,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> None:
        self.name = name
        self.source = source
        self.config = config or ETLConfig(
            name=name, source=source,
            checkpoint=checkpoint,
            parallel=parallel,
            max_workers=max_workers,
        )

        # Stages (in order)
        self.extractor = extractor
        self.validator = validator or DataValidator()
        self.cleaner = cleaner or DataCleaner()
        self.normalizer = normalizer or DataNormalizer()
        self.transformer = transformer or DataTransformer()
        self.store = store

        # Cross-cutting
        self.tracker = JobTracker() if self.config.checkpoint else None
        self.progress = ProgressReporter(enabled=True, desc=name)

    def run(self, **kwargs: Any) -> ETLResult:
        """Execute the full ETL pipeline.

        Parameters
        ----------
        **kwargs
            Extra parameters passed to the Extract stage.

        Returns
        -------
        ETLResult
            Complete pipeline result with per-stage metrics.
        """
        etl_result = ETLResult(
            pipeline_name=self.name,
            source=self.source,
            started_at=datetime.now(timezone.utc),
        )
        start_total = datetime.now(timezone.utc)

        # Checkpoint: try to resume
        resume_from: PipelineStage | None = None
        if self.tracker is not None:
            # If a job_id is provided, attempt resume
            job_id = kwargs.pop("job_id", None)
            if job_id:
                state = self.tracker.resume_job(job_id)
                if state is not None:
                    resume_from = state.resume_from
                    logger.info(
                        "Resuming job %s from stage %s",
                        job_id,
                        resume_from.value if resume_from else "start",
                    )
                else:
                    state = self.tracker.create_job(self.name, job_id)
            else:
                state = self.tracker.create_job(self.name)
            etl_result.checkpoint_id = state.job_id
        else:
            state = None

        stages = [
            (PipelineStage.EXTRACT, self._run_extract),
            (PipelineStage.VALIDATE, self._run_validate),
            (PipelineStage.CLEAN, self._run_clean),
            (PipelineStage.NORMALIZE, self._run_normalize),
            (PipelineStage.TRANSFORM, self._run_transform),
            (PipelineStage.STORE, self._run_store),
        ]

        # Determine total stages for progress bar
        if resume_from:
            start_idx = next(
                i for i, (s, _) in enumerate(stages) if s == resume_from
            )
            stages = stages[start_idx:]

        self.progress.start_pipeline(len(stages))
        current_data: list[dict[str, Any]] = []

        for stage, stage_fn in stages:
            # Checkpoint: skip completed stages
            if state and stage in state.completed_stages:
                self.progress.update(1)
                continue

            self.progress.start_stage(
                stage,
                record_count=len(current_data) if current_data else 0,
            )

            try:
                stage_result = stage_fn(current_data, **kwargs)
            except Exception as exc:
                logger.exception("Fatal error in stage %s: %s", stage.value, exc)
                stage_result = StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    errors=[f"Fatal: {exc}"],
                )

            etl_result.stages[stage] = stage_result
            etl_result.total_errors += len(stage_result.errors)

            self.progress.finish_stage(
                stage, stage_result.status, stage_result.records_out,
            )

            # Pass data to next stage
            current_data = stage_result.data or []

            # Checkpoint: save
            if state:
                if stage_result.status in (StageStatus.SUCCESS, StageStatus.WARNING):
                    self.tracker.mark_stage_done(state, stage)
                else:
                    self.tracker.mark_stage_failed(state, stage)
                    break  # abort on failure

            # Abort on failure (unless continuing through warnings)
            if stage_result.status == StageStatus.FAILED:
                logger.error("Pipeline aborted at stage: %s", stage.value)
                break

            # Progress bar
            self.progress.update(1)

        # ── Finalise ──────────────────────────────────
        etl_result.completed_at = datetime.now(timezone.utc)
        etl_result.total_duration_seconds = (
            etl_result.completed_at - etl_result.started_at
        ).total_seconds()

        # Determine overall status
        statuses = [s.status for s in etl_result.stages.values()]
        if any(s == StageStatus.FAILED for s in statuses):
            etl_result.overall_status = StageStatus.FAILED
        elif any(s == StageStatus.WARNING for s in statuses):
            etl_result.overall_status = StageStatus.WARNING
        else:
            etl_result.overall_status = StageStatus.SUCCESS

        etl_result.total_records = sum(
            s.records_out for s in etl_result.stages.values()
        )

        # Clean up checkpoint on success
        if state and etl_result.success and self.tracker is not None:
            self.tracker.delete_checkpoint(state.job_id)

        self.progress.finish_pipeline(
            etl_result.overall_status,
            etl_result.total_duration_seconds,
            etl_result.total_records,
        )

        logger.info(
            "%s pipeline %s: %d records in %.1fs",
            self.name,
            etl_result.overall_status.value,
            etl_result.total_records,
            etl_result.total_duration_seconds,
        )

        return etl_result

    # ── Stage runners ──────────────────────────────────

    def _run_extract(
        self, data: list[dict[str, Any]], **kwargs: Any
    ) -> StageResult:
        return self.extractor.run(**kwargs)

    def _run_validate(
        self, data: list[dict[str, Any]], **kwargs: Any
    ) -> StageResult:
        return self.validator.run(data)

    def _run_clean(
        self, data: list[dict[str, Any]], **kwargs: Any
    ) -> StageResult:
        return self.cleaner.run(data)

    def _run_normalize(
        self, data: list[dict[str, Any]], **kwargs: Any
    ) -> StageResult:
        return self.normalizer.run(data)

    def _run_transform(
        self, data: list[dict[str, Any]], **kwargs: Any
    ) -> StageResult:
        return self.transformer.run(data)

    def _run_store(
        self, data: list[dict[str, Any]], **kwargs: Any
    ) -> StageResult:
        return self.store.write(data)
