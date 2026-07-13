"""
Progress reporting — tqdm-based progress bars with timing and throughput.

Provides a central ``ProgressReporter`` that all ETL stages can use
to show:
- Current stage name and status
- Record count and rate (rec/s)
- Elapsed and estimated time remaining
- Per-stage timing breakdown
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from tqdm import tqdm

from src.etl.models import PipelineStage, StageStatus

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Configurable progress reporter for ETL pipelines.

    Maintains an ordered list of stages and displays a live
    progress bar with throughput statistics.

    Parameters
    ----------
    enabled : bool
        Enable progress bars (default True). Set to False for CI.
    desc : str
        Pipeline description (default ``"ETL Pipeline"``).
    """

    def __init__(self, enabled: bool = True, desc: str = "ETL Pipeline") -> None:
        self.enabled = enabled
        self.desc = desc
        self._bar: tqdm | None = None
        self._stage_times: dict[str, float] = {}
        self._current_stage: str = ""

    def start_pipeline(
        self,
        total_stages: int,
        total_records: int = 0,
    ) -> None:
        """Initialise the progress bar for the full pipeline.

        Parameters
        ----------
        total_stages : int
            Number of stages in the pipeline.
        total_records : int
            Estimated total records (0 = unknown).
        """
        if not self.enabled:
            return
        self._bar = tqdm(
            total=total_stages,
            desc=self.desc,
            unit="stage",
            ncols=100,
            bar_format=(
                "{desc:<30s} |{bar:30}| {n_fmt}/{total_fmt} stages "
                "[{elapsed}<{remaining}]"
            ),
        )

    def start_stage(self, stage: PipelineStage, record_count: int = 0) -> None:
        """Mark the start of a pipeline stage.

        Parameters
        ----------
        stage : PipelineStage
            The stage being started.
        record_count : int
            Number of records entering this stage.
        """
        self._current_stage = stage.value
        self._stage_times[stage.value] = time.perf_counter()

        if self._bar is not None:
            self._bar.set_postfix(
                stage=stage.value,
                records=f"{record_count:,}" if record_count else "?",
                refresh=False,
            )

    def update(self, n: int = 1, **kwargs: Any) -> None:
        """Advance the progress bar by ``n`` steps.

        Parameters
        ----------
        n : int
            Number of steps (default 1 = one stage).
        """
        if self._bar is not None:
            self._bar.update(n)
            if kwargs:
                self._bar.set_postfix(**kwargs, refresh=False)

    def finish_stage(
        self,
        stage: PipelineStage,
        status: StageStatus,
        records: int = 0,
    ) -> None:
        """Record that a stage has completed.

        Parameters
        ----------
        stage : PipelineStage
            The completed stage.
        status : StageStatus
            Final status.
        records : int
            Records output from this stage.
        """
        elapsed = time.perf_counter() - self._stage_times.get(stage.value, 0.0)
        rate = f"{records / elapsed:.0f}" if elapsed > 0 and records > 0 else "?"

        if self._bar is not None:
            self._bar.set_postfix(
                stage=stage.value,
                status=status.value,
                records=f"{records:,}",
                rate=f"{rate} rec/s",
                refresh=True,
            )
        self._current_stage = ""

    def finish_pipeline(
        self,
        overall_status: StageStatus,
        total_duration: float,
        total_records: int,
    ) -> None:
        """Close the progress bar and log pipeline summary."""
        if self._bar is not None:
            self._bar.set_postfix(
                status=overall_status.value,
                total=f"{total_records:,}",
                duration=f"{total_duration:.1f}s",
            )
            self._bar.close()
            self._bar = None

        logger.info(
            "Pipeline %s: %d records in %.1fs",
            overall_status.value,
            total_records,
            total_duration,
        )

    def stage_progress(
        self,
        stage: PipelineStage,
        current: int,
        total: int,
    ) -> None:
        """Update per-record progress within a stage.

        Creates or updates a nested progress bar for the current
        stage's individual records.
        """
        if not self.enabled:
            return

        # Use a simple log-based progress for record-level progress
        # to avoid nested tqdm complexity
        if total > 0 and current % max(1, total // 20) == 0:
            pct = current / total * 100
            logger.info(
                "  [%s] %d / %d (%d%%)",
                stage.value,
                current,
                total,
                int(pct),
            )

    @contextmanager
    def stage_context(
        self,
        stage: PipelineStage,
        record_count: int = 0,
    ) -> Generator[None, None, None]:
        """Context manager for a stage — auto starts and finishes.

        Usage::

            with progress.stage_context(PipelineStage.EXTRACT, 1000):
                data = extractor.run()
        """
        self.start_stage(stage, record_count)
        try:
            yield
            self.finish_stage(stage, StageStatus.SUCCESS, record_count)
        except Exception:
            self.finish_stage(stage, StageStatus.FAILED, record_count)
            raise
