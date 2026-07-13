"""
ETL data models — shared types for the pipeline framework.

All pipeline stages produce ``StageResult`` objects. The overall
pipeline produces one ``ETLResult``. Validation outputs are
captured as ``ValidationReport``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PipelineStage(str, Enum):
    """Stages of the ETL pipeline, in execution order."""

    EXTRACT = "extract"
    VALIDATE = "validate"
    CLEAN = "clean"
    NORMALIZE = "normalize"
    TRANSFORM = "transform"
    STORE = "store"

    def __str__(self) -> str:
        return self.value


class StageStatus(str, Enum):
    """Outcome of a single pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"  # completed with warnings
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result produced by one ETL pipeline stage.

    Attributes
    ----------
    stage : PipelineStage
        Which stage produced this result.
    status : StageStatus
        Outcome of the stage.
    records_in : int
        Number of input records received.
    records_out : int
        Number of output records produced.
    data : Any
        The transformed data (DataFrame, list[dict], etc.) to pass
        to the next stage.
    errors : list[str]
        Non-fatal errors or warnings encountered.
    metrics : dict[str, float]
        Stage-specific metrics (e.g. null_pct, duplicate_count).
    duration_seconds : float
        Wall-clock time for this stage.
    """

    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    records_in: int = 0
    records_out: int = 0
    data: Any = None
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class ETLResult:
    """Complete result of a full ETL pipeline run.

    Serialisable to JSON for logging and reporting.
    """

    pipeline_name: str = ""
    source: str = ""
    stages: dict[PipelineStage, StageResult] = field(default_factory=dict)
    overall_status: StageStatus = StageStatus.PENDING
    total_records: int = 0
    total_errors: int = 0
    total_duration_seconds: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    checkpoint_id: str | None = None

    @property
    def success(self) -> bool:
        return self.overall_status in (StageStatus.SUCCESS, StageStatus.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "source": self.source,
            "overall_status": self.overall_status.value,
            "total_records": self.total_records,
            "total_errors": self.total_errors,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "checkpoint_id": self.checkpoint_id,
            "stages": {
                k.value: {
                    "status": v.status.value,
                    "records_in": v.records_in,
                    "records_out": v.records_out,
                    "errors": v.errors[:5],
                    "metrics": v.metrics,
                    "duration_seconds": round(v.duration_seconds, 3),
                }
                for k, v in self.stages.items()
            },
        }


@dataclass
class ValidationRuleResult:
    """Result of a single validation rule check."""

    rule_name: str
    passed: bool
    record_count: int = 0
    failure_count: int = 0
    details: str = ""


@dataclass
class ValidationReport:
    """Aggregated report from the Validation stage."""

    total_checks: int = 0
    passed: int = 0
    warnings: int = 0
    failures: int = 0
    rules: list[ValidationRuleResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.failures == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "passed": self.passed,
            "warnings": self.warnings,
            "failures": self.failures,
            "is_valid": self.is_valid,
        }


@dataclass
class ETLConfig:
    """Configuration for an ETL pipeline run.

    Can be populated from a YAML file or a dict, enabling
    config-driven pipeline execution.

    Attributes
    ----------
    name : str
        Pipeline name for logging and tracking.
    source : str
        Data source identifier (e.g. ``football-data-co-uk``).
    stages : dict[PipelineStage, dict]
        Per-stage configuration parameters.
    batch_size : int
        Number of records to process per batch (default 1000).
    parallel : bool
        Enable parallel processing (default False).
    max_workers : int
        Thread/process pool size (default 4).
    retry_attempts : int
        Max retries for failed API calls (default 3).
    retry_backoff : float
        Exponential backoff base in seconds (default 2.0).
    checkpoint : bool
        Enable checkpoint/resume (default False).
    validate_strict : bool
        If True, validation failures abort the pipeline (default False).
    """

    name: str = ""
    source: str = ""
    stages: dict[PipelineStage, dict] = field(default_factory=dict)
    batch_size: int = 1000
    parallel: bool = False
    max_workers: int = 4
    retry_attempts: int = 3
    retry_backoff: float = 2.0
    checkpoint: bool = False
    validate_strict: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ETLConfig:
        """Create config from a dictionary (e.g. from YAML or JSON)."""
        stages = {}
        for stage_name, stage_cfg in d.get("stages", {}).items():
            try:
                stage = PipelineStage(stage_name)
            except ValueError:
                continue
            stages[stage] = stage_cfg

        return cls(
            name=d.get("name", ""),
            source=d.get("source", ""),
            stages=stages,
            batch_size=d.get("batch_size", 1000),
            parallel=d.get("parallel", False),
            max_workers=d.get("max_workers", 4),
            retry_attempts=d.get("retry_attempts", 3),
            retry_backoff=d.get("retry_backoff", 2.0),
            checkpoint=d.get("checkpoint", False),
            validate_strict=d.get("validate_strict", False),
        )
