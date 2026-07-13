"""
Data Models — ComputationResult, PipelineReport, FeatureMetadata, and
related structures for the feature engineering framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ═══════════════════════════════════════════════════════════
#  Computation Context & Results
# ═══════════════════════════════════════════════════════════


@dataclass
class ComputationResult:
    """Result of computing a single feature for a single entity.

    Parameters
    ----------
    feature_name : str
        Name of the computed feature.
    entity_id : int
        Entity ID (match, team, or league).
    entity_type : str
        Type of entity (match/team/league).
    value : float | str | dict | None
        The computed value.
    success : bool
        Whether computation succeeded.
    duration_seconds : float
        Computation time in seconds.
    error : str | None
        Error message if failed.
    metadata : dict
        Additional context (computer version, params used, etc.).
    """
    feature_name: str
    entity_id: int
    entity_type: str = "match"
    value: float | str | dict[str, Any] | None = None
    success: bool = True
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "value": self.value,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 4),
            "error": self.error,
        }


@dataclass
class FeatureMetadata:
    """Rich metadata for a feature definition (beyond DB columns).

    Parameters
    ----------
    name : str
        Unique feature name.
    version : int
        Feature version.
    description : str
        Human-readable description.
    dependencies : list[str]
        Feature names this depends on.
    output_columns : list[str]
        Column names produced by this feature.
    data_type : str
        Expected data type (float, int, str, bool, categorical).
    computation_time : str
        Expected computation time category (fast, medium, slow).
    validation_rules : dict
        Validation constraints (min, max, nullable, etc.).
    category : str
        Thematic category.
    author : str
        Who created this feature.
    tags : list[str]
        Searchable tags.
    source : str
        Source data (e.g. ``football-data.co.uk``, ``understat``).
    """
    name: str = ""
    version: int = 1
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    data_type: str = "float"
    computation_time: str = "fast"
    validation_rules: dict[str, Any] = field(default_factory=dict)
    category: str = "other"
    author: str = "system"
    tags: list[str] = field(default_factory=list)
    source: str = ""


@dataclass
class TransformContext:
    """Context passed through the feature transform pipeline.

    Carries shared data (e.g. pre-loaded DataFrames, configuration)
    that multiple features need without recomputing.

    Parameters
    ----------
    entity_type : str
        Type of entity being processed.
    entity_ids : list[int]
        Entity IDs being processed.
    trigger : str
        Computation trigger (manual, scheduled, pipeline).
    raw_data : dict, optional
        Pre-loaded raw data shared across features.
    params : dict, optional
        Pipeline parameters.
    """
    entity_type: str = "match"
    entity_ids: list[int] = field(default_factory=list)
    trigger: str = "manual"
    raw_data: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def elapsed_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()


# ═══════════════════════════════════════════════════════════
#  Pipeline Reporting
# ═══════════════════════════════════════════════════════════


@dataclass
class PipelineReport:
    """Full report from a FeaturePipeline run.

    Parameters
    ----------
    success : bool
        Whether the entire pipeline succeeded.
    n_features : int
        Number of features configured.
    n_computed : int
        Number of features actually computed.
    n_skipped : int
        Number of features skipped (already fresh).
    n_failed : int
        Number of features that failed.
    n_entities : int
        Number of entities processed.
    total_duration : float
        Total pipeline duration in seconds.
    per_feature_stats : dict
        Stats per feature: computed, skipped, failed, duration.
    errors : list[str]
        Error messages from failures.
    trigger : str
        Computation trigger.
    batch_id : str | None
        ID of the computation batch in FeatureStore.
    started_at : datetime
        Pipeline start time.
    ended_at : datetime | None
        Pipeline end time.
    metadata : dict
        Additional metadata.
    """
    success: bool = True
    n_features: int = 0
    n_computed: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_entities: int = 0
    total_duration: float = 0.0
    per_feature_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    trigger: str = "manual"
    batch_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "n_features": self.n_features,
            "n_computed": self.n_computed,
            "n_skipped": self.n_skipped,
            "n_failed": self.n_failed,
            "n_entities": self.n_entities,
            "total_duration": round(self.total_duration, 2),
            "per_feature_stats": self.per_feature_stats,
            "errors": self.errors[:10],  # Truncate for readability
            "trigger": self.trigger,
            "batch_id": self.batch_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to the console."""
        print("\n" + "=" * 70)
        print("  FEATURE PIPELINE REPORT")
        print("=" * 70)
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        print(f"  Status:           {status}")
        print(f"  Features:         {self.n_features} configured, "
              f"{self.n_computed} computed, {self.n_skipped} skipped, "
              f"{self.n_failed} failed")
        print(f"  Entities:         {self.n_entities}")
        print(f"  Duration:         {self.total_duration:.2f}s")
        if self.batch_id:
            print(f"  Batch ID:         {self.batch_id}")
        print(f"  Trigger:          {self.trigger}")
        print(f"  Errors:           {len(self.errors)}")
        if self.errors:
            for err in self.errors[:5]:
                print(f"    • {err}")
        if self.per_feature_stats:
            print(f"\n  Per-Feature Stats:")
            for feat, stats in sorted(self.per_feature_stats.items()):
                status_icon = "✅" if stats.get("status") == "ok" else "⚠️"
                print(f"    {status_icon} {feat:<35s} "
                      f"computed={stats.get('computed', 0)} "
                      f"skipped={stats.get('skipped', 0)} "
                      f"failed={stats.get('failed', 0)} "
                      f"({stats.get('duration', 0):.2f}s)")
        print("=" * 70)


@dataclass
class FeatureSet:
    """A named collection of features that should be computed together.

    Useful for grouping related features (e.g. ``elo_features``,
    ``rolling_stats``, ``h2h_stats``) for selective computation.
    """
    name: str = ""
    description: str = ""
    features: list[str] = field(default_factory=list)
    enabled: bool = True
    parallel: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
