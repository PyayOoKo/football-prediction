"""
SQLAlchemy ORM models for the Feature Store.

Tables
------
feature_definitions
    Registry of all features with type, category, version, and metadata.
feature_values
    Computed feature values keyed by definition + entity (match/team).
feature_dependencies
    Directed edges in the feature dependency DAG.
feature_versions
    Version history for feature definitions.
feature_computation_batches
    Audit trail for batch computation runs.

All tables use ``UUID`` primary keys for distributed compatibility
and ``TIMESTAMP WITH TIME ZONE`` for timezone-safe storage.
Note: Uses ``sqlalchemy.JSON`` (not PostgreSQL-specific ``JSONB``) so
the models work with both SQLite (tests) and PostgreSQL (production).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base


# ═══════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════


class FeatureStatus(str, enum.Enum):
    """Lifecycle status of a feature definition."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class FeatureCategory(str, enum.Enum):
    """Thematic category grouping related features."""

    ROLLING_STAT = "rolling_stat"
    TEAM_FORM = "team_form"
    ELO_RATING = "elo_rating"
    ATTACK_STRENGTH = "attack_strength"
    DEFENSE_STRENGTH = "defense_strength"
    HOME_ADVANTAGE = "home_advantage"
    AWAY_ADVANTAGE = "away_advantage"
    REST_DAYS = "rest_days"
    FIXTURE_CONGESTION = "fixture_congestion"
    LEAGUE_STRENGTH = "league_strength"
    TEAM_MOMENTUM = "team_momentum"
    MARKET_MOVEMENT = "market_movement"
    H2H_STAT = "h2h_stat"
    XG_FEATURE = "xg_feature"
    ODDS_FEATURE = "odds_feature"
    COMPOSITE = "composite"


class EntityType(str, enum.Enum):
    """The entity to which a feature value belongs."""

    MATCH = "match"
    TEAM = "team"
    LEAGUE = "league"
    PLAYER = "player"
    GLOBAL = "global"


# ═══════════════════════════════════════════════════════════
#  Helper: UUID primary key column
# ═══════════════════════════════════════════════════════════

# Use String(36) for SQLite compatibility; PostgreSQL via sqlalchemy.UUID
# works transparently if the engine supports it.  We keep PK as text so
# the same model works in tests (SQLite) and production (PostgreSQL).
def _uuid_pk() -> Any:
    return mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))


# ═══════════════════════════════════════════════════════════
#  ORM Models
# ═══════════════════════════════════════════════════════════


class FeatureDefinition(Base):
    """
    Feature registry — defines what features exist and how they behave.

    Each row represents a single, versioned feature definition. Features
    can be versioned independently: changing the computation logic or
    parameters produces a new row with an incremented ``version``.

    Columns
    -------
    id : str (UUID)
        Primary key.
    name : str
        Unique feature name (e.g. ``home_attack_strength_10``).
    version : int
        Feature version (incremented when definition changes).
    feature_type : str
        Semantic type label (e.g. ``rolling_stat``, ``elo``).
    category : FeatureCategory
        Thematic category for grouping.
    entity_type : EntityType
        What kind of entity this feature describes.
    description : str
        Human-readable documentation.
    computation_params : dict
        JSON parameters used by the computer (window size, K-factor, etc.).
    validation_rules : dict
        JSON validation rules (bounds, nullability, cardinality).
    dependencies : list[str]
        Feature names this feature depends on (soft references).
    status : FeatureStatus
        Lifecycle status.
    extra_metadata : dict
        Arbitrary JSON metadata (author, source, tags, etc.).
    is_active : bool
        Quick filter for active features.
    created_at : datetime
        Row creation timestamp.
    updated_at : datetime
        Row last-update timestamp.
    """

    __tablename__ = "feature_definitions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_feature_def_name_version"),
    )

    id: Mapped[str] = _uuid_pk()
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True,
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
    )
    feature_type: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True,
    )
    category: Mapped[FeatureCategory] = mapped_column(
        Enum(FeatureCategory, name="feature_category"), nullable=False,
    )
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType, name="entity_type"), nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    computation_params: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    validation_rules: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    dependencies: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True, default=list,
    )
    status: Mapped[FeatureStatus] = mapped_column(
        Enum(FeatureStatus, name="feature_status"),
        nullable=False, default=FeatureStatus.DRAFT,
    )
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True, default=dict,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )

    # Relationships
    values: Mapped[list[FeatureValue]] = relationship(
        "FeatureValue", back_populates="definition",
        cascade="all, delete-orphan",
    )
    dependency_edges: Mapped[list[FeatureDependency]] = relationship(
        "FeatureDependency",
        foreign_keys="FeatureDependency.dependent_feature_id",
        back_populates="dependent_feature",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<FeatureDefinition {self.name!r} v{self.version}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "feature_type": self.feature_type,
            "category": self.category.value if self.category else None,
            "entity_type": self.entity_type.value if self.entity_type else None,
            "description": self.description,
            "status": self.status.value if self.status else None,
            "is_active": self.is_active,
            "computation_params": self.computation_params or {},
            "validation_rules": self.validation_rules or {},
            "dependencies": self.dependencies or [],
            "metadata": self.extra_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class FeatureValue(Base):
    """
    Stores a single computed feature value for a specific entity.

    Values are stored in a flexible format to support different
    feature types. At least one of ``numeric_value``, ``text_value``,
    or ``json_value`` should be populated.

    Columns
    -------
    id : str (UUID)
        Primary key.
    feature_definition_id : str
        FK to ``feature_definitions``.
    match_id : int, optional
        FK to ``matches.id`` (for match-level features).
    team_id : int, optional
        FK to ``teams.id`` (for team-level features).
    league_id : int, optional
        FK to ``competitions.id`` (for league-level features).
    numeric_value : float, optional
        Primary value storage for scalar features.
    text_value : str, optional
        String value for categorical/text features.
    json_value : dict, optional
        Complex value for multi-dimensional features.
    computed_at : datetime
        When this value was computed.
    computed_by : str
        Identifier of the computer that produced this value.
    batch_id : str, optional
        FK to ``feature_computation_batches``.
    created_at : datetime
        Row creation timestamp.
    """

    __tablename__ = "feature_values"
    __table_args__ = (
        UniqueConstraint(
            "feature_definition_id", "match_id", "team_id",
            name="uq_feature_value_entity",
        ),
    )

    id: Mapped[str] = _uuid_pk()
    feature_definition_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("feature_definitions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    match_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True,
    )
    team_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True,
    )
    league_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True,
    )
    numeric_value: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    text_value: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    json_value: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    computed_by: Mapped[str] = mapped_column(
        String(255), nullable=False, default="",
    )
    batch_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("feature_computation_batches.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    # Relationships
    definition: Mapped[FeatureDefinition] = relationship(
        "FeatureDefinition", back_populates="values",
    )
    batch: Mapped[FeatureComputationBatch | None] = relationship(
        "FeatureComputationBatch", back_populates="values",
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureValue def={self.feature_definition_id} "
            f"match={self.match_id} team={self.team_id} "
            f"val={self.numeric_value or self.text_value}>"
        )


class FeatureDependency(Base):
    """
    Directed edge in the feature dependency DAG.

    Defines which features must be computed before others. The DAG is
    enforced by ``FeatureRegistry`` when planning computation batches.

    Columns
    -------
    id : str (UUID)
        Primary key.
    dependent_feature_id : str
        The feature that *depends on* another (FK to ``feature_definitions``).
    dependency_feature_id : str
        The feature that is a *prerequisite* (FK to ``feature_definitions``).
    is_hard : bool
        ``True`` = this dependency is required (value must exist).
        ``False`` = soft dependency (used if available, not an error if missing).
    created_at : datetime
        Row creation timestamp.
    """

    __tablename__ = "feature_dependencies"
    __table_args__ = (
        UniqueConstraint(
            "dependent_feature_id", "dependency_feature_id",
            name="uq_feature_dep_edge",
        ),
    )

    id: Mapped[str] = _uuid_pk()
    dependent_feature_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("feature_definitions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    dependency_feature_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("feature_definitions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    is_hard: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    # Relationships
    dependent_feature: Mapped[FeatureDefinition] = relationship(
        "FeatureDefinition",
        foreign_keys=[dependent_feature_id],
        back_populates="dependency_edges",
    )
    dependency_feature: Mapped[FeatureDefinition] = relationship(
        "FeatureDefinition",
        foreign_keys=[dependency_feature_id],
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureDependency {self.dependent_feature_id} "
            f"-> {self.dependency_feature_id} hard={self.is_hard}>"
        )


class FeatureVersion(Base):
    """
    Version history for feature definitions.

    Each time a feature definition is updated, a new ``FeatureVersion``
    row records the change alongside the updated ``FeatureDefinition``.
    Enables rollback and audit of feature changes over time.

    Columns
    -------
    id : str (UUID)
        Primary key.
    feature_definition_id : str
        FK to ``feature_definitions``.
    version : int
        Version number.
    is_current : bool
        Whether this is the active version.
    changelog : str
        Description of what changed.
    snapshot : dict
        Full JSON snapshot of the definition at this version.
    created_at : datetime
        When this version was created.
    """

    __tablename__ = "feature_versions"
    __table_args__ = (
        UniqueConstraint(
            "feature_definition_id", "version",
            name="uq_feature_version_number",
        ),
    )

    id: Mapped[str] = _uuid_pk()
    feature_definition_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("feature_definitions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    changelog: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureVersion def={self.feature_definition_id} "
            f"v{self.version} current={self.is_current}>"
        )


class FeatureComputationBatch(Base):
    """
    Audit trail for a batch computation run.

    Records metadata about a single computation batch so you can trace
    which values were produced by which run, compare performance, and
    debug issues.

    Columns
    -------
    id : str (UUID)
        Primary key.
    batch_label : str
        Human-readable label (e.g. ``daily-2026-07-13``).
    trigger : str
        How the batch was triggered: ``manual``, ``scheduled``, ``pipeline``.
    features_computed : list[str]
        Feature names that were computed in this batch.
    entity_count : int
        Number of entities (matches/teams) processed.
    started_at : datetime
        Batch start timestamp.
    completed_at : datetime, optional
        Batch completion timestamp.
    duration_seconds : float, optional
        Total computation duration.
    success : bool
        Whether the batch completed successfully.
    error_message : str, optional
        Error details if failed.
    extra_metadata : dict
        Arbitrary JSON metadata (hostname, version info, etc.).
    created_at : datetime
        Row creation timestamp.
    """

    __tablename__ = "feature_computation_batches"

    id: Mapped[str] = _uuid_pk()
    batch_label: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True,
    )
    trigger: Mapped[str] = mapped_column(
        String(50), nullable=False, default="manual",
    )
    features_computed: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    entity_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    success: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True, default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    # Relationships
    values: Mapped[list[FeatureValue]] = relationship(
        "FeatureValue", back_populates="batch",
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureComputationBatch {self.batch_label!r} "
            f"trigger={self.trigger} success={self.success}>"
        )

    def complete(self, success: bool = True, error: str | None = None) -> None:
        """Mark this batch as completed with timing and status."""
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()
        self.success = success
        if error:
            self.error_message = error
