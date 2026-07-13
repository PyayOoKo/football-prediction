"""
FeatureLineage — track the provenance of every computed feature value.

Provides a complete audit trail from source data → feature computation →
model training, enabling debugging, reproducibility, and governance.

Concepts
--------
- **Source** — an origin of data (CSV file, API endpoint, database table)
- **Transform** — a computation step (feature engineering function, computer)
- **Feature** — a computed feature value in the store
- **Model** — an ML model that consumes features

The lineage graph is stored as directed edges with metadata in the
database, and can be queried for:
- *Provenance*: given a feature value, trace back to source data
- *Impact*: given a source data change, find all affected features and models
- *Lineage*: full DAG from sources through transforms to features

Architecture
------------
Lineage is recorded as metadata on ``FeatureValue`` rows and in a
dedicated ``feature_lineage`` table (via ``FeatureLineageEntry`` model).
The ``FeatureLineage`` service provides high-level tracking and querying.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text, func, select
from sqlalchemy.orm import Session, Mapped, mapped_column

from src.database.base import Base
from src.feature_store.models import FeatureDefinition, FeatureValue

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  ORM Model
# ═══════════════════════════════════════════════════════════


class FeatureLineageEntry(Base):
    """Records a single lineage edge in the feature provenance DAG.

    Each entry traces one step in the lineage: from a **source**
    or **transform** to a **feature value** or **model**.

    Columns
    -------
    id : str (UUID)
        Primary key.
    feature_definition_id : str, optional
        FK to the feature definition (if this entry is about a feature).
    feature_value_id : str, optional
        FK to the specific feature value row.
    source_type : str
        Type of the source: ``source``, ``transform``, ``feature``, ``model``.
    source_name : str
        Name of the source/transform/model.
    source_version : str, optional
        Version identifier of the source/transform.
    source_metadata : dict
        Arbitrary metadata about this lineage step.
    parent_entry_id : str, optional
        FK to the parent lineage entry (for chaining).
    created_at : datetime
        When this lineage entry was recorded.
    """

    __tablename__ = "feature_lineage"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True,
        default=lambda: __import__("uuid").uuid4().hex[:36],
    )
    feature_definition_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("feature_definitions.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    feature_value_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("feature_values.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
    )
    source_name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True,
    )
    source_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True,
    )
    source_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    parent_entry_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("feature_lineage.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureLineageEntry {self.source_name!r} "
            f"type={self.source_type}>"
        )


# ═══════════════════════════════════════════════════════════
#  Data classes for lineage queries
# ═══════════════════════════════════════════════════════════


@dataclass
class LineageNode:
    """A single node in the lineage graph.

    Attributes
    ----------
    name : str
        Node name (source name, feature name, model name).
    node_type : str
        ``source``, ``transform``, ``feature``, or ``model``.
    version : str, optional
        Version identifier.
    metadata : dict
        Node metadata.
    children : list[LineageNode]
        Downstream nodes (features/models that depend on this).
    parents : list[LineageNode]
        Upstream nodes (sources/transforms this depends on).
    """

    name: str
    node_type: str
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[LineageNode] = field(default_factory=list)
    parents: list[LineageNode] = field(default_factory=list)


@dataclass
class LineageProvenance:
    """Full provenance for a feature value.

    Attributes
    ----------
    feature_name : str
        Feature name.
    feature_version : int
        Feature definition version.
    value : float | str | dict | None
        The actual feature value.
    computed_at : datetime, optional
        When the value was computed.
    computed_by : str
        Computer identifier.
    source_chain : list[dict]
        Ordered list of sources/transforms that produced this value.
    model_consumers : list[str]
        Names of models that consume this feature.
    """

    feature_name: str = ""
    feature_version: int = 0
    value: Any = None
    computed_at: datetime | None = None
    computed_by: str = ""
    source_chain: list[dict[str, Any]] = field(default_factory=list)
    model_consumers: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
#  FeatureLineage Service
# ═══════════════════════════════════════════════════════════


class FeatureLineage:
    """Track and query feature provenance.

    Records lineage entries as features are computed and consumed,
    enabling full traceability from source data through to models.

    Parameters
    ----------
    session : Session
        SQLAlchemy ORM session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Recording ─────────────────────────────────────────

    def record_source(
        self,
        source_name: str,
        *,
        source_version: str | None = None,
        source_metadata: dict[str, Any] | None = None,
    ) -> FeatureLineageEntry:
        """Record a data source (CSV file, API, database table).

        Parameters
        ----------
        source_name : str
            Source identifier (e.g. ``football-data.co.uk``, ``understat``).
        source_version : str, optional
            Data version or fetch timestamp.
        source_metadata : dict, optional
            Extra metadata about the source.

        Returns
        -------
        FeatureLineageEntry
        """
        entry = FeatureLineageEntry(
            source_type="source",
            source_name=source_name,
            source_version=source_version,
            source_metadata=source_metadata or {},
        )
        self._session.add(entry)
        self._session.flush()
        logger.info("Recorded source: %s (v%s)", source_name, source_version or "?")
        return entry

    def record_transform(
        self,
        transform_name: str,
        *,
        transform_version: str | None = None,
        parent_entry: FeatureLineageEntry | None = None,
        transform_metadata: dict[str, Any] | None = None,
    ) -> FeatureLineageEntry:
        """Record a transform/computation step.

        Parameters
        ----------
        transform_name : str
            Transform identifier (e.g. ``elo_computer``, ``feature_engineering``).
        transform_version : str, optional
            Version of the transform.
        parent_entry : FeatureLineageEntry, optional
            Parent lineage entry (the source or transform that feeds this).
        transform_metadata : dict, optional

        Returns
        -------
        FeatureLineageEntry
        """
        entry = FeatureLineageEntry(
            source_type="transform",
            source_name=transform_name,
            source_version=transform_version,
            source_metadata=transform_metadata or {},
            parent_entry_id=parent_entry.id if parent_entry else None,
        )
        self._session.add(entry)
        self._session.flush()
        logger.debug("Recorded transform: %s", transform_name)
        return entry

    def record_feature_computation(
        self,
        definition: FeatureDefinition,
        value: FeatureValue,
        *,
        computed_by: str = "",
        parent_entries: list[FeatureLineageEntry] | None = None,
        computation_metadata: dict[str, Any] | None = None,
    ) -> FeatureLineageEntry:
        """Record that a feature value was computed.

        Links the feature value back to its parent transforms/sources.

        Parameters
        ----------
        definition : FeatureDefinition
            The feature definition.
        value : FeatureValue
            The computed feature value.
        computed_by : str
            Computer identifier.
        parent_entries : list[FeatureLineageEntry], optional
            Lineage entries for transforms/sources that fed this computation.
        computation_metadata : dict, optional

        Returns
        -------
        FeatureLineageEntry
        """
        # Record the transform (computation) step
        transform_entry = self.record_transform(
            transform_name=computed_by or definition.name,
            transform_version=f"v{definition.version}",
            parent_entry=parent_entries[0] if parent_entries else None,
            transform_metadata={
                **(computation_metadata or {}),
                "definition_name": definition.name,
                "definition_version": definition.version,
                "feature_type": definition.feature_type,
            },
        )

        # Record the feature value as an output of the transform
        feature_entry = FeatureLineageEntry(
            feature_definition_id=definition.id,
            feature_value_id=value.id,
            source_type="feature",
            source_name=definition.name,                    source_version=f"v{definition.version}",
            source_metadata={
                "computed_by": computed_by,
                "numeric_value": value.numeric_value,
                "match_id": value.match_id,
                "team_id": value.team_id,
                **(computation_metadata or {}),
            },
            parent_entry_id=transform_entry.id,
        )
        self._session.add(feature_entry)
        self._session.flush()

        if parent_entries and len(parent_entries) > 1:
            for parent in parent_entries[1:]:
                link = FeatureLineageEntry(
                    source_type="transform",
                    source_name=f"parent:{parent.source_name}",
                    source_version=parent.source_version,
                    source_metadata={"note": "additional parent"},
                    parent_entry_id=feature_entry.id,
                )
                self._session.add(link)
            self._session.flush()

        logger.debug(
            "Recorded feature computation: %s v%d for entity",
            definition.name, definition.version,
        )
        return feature_entry

    def record_model_consumption(
        self,
        model_name: str,
        *,
        model_version: str | None = None,
        features_used: list[FeatureDefinition],
        model_metadata: dict[str, Any] | None = None,
    ) -> list[FeatureLineageEntry]:
        """Record that a model consumes a set of features.

        Parameters
        ----------
        model_name : str
            Model name (e.g. ``xgboost_ensemble``).
        model_version : str, optional
            Model version or run ID.
        features_used : list[FeatureDefinition]
            Features consumed by the model.
        model_metadata : dict, optional

        Returns
        -------
        list[FeatureLineageEntry]
            Lineage entries created.
        """
        entries: list[FeatureLineageEntry] = []
        for definition in features_used:
            entry = FeatureLineageEntry(
                feature_definition_id=definition.id,
                source_type="model",
                source_name=model_name,
                source_version=model_version,
                source_metadata={
                    "feature_name": definition.name,
                    "feature_version": definition.version,
                    **(model_metadata or {}),
                },
            )
            self._session.add(entry)
            entries.append(entry)

        self._session.flush()
        logger.info(
            "Recorded model %s (v%s) consuming %d features",
            model_name, model_version or "?", len(features_used),
        )
        return entries

    # ── Querying ──────────────────────────────────────────

    def get_provenance(
        self,
        definition: FeatureDefinition,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
    ) -> LineageProvenance:
        """Get the full provenance chain for a feature value.

        Traces backwards from the feature value through transforms
        to source data, and forwards to model consumers.

        Parameters
        ----------
        definition : FeatureDefinition
        match_id : int, optional
        team_id : int, optional

        Returns
        -------
        LineageProvenance
        """
        provenance = LineageProvenance(
            feature_name=definition.name,
            feature_version=definition.version,
        )

        # Find the feature value
        # Find the feature value
        filters = [FeatureValue.feature_definition_id == definition.id]
        if match_id is not None:
            filters.append(FeatureValue.match_id == match_id)
        if team_id is not None:
            filters.append(FeatureValue.team_id == team_id)

        stmt = select(FeatureValue).where(*filters).order_by(
            FeatureValue.computed_at.desc(),
        ).limit(1)
        fv = self._session.execute(stmt).scalar_one_or_none()

        if fv is None:
            return provenance

        provenance.value = (
            fv.numeric_value if fv.numeric_value is not None
            else fv.text_value or fv.json_value
        )
        provenance.computed_at = fv.computed_at
        provenance.computed_by = fv.computed_by

        # Trace back through lineage entries for this value
        lineage_stmt = (
            select(FeatureLineageEntry)
            .where(
                FeatureLineageEntry.feature_value_id == fv.id,
            )
            .order_by(FeatureLineageEntry.created_at)
        )
        feature_entry = self._session.execute(lineage_stmt).scalar_one_or_none()

        if feature_entry is not None:
            # Walk up the parent chain
            current = feature_entry
            while current is not None:
                provenance.source_chain.insert(0, {
                    "type": current.source_type,
                    "name": current.source_name,
                    "version": current.source_version,
                    "metadata": current.source_metadata or {},
                    "timestamp": current.created_at.isoformat() if current.created_at else None,
                })
                if current.parent_entry_id:
                    parent_stmt = select(FeatureLineageEntry).where(
                        FeatureLineageEntry.id == current.parent_entry_id,
                    )
                    current = self._session.execute(parent_stmt).scalar_one_or_none()
                else:
                    current = None

        # Find model consumers
        model_stmt = (
            select(FeatureLineageEntry)
            .where(
                FeatureLineageEntry.feature_definition_id == definition.id,
                FeatureLineageEntry.source_type == "model",
            )
        )
        model_entries = list(self._session.execute(model_stmt).scalars().all())
        provenance.model_consumers = sorted(set(
            e.source_name for e in model_entries
        ))

        return provenance

    def get_upstream(
        self,
        definition: FeatureDefinition,
        *,
        max_depth: int = 10,
    ) -> list[dict[str, Any]]:
        """Get all upstream sources/transforms for a feature definition.

        Parameters
        ----------
        definition : FeatureDefinition
        max_depth : int
            Maximum traversal depth.

        Returns
        -------
        list[dict]
            Ordered list of upstream lineage nodes.
        """
        stmt = (
            select(FeatureLineageEntry)
            .where(
                FeatureLineageEntry.feature_definition_id == definition.id,
            )
            .order_by(FeatureLineageEntry.created_at.desc())
            .limit(max_depth)
        )
        return [
            {
                "type": e.source_type,
                "name": e.source_name,
                "version": e.source_version,
                "timestamp": e.created_at.isoformat() if e.created_at else None,
            }
            for e in self._session.execute(stmt).scalars().all()
        ]

    def get_downstream(
        self,
        source_name: str,
        *,
        max_depth: int = 10,
    ) -> list[dict[str, Any]]:
        """Find all features that depend on a given source or transform.

        Parameters
        ----------
        source_name : str
            Source or transform name.
        max_depth : int
            Maximum traversal depth.

        Returns
        -------
        list[dict]
            Downstream feature names and versions.
        """
        # Find lineage entries with this source name
        stmt = (
            select(FeatureLineageEntry)
            .where(FeatureLineageEntry.source_name == source_name)
        )
        entries = list(self._session.execute(stmt).scalars().all())

        # Follow parent_entry_id chain forward
        seen: set[str] = set()
        downstream: list[dict[str, Any]] = []
        queue = [e.id for e in entries]

        depth = 0
        while queue and depth < max_depth:
            current_id = queue.pop(0)
            if current_id in seen:
                continue
            seen.add(current_id)

            # Find children (entries that have this as parent)
            child_stmt = (
                select(FeatureLineageEntry)
                .where(FeatureLineageEntry.parent_entry_id == current_id)
            )
            children = list(self._session.execute(child_stmt).scalars().all())

            for child in children:
                downstream.append({
                    "type": child.source_type,
                    "name": child.source_name,
                    "version": child.source_version,
                    "feature_definition_id": child.feature_definition_id,
                })
                queue.append(child.id)

            depth += 1

        return downstream

    def get_source_summary(self) -> list[dict[str, Any]]:
        """Get a summary of all sources and their downstream feature counts.

        Returns
        -------
        list[dict]
            Each dict has: ``source_name``, ``source_type``, ``downstream_features`` count.
        """
        stmt = (
            select(FeatureLineageEntry)
            .where(FeatureLineageEntry.source_type == "source")
            .distinct()
        )
        sources = list(self._session.execute(stmt).scalars().all())

        summary: list[dict[str, Any]] = []
        for source in sources:
            downstream = self.get_downstream(source.source_name)
            feature_names = sorted(set(
                d["name"] for d in downstream
                if d["type"] == "feature"
            ))
            summary.append({
                "source_name": source.source_name,
                "source_version": source.source_version,
                "downstream_features": feature_names,
                "feature_count": len(feature_names),
                "total_downstream_entries": len(downstream),
            })

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize all lineage entries for export/reporting."""
        stmt = select(FeatureLineageEntry).order_by(
            FeatureLineageEntry.created_at,
        )
        entries = list(self._session.execute(stmt).scalars().all())

        return {
            "total_entries": len(entries),
            "sources": len([e for e in entries if e.source_type == "source"]),
            "transforms": len([e for e in entries if e.source_type == "transform"]),
            "features": len([e for e in entries if e.source_type == "feature"]),
            "models": len([e for e in entries if e.source_type == "model"]),
            "entries": [
                {
                    "id": e.id,
                    "source_type": e.source_type,
                    "source_name": e.source_name,
                    "source_version": e.source_version,
                    "feature_definition_id": e.feature_definition_id,
                    "parent_entry_id": e.parent_entry_id,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in entries
            ],
        }
