"""
FeatureStore — core CRUD operations for feature values.

Provides:
- Store and retrieve feature values by definition + entity
- Batch inserts for efficient bulk computation
- Incremental update support (only update stale/missing values)
- Computation batch tracking with timing and audit
- Query methods for model input assembly
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from src.feature_store.models import (
    FeatureComputationBatch,
    FeatureDefinition,
    FeatureValue,
)

logger = logging.getLogger(__name__)


class FeatureStore:
    """Core CRUD store for feature values.

    Handles storing, retrieving, and batch-managing computed feature
    values. All operations go through the provided SQLAlchemy ``Session``.

    Parameters
    ----------
    session : Session
        SQLAlchemy ORM session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Single-value operations ───────────────────────────

    def set(
        self,
        definition_id: str,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
        league_id: int | None = None,
        numeric_value: float | None = None,
        text_value: str | None = None,
        json_value: dict[str, Any] | None = None,
        computed_by: str = "",
        batch_id: str | None = None,
    ) -> FeatureValue:
        """Set a single feature value.

        Creates a new ``FeatureValue`` or updates the existing one for
        the specified definition + entity.

        Parameters
        ----------
        definition_id : UUID
            FK to ``feature_definitions``.
        match_id : int, optional
            Match entity ID.
        team_id : int, optional
            Team entity ID.
        league_id : int, optional
            League entity ID.
        numeric_value : float, optional
            Scalar value.
        text_value : str, optional
            String value.
        json_value : dict, optional
            Complex value.
        computed_by : str
            Computer identifier.
        batch_id : UUID, optional
            Computation batch ID.

        Returns
        -------
        FeatureValue
            The stored value (new or updated).
        """
        existing = self._find_existing(definition_id, match_id, team_id, league_id)

        if existing is not None:
            existing.numeric_value = numeric_value
            existing.text_value = text_value
            existing.json_value = json_value
            existing.computed_at = datetime.now(timezone.utc)
            existing.computed_by = computed_by
            if batch_id:
                existing.batch_id = batch_id
            value = existing
        else:
            value = FeatureValue(
                feature_definition_id=definition_id,
                match_id=match_id,
                team_id=team_id,
                league_id=league_id,
                numeric_value=numeric_value,
                text_value=text_value,
                json_value=json_value,
                computed_by=computed_by,
                batch_id=batch_id,
            )
            self._session.add(value)

        self._session.flush()
        return value

    def get(
        self,
        definition_id: str,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
        league_id: int | None = None,
    ) -> FeatureValue | None:
        """Get a single feature value for the specified entity.

        Parameters
        ----------
        definition_id : UUID
            Feature definition ID.
        match_id : int, optional
        team_id : int, optional
        league_id : int, optional

        Returns
        -------
        FeatureValue | None
        """
        return self._find_existing(definition_id, match_id, team_id, league_id)

    def get_value(
        self,
        definition_id: str,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
        league_id: int | None = None,
    ) -> float | str | dict[str, Any] | None:
        """Get only the value (numeric, text, or json) for a feature.

        Convenience method that unwraps the ``FeatureValue`` object.

        Returns
        -------
        float | str | dict | None
        """
        fv = self.get(
            definition_id, match_id=match_id, team_id=team_id, league_id=league_id,
        )
        if fv is None:
            return None
        return fv.numeric_value or fv.text_value or fv.json_value

    def _find_existing(
        self,
        definition_id: str,
        match_id: int | None,
        team_id: int | None,
        league_id: int | None,
    ) -> FeatureValue | None:
        """Find an existing feature value row for the given entity."""
        filters = [FeatureValue.feature_definition_id == definition_id]

        if match_id is not None:
            filters.append(FeatureValue.match_id == match_id)
        elif team_id is not None:
            filters.append(FeatureValue.team_id == team_id)
        elif league_id is not None:
            filters.append(FeatureValue.league_id == league_id)
        else:
            # Global feature — match_id, team_id, league_id are all NULL
            filters.append(FeatureValue.match_id.is_(None))
            filters.append(FeatureValue.team_id.is_(None))
            filters.append(FeatureValue.league_id.is_(None))

        stmt = select(FeatureValue).where(and_(*filters))
        return self._session.execute(stmt).scalar_one_or_none()

    # ── Batch operations ──────────────────────────────────

    def set_many(
        self,
        values: list[dict[str, Any]],
    ) -> int:
        """Insert or update multiple feature values efficiently.

        Each dict in ``values`` should contain:
        ``definition_id``, and at least one entity ID key
        (``match_id`` / ``team_id`` / ``league_id``) plus at least
        one value key (``numeric_value`` / ``text_value`` / ``json_value``).

        Parameters
        ----------
        values : list[dict]
            Feature value records to upsert.

        Returns
        -------
        int
            Number of records processed.
        """
        count = 0
        for v in values:
            self.set(
                definition_id=v["definition_id"],
                match_id=v.get("match_id"),
                team_id=v.get("team_id"),
                league_id=v.get("league_id"),
                numeric_value=v.get("numeric_value"),
                text_value=v.get("text_value"),
                json_value=v.get("json_value"),
                computed_by=v.get("computed_by", ""),
                batch_id=v.get("batch_id"),
            )
            count += 1

        self._session.flush()
        logger.debug("Stored %d feature values in batch", count)
        return count

    def get_many(
        self,
        definition_ids: list[str],
        *,
        match_ids: list[int] | None = None,
        team_ids: list[int] | None = None,
    ) -> list[FeatureValue]:
        """Get multiple feature values by definition + entity keys.

        Parameters
        ----------
        definition_ids : list[UUID]
        match_ids : list[int], optional
        team_ids : list[int], optional

        Returns
        -------
        list[FeatureValue]
        """
        filters = [FeatureValue.feature_definition_id.in_(definition_ids)]

        if match_ids:
            filters.append(FeatureValue.match_id.in_(match_ids))
        if team_ids:
            filters.append(FeatureValue.team_id.in_(team_ids))

        stmt = select(FeatureValue).where(and_(*filters))
        return list(self._session.execute(stmt).scalars().all())

    def delete(
        self,
        definition_id: str,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
    ) -> bool:
        """Delete a single feature value.

        Parameters
        ----------
        definition_id : UUID
        match_id : int, optional
        team_id : int, optional

        Returns
        -------
        bool
            True if a row was deleted.
        """
        filters = [FeatureValue.feature_definition_id == definition_id]
        if match_id is not None:
            filters.append(FeatureValue.match_id == match_id)
        if team_id is not None:
            filters.append(FeatureValue.team_id == team_id)

        stmt = select(FeatureValue).where(and_(*filters))
        fv = self._session.execute(stmt).scalar_one_or_none()
        if fv is not None:
            self._session.delete(fv)
            self._session.flush()
            return True
        return False

    def delete_all_for_definition(self, definition_id: uuid.UUID) -> int:
        """Delete all values for a feature definition.

        Parameters
        ----------
        definition_id : UUID

        Returns
        -------
        int
            Number of rows deleted.
        """
        stmt = select(FeatureValue).where(
            FeatureValue.feature_definition_id == definition_id,
        )
        rows = list(self._session.execute(stmt).scalars().all())
        for row in rows:
            self._session.delete(row)
        self._session.flush()
        logger.info("Deleted %d values for definition %s", len(rows), definition_id)
        return len(rows)

    # ── Computation batch tracking ────────────────────────

    def start_batch(
        self,
        batch_label: str,
        trigger: str = "manual",
        features_computed: list[str] | None = None,
        entity_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureComputationBatch:
        """Start a new computation batch.

        Parameters
        ----------
        batch_label : str
            Human-readable label.
        trigger : str
            How triggered: ``manual``, ``scheduled``, ``pipeline``.
        features_computed : list[str], optional
            Feature names to compute.
        entity_count : int
            Entities to process.
        metadata : dict, optional
            Extra metadata.

        Returns
        -------
        FeatureComputationBatch
        """
        batch = FeatureComputationBatch(
            batch_label=batch_label,
            trigger=trigger,
            features_computed=features_computed or [],
            entity_count=entity_count,
            extra_metadata=metadata or {},
        )
        self._session.add(batch)
        self._session.flush()
        logger.info(
            "Started batch %s: %s (%d entities)",
            batch.id, batch_label, entity_count,
        )
        return batch

    def complete_batch(
        self,
        batch_id: str,
        success: bool = True,
        error: str | None = None,
    ) -> FeatureComputationBatch:
        """Mark a computation batch as completed.

        Parameters
        ----------
        batch_id : UUID
        success : bool
        error : str, optional

        Returns
        -------
        FeatureComputationBatch
        """
        batch = self._session.get(FeatureComputationBatch, batch_id)
        if batch is None:
            raise ValueError(f"Batch {batch_id} not found.")
        batch.complete(success=success, error=error)
        self._session.flush()
        logger.info(
            "Completed batch %s: success=%s duration=%.2fs",
            batch_id, success, batch.duration_seconds or 0,
        )
        return batch

    def get_batch(self, batch_id: uuid.UUID) -> FeatureComputationBatch | None:
        """Get a computation batch by ID."""
        return self._session.get(FeatureComputationBatch, batch_id)

    def list_batches(
        self,
        *,
        trigger: str | None = None,
        success: bool | None = None,
        limit: int = 20,
    ) -> list[FeatureComputationBatch]:
        """List recent computation batches.

        Parameters
        ----------
        trigger : str, optional
            Filter by trigger type.
        success : bool, optional
            Filter by success status.
        limit : int
            Max results (default 20).

        Returns
        -------
        list[FeatureComputationBatch]
        """
        stmt = select(FeatureComputationBatch).order_by(
            FeatureComputationBatch.created_at.desc(),
        ).limit(limit)

        if trigger is not None:
            stmt = stmt.where(FeatureComputationBatch.trigger == trigger)
        if success is not None:
            stmt = stmt.where(FeatureComputationBatch.success == success)

        return list(self._session.execute(stmt).scalars().all())

    # ── Incremental updates ───────────────────────────────

    def needs_update(
        self,
        definition_id: str,
        entity_ids: list[int],
        *,
        entity_type: str = "match",
        max_age_hours: float = 24,
    ) -> list[int]:
        """Identify entities that need a feature value recomputed.

        An entity "needs update" if:
        - No value exists for it, OR
        - The existing value is older than ``max_age_hours``

        Parameters
        ----------
        definition_id : UUID
            Feature definition to check.
        entity_ids : list[int]
            Entity IDs to check (match or team IDs).
        entity_type : str
            ``match`` or ``team``.
        max_age_hours : float
            Maximum age for a value to be considered fresh.

        Returns
        -------
        list[int]
            Entity IDs that need recomputation.
        """
        if not entity_ids:
            return []

        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)

        # Load existing values for these entities
        id_col = (
            FeatureValue.match_id if entity_type == "match"
            else FeatureValue.team_id
        )
        stmt = select(FeatureValue).where(
            FeatureValue.feature_definition_id == definition_id,
            id_col.in_(entity_ids),
        )
        existing = {
            (getattr(v, id_col.key) if hasattr(v, entity_type + "_id") else 0): v
            for v in self._session.execute(stmt).scalars().all()
        }

        stale: list[int] = []
        for eid in entity_ids:
            value = existing.get(eid)
            if value is None:
                stale.append(eid)
            elif value.computed_at.timestamp() < cutoff:
                stale.append(eid)

        return stale

    # ── Assembly for model input ──────────────────────────

    def assemble_feature_vector(
        self,
        definition_ids: list[str],
        *,
        match_id: int | None = None,
        team_ids: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """Assemble a flat feature vector for model inference.

        For a given match, retrieves all specified feature values
        and returns them as a ``{feature_name: value}`` dict.

        Parameters
        ----------
        definition_ids : list[UUID]
            Feature definitions to include.
        match_id : int, optional
            Match ID for match-level features.
        team_ids : dict[str, int], optional
            Team role -> ID mapping (e.g. ``{"home": 42, "away": 17}``).

        Returns
        -------
        dict[str, float]
            Feature names to numeric values.
        """
        from sqlalchemy.orm import joinedload

        result: dict[str, float] = {}

        # Load definitions for names
        stmt = select(FeatureDefinition).where(
            FeatureDefinition.id.in_(definition_ids),
        )
        definitions = {
            d.id: d for d in self._session.execute(stmt).scalars().all()
        }

        # Load values
        values = self._session.execute(
            select(FeatureValue)
            .options(joinedload(FeatureValue.definition))
            .where(FeatureValue.feature_definition_id.in_(definition_ids))
        ).scalars().all()

        for fv in values:
            # Check entity match
            if fv.match_id is not None and fv.match_id != match_id:
                continue

            # For team-level features, check if the team matches
            if fv.team_id is not None and team_ids:
                if fv.team_id not in team_ids.values():
                    continue

            # For league-level features
            if fv.league_id is not None:
                # Include all league-level features
                pass

            name = definitions.get(fv.feature_definition_id)
            if name is None:
                continue

            value = fv.numeric_value
            if value is not None:
                result[name.name] = value

        return result
