"""
FeatureRegistry — central catalog for feature definitions.

Responsibilities:
- Register new feature definitions with versioning
- Look up features by name, category, entity type, status
- Manage the dependency DAG (topological sort, cycle detection)
- Validate feature definitions before registration
- Track feature lifecycle (draft → active → deprecated → retired)
- Record version history for audit and rollback
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.feature_store.models import (
    EntityType,
    FeatureCategory,
    FeatureDefinition,
    FeatureDependency,
    FeatureStatus,
    FeatureVersion,
)

logger = logging.getLogger(__name__)

# ── Feature type constants ────────────────────────────────
FEATURE_TYPES = [
    "rolling_stat", "team_form", "elo", "attack_strength",
    "defense_strength", "home_advantage", "away_advantage",
    "rest_days", "fixture_congestion", "league_strength",
    "team_momentum", "market_movement", "h2h_stat",
    "xg_feature", "odds_feature", "composite",
]


class FeatureRegistry:
    """Central feature definition registry backed by SQLAlchemy.

    Provides lookup, registration, versioning, dependency management,
    and lifecycle transitions for all features in the store.

    Parameters
    ----------
    session : Session
        SQLAlchemy ORM session for database access.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Registration ──────────────────────────────────────

    def register(
        self,
        name: str,
        feature_type: str,
        category: FeatureCategory | str,
        entity_type: str,
        *,
        description: str | None = None,
        computation_params: dict[str, Any] | None = None,
        validation_rules: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        version: int = 1,
        status: FeatureStatus | str = FeatureStatus.DRAFT,
        changelog: str | None = None,
    ) -> FeatureDefinition:
        """Register a new feature definition.

        Parameters
        ----------
        name : str
            Unique feature name (e.g. ``home_attack_strength_10``).
        feature_type : str
            Semantic type label (e.g. ``rolling_stat``, ``elo``).
        category : FeatureCategory | str
            Thematic category.
        entity_type : str
            Entity this feature belongs to (match/team/league/player/global).
        description : str, optional
            Human-readable documentation.
        computation_params : dict, optional
            Parameters used by the computer (window, K-factor, etc.).
        validation_rules : dict, optional
            Validation constraints (bounds, nullable, cardinality).
        dependencies : list[str], optional
            Feature names this feature depends on.
        metadata : dict, optional
            Arbitrary metadata (author, source, tags).
        version : int
            Version number (default 1).
        status : FeatureStatus | str
            Lifecycle status (default DRAFT).
        changelog : str, optional
            Description of changes for this version.

        Returns
        -------
        FeatureDefinition
            The newly created definition.

        Raises
        ------
        ValueError
            If validation fails or a duplicate exists.
        """
        if isinstance(category, str):
            category = FeatureCategory(category)
        if isinstance(status, str):
            status = FeatureStatus(status)

        # Validate
        self._validate_registration(
            name, feature_type, category, entity_type, version,
        )

        # Check for existing definition with same name + version
        existing = self._get_by_name_version(name, version)
        if existing is not None:
            raise ValueError(
                f"Feature {name!r} version {version} already exists "
                f"(id={existing.id}). Use ``new_version()`` to create a "
                f"new version."
            )

        definition = FeatureDefinition(
            name=name,
            version=version,
            feature_type=feature_type,
            category=category,
            entity_type=EntityType(entity_type) if isinstance(entity_type, str) else entity_type,
            description=description,
            computation_params=computation_params or {},
            validation_rules=validation_rules or {},
            dependencies=dependencies or [],
            status=status,
            extra_metadata=metadata or {},
            is_active=(status == FeatureStatus.ACTIVE),
        )
        self._session.add(definition)
        self._session.flush()  # Get ID

        # Record version history
        self._record_version(definition, changelog or "Initial registration")

        # Create dependency edges
        if dependencies:
            self._create_dependency_edges(definition, dependencies)

        logger.info(
            "Registered feature %r v%d (type=%s, category=%s)",
            name, version, feature_type, category.value,
        )
        return definition

    def new_version(
        self,
        name: str,
        *,
        changelog: str = "",
        **updates: Any,
    ) -> FeatureDefinition:
        """Create a new version of an existing feature definition.

        Increments the version number and copies fields from the
        current definition, applying any ``**updates`` overrides.

        Parameters
        ----------
        name : str
            Feature name.
        changelog : str
            Description of what changed in this version.
        **updates : Any
            Fields to override in the new version (e.g. ``computation_params``,
            ``validation_rules``, ``description``, ``status``).

        Returns
        -------
        FeatureDefinition
            The new version.

        Raises
        ------
        ValueError
            If no active definition exists for *name*.
        """
        current = self.latest(name)
        if current is None:
            raise ValueError(
                f"No active feature definition found for {name!r}. "
                f"Register it first with ``register()``."
            )

        new_version = current.version + 1

        # Apply defaults from current, overrides from **updates
        kwargs = {
            "name": current.name,
            "feature_type": current.feature_type,
            "category": current.category,
            "entity_type": current.entity_type,
            "description": updates.pop("description", current.description),
            "computation_params": updates.pop(
                "computation_params", current.computation_params,
            ),
            "validation_rules": updates.pop(
                "validation_rules", current.validation_rules,
            ),
            "dependencies": updates.pop("dependencies", current.dependencies),
            "metadata": updates.pop("metadata", current.extra_metadata),
            "version": new_version,
            "status": updates.pop("status", FeatureStatus.DRAFT),
            "changelog": changelog,
        }
        # Any remaining **updates go into metadata
        if updates:
            kwargs["metadata"] = {
                **(kwargs.get("metadata") or {}),
                **updates,
            }

        return self.register(**kwargs)

    def _validate_registration(
        self,
        name: str,
        feature_type: str,
        category: FeatureCategory,
        entity_type: str,
        version: int,
    ) -> None:
        """Validate registration parameters before creating.

        Raises
        ------
        ValueError
            If any parameter fails validation.
        """
        if not name or not name.strip():
            raise ValueError("Feature name cannot be empty.")
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Feature name {name!r} contains invalid characters. "
                f"Use only letters, numbers, underscores, and hyphens."
            )
        if feature_type not in FEATURE_TYPES:
            raise ValueError(
                f"Unknown feature type {feature_type!r}. "
                f"Must be one of {FEATURE_TYPES}."
            )
        if entity_type not in [e.value for e in EntityType]:
            raise ValueError(
                f"Unknown entity type {entity_type!r}. "
                f"Must be one of {[e.value for e in EntityType]}."
            )
        if version < 1:
            raise ValueError("Feature version must be >= 1.")

    # ── Lookup ────────────────────────────────────────────

    def get(self, definition_id: str) -> FeatureDefinition | None:
        """Get a feature definition by string ID."""
        return self._session.get(FeatureDefinition, definition_id)

    def latest(self, name: str) -> FeatureDefinition | None:
        """Get the latest (highest version) definition for a feature name."""
        stmt = (
            select(FeatureDefinition)
            .where(FeatureDefinition.name == name)
            .order_by(FeatureDefinition.version.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_name_version(
        self, name: str, version: int,
    ) -> FeatureDefinition | None:
        """Get a specific version of a feature definition."""
        return self._get_by_name_version(name, version)

    def _get_by_name_version(
        self, name: str, version: int,
    ) -> FeatureDefinition | None:
        stmt = (
            select(FeatureDefinition)
            .where(
                FeatureDefinition.name == name,
                FeatureDefinition.version == version,
            )
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list(
        self,
        *,
        category: FeatureCategory | str | None = None,
        feature_type: str | None = None,
        entity_type: str | None = None,
        status: FeatureStatus | str | None = None,
        is_active: bool | None = None,
    ) -> list[FeatureDefinition]:
        """List feature definitions with optional filters.

        Parameters
        ----------
        category : FeatureCategory, optional
            Filter by category.
        feature_type : str, optional
            Filter by feature type.
        entity_type : str, optional
            Filter by entity type.
        status : FeatureStatus, optional
            Filter by lifecycle status.
        is_active : bool, optional
            Filter by active flag.

        Returns
        -------
        list[FeatureDefinition]
        """
        stmt = select(FeatureDefinition)

        if category is not None:
            if isinstance(category, str):
                category = FeatureCategory(category)
            stmt = stmt.where(FeatureDefinition.category == category)
        if feature_type is not None:
            stmt = stmt.where(FeatureDefinition.feature_type == feature_type)
        if entity_type is not None:
            stmt = stmt.where(FeatureDefinition.entity_type == EntityType(entity_type))
        if status is not None:
            if isinstance(status, str):
                status = FeatureStatus(status)
            stmt = stmt.where(FeatureDefinition.status == status)
        if is_active is not None:
            stmt = stmt.where(FeatureDefinition.is_active == is_active)

        stmt = stmt.order_by(FeatureDefinition.name, FeatureDefinition.version.desc())
        return list(self._session.execute(stmt).scalars().all())

    def count(self) -> int:
        """Return the total number of feature definitions."""
        from sqlalchemy import func as sa_func

        stmt = select(sa_func.count(FeatureDefinition.id))
        return self._session.execute(stmt).scalar() or 0

    # ── Lifecycle management ──────────────────────────────

    def activate(self, name: str, version: int | None = None) -> FeatureDefinition:
        """Set a feature definition to ACTIVE status.

        Parameters
        ----------
        name : str
            Feature name.
        version : int, optional
            Specific version to activate. Defaults to latest.

        Returns
        -------
        FeatureDefinition
        """
        if version is not None:
            definition = self._get_by_name_version(name, version)
        else:
            definition = self.latest(name)

        if definition is None:
            raise ValueError(f"Feature {name!r} v{version or 'latest'} not found.")

        definition.status = FeatureStatus.ACTIVE
        definition.is_active = True
        self._session.flush()
        self._record_version(definition, "Activated")
        logger.info("Activated feature %r v%d", name, definition.version)
        return definition

    def deprecate(self, name: str, reason: str = "") -> FeatureDefinition:
        """Mark a feature definition as DEPRECATED.

        Deprecated features remain available for existing values but
        are not used for new computation by default.

        Parameters
        ----------
        name : str
            Feature name.
        reason : str
            Reason for deprecation.

        Returns
        -------
        FeatureDefinition
        """
        definition = self.latest(name)
        if definition is None:
            raise ValueError(f"Feature {name!r} not found.")

        definition.status = FeatureStatus.DEPRECATED
        definition.is_active = False
        self._session.flush()
        self._record_version(definition, f"Deprecated: {reason}" if reason else "Deprecated")
        logger.info("Deprecated feature %r v%d", name, definition.version)
        return definition

    def retire(self, name: str, reason: str = "") -> FeatureDefinition:
        """Mark a feature definition as RETIRED (permanently removed).

        Retired features are kept for historical audit but should not
        be used in any new models or computations.

        Parameters
        ----------
        name : str
            Feature name.
        reason : str
            Reason for retirement.

        Returns
        -------
        FeatureDefinition
        """
        # Retire ALL versions
        stmt = select(FeatureDefinition).where(FeatureDefinition.name == name)
        definitions = list(self._session.execute(stmt).scalars().all())
        if not definitions:
            raise ValueError(f"Feature {name!r} not found.")

        for definition in definitions:
            definition.status = FeatureStatus.RETIRED
            definition.is_active = False
        self._session.flush()
        logger.info("Retired feature %r (%d versions)", name, len(definitions))
        return definitions[0]

    # ── Version history ───────────────────────────────────

    def get_history(
        self, name: str,
    ) -> list[FeatureVersion]:
        """Get version history for a feature definition.

        Parameters
        ----------
        name : str
            Feature name.

        Returns
        -------
        list[FeatureVersion]
            Sorted by version descending.
        """
        stmt = (
            select(FeatureVersion)
            .join(FeatureDefinition)
            .where(FeatureDefinition.name == name)
            .order_by(FeatureVersion.version.desc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def _record_version(
        self, definition: FeatureDefinition, changelog: str,
    ) -> FeatureVersion:
        """Record a version entry for a definition.

        If a version entry already exists for this definition + version,
        update it in-place (e.g. for lifecycle changes like activate/deprecate
        that don't increment the version number).
        """
        # Check for existing version entry
        stmt = select(FeatureVersion).where(
            FeatureVersion.feature_definition_id == definition.id,
            FeatureVersion.version == definition.version,
        )
        existing = self._session.execute(stmt).scalar_one_or_none()

        if existing is not None:
            existing.changelog = changelog
            existing.snapshot = definition.to_dict()
            existing.is_current = True
            self._session.flush()
            return existing

        # Mark other versions as not current
        current_history = self.get_history(definition.name)
        for v in current_history:
            v.is_current = False

        version = FeatureVersion(
            feature_definition_id=definition.id,
            version=definition.version,
            is_current=True,
            changelog=changelog,
            snapshot=definition.to_dict(),
        )
        self._session.add(version)
        self._session.flush()
        return version

    # ── Dependency management ─────────────────────────────

    def _create_dependency_edges(
        self,
        definition: FeatureDefinition,
        dependency_names: list[str],
    ) -> None:
        """Create dependency edges for a feature definition.

        Resolves dependency names to definition IDs. Edges with
        unresolvable names are logged as warnings (soft dependencies).
        """
        for dep_name in dependency_names:
            dep_def = self.latest(dep_name)
            if dep_def is None:
                logger.warning(
                    "Dependency %r for feature %r not found — skipping edge.",
                    dep_name, definition.name,
                )
                continue

            # Check for duplicate edge
            existing = (
                self._session.execute(
                    select(FeatureDependency).where(
                        FeatureDependency.dependent_feature_id == definition.id,
                        FeatureDependency.dependency_feature_id == dep_def.id,
                    )
                ).scalar_one_or_none()
            )
            if existing is not None:
                continue

            edge = FeatureDependency(
                dependent_feature_id=definition.id,
                dependency_feature_id=dep_def.id,
                is_hard=True,
            )
            self._session.add(edge)

    def get_dependencies(
        self,
        definition_id: str,
        *,
        hard_only: bool = False,
    ) -> list[FeatureDefinition]:
        """Get all features that *this* feature depends on (prerequisites).

        Parameters
        ----------
        definition_id : UUID
            Feature definition ID.
        hard_only : bool
            If True, only return hard (required) dependencies.

        Returns
        -------
        list[FeatureDefinition]
        """
        stmt = (
            select(FeatureDependency)
            .where(FeatureDependency.dependent_feature_id == definition_id)
        )
        edges = self._session.execute(stmt).scalars().all()

        dep_ids = []
        for edge in edges:
            if hard_only and not edge.is_hard:
                continue
            dep_ids.append(edge.dependency_feature_id)

        if not dep_ids:
            return []

        stmt = select(FeatureDefinition).where(
            FeatureDefinition.id.in_(dep_ids),
        )
        return list(self._session.execute(stmt).scalars().all())

    def get_dependents(
        self,
        definition_id: str,
    ) -> list[FeatureDefinition]:
        """Get all features that *depend on* this feature (reverse edge).

        Parameters
        ----------
        definition_id : UUID
            Feature definition ID.

        Returns
        -------
        list[FeatureDefinition]
        """
        stmt = (
            select(FeatureDependency)
            .where(FeatureDependency.dependency_feature_id == definition_id)
        )
        edges = self._session.execute(stmt).scalars().all()

        dep_ids = [e.dependent_feature_id for e in edges]
        if not dep_ids:
            return []

        stmt = select(FeatureDefinition).where(
            FeatureDefinition.id.in_(dep_ids),
        )
        return list(self._session.execute(stmt).scalars().all())

    def topological_sort(
        self,
        feature_ids: list[str] | None = None,
    ) -> list[FeatureDefinition]:
        """Return features in topological order (dependencies first).

        Uses Kahn's algorithm for topological sorting of the dependency
        DAG. Raises ``ValueError`` if a cycle is detected.

        Parameters
        ----------
        feature_ids : list[str], optional
            Subset of features to sort. If None, sorts ALL active features.

        Returns
        -------
        list[FeatureDefinition]
            Features in computation order (dependencies first).

        Raises
        ------
        ValueError
            If a dependency cycle is detected.
        """
        if feature_ids is not None:
            # Load all edges among the subset
            stmt = select(FeatureDependency).where(
                FeatureDependency.dependent_feature_id.in_(feature_ids),
            ).where(
                FeatureDependency.dependency_feature_id.in_(feature_ids),
            )
        else:
            stmt = select(FeatureDependency)

        edges = list(self._session.execute(stmt).scalars().all())

        # Build adjacency and in-degree maps
        in_degree: dict[str, int] = {}
        adjacency: dict[str, list[str]] = {}

        def _ensure(id_: str) -> None:
            if id_ not in in_degree:
                in_degree[id_] = 0
                adjacency[id_] = []

        for edge in edges:
            _ensure(edge.dependent_feature_id)
            _ensure(edge.dependency_feature_id)
            adjacency[edge.dependency_feature_id].append(
                edge.dependent_feature_id,
            )
            in_degree[edge.dependent_feature_id] += 1

        if not in_degree:
            return []

        # Kahn's algorithm
        queue = deque(
            id_ for id_, deg in in_degree.items() if deg == 0
        )
        sorted_ids: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_ids.append(node)
            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_ids) != len(in_degree):
            cycle_nodes = [str(i) for i in in_degree if in_degree[i] > 0]
            raise ValueError(
                f"Cycle detected in feature dependency DAG. "
                f"Affected feature IDs: {cycle_nodes[:10]}..."
            )

        # Load full FeatureDefinition objects
        if not sorted_ids:
            return []

        stmt = select(FeatureDefinition).where(
            FeatureDefinition.id.in_(sorted_ids),
        )
        defs = {
            d.id: d for d in self._session.execute(stmt).scalars().all()
        }
        return [defs[i] for i in sorted_ids if i in defs]

    def has_cycle(self) -> bool:
        """Check whether any dependency cycles exist in the DAG.

        Returns
        -------
        bool
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    # ─── Convenience ──────────────────────────────────────

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize all feature definitions as a list of dicts."""
        defs = self.list()
        return [d.to_dict() for d in defs]

    def search(self, query: str) -> list[FeatureDefinition]:
        """Search feature definitions by name (case-insensitive partial match).

        Parameters
        ----------
        query : str
            Search string.

        Returns
        -------
        list[FeatureDefinition]
        """
        stmt = (
            select(FeatureDefinition)
            .where(FeatureDefinition.name.ilike(f"%{query}%"))
            .order_by(FeatureDefinition.name)
        )
        return list(self._session.execute(stmt).scalars().all())



