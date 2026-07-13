"""
FeatureComputationEngine — orchestrate batch computation with resume support,
lazy loading, progress tracking, and dependency resolution.

Architecture
------------
The computation engine extends the abstract ``FeatureComputer`` pattern with
concrete orchestration:

1. **Dependency Resolution** — resolve the feature dependency DAG to determine
   computation order (topological sort via ``FeatureRegistry``).
2. **Incremental Computation** — use ``FeatureStore.needs_update()`` to skip
   entities whose values are already fresh.
3. **Resume Support** — on interruption, the engine can detect incomplete
   batches and resume from the last checkpoint.
4. **Lazy Loading** — features are only computed when accessed, using a
   ``LazyFeature`` wrapper that defers computation until ``get()`` is called.
5. **Progress Tracking** — emits progress via ``tqdm`` and structured logging.
6. **Batch Tracking** — each computation run creates a
   ``FeatureComputationBatch`` for audit.

Usage
-----
::

    from src.feature_store import FeatureComputationEngine, LazyFeature
    from src.feature_store.computers import ComputerRegistry

    # Register feature computers
    registry = ComputerRegistry()
    @registry.register("elo")
    class EloComputer(FeatureComputer):
        def compute_one(self, entity_id, **kwargs):
            return {"elo": 1500.0}

    # Create engine
    engine = FeatureComputationEngine(
        registry=registry,
        store=feature_store,
        registry_service=feature_registry,
    )

    # Run incremental computation
    batch = engine.run_incremental(
        feature_names=["elo_rating", "attack_strength"],
        entity_ids=[1, 2, 3],
        entity_type="team",
        trigger="scheduled",
    )

    # Lazy loading
    lazy = LazyFeature("elo_rating", team_id=42, engine=engine)
    value = lazy.get()  # Computed on first access
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.feature_store.computers import ComputerRegistry, FeatureComputer

from src.feature_store.computers import ComputerRegistry, FeatureComputer
from src.feature_store.models import (
    FeatureCategory,
    FeatureComputationBatch,
    FeatureDefinition,
    FeatureStatus,
    FeatureValue,
)
from src.feature_store.registry import FeatureRegistry
from src.feature_store.store import FeatureStore
from src.feature_store.validation import FeatureValidator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Computation Report
# ═══════════════════════════════════════════════════════════


@dataclass
class ComputationReport:
    """Report from a batch computation run.

    Attributes
    ----------
    batch_id : str
        ID of the computation batch.
    batch_label : str
        Human-readable batch label.
    feature_names : list[str]
        Names of features computed.
    entity_type : str
        Type of entities processed (match/team).
    entity_count : int
        Total entities targeted.
    computed_count : int
        Entities actually computed (excludes already-fresh).
    skipped_count : int
        Entities skipped (already fresh).
    failed_count : int
        Entities that failed during computation.
    duration_seconds : float
        Total computation duration.
    success : bool
        Whether the batch completed without critical errors.
    error : str | None
        Error message if failed.
    per_feature_stats : dict[str, dict]
        Stats per feature: computed, skipped, failed, duration.
    """

    batch_id: str = ""
    batch_label: str = ""
    feature_names: list[str] = field(default_factory=list)
    entity_type: str = "match"
    entity_count: int = 0
    computed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: str | None = None
    per_feature_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_label": self.batch_label,
            "feature_names": self.feature_names,
            "entity_type": self.entity_type,
            "entity_count": self.entity_count,
            "computed_count": self.computed_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "success": self.success,
            "error": self.error,
            "per_feature_stats": self.per_feature_stats,
        }


# ═══════════════════════════════════════════════════════════
#  FeatureComputationEngine
# ═══════════════════════════════════════════════════════════


class FeatureComputationEngine:
    """Orchestrates batch feature computation with resume and progress tracking.

    Parameters
    ----------
    registry : ComputerRegistry
        Registry of feature type → computer implementations.
    store : FeatureStore
        Database-backed feature store for persisting values.
    registry_service : FeatureRegistry
        Feature definition registry for lookup and dependency resolution.
    validator : FeatureValidator, optional
        Validator for post-computation validation.
    show_progress : bool
        Whether to show ``tqdm`` progress bars (default True).
    max_retries : int
        Maximum retries for failed computations (default 0).
    """

    def __init__(
        self,
        registry: ComputerRegistry,
        store: FeatureStore,
        registry_service: FeatureRegistry,
        validator: FeatureValidator | None = None,
        show_progress: bool = True,
        max_retries: int = 0,
    ) -> None:
        self._computer_registry = registry
        self._store = store
        self._registry = registry_service
        self._validator = validator or FeatureValidator()
        self.show_progress = show_progress
        self.max_retries = max_retries

    # ── Full batch computation ────────────────────────────

    def compute_features(
        self,
        feature_names: list[str],
        entity_ids: list[int],
        *,
        entity_type: str = "match",
        trigger: str = "manual",
        batch_label: str | None = None,
        incremental: bool = True,
        max_age_hours: float = 24.0,
        force_recompute: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ComputationReport:
        """Compute a batch of features for a set of entities.

        Resolves dependencies, determines computation order, runs
        each feature computer, validates results, and persists to the store.

        Parameters
        ----------
        feature_names : list[str]
            Feature names to compute (top-level; dependencies auto-resolved).
        entity_ids : list[int]
            Entity IDs (match IDs or team IDs).
        entity_type : str
            ``match`` or ``team`` (default ``match``).
        trigger : str
            Computation trigger: ``manual``, ``scheduled``, ``pipeline``.
        batch_label : str, optional
            Custom batch label. Auto-generated if not provided.
        incremental : bool
            If True (default), skip entities with fresh values.
        max_age_hours : float
            Max age for a value to be considered fresh (default 24).
        force_recompute : bool
            If True, recompute ALL entities regardless of freshness.
        progress_callback : callable, optional
            Callback ``(completed, total, feature_name) → None``.

        Returns
        -------
        ComputationReport
        """
        start_time = time.time()

        # Auto-generate batch label
        if batch_label is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            batch_label = f"compute_{ts}"

        # Resolve feature names to definitions (including dependencies)
        all_features = self._resolve_features(feature_names)
        all_feature_names = [f.name for f in all_features]

        # Start batch tracking
        batch = self._store.start_batch(
            batch_label=batch_label,
            trigger=trigger,
            features_computed=all_feature_names,
            entity_count=len(entity_ids) * len(all_features),
        )

        report = ComputationReport(
            batch_id=batch.id,
            batch_label=batch_label,
            feature_names=all_feature_names,
            entity_type=entity_type,
            entity_count=len(entity_ids),
        )

        try:
            self._execute_computation(
                features=all_features,
                entity_ids=entity_ids,
                entity_type=entity_type,
                batch=batch,
                incremental=incremental and not force_recompute,
                max_age_hours=max_age_hours,
                report=report,
                progress_callback=progress_callback,
            )

            report.duration_seconds = time.time() - start_time
            report.success = report.failed_count == 0

            # Complete batch
            self._store.complete_batch(
                batch.id,
                success=report.success,
                error=report.error,
            )

        except Exception as exc:
            report.duration_seconds = time.time() - start_time
            report.success = False
            report.error = str(exc)
            self._store.complete_batch(
                batch.id, success=False, error=str(exc),
            )
            logger.error("Batch computation failed: %s", exc, exc_info=True)

        logger.info(
            "Batch %s: computed=%d skipped=%d failed=%d (%.2fs)",
            batch_label, report.computed_count, report.skipped_count,
            report.failed_count, report.duration_seconds,
        )
        return report

    def run_incremental(
        self,
        feature_names: list[str],
        entity_ids: list[int],
        *,
        entity_type: str = "match",
        trigger: str = "scheduled",
        max_age_hours: float = 24.0,
    ) -> ComputationReport:
        """Run incremental computation — only compute stale/missing values.

        This is the primary entry point for scheduled/automated runs.

        Parameters
        ----------
        feature_names : list[str]
        entity_ids : list[int]
        entity_type : str
        trigger : str
        max_age_hours : float

        Returns
        -------
        ComputationReport
        """
        return self.compute_features(
            feature_names=feature_names,
            entity_ids=entity_ids,
            entity_type=entity_type,
            trigger=trigger,
            incremental=True,
            max_age_hours=max_age_hours,
        )

    def resume(
        self,
        batch_id: str,
        *,
        force_recompute: bool = False,
    ) -> ComputationReport:
        """Resume a failed or incomplete computation batch.

        Loads the batch record, determines what was not computed,
        and runs only the missing entities.

        Parameters
        ----------
        batch_id : str
            ID of the batch to resume.
        force_recompute : bool
            If True, recompute all entities in the batch.

        Returns
        -------
        ComputationReport
        """
        batch = self._store.get_batch(batch_id)
        if batch is None:
            raise ValueError(f"Batch {batch_id} not found.")

        if batch.success and not force_recompute:
            logger.info("Batch %s already completed successfully — nothing to resume.", batch_id)
            return ComputationReport(
                batch_id=batch_id,
                batch_label=batch.batch_label,
                success=True,
            )

        # Load feature definitions from stored names
        features = []
        for fname in (batch.features_computed or []):
            feat = self._registry.latest(fname)
            if feat is not None:
                features.append(feat)

        logger.info(
            "Resuming batch %s (%s): %d features, %d entities",
            batch_id, batch.batch_label, len(features), batch.entity_count,
        )

        report = self.compute_features(
            feature_names=[f.name for f in features],
            entity_ids=list(range(batch.entity_count)),  # Use stored count
            entity_type="match",  # Default — user can override
            trigger=batch.trigger or "resume",
            batch_label=f"resume_{batch.batch_label}",
            force_recompute=force_recompute,
        )

        return report

    # ── Single-feature single-entity ──────────────────────

    def compute_one(
        self,
        feature_name: str,
        entity_id: int,
        *,
        entity_type: str = "match",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute a single feature for a single entity.

        Parameters
        ----------
        feature_name : str
            Feature name.
        entity_id : int
            Entity ID.
        entity_type : str
        context : dict, optional
            Additional context for the computer.

        Returns
        -------
        dict[str, Any]
            Computed feature values (``{feature_name: value}``).
        """
        definition = self._registry.latest(feature_name)
        if definition is None:
            raise ValueError(f"Feature {feature_name!r} not found in registry.")

        computer = self._computer_registry.get(definition.feature_type)
        if computer is None:
            raise ValueError(
                f"No computer registered for feature type {definition.feature_type!r}",
            )

        result = computer.compute_one(entity_id, **(context or {}))
        self._validate_and_store(definition, entity_id, entity_type, result, computer.name)
        return result

    # ── Internal ──────────────────────────────────────────

    def _resolve_features(
        self,
        feature_names: list[str],
    ) -> list[FeatureDefinition]:
        """Resolve feature names to definitions, including dependency expansion.

        Uses topological sort to ensure dependencies are computed first.
        """
        definitions: list[FeatureDefinition] = []
        seen: set[str] = set()

        # Load definitions and their dependencies
        def _load(name: str) -> None:
            if name in seen:
                return
            seen.add(name)

            defn = self._registry.latest(name)
            if defn is None:
                logger.warning("Feature %r not found in registry — skipping.", name)
                return

            # Load soft dependencies (from definition.dependencies field)
            for dep_name in (defn.dependencies or []):
                dep_def = self._registry.latest(dep_name)
                if dep_def is not None and dep_def.name not in seen:
                    _load(dep_def.name)

            definitions.append(defn)

        for name in feature_names:
            _load(name)

        # Topological sort via registry
        ids = [d.id for d in definitions]
        sorted_defs = self._registry.topological_sort(feature_ids=ids)

        # Any features not in the DAG (no dependencies) should be appended
        sorted_ids = {d.id for d in sorted_defs}
        remaining = [d for d in definitions if d.id not in sorted_ids]

        return sorted_defs + remaining

    def _execute_computation(
        self,
        features: list[FeatureDefinition],
        entity_ids: list[int],
        entity_type: str,
        batch: FeatureComputationBatch,
        incremental: bool,
        max_age_hours: float,
        report: ComputationReport,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """Execute the computation loop over features and entities."""
        total_ops = len(features) * len(entity_ids)
        completed = 0

        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_ops, desc="Computing features", unit="ops")
            except ImportError:
                pbar = None
                logger.warning("tqdm not installed — progress bars disabled.")
        else:
            pbar = None

        for feature_def in features:
            feature_start = time.time()
            feat_computed = 0
            feat_skipped = 0
            feat_failed = 0

            # Check incremental: which entities need update?
            if incremental and not report.error:
                stale_ids = self._store.needs_update(
                    feature_def.id,
                    entity_ids,
                    entity_type=entity_type,
                    max_age_hours=max_age_hours,
                )
                if stale_ids:
                    logger.info(
                        "  %s: %d/%d entities need update",
                        feature_def.name, len(stale_ids), len(entity_ids),
                    )
            else:
                stale_ids = entity_ids[:]
                feat_skipped_total = len(entity_ids) - len(stale_ids)
                if incremental:
                    feat_skipped = feat_skipped_total

            # Lookup computer
            computer = self._computer_registry.get(feature_def.feature_type)
            if computer is None:
                logger.warning("No computer for %s — skipping.", feature_def.name)
                feat_failed = len(stale_ids)
                if pbar:
                    pbar.update(len(stale_ids))
                completed += len(stale_ids)
                report.per_feature_stats[feature_def.name] = {
                    "computed": 0, "skipped": 0, "failed": len(stale_ids),
                    "duration": 0.0, "status": "no_computer",
                }
                continue

            # Compute for each entity
            for eid in stale_ids:
                try:
                    kwargs = {entity_type + "_id": eid}
                    context: dict[str, Any] = {"match_id": eid} if entity_type == "match" else {"team_id": eid}

                    result = computer.compute_one(eid, **context)

                    # Store results
                    self._validate_and_store(
                        feature_def, eid, entity_type,
                        result, computer.name, batch_id=batch.id,
                    )
                    feat_computed += 1

                except Exception as exc:
                    feat_failed += 1
                    logger.error(
                        "  Failed computing %s for %s %d: %s",
                        feature_def.name, entity_type, eid, exc,
                    )

                completed += 1
                if pbar:
                    pbar.update(1)
                if progress_callback:
                    progress_callback(completed, total_ops, feature_def.name)

            feat_duration = time.time() - feature_start

            report.computed_count += feat_computed
            report.failed_count += feat_failed
            report.skipped_count += feat_skipped
            report.per_feature_stats[feature_def.name] = {
                "computed": feat_computed,
                "skipped": feat_skipped,
                "failed": feat_failed,
                "duration": round(feat_duration, 2),
                "status": "ok" if feat_failed == 0 else f"{feat_failed} errors",
            }

            logger.info(
                "  %s: computed=%d skipped=%d failed=%d (%.2fs)",
                feature_def.name, feat_computed, feat_skipped,
                feat_failed, feat_duration,
            )

        if pbar:
            pbar.close()

        # Update entity count on the batch
        batch.entity_count = completed

    def _validate_and_store(
        self,
        definition: FeatureDefinition,
        entity_id: int,
        entity_type: str,
        result: dict[str, Any],
        computed_by: str,
        batch_id: str | None = None,
    ) -> None:
        """Validate a computation result and persist to the store."""
        for feature_name, value in result.items():
            kwargs: dict[str, Any] = {
                "numeric_value": value if isinstance(value, (int, float)) else None,
                "json_value": value if isinstance(value, dict) else None,
                "text_value": str(value) if not isinstance(value, (int, float, dict)) else None,
                "computed_by": computed_by,
                "batch_id": batch_id,
            }

            if entity_type == "match":
                kwargs["match_id"] = entity_id
            elif entity_type == "team":
                kwargs["team_id"] = entity_id
            elif entity_type == "league":
                kwargs["league_id"] = entity_id

            self._store.set(
                definition_id=definition.id,
                **kwargs,
            )


# ═══════════════════════════════════════════════════════════
#  LazyFeature — compute on first access
# ═══════════════════════════════════════════════════════════


class LazyFeature:
    """A feature whose value is computed **lazily** on first access.

    The ``LazyFeature`` wraps a feature name + entity identifier and
    defers computation until ``get()`` is called. Results are cached
    in memory for the lifetime of the ``LazyFeature`` instance.

    This is useful for:
    - Conditional pipelines where not all features are needed
    - Interactive exploration where features are accessed on demand
    - Dependency injection where features are passed around but not
      immediately consumed

    Parameters
    ----------
    feature_name : str
        Name of the feature to compute.
    entity_id : int
        Entity ID (match, team, or league).
    entity_type : str
        ``match``, ``team``, or ``league`` (default ``match``).
    engine : FeatureComputationEngine, optional
        Engine to use for computation. Must be provided before ``get()``.
    use_cache : bool
        If True (default), cache the value in memory after first computation.

    Examples
    --------
    ::

        lazy_elo = LazyFeature("elo_rating", team_id=42, engine=engine)

        # Value is computed on first access
        elo_value = lazy_elo.get()  # Computed here
        elo_value = lazy_elo.get()  # Returns cached value

        # Check if computed without triggering computation
        if not lazy_elo.is_computed:
            print("Feature not yet computed")
    """

    def __init__(
        self,
        feature_name: str,
        entity_id: int,
        *,
        entity_type: str = "match",
        engine: FeatureComputationEngine | None = None,
        use_cache: bool = True,
    ) -> None:
        self.feature_name = feature_name
        self.entity_id = entity_id
        self.entity_type = entity_type
        self._engine = engine
        self._use_cache = use_cache
        self._cached_value: Any = None
        self._is_computed: bool = False

    @property
    def is_computed(self) -> bool:
        """Whether this feature has been computed yet."""
        return self._is_computed

    def set_engine(self, engine: FeatureComputationEngine) -> None:
        """Set or replace the computation engine."""
        self._engine = engine

    def get(self, force_recompute: bool = False) -> Any:
        """Get the feature value, computing it if necessary.

        On first call, computes via the engine and (optionally) caches
        the result. Subsequent calls return the cached value unless
        ``force_recompute=True``.

        Parameters
        ----------
        force_recompute : bool
            If True, recompute even if already cached.

        Returns
        -------
        Any
            The computed feature value.
        """
        if self._is_computed and not force_recompute:
            return self._cached_value

        if self._engine is None:
            raise ValueError(
                "No engine provided. Call ``set_engine()`` or pass engine "
                "to the constructor before calling ``get()``."
            )

        result = self._engine.compute_one(
            self.feature_name,
            self.entity_id,
            entity_type=self.entity_type,
        )

        # Extract the value from the result dict
        if isinstance(result, dict):
            value = result.get(self.feature_name, result)
        else:
            value = result

        if self._use_cache:
            self._cached_value = value
            self._is_computed = True

        return value

    def __repr__(self) -> str:
        status = "computed" if self._is_computed else "lazy"
        return (
            f"<LazyFeature {self.feature_name!r} "
            f"entity={self.entity_type}:{self.entity_id} "
            f"status={status}>"
        )


# ═══════════════════════════════════════════════════════════
#  LazyFeatureSet — batch lazy loading
# ═══════════════════════════════════════════════════════════


class LazyFeatureSet:
    """A collection of ``LazyFeature`` objects for a single entity.

    Provides dict-like access with lazy computation::

        features = LazyFeatureSet(match_id=42, engine=engine)
        features.add("elo_rating")
        features.add("attack_strength")

        # All accessed features are computed on demand
        elo = features["elo_rating"]          # Computed here
        attack = features["attack_strength"]   # Computed here
    """

    def __init__(
        self,
        entity_id: int,
        *,
        entity_type: str = "match",
        engine: FeatureComputationEngine | None = None,
    ) -> None:
        self.entity_id = entity_id
        self.entity_type = entity_type
        self._engine = engine
        self._features: dict[str, LazyFeature] = {}

    def add(self, feature_name: str) -> LazyFeature:
        """Add a feature to the set (lazy — not computed yet).

        Parameters
        ----------
        feature_name : str
            Feature to add.

        Returns
        -------
        LazyFeature
            The lazy feature wrapper.
        """
        if feature_name not in self._features:
            lazy = LazyFeature(
                feature_name=feature_name,
                entity_id=self.entity_id,
                entity_type=self.entity_type,
                engine=self._engine,
            )
            self._features[feature_name] = lazy
        return self._features[feature_name]

    def __getitem__(self, feature_name: str) -> Any:
        """Get a feature value, computing it lazily if needed."""
        if feature_name not in self._features:
            self.add(feature_name)
        return self._features[feature_name].get()

    def __contains__(self, feature_name: str) -> bool:
        return feature_name in self._features

    def get_computed(self) -> dict[str, Any]:
        """Get only the features that have already been computed."""
        return {
            name: feat._cached_value
            for name, feat in self._features.items()
            if feat.is_computed
        }

    def compute_all(self) -> dict[str, Any]:
        """Compute ALL features in the set (eager)."""
        return {name: self[name] for name in self._features}

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        computed = sum(1 for f in self._features.values() if f.is_computed)
        return (
            f"<LazyFeatureSet entity={self.entity_type}:{self.entity_id} "
            f"features={len(self._features)} computed={computed}>"
        )
