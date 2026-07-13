"""
FeatureCache — caching layer for computed feature values.

Integrates the existing ``CacheManager`` framework with the Feature Store,
providing transparent caching of feature values with tag-based invalidation,
batch cache warming, and stale-while-revalidate semantics.

Architecture
------------
FeatureCache wraps both a ``FeatureStore`` (database) and a ``CacheManager``
(cache backend), implementing a look-aside cache pattern:

    1. ``get()`` → check cache → if miss, load from FeatureStore → populate cache
    2. ``set()`` → write to FeatureStore → populate cache
    3. ``delete()`` → delete from FeatureStore → invalidate cache

Cache keys follow the convention::

    feature:{definition_name}:v{version}:match:{match_id}
    feature:{definition_name}:v{version}:team:{team_id}
    feature:{definition_name}:v{version}:global

Tags are applied per:
- **Entity** — ``match:{id}``, ``team:{id}``
- **Feature** — ``feature:{name}``
- **Category** — ``category:{category}``
"""

from __future__ import annotations

import logging
from typing import Any

from src.cache import CacheManager, CacheStats
from src.cache.decorators import _run_async
from src.feature_store.models import FeatureDefinition, FeatureValue
from src.feature_store.store import FeatureStore

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "feature"


class FeatureCache:
    """Caching layer for the Feature Store.

    Wraps a ``FeatureStore`` with a ``CacheManager`` for transparent
    caching of feature values. Operates in **look-aside** mode:
    cache is checked first; on miss, the underlying store is queried
    and the result is populated into the cache.

    All methods are **synchronous** (the cache calls are async internally,
    bridged via ``_run_async`` — same pattern as the ``@cached`` decorator).

    Parameters
    ----------
    store : FeatureStore
        The underlying (database-backed) feature store.
    cache : CacheManager
        The cache manager (backed by SQLite or Redis).
    default_ttl : float
        Default cache TTL in seconds (default 3600).
    """

    def __init__(
        self,
        store: FeatureStore,
        cache: CacheManager,
        default_ttl: float = 3600.0,
    ) -> None:
        self._store = store
        self._cache = cache
        self.default_ttl = default_ttl

    # ── Key builders ──────────────────────────────────────

    @staticmethod
    def _key_for_match(name: str, version: int, match_id: int) -> str:
        return f"{CACHE_NAMESPACE}:{name}:v{version}:match:{match_id}"

    @staticmethod
    def _key_for_team(name: str, version: int, team_id: int) -> str:
        return f"{CACHE_NAMESPACE}:{name}:v{version}:team:{team_id}"

    @staticmethod
    def _key_for_league(name: str, version: int, league_id: int) -> str:
        return f"{CACHE_NAMESPACE}:{name}:v{version}:league:{league_id}"

    @staticmethod
    def _key_for_global(name: str, version: int) -> str:
        return f"{CACHE_NAMESPACE}:{name}:v{version}:global"

    @staticmethod
    def _entity_tags(
        match_id: int | None = None,
        team_id: int | None = None,
        league_id: int | None = None,
    ) -> set[str]:
        tags: set[str] = set()
        if match_id is not None:
            tags.add(f"match:{match_id}")
        if team_id is not None:
            tags.add(f"team:{team_id}")
        if league_id is not None:
            tags.add(f"league:{league_id}")
        return tags

    # ── Single-value operations ───────────────────────────

    def get(
        self,
        definition: FeatureDefinition,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
        league_id: int | None = None,
    ) -> FeatureValue | None:
        """Get a cached feature value, falling through to the store on miss.

        Look-aside cache pattern:
        - Check cache → if hit, return
        - If miss, query ``FeatureStore`` → populate cache → return

        Parameters
        ----------
        definition : FeatureDefinition
        match_id : int, optional
        team_id : int, optional
        league_id : int, optional

        Returns
        -------
        FeatureValue | None
        """
        key = self._build_key(definition, match_id, team_id, league_id)
        if key is None:
            return self._store.get(
                definition.id, match_id=match_id, team_id=team_id, league_id=league_id,
            )

        # Check cache
        cached = _run_async(self._cache.get(key))
        if cached is not None:
            if isinstance(cached, dict):
                return FeatureValue(**cached)
            return cached  # type: ignore[return-value]

        # Cache miss — load from store and populate
        value = self._store.get(
            definition.id, match_id=match_id, team_id=team_id, league_id=league_id,
        )
        if value is not None:
            tags = {f"feature:{definition.name}", f"category:{definition.category.value}"}
            tags |= self._entity_tags(match_id, team_id, league_id)
            _run_async(self._cache.set(
                key,
                value.to_dict() if hasattr(value, "to_dict") else value,
                ttl=self.default_ttl,
                tags=tags,
            ))
        return value

    def set(
        self,
        definition: FeatureDefinition,
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
        """Set a feature value, writing to both store and cache.

        Parameters
        ----------
        Same as ``FeatureStore.set()``.

        Returns
        -------
        FeatureValue
        """
        value = self._store.set(
            definition_id=definition.id,
            match_id=match_id,
            team_id=team_id,
            league_id=league_id,
            numeric_value=numeric_value,
            text_value=text_value,
            json_value=json_value,
            computed_by=computed_by,
            batch_id=batch_id,
        )

        # Populate cache
        key = self._build_key(definition, match_id, team_id, league_id)
        if key is not None:
            tags = {f"feature:{definition.name}", f"category:{definition.category.value}"}
            tags |= self._entity_tags(match_id, team_id, league_id)
            _run_async(self._cache.set(
                key,
                value.to_dict() if hasattr(value, "to_dict") else value,
                ttl=self.default_ttl,
                tags=tags,
            ))

        return value

    def delete(
        self,
        definition: FeatureDefinition,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
    ) -> bool:
        """Delete a feature value from both store and cache.

        Parameters
        ----------
        definition : FeatureDefinition
        match_id : int, optional
        team_id : int, optional

        Returns
        -------
        bool
        """
        # Delete from store
        result = self._store.delete(
            definition.id, match_id=match_id, team_id=team_id,
        )

        # Invalidate cache
        key = self._build_key(definition, match_id, team_id)
        if key is not None:
            _run_async(self._cache.invalidate(key))

        return result

    # ── Batch operations ──────────────────────────────────

    def get_many(
        self,
        definitions: list[tuple[FeatureDefinition, int | None, int | None]],
    ) -> dict[str, FeatureValue | None]:
        """Get multiple feature values with cache-first semantics.

        Uses entity-qualified keys (``{name}:match:{id}``, ``{name}:team:{id}``)
        in the result dict so the same feature can be queried for multiple
        entities without key collision.

        Parameters
        ----------
        definitions : list of (FeatureDefinition, match_id | None, team_id | None)
            Each tuple is ``(definition, match_id, team_id)`` where typically
            only one of ``match_id`` or ``team_id`` is non-None.

        Returns
        -------
        dict[str, FeatureValue | None]
            Entity-qualified keys::

                {
                    "elo_rating:match:42": <FeatureValue>,
                    "elo_rating:match:43": <FeatureValue>,
                    "attack_strength:team:7": None,
                }
        """
        # Build unique entity-qualified result keys
        def _result_key(defn: FeatureDefinition, match_id: int | None, team_id: int | None) -> str:
            if match_id is not None:
                return f"{defn.name}:match:{match_id}"
            if team_id is not None:
                return f"{defn.name}:team:{team_id}"
            return f"{defn.name}:global"

        keys_to_defs: dict[str, tuple[FeatureDefinition, int | None, int | None]] = {}
        for defn, match_id, team_id in definitions:
            key = self._build_key(defn, match_id, team_id)
            if key:
                keys_to_defs[key] = (defn, match_id, team_id)

        # Batch cache check
        cached_results = _run_async(self._cache.get_many(list(keys_to_defs.keys())))

        results: dict[str, FeatureValue | None] = {}
        uncached: list[tuple[FeatureDefinition, int | None, int | None]] = []

        for key, (defn, match_id, team_id) in keys_to_defs.items():
            cached = cached_results.get(key)
            if cached is not None:
                if isinstance(cached, dict):
                    results[_result_key(defn, match_id, team_id)] = FeatureValue(**cached)
                else:
                    results[_result_key(defn, match_id, team_id)] = cached  # type: ignore
            else:
                uncached.append((defn, match_id, team_id))

        # Load uncached from store (individual queries)
        for defn, match_id, team_id in uncached:
            value = self._store.get(defn.id, match_id=match_id, team_id=team_id)
            result_key = _result_key(defn, match_id, team_id)
            results[result_key] = value
            if value is not None:
                key = self._build_key(defn, match_id, team_id)
                if key:
                    tags = {f"feature:{defn.name}", f"category:{defn.category.value}"}
                    tags |= self._entity_tags(match_id, team_id)
                    _run_async(self._cache.set(
                        key,
                        value.to_dict() if hasattr(value, "to_dict") else value,
                        ttl=self.default_ttl,
                        tags=tags,
                    ))

        return results

    # ── Cache invalidation ────────────────────────────────

    def invalidate_feature(self, name: str) -> int:
        """Invalidate all cached values for a feature definition."""
        return _run_async(self._cache.invalidate_by_tag(f"feature:{name}"))

    def invalidate_entity(self, entity_type: str, entity_id: int) -> int:
        """Invalidate all cached feature values for an entity."""
        return _run_async(self._cache.invalidate_by_tag(f"{entity_type}:{entity_id}"))

    def invalidate_category(self, category: str) -> int:
        """Invalidate all cached values for a feature category."""
        return _run_async(self._cache.invalidate_by_tag(f"category:{category}"))

    # ── Cache warming ─────────────────────────────────────

    def warm(
        self,
        definition: FeatureDefinition,
        entity_ids: list[int],
        *,
        entity_type: str = "match",
    ) -> int:
        """Pre-populate the cache for a batch of entities (cache warming).

        Parameters
        ----------
        definition : FeatureDefinition
        entity_ids : list[int]
        entity_type : str

        Returns
        -------
        int
            Number of cache entries written.
        """
        warmed = 0
        for eid in entity_ids:
            kwargs: dict[str, int] = (
                {"match_id": eid} if entity_type == "match" else {"team_id": eid}
            )
            key = self._build_key(definition, **kwargs)  # type: ignore[arg-type]
            if key is None:
                continue

            # Skip if already cached
            cached = _run_async(self._cache.get(key))
            if cached is not None:
                continue

            value = self._store.get(definition.id, **kwargs)  # type: ignore[arg-type]
            if value is not None:
                tags = {
                    f"feature:{definition.name}",
                    f"category:{definition.category.value}",
                    f"{entity_type}:{eid}",
                }
                _run_async(self._cache.set(
                    key,
                    value.to_dict() if hasattr(value, "to_dict") else value,
                    ttl=self.default_ttl,
                    tags=tags,
                ))
                warmed += 1

        logger.info(
            "Warmed %d/%d cache entries for %s (%s)",
            warmed, len(entity_ids), definition.name, entity_type,
        )
        return warmed

    # ── Stale-while-revalidate ────────────────────────────

    def get_with_stale(
        self,
        definition: FeatureDefinition,
        *,
        match_id: int | None = None,
        team_id: int | None = None,
    ) -> tuple[FeatureValue | None, bool]:
        """Get a feature value with stale-while-revalidate support.

        Returns (value, is_stale). A stale value is served from cache
        past the TTL but not yet fully expired, with an indication
        that it should be revalidated.

        Parameters
        ----------
        definition : FeatureDefinition
        match_id : int, optional
        team_id : int, optional

        Returns
        -------
        tuple[FeatureValue | None, bool]
        """
        key = self._build_key(definition, match_id=match_id, team_id=team_id)
        if key is None:
            return self._store.get(definition.id, match_id=match_id, team_id=team_id), False

        entry = _run_async(self._cache.get_entry(key))
        if entry is not None:
            is_stale = entry.age > self.default_ttl
            # Serve stale but within stale TTL
            if is_stale and entry.age <= (self.default_ttl + 300):
                return entry.value, True  # type: ignore[return-value]
            if not is_stale:
                return entry.value, False  # type: ignore[return-value]

        # Cache miss — load from store
        value = self._store.get(definition.id, match_id=match_id, team_id=team_id)
        if value is not None:
            tags = {f"feature:{definition.name}", f"category:{definition.category.value}"}
            tags |= self._entity_tags(match_id, team_id)
            _run_async(self._cache.set(
                key,
                value.to_dict() if hasattr(value, "to_dict") else value,
                ttl=self.default_ttl,
                tags=tags,
            ))
        return value, False

    # ── Utility ───────────────────────────────────────────

    def _build_key(
        self,
        definition: FeatureDefinition,
        match_id: int | None = None,
        team_id: int | None = None,
        league_id: int | None = None,
    ) -> str | None:
        if match_id is not None:
            return self._key_for_match(definition.name, definition.version, match_id)
        if team_id is not None:
            return self._key_for_team(definition.name, definition.version, team_id)
        if league_id is not None:
            return self._key_for_league(definition.name, definition.version, league_id)
        return self._key_for_global(definition.name, definition.version)

    def cache_stats(self) -> CacheStats:
        """Return cache statistics from the underlying cache manager."""
        return _run_async(self._cache.stats())

    def clear_cache(self) -> int:
        """Clear ALL cached feature values. Use with caution."""
        return _run_async(self._cache.clear())
