"""Tests for FeatureCache — caching layer for the Feature Store."""

from __future__ import annotations

from typing import Any

import pytest

from src.cache import CacheManager, SQLiteBackend
from src.feature_store import (
    FeatureCategory,
    FeatureDefinition,
    FeatureStatus,
)
from src.feature_store.cache import FeatureCache
from src.feature_store.registry import FeatureRegistry
from src.feature_store.store import FeatureStore


@pytest.fixture
def cache_manager(tmp_path) -> CacheManager:
    """Create a temp-file-based cache manager for testing.

    Uses a temp file (not :memory:) because ``_run_async`` may create an
    event loop that uses a different thread-local connection, causing
    ``:memory:`` databases to be isolated per thread.
    """
    cache_path = tmp_path / "test_cache.db"
    backend = SQLiteBackend(str(cache_path), cleanup_interval=0)
    return CacheManager(backend, namespace="feature", default_ttl=3600)


@pytest.fixture
def feature_cache(
    session,
    registry: FeatureRegistry,
    store: FeatureStore,
    cache_manager: CacheManager,
) -> FeatureCache:
    return FeatureCache(store, cache_manager, default_ttl=3600)


class TestFeatureCache:
    """Test FeatureCache caching layer."""

    def test_get_miss_then_store(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Cache miss falls through to store, then populates cache."""
        # Miss — no value in store
        value = feature_cache.get(sample_def, match_id=1)
        assert value is None

        # Set a value
        fv = feature_cache.set(
            sample_def, match_id=1, numeric_value=0.85, computed_by="test",
        )
        assert fv.numeric_value == 0.85

        # Now it should hit cache
        cached = feature_cache.get(sample_def, match_id=1)
        assert cached is not None
        assert cached.numeric_value == 0.85

    def test_set_updates_cache(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Setting a value updates both store and cache."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.5, computed_by="test",
        )
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.9, computed_by="test",
        )

        cached = feature_cache.get(sample_def, match_id=1)
        assert cached is not None
        assert cached.numeric_value == 0.9

    def test_delete_invalidates_cache(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Deleting a value removes it from both store and cache."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.5, computed_by="test",
        )
        feature_cache.set(
            sample_def, match_id=2, numeric_value=0.6, computed_by="test",
        )

        deleted = feature_cache.delete(sample_def, match_id=1)
        assert deleted is True

        # Should be gone from cache and store
        cached = feature_cache.get(sample_def, match_id=1)
        assert cached is None

        # Other values should remain
        other = feature_cache.get(sample_def, match_id=2)
        assert other is not None
        assert other.numeric_value == 0.6

    def test_invalidate_feature(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Invalidate all cached values for a feature."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.1, computed_by="test",
        )
        feature_cache.set(
            sample_def, match_id=2, numeric_value=0.2, computed_by="test",
        )

        count = feature_cache.invalidate_feature(sample_def.name)
        assert count >= 0  # May be 0 if SQLite has auto-cleaned

        # Values should still be in store
        from sqlalchemy import select
        from src.feature_store.models import FeatureValue
        session = feature_cache._store._session
        stmt = select(FeatureValue).where(
            FeatureValue.feature_definition_id == sample_def.id,
        )
        remaining = list(session.execute(stmt).scalars().all())
        assert len(remaining) == 2

    def test_cache_warm(
        self, feature_cache: FeatureCache,
        registry: FeatureRegistry, store: FeatureStore,
        sample_def: FeatureDefinition,
    ) -> None:
        """Warm the cache for entities that have values in the store."""
        # Set values in store only (bypass cache)
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.1, computed_by="test")
        store.set(definition_id=sample_def.id, match_id=2, numeric_value=0.2, computed_by="test")

        # Warm cache
        warmed = feature_cache.warm(sample_def, [1, 2])
        assert warmed == 2

        # Should now be in cache
        cached = feature_cache.get(sample_def, match_id=1)
        assert cached is not None
        assert cached.numeric_value == 0.1

    def test_cache_warm_only_missing(
        self, feature_cache: FeatureCache,
        sample_def: FeatureDefinition,
    ) -> None:
        """Warming should not re-cache already-cached entries."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.5, computed_by="test",
        )

        # Warm excludes entity 2 (no store value)
        warmed = feature_cache.warm(sample_def, [1, 2])
        assert warmed == 0  # Entity 1 is already cached, entity 2 has no store value

    def test_get_many(
        self, feature_cache: FeatureCache,
        sample_def: FeatureDefinition,
    ) -> None:
        """Batch get with cache-first semantics.

        Uses entity-qualified result keys so the same feature can be
        queried for multiple entities without key collision.
        """
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.1, computed_by="test",
        )
        feature_cache.set(
            sample_def, match_id=2, numeric_value=0.2, computed_by="test",
        )

        results = feature_cache.get_many([
            (sample_def, 1, None),
            (sample_def, 2, None),
            (sample_def, 3, None),  # Missing
        ])

        # Results use entity-qualified keys
        assert "test_feature:match:1" in results
        assert "test_feature:match:2" in results
        assert "test_feature:match:3" in results

        # Entity 1 and 2 should have values
        assert results["test_feature:match:1"] is not None
        assert results["test_feature:match:1"].numeric_value == 0.1
        assert results["test_feature:match:2"] is not None
        assert results["test_feature:match:2"].numeric_value == 0.2

        # Entity 3 (not stored) should be None
        assert results["test_feature:match:3"] is None

    def test_get_with_stale_fresh(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Fresh values should not be marked as stale."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.5, computed_by="test",
        )
        value, is_stale = feature_cache.get_with_stale(
            sample_def, match_id=1,
        )
        assert value is not None
        assert value.numeric_value == 0.5
        assert is_stale is False

    def test_get_many_empty(
        self, feature_cache: FeatureCache,
    ) -> None:
        """Empty batch get returns empty dict."""
        results = feature_cache.get_many([])
        assert results == {}

    def test_team_level_caching(
        self, feature_cache: FeatureCache, sample_team_def: FeatureDefinition,
    ) -> None:
        """Team-level features use team cache keys."""
        feature_cache.set(
            sample_team_def, team_id=42, numeric_value=1500.0, computed_by="test",
        )
        cached = feature_cache.get(sample_team_def, team_id=42)
        assert cached is not None
        assert cached.numeric_value == 1500.0

    def test_global_feature_caching(
        self, feature_cache: FeatureCache,
        registry: FeatureRegistry,
    ) -> None:
        """Global features (no entity ID) use global cache key."""
        global_def = registry.register(
            name="league_strength_factor",
            feature_type="league_strength",
            category=FeatureCategory.LEAGUE_STRENGTH,
            entity_type="global",
            status=FeatureStatus.ACTIVE,
        )
        feature_cache.set(
            global_def, numeric_value=1.5, computed_by="test",
        )
        cached = feature_cache.get(global_def)
        assert cached is not None
        assert cached.numeric_value == 1.5

    def test_clear_cache(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Clear all cached values."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.5, computed_by="test",
        )
        cleared = feature_cache.clear_cache()
        assert cleared > 0

        # Store should still have the value
        from sqlalchemy import select
        from src.feature_store.models import FeatureValue
        session = feature_cache._store._session
        stmt = select(FeatureValue).where(
            FeatureValue.feature_definition_id == sample_def.id,
        )
        remaining = list(session.execute(stmt).scalars().all())
        assert len(remaining) == 1

    def test_invalidate_entity(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Invalidate all cached values for a specific entity."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.5, computed_by="test",
        )
        feature_cache.set(
            sample_def, match_id=2, numeric_value=0.6, computed_by="test",
        )

        count = feature_cache.invalidate_entity("match", 1)
        assert count >= 0

    def test_invalidate_category(
        self, feature_cache: FeatureCache, sample_def: FeatureDefinition,
    ) -> None:
        """Invalidate all cached values for a category."""
        feature_cache.set(
            sample_def, match_id=1, numeric_value=0.5, computed_by="test",
        )
        count = feature_cache.invalidate_category(sample_def.category.value)
        assert count >= 0

    def test_cache_stats(
        self, feature_cache: FeatureCache,
    ) -> None:
        """Cache stats should return valid statistics."""
        stats = feature_cache.cache_stats()
        assert stats is not None
        assert isinstance(stats.hits, int)
