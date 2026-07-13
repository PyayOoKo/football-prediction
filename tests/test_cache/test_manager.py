"""
Tests for ``src.cache.manager``.

Covers CacheManager: namespacing, get/set/delete, get_or_compute,
invalidation by key and tag, stats, build_key, and multi operations.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio

from src.cache.backend import SQLiteBackend
from src.cache.manager import CacheManager


@pytest.fixture
def cache() -> CacheManager:
    backend = SQLiteBackend(
        db_path=":memory:",
        table_name="test_manager",
        cleanup_interval=0,
        vacuum_after_n_cleanups=0,
    )
    return CacheManager(backend, namespace="test_ns", default_ttl=3600)


@pytest.mark.asyncio
class TestCacheManagerBasics:
    async def test_set_and_get(self, cache: CacheManager) -> None:
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"

    async def test_get_missing(self, cache: CacheManager) -> None:
        value = await cache.get("nonexistent")
        assert value is None

    async def test_get_entry(self, cache: CacheManager) -> None:
        await cache.set("key1", "value1", ttl=60)
        entry = await cache.get_entry("key1")
        assert entry is not None
        assert entry.value == "value1"
        assert entry.ttl == 60

    async def test_delete(self, cache: CacheManager) -> None:
        await cache.set("key1", "value1")
        assert await cache.delete("key1")
        assert not await cache.has("key1")

    async def test_delete_missing(self, cache: CacheManager) -> None:
        assert not await cache.delete("nonexistent")

    async def test_has(self, cache: CacheManager) -> None:
        await cache.set("key1", "value1")
        assert await cache.has("key1")
        assert not await cache.has("nonexistent")

    async def test_overwrite(self, cache: CacheManager) -> None:
        await cache.set("key", "old")
        await cache.set("key", "new")
        assert await cache.get("key") == "new"


@pytest.mark.asyncio
class TestCacheManagerNamespace:
    async def test_automatic_namespace(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1")
        # The backend stores the namespaced key
        entry = await cache.get("k1")
        assert entry == "v1"

    async def test_key_prefixed(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1")
        # Check raw backend has the namespaced key
        entry = await cache.backend.get("test_ns:k1")
        assert entry is not None
        assert entry.value == "v1"

    async def test_build_key(self) -> None:
        key = CacheManager.build_key("team", "42")
        assert key == "team:42"

    async def test_build_key_single(self) -> None:
        key = CacheManager.build_key("elo")
        assert key == "elo"


@pytest.mark.asyncio
class TestCacheManagerGetOrCompute:
    async def test_get_or_compute_hit(self, cache: CacheManager) -> None:
        await cache.set("k1", "cached_value")
        computed = await cache.get_or_compute("k1", "should_not_call")
        assert computed == "cached_value"

    async def test_get_or_compute_miss(self, cache: CacheManager) -> None:
        computed = await cache.get_or_compute("k2", "fresh_value")
        assert computed == "fresh_value"

    async def test_get_or_compute_async(self, cache: CacheManager) -> None:
        async def async_compute() -> str:
            return "async_value"

        computed = await cache.get_or_compute("k3", async_compute())
        assert computed == "async_value"

    async def test_get_or_compute_caches(self, cache: CacheManager) -> None:
        await cache.get_or_compute("k1", "computed_once")
        # Should hit cache now
        value = await cache.get("k1")
        assert value == "computed_once"


@pytest.mark.asyncio
class TestCacheManagerInvalidation:
    async def test_invalidate_key(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1")
        assert await cache.invalidate("k1")
        assert not await cache.has("k1")

    async def test_invalidate_by_tag(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1", tags={"team"})
        await cache.set("k2", "v2", tags={"team"})
        await cache.set("k3", "v3", tags={"other"})
        deleted = await cache.invalidate_by_tag("team")
        assert deleted >= 2
        assert not await cache.has("k1")
        assert not await cache.has("k2")
        assert await cache.has("k3")


@pytest.mark.asyncio
class TestCacheManagerStats:
    async def test_stats_initial(self, cache: CacheManager) -> None:
        stats = await cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0

    async def test_stats_tracking(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1")
        await cache.get("k1")  # hit
        await cache.get("missing")  # miss
        stats = await cache.stats()
        assert stats.hits >= 1
        assert stats.misses >= 1

    async def test_to_dict(self, cache: CacheManager) -> None:
        await cache.get("miss1")
        await cache.get("miss2")
        d = cache.to_dict()
        assert d["namespace"] == "test_ns"
        assert d["misses"] >= 2
        assert "hit_ratio" in d


@pytest.mark.asyncio
class TestCacheManagerBatch:
    async def test_get_many(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1")
        await cache.set("k2", "v2")
        results = await cache.get_many(["k1", "k2", "missing"])
        assert results == {"k1": "v1", "k2": "v2", "missing": None}

    async def test_set_many(self, cache: CacheManager) -> None:
        await cache.set_many({"k1": "v1", "k2": "v2"}, ttl=60)
        assert await cache.get("k1") == "v1"
        assert await cache.get("k2") == "v2"


@pytest.mark.asyncio
class TestCacheManagerTTL:
    async def test_default_ttl(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1")  # Uses default_ttl=3600
        entry = await cache.get_entry("k1")
        assert entry is not None
        assert entry.ttl == 3600

    async def test_custom_ttl(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1", ttl=60)
        entry = await cache.get_entry("k1")
        assert entry is not None
        assert entry.ttl == 60

    async def test_zero_ttl(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1", ttl=0)
        entry = await cache.get_entry("k1")
        assert entry is not None
        assert entry.ttl == 0


@pytest.mark.asyncio
class TestCacheManagerCleanup:
    async def test_cleanup(self, cache: CacheManager) -> None:
        await cache.set("temp", "gone", ttl=0.05)
        await cache.set("perm", "here", ttl=0)
        import time
        time.sleep(0.1)
        result = await cache.cleanup()
        assert result.expired_cleared >= 1
        assert await cache.get("perm") == "here"
        assert await cache.get("temp") is None
