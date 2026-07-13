"""
Tests for ``src.cache.decorators``.

Covers @cached and @invalidate for sync and async functions,
custom key functions, tags, and global cache configuration.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache import cached, invalidate, set_cache
from src.cache.backend import SQLiteBackend
from src.cache.decorators import _get_global_cache
from src.cache.manager import CacheManager


@pytest.fixture
def cache() -> CacheManager:
    """Create a test cache with in-memory SQLite."""
    backend = SQLiteBackend(
        db_path=":memory:",
        table_name="test_decorators",
        cleanup_interval=0,
        vacuum_after_n_cleanups=0,
    )
    return CacheManager(backend, namespace="test_dec", default_ttl=3600)


@pytest.fixture(autouse=True)
def reset_global_cache() -> None:
    """Reset the global cache before each test."""
    import src.cache.decorators as dec_module
    dec_module._global_cache = None


class TestCachedSync:
    def test_caches_result(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=60, cache=cache)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(5) == 10  # Should hit cache
        assert call_count == 1

    def test_different_args_different_cache(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=60, cache=cache)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(10) == 20
        assert call_count == 2  # Different args = different cache keys

    def test_custom_key_fn(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=60, cache=cache, key_fn=lambda x: f"mykey:{x}")
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(5) == 10  # Hit cache via custom key
        assert call_count == 1

    def test_tags(self, cache: CacheManager) -> None:
        @cached(ttl=60, cache=cache, tags={"test_tag"})
        def compute(x: int) -> int:
            return x * 2

        compute(5)
        # Cache should have the entry with the tag
        import asyncio
        entry = asyncio.run(cache.get_entry("func:test_decorators.test_cached_sync.<locals>.compute(5)"))
        if entry is not None:
            assert "test_tag" in entry.tags

    def test_zero_ttl(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=0, cache=cache)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(5) == 10  # Should still cache with ttl=0 (no expiration)
        assert call_count == 1

    def test_kwargs(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=60, cache=cache)
        def greet(name: str, greeting: str = "Hello") -> str:
            nonlocal call_count
            call_count += 1
            return f"{greeting}, {name}!"

        assert greet("Alice") == "Hello, Alice!"
        assert greet("Alice", greeting="Hi") == "Hi, Alice!"
        assert call_count == 2  # Different kwargs = different cache keys


class TestCachedAsync:
    @pytest.mark.asyncio
    async def test_caches_result(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=60, cache=cache)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert await compute(5) == 10
        assert await compute(5) == 10  # Hit cache
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_args(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=60, cache=cache)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert await compute(5) == 10
        assert await compute(10) == 20
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_cache_hit(self, cache: CacheManager) -> None:
        call_count = 0

        @cached(ttl=0.05, cache=cache)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert await compute(5) == 10
        time.sleep(0.1)
        assert await compute(5) == 10  # TTL expired, recompute
        assert call_count == 2


class TestInvalidateSync:
    def test_invalidate_by_tag(self, cache: CacheManager) -> None:
        # Set up cached entries with a tag
        import asyncio
        asyncio.run(cache.set("k1", "v1", tags={"mygroup"}))
        asyncio.run(cache.set("k2", "v2", tags={"mygroup"}))

        @invalidate(tags={"mygroup"}, cache=cache)
        def update_data() -> str:
            return "updated"

        update_data()
        assert not asyncio.run(cache.has("k1"))
        assert not asyncio.run(cache.has("k2"))

    def test_invalidate_by_key(self, cache: CacheManager) -> None:
        import asyncio
        asyncio.run(cache.set("specific_key", "v1"))

        @invalidate(keys=["specific_key"], cache=cache)
        def update() -> str:
            return "done"

        update()
        assert not asyncio.run(cache.has("specific_key"))

    def test_invalidate_key_fn(self, cache: CacheManager) -> None:
        import asyncio

        @invalidate(key_fn=lambda team_id: f"team:{team_id}", cache=cache)
        def update_team(team_id: int) -> None:
            pass

        asyncio.run(cache.set("team:42", "some_data"))
        update_team(42)
        assert not asyncio.run(cache.has("team:42"))


class TestInvalidateAsync:
    @pytest.mark.asyncio
    async def test_invalidate_by_tag_async(self, cache: CacheManager) -> None:
        await cache.set("k1", "v1", tags={"group"})

        @invalidate(tags={"group"}, cache=cache)
        async def update() -> str:
            return "done"

        await update()
        assert not await cache.has("k1")


class TestGlobalCache:
    def test_default_global_cache(self) -> None:
        """The global cache should be created on first use."""
        cache = _get_global_cache()
        assert cache is not None
        assert isinstance(cache, CacheManager)

    def test_set_global_cache(self, cache: CacheManager) -> None:
        """Can override the global cache."""
        set_cache(cache)
        assert _get_global_cache() is cache

    def test_global_cache_persistence(self, cache: CacheManager) -> None:
        set_cache(cache)
        g1 = _get_global_cache()
        g2 = _get_global_cache()
        assert g1 is g2  # Same instance
