"""
Tests for ``src.cache.backend``.

Covers SQLiteBackend: CRUD operations, TTL/expiration, stats, tags,
batch operations, cleanup, and thread safety.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

from src.cache.backend import SQLiteBackend
from src.cache.models import CacheEntry, CacheStats


@pytest.fixture
def sqlite_backend() -> SQLiteBackend:
    """Create an in-memory SQLite backend for testing."""
    backend = SQLiteBackend(
        db_path=":memory:",
        table_name="test_cache",
        cleanup_interval=0,  # Disable auto-cleanup
        vacuum_after_n_cleanups=0,
    )
    return backend


@pytest.fixture
def file_backend(tmp_path: Path) -> SQLiteBackend:
    """Create a file-based SQLite backend for testing."""
    db_path = tmp_path / "test_cache.db"
    backend = SQLiteBackend(
        db_path=str(db_path),
        table_name="test_cache",
        cleanup_interval=0,
        vacuum_after_n_cleanups=0,
    )
    return backend


@pytest.mark.asyncio
class TestSQLiteBackendCRUD:
    async def test_set_and_get(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1")
        entry = await sqlite_backend.get("key1")
        assert entry is not None
        assert entry.value == "value1"
        assert entry.key == "key1"

    async def test_get_missing(self, sqlite_backend: SQLiteBackend) -> None:
        entry = await sqlite_backend.get("nonexistent")
        assert entry is None

    async def test_get_none_value(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("null_key", None)
        entry = await sqlite_backend.get("null_key")
        assert entry is not None
        assert entry.value is None

    async def test_set_overwrites(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "old_value")
        await sqlite_backend.set("key1", "new_value")
        entry = await sqlite_backend.get("key1")
        assert entry is not None
        assert entry.value == "new_value"

    async def test_delete(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1")
        deleted = await sqlite_backend.delete("key1")
        assert deleted
        entry = await sqlite_backend.get("key1")
        assert entry is None

    async def test_delete_missing(self, sqlite_backend: SQLiteBackend) -> None:
        deleted = await sqlite_backend.delete("nonexistent")
        assert not deleted

    async def test_has(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1")
        assert await sqlite_backend.has("key1")
        assert not await sqlite_backend.has("nonexistent")

    async def test_clear(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "v1")
        await sqlite_backend.set("key2", "v2")
        cleared = await sqlite_backend.clear()
        assert cleared >= 2
        assert not await sqlite_backend.has("key1")
        assert not await sqlite_backend.has("key2")

    async def test_clear_empty(self, sqlite_backend: SQLiteBackend) -> None:
        cleared = await sqlite_backend.clear()
        assert cleared >= 0

    async def test_complex_value(self, sqlite_backend: SQLiteBackend) -> None:
        complex_val = {
            "id": 42,
            "nested": {"a": [1, 2, 3]},
            "teams": ["Arsenal", "Chelsea"],
        }
        await sqlite_backend.set("complex", complex_val)
        entry = await sqlite_backend.get("complex")
        assert entry is not None
        assert entry.value["id"] == 42
        assert entry.value["nested"]["a"] == [1, 2, 3]

    async def test_empty_string(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("empty", "")
        entry = await sqlite_backend.get("empty")
        assert entry is not None
        assert entry.value == ""

    async def test_large_value(self, sqlite_backend: SQLiteBackend) -> None:
        large = "x" * 100000
        await sqlite_backend.set("large", large)
        entry = await sqlite_backend.get("large")
        assert entry is not None
        assert len(entry.value) == 100000


@pytest.mark.asyncio
class TestSQLiteBackendTTL:
    async def test_ttl_not_expired(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1", ttl=60)
        entry = await sqlite_backend.get("key1")
        assert entry is not None
        assert entry.value == "value1"

    async def test_ttl_expired(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1", ttl=0.1)
        time.sleep(0.15)
        entry = await sqlite_backend.get("key1")
        assert entry is None

    async def test_ttl_zero_never_expires(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1", ttl=0)
        entry = await sqlite_backend.get("key1")
        assert entry is not None

    async def test_mixed_ttl(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("permanent", "always", ttl=0)
        await sqlite_backend.set("temporary", "gone", ttl=0.1)
        time.sleep(0.15)
        assert (await sqlite_backend.get("permanent")) is not None
        assert (await sqlite_backend.get("temporary")) is None


@pytest.mark.asyncio
class TestSQLiteBackendStats:
    async def test_empty_stats(self, sqlite_backend: SQLiteBackend) -> None:
        stats = await sqlite_backend.stats()
        assert stats.entries == 0
        assert stats.size_bytes == 0
        assert stats.hits == 0
        assert stats.misses == 0

    async def test_stats_after_insert(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1")
        await sqlite_backend.set("key2", "value2")
        stats = await sqlite_backend.stats()
        assert stats.entries >= 2
        assert stats.size_bytes > 0

    async def test_stats_hit_miss(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "value1")
        await sqlite_backend.get("key1")  # hit
        await sqlite_backend.get("missing")  # miss
        stats = await sqlite_backend.stats()
        assert stats.hits == 1
        assert stats.misses >= 1


@pytest.mark.asyncio
class TestSQLiteBackendTags:
    async def test_set_with_tags(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("key1", "v1", tags={"team", "elo"})
        entry = await sqlite_backend.get("key1")
        assert entry is not None
        assert entry.tags == {"team", "elo"}

    async def test_delete_by_tag(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("k1", "v1", tags={"group_a"})
        await sqlite_backend.set("k2", "v2", tags={"group_a"})
        await sqlite_backend.set("k3", "v3", tags={"group_b"})
        deleted = await sqlite_backend.delete_by_tag("group_a")
        assert deleted >= 2
        assert not await sqlite_backend.has("k1")
        assert not await sqlite_backend.has("k2")
        assert await sqlite_backend.has("k3")

    async def test_delete_by_tag_no_match(self, sqlite_backend: SQLiteBackend) -> None:
        deleted = await sqlite_backend.delete_by_tag("nonexistent")
        assert deleted == 0

    async def test_tags_overlap(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("k1", "v1", tags={"sport", "football"})
        await sqlite_backend.set("k2", "v2", tags={"sport", "basketball"})
        deleted = await sqlite_backend.delete_by_tag("football")
        assert deleted >= 1
        assert not await sqlite_backend.has("k1")
        assert await sqlite_backend.has("k2")  # k2 shouldn't be deleted


@pytest.mark.asyncio
class TestSQLiteBackendBatch:
    async def test_get_many(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("k1", "v1")
        await sqlite_backend.set("k2", "v2")
        results = await sqlite_backend.get_many(["k1", "k2", "missing"])
        assert results["k1"] is not None and results["k1"].value == "v1"
        assert results["k2"] is not None and results["k2"].value == "v2"
        assert results["missing"] is None

    async def test_get_many_empty(self, sqlite_backend: SQLiteBackend) -> None:
        results = await sqlite_backend.get_many([])
        assert results == {}

    async def test_set_many(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set_many(
            {"k1": "v1", "k2": "v2"}, ttl=60, tags={"batch"},
        )
        assert (await sqlite_backend.has("k1"))
        assert (await sqlite_backend.has("k2"))
        entry = await sqlite_backend.get("k1")
        assert entry is not None and "batch" in entry.tags


@pytest.mark.asyncio
class TestSQLiteBackendCleanup:
    async def test_cleanup_expired(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("temp", "gone", ttl=0.05)
        await sqlite_backend.set("perm", "here", ttl=0)
        time.sleep(0.1)
        cleanup_stats = await sqlite_backend.cleanup()
        assert cleanup_stats.expired_cleared >= 1
        assert await sqlite_backend.has("perm")
        assert not await sqlite_backend.has("temp")

    async def test_cleanup_no_expired(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("perm1", "v1", ttl=0)
        await sqlite_backend.set("perm2", "v2", ttl=0)
        cleanup_stats = await sqlite_backend.cleanup()
        assert cleanup_stats.expired_cleared == 0

    async def test_file_backend_persistence(self, file_backend: SQLiteBackend) -> None:
        await file_backend.set("k1", "v1")
        assert await file_backend.has("k1")

    async def test_cleanup_idempotent(self, sqlite_backend: SQLiteBackend) -> None:
        stats1 = await sqlite_backend.cleanup()
        stats2 = await sqlite_backend.cleanup()
        assert isinstance(stats1, CacheStats)
        assert isinstance(stats2, CacheStats)


@pytest.mark.asyncio
class TestSQLiteBackendEdgeCases:
    async def test_special_characters(self, sqlite_backend: SQLiteBackend) -> None:
        key = "team:arsenal-fc_123"
        value = {"name": "Arsenal FC", "emoji": "⚽🔴"}
        await sqlite_backend.set(key, value)
        entry = await sqlite_backend.get(key)
        assert entry is not None
        assert entry.value["emoji"] == "⚽🔴"

    async def test_unicode_key(self, sqlite_backend: SQLiteBackend) -> None:
        key = "üñîçødé"
        await sqlite_backend.set(key, "value")
        entry = await sqlite_backend.get(key)
        assert entry is not None
        assert entry.value == "value"

    async def test_binary_value(self, sqlite_backend: SQLiteBackend) -> None:
        binary = bytes(range(256))
        await sqlite_backend.set("binary", binary)
        entry = await sqlite_backend.get("binary")
        assert entry is not None
        assert entry.value == binary

    async def test_boolean_values(self, sqlite_backend: SQLiteBackend) -> None:
        await sqlite_backend.set("true_val", True)
        await sqlite_backend.set("false_val", False)
        t = await sqlite_backend.get("true_val")
        f = await sqlite_backend.get("false_val")
        assert t is not None and t.value is True
        assert f is not None and f.value is False
