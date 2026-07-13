"""
Tests for ``src.cache.models``.

Covers CacheEntry, CacheStats, CacheKey — serialization, expiration,
statistics computation, and key construction.
"""

from __future__ import annotations

import time
import pickle

import pytest

from src.cache.models import CacheEntry, CacheStats, CacheKey


class TestCacheEntry:
    def test_create_minimal(self) -> None:
        entry = CacheEntry(key="test", value=42)
        assert entry.key == "test"
        assert entry.value == 42
        assert entry.ttl == 0.0
        assert not entry.is_expired
        assert entry.age >= 0

    def test_expiration(self) -> None:
        entry = CacheEntry(key="test", value="x", ttl=0.1)
        assert not entry.is_expired
        time.sleep(0.15)
        assert entry.is_expired

    def test_no_expiration(self) -> None:
        entry = CacheEntry(key="test", value="x", ttl=0)
        time.sleep(0.01)
        assert not entry.is_expired

    def test_remaining_ttl(self) -> None:
        entry = CacheEntry(key="test", value="x", ttl=10)
        assert entry.remaining_ttl <= 10
        assert entry.remaining_ttl > 0

    def test_no_ttl_remaining(self) -> None:
        entry = CacheEntry(key="test", value="x", ttl=0)
        assert entry.remaining_ttl == float("inf")

    def test_serialization_roundtrip(self) -> None:
        entry = CacheEntry(
            key="test", value={"a": 1, "b": [2, 3]},
            ttl=3600, tags={"team", "elo"},
        )
        data = entry.to_bytes()
        restored = CacheEntry.from_bytes(data)
        assert restored.key == "test"
        assert restored.value == {"a": 1, "b": [2, 3]}
        assert restored.ttl == 3600
        assert restored.tags == {"team", "elo"}

    def test_serialization_complex(self) -> None:
        value = {
            "id": 42,
            "name": "Arsenal",
            "stats": {"wins": 20, "draws": 5, "losses": 3},
            "tags": ["premier_league", "england"],
        }
        entry = CacheEntry(key="team:42", value=value, ttl=7200)
        restored = CacheEntry.from_bytes(entry.to_bytes())
        assert restored.value["id"] == 42
        assert restored.value["name"] == "Arsenal"
        assert restored.value["stats"]["wins"] == 20

    def test_empty_tags(self) -> None:
        entry = CacheEntry(key="test", value=1)
        assert entry.tags == set()

    def test_tags_preserved(self) -> None:
        entry = CacheEntry(key="test", value=1, tags={"a", "b", "c"})
        assert entry.tags == {"a", "b", "c"}

    def test_sizeof(self) -> None:
        entry = CacheEntry(key="small", value=1)
        assert entry.__sizeof__() > 0


class TestCacheStats:
    def test_empty_stats(self) -> None:
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_ratio == 0.0
        assert stats.total_requests == 0

    def test_hit_ratio(self) -> None:
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_ratio == 0.8
        assert stats.total_requests == 100

    def test_hit_ratio_no_requests(self) -> None:
        stats = CacheStats()
        assert stats.hit_ratio == 0.0

    def test_to_dict(self) -> None:
        stats = CacheStats(hits=10, misses=5, entries=100, size_bytes=4096)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["entries"] == 100
        assert d["size_bytes"] == 4096
        assert d["hit_ratio"] == pytest.approx(10 / 15, rel=1e-3)

    def test_iadd(self) -> None:
        s1 = CacheStats(hits=10, misses=5)
        s2 = CacheStats(hits=20, misses=10, entries=50, size_bytes=1024)
        s1 += s2
        assert s1.hits == 30
        assert s1.misses == 15
        assert s1.entries == 50  # Uses latest count
        assert s1.size_bytes == 1024

    def test_repr(self) -> None:
        stats = CacheStats(hits=90, misses=10)
        r = repr(stats)
        assert "CacheStats" in r
        assert "90" in r
        assert "90.0%" in r


class TestCacheKey:
    def test_basic_key(self) -> None:
        key = CacheKey(namespace="api", parts=["team", "42"])
        assert str(key) == "api:team:42"

    def test_single_part(self) -> None:
        key = CacheKey(namespace="elo", parts=["123"])
        assert str(key) == "elo:123"

    def test_default_namespace(self) -> None:
        key = CacheKey(parts=["test"])
        assert str(key) == "default:test"

    def test_hash(self) -> None:
        k1 = CacheKey(namespace="ns", parts=["a", "b"])
        k2 = CacheKey(namespace="ns", parts=["a", "b"])
        assert hash(k1) == hash(k2)

    def test_from_str(self) -> None:
        key = CacheKey.from_str("api:team:42")
        assert key.namespace == "api"
        assert key.parts == ["team", "42"]

    def test_from_str_no_namespace(self) -> None:
        key = CacheKey.from_str("simple")
        assert key.namespace == "default"
        assert key.parts == ["simple"]

    def test_for_url(self) -> None:
        url = "https://api.example.com/data"
        key = CacheKey.for_url(url)
        assert key.namespace == "url"
        assert len(key.parts) == 1
        assert len(key.parts[0]) == 16  # SHA256 prefix

    def test_for_func(self) -> None:
        key = CacheKey.for_func("my_func", (1, 2), {"kwarg": "val"})
        assert key.namespace == "func"
        assert key.parts[0] == "my_func"
        assert len(key.parts[1]) == 16  # SHA256 hash
