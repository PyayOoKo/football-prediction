"""
Cache data models — shared types used across all cache backends.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cached item with metadata.

    Attributes
    ----------
    key : str
        Unique cache key.
    value : T
        Cached value (any pickle-serializable object).
    ttl : float
        Time-to-live in seconds. 0 means no expiration.
    created_at : float
        Unix timestamp when the entry was created.
    size_bytes : int
        Approximate byte size of the serialized entry.
    tags : set[str]
        Optional tags for group invalidation.
    """

    key: str
    value: T
    ttl: float = 0.0
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    tags: set[str] = field(default_factory=set)

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl

    @property
    def age(self) -> float:
        """Age of the entry in seconds."""
        return time.time() - self.created_at

    @property
    def remaining_ttl(self) -> float:
        """Seconds until this entry expires (0 if no TTL)."""
        if self.ttl <= 0:
            return float("inf")
        remaining = self.ttl - self.age
        return max(0.0, remaining)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        return pickle.dumps({
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
        })

    @staticmethod
    def from_bytes(data: bytes) -> CacheEntry[Any]:
        """Deserialize from bytes."""
        raw = pickle.loads(data)
        return CacheEntry(
            key=raw["key"],
            value=raw["value"],
            ttl=raw["ttl"],
            created_at=raw["created_at"],
            size_bytes=raw.get("size_bytes", 0),
            tags=raw.get("tags", set()),
        )

    def __sizeof__(self) -> int:
        """Return approximate memory size."""
        return len(self.to_bytes())


@dataclass
class CacheStats:
    """Statistics for a cache backend or manager.

    Attributes
    ----------
    hits : int
        Number of cache hits.
    misses : int
        Number of cache misses.
    entries : int
        Current number of cached entries.
    size_bytes : int
        Approximate total byte size of cached data.
    expired_cleared : int
        Number of expired entries cleared during cleanup.
    hit_ratio : float
        Cache hit ratio (0.0–1.0).
    """

    hits: int = 0
    misses: int = 0
    entries: int = 0
    size_bytes: int = 0
    expired_cleared: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": self.entries,
            "size_bytes": self.size_bytes,
            "expired_cleared": self.expired_cleared,
            "hit_ratio": round(self.hit_ratio, 4),
            "total_requests": self.total_requests,
        }

    def __iadd__(self, other: CacheStats) -> CacheStats:
        self.hits += other.hits
        self.misses += other.misses
        self.entries = other.entries  # Use latest count
        self.size_bytes = other.size_bytes
        self.expired_cleared += other.expired_cleared
        return self

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"entries={self.entries}, hit_ratio={self.hit_ratio:.1%})"
        )


@dataclass
class CacheKey:
    """Structured cache key with namespace support.

    Examples
    --------
    ::

        key = CacheKey(namespace="api", parts=["team", "42"])
        assert str(key) == "api:team:42"
    """

    namespace: str = "default"
    parts: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.namespace}:{':'.join(self.parts)}"

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def from_str(key: str) -> CacheKey:
        """Parse a colon-separated key string."""
        if ":" in key:
            namespace, *parts = key.split(":")
            return CacheKey(namespace=namespace, parts=parts)
        return CacheKey(namespace="default", parts=[key])

    @staticmethod
    def for_url(url: str) -> CacheKey:
        """Create a cache key from a URL."""
        import hashlib
        hash_str = hashlib.sha256(url.encode()).hexdigest()[:16]
        return CacheKey(namespace="url", parts=[hash_str])

    @staticmethod
    def for_func(func_name: str, args: tuple[Any, ...],
                 kwargs: dict[str, Any]) -> CacheKey:
        """Create a cache key from a function call."""
        import hashlib
        raw = f"{func_name}:{pickle.dumps((args, sorted(kwargs.items())))}"
        hash_str = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return CacheKey(namespace="func", parts=[func_name, hash_str])
