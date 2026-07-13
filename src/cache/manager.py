"""
CacheManager — high-level cache orchestration.

Coordinates one or more backends, providing:
- Unified get/set/delete API
- Tag-based invalidation
- Cache statistics aggregation
- Lazy cleanup scheduling
- Namespace support for multi-tenant setups
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from src.cache.backend import CacheBackend
from src.cache.models import CacheEntry, CacheStats, CacheKey

logger = logging.getLogger(__name__)


class CacheManager:
    """High-level cache orchestrator.

    Wraps a CacheBackend and adds:
    - **Namespaces** — Automatically prefix keys with a namespace.
    - **Statistics** — Tracks hits/misses across all operations.
    - **Invocation patterns** — Common patterns like
      ``get_or_compute`` and ``get_or_fetch``.
    - **Monitoring** — ``to_dict()`` for dashboard integration.

    Parameters
    ----------
    backend : CacheBackend
        The storage backend (e.g. SQLiteBackend, RedisBackend).
    namespace : str
        Default namespace (default ``football``).
        All keys get prefixed with ``{namespace}:``.
    default_ttl : float
        Default TTL for entries that don't specify one (default 3600).
    enable_stats : bool
        Track hit/miss statistics (default True).
    """

    def __init__(
        self,
        backend: CacheBackend,
        namespace: str = "football",
        default_ttl: float = 3600.0,
        enable_stats: bool = True,
    ) -> None:
        self.backend = backend
        self.namespace = namespace
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats

        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._get_count: int = 0
        self._set_count: int = 0
        self._delete_count: int = 0

    # ── Key management ─────────────────────────────────

    def _key(self, key: str) -> str:
        """Prefix a key with the namespace."""
        if key.startswith(f"{self.namespace}:"):
            return key
        return f"{self.namespace}:{key}"

    @staticmethod
    def build_key(*parts: str) -> str:
        """Build a colon-separated cache key.

        Examples
        --------
        ::

            CacheManager.build_key("team", "42")      → "team:42"
            CacheManager.build_key("elo", "123")      → "elo:123"
            CacheManager.build_key("match", "1", "2") → "match:1:2"
        """
        return ":".join(parts)

    # ── Core operations ────────────────────────────────

    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key.

        Returns the cached value (unwrapped from CacheEntry)
        or None if not found / expired.
        """
        full_key = self._key(key)
        entry = await self.backend.get(full_key)

        with self._lock:
            self._get_count += 1
            if entry is not None:
                self._hits += 1
            else:
                self._misses += 1

        return entry.value if entry is not None else None

    async def get_entry(self, key: str) -> CacheEntry | None:
        """Retrieve a full CacheEntry (with metadata)."""
        full_key = self._key(key)
        return await self.backend.get(full_key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        tags: set[str] | None = None,
    ) -> None:
        """Store a value.

        Parameters
        ----------
        key : str
            Cache key.
        value : Any
            Value to cache (must be pickle-serializable).
        ttl : float, optional
            Time-to-live in seconds. Defaults to ``default_ttl``.
        tags : set[str], optional
            Tags for group invalidation.
        """
        full_key = self._key(key)
        actual_ttl = ttl if ttl is not None else self.default_ttl
        await self.backend.set(full_key, value, ttl=actual_ttl, tags=tags)

        with self._lock:
            self._set_count += 1

    async def delete(self, key: str) -> bool:
        """Delete a single cache entry."""
        full_key = self._key(key)
        result = await self.backend.delete(full_key)

        with self._lock:
            self._delete_count += 1

        return result

    async def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        full_key = self._key(key)
        return await self.backend.has(full_key)

    async def clear(self) -> int:
        """Clear ALL entries in this namespace.

        WARNING: This clears the entire cache, not just the namespace.
        For targeted clearing, use ``delete_by_tag`` instead.
        """
        return await self.backend.clear()

    async def get_many(self, keys: list[str]) -> dict[str, Any | None]:
        """Retrieve multiple values at once."""
        full_keys = [self._key(k) for k in keys]
        entries = await self.backend.get_many(full_keys)

        results: dict[str, Any | None] = {}
        for key, full_key in zip(keys, full_keys):
            entry = entries.get(full_key)
            results[key] = entry.value if entry is not None else None

        with self._lock:
            self._get_count += len(keys)

        return results

    async def set_many(
        self,
        entries: dict[str, Any],
        ttl: float | None = None,
        tags: set[str] | None = None,
    ) -> None:
        """Store multiple values at once."""
        actual_ttl = ttl if ttl is not None else self.default_ttl
        prefixed = {self._key(k): v for k, v in entries.items()}
        await self.backend.set_many(prefixed, ttl=actual_ttl, tags=tags)

        with self._lock:
            self._set_count += len(entries)

    # ── Convenience patterns ───────────────────────────

    async def get_or_compute(
        self,
        key: str,
        compute: Any,
        ttl: float | None = None,
        tags: set[str] | None = None,
    ) -> Any:
        """Return cached value or compute and cache it.

        Parameters
        ----------
        key : str
            Cache key.
        compute : Callable | Coroutine
            Function to compute the value if not cached.
            Can be sync or async.
        ttl : float, optional
            Custom TTL. Uses default_ttl if not provided.
        tags : set[str], optional
            Tags for invalidation.

        Returns
        -------
        Any
            The cached or computed value.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Compute the value
        if asyncio.iscoroutine(compute):
            value = await compute
        else:
            value = compute

        await self.set(key, value, ttl=ttl, tags=tags)
        return value

    # ── Invalidation ───────────────────────────────────

    async def invalidate(self, key: str) -> bool:
        """Delete a specific key (alias for ``delete()``)."""
        return await self.delete(key)

    async def invalidate_by_tag(self, tag: str) -> int:
        """Delete all entries with the given tag.

        Examples
        --------
        ::

            # Invalidate all team-related cache entries
            await cache.invalidate_by_tag("team")
        """
        return await self.backend.delete_by_tag(tag)

    # ── Maintenance ────────────────────────────────────

    async def cleanup(self) -> CacheStats:
        """Remove expired entries and return cleanup stats."""
        return await self.backend.cleanup()

    async def stats(self) -> CacheStats:
        """Return aggregated cache statistics."""
        backend_stats = await self.backend.stats()
        with self._lock:
            backend_stats.hits = self._hits
            backend_stats.misses = self._misses
        return backend_stats

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of cache stats for monitoring dashboards."""
        stats_dict = {
            "namespace": self.namespace,
            "default_ttl": self.default_ttl,
            "get_count": self._get_count,
            "set_count": self._set_count,
            "delete_count": self._delete_count,
            "hits": self._hits,
            "misses": self._misses,
        }
        total = self._hits + self._misses
        stats_dict["hit_ratio"] = round(self._hits / total, 4) if total > 0 else 0.0
        return stats_dict

    async def close(self) -> None:
        """Close the backend connection."""
        if hasattr(self.backend, "close"):
            await self.backend.close()
