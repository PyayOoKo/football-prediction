"""
Reusable caching framework for the football prediction platform.

Supports:
- **SQLite** cache backend (local, zero-config, thread-safe)
- **Redis** cache backend (distributed, production)
- Time-to-live (TTL) with automatic expiration
- Tag-based cache invalidation
- Decorator interface for sync and async functions
- Cache statistics and monitoring

Quick start
-----------
::

    from src.cache import CacheManager, SQLiteBackend, cached

    # Option 1: Use the decorator
    @cached(ttl=3600)
    def get_team(team_id: int) -> dict:
        return {"id": team_id, "name": "Arsenal"}

    # Option 2: Use the CacheManager directly
    cache = CacheManager(SQLiteBackend("data/cache/app_cache.db"))
    await cache.set("team:42", {"name": "Arsenal"}, ttl=3600)
    team = await cache.get("team:42")
    print(cache.stats)

Backends
--------
- ``SQLiteBackend`` — Local SQLite database, no external deps beyond stdlib.
- ``RedisBackend`` — Remote Redis server, requires ``redis-py``.

See Also
--------
- ``CacheManager`` — Orchestrator with invalidation, stats, and cleanup.
- ``cached`` — Decorator for transparent function caching.
"""

from __future__ import annotations

from src.cache.backend import CacheBackend, SQLiteBackend, RedisBackend
from src.cache.decorators import cached, invalidate, set_cache
from src.cache.manager import CacheManager
from src.cache.models import CacheEntry, CacheStats, CacheKey

__all__ = [
    "CacheBackend",
    "SQLiteBackend",
    "RedisBackend",
    "CacheManager",
    "CacheEntry",
    "CacheStats",
    "CacheKey",
    "cached",
    "invalidate",
    "set_cache",
]
