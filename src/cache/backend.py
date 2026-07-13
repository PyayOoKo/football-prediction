"""
Cache backends — pluggable storage for cached data.

Provides:
- ``CacheBackend`` — Abstract base class that all backends implement.
- ``SQLiteBackend`` — Zero-config local cache using SQLite (thread-safe).
- ``RedisBackend`` — Distributed cache using Redis (optional, requires ``redis``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.cache.models import CacheEntry, CacheStats, CacheKey

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Abstract base
# ═══════════════════════════════════════════════════════════


class CacheBackend(ABC):
    """Abstract base for all cache storage backends.

    All backends implement: get, set, delete, clear, has, stats,
    get_many, set_many, delete_by_tag, cleanup.
    """

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cached entry by key.

        Returns ``None`` if the key doesn't exist or has expired.
        """
        ...

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: float = 0.0,
        tags: set[str] | None = None,
    ) -> None:
        """Store a value in the cache.

        Parameters
        ----------
        key : str
            Unique cache key.
        value : Any
            Any pickle-serializable object.
        ttl : float
            Time-to-live in seconds. 0 = no expiration.
        tags : set[str], optional
            Tags for group invalidation.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a single cache entry. Returns True if existed."""
        ...

    @abstractmethod
    async def clear(self) -> int:
        """Clear all entries. Returns number of deleted entries."""
        ...

    @abstractmethod
    async def has(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        ...

    @abstractmethod
    async def stats(self) -> CacheStats:
        """Return current cache statistics."""
        ...

    async def get_many(self, keys: list[str]) -> dict[str, CacheEntry | None]:
        """Retrieve multiple entries at once.

        Default implementation calls ``get()`` for each key.
        Backends may override for batch-optimized access.
        """
        return {key: await self.get(key) for key in keys}

    async def set_many(
        self,
        entries: dict[str, Any],
        ttl: float = 0.0,
        tags: set[str] | None = None,
    ) -> None:
        """Store multiple entries at once.

        Default implementation calls ``set()`` for each entry.
        Backends may override for batch-optimized access.
        """
        for key, value in entries.items():
            await self.set(key, value, ttl=ttl, tags=tags)

    @abstractmethod
    async def delete_by_tag(self, tag: str) -> int:
        """Delete all entries with the given tag. Returns count."""
        ...

    @abstractmethod
    async def cleanup(self) -> CacheStats:
        """Remove all expired entries. Returns stats about the cleanup."""
        ...


# ═══════════════════════════════════════════════════════════
#  SQLite backend
# ═══════════════════════════════════════════════════════════


class SQLiteBackend(CacheBackend):
    """SQLite-based cache backend.

    Stores entries in a local SQLite database with:
    - TTL-based auto-expiration on reads
    - Background cleanup thread for expired entries
    - Tag-based group invalidation
    - Thread-safe via connection pooling with WAL mode

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file. Use ``:memory:`` for temp.
    table_name : str
        SQLite table name (default ``cache_entries``).
    cleanup_interval : float
        Seconds between automatic cleanup runs (default 300 = 5 min).
        Set to 0 to disable auto-cleanup.
    vacuum_after_n_cleanups : int
        Run PRAGMA vacuum after every N cleanups (default 10).
        Helps prevent unbounded database growth.
    """

    def __init__(
        self,
        db_path: str | Path = "data/cache/cache.db",
        table_name: str = "cache_entries",
        cleanup_interval: float = 300.0,
        vacuum_after_n_cleanups: int = 10,
    ) -> None:
        self.db_path = str(db_path)
        self.table_name = table_name
        self.cleanup_interval = cleanup_interval
        self.vacuum_after_n_cleanups = vacuum_after_n_cleanups

        # Stats (thread-safe via lock)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._cleanup_count = 0

        # Thread-local connections for thread safety
        self._local = threading.local()

        # Initialize the database schema
        self._init_db()

        # Start background cleanup
        if self.cleanup_interval > 0:
            self._start_cleanup_thread()

    # ── Connection management ──────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False,
                isolation_level=None,  # Auto-commit mode
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA cache_size=-8000")  # 8 MB cache
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Create the cache table and indexes if they don't exist."""
        conn = self._get_conn()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                key TEXT PRIMARY KEY NOT NULL,
                value BLOB NOT NULL,
                ttl REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0,
                tags TEXT NOT NULL DEFAULT '[]',
                size_bytes INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_expires
            ON {self.table_name}(expires_at)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tags
            ON {self.table_name}(tags)
        """)
        conn.execute("PRAGMA schema_version")

    # ── Core operations ────────────────────────────────

    async def get(self, key: str) -> CacheEntry | None:
        conn = self._get_conn()
        cursor = conn.execute(
            f"SELECT * FROM {self.table_name} WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()

        if row is None:
            with self._lock:
                self._misses += 1
            return None

        # Check expiration
        now = time.time()
        expires_at = row["expires_at"]
        if 0 < expires_at < now:
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE key = ?",
                (key,),
            )
            with self._lock:
                self._misses += 1
            return None

        # Deserialize
        try:
            entry = pickle.loads(row["value"])
            with self._lock:
                self._hits += 1
            return entry
        except (pickle.PickleError, EOFError) as exc:
            logger.warning("Cache deserialization error for %s: %s", key, exc)
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE key = ?",
                (key,),
            )
            with self._lock:
                self._misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float = 0.0,
        tags: set[str] | None = None,
    ) -> None:
        now = time.time()
        expires_at = now + ttl if ttl > 0 else 0
        tags_json = json.dumps(list(tags or []))

        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            created_at=now,
            tags=tags or set(),
        )
        serialized = pickle.dumps(entry)
        size = len(serialized)

        conn = self._get_conn()
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.table_name}
                (key, value, ttl, created_at, expires_at, tags, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (key, serialized, ttl, now, expires_at, tags_json, size),
        )

    async def delete(self, key: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            f"DELETE FROM {self.table_name} WHERE key = ?",
            (key,),
        )
        return cursor.rowcount > 0

    async def clear(self) -> int:
        conn = self._get_conn()
        cursor = conn.execute(f"DELETE FROM {self.table_name}")
        return cursor.rowcount

    async def has(self, key: str) -> bool:
        return (await self.get(key)) is not None

    async def stats(self) -> CacheStats:
        conn = self._get_conn()
        cursor = conn.execute(
            f"SELECT COUNT(*) as entries, "
            f"COALESCE(SUM(size_bytes), 0) as total_bytes "
            f"FROM {self.table_name}",
        )
        row = cursor.fetchone()

        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=row["entries"] if row else 0,
                size_bytes=row["total_bytes"] if row else 0,
            )

    async def get_many(self, keys: list[str]) -> dict[str, CacheEntry | None]:
        """Batch get using SQL IN clause."""
        if not keys:
            return {}

        placeholders = ",".join("?" * len(keys))
        conn = self._get_conn()
        cursor = conn.execute(
            f"SELECT * FROM {self.table_name} WHERE key IN ({placeholders})",
            keys,
        )
        rows = cursor.fetchall()
        found = {row["key"]: row for row in rows}

        results: dict[str, CacheEntry | None] = {}
        now = time.time()

        for key in keys:
            row = found.get(key)
            if row is None:
                with self._lock:
                    self._misses += 1
                results[key] = None
                continue

            expires_at = row["expires_at"]
            if 0 < expires_at < now:
                conn.execute(
                    f"DELETE FROM {self.table_name} WHERE key = ?",
                    (key,),
                )
                with self._lock:
                    self._misses += 1
                results[key] = None
                continue

            try:
                results[key] = pickle.loads(row["value"])
                with self._lock:
                    self._hits += 1
            except (pickle.PickleError, EOFError):
                with self._lock:
                    self._misses += 1
                results[key] = None

        return results

    async def set_many(
        self,
        entries: dict[str, Any],
        ttl: float = 0.0,
        tags: set[str] | None = None,
    ) -> None:
        now = time.time()
        expires_at = now + ttl if ttl > 0 else 0
        tags_json = json.dumps(list(tags or []))
        conn = self._get_conn()

        conn.execute("BEGIN")
        try:
            for key, value in entries.items():
                entry = CacheEntry(
                    key=key, value=value, ttl=ttl,
                    created_at=now, tags=tags or set(),
                )
                serialized = pickle.dumps(entry)
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.table_name}
                        (key, value, ttl, created_at, expires_at, tags, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key, serialized, ttl, now, expires_at, tags_json, len(serialized)),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def delete_by_tag(self, tag: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            f"DELETE FROM {self.table_name} WHERE tags LIKE ?",
            (f"%{tag}%",),
        )
        deleted = cursor.rowcount
        logger.info("Deleted %d cache entries with tag '%s'", deleted, tag)
        return deleted

    async def cleanup(self) -> CacheStats:
        """Remove expired entries."""
        conn = self._get_conn()
        cursor = conn.execute(
            f"DELETE FROM {self.table_name} "
            f"WHERE expires_at > 0 AND expires_at < ?",
            (time.time(),),
        )
        cleared = cursor.rowcount

        with self._lock:
            self._cleanup_count += 1
            if self.vacuum_after_n_cleanups > 0 and \
               self._cleanup_count % self.vacuum_after_n_cleanups == 0:
                try:
                    conn.execute("PRAGMA vacuum")
                    logger.debug("Vacuumed cache database")
                except Exception as exc:
                    logger.warning("Vacuum failed: %s", exc)

        logger.info("Cache cleanup removed %d expired entries", cleared)

        return CacheStats(
            expired_cleared=cleared,
        )

    def _start_cleanup_thread(self) -> None:
        """Start a daemon thread for periodic cleanup."""
        thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="cache-cleanup",
        )
        thread.start()
        logger.debug(
            "Cache cleanup thread started (interval=%ds)",
            self.cleanup_interval,
        )

    def _cleanup_loop(self) -> None:
        """Periodic cleanup loop running in a daemon thread."""
        while True:
            time.sleep(self.cleanup_interval)
            try:
                # Reset connection for the cleanup thread
                if hasattr(self._local, "conn"):
                    del self._local.conn
                import asyncio
                asyncio.run(self.cleanup())
            except Exception as exc:
                logger.warning("Cache cleanup error: %s", exc)

    def close(self) -> None:
        """Close all connections and release resources."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# ═══════════════════════════════════════════════════════════
#  Redis backend (optional — requires redis-py)
# ═══════════════════════════════════════════════════════════


class RedisBackend(CacheBackend):
    """Redis-based cache backend for distributed deployments.

    Requires the ``redis`` package: ``pip install redis``

    Stores values as pickled bytes with Redis TTL and SET/GET.
    Tags are stored as Redis sets for efficient group invalidation.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (default ``redis://localhost:6379/0``).
    key_prefix : str
        Prefix for all cache keys (default ``cache:``).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "cache:",
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix

        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

        # Lazy import of redis to keep it optional
        try:
            import redis.asyncio as redis_asyncio
        except ImportError:
            raise ImportError(
                "RedisBackend requires the 'redis' package. "
                "Install it with: pip install redis"
            )

        self._redis = redis_asyncio.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=10,
        )

    def _mk_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def _tag_key(self, tag: str) -> str:
        return f"{self.key_prefix}tag:{tag}"

    async def get(self, key: str) -> CacheEntry | None:
        full_key = self._mk_key(key)
        data = await self._redis.get(full_key)

        if data is None:
            with self._lock:
                self._misses += 1
            return None

        try:
            entry = CacheEntry.from_bytes(data)
            # Redis handles TTL natively, but check our embedded TTL too
            if entry.is_expired:
                await self._redis.delete(full_key)
                with self._lock:
                    self._misses += 1
                return None
            with self._lock:
                self._hits += 1
            return entry
        except Exception:
            with self._lock:
                self._misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: float = 0.0,
        tags: set[str] | None = None,
    ) -> None:
        full_key = self._mk_key(key)
        tags = tags or set()

        entry = CacheEntry(
            key=key, value=value, ttl=ttl,
            tags=tags,
        )
        data = entry.to_bytes()

        # Store with Redis TTL (convert seconds to ms for redis-py >= 5.0)
        ttl_ms = int(ttl * 1000) if ttl > 0 else None
        if ttl_ms:
            await self._redis.psetex(full_key, ttl_ms, data)
        else:
            await self._redis.set(full_key, data)

        # Update tag sets
        if tags:
            pipeline = self._redis.pipeline()
            tag_key = self._tag_key
            for tag in tags:
                pipeline.sadd(tag_key(tag), key)
                # Tag sets expire after the longest TTL to clean up
                if ttl > 0:
                    pipeline.pexpire(tag_key(tag), ttl_ms or 0)
            await pipeline.execute()

    async def delete(self, key: str) -> bool:
        full_key = self._mk_key(key)
        result = await self._redis.delete(full_key)
        return result > 0

    async def clear(self) -> int:
        # Delete all keys with our prefix
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await self._redis.scan(
                cursor, match=f"{self.key_prefix}*", count=100,
            )
            if keys:
                deleted += await self._redis.delete(*keys)
            if cursor == 0:
                break
        return deleted

    async def has(self, key: str) -> bool:
        full_key = self._mk_key(key)
        exists = await self._redis.exists(full_key)
        return exists > 0

    async def stats(self) -> CacheStats:
        # Count keys with our prefix (approximate)
        cursor = 0
        count = 0
        total_size = 0
        while True:
            cursor, keys = await self._redis.scan(
                cursor, match=f"{self.key_prefix}*", count=1000,
            )
            if keys:
                count += len(keys)
                # Get memory for each key (sample if too many)
                for key in keys[:100]:  # Limit sampling
                    try:
                        total_size += await self._redis.memory_usage(key) or 0
                    except Exception:
                        pass
            if cursor == 0:
                break

        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=count,
                size_bytes=total_size,
            )

    async def get_many(self, keys: list[str]) -> dict[str, CacheEntry | None]:
        if not keys:
            return {}

        full_keys = [self._mk_key(k) for k in keys]
        data_list = await self._redis.mget(*full_keys)

        results: dict[str, CacheEntry | None] = {}
        for key, data in zip(keys, data_list):
            if data is None:
                with self._lock:
                    self._misses += 1
                results[key] = None
            else:
                try:
                    entry = CacheEntry.from_bytes(data)
                    if entry.is_expired:
                        await self._redis.delete(self._mk_key(key))
                        results[key] = None
                        with self._lock:
                            self._misses += 1
                    else:
                        results[key] = entry
                        with self._lock:
                            self._hits += 1
                except Exception:
                    results[key] = None
                    with self._lock:
                        self._misses += 1
        return results

    async def delete_by_tag(self, tag: str) -> int:
        tag_key = self._tag_key(tag)
        # Get all keys with this tag
        keys = await self._redis.smembers(tag_key)
        if not keys:
            return 0

        full_keys = [self._mk_key(k.decode() if isinstance(k, bytes) else k) for k in keys]
        full_keys.append(tag_key)  # Also delete the tag set itself

        deleted = await self._redis.delete(*full_keys)
        logger.info("Deleted %d cache entries with tag '%s'", deleted, tag)
        return deleted

    async def cleanup(self) -> CacheStats:
        # Redis handles TTL natively — no explicit cleanup needed.
        # This method exists for interface compatibility.
        return CacheStats()

    async def close(self) -> None:
        await self._redis.aclose()
