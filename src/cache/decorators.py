"""
Cache decorators — transparently cache function results.

Provides ``@cached`` for sync/async functions and ``@invalidate``
for post-operation cache clearing.

Usage
-----
::

    from src.cache import cached, invalidate

    @cached(ttl=3600)
    def get_team_stats(team_id: int) -> dict:
        \"\"\"Expensive computation, cached for 1 hour.\"\"\"
        return fetch_from_db(team_id)

    @cached(ttl=300, tags={"api", "odds"})
    async def get_live_odds(match_id: int) -> list[dict]:
        \"\"\"Expensive API call, cached for 5 minutes.\"\"\"
        return await odds_api.fetch(match_id)

    @invalidate(tags={"team", "elo"})
    def update_elo(team_id: int) -> None:
        \"\"\"After updating Elo, invalidate related cache entries.\"\"\"
        save_to_db(team_id)

    # Custom key function
    @cached(key_fn=lambda team_id, **kw: f\"team:{team_id}\")
    def get_team(team_id: int) -> dict:
        return fetch_team(team_id)

    # Use the default global cache
    from src.cache import CacheManager, SQLiteBackend
    cache = CacheManager(SQLiteBackend())
    cached.set_cache(cache)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

from src.cache.backend import CacheBackend
from src.cache.manager import CacheManager
from src.cache.models import CacheKey

logger = logging.getLogger(__name__)

# Default global cache manager (lazy-initialized)
_global_cache: CacheManager | None = None

P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def _get_global_cache() -> CacheManager:
    """Get or create the global cache instance.

    Creates a default SQLite-backed cache in ``data/cache/func_cache.db``
    if no cache has been configured.
    """
    global _global_cache
    if _global_cache is None:
        from src.cache.backend import SQLiteBackend
        _global_cache = CacheManager(
            SQLiteBackend("data/cache/func_cache.db", cleanup_interval=600),
            namespace="func",
            default_ttl=3600,
        )
    return _global_cache


def set_cache(cache: CacheManager) -> None:
    """Set the global cache instance used by ``@cached``.

    Call this once at application startup to use a custom backend::

        from src.cache import CacheManager, RedisBackend, set_cache

        cache = CacheManager(RedisBackend("redis://..."))
        set_cache(cache)
    """
    global _global_cache
    _global_cache = cache


# ── Default key function ───────────────────────────────

def _default_key_fn(func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from a function call.

    Format: ``{module}.{qualname}({arg1},{arg2},...)``
    """
    parts = [func.__module__, func.__qualname__]

    # Add positional args
    arg_strs = [repr(a) for a in args]
    # Add keyword args (sorted for determinism)
    kwarg_strs = [f"{k}={v!r}" for k, v in sorted(kwargs.items())]
    all_args = ",".join(arg_strs + kwarg_strs)
    parts.append(f"({all_args})")

    return ":".join(parts)


# ── @cached decorator ──────────────────────────────────

class cached:
    """Decorator that caches function results.

    Supports both sync and async functions.

    Parameters
    ----------
    ttl : float
        Time-to-live in seconds (default 3600).
    key_fn : callable, optional
        Custom key function: ``key_fn(func, *args, **kwargs) → str``.
        If not provided, generates a key from the function name + args.
    tags : set[str], optional
        Tags for group invalidation.
    cache : CacheManager, optional
        Cache manager to use. Defaults to the global cache.
    include_self : bool
        If True (default), includes ``self`` in the cache key for
        bound methods. Set to False to share cache across instances.

    Examples
    --------
    ::

        @cached(ttl=300)
        def expensive_query(user_id: int) -> dict:
            return db.fetch(user_id)

        @cached(tags={\"api\", \"odds\"})
        async def fetch_odds(match_id: int) -> list:
            return await api.get_odds(match_id)

        @cached(key_fn=lambda team_id, **kw: f\"team:{team_id}\")
        def get_team(team_id: int) -> dict:
            return load_team(team_id)
    """

    def __init__(
        self,
        ttl: float = 3600.0,
        key_fn: Callable[..., str] | None = None,
        tags: set[str] | None = None,
        cache: CacheManager | None = None,
        include_self: bool = True,
    ) -> None:
        self.ttl = ttl
        self.custom_key_fn = key_fn
        self.tags = tags or set()
        self.cache = cache
        self.include_self = include_self

    def __call__(self, func: F) -> F:
        """Apply the caching decorator to a function."""
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            return self._decorate_async(func)  # type: ignore[return-value]
        else:
            return self._decorate_sync(func)  # type: ignore[return-value]

    def _make_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for the given function call."""
        if self.custom_key_fn is not None:
            return self.custom_key_fn(*args, **kwargs)

        # Determine if this is a bound method
        bound_method = inspect.ismethod(func) or (
            hasattr(func, "__self__") and func.__self__ is not None
        )

        if bound_method and not self.include_self:
            # Skip self/cls
            return _default_key_fn(func, args[1:], kwargs)
        return _default_key_fn(func, args, kwargs)

    def _decorate_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap a sync function."""
        import threading

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = self.cache or _get_global_cache()
            key = self._make_key(func, args, kwargs) if not self.custom_key_fn else self.custom_key_fn(*args, **kwargs)

            # Try to get from cache using a new event loop in the current thread
            try:
                entry = _run_async(cache.get(key))
                if entry is not None:
                    return entry  # type: ignore[return-value]
            except Exception as exc:
                logger.debug("Cache get failed, computing: %s", exc)

            # Compute the value
            value = func(*args, **kwargs)

            # Cache it (fire-and-forget)
            try:
                _run_async(cache.set(key, value, ttl=self.ttl, tags=self.tags))
            except Exception as exc:
                logger.debug("Cache set failed: %s", exc)

            return value

        return wrapper

    def _decorate_async(
        self, func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        """Wrap an async function."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = self.cache or _get_global_cache()
            key = self._make_key(func, args, kwargs) if not self.custom_key_fn else self.custom_key_fn(*args, **kwargs)

            # Try cache first
            try:
                entry = await cache.get(key)
                if entry is not None:
                    return entry  # type: ignore[return-value]
            except Exception as exc:
                logger.debug("Async cache get failed, computing: %s", exc)

            # Compute
            value = await func(*args, **kwargs)

            # Cache
            try:
                await cache.set(key, value, ttl=self.ttl, tags=self.tags)
            except Exception as exc:
                logger.debug("Async cache set failed: %s", exc)

            return value

        return wrapper


# ── @invalidate decorator ──────────────────────────────

class invalidate:
    """Decorator that invalidates cache entries after a function runs.

    Useful for clearing cached data after update/delete operations.

    Parameters
    ----------
    keys : list[str], optional
        Specific cache keys to invalidate.
    tags : set[str], optional
        Tags whose entries should be invalidated.
    key_fn : callable, optional
        Function that returns the key(s) to invalidate.
        Receives the same args as the decorated function.
    cache : CacheManager, optional
        Cache manager to use. Defaults to the global cache.

    Examples
    --------
    ::

        @invalidate(tags={\"team\", \"elo\"})
        def update_elo(team_id: int, new_rating: float) -> None:
            save_to_db(team_id)

        @invalidate(key_fn=lambda team_id: f\"team:{team_id}\")
        def delete_team(team_id: int) -> None:
            db.delete_team(team_id)
    """

    def __init__(
        self,
        keys: list[str] | None = None,
        tags: set[str] | None = None,
        key_fn: Callable[..., str | list[str]] | None = None,
        cache: CacheManager | None = None,
    ) -> None:
        self.keys = keys or []
        self.tags = tags or set()
        self.key_fn = key_fn
        self.cache = cache

    def __call__(self, func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)
        if is_async:
            return self._decorate_async(func)  # type: ignore[return-value]
        else:
            return self._decorate_sync(func)  # type: ignore[return-value]

    def _decorate_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = self.cache or _get_global_cache()
            value = func(*args, **kwargs)

            self._do_invalidate(cache, *args, **kwargs)
            return value

        return wrapper

    def _decorate_async(
        self, func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            cache = self.cache or _get_global_cache()
            value = await func(*args, **kwargs)

            await self._do_invalidate_async(cache, *args, **kwargs)
            return value

        return wrapper

    def _do_invalidate(
        self, cache: CacheManager, *args: Any, **kwargs: Any,
    ) -> None:
        """Synchronous invalidation (fire-and-forget)."""
        for key in self.keys:
            _run_async(cache.invalidate(key))
        for tag in self.tags:
            _run_async(cache.invalidate_by_tag(tag))
        if self.key_fn:
            result = self.key_fn(*args, **kwargs)
            if isinstance(result, list):
                for key in result:
                    _run_async(cache.invalidate(key))
            else:
                _run_async(cache.invalidate(result))

    async def _do_invalidate_async(
        self, cache: CacheManager, *args: Any, **kwargs: Any,
    ) -> None:
        """Async invalidation."""
        for key in self.keys:
            await cache.invalidate(key)
        for tag in self.tags:
            await cache.invalidate_by_tag(tag)
        if self.key_fn:
            result = self.key_fn(*args, **kwargs)
            if isinstance(result, list):
                for key in result:
                    await cache.invalidate(key)
            else:
                await cache.invalidate(result)


# ── Helper: run async in sync context ─────────────────

def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Works both when an event loop is running (by using
    ``asyncio.run_coroutine_threadsafe``) and when it isn't
    (by using ``asyncio.run``).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — use asyncio.run
        return asyncio.run(coro)

    # Loop is running — schedule in a new thread
    import concurrent.futures
    import threading

    result: list[Any] = []
    error: Exception | None = None
    error_lock = threading.Lock()

    def run_in_thread() -> None:
        nonlocal error
        try:
            result.append(asyncio.run(coro))
        except Exception as e:
            with error_lock:
                error = e

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    return result[0]
