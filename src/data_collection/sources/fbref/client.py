"""
FBrefClient — async HTTP client with caching, retry, and rate limiting.

Wraps httpx to provide:
- Polite rate limiting via RobotsChecker
- Exponential backoff retry with jitter
- LRU response caching with configurable TTL
- Concurrent page fetching with asyncio
- Session reuse with connection pooling
- Request timing and logging

Usage
-----
::

    from src.data_collection.sources.fbref import FBrefClient

    async with FBrefClient() as client:
        html = await client.get("/en/comps/9/Premier-League-Stats")
        # html is cached for subsequent requests within TTL
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import httpx

from src.data_collection.sources.fbref.robots import RobotsChecker
from src.etl.extract import RetryWithBackoff

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────

FBREF_BASE = "https://fbref.com"
DEFAULT_CACHE_DIR = "data/cache/fbref"
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_CONCURRENCY = 2  # Max concurrent requests (polite)
DEFAULT_REQUEST_TIMEOUT = 30.0

# User-agent used for all FBref requests
FBREF_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class CachedResponse:
    """A cached HTTP response."""

    url: str
    text: str
    status_code: int
    headers: dict[str, str]
    cached_at: float
    ttl: float

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.cached_at) > self.ttl

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> CachedResponse:
        return pickle.loads(data)


class FBrefClient:
    """Async HTTP client for FBref with built-in politeness.

    Parameters
    ----------
    base_url : str
        FBref base URL (default ``https://fbref.com``).
    cache_dir : str | Path
        Directory for disk cache (default ``data/cache/fbref``).
    cache_ttl : float
        Cache TTL in seconds (default 3600 = 1 hour).
    max_concurrent : int
        Maximum concurrent requests (default 2 — polite).
    request_timeout : float
        HTTP request timeout in seconds (default 30).
    respect_robots : bool
        Whether to check robots.txt before each request (default True).
    user_agent : str
        User-Agent header value (defaults to Chrome UA).
    """

    def __init__(
        self,
        base_url: str = FBREF_BASE,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        max_concurrent: int = DEFAULT_CONCURRENCY,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        respect_robots: bool = True,
        user_agent: str = FBREF_USER_AGENT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout
        self.respect_robots = respect_robots
        self.user_agent = user_agent

        # Rate limiting
        self._robots = RobotsChecker(
            base_url=base_url,
            user_agent="FootballPredictionBot/1.0",
        )
        self._retry = RetryWithBackoff(
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Cache: memory (dict) + disk
        self._memory_cache: dict[str, CachedResponse] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0

        # HTTP client
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

        # Ensure cache dir exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Properties ─────────────────────────────────────

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
                timeout=self.request_timeout,
                limits=httpx.Limits(
                    max_connections=self.max_concurrent + 2,
                    max_keepalive_connections=self.max_concurrent,
                ),
            )
        return self._client

    @property
    def sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
                timeout=self.request_timeout,
            )
        return self._sync_client

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": self._total_requests,
            "hit_ratio": round(
                self._cache_hits / max(self._total_requests, 1) * 100, 1
            ),
        }

    # ── Async API ──────────────────────────────────────

    async def get(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> str:
        """Fetch a page asynchronously with caching, rate limiting, and retry.

        Parameters
        ----------
        url : str
            Full URL or path (e.g. ``/en/comps/9/...``).
        force_refresh : bool
            Bypass cache and re-fetch.

        Returns
        -------
        str
            HTML text content.
        """
        full_url = self._resolve_url(url)
        cache_key = self._cache_key(full_url)

        self._total_requests += 1

        # Check cache
        if not force_refresh:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached.text

        self._cache_misses += 1

        # Check robots.txt
        if self.respect_robots:
            parsed = urlparse(full_url)
            if not self._robots.check_path(path=parsed.path):
                logger.warning("robots.txt disallows: %s", parsed.path)
                raise PermissionError(
                    f"robots.txt disallows access to {parsed.path}"
                )

        # Rate limit: wait if needed
        waited = self._robots.wait_if_needed()

        # Fetch with retry (synchronous retry wrapper)
        async with self._semaphore:
            try:
                # Use a thread pool to run the sync retry logic
                response = await asyncio.to_thread(
                    self._retry.execute,
                    self.sync_client.get,
                    full_url,
                )
                html = response.text
                status = response.status_code

                self._robots.record_request()

                if status != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {status} for {full_url}",
                        request=response.request,
                        response=response,
                    )

                # Cache the result
                cached_resp = CachedResponse(
                    url=full_url,
                    text=html,
                    status_code=status,
                    headers=dict(response.headers),
                    cached_at=time.time(),
                    ttl=self.cache_ttl,
                )
                self._set_cache(cache_key, cached_resp)

                if waited > 0:
                    logger.debug("Waited %.1fs before fetching %s", waited, full_url)

                return html

            except Exception as exc:
                logger.error("Failed to fetch %s: %s", full_url, exc)
                raise

    async def get_json(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch a URL and parse as JSON."""
        html = await self.get(url, force_refresh)
        return json.loads(html)

    async def get_multiple(
        self,
        urls: list[str],
        force_refresh: bool = False,
    ) -> list[str]:
        """Fetch multiple pages concurrently.

        Parameters
        ----------
        urls : list[str]
            URLs to fetch.
        force_refresh : bool
            Bypass cache.

        Returns
        -------
        list[str]
            HTML responses in the same order as input URLs.
        """
        tasks = [self.get(url, force_refresh) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle errors gracefully
        output: list[str] = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error("Failed to fetch %s: %s", url, result)
                output.append("")
            else:
                output.append(result)
        return output

    # ── Sync API ───────────────────────────────────────

    def get_sync(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> str:
        """Synchronous version of ``get()``.

        Useful for scripts that don't want async.
        """
        full_url = self._resolve_url(url)
        cache_key = self._cache_key(full_url)

        self._total_requests += 1

        if not force_refresh:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached.text

        self._cache_misses += 1

        if self.respect_robots:
            parsed = urlparse(full_url)
            if not self._robots.check_path(path=parsed.path):
                raise PermissionError(
                    f"robots.txt disallows access to {parsed.path}"
                )

        self._robots.wait_if_needed()

        try:
            response = self._retry.execute(
                self.sync_client.get, full_url,
            )
            html = response.text
            self._robots.record_request()

            cached_resp = CachedResponse(
                url=full_url,
                text=html,
                status_code=response.status_code,
                headers=dict(response.headers),
                cached_at=time.time(),
                ttl=self.cache_ttl,
            )
            self._set_cache(cache_key, cached_resp)

            return html

        except Exception as exc:
            logger.error("Failed to fetch %s: %s", full_url, exc)
            raise

    # ── Cache management ───────────────────────────────

    def clear_cache(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.cache"):
            f.unlink()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("FBref cache cleared")

    def _cache_key(self, url: str) -> str:
        """Generate a cache key from a URL."""
        return hashlib.sha256(url.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> CachedResponse | None:
        """Try memory cache first, then disk cache."""
        # Memory
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired:
                return entry
            del self._memory_cache[key]

        # Disk
        disk_path = self.cache_dir / f"{key}.cache"
        if disk_path.exists():
            try:
                data = disk_path.read_bytes()
                entry = CachedResponse.from_bytes(data)
                if not entry.is_expired:
                    # Promote to memory
                    self._memory_cache[key] = entry
                    return entry
                disk_path.unlink()
            except Exception as exc:
                logger.debug("Cache read error: %s", exc)

        return None

    def _set_cache(self, key: str, entry: CachedResponse) -> None:
        """Store in memory cache and optionally disk."""
        self._memory_cache[key] = entry

        # Write to disk periodically (sample based on URL hash)
        if int(key[:8], 16) % 100 < 20:  # 20% sample to reduce disk I/O
            disk_path = self.cache_dir / f"{key}.cache"
            try:
                disk_path.write_bytes(entry.to_bytes())
            except Exception as exc:
                logger.debug("Cache write error: %s", exc)

    def _resolve_url(self, url: str) -> str:
        """Resolve relative paths to absolute FBref URLs."""
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return urljoin(self.base_url + "/", url.lstrip("/"))

    # ── Lifecycle ──────────────────────────────────────

    async def close(self) -> None:
        """Close all connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        self._robots.close()

    async def __aenter__(self) -> FBrefClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
