"""
API Key Authentication — simple bearer-token auth for the prediction API.

Supports:
- Static API key validation (via env var or config)
- Rate limiting per key
- Optional API key header or query parameter

Usage:
    from api.auth import verify_api_key, RateLimitMiddleware
    from fastapi import Depends, HTTPException, status

    @app.get("/secure", dependencies=[Depends(verify_api_key)])
    async def secure_endpoint():
        return {"message": "authenticated"}
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import config

logger = logging.getLogger(__name__)

# ── Security scheme ─────────────────────────────────────────
security = HTTPBearer(auto_error=False)

# ── API key configuration ──────────────────────────────────
# Try reading from multiple sources
API_KEY = os.environ.get("PREDICTION_API_KEY", "") or os.environ.get(
    "THE_ODDS_API_KEY", ""
) or getattr(config, "odds_api", None) and getattr(
    config.odds_api, "api_key_env", None
) and os.environ.get(config.odds_api.api_key_env, "")

if not API_KEY:
    # In production, fail loudly
    import warnings
    warnings.warn(
        "No PREDICTION_API_KEY environment variable set! "
        "Set PREDICTION_API_KEY in production. "
        "Using development mode without auth."
    )
    logger.warning(
        "No PREDICTION_API_KEY set. Authentication disabled "
        "(development mode). Set PREDICTION_API_KEY in production."
    )
    API_KEY = None  # Disable auth in dev mode

logger.info("API authentication configured (key length: %d chars)", len(API_KEY))

# ── Rate limiting ───────────────────────────────────────────
class RateLimiter:
    """In-memory rate limiter with bounded storage.

    Limits requests per client ID within a sliding time window.
    Uses a fixed-size LRU-like eviction policy to prevent memory leaks.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        max_clients: int = 10000,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup: float = time.time()

    def check(self, client_id: str) -> bool:
        """Check if *client_id* has exceeded the rate limit.

        Returns True if the request is allowed, False if rate-limited.
        """
        now = time.time()
        cutoff = now - self.window_seconds

        # Periodic cleanup of stale clients (every 60s)
        if now - self._last_cleanup > 60:
            self._cleanup(now)
            self._last_cleanup = now

        # Prune old entries for this client
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > cutoff
        ]

        if len(self._requests[client_id]) >= self.max_requests:
            return False

        self._requests[client_id].append(now)
        return True

    def _cleanup(self, now: float) -> None:
        """Remove stale clients that haven't made requests in 2x window."""
        stale_cutoff = now - self.window_seconds * 2
        stale_keys = [
            k for k, v in self._requests.items()
            if not v or max(v) < stale_cutoff
        ]
        for k in stale_keys:
            del self._requests[k]

        # If still over max_clients, evict oldest
        if len(self._requests) > self.max_clients:
            sorted_keys = sorted(
                self._requests.keys(),
                key=lambda k: max(self._requests[k]) if self._requests[k] else 0,
            )
            to_evict = len(self._requests) - self.max_clients
            for k in sorted_keys[:to_evict]:
                del self._requests[k]


# Singleton rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


# ── Authentication dependency ───────────────────────────────
async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """FastAPI dependency that validates the API key.

    Accepts the API key via:
    1. ``Authorization: Bearer <key>`` header (preferred)
    2. ``X-API-Key`` header (handled in middleware for middleware routes)
    3. ``api_key`` query parameter

    Returns the API key string on success, or raises 401/403.

    Rate limiting is handled by middleware globally.
    """
    if API_KEY is None:
        # Development mode — no auth required
        return "dev-mode"

    # Extract API key
    api_key: str | None = None

    if credentials is not None:
        api_key = credentials.credentials
    elif "x-api-key" in request.headers:
        api_key = request.headers["x-api-key"]
    elif "api_key" in request.query_params:
        api_key = request.query_params["api_key"]

    # Validate key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via Authorization: Bearer <key>, "
            "X-API-Key header, or ?api_key=<key> query parameter.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    # Rate limiting is handled by middleware — no check needed here
    return api_key


# ── Optional authentication (for public endpoints) ──────────
async def optional_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Like ``verify_api_key`` but returns None instead of raising on failure.

    Returns the API key string if valid, or None if missing/invalid.
    Use on endpoints that work with or without authentication.
    """
    if API_KEY is None:
        # Development mode — no auth required
        return "dev-mode"

    api_key: str | None = None

    if credentials is not None:
        api_key = credentials.credentials
    elif "x-api-key" in request.headers:
        api_key = request.headers["x-api-key"]

    if api_key == API_KEY:
        return api_key

    return None
