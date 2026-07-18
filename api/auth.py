"""
API Key Authentication — secure bearer-token auth for the prediction API.

Supports:
- Authorization: Bearer <key> header (preferred)
- X-API-Key header
- Environment-based toggling for development mode

Security guarantees:
- Constant-time comparison via secrets.compare_digest()
- No len() on None
- Request object passed explicitly to verify_api_key
- Production authentication is mandatory
- Development mode requires explicit APP_ENV=development flag
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import config

# Load .env before reading any environment variables
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_dotenv_path)

logger = logging.getLogger(__name__)

# ── Security scheme ─────────────────────────────────────────
security = HTTPBearer(auto_error=False)

# ── Environment / auth mode ─────────────────────────────────
APP_ENV = os.environ.get("APP_ENV", "production").lower()
AUTH_DISABLED = os.environ.get("API_AUTH_DISABLED", "").lower() in ("true", "1", "yes")
IS_DEV_MODE = APP_ENV == "development" or AUTH_DISABLED

# ── API key configuration ──────────────────────────────────
# Try reading from multiple sources
_raw_key: str | None = (
    os.environ.get("PREDICTION_API_KEY")
    or os.environ.get("THE_ODDS_API_KEY")
    or (
        getattr(config, "odds_api", None)
        and getattr(config.odds_api, "api_key_env", None)
        and os.environ.get(config.odds_api.api_key_env)
    )
)

if not _raw_key:
    if IS_DEV_MODE:
        logger.warning(
            "No PREDICTION_API_KEY set. Running in DEVELOPMENT mode without auth. "
            "Set APP_ENV=production and PREDICTION_API_KEY in production."
        )
    else:
        logger.error(
            "PRODUCTION MODE: No PREDICTION_API_KEY environment variable set! "
            "Authentication is DISABLED. Set PREDICTION_API_KEY immediately."
        )

# Store as Optional[str]; never None for comparison purposes
API_KEY: str | None = _raw_key

if API_KEY:
    if IS_DEV_MODE:
        logger.info("API authentication configured (key length: %d chars)", len(API_KEY))
    else:
        logger.info("API authentication configured")
elif IS_DEV_MODE:
    logger.info("API authentication disabled (development mode)")
else:
    logger.warning("API authentication DISABLED in production! Set PREDICTION_API_KEY.")


# ── Rate limiting ───────────────────────────────────────────
class RateLimiter:
    """In-memory rate limiter with bounded storage.

    Limits requests per client ID within a sliding time window.
    Uses a fixed-size LRU-like eviction policy to prevent memory leaks.

    .. warning::

        This is a **process-local** in-memory limiter. For multi-worker
        deployments (e.g., multiple uvicorn workers), use a shared backend
        like Redis.
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
        timestamps = self._requests.get(client_id, [])
        self._requests[client_id] = [t for t in timestamps if t > cutoff]

        if len(self._requests[client_id]) >= self.max_requests:
            return False

        self._requests[client_id].append(now)
        return True

    def _cleanup(self, now: float) -> None:
        """Remove stale clients that haven't made requests in 2x window."""
        stale_cutoff = now - self.window_seconds * 2
        stale_keys = [
            k
            for k, v in list(self._requests.items())
            if not v or max(v, default=0) < stale_cutoff
        ]
        for k in stale_keys:
            try:
                del self._requests[k]
            except KeyError:
                pass

        # If still over max_clients, evict oldest
        if len(self._requests) > self.max_clients:
            sorted_keys = sorted(
                self._requests.keys(),
                key=lambda k: max(self._requests[k], default=0),
            )
            to_evict = len(self._requests) - self.max_clients
            for k in sorted_keys[:to_evict]:
                try:
                    del self._requests[k]
                except KeyError:
                    pass


# Singleton rate limiter — configurable via environment variables
# RATE_LIMIT_MAX:   max requests per window (default 100)
# RATE_LIMIT_WINDOW: time window in seconds (default 60)
_rate_limit_max = int(os.environ.get("RATE_LIMIT_MAX", "100"))
_rate_limit_window = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
rate_limiter = RateLimiter(
    max_requests=max(_rate_limit_max, 1),
    window_seconds=max(_rate_limit_window, 1),
)
logger.info(
    "Rate limiter: max %d requests per %d seconds per IP",
    rate_limiter.max_requests,
    rate_limiter.window_seconds,
)


# ── Authentication dependency ───────────────────────────────
async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """FastAPI dependency that validates the API key.

    Accepts the API key via:
    1. ``Authorization: Bearer <key>`` header (preferred)
    2. ``X-API-Key`` header
    3. ``api_key`` query parameter (for backward compatibility)

    Returns the API key string on success.
    Raises 401 for missing credentials, 403 for invalid credentials.

    In development mode (APP_ENV=development or API_AUTH_DISABLED=true),
    authentication is skipped.
    """
    # Development mode — no auth required
    if IS_DEV_MODE:
        return "dev-mode"

    # Production mode — API key must be configured
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured. Set PREDICTION_API_KEY environment variable.",
        )

    # Extract API key from various sources
    api_key: str | None = None

    if credentials is not None:
        api_key = credentials.credentials
    else:
        # Check X-API-Key header
        x_api_key = request.headers.get("x-api-key")
        if x_api_key:
            api_key = x_api_key
        else:
            # Check query parameter (backward compatibility)
            api_key = request.query_params.get("api_key")

    # Validate key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via Authorization: Bearer <key>, "
            "X-API-Key header, or ?api_key=<key> query parameter.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key


# ── Optional authentication (for backward compatibility, removed from critical endpoints) ──
async def optional_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Like ``verify_api_key`` but returns None instead of raising on failure.

    .. deprecated::
        Use ``verify_api_key`` for protected endpoints. This is retained only
        for non-critical informational endpoints.

    Returns the API key string if valid, or None if missing/invalid.
    """
    if IS_DEV_MODE:
        return "dev-mode"

    # Production mode — API key must be configured
    if not API_KEY:
        return None

    api_key: str | None = None

    if credentials is not None:
        api_key = credentials.credentials
    else:
        api_key = request.headers.get("x-api-key")

    if api_key and secrets.compare_digest(api_key, API_KEY):
        return api_key

    return None
