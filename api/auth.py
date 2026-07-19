"""
Authentication and rate limiting for the API.

Provides API-key-based authentication with support for:
- Bearer tokens (Authorization header)
- X-API-Key header
- Query parameter (?api_key=) [backward compatible]
- Optional authentication for public endpoints
- Rate limiting with configurable thresholds
- JWT-based authentication for user sessions
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt

# ── Environment helpers ─────────────────────────────────────

IS_DEV_MODE = os.getenv("APP_ENV", "development").lower() in (
    "development",
    "dev",
    "",
)


def _get_api_key() -> str | None:
    """Get the configured API key from environment."""
    key = os.getenv("PREDICTION_API_KEY", "")
    return key.strip() if key else None


# ── Rate Limiter ────────────────────────────────────────────


class _RateLimiter:
    """Simple in-memory rate limiter (per-IP sliding window).

    In production, replace with Redis-based rate limiting.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._store: dict[str, list[float]] = {}

    def check(self, client_ip: str) -> bool:
        """Check if the client IP is within the rate limit.

        Returns True if allowed, False if rate limited.
        """
        now = time.time()

        # Clean old entries for this IP
        timestamps = self._store.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < self.window_seconds]

        if len(timestamps) >= self.max_requests:
            return False

        timestamps.append(now)
        self._store[client_ip] = timestamps
        return True


# Default rate limiter instance
rate_limiter = _RateLimiter()


# ── Authentication Functions ───────────────────────────────


def _extract_credentials(request: Request) -> str | None:
    """Extract API key from request, checking multiple sources.

    Priority: Bearer token > X-API-Key header > query parameter.
    Headers are matched case-insensitively per HTTP spec.
    """
    # Build a case-insensitive header map
    headers_lower = {k.lower(): v for k, v in request.headers.items()}

    # 1. Authorization: Bearer <token>
    auth_header = headers_lower.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    # 2. X-API-Key header (any casing)
    api_key_header = headers_lower.get("x-api-key", "")
    if api_key_header:
        return api_key_header

    # 3. Query parameter (?api_key=)
    api_key_param = request.query_params.get("api_key", "")
    if api_key_param:
        return api_key_param

    return None


async def verify_api_key(
    request: Request,
    credentials: Any = None,
) -> str:
    """Verify API key from request or explicit credentials.

    Primary authentication dependency for protected endpoints.
    Returns the validated API key string on success.

    Raises:
        HTTPException(503): No API key configured (server misconfiguration)
        HTTPException(401): Missing credentials
        HTTPException(403): Invalid credentials
    """
    expected_key = _get_api_key()

    # Server misconfiguration: no key set
    if not expected_key:
        if IS_DEV_MODE:
            return "dev-mode"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "PREDICTION_API_KEY is not configured. "
                "Set the environment variable and restart the server."
            ),
        )

    # Extract credentials from request or explicit parameter
    token = None
    if credentials is not None:
        # credentials is an HTTPAuthorizationCredentials-like object
        if hasattr(credentials, "credentials"):
            token = credentials.credentials
        elif isinstance(credentials, str):
            token = credentials
    else:
        token = _extract_credentials(request)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_stripped = token.strip()
    if not token_stripped:
        # Key provided but is only whitespace — treat as invalid, not missing
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    if token_stripped != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return token


async def optional_auth(
    request: Request,
    credentials: Any = None,
) -> str | None:
    """Optional authentication — returns None instead of raising on failure.

    Use for endpoints where auth is nice-to-have but not required.
    """
    try:
        return await verify_api_key(request, credentials)
    except HTTPException:
        return None


# ═══════════════════════════════════════════════════════════
#  JWT-based Authentication (legacy / user sessions)
# ═══════════════════════════════════════════════════════════

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class TokenData:
    """Token payload data."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    roles: list = []

    def __init__(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        roles: Optional[list] = None,
    ):
        self.user_id = user_id
        self.username = username
        self.roles = roles or []


class TokenResponse:
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        expires_in: int,
        token_type: str = "bearer",
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_type = token_type
        self.expires_in = expires_in


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        roles: list = payload.get("roles", [])

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token claims",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return TokenData(user_id=user_id, username=username, roles=roles)

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash (placeholder - use bcrypt in production)."""
    return plain_password == hashed_password  # Placeholder


def hash_password(password: str) -> str:
    """Hash password (placeholder - use bcrypt in production)."""
    return password  # Placeholder
