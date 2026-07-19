"""
Security middleware for the API.
Includes rate limiting, input sanitization, and authentication.
"""
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional
import time
import re

# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting storage (in production, use Redis)
_rate_limit_store: Dict[str, list] = {}
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse."""
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old entries
        if client_ip in _rate_limit_store:
            _rate_limit_store[client_ip] = [
                t for t in _rate_limit_store[client_ip] 
                if current_time - t < RATE_LIMIT_WINDOW
            ]
        else:
            _rate_limit_store[client_ip] = []
        
        # Check rate limit
        if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later."
            )
        
        # Record request
        _rate_limit_store[client_ip].append(current_time)
        
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(
            RATE_LIMIT_REQUESTS - len(_rate_limit_store[client_ip])
        )
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

class SanitizationMiddleware(BaseHTTPMiddleware):
    """Input sanitization middleware to prevent injection attacks."""
    
    # Patterns that might indicate injection attempts
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>",  # XSS
        r"javascript:",    # XSS
        r"on\w+\s*=",      # Event handlers
        r"--\s*$",         # SQL comment
        r";\s*DROP\s+",    # SQL DROP
        r";\s*DELETE\s+",  # SQL DELETE
        r"\.\./",          # Path traversal
    ]
    
    async def dispatch(self, request: Request, call_next):
        # Only sanitize body for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body_str = body.decode('utf-8')
                    for pattern in self.DANGEROUS_PATTERNS:
                        if re.search(pattern, body_str, re.IGNORECASE):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Potentially malicious input detected"
                            )
            except UnicodeDecodeError:
                pass  # Binary data, skip sanitization
        
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

async def validate_api_key(request: Request):
    """Validate API key from header."""
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing"
        )
    
    # In production, validate against database or secrets manager
    # For now, check environment variable
    import os
    expected_key = os.getenv("API_KEY")
    
    if expected_key and api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return api_key
