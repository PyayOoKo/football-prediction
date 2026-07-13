"""
RobotsChecker — respects FBref's robots.txt and enforces polite access.

Features
--------
- Fetches and caches robots.txt content (with TTL)
- Checks URL paths against disallowed rules
- Parses Crawl-delay directives
- Enforces minimum request intervals
- Signals when scraping is not permitted at all

FBref robots.txt notes
----------------------
FBref disallows several paths (/my/, /feedback/, /linker/, /fbref/).
The main /en/comps/ and /en/squads/ paths are not disallowed, but
automated access is technically against their terms of service.
This module ensures polite, rate-limited access.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ── Default minimum delay (seconds) between requests ─────
# FBref doesn't specify a Crawl-delay, so we use a polite default.
DEFAULT_DELAY = 4.0

# Paths that FBref disallows in robots.txt (as of 2026)
KNOWN_DISALLOWED_PREFIXES: tuple[str, ...] = (
    "/fbref/",
    "/feedback/",
    "/linker/",
    "/my/",
    "/news/",
    "/players/",
    "/sharelink/",
    "/sharetools/",
    "/search/",
)


@dataclass
class RobotsPolicy:
    """Parsed robots.txt policy for FBref.

    Attributes
    ----------
    disallowed_prefixes : list[str]
        URL path prefixes that are disallowed.
    crawl_delay : float
        Minimum seconds between requests.
    is_allowed : bool
        Whether scraping FBref is permitted at all (per robots.txt).
    fetched_at : float
        Timestamp when the policy was fetched.
    ttl : float
        Seconds before the policy should be re-fetched (default 3600).
    """

    disallowed_prefixes: list[str] = field(default_factory=list)
    crawl_delay: float = DEFAULT_DELAY
    is_allowed: bool = True
    fetched_at: float = 0.0
    ttl: float = 3600.0

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.fetched_at) > self.ttl

    def is_path_allowed(self, path: str) -> bool:
        """Check if a URL path is allowed under this policy.

        Parameters
        ----------
        path : str
            URL path (e.g. ``/en/comps/9/Premier-League-Stats``).

        Returns
        -------
        bool
            True if the path is allowed.
        """
        if not self.is_allowed:
            return False
        for prefix in self.disallowed_prefixes:
            if path.startswith(prefix):
                return False
        return True


class RobotsChecker:
    """Checks and caches robots.txt policies.

    Parameters
    ----------
    base_url : str
        Site base URL (default ``https://fbref.com``).
    user_agent : str
        Bot user-agent name (default ``FootballPredictionBot/1.0``).
    default_delay : float
        Default delay if no Crawl-delay is found (default 4.0s).
    """

    def __init__(
        self,
        base_url: str = "https://fbref.com",
        user_agent: str = "FootballPredictionBot/1.0",
        default_delay: float = DEFAULT_DELAY,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.default_delay = default_delay

        self._policy: RobotsPolicy | None = None
        self._last_request_time: float = 0.0
        self._client: httpx.Client | None = None

    @property
    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/plain,*/*",
                },
                follow_redirects=True,
                timeout=10.0,
            )
        return self._client

    def get_policy(self) -> RobotsPolicy:
        """Fetch and parse robots.txt, or return cached policy.

        Returns
        -------
        RobotsPolicy
            Current policy with disallowed paths and delay.
        """
        if self._policy is not None and not self._policy.is_stale:
            return self._policy

        policy = RobotsPolicy(fetched_at=time.time(), ttl=self.default_delay * 100)

        try:
            url = f"{self.base_url}/robots.txt"
            resp = self._http.get(url)
            resp.raise_for_status()

            policy = self._parse_robots_txt(resp.text)
            policy.fetched_at = time.time()

            logger.info(
                "Robots.txt loaded: %d disallowed prefixes, delay=%.1fs, allowed=%s",
                len(policy.disallowed_prefixes),
                policy.crawl_delay,
                policy.is_allowed,
            )

        except Exception as exc:
            logger.warning(
                "Could not fetch robots.txt (%s) — using default polite policy",
                exc,
            )

        self._policy = policy
        return policy

    def check_path(self, url: str | None = None, path: str | None = None) -> bool:
        """Check if a URL is allowed by robots.txt.

        Parameters
        ----------
        url : str, optional
            Full URL to check.
        path : str, optional
            URL path to check (alternative to url).

        Returns
        -------
        bool
            True if the path is allowed.
        """
        policy = self.get_policy()

        if not policy.is_allowed:
            return False

        if path is None and url is not None:
            parsed = urlparse(url)
            path = parsed.path or "/"

        if path is None:
            return True

        return policy.is_path_allowed(path)

    def wait_if_needed(self) -> float:
        """Block until the minimum delay since the last request has passed.

        Returns
        -------
        float
            Actual seconds waited.
        """
        policy = self.get_policy()
        delay = max(policy.crawl_delay, self.default_delay)

        elapsed = time.time() - self._last_request_time
        if elapsed < delay:
            wait = delay - elapsed
            time.sleep(wait)
            return wait
        return 0.0

    def record_request(self) -> None:
        """Record that a request was made (for rate limiting)."""
        self._last_request_time = time.time()

    @staticmethod
    def _parse_robots_txt(text: str) -> RobotsPolicy:
        """Parse raw robots.txt content into a policy.

        Handles User-agent matching, Disallow lines, and Crawl-delay.
        """
        policy = RobotsPolicy(
            crawl_delay=DEFAULT_DELAY,
            fetched_at=time.time(),
        )

        current_agent = "*"
        in_relevant_section = True
        found_specific = False

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse User-agent line
            ua_match = re.match(r"^User-agent:\s*(.+)$", line, re.IGNORECASE)
            if ua_match:
                current_agent = ua_match.group(1).strip()
                in_relevant_section = (
                    current_agent == "*"
                    or "FootballPredictionBot" in current_agent
                    or "bot" in current_agent.lower()
                )
                if current_agent != "*" and not found_specific:
                    found_specific = True
                continue

            if not in_relevant_section:
                continue

            # Parse Disallow line
            dis_match = re.match(r"^Disallow:\s*(.*)$", line, re.IGNORECASE)
            if dis_match:
                path = dis_match.group(1).strip()
                if path:
                    policy.disallowed_prefixes.append(path)
                continue

            # Parse Crawl-delay line
            cd_match = re.match(r"^Crawl-delay:\s*(\d+\.?\d*)$", line, re.IGNORECASE)
            if cd_match:
                policy.crawl_delay = float(cd_match.group(1))
                continue

            # Parse Allow line (exceptions)
            allow_match = re.match(r"^Allow:\s*(.*)$", line, re.IGNORECASE)
            if allow_match:
                allowed_path = allow_match.group(1).strip()
                # Remove any disallowed prefixes that start with the allowed path
                policy.disallowed_prefixes = [
                    p for p in policy.disallowed_prefixes
                    if not p.startswith(allowed_path)
                ]
                continue

        # Fallback: add known FBref disallowed paths if none found
        if not policy.disallowed_prefixes:
            logger.info("No disallow rules found in robots.txt — using known FBref paths")
            policy.disallowed_prefixes.extend(KNOWN_DISALLOWED_PREFIXES)

        return policy

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> RobotsChecker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
