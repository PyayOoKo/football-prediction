"""
UnderstatClient — fetches and extracts JSON data from Understat pages.

Understat embeds data in JavaScript variables inside HTML script tags:
    var teamsData = JSON.parse('{"team1": {...}}');
    var shotsData = JSON.parse('{"12345": [...]}');

This client:
1. Fetches the HTML page
2. Extracts the relevant <script> tag content
3. Parses the JavaScript variable assignment
4. Decodes Unicode-escaped strings in the JSON
5. Returns clean Python dictionaries

Supports caching, rate limiting, and retry via the client config.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from src.data_collection.sources.fbref.client import FBrefClient

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────

UNDERSTAT_BASE = "https://understat.com"

# JavaScript variable names containing the data we need
_VAR_TEAMS_DATA = r"var\s+teamsData\s*=\s*JSON\.parse\s*\(\s*'"
_VAR_SHOTS_DATA = r"var\s+shotsData\s*=\s*JSON\.parse\s*\(\s*'"
_VAR_PLAYERS_DATA = r"var\s+playersData\s*=\s*JSON\.parse\s*\(\s*'"
_VAR_DATES_DATA = r"var\s+datesData\s*=\s*JSON\.parse\s*\(\s*'"

# Regex patterns for each variable
_VAR_PATTERNS: dict[str, re.Pattern] = {
    "teamsData": re.compile(_VAR_TEAMS_DATA + r"(.*?)'\s*\)\s*;", re.DOTALL),
    "shotsData": re.compile(_VAR_SHOTS_DATA + r"(.*?)'\s*\)\s*;", re.DOTALL),
    "playersData": re.compile(_VAR_PLAYERS_DATA + r"(.*?)'\s*\)\s*;", re.DOTALL),
    "datesData": re.compile(_VAR_DATES_DATA + r"(.*?)'\s*\)\s*;", re.DOTALL),
}

# Headers to mimic a real browser (Understat is more lenient than FBref)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Default request delay (Understat is relatively tolerant, but be polite)
DEFAULT_DELAY = 1.5


class UnderstatClient:
    """HTTP client for fetching Understat pages and extracting embedded JSON.

    Parameters
    ----------
    base_url : str
        Understat base URL (default ``https://understat.com``).
    cache_dir : str
        Cache directory for responses (default ``data/cache/understat``).
    cache_ttl : float
        Cache TTL in seconds (default 3600).
    request_delay : float
        Minimum seconds between requests (default 1.5).
    request_timeout : float
        HTTP timeout in seconds (default 30).
    """

    def __init__(
        self,
        base_url: str = UNDERSTAT_BASE,
        cache_dir: str = "data/cache/understat",
        cache_ttl: float = 3600.0,
        request_delay: float = DEFAULT_DELAY,
        request_timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_delay = request_delay
        self.request_timeout = request_timeout

        # Reuse FBref's client for caching and retry (it's generic enough)
        self._http = FBrefClient(
            base_url=base_url,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
            max_concurrent=3,
            request_timeout=request_timeout,
            respect_robots=False,  # Understat doesn't block known paths
            user_agent=_HEADERS["User-Agent"],
        )

        self._last_request: float = 0.0

    # ── Public API ─────────────────────────────────────

    async def get_league_page(self, league: str, year: int) -> str:
        """Fetch a league page HTML.

        Parameters
        ----------
        league : str
            League code (e.g. ``EPL``, ``La_liga``).
        year : int
            Season starting year (e.g. ``2024``).

        Returns
        -------
        str
            HTML content.
        """
        url = f"{self.base_url}/league/{league}/{year}"
        return await self._http.get(url)

    async def get_match_page(self, match_id: int) -> str:
        """Fetch a match page HTML.

        Parameters
        ----------
        match_id : int
            Understat match identifier.

        Returns
        -------
        str
            HTML content.
        """
        url = f"{self.base_url}/match/{match_id}"
        return await self._http.get(url)

    async def get_team_page(self, team_name: str, year: int) -> str:
        """Fetch a team page HTML.

        Parameters
        ----------
        team_name : str
            Team name slug (e.g. ``Manchester_United``).
        year : int
            Season starting year.

        Returns
        -------
        str
            HTML content.
        """
        url = f"{self.base_url}/team/{team_name}/{year}"
        return await self._http.get(url)

    # ── JSON extraction ────────────────────────────────

    async def get_league_data(
        self, league: str, year: int,
    ) -> dict[str, Any]:
        """Extract teamsData JSON from a league page.

        Parameters
        ----------
        league : str
            League code.
        year : int
            Season starting year.

        Returns
        -------
        dict
            Parsed teamsData dictionary.
        """
        html = await self.get_league_page(league, year)
        return self._extract_json(html, "teamsData")

    async def get_match_data(
        self, match_id: int,
    ) -> dict[str, Any]:
        """Extract shotsData JSON from a match page.

        Parameters
        ----------
        match_id : int
            Understat match identifier.

        Returns
        -------
        dict
            Parsed shotsData dictionary.
        """
        html = await self.get_match_page(match_id)
        return self._extract_json(html, "shotsData")

    async def get_league_matches_data(
        self, league: str, year: int,
    ) -> dict[str, Any]:
        """Extract datesData JSON from a league page (match list).

        The datesData variable contains the match schedule/results
        for the league season.

        Parameters
        ----------
        league : str
            League code.
        year : int
            Season starting year.

        Returns
        -------
        dict
            Parsed datesData dictionary.
        """
        html = await self.get_league_page(league, year)
        return self._extract_json(html, "datesData")

    # ── Static extraction method ───────────────────────

    @staticmethod
    def _extract_json(html: str, var_name: str) -> dict[str, Any]:
        """Extract JSON data from a JavaScript variable in HTML.

        Works with Understat's format:
            var teamsData = JSON.parse('{...}');

        Parameters
        ----------
        html : str
            Page HTML content.
        var_name : str
            JavaScript variable name (teamsData, shotsData, etc.).

        Returns
        -------
        dict
            Parsed JSON data. Empty dict if not found.
        """
        pattern = _VAR_PATTERNS.get(var_name)
        if not pattern:
            logger.warning("Unknown variable name: %s", var_name)
            return {}

        match = pattern.search(html)
        if not match:
            logger.warning(
                "Could not find variable '%s' in page", var_name,
            )
            return {}

        raw = match.group(1).strip()

        # Try direct JSON parse first (fast path for clean JSON)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fallback: decode JavaScript/Unicode escapes then retry
        try:
            decoded = raw.encode().decode("unicode_escape", errors="replace")
            return json.loads(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error(
                "Could not parse %s from page: %s", var_name, exc,
            )
            return {}

    # ── Lifecycle ──────────────────────────────────────

    @property
    def cache_stats(self) -> dict[str, int]:
        return self._http.cache_stats

    def clear_cache(self) -> None:
        self._http.clear_cache()

    def wait_if_needed(self) -> float:
        """Block until the minimum delay since the last request has passed.

        Returns
        -------
        float
            Actual seconds waited.
        """
        elapsed = time.time() - self._last_request
        if elapsed < self.request_delay:
            wait = self.request_delay - elapsed
            time.sleep(wait)
            self._last_request = time.time()
            return wait
        self._last_request = time.time()
        return 0.0

    async def close(self) -> None:
        await self._http.close()

    async def __aenter__(self) -> UnderstatClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
