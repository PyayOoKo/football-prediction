"""
Odds API — fetch live sports betting odds from The Odds API (the-odds-api.com).

Fetches upcoming match odds for soccer/football leagues and the World Cup,
caches results to stay within the free-tier rate limit (500 requests/month),
and returns odds in a format compatible with ``src.value_betting``.

Usage
-----
    from src.odds_api import OddsAPIClient

    client = OddsAPIClient()
    odds = client.get_match_odds("Brazil", "Norway")
    # Returns: {"home_odds": 1.83, "draw_odds": 3.70, "away_odds": 4.20}
    # Or None if the match isn't found / API unavailable

Environment
-----------
Requires ``THE_ODDS_API_KEY`` environment variable.
Get a free key at https://the-odds-api.com/ (500 requests/month free).

Caching
-------
Results are cached to ``data/external/odds_cache.json`` for ``CACHE_TTL`` seconds
(default 3600 = 1 hour) to avoid unnecessary API calls.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path


from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

API_BASE_URL = "https://api.the-odds-api.com/v4"
"""Base URL for The Odds API v4."""

DEFAULT_CACHE_TTL = 3600
"""Default cache TTL in seconds (1 hour)."""

DEFAULT_REGIONS = "us,uk,eu"
"""Default bookmaker regions: UK, Ireland, and European."""

CACHE_DIR = Path("data") / "external"
"""Directory for API response caches."""

CACHE_FILE = CACHE_DIR / "odds_cache.json"
"""File path for the odds cache."""

# ── Soccer league keys (from The Odds API) ─────────────
# These can be refreshed dynamically, but we include common ones
# for quick reference. The full list comes from /v4/sports.
SOCCER_LEAGUES: dict[str, str] = {
    "soccer_epl": "English Premier League",
    "soccer_la_liga": "Spanish La Liga",
    "soccer_bundesliga": "German Bundesliga",
    "soccer_serie_a": "Italian Serie A",
    "soccer_ligue_one": "French Ligue 1",
    "soccer_uefa_champs_league": "UEFA Champions League",
    "soccer_fifa_world_cup": "FIFA World Cup",
    "soccer_euro_cup": "UEFA Euro",
    "soccer_copa_america": "Copa America",
}

# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class MatchOdds:
    """Odds for a single match from a specific bookmaker."""

    home_team: str
    away_team: str
    match_date: str
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmaker: str
    sport_key: str = ""
    sport_title: str = ""


@dataclass
class OddsAPIConfig:
    """Configuration for the Odds API client."""

    api_key: str = ""
    regions: str = DEFAULT_REGIONS
    markets: str = "h2h"
    cache_ttl: int = DEFAULT_CACHE_TTL
    timeout: int = 15


# ═══════════════════════════════════════════════════════════
#  Main client
# ═══════════════════════════════════════════════════════════

class OddsAPIClient:
    """Client for fetching live odds from The Odds API.

    Parameters
    ----------
    api_key : str, optional
        API key. If not provided, reads from ``THE_ODDS_API_KEY`` env var.
    regions : str, optional
        Bookmaker regions to query (default ``"uk,ie,eu"``).
    markets : str, optional
        Markets to fetch (default ``"h2h"``).
    cache_ttl : int, optional
        Cache TTL in seconds (default 3600 = 1 hour).
    timeout : int, optional
        HTTP request timeout in seconds (default 15).
    """

    def __init__(
        self,
        api_key: str | None = None,
        regions: str = DEFAULT_REGIONS,
        markets: str = "h2h",
        cache_ttl: int = DEFAULT_CACHE_TTL,
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY", "")
        self.regions = regions
        self.markets = markets
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._cache: dict[str, Any] = {}
        self._cache_loaded = False
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
        })

        if not self.api_key:
            logger.warning(
                "THE_ODDS_API_KEY not set. Live odds unavailable. "
                "Get a free key at https://the-odds-api.com/"
            )

    # ── Public API ──────────────────────────────────────

    def get_available_sports(self) -> list[dict[str, Any]]:
        """Fetch the list of available sports/leagues.

        Returns
        -------
        list[dict[str, Any]]
            List of sport objects with ``key``, ``group``, ``title``,
            ``active``, and ``has_outrights`` fields.
        """
        if not self.api_key:
            return []

        url = f"{API_BASE_URL}/sports"
        return self._get(url)

    def get_upcoming_odds(
        self,
        sport_key: str = "upcoming",
        bookmaker: str | None = None,
    ) -> list[MatchOdds]:
        """Fetch upcoming match odds for a sport.

        Parameters
        ----------
        sport_key : str
            Sport key (e.g. ``"soccer_epl"``, ``"soccer_fifa_world_cup"``).
            Use ``"upcoming"`` for all sports combined.
        bookmaker : str, optional
            If set, only return odds from this bookmaker (e.g. ``"bet365"``).
            Otherwise returns the best available odds across all bookmakers.

        Returns
        -------
        list[MatchOdds]
            List of ``MatchOdds`` objects.
        """
        if not self.api_key:
            return []

        url = f"{API_BASE_URL}/sports/{sport_key}/odds"
        params = {
            "regions": self.regions,
            "markets": self.markets,
        }
        data = self._get(url, params=params)

        if not data:
            return []

        return self._parse_response(data, bookmaker)

    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        sport_key: str = "soccer_fifa_world_cup",
        bookmaker: str | None = None,
    ) -> dict[str, float] | None:
        """Get odds for a specific match by team names.

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name.
        sport_key : str
            Sport key to search within (default ``"soccer_fifa_world_cup"``).
        bookmaker : str, optional
            Specific bookmaker to use.

        Returns
        -------
        dict[str, float] | None
            ``{"home_odds": ..., "draw_odds": ..., "away_odds": ...}``
            or ``None`` if the match isn't found.
        """
        matches = self.get_upcoming_odds(sport_key=sport_key, bookmaker=bookmaker)
        for match in matches:
            h = match.home_team.lower().strip()
            a = match.away_team.lower().strip()
            if h == home_team.lower().strip() and a == away_team.lower().strip():
                return {
                    "home_odds": match.home_odds,
                    "draw_odds": match.draw_odds,
                    "away_odds": match.away_odds,
                    "bookmaker": match.bookmaker,
                    "match_date": match.match_date,
                }
            # Try swapped (in case home/away labels differ)
            if h == away_team.lower().strip() and a == home_team.lower().strip():
                return {
                    "home_odds": match.away_odds,
                    "draw_odds": match.draw_odds,
                    "away_odds": match.home_odds,
                    "bookmaker": match.bookmaker,
                    "match_date": match.match_date,
                }
        return None

    def get_value_bet_odds(
        self,
        team_pairs: list[tuple[str, str]],
        sport_key: str = "soccer_fifa_world_cup",
        bookmaker: str | None = None,
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Get odds for multiple matches in bulk.

        This is the primary entry point for the value betting pipeline.
        Fetches all upcoming matches once, then matches by team names.

        Parameters
        ----------
        team_pairs : list[tuple[str, str]]
            List of ``(home_team, away_team)`` tuples.
        sport_key : str
            Sport key to search within.
        bookmaker : str, optional
            Specific bookmaker.

        Returns
        -------
        dict[tuple[str, str], dict[str, float]]
            Mapping of ``(home_team, away_team) → odds_dict``.
            Missing matches are excluded from the result.
        """
        if not self.api_key:
            return {}

        matches = self.get_upcoming_odds(sport_key=sport_key, bookmaker=bookmaker)

        # Build a lookup by team name (case-insensitive)
        lookup: dict[str, MatchOdds] = {}
        for m in matches:
            lookup[(m.home_team.lower(), m.away_team.lower())] = m

        results: dict[tuple[str, str], dict[str, float]] = {}
        for h, a in team_pairs:
            key = (h.lower(), a.lower())
            m = lookup.get(key)
            if m is None:
                # Try swapped
                m = lookup.get((a.lower(), h.lower()))
                if m is not None:
                    results[(h, a)] = {
                        "home_odds": m.away_odds,
                        "draw_odds": m.draw_odds,
                        "away_odds": m.home_odds,
                        "bookmaker": m.bookmaker,
                        "match_date": m.match_date,
                    }
                continue
            results[(h, a)] = {
                "home_odds": m.home_odds,
                "draw_odds": m.draw_odds,
                "away_odds": m.away_odds,
                "bookmaker": m.bookmaker,
                "match_date": m.match_date,
            }

        return results

    # ── Internal helpers ────────────────────────────────

    def _get(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Make a GET request with caching."""
        # Check cache first
        cached = self._load_cache(url, params)
        if cached is not None:
            return cached

        # Make actual request
        all_params = {"apiKey": self.api_key}
        if params:
            all_params.update(params)

        try:
            logger.debug("GET %s (params=%s)", url, all_params)
            resp = self._session.get(url, params=all_params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Cache the result
            self._save_cache(url, params, data)
            return data

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.error("Odds API HTTP %s: %s", status, exc)
            return []
        except requests.ConnectionError as exc:
            logger.error("Odds API connection failed: %s", exc)
            return []
        except requests.Timeout:
            logger.error("Odds API request timed out (%ss)", self.timeout)
            return []
        except json.JSONDecodeError as exc:
            logger.error("Odds API returned invalid JSON: %s", exc)
            return []

    def _parse_response(
        self,
        data: list[dict[str, Any]],
        bookmaker: str | None = None,
    ) -> list[MatchOdds]:
        """Parse the API response into ``MatchOdds`` objects.

        For each event, extracts the best available odds across all
        bookmakers (or from a specified bookmaker).
        """
        results: list[MatchOdds] = []

        for event in data:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence_time = event.get("commence_time", "")
            sport_key = event.get("sport_key", "")
            sport_title = event.get("sport_title", "")
            bookmakers = event.get("bookmakers", [])

            if not bookmakers:
                continue

            best_home = 0.0
            best_draw = 0.0
            best_away = 0.0
            best_bookmaker = ""

            for bk in bookmakers:
                bk_title = bk.get("title", "unknown")
                markets = bk.get("markets", [])

                # If filtering by bookmaker, skip others
                if bookmaker and bk_title.lower() != bookmaker.lower():
                    continue

                for market in markets:
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    odds_map: dict[str, float] = {}
                    for outcome in outcomes:
                        odds_map[outcome["name"]] = outcome["price"]

                    # Map outcomes to home/draw/away
                    home_odds = odds_map.get(home_team, 0.0)
                    draw_odds = odds_map.get("Draw", 0.0)
                    away_odds = odds_map.get(away_team, 0.0)

                    # Track best odds independently per outcome
                    # (the best home_odds, draw_odds, and away_odds may come
                    # from different bookmakers)
                    if home_odds > best_home:
                        best_home = home_odds
                    if draw_odds > best_draw:
                        best_draw = draw_odds
                    if away_odds > best_away:
                        best_away = away_odds
                    # Track which bookmaker provided the best home odds
                    if home_odds >= best_home:
                        best_bookmaker = bk_title

            if best_home > 0 and best_away > 0:
                results.append(MatchOdds(
                    home_team=home_team,
                    away_team=away_team,
                    match_date=commence_time,
                    home_odds=best_home,
                    draw_odds=best_draw,
                    away_odds=best_away,
                    bookmaker=best_bookmaker,
                    sport_key=sport_key,
                    sport_title=sport_title,
                ))

        return results

    # ── Caching ─────────────────────────────────────────

    def _load_cache(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        """Load a cached response if it's still fresh."""
        if not self._cache_loaded:
            self._load_cache_file()

        cache_key = self._make_cache_key(url, params)
        entry = self._cache.get(cache_key)

        if entry is None:
            return None

        # Check TTL
        age = time.time() - entry["timestamp"]
        if age > self.cache_ttl:
            logger.debug("Cache expired for %s (%.0fs old)", url, age)
            del self._cache[cache_key]
            return None

        logger.debug("Cache hit for %s (%.0fs old)", url, age)
        return entry["data"]

    def _save_cache(
        self,
        url: str,
        params: dict[str, str] | None,
        data: list[dict[str, Any]] | dict[str, Any],
    ) -> None:
        """Save a response to the cache."""
        if not self._cache_loaded:
            self._load_cache_file()

        cache_key = self._make_cache_key(url, params)
        self._cache[cache_key] = {
            "timestamp": time.time(),
            "url": url,
            "params": params,
            "data": data,
        }
        self._persist_cache()

    def _make_cache_key(self, url: str, params: dict[str, str] | None = None) -> str:
        """Create a unique cache key from URL + params."""
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            return f"{url}?{param_str}"
        return url

    def _load_cache_file(self) -> None:
        """Load the cache file from disk."""
        self._cache_loaded = True
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    self._cache = json.load(f)
                # Remove expired entries
                now = time.time()
                expired_keys = [
                    k for k, v in self._cache.items()
                    if now - v.get("timestamp", 0) > self.cache_ttl
                ]
                for k in expired_keys:
                    del self._cache[k]
                if expired_keys:
                    logger.debug("Cleared %d expired cache entries", len(expired_keys))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load odds cache: %s", exc)
                self._cache = {}
        else:
            self._cache = {}

    def _persist_cache(self) -> None:
        """Write the cache to disk."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to persist odds cache: %s", exc)

    def clear_cache(self) -> None:
        """Clear the odds cache."""
        self._cache = {}
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        logger.info("Odds cache cleared")


# ═══════════════════════════════════════════════════════════
#  Convenience function (single-call entry point)
# ═══════════════════════════════════════════════════════════

def fetch_live_odds(
    team_pairs: list[tuple[str, str]],
    sport_key: str = "soccer_fifa_world_cup",
    bookmaker: str | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """One-shot convenience function to fetch live odds for a set of matches.

    Parameters
    ----------
    team_pairs : list[tuple[str, str]]
        List of ``(home_team, away_team)`` tuples.
    sport_key : str
        Sport key (default ``"soccer_fifa_world_cup"``).
    bookmaker : str, optional
        Specific bookmaker (e.g. ``"bet365"``).

    Returns
    -------
    dict[tuple[str, str], dict[str, float]]
        Mapping of match tuple → ``{"home_odds": ..., "draw_odds": ..., "away_odds": ...}``.
        Empty dict if API is unavailable.
    """
    client = OddsAPIClient()
    return client.get_value_bet_odds(team_pairs, sport_key=sport_key, bookmaker=bookmaker)


# ═══════════════════════════════════════════════════════════
#  CLI for testing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    client = OddsAPIClient()

    if not client.api_key:
        print("\n  [X] THE_ODDS_API_KEY not set.")
        print("  Get a free key at https://the-odds-api.com/")
        print("  Then set: export THE_ODDS_API_KEY='your_key_here'")
        sys.exit(1)

    # Test: list available sports
    print("\n  Fetching available sports ...")
    sports = client.get_available_sports()
    soccer_sports = [s for s in sports if "soccer" in s.get("key", "")]
    print(f"  Found {len(soccer_sports)} soccer leagues:\n")
    for s in soccer_sports:
        print(f"    {s['key']:<40} {s['title']}")

    # Test: fetch upcoming World Cup odds
    print("\n  Fetching World Cup odds ...")
    odds = client.get_upcoming_odds(sport_key="soccer_fifa_world_cup")
    if odds:
        print(f"  Found {len(odds)} matches:\n")
        for m in odds:
            print(f"    {m.home_team:<22} vs {m.away_team:<22}  "
                  f"{m.home_odds:<6.2f}  {m.draw_odds:<6.2f}  {m.away_odds:<6.2f}  [{m.bookmaker}]")
    else:
        print("  No matches found. The World Cup league key may differ.")
        print("  Trying 'upcoming' sport key ...")
        odds = client.get_upcoming_odds(sport_key="upcoming")
        if odds:
            # Filter to likely World Cup matches
            wc_matches = [m for m in odds if any(t in m.home_team for t in
                          ["Brazil", "Mexico", "England", "France", "Argentina"])]
            print(f"  Found {len(wc_matches)} potential World Cup matches:\n")
            for m in wc_matches:
                print(f"    {m.home_team:<22} vs {m.away_team:<22}  "
                      f"{m.home_odds:<6.2f}  {m.draw_odds:<6.2f}  {m.away_odds:<6.2f}  [{m.bookmaker}]")
        else:
            print("  No matches found via 'upcoming' either.")
