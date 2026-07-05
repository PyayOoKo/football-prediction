"""Unit tests for the Odds API client (``src.odds_api``).

Uses ``unittest.mock`` to simulate HTTP responses so tests never hit
the real API — free tier rate limits (500 req/month) stay untouched.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, PropertyMock, call, patch

import pytest
import requests

from src.odds_api import (
    API_BASE_URL,
    CACHE_DIR,
    CACHE_FILE,
    MatchOdds,
    OddsAPIClient,
    OddsAPIConfig,
    fetch_live_odds,
)


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def mock_api_key() -> str:
    """A dummy API key for tests that need one."""
    return "test_api_key_12345"


@pytest.fixture
def sample_sports_response() -> list[dict[str, Any]]:
    """Mock response from ``/v4/sports``."""
    return [
        {
            "key": "soccer_epl",
            "group": "Soccer",
            "title": "English Premier League",
            "description": "England's top football division",
            "active": True,
            "has_outrights": False,
        },
        {
            "key": "soccer_fifa_world_cup",
            "group": "Soccer (International)",
            "title": "FIFA World Cup",
            "description": "International soccer's premier tournament",
            "active": True,
            "has_outrights": True,
        },
        {
            "key": "americanfootball_nfl",
            "group": "American Football",
            "title": "NFL",
            "active": True,
            "has_outrights": True,
        },
    ]


@pytest.fixture
def sample_odds_response() -> list[dict[str, Any]]:
    """Mock response from ``/v4/sports/{sport}/odds``.

    Simulates two World Cup R16 matches with odds from two bookmakers.
    """
    return [
        {
            "id": "match_001",
            "sport_key": "soccer_fifa_world_cup",
            "sport_title": "FIFA World Cup",
            "commence_time": "2026-07-05T20:00:00Z",
            "home_team": "Brazil",
            "away_team": "Norway",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Brazil", "price": 1.83},
                                {"name": "Draw", "price": 3.70},
                                {"name": "Norway", "price": 4.20},
                            ],
                        }
                    ],
                },
                {
                    "key": "william_hill",
                    "title": "William Hill",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Brazil", "price": 1.80},
                                {"name": "Draw", "price": 3.75},
                                {"name": "Norway", "price": 4.30},
                            ],
                        }
                    ],
                },
            ],
        },
        {
            "id": "match_002",
            "sport_key": "soccer_fifa_world_cup",
            "sport_title": "FIFA World Cup",
            "commence_time": "2026-07-05T23:00:00Z",
            "home_team": "Mexico",
            "away_team": "England",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Mexico", "price": 3.10},
                                {"name": "Draw", "price": 3.15},
                                {"name": "England", "price": 2.42},
                            ],
                        }
                    ],
                }
            ],
        },
    ]


@pytest.fixture
def sample_empty_odds_response() -> list[dict[str, Any]]:
    """Empty odds response — no matches available."""
    return []


@pytest.fixture
def client_with_key(mock_api_key: str) -> OddsAPIClient:
    """OddsAPIClient with a preset API key (no env var needed)."""
    return OddsAPIClient(api_key=mock_api_key)


@pytest.fixture
def client_no_key() -> OddsAPIClient:
    """OddsAPIClient without any API key."""
    return OddsAPIClient(api_key="")


# ═══════════════════════════════════════════════════════════
#  Initialisation
# ═══════════════════════════════════════════════════════════

class TestInit:
    """OddsAPIClient construction and config."""

    def test_default_construction_no_key(self) -> None:
        """No API key passed and no env var → empty api_key, warning logged."""
        with patch.dict(os.environ, clear=True):
            with patch("src.odds_api.logger.warning") as mock_warn:
                client = OddsAPIClient()
                assert client.api_key == ""
                mock_warn.assert_called_once_with(
                    "THE_ODDS_API_KEY not set. Live odds unavailable. "
                    "Get a free key at https://the-odds-api.com/"
                )

    def test_construction_with_key_argument(self, mock_api_key: str) -> None:
        """API key passed explicitly is stored."""
        client = OddsAPIClient(api_key=mock_api_key)
        assert client.api_key == mock_api_key

    def test_construction_with_env_var(self, mock_api_key: str) -> None:
        """API key from env var is picked up."""
        with patch.dict(os.environ, {"THE_ODDS_API_KEY": mock_api_key}, clear=True):
            client = OddsAPIClient()
            assert client.api_key == mock_api_key

    def test_key_argument_overrides_env_var(self, mock_api_key: str) -> None:
        """Explicit key argument takes precedence over env var."""
        with patch.dict(os.environ, {"THE_ODDS_API_KEY": "env_key"}, clear=True):
            client = OddsAPIClient(api_key=mock_api_key)
            assert client.api_key == mock_api_key  # not "env_key"

    def test_construction_with_custom_settings(self) -> None:
        """Custom timeout, regions, and cache_ttl are set."""
        client = OddsAPIClient(
            api_key="key",
            regions="us",
            markets="h2h,spreads",
            cache_ttl=1800,
            timeout=30,
        )
        assert client.regions == "us"
        assert client.markets == "h2h,spreads"
        assert client.cache_ttl == 1800
        assert client.timeout == 30

    def test_session_headers(self, client_with_key: OddsAPIClient) -> None:
        """Session is configured with JSON accept header."""
        assert client_with_key._session.headers["Accept"] == "application/json"

    def test_odds_api_config_dataclass(self) -> None:
        """OddsAPIConfig dataclass has expected defaults."""
        cfg = OddsAPIConfig()
        assert cfg.api_key == ""
        assert cfg.regions == "uk,ie,eu"
        assert cfg.markets == "h2h"
        assert cfg.cache_ttl == 3600
        assert cfg.timeout == 15


# ═══════════════════════════════════════════════════════════
#  Cache key generation
# ═══════════════════════════════════════════════════════════

class TestCacheKey:
    """_make_cache_key utility."""

    def test_url_only(self, client_with_key: OddsAPIClient) -> None:
        """URL without params returns the URL itself."""
        key = client_with_key._make_cache_key("https://api.example.com/sports")
        assert key == "https://api.example.com/sports"

    def test_url_with_params(self, client_with_key: OddsAPIClient) -> None:
        """Params are sorted alphabetically and appended as query string."""
        key = client_with_key._make_cache_key(
            "https://api.example.com/odds",
            {"regions": "uk", "markets": "h2h", "apiKey": "secret"},
        )
        assert key == "https://api.example.com/odds?apiKey=secret&markets=h2h&regions=uk"

    def test_same_params_produce_same_key(self, client_with_key: OddsAPIClient) -> None:
        """Same URL + same params → same cache key (deterministic)."""
        p = {"a": "1", "b": "2"}
        k1 = client_with_key._make_cache_key("url", p)
        k2 = client_with_key._make_cache_key("url", p)
        assert k1 == k2

    def test_different_params_different_key(self, client_with_key: OddsAPIClient) -> None:
        """Different params → different cache keys."""
        k1 = client_with_key._make_cache_key("url", {"a": "1"})
        k2 = client_with_key._make_cache_key("url", {"a": "2"})
        assert k1 != k2


# ═══════════════════════════════════════════════════════════
#  Response parsing
# ═══════════════════════════════════════════════════════════

class TestParseResponse:
    """_parse_response converts raw API data into MatchOdds objects."""

    def test_basic_parsing(self, client_with_key: OddsAPIClient,
                           sample_odds_response: list[dict[str, Any]]) -> None:
        """Correctly parses two matches with odds."""
        odds = client_with_key._parse_response(sample_odds_response)
        assert len(odds) == 2

        brazil_vs_norway = odds[0]
        assert brazil_vs_norway.home_team == "Brazil"
        assert brazil_vs_norway.away_team == "Norway"
        assert brazil_vs_norway.home_odds == pytest.approx(1.83)   # Bet365
        assert brazil_vs_norway.draw_odds == pytest.approx(3.75)   # William Hill better
        assert brazil_vs_norway.away_odds == pytest.approx(4.30)   # William Hill better

        mexico_vs_england = odds[1]
        assert mexico_vs_england.home_team == "Mexico"
        assert mexico_vs_england.away_team == "England"
        assert mexico_vs_england.home_odds == pytest.approx(3.10)
        assert mexico_vs_england.away_odds == pytest.approx(2.42)

    def test_best_odds_selected_independently_per_outcome(
        self, client_with_key: OddsAPIClient,
    ) -> None:
        """When multiple bookmakers, the best odds for each outcome are chosen
        independently (different outcomes can come from different bookmakers)."""
        data = [
            {
                "id": "m1",
                "home_team": "TeamA",
                "away_team": "TeamB",
                "commence_time": "2026-07-05T20:00:00Z",
                "sport_key": "soccer_test",
                "sport_title": "Test League",
                "bookmakers": [
                    {
                        "key": "bk1",
                        "title": "Bookie1",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "TeamA", "price": 2.0},
                                    {"name": "Draw", "price": 3.0},
                                    {"name": "TeamB", "price": 4.0},
                                ],
                            }
                        ],
                    },
                    {
                        "key": "bk2",
                        "title": "Bookie2",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "TeamA", "price": 2.2},
                                    {"name": "Draw", "price": 3.2},
                                    {"name": "TeamB", "price": 3.8},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
        odds = client_with_key._parse_response(data)
        assert len(odds) == 1
        m = odds[0]
        # Each outcome picks the best independently:
        #   Best home: 2.2 (Bookie2)
        #   Best draw: 3.2 (Bookie2)
        #   Best away: 4.0 (Bookie1)
        assert m.home_odds == pytest.approx(2.2)
        assert m.draw_odds == pytest.approx(3.2)
        assert m.away_odds == pytest.approx(4.0)

    def test_bookmaker_filter(self, client_with_key: OddsAPIClient,
                              sample_odds_response: list[dict[str, Any]]) -> None:
        """When bookmaker is specified, only odds from that bookie are used."""
        odds = client_with_key._parse_response(sample_odds_response, bookmaker="William Hill")
        assert len(odds) == 1  # only Brazil-Norway has William Hill
        assert odds[0].home_team == "Brazil"
        assert odds[0].home_odds == pytest.approx(1.80)
        assert odds[0].away_odds == pytest.approx(4.30)

    def test_bookmaker_filter_case_insensitive(
        self, client_with_key: OddsAPIClient,
        sample_odds_response: list[dict[str, Any]],
    ) -> None:
        """Bookmaker filter is case-insensitive."""
        odds = client_with_key._parse_response(sample_odds_response, bookmaker="william hill")
        assert len(odds) == 1

    def test_no_bookmakers(self, client_with_key: OddsAPIClient) -> None:
        """Match with no bookmakers is skipped."""
        data: list[dict[str, Any]] = [
            {
                "id": "m1",
                "home_team": "TeamA",
                "away_team": "TeamB",
                "commence_time": "2026-07-05T20:00:00Z",
                "sport_key": "soccer_test",
                "sport_title": "Test League",
                "bookmakers": [],
            }
        ]
        odds = client_with_key._parse_response(data)
        assert len(odds) == 0

    def test_non_h2h_markets_ignored(self, client_with_key: OddsAPIClient) -> None:
        """Only h2h markets are processed; others (e.g. spreads) are ignored."""
        data: list[dict[str, Any]] = [
            {
                "id": "m1",
                "home_team": "TeamA",
                "away_team": "TeamB",
                "commence_time": "2026-07-05T20:00:00Z",
                "sport_key": "soccer_test",
                "sport_title": "Test League",
                "bookmakers": [
                    {
                        "key": "bk1",
                        "title": "Bookie1",
                        "markets": [
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "TeamA", "price": 1.5},
                                    {"name": "TeamB", "price": 2.5},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        odds = client_with_key._parse_response(data)
        assert len(odds) == 0  # no h2h market → skipped

    def test_empty_input(self, client_with_key: OddsAPIClient) -> None:
        """Empty list input returns empty results."""
        assert client_with_key._parse_response([]) == []


# ═══════════════════════════════════════════════════════════
#  Public methods — no API key
# ═══════════════════════════════════════════════════════════

class TestPublicNoKey:
    """All public methods return safe empty values when no API key is set."""

    def test_get_available_sports_no_key(self, client_no_key: OddsAPIClient) -> None:
        assert client_no_key.get_available_sports() == []

    def test_get_upcoming_odds_no_key(self, client_no_key: OddsAPIClient) -> None:
        assert client_no_key.get_upcoming_odds() == []

    def test_get_match_odds_no_key(self, client_no_key: OddsAPIClient) -> None:
        assert client_no_key.get_match_odds("Brazil", "Norway") is None

    def test_get_value_bet_odds_no_key(self, client_no_key: OddsAPIClient) -> None:
        assert client_no_key.get_value_bet_odds([("Brazil", "Norway")]) == {}

    def test_fetch_live_odds_no_key(self) -> None:
        """fetch_live_odds convenience function with no API key."""
        with patch.dict(os.environ, clear=True):
            result = fetch_live_odds([("Brazil", "Norway")])
            assert result == {}


# ═══════════════════════════════════════════════════════════
#  Public methods — with API key (mocked HTTP)
# ═══════════════════════════════════════════════════════════

class TestGetAvailableSports:
    """get_available_sports endpoint."""

    def test_returns_sports(self, client_with_key: OddsAPIClient,
                            sample_sports_response: list[dict[str, Any]]) -> None:
        """Returns the list of available sports."""
        with patch.object(client_with_key, "_get", return_value=sample_sports_response):
            sports = client_with_key.get_available_sports()
            assert len(sports) == 3
            assert sports[0]["key"] == "soccer_epl"

    def test_calls_correct_url(self, client_with_key: OddsAPIClient,
                                sample_sports_response: list[dict[str, Any]]) -> None:
        """Calls _get with the correct sports URL."""
        with patch.object(client_with_key, "_get", return_value=sample_sports_response) as mock:
            client_with_key.get_available_sports()
            mock.assert_called_once_with(f"{API_BASE_URL}/sports")


class TestGetUpcomingOdds:
    """get_upcoming_odds endpoint."""

    def test_returns_match_odds(self, client_with_key: OddsAPIClient,
                                sample_odds_response: list[dict[str, Any]]) -> None:
        """Returns parsed MatchOdds objects."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            odds = client_with_key.get_upcoming_odds(sport_key="soccer_fifa_world_cup")
            assert len(odds) == 2
            assert all(isinstance(o, MatchOdds) for o in odds)

    def test_calls_correct_url(self, client_with_key: OddsAPIClient,
                                sample_odds_response: list[dict[str, Any]]) -> None:
        """Calls _get with the correct sport URL and params."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response) as mock:
            client_with_key.get_upcoming_odds(
                sport_key="soccer_fifa_world_cup",
                bookmaker="bet365",
            )
            expected_url = f"{API_BASE_URL}/sports/soccer_fifa_world_cup/odds"
            mock.assert_called_once_with(expected_url, params={"regions": "uk,ie,eu", "markets": "h2h"})

    def test_empty_response(self, client_with_key: OddsAPIClient,
                            sample_empty_odds_response: list[dict[str, Any]]) -> None:
        """Empty API response returns empty list."""
        with patch.object(client_with_key, "_get", return_value=sample_empty_odds_response):
            odds = client_with_key.get_upcoming_odds()
            assert odds == []

    def test_none_response(self, client_with_key: OddsAPIClient) -> None:
        """_get returning None (e.g. from cache miss) returns empty list."""
        with patch.object(client_with_key, "_get", return_value=[]):
            odds = client_with_key.get_upcoming_odds()
            assert odds == []


class TestGetMatchOdds:
    """get_match_odds — find a specific match by team names."""

    def test_finds_match(self, client_with_key: OddsAPIClient,
                         sample_odds_response: list[dict[str, Any]]) -> None:
        """Returns correct odds dict for a known match."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            result = client_with_key.get_match_odds("Brazil", "Norway")
            assert result is not None
            assert result["home_odds"] == pytest.approx(1.83)   # Bet365
            assert result["draw_odds"] == pytest.approx(3.75)   # William Hill better
            assert result["away_odds"] == pytest.approx(4.30)   # William Hill better
            assert "match_date" in result

    def test_case_insensitive(self, client_with_key: OddsAPIClient,
                              sample_odds_response: list[dict[str, Any]]) -> None:
        """Team name matching is case-insensitive."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            result = client_with_key.get_match_odds("brazil", "norway")
            assert result is not None
            assert result["home_odds"] == pytest.approx(1.83)

    def test_whitespace_insensitive(self, client_with_key: OddsAPIClient,
                                     sample_odds_response: list[dict[str, Any]]) -> None:
        """Leading/trailing whitespace is stripped."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            result = client_with_key.get_match_odds("  Brazil  ", "  Norway  ")
            assert result is not None

    def test_swapped_home_away(self, client_with_key: OddsAPIClient,
                                sample_odds_response: list[dict[str, Any]]) -> None:
        """If we pass teams in the wrong order, still finds the match (swaps odds)."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            # Norway is away, but we search "Norway" as home
            result = client_with_key.get_match_odds("Norway", "Brazil")
            assert result is not None
            # home_odds should be Norway's best odds (4.30), away_odds Brazil's (1.83)
            assert result["home_odds"] == pytest.approx(4.30)   # best away=Norway from William Hill
            assert result["away_odds"] == pytest.approx(1.83)    # best home=Brazil from Bet365
            assert result["draw_odds"] == pytest.approx(3.75)    # best draw from William Hill

    def test_match_not_found(self, client_with_key: OddsAPIClient,
                             sample_odds_response: list[dict[str, Any]]) -> None:
        """Unknown teams return None."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            result = client_with_key.get_match_odds("FakeTeam", "Nonexistent")
            assert result is None


class TestGetValueBetOdds:
    """get_value_bet_odds — bulk lookup for the value betting pipeline."""

    def test_finds_multiple_matches(self, client_with_key: OddsAPIClient,
                                     sample_odds_response: list[dict[str, Any]]) -> None:
        """Returns odds for all requested matches that exist."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            pairs = [("Brazil", "Norway"), ("Mexico", "England")]
            results = client_with_key.get_value_bet_odds(pairs)
            assert len(results) == 2

            bn = results[("Brazil", "Norway")]
            assert bn["home_odds"] == pytest.approx(1.83)   # Bet365
            assert bn["away_odds"] == pytest.approx(4.30)   # William Hill better

            me = results[("Mexico", "England")]
            assert me["home_odds"] == pytest.approx(3.10)

    def test_skips_missing_matches(self, client_with_key: OddsAPIClient,
                                    sample_odds_response: list[dict[str, Any]]) -> None:
        """Unknown matches are not included in the result dict."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            pairs = [("Brazil", "Norway"), ("Fake", "Teams")]
            results = client_with_key.get_value_bet_odds(pairs)
            assert len(results) == 1
            assert ("Fake", "Teams") not in results

    def test_swapped_teams_in_bulk(self, client_with_key: OddsAPIClient,
                                    sample_odds_response: list[dict[str, Any]]) -> None:
        """Swapped home/away pairs are still matched correctly."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            # Norway is technically away, but user passes it as home
            pairs = [("Norway", "Brazil")]
            results = client_with_key.get_value_bet_odds(pairs)
            assert len(results) == 1
            r = results[("Norway", "Brazil")]
            assert r["home_odds"] == pytest.approx(4.30)  # Norway's best odds (William Hill)
            assert r["away_odds"] == pytest.approx(1.83)  # Brazil's best odds (Bet365)

    def test_empty_pairs(self, client_with_key: OddsAPIClient,
                          sample_odds_response: list[dict[str, Any]]) -> None:
        """Empty team pairs returns empty dict."""
        with patch.object(client_with_key, "_get", return_value=sample_odds_response):
            results = client_with_key.get_value_bet_odds([])
            assert results == {}


# ═══════════════════════════════════════════════════════════
#  Error handling
# ═══════════════════════════════════════════════════════════

class TestErrorHandling:
    """_get() should gracefully handle all HTTP-related errors."""

    def test_http_404(self, client_with_key: OddsAPIClient) -> None:
        """404 HTTP error returns empty list."""
        with patch.object(client_with_key._session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "404 Client Error", response=mock_response
            )
            mock_get.return_value = mock_response
            result = client_with_key._get(f"{API_BASE_URL}/sports/bad_sport")
            assert result == []

    def test_connection_error(self, client_with_key: OddsAPIClient) -> None:
        """Connection error (e.g. no internet) returns empty list."""
        with patch.object(client_with_key._session, "get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")
            result = client_with_key._get(f"{API_BASE_URL}/sports")
            assert result == []

    def test_timeout(self, client_with_key: OddsAPIClient) -> None:
        """Request timeout returns empty list."""
        with patch.object(client_with_key._session, "get") as mock_get:
            mock_get.side_effect = requests.Timeout("Timed out")
            result = client_with_key._get(f"{API_BASE_URL}/sports")
            assert result == []

    def test_invalid_json(self, client_with_key: OddsAPIClient) -> None:
        """Non-JSON response returns empty list."""
        with patch.object(client_with_key._session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.side_effect = json.JSONDecodeError("Bad JSON", "", 0)
            mock_get.return_value = mock_response
            result = client_with_key._get(f"{API_BASE_URL}/sports")
            assert result == []

    def test_rate_limit_429(self, client_with_key: OddsAPIClient) -> None:
        """429 rate-limit response is treated as HTTP error -> empty list."""
        with patch.object(client_with_key._session, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "429 Rate Limited", response=mock_response
            )
            mock_get.return_value = mock_response
            result = client_with_key._get(f"{API_BASE_URL}/sports")
            assert result == []


# ═══════════════════════════════════════════════════════════
#  Caching behavior
# ═══════════════════════════════════════════════════════════

class TestCaching:
    """Caching layer inside _get.

    Uses ``tmp_path`` to avoid the real ``data/external/odds_cache.json``
    file interfering with test assertions.
    """

    def test_cache_miss_hits_api(self, client_with_key: OddsAPIClient,
                                  tmp_path: Path) -> None:
        """On cache miss, the API is called and result is cached."""
        with patch("src.odds_api.CACHE_FILE", tmp_path / "odds_cache.json"):
            with patch.object(client_with_key._session, "get") as mock_get:
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = [{"key": "soccer_epl"}]
                mock_get.return_value = mock_response

                # First call — cache miss, should call API
                result = client_with_key._get(f"{API_BASE_URL}/sports")
                assert result == [{"key": "soccer_epl"}]
                mock_get.assert_called_once()

    def test_cache_hit_skips_api(self, client_with_key: OddsAPIClient,
                                 tmp_path: Path) -> None:
        """On cache hit, the cached result is returned without API call."""
        with patch("src.odds_api.CACHE_FILE", tmp_path / "odds_cache.json"):
            with patch.object(client_with_key._session, "get") as mock_get:
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = [{"key": "soccer_epl"}]
                mock_get.return_value = mock_response

                # First call populates cache
                client_with_key._get(f"{API_BASE_URL}/sports")

                # Second call — cache hit, should NOT call API
                result = client_with_key._get(f"{API_BASE_URL}/sports")
                assert result == [{"key": "soccer_epl"}]
                mock_get.assert_called_once()  # still only one call

    def test_cache_expiry(self, client_with_key: OddsAPIClient,
                          tmp_path: Path) -> None:
        """After TTL expires, the cache is skipped and API is called again."""
        client_with_key.cache_ttl = 0.001  # 1ms TTL

        with patch("src.odds_api.CACHE_FILE", tmp_path / "odds_cache.json"):
            with patch.object(client_with_key._session, "get") as mock_get:
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = [{"key": "soccer_epl"}]
                mock_get.return_value = mock_response

                # First call — cache miss
                client_with_key._get(f"{API_BASE_URL}/sports")
                time.sleep(0.02)  # wait past TTL

                # Second call — cache expired, should call API again
                client_with_key._get(f"{API_BASE_URL}/sports")
                assert mock_get.call_count == 2

    def test_cache_key_isolation(self, client_with_key: OddsAPIClient,
                                 tmp_path: Path) -> None:
        """Different URLs have separate cache entries."""
        with patch("src.odds_api.CACHE_FILE", tmp_path / "odds_cache.json"):
            with patch.object(client_with_key._session, "get") as mock_get:
                mock_response = MagicMock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.side_effect = [
                    [{"key": "sports"}],   # first URL
                    [{"key": "odds"}],     # second URL
                ]
                mock_get.return_value = mock_response

                r1 = client_with_key._get(f"{API_BASE_URL}/sports")
                r2 = client_with_key._get(f"{API_BASE_URL}/sports/odds")
                r3 = client_with_key._get(f"{API_BASE_URL}/sports")  # cache hit

                assert r1 == [{"key": "sports"}]
                assert r2 == [{"key": "odds"}]
                assert r3 == [{"key": "sports"}]
                assert mock_get.call_count == 2  # not 3

    def test_clear_cache(self, client_with_key: OddsAPIClient,
                         tmp_path: Path) -> None:
        """clear_cache empties in-memory cache and deletes cache file."""
        cache_file = tmp_path / "odds_cache.json"
        with patch("src.odds_api.CACHE_FILE", cache_file):
            # Populate cache and write to file
            client_with_key._cache = {"test": {"data": "value"}}
            client_with_key._persist_cache()
            assert cache_file.exists()

            client_with_key.clear_cache()
            assert client_with_key._cache == {}
            assert not cache_file.exists()


# ═══════════════════════════════════════════════════════════
#  Cache persistence (file I/O)
# ═══════════════════════════════════════════════════════════

class TestCachePersistence:
    """Cache file load/save operations."""

    def test_persist_and_load(self, client_with_key: OddsAPIClient, tmp_path: Path) -> None:
        """Cache is persisted to disk and can be reloaded."""
        with patch("src.odds_api.CACHE_DIR", tmp_path), \
             patch("src.odds_api.CACHE_FILE", tmp_path / "odds_cache.json"):
            cache_file = tmp_path / "odds_cache.json"

            client_with_key._cache = {"my_key": {"data": "stored", "timestamp": time.time()}}
            client_with_key._persist_cache()
            assert cache_file.exists()

            # New client reads the cache
            client_with_key._cache = {}
            client_with_key._cache_loaded = False
            client_with_key._load_cache_file()
            assert "my_key" in client_with_key._cache
            assert client_with_key._cache["my_key"]["data"] == "stored"

    def test_clear_cache_no_file(self, client_with_key: OddsAPIClient) -> None:
        """clear_cache works even when no cache file exists."""
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        client_with_key.clear_cache()  # should not raise
        assert client_with_key._cache == {}

    def test_corrupted_cache_file(self, client_with_key: OddsAPIClient, tmp_path: Path) -> None:
        """Corrupted cache file is handled gracefully (falls back to empty cache)."""
        cache_file = tmp_path / "odds_cache.json"
        cache_file.write_text("this is not valid json")
        with patch("src.odds_api.CACHE_FILE", cache_file):
            client_with_key._load_cache_file()
            assert client_with_key._cache == {}


# ═══════════════════════════════════════════════════════════
#  MatchOdds dataclass
# ═══════════════════════════════════════════════════════════

class TestMatchOdds:
    """MatchOdds dataclass construction and defaults."""

    def test_minimal_creation(self) -> None:
        """MatchOdds can be created with the minimum required fields."""
        m = MatchOdds(
            home_team="Brazil",
            away_team="Norway",
            match_date="2026-07-05T20:00:00Z",
            home_odds=1.83,
            draw_odds=3.70,
            away_odds=4.20,
            bookmaker="Bet365",
        )
        assert m.sport_key == ""
        assert m.sport_title == ""

    def test_full_creation(self) -> None:
        """MatchOdds can be created with all fields."""
        m = MatchOdds(
            home_team="Brazil",
            away_team="Norway",
            match_date="2026-07-05T20:00:00Z",
            home_odds=1.83,
            draw_odds=3.70,
            away_odds=4.20,
            bookmaker="Bet365",
            sport_key="soccer_fifa_world_cup",
            sport_title="FIFA World Cup",
        )
        assert m.sport_key == "soccer_fifa_world_cup"
        assert m.sport_title == "FIFA World Cup"


# ═══════════════════════════════════════════════════════════
#  Integration-style: public methods exercise the full chain
# ═══════════════════════════════════════════════════════════

class TestIntegrationFlow:
    """End-to-end style tests through the public API (still mocked)."""

    def test_get_match_odds_via_get_upcoming_odds(
        self, client_with_key: OddsAPIClient,
        sample_odds_response: list[dict[str, Any]],
    ) -> None:
        """get_match_odds internally calls get_upcoming_odds -> _get -> _parse_response."""
        with patch.object(client_with_key._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = sample_odds_response
            mock_get.return_value = mock_resp

            result = client_with_key.get_match_odds("Mexico", "England")

            assert result is not None
            assert result["home_odds"] == pytest.approx(3.10)

    def test_get_value_bet_odds_via_get_upcoming_odds(
        self, client_with_key: OddsAPIClient,
        sample_odds_response: list[dict[str, Any]],
    ) -> None:
        """get_value_bet_odds internally calls get_upcoming_odds."""
        with patch.object(client_with_key._session, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = sample_odds_response
            mock_get.return_value = mock_resp

            results = client_with_key.get_value_bet_odds([
                ("Brazil", "Norway"), ("Mexico", "England"),
            ])
            assert len(results) == 2

    def test_fetch_live_odds_convenience(
        self, mock_api_key: str,
        sample_odds_response: list[dict[str, Any]],
    ) -> None:
        """fetch_live_odds convenience function works end-to-end."""
        with patch("src.odds_api.OddsAPIClient._get", return_value=sample_odds_response), \
             patch.dict(os.environ, {"THE_ODDS_API_KEY": mock_api_key}, clear=True):
            results = fetch_live_odds([("Brazil", "Norway")])
            assert len(results) == 1
            assert results[("Brazil", "Norway")]["home_odds"] == pytest.approx(1.83)
