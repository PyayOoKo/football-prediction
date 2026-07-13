"""
Unit tests for the EntityResolver — FK ID resolution.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from src.importers.resolver import EntityResolver


# ── Team resolution (with mocked DB) ─────────────────────


class TestTeamResolution:
    def test_resolve_empty_name(self) -> None:
        """Empty or None team names return None."""
        resolver = EntityResolver()
        assert resolver.resolve_team("") is None
        assert resolver.resolve_team("  ") is None
        assert resolver.resolve_team(None) is None  # type: ignore[arg-type]

    @patch(
        "src.importers.resolver.EntityResolver._query_team",
        return_value=42,
    )
    def test_resolve_exact_match(self, mock_query) -> None:
        """Exact team name resolves to the correct ID."""
        resolver = EntityResolver()
        team_id = resolver.resolve_team("Manchester United")
        assert team_id == 42
        mock_query.assert_called_once_with("Manchester United")

    @patch(
        "src.importers.resolver.EntityResolver._query_team",
        return_value=42,
    )
    def test_resolve_alias(self, mock_query) -> None:
        """Alias 'Man Utd' resolves to 'Manchester United' ID."""
        resolver = EntityResolver()
        team_id = resolver.resolve_team("Man Utd")
        assert team_id == 42
        # Should resolve to canonical name
        mock_query.assert_called_once_with("Manchester United")

    @patch(
        "src.importers.resolver.EntityResolver._query_team",
        return_value=55,
    )
    def test_resolve_fuzzy(self, mock_query) -> None:
        """Fuzzy match 'Arsnal' resolves to 'Arsenal'."""
        resolver = EntityResolver()
        team_id = resolver.resolve_team("Arsnal")
        assert team_id == 55
        mock_query.assert_called_once_with("Arsenal")

    @patch(
        "src.importers.resolver.EntityResolver._query_team",
        return_value=None,
    )
    def test_resolve_not_found(self, mock_query) -> None:
        """Unknown team returns None."""
        resolver = EntityResolver()
        team_id = resolver.resolve_team("Totally Fake FC")
        assert team_id is None

    def test_resolve_caching(self) -> None:
        """Repeated lookups use cache, not DB."""
        resolver = EntityResolver()

        # First call: miss cache, query DB
        with patch.object(
            resolver, "_query_team", return_value=42,
        ) as mock_query:
            t1 = resolver.resolve_team("Man Utd")
            assert t1 == 42
            assert mock_query.call_count == 1

        # Second call: hit cache, no DB query
        with patch.object(
            resolver, "_query_team", return_value=999,
        ) as mock_query:
            t2 = resolver.resolve_team("Man Utd")
            assert t2 == 42  # Original cached value
            mock_query.assert_not_called()

    @patch(
        "src.importers.resolver.EntityResolver._create_team",
        return_value=100,
    )
    @patch(
        "src.importers.resolver.EntityResolver._query_team",
        return_value=None,
    )
    def test_auto_create_team(self, mock_query, mock_create) -> None:
        """Auto-create mode creates unknown teams."""
        resolver = EntityResolver(auto_create_teams=True)
        team_id = resolver.resolve_team("NewTeam FC")
        assert team_id == 100
        mock_create.assert_called_once_with("NewTeam FC")


# ── Competition resolution ────────────────────────────────


class TestCompetitionResolution:
    @patch(
        "src.importers.resolver.EntityResolver._query_competition_by_code",
        return_value=1,
    )
    def test_resolve_by_code(self, mock_query) -> None:
        """League code 'E0' resolves to competition ID 1."""
        resolver = EntityResolver()
        comp_id = resolver.resolve_competition(code="E0")
        assert comp_id == 1
        mock_query.assert_called_once_with("E0")

    @patch(
        "src.importers.resolver.EntityResolver._query_competition_by_name",
        return_value=2,
    )
    def test_resolve_by_name(self, mock_query) -> None:
        """Full name resolves to competition ID."""
        resolver = EntityResolver()
        comp_id = resolver.resolve_competition(name="Premier League")
        assert comp_id == 2
        mock_query.assert_called_once_with("Premier League")

    @patch(
        "src.importers.resolver.EntityResolver._query_competition_by_code",
        return_value=None,
    )
    @patch(
        "src.importers.resolver.EntityResolver._query_competition_by_name",
        return_value=None,
    )
    def test_resolve_no_match(self, mock_code, mock_name) -> None:
        """Unknown competition returns None."""
        resolver = EntityResolver()
        comp_id = resolver.resolve_competition(code="XX")
        assert comp_id is None

    @patch(
        "src.importers.resolver.EntityResolver._create_competition",
        return_value=10,
    )
    @patch(
        "src.importers.resolver.EntityResolver._query_competition_by_code",
        return_value=None,
    )
    def test_auto_create_competition(self, mock_query, mock_create) -> None:
        """Auto-create mode creates unknown competitions."""
        resolver = EntityResolver(auto_create_competitions=True)
        comp_id = resolver.resolve_competition(code="XX", name="Unknown League")
        assert comp_id == 10


# ── Season resolution ─────────────────────────────────────


class TestSeasonResolution:
    @patch(
        "src.importers.resolver.EntityResolver._query_season",
        return_value=5,
    )
    def test_resolve_by_name(self, mock_query) -> None:
        """Season name resolves to ID."""
        resolver = EntityResolver()
        season_id = resolver.resolve_season(
            season_name="2024/2025",
            competition_id=1,
        )
        assert season_id == 5

    @patch(
        "src.importers.resolver.EntityResolver._query_season",
        return_value=None,
    )
    def test_resolve_no_match(self, mock_query) -> None:
        """Unknown season returns None."""
        resolver = EntityResolver()
        season_id = resolver.resolve_season(
            season_name="2099/2100",
            competition_id=1,
        )
        assert season_id is None

    @patch(
        "src.importers.resolver.EntityResolver._create_season",
        return_value=20,
    )
    @patch(
        "src.importers.resolver.EntityResolver._query_season",
        return_value=None,
    )
    def test_auto_create_season(self, mock_query, mock_create) -> None:
        """Auto-create mode creates unknown seasons."""
        resolver = EntityResolver(auto_create_seasons=True)
        season_id = resolver.resolve_season(
            season_name="2025/2026",
            competition_id=1,
            start_date=date(2025, 8, 1),
        )
        assert season_id == 20


# ── Row resolution ────────────────────────────────────────


class TestRowResolution:
    @patch(
        "src.importers.resolver.EntityResolver.resolve_team",
        side_effect=lambda n: {"Arsenal": 1, "Chelsea": 2}.get(n),
    )
    @patch(
        "src.importers.resolver.EntityResolver.resolve_competition",
        return_value=5,
    )
    @patch(
        "src.importers.resolver.EntityResolver.resolve_season",
        return_value=10,
    )
    def test_resolve_row(self, mock_season, mock_comp, mock_team) -> None:
        """A parsed row gets all FK IDs resolved."""
        resolver = EntityResolver()
        row = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league": "E0",
            "season": "2024/2025",
            "match_date": date(2024, 1, 7),
        }
        result = resolver.resolve_row(row)

        assert result["team_home_id"] == 1
        assert result["team_away_id"] == 2
        assert result["competition_id"] == 5
        assert result["season_id"] == 10

    @patch(
        "src.importers.resolver.EntityResolver.resolve_team",
        side_effect=lambda n: {"Arsenal": 1, "Chelsea": 2}.get(n),
    )
    def test_resolve_rows_batch(self, mock_team) -> None:
        """Batch resolution returns same number of rows with FK IDs."""
        resolver = EntityResolver()
        rows = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "match_date": date(2024, 1, 7)},
            {"home_team": "Arsenal", "away_team": "Chelsea", "match_date": date(2024, 2, 1)},
        ]
        results = resolver.resolve_rows(rows)
        assert len(results) == 2
        for r in results:
            assert "team_home_id" in r
            assert r["team_home_id"] == 1
            assert r["team_away_id"] == 2


# ── Cache management ──────────────────────────────────────


class TestCache:
    def test_clear_cache(self) -> None:
        resolver = EntityResolver()
        resolver._team_cache["test"] = 1
        resolver._competition_cache["test"] = 2
        resolver._season_cache["test"] = 3

        resolver.clear_cache()

        assert len(resolver._team_cache) == 0
        assert len(resolver._competition_cache) == 0
        assert len(resolver._season_cache) == 0

    def test_cache_stats(self) -> None:
        resolver = EntityResolver()
        resolver._team_cache["a"] = 1
        resolver._competition_cache["b"] = 2
        resolver._season_cache["c"] = 3

        stats = resolver.cache_stats
        assert stats["teams"] == 1
        assert stats["competitions"] == 1
        assert stats["seasons"] == 1
