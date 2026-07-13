"""
Tests for repository pattern — CRUD and domain-specific queries.
"""

from __future__ import annotations

from datetime import date

from src.database.models import Match, Team
from src.database.repositories import MatchRepository, TeamRepository


class TestTeamRepository:
    """Verify TeamRepository CRUD operations."""

    def test_add_and_get_by_name(self, db_session) -> None:
        repo = TeamRepository(db_session)
        team = repo.get_or_create(name="Manchester City", short_name="MCI")
        assert team.id is not None

        found = repo.get_by_name("Manchester City")
        assert found is not None
        assert found.short_name == "MCI"

    def test_get_or_create_returns_existing(self, db_session) -> None:
        repo = TeamRepository(db_session)
        team1 = repo.get_or_create(name="Arsenal")
        team2 = repo.get_or_create(name="Arsenal")
        assert team1.id == team2.id

    def test_search_by_name(self, db_session) -> None:
        repo = TeamRepository(db_session)
        repo.get_or_create(name="Manchester United")
        repo.get_or_create(name="Manchester City")
        repo.get_or_create(name="Liverpool")

        results = repo.search_by_name("Manchester")
        assert len(results) == 2


class TestMatchRepository:
    """Verify MatchRepository query methods."""

    def test_get_upcoming(self, db_session, sample_team, other_team) -> None:
        repo = MatchRepository(db_session)
        match = Match(
            home_team_id=sample_team.id,
            away_team_id=other_team.id,
            match_date=date.today(),
        )
        db_session.add(match)
        db_session.flush()

        upcoming = repo.get_upcoming()
        assert len(upcoming) >= 1
