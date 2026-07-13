"""
Tests for SQLAlchemy ORM models — schema, relationships, and constraints.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from src.database.models import Competition, Match, Prediction, Season, Team


class TestTeamModel:
    """Verify Team model fields and constraints."""

    def test_create_team(self, db_session) -> None:
        team = Team(name="Liverpool", short_name="LIV")
        db_session.add(team)
        db_session.flush()

        assert team.id is not None
        assert team.name == "Liverpool"

    def test_team_name_unique_constraint(self, db_session) -> None:
        team1 = Team(name="Chelsea")
        db_session.add(team1)
        db_session.flush()

        team2 = Team(name="Chelsea")
        db_session.add(team2)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_team_table_exists(self, db_session) -> None:
        inspector = inspect(db_session.bind)
        tables = inspector.get_table_names()
        assert "teams" in tables


class TestMatchModel:
    """Verify Match model fields and relationships."""

    def test_create_match(self, db_session, sample_team, other_team, sample_league) -> None:
        match = Match(
            home_team_id=sample_team.id,
            away_team_id=other_team.id,
            competition_id=sample_league.id,
            match_date=date(2024, 1, 1),
            home_goals=2,
            away_goals=0,
            result="H",
        )
        db_session.add(match)
        db_session.flush()

        assert match.id is not None
        assert match.home_team_id == sample_team.id

    def test_match_table_exists(self, db_session) -> None:
        inspector = inspect(db_session.bind)
        tables = inspector.get_table_names()
        assert "matches" in tables
