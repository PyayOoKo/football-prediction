"""
Tests for the ``SoftDeleteMixin`` — soft delete lifecycle and query behaviour.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, select

from src.database.models import Team


class TestSoftDeleteMixin:
    """Verify SoftDeleteMixin column, properties, and methods."""

    def test_deleted_at_defaults_to_none(self, db_session) -> None:
        """Newly created records have ``deleted_at IS NULL``."""
        team = Team(name="Test Team", short_name="TST")
        db_session.add(team)
        db_session.flush()

        assert team.deleted_at is None

    def test_is_deleted_false_for_active(self, db_session) -> None:
        """``is_deleted`` is ``False`` for records that have not been soft-deleted."""
        team = Team(name="Active Team", short_name="ACT")
        db_session.add(team)
        db_session.flush()

        assert team.is_deleted is False

    def test_soft_delete_sets_deleted_at(self, db_session) -> None:
        """Calling ``soft_delete()`` sets ``deleted_at`` to a UTC datetime."""
        team = Team(name="Delete Me", short_name="DEL")
        db_session.add(team)
        db_session.flush()
        original_id = team.id

        before = datetime.now(timezone.utc)
        team.soft_delete()
        after = datetime.now(timezone.utc)

        assert team.deleted_at is not None
        assert before <= team.deleted_at.replace(tzinfo=timezone.utc) <= after

    def test_is_deleted_true_after_soft_delete(self, db_session) -> None:
        """``is_deleted`` is ``True`` after calling ``soft_delete()``."""
        team = Team(name="Gone Team", short_name="GON")
        db_session.add(team)
        db_session.flush()

        team.soft_delete()

        assert team.is_deleted is True

    def test_soft_delete_returns_self(self, db_session) -> None:
        """``soft_delete()`` returns ``self`` so calls can be chained."""
        team = Team(name="Chain Test", short_name="CHN")
        db_session.add(team)
        db_session.flush()

        result = team.soft_delete()

        assert result is team

    def test_record_still_exists_after_soft_delete(self, db_session) -> None:
        """Soft-deleted records are NOT removed from the database."""
        team = Team(name="Still Here", short_name="HRH")
        db_session.add(team)
        db_session.flush()
        original_id = team.id

        team.soft_delete()
        db_session.flush()  # commit the soft delete

        # Should still be queryable (no hard delete)
        found = db_session.get(Team, original_id)
        assert found is not None
        assert found.id == original_id
        assert found.is_deleted is True

    def test_soft_delete_multiple_records(self, db_session) -> None:
        """Multiple records can be soft-deleted independently."""
        teams = [
            Team(name="Team A", short_name="TA"),
            Team(name="Team B", short_name="TB"),
            Team(name="Team C", short_name="TC"),
        ]
        db_session.add_all(teams)
        db_session.flush()

        # Soft-delete only Team B
        teams[1].soft_delete()
        db_session.flush()

        # Refresh all from DB
        for t in teams:
            db_session.refresh(t)

        assert teams[0].is_deleted is False
        assert teams[1].is_deleted is True
        assert teams[2].is_deleted is False

    def test_column_exists_in_table(self, db_session) -> None:
        """The ``deleted_at`` column is present in the database schema."""
        inspector = inspect(db_session.bind)
        columns = {c["name"] for c in inspector.get_columns("teams")}
        assert "deleted_at" in columns

    def test_deleted_at_nullable(self, db_session) -> None:
        """The ``deleted_at`` column is nullable in the schema."""
        inspector = inspect(db_session.bind)
        team_cols = inspector.get_columns("teams")
        deleted_at_col = next(c for c in team_cols if c["name"] == "deleted_at")
        assert deleted_at_col["nullable"] is True


class TestSoftDeleteWithModels:
    """Verify soft delete works across Match, Prediction, and Team models."""

    def test_match_model_has_mixin(self) -> None:
        """Match model inherits SoftDeleteMixin (has deleted_at)."""
        from src.database.models import Match

        assert hasattr(Match, "deleted_at")

    def test_prediction_model_has_mixin(self) -> None:
        """Prediction model inherits SoftDeleteMixin (has deleted_at)."""
        from src.database.models import Prediction

        assert hasattr(Prediction, "deleted_at")

    def test_team_model_has_mixin(self) -> None:
        """Team model inherits SoftDeleteMixin (has deleted_at)."""
        assert hasattr(Team, "deleted_at")
