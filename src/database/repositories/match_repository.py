"""
Match repository — domain-specific queries for matches.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import and_, or_, select

from src.database.models.match import Match
from src.database.models.team import Team
from src.database.repositories.base import BaseRepository


class MatchRepository(BaseRepository[Match]):
    """Repository for ``Match`` entities with football-specific queries."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Match)

    def get_by_date_range(self, start: date, end: date, limit: int = 10000) -> list[Match]:
        stmt = (
            select(Match)
            .where(and_(Match.match_date >= start, Match.match_date <= end))
            .order_by(Match.match_date)
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())

    def get_upcoming(self, limit: int = 10) -> list[Match]:
        today = date.today()
        stmt = (
            select(Match)
            .where(Match.match_date >= today, Match.home_goals.is_(None))
            .order_by(Match.match_date)
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())

    def get_recent(self, limit: int = 10) -> list[Match]:
        stmt = (
            select(Match)
            .where(Match.home_goals.isnot(None))
            .order_by(Match.match_date.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())

    def get_by_team(self, team_id: int, limit: int = 50) -> list[Match]:
        stmt = (
            select(Match)
            .where(
                or_(Match.home_team_id == team_id, Match.away_team_id == team_id)
            )
            .order_by(Match.match_date.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())

    def get_by_competition(self, competition_id: int, limit: int = 100) -> list[Match]:
        stmt = (
            select(Match)
            .where(Match.competition_id == competition_id)
            .order_by(Match.match_date.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())

    def get_team_vs_team(
        self, home_team_id: int, away_team_id: int, limit: int = 10
    ) -> list[Match]:
        stmt = (
            select(Match)
            .where(
                and_(
                    Match.home_team_id == home_team_id,
                    Match.away_team_id == away_team_id,
                )
            )
            .order_by(Match.match_date.desc())
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())
