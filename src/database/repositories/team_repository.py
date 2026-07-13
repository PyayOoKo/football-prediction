"""
Team repository — domain-specific queries for teams.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from src.database.models.team import Team
from src.database.repositories.base import BaseRepository


class TeamRepository(BaseRepository[Team]):
    """Repository for ``Team`` entities."""

    def __init__(self, session: Any) -> None:
        super().__init__(session, Team)

    def get_by_name(self, name: str) -> Team | None:
        stmt = select(Team).where(Team.name == name).limit(1)
        return self._session.scalars(stmt).first()

    def search_by_name(self, query: str, limit: int = 10) -> list[Team]:
        stmt = (
            select(Team)
            .where(Team.name.ilike(f"%{query}%"))
            .order_by(Team.name)
            .limit(limit)
        )
        return list(self._session.scalars(stmt).all())

    def get_or_create(self, name: str, **defaults: Any) -> Team:
        existing = self.get_by_name(name)
        if existing is not None:
            return existing
        team = Team(name=name, **defaults)
        return self.add(team)
