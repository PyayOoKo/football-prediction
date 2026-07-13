"""
Lineup model — team formations and player lineups.

Stores the starting XI, formation, and substitutions for
each team in a match. The ``starting_xi`` and ``substitutes``
columns use PostgreSQL JSONB for flexible schema-less storage
since lineup data sources vary in what they provide.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.database.models.match import Match
from src.database.models.team import Team


class Lineup(Base):
    __tablename__ = "lineups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )

    # ── Formation & players ────────────────────────────
    formation: Mapped[str | None] = mapped_column(
        String(8), nullable=True, comment="e.g. 4-3-3, 4-4-2, 3-5-2"
    )
    starting_xi: Mapped[dict | None] = mapped_column(
        JSON, nullable=True,
        comment="List of {player_id, name, position, shirt_number}",
    )
    substitutes: Mapped[dict | None] = mapped_column(
        JSON, nullable=True,
        comment="List of {player_id, name, minute_on}",
    )
    substitutions_made: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=0
    )
    coach: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="Manager/head coach name"
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "match_id", "team_id",
            name="uq_lineups_match_team",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="lineups")
    team: Mapped[Team] = relationship(Team)

    def __repr__(self) -> str:
        return (
            f"<Lineup(match_id={self.match_id}, team_id={self.team_id}, "
            f"formation='{self.formation}')>"
        )
