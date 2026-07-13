"""
Team Elo History — Elo rating snapshot per match.

Records each team's Elo rating before and after every match.
This enables time-series analysis of team strength, visualising
rating trajectories, and computing rating-derived features.

Only one row per (team_id, match_id) — computed after the
match outcome is known.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.database.models.match import Match
from src.database.models.team import Team


class TeamEloHistory(Base):
    __tablename__ = "team_elo_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    side: Mapped[str] = mapped_column(
        String(4), nullable=False, comment="home or away"
    )

    # ── Elo values ─────────────────────────────────────
    elo_before: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Elo before the match"
    )
    elo_after: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Elo after the match"
    )
    elo_change: Mapped[float] = mapped_column(
        Float, nullable=False, comment="elo_after - elo_before"
    )
    k_factor: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="K-factor used for this match"
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "team_id", "match_id",
            name="uq_team_elo_match"
        ),
        CheckConstraint(
            side.in_(["home", "away"]),
            name="ck_team_elo_side",
        ),
    )

    # ── Relationships ──────────────────────────────────
    team: Mapped[Team] = relationship(Team)
    match: Mapped[Match] = relationship(Match)

    def __repr__(self) -> str:
        return (
            f"<TeamEloHistory(team_id={self.team_id}, "
            f"elo_before={self.elo_before:.1f})>"
        )
