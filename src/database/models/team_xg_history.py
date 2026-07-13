"""
Team xG History — expected goals per match per team.

Separate from match statistics because xG data:
1. Comes from different providers (Opta, Understat, StatsBomb)
2. Needs its own source tracking
3. Is often computed/purchased separately from other stats
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
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


class TeamXgHistory(Base):
    __tablename__ = "team_xg_history"

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
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="opta",
        comment="Provider: opta, understat, statsbomb, manual"
    )

    # ── xG values ────────────────────────────────────
    xg: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Total expected goals"
    )
    xg_open_play: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="xG from open play only"
    )
    xg_set_piece: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="xG from set pieces only"
    )
    xg_penalty: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="xG from penalties only"
    )
    xg_first_half: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="xG in first half only"
    )
    xg_second_half: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="xG in second half only"
    )

    # ── Expected assists ───────────────────────────────
    xa: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Expected assists"
    )

    # ── Shots ──────────────────────────────────────────
    shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deep_completions: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="Passes completed into the box"
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "team_id", "match_id", "source",
            name="uq_team_xg_match_source",
        ),
    )

    # ── Relationships ──────────────────────────────────
    team: Mapped[Team] = relationship(Team)
    match: Mapped[Match] = relationship(Match)

    def __repr__(self) -> str:
        return (
            f"<TeamXgHistory(team_id={self.team_id}, match_id={self.match_id}, "
            f"xg={self.xg:.2f})>"
        )
