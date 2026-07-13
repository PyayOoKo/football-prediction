"""
Team Form — pre-computed rolling form metrics per match per team.

Avoids recomputing rolling averages on every query. Updated
after each match. Stores:
- Points per game over the last N matches
- Goals scored/conceded averages
- Win/draw/loss streaks
- Clean sheet and BTTS rates

Indexed by (team_id, match_id) for fast lookup when building
feature matrices.
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


class TeamForm(Base):
    __tablename__ = "team_form"

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

    # ── Rolling form (last 5 matches before this match) ─
    last_5_ppg: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Points per game (last 5)"
    )
    last_5_goals_scored: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Avg goals scored (last 5)"
    )
    last_5_goals_conceded: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Avg goals conceded (last 5)"
    )
    last_5_wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_5_draws: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_5_losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_5_clean_sheets: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_5_btts: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Both teams scored"
    )
    current_streak: Mapped[str | None] = mapped_column(
        String(8), nullable=True, comment="W, D, L (e.g. 'WWDLW')"
    )

    # ── Extended windows ───────────────────────────────
    last_10_ppg: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_10_goals_scored: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_10_goals_conceded: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_20_ppg: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_20_goals_scored: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_20_goals_conceded: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Season totals ──────────────────────────────────
    season_ppg: Mapped[float | None] = mapped_column(Float, nullable=True)
    season_matches_played: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "team_id", "match_id",
            name="uq_team_form_match"
        ),
    )

    # ── Relationships ──────────────────────────────────
    team: Mapped[Team] = relationship(Team)
    match: Mapped[Match] = relationship(Match)

    def __repr__(self) -> str:
        return (
            f"<TeamForm(team_id={self.team_id}, "
            f"match_id={self.match_id}, ppg={self.last_5_ppg})>"
        )
