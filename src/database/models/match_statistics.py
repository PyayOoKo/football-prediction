"""
Match Statistics model — granular in-play stats.

Stored 1:1 with matches for efficient JOIN perf.
Separated from the Match table because stats are often loaded
lazily (only when computing advanced features), keeping the
main matches table lean for fast scans.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.database.models.match import Match


class MatchStatistics(Base):
    __tablename__ = "match_statistics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── 1:1 link ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), unique=True, nullable=False, index=True
    )

    # ── Home stats ─────────────────────────────────────
    home_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_possession: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Percentage 0-100"
    )
    home_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_offsides: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_inside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_outside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Away stats ─────────────────────────────────────
    away_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_possession: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Percentage 0-100"
    )
    away_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_offsides: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_inside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_outside_box: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(
        Match, back_populates="statistics"
    )

    def __repr__(self) -> str:
        return f"<MatchStatistics(match_id={self.match_id})>"
