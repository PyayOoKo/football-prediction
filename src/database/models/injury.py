"""
Injury model — player injuries and recovery.

Tracks injury type, severity, and expected return date.
Useful for computing a team's "injury burden" feature
and adjusting player availability predictions.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.database.models.player import Player


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id"), nullable=False, index=True
    )

    # ── Injury details ─────────────────────────────────
    injury_type: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="e.g. Hamstring, ACL, Ankle"
    )
    severity: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="minor, moderate, severe, career-ending"
    )
    injury_date: Mapped[date] = mapped_column(Date, nullable=False)
    expected_return: Mapped[date | None] = mapped_column(Date, nullable=True)
    actual_return: Mapped[date | None] = mapped_column(Date, nullable=True)
    missed_matches: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ──────────────────────────────────
    player: Mapped[Player] = relationship(Player, back_populates="injuries")

    def __repr__(self) -> str:
        return (
            f"<Injury(id={self.id}, player_id={self.player_id}, "
            f"type='{self.injury_type}')>"
        )
