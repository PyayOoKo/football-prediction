"""
Season model — temporal grouping for competitions.

A season belongs to one competition and aggregates all matches
played in that time period. Examples:
- Premier League 2024/2025
- 2022 FIFA World Cup
- La Liga 2023/2024

This enables hierarchical queries like "all matches in the
2024/2025 Premier League season" without string parsing.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.competition import Competition
    from src.database.models.match import Match


class Season(Base):
    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    name: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="e.g. '2024/2025', '2022'"
    )
    competition_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("competitions.id"), nullable=False, index=True
    )

    # ── Date range ─────────────────────────────────────
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)

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
            "competition_id", "name",
            name="uq_seasons_competition_name"
        ),
        CheckConstraint(
            "start_date <= end_date",
            name="ck_seasons_date_range",
        ),
    )

    # ── Relationships ──────────────────────────────────
    competition: Mapped[Competition] = relationship(
        "Competition", back_populates="seasons"
    )
    matches: Mapped[list[Match]] = relationship(
        "Match", back_populates="season"
    )

    def __repr__(self) -> str:
        return (
            f"<Season(id={self.id}, name='{self.name}', "
            f"competition_id={self.competition_id})>"
        )
