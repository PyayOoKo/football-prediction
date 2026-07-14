"""
Odds model — bookmaker odds for match outcomes.

Separated from the Match table because:
1. Multiple bookmakers provide odds for the same match
2. Opening, closing, and live odds exist per bookmaker
3. Historical odds data is essential for value betting backtests

The ``source`` column identifies the bookmaker or data provider
(e.g. Bet365, Pinnacle, Bwin, averages like BbAvH).

The ``timestamp`` column tracks when the odds were recorded,
enabling opening-to-closing line movement analysis.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
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


class Odds(Base):
    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="Bookmaker or 'consensus'"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        comment="When the odds were recorded",
    )

    # ── Odds (decimal) ─────────────────────────────────
    odds_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_away: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Derived ────────────────────────────────────────
    implied_prob_home: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="1 / odds_home (no margin adjustment)"
    )
    implied_prob_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_prob_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    margin: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Bookmaker margin = sum(1/odds) - 1, as percentage",
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "match_id", "source", "timestamp",
            name="uq_odds_match_source_time",
        ),
        CheckConstraint(
            "odds_home IS NULL OR odds_home > 1.0",
            name="ck_odds_home_positive",
        ),
        CheckConstraint(
            "odds_draw IS NULL OR odds_draw > 1.0",
            name="ck_odds_draw_positive",
        ),
        CheckConstraint(
            "odds_away IS NULL OR odds_away > 1.0",
            name="ck_odds_away_positive",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="odds")

    def __repr__(self) -> str:
        return (
            f"<Odds(match_id={self.match_id}, source='{self.source}')>"
        )
