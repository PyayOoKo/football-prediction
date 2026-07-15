"""
ClosingOdds model — closing odds per match from multiple sources.

Stores the final (kick-off) market prices for each match across
multiple sources (Football-Data.co.uk, OddsPortal, BetExplorer)
including 1X2, BTTS, and Over/Under markets.

This enables CLV computation, market consensus, and backtesting
with realistic closing-line pricing.

Columns
-------
match_id  FK to matches table
source    Data source identifier (e.g. ``football-data``, ``oddsportal``)
timestamp When the odds were recorded (close to kick-off)

1X2 odds:
  odds_home, odds_draw, odds_away  — Closing decimal odds

BTTS odds:
  btts_yes, btts_no  — Both Teams To Score decimal odds

Over/Under odds:
  over25, under25  — Over/Under 2.5 goals decimal odds

Constraints
-----------
- Unique on (match_id, source) — one closing odds row per source per match
- All odds must be > 1.0 or NULL
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


class ClosingOdds(Base):
    __tablename__ = "closing_odds"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="e.g. football-data, oddsportal, betexplorer"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        comment="When the odds were recorded (close to kick-off)",
    )

    # ── 1X2 closing odds (decimal) ─────────────────────
    odds_home: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Closing decimal odds for home win"
    )
    odds_draw: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Closing decimal odds for draw"
    )
    odds_away: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Closing decimal odds for away win"
    )

    # ── BTTS closing odds (decimal) ────────────────────
    btts_yes: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Closing decimal odds for Both Teams To Score (Yes)"
    )
    btts_no: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Closing decimal odds for Both Teams To Score (No)"
    )

    # ── Over/Under closing odds (decimal) ─────────────
    over25: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Closing decimal odds for Over 2.5 goals"
    )
    under25: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Closing decimal odds for Under 2.5 goals"
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "match_id", "source",
            name="uq_closing_odds_match_source",
        ),
        CheckConstraint(
            "odds_home IS NULL OR odds_home > 1.0",
            name="ck_closing_odds_home_positive",
        ),
        CheckConstraint(
            "odds_draw IS NULL OR odds_draw > 1.0",
            name="ck_closing_odds_draw_positive",
        ),
        CheckConstraint(
            "odds_away IS NULL OR odds_away > 1.0",
            name="ck_closing_odds_away_positive",
        ),
        CheckConstraint(
            "btts_yes IS NULL OR btts_yes > 1.0",
            name="ck_closing_odds_btts_yes_positive",
        ),
        CheckConstraint(
            "btts_no IS NULL OR btts_no > 1.0",
            name="ck_closing_odds_btts_no_positive",
        ),
        CheckConstraint(
            "over25 IS NULL OR over25 > 1.0",
            name="ck_closing_odds_over25_positive",
        ),
        CheckConstraint(
            "under25 IS NULL OR under25 > 1.0",
            name="ck_closing_odds_under25_positive",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="closing_odds")

    def __repr__(self) -> str:
        return (
            f"<ClosingOdds(match_id={self.match_id}, source='{self.source}', "
            f"odds=({self.odds_home}/{self.odds_draw}/{self.odds_away}))>"
        )
