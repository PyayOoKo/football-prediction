"""
Betting Result — actual bet outcomes and profit tracking.

Records every placed bet (real or simulated), its outcome,
and the resulting P&L. This is the source of truth for
evaluating betting strategy performance over time.

Each row represents one bet on one outcome of one match.
If a strategy bets on home, draw, AND away at different
stakes, that's three rows.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.database.models.match import Match


class BettingResult(Base):
    __tablename__ = "betting_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    strategy: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="Strategy name, e.g. 'kelly_0.25'"
    )

    # ── Bet details ────────────────────────────────────
    bookmaker: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="Bookmaker placed with"
    )
    bet_outcome: Mapped[str] = mapped_column(
        String(4), nullable=False, comment="H, D, A"
    )
    decimal_odds: Mapped[float] = mapped_column(Float, nullable=False)
    stake: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Stake amount in base currency"
    )

    # ── Result ─────────────────────────────────────────
    won: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, comment="True = won, False = lost"
    )
    profit: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Negative = loss, positive = profit"
    )
    roi_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Return on investment for this bet as percentage"
    )

    # ── Metadata ───────────────────────────────────────
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        CheckConstraint(
            bet_outcome.in_(["H", "D", "A"]),
            name="ck_betting_results_outcome",
        ),
        CheckConstraint(
            "stake > 0",
            name="ck_betting_results_stake_positive",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="betting_results")

    def __repr__(self) -> str:
        return (
            f"<BettingResult(match_id={self.match_id}, "
            f"outcome='{self.bet_outcome}', won={self.won})>"
        )
