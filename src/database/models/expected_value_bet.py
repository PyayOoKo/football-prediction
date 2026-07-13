"""
Expected Value (EV) Bet — value betting calculations.

For each match + bookmaker combination, computes the expected
value of betting on each outcome using the model's predicted
probabilities vs the bookmaker's implied probabilities.

This table is populated by the value betting pipeline and
enables backtesting EV-based betting strategies.
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


class ExpectedValueBet(Base):
    __tablename__ = "expected_value_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    bookmaker: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="Bookmaker source"
    )

    # ── Model probabilities ────────────────────────────
    model_prob_home: Mapped[float] = mapped_column(Float, nullable=False)
    model_prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    model_prob_away: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Bookmaker implied probabilities ────────────────
    book_prob_home: Mapped[float] = mapped_column(Float, nullable=False)
    book_prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    book_prob_away: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Expected value ─────────────────────────────────
    ev_home: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="model_prob_home / book_prob_home - 1"
    )
    ev_draw: Mapped[float] = mapped_column(Float, nullable=False)
    ev_away: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Kelly stake ────────────────────────────────────
    kelly_stake_home: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Full Kelly fraction for home bet"
    )
    kelly_stake_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    kelly_stake_away: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Best bet ──────────────────────────────────────
    recommended_bet: Mapped[str | None] = mapped_column(
        String(4), nullable=True, comment="H, D, A"
    )
    recommended_ev: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_kelly: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "match_id", "bookmaker",
            name="uq_ev_bet_match_bookmaker",
        ),
        CheckConstraint(
            "model_prob_home + model_prob_draw + model_prob_away BETWEEN 0.98 AND 1.02",
            name="ck_ev_bet_model_probs_sum",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="expected_value_bets")

    def __repr__(self) -> str:
        return (
            f"<ExpectedValueBet(match_id={self.match_id}, "
            f"bookmaker='{self.bookmaker}')>"
        )
