"""
Closing Line Value (CLV) — a key betting performance metric.

CLV measures how the opening odds compare to the closing odds:
    CLV = (closing_price - opening_price) / opening_price

A positive CLV means the market moved toward your bet (you
beat the closing line), which is a strong indicator of
predictive skill vs luck.

Stored per match so we can aggregate CLV across a betting
strategy over time.
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


class ClosingLineValue(Base):
    __tablename__ = "closing_line_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    bookmaker: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="Bookmaker source"
    )
    outcome: Mapped[str] = mapped_column(
        String(4), nullable=False, comment="H, D, A"
    )

    # ── Prices ─────────────────────────────────────────
    opening_price: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Decimal odds at market open"
    )
    closing_price: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Decimal odds at market close (kickoff)"
    )
    clv: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="(closing_price - opening_price) / opening_price"
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "match_id", "bookmaker", "outcome",
            name="uq_clv_match_bookmaker_outcome",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="closing_line_values")

    def __repr__(self) -> str:
        return (
            f"<ClosingLineValue(match_id={self.match_id}, "
            f"clv={self.clv:.4f})>"
        )
