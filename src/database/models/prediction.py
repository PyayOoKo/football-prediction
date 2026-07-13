"""
Prediction ORM model.

Stores model output for each match: predicted probabilities,
confidence scores, and metadata about the model version used.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.database.models.match import Match


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Foreign keys ───────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )

    # ── Model output ───────────────────────────────────
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    prob_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_result: Mapped[str | None] = mapped_column(
        String(4), nullable=True  # "H", "D", "A"
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Value betting ──────────────────────────────────
    expected_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    kelly_stake: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Metadata ───────────────────────────────────────
    features_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, match_id={self.match_id}, "
            f"model='{self.model_name}')>"
        )
