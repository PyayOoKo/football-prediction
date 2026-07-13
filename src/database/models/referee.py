"""
Referee model — officials who oversee matches.

Tracks the referee's name, nationality, and which matches
they have officiated.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.country import Country
    from src.database.models.match import Match


class Referee(Base):
    __tablename__ = "referees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    country_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("countries.id"), nullable=True, index=True
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ──────────────────────────────────
    country_obj: Mapped[Country | None] = relationship(
        "Country", back_populates="referees"
    )
    matches: Mapped[list[Match]] = relationship(
        "Match", back_populates="referee"
    )

    def __repr__(self) -> str:
        return f"<Referee(id={self.id}, name='{self.full_name}')>"
