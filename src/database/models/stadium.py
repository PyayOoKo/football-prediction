"""
Stadium model — venue information for matches.

Includes geographic location (city, country), capacity,
pitch surface type, and whether it has a roof.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.country import Country
    from src.database.models.match import Match


class Stadium(Base):
    __tablename__ = "stadiums"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    country_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("countries.id"), nullable=True, index=True
    )

    # ── Physical ───────────────────────────────────────
    capacity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    surface: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="Grass, artificial, hybrid"
    )
    roofed: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ──────────────────────────────────
    country_obj: Mapped[Country | None] = relationship(
        "Country", back_populates="stadiums"
    )
    home_matches: Mapped[list[Match]] = relationship(
        "Match", back_populates="stadium"
    )

    def __repr__(self) -> str:
        return f"<Stadium(id={self.id}, name='{self.name}')>"
