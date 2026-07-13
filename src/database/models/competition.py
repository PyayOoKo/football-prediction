"""
Competition model — replaces the old League model.

Represents any organised football competition: domestic leagues
(Premier League, La Liga), cups (FA Cup, Champions League),
or international tournaments (World Cup, Euros).

Separating competitions from seasons allows a single competition
(e.g. Premier League) to span multiple seasons cleanly.

Permissions
-----------
``type`` values are constrained via a CHECK constraint:
- ``league`` — domestic round-robin league
- ``cup`` — knockout or group+knockout tournament
- ``playoff`` — promotion/relegation playoffs
- ``friendly`` — exhibition matches
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.country import Country
    from src.database.models.match import Match
    from src.database.models.season import Season


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    code: Mapped[str | None] = mapped_column(String(16), nullable=True, unique=True)
    type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="league"
    )

    # ── Hierarchy ──────────────────────────────────────
    country_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("countries.id"), nullable=True, index=True
    )
    level: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Tier level (1=top division)"
    )

    # ── Optional ───────────────────────────────────────
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    logo_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Constraints ────────────────────────────────────
    __table_args__ = (
        CheckConstraint(
            type.in_(["league", "cup", "playoff", "friendly"]),
            name="ck_competitions_type",
        ),
    )

    # ── Relationships ──────────────────────────────────
    country_obj: Mapped[Country | None] = relationship(
        "Country", back_populates="competitions"
    )
    seasons: Mapped[list[Season]] = relationship(
        "Season", back_populates="competition", cascade="all, delete-orphan"
    )
    matches: Mapped[list[Match]] = relationship(
        "Match", back_populates="competition"
    )

    def __repr__(self) -> str:
        return f"<Competition(id={self.id}, name='{self.name}', type='{self.type}')>"
