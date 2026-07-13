"""
Country model — ISO-coded country reference data.

Drives team nationality, competition regions, referee nationality,
player citizenship, and stadium location. Small, heavily referenced
table (many FKs point here) so every query JOINs to it efficiently.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.competition import Competition
    from src.database.models.player import Player
    from src.database.models.referee import Referee
    from src.database.models.stadium import Stadium
    from src.database.models.team import Team


class Country(Base):
    __tablename__ = "countries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    iso_alpha2: Mapped[str] = mapped_column(
        String(2), unique=True, nullable=False, index=True
    )
    iso_alpha3: Mapped[str] = mapped_column(
        String(3), unique=True, nullable=False, index=True
    )
    fifa_code: Mapped[str | None] = mapped_column(String(3), unique=True, nullable=True)
    continent: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ──────────────────────────────────
    teams: Mapped[list[Team]] = relationship("Team", back_populates="country_obj")
    competitions: Mapped[list[Competition]] = relationship(
        "Competition", back_populates="country_obj"
    )
    stadiums: Mapped[list[Stadium]] = relationship(
        "Stadium", back_populates="country_obj"
    )
    referees: Mapped[list[Referee]] = relationship(
        "Referee", back_populates="country_obj"
    )
    players: Mapped[list[Player]] = relationship(
        "Player", back_populates="country_obj"
    )

    def __repr__(self) -> str:
        return f"<Country(id={self.id}, name='{self.name}', iso='{self.iso_alpha2}')>"
