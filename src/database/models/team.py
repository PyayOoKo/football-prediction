"""
Team ORM model — a football club or national team.

This is a foundational entity referenced by matches, lineups,
players, transfers, Elo history, and more.

Year founded is stored as an integer (e.g. 1886) for easy
age-at-match computation.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.country import Country
    from src.database.models.match import Match
    from src.database.models.player import Player
    from src.database.models.stadium import Stadium
    from src.database.models.transfer import Transfer


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    name: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    short_name: Mapped[str | None] = mapped_column(String(8), nullable=True)
    country_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("countries.id"), nullable=True, index=True
    )
    stadium_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("stadiums.id"), nullable=True, index=True
    )

    # ── Details ────────────────────────────────────────
    year_founded: Mapped[int | None] = mapped_column(Integer, nullable=True)
    logo_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    website: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ──────────────────────────────────
    country_obj: Mapped[Country | None] = relationship(
        "Country", back_populates="teams"
    )
    home_stadium: Mapped[Stadium | None] = relationship("Stadium")
    home_matches: Mapped[list[Match]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[list[Match]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    players: Mapped[list[Player]] = relationship(
        "Player", back_populates="current_team"
    )
    transfers_in: Mapped[list[Transfer]] = relationship(
        "Transfer", foreign_keys="Transfer.to_team_id", back_populates="to_team"
    )
    transfers_out: Mapped[list[Transfer]] = relationship(
        "Transfer", foreign_keys="Transfer.from_team_id", back_populates="from_team"
    )

    def __repr__(self) -> str:
        return f"<Team(id={self.id}, name='{self.name}')>"
