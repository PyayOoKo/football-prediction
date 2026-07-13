"""
Player model — individual footballer.

Tracks career metadata: position, date of birth, nationality,
current team, and market value. Referenced by player match
stats, injuries, and transfers.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.country import Country
    from src.database.models.injury import Injury
    from src.database.models.player_match_stats import PlayerMatchStats
    from src.database.models.team import Team
    from src.database.models.transfer import Transfer


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Personal ───────────────────────────────────────
    full_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    country_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("countries.id"), nullable=True, index=True
    )

    # ── Football ───────────────────────────────────────
    position: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
        comment="GK, CB, LB, RB, CM, LM, RM, CAM, LW, RW, CF, ST"
    )
    preferred_foot: Mapped[str | None] = mapped_column(
        String(8), nullable=True, comment="left, right, both"
    )
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_kg: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Current team ───────────────────────────────────
    current_team_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=True, index=True
    )
    shirt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Market value ───────────────────────────────────
    market_value_eur: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Estimated market value in EUR"
    )

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
            position.in_([
                "GK", "CB", "LB", "RB", "LWB", "RWB",
                "CM", "CDM", "CAM", "LM", "RM",
                "LW", "RW", "CF", "ST", "SS",
            ]),
            name="ck_players_position",
        ),
        CheckConstraint(
            preferred_foot.in_(["left", "right", "both"]),
            name="ck_players_foot",
        ),
    )

    # ── Relationships ──────────────────────────────────
    country_obj: Mapped[Country | None] = relationship(
        "Country", back_populates="players"
    )
    current_team: Mapped[Team | None] = relationship(
        "Team", back_populates="players"
    )
    match_stats: Mapped[list[PlayerMatchStats]] = relationship(
        "PlayerMatchStats", back_populates="player", cascade="all, delete-orphan"
    )
    injuries: Mapped[list[Injury]] = relationship(
        "Injury", back_populates="player", cascade="all, delete-orphan"
    )
    transfers: Mapped[list[Transfer]] = relationship(
        "Transfer", back_populates="player", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Player(id={self.id}, name='{self.full_name}')>"
