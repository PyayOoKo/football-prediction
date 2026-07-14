"""
Player Match Statistics — individual player performance in a match.

Stores all per-player stats for each match appearance: minutes
played, goals, assists, cards, and advanced metrics.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
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
from src.database.models.player import Player


class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False, index=True
    )
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id"), nullable=False, index=True
    )
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )

    # ── Performance ────────────────────────────────────
    minutes_played: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_starter: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=False
    )
    position: Mapped[str | None] = mapped_column(
        String(8), nullable=True, comment="Actual position played"
    )

    # ── Counting stats ─────────────────────────────────
    goals: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    assists: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    shots: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    passes: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    pass_accuracy: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Percentage 0-100"
    )
    tackles: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    interceptions: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    fouls_committed: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    fouls_drawn: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    yellow_card: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    red_card: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    saves: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=0,
        comment="Goalkeeper saves only"
    )

    # ── Advanced ───────────────────────────────────────
    rating: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Whoscored / FotMob / SofaScore rating (0-10)"
    )
    xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    xa: Mapped[float | None] = mapped_column(Float, nullable=True, comment="Expected assists")
    xg_chain: Mapped[float | None] = mapped_column(Float, nullable=True)
    xg_buildup: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "match_id", "player_id",
            name="uq_pms_match_player",
        ),
        CheckConstraint(
            "minutes_played IS NULL OR (minutes_played >= 0 AND minutes_played <= 120)",
            name="ck_pms_minutes",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match)
    player: Mapped[Player] = relationship(Player, back_populates="match_stats")

    def __repr__(self) -> str:
        return (
            f"<PlayerMatchStats(match_id={self.match_id}, "
            f"player_id={self.player_id})>"
        )
