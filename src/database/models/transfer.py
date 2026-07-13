"""
Transfer model — player transfers between teams.

Tracks the movement of players between clubs including
transfer fee, date, and optional loan details.
Useful for computing team strength changes over time.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
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
from src.database.models.player import Player
from src.database.models.team import Team


class Transfer(Base):
    __tablename__ = "transfers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Identity ───────────────────────────────────────
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id"), nullable=False, index=True
    )
    from_team_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=True, index=True
    )
    to_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )

    # ── Transfer details ───────────────────────────────
    transfer_date: Mapped[date] = mapped_column(Date, nullable=False)
    transfer_fee_eur: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Transfer fee in EUR (NULL = undisclosed)"
    )
    is_loan: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    loan_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    contract_length_months: Mapped[int | None] = mapped_column(Integer, nullable=True)

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
            "from_team_id IS NULL OR from_team_id != to_team_id",
            name="ck_transfers_different_teams",
        ),
        CheckConstraint(
            "transfer_fee_eur IS NULL OR transfer_fee_eur >= 0",
            name="ck_transfers_fee_non_negative",
        ),
    )

    # ── Relationships ──────────────────────────────────
    player: Mapped[Player] = relationship(Player, back_populates="transfers")
    from_team: Mapped[Team | None] = relationship(
        Team, foreign_keys=[from_team_id], back_populates="transfers_out"
    )
    to_team: Mapped[Team] = relationship(
        Team, foreign_keys=[to_team_id], back_populates="transfers_in"
    )

    def __repr__(self) -> str:
        return (
            f"<Transfer(id={self.id}, player_id={self.player_id}, "
            f"from={self.from_team_id} -> to={self.to_team_id})>"
        )
