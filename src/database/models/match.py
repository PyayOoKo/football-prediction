"""
Match ORM model — the central entity in the prediction pipeline.

A match connects teams, competition, season, stadium, and referee.
Detailed statistics (shots, possession, xG) and odds live in
separate 1:1 or 1:N tables to keep this table lean for fast
scanning over millions of rows.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base

if TYPE_CHECKING:
    from src.database.models.betting_result import BettingResult
    from src.database.models.closing_line_value import ClosingLineValue
    from src.database.models.competition import Competition
    from src.database.models.expected_value_bet import ExpectedValueBet
    from src.database.models.lineup import Lineup
    from src.database.models.match_statistics import MatchStatistics
    from src.database.models.odds import Odds
    from src.database.models.prediction import Prediction
    from src.database.models.referee import Referee
    from src.database.models.season import Season
    from src.database.models.stadium import Stadium
    from src.database.models.team import Team
    from src.database.models.weather import Weather


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Foreign keys ───────────────────────────────────
    competition_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("competitions.id"), nullable=True, index=True
    )
    season_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("seasons.id"), nullable=True, index=True
    )
    home_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )
    away_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False, index=True
    )
    stadium_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("stadiums.id"), nullable=True
    )
    referee_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("referees.id"), nullable=True
    )

    # ── Match details ──────────────────────────────────
    match_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    round: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
        comment="e.g. 'Group A', 'Round 1', 'Semi-final'"
    )
    is_neutral_venue: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=False
    )
    attendance: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Outcome (nullable for upcoming matches) ────────
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped[str | None] = mapped_column(
        String(4), nullable=True,
        comment="H = home win, D = draw, A = away win"
    )
    duration: Mapped[str | None] = mapped_column(
        String(8), nullable=True, default="regular",
        comment="regular, extra_time, penalties"
    )

    # ── Status ─────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="scheduled",
        comment="scheduled, live, finished, postponed, cancelled, abandoned",
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # ── Constraints & indexes ──────────────────────────
    __table_args__ = (
        CheckConstraint(
            "home_team_id != away_team_id",
            name="ck_matches_different_teams",
        ),
        CheckConstraint(
            result.in_(["H", "D", "A"]),
            name="ck_matches_result",
        ),
        CheckConstraint(
            duration.in_(["regular", "extra_time", "penalties"]),
            name="ck_matches_duration",
        ),
        CheckConstraint(
            status.in_([
                "scheduled", "live", "finished",
                "postponed", "cancelled", "abandoned",
            ]),
            name="ck_matches_status",
        ),
        # Composite indexes for common query patterns
        Index(
            "ix_matches_comp_season_date",
            "competition_id", "season_id", "match_date",
        ),
        Index(
            "ix_matches_home_date",
            "home_team_id", "match_date",
        ),
        Index(
            "ix_matches_away_date",
            "away_team_id", "match_date",
        ),
    )

    # ── Relationships ──────────────────────────────────
    competition: Mapped[Competition | None] = relationship(
        "Competition", back_populates="matches"
    )
    season: Mapped[Season | None] = relationship(
        "Season", back_populates="matches"
    )
    home_team: Mapped[Team] = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_matches"
    )
    away_team: Mapped[Team] = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_matches"
    )
    stadium: Mapped[Stadium | None] = relationship(
        "Stadium", back_populates="home_matches"
    )
    referee: Mapped[Referee | None] = relationship(
        "Referee", back_populates="matches"
    )

    # Detail tables (1:1 or 1:N)
    statistics: Mapped[MatchStatistics | None] = relationship(
        "MatchStatistics", back_populates="match", uselist=False,
        cascade="all, delete-orphan"
    )
    odds: Mapped[list[Odds]] = relationship(
        "Odds", back_populates="match", cascade="all, delete-orphan"
    )
    weather: Mapped[Weather | None] = relationship(
        "Weather", back_populates="match", uselist=False,
        cascade="all, delete-orphan"
    )
    predictions: Mapped[list[Prediction]] = relationship(
        "Prediction", back_populates="match", cascade="all, delete-orphan"
    )
    lineups: Mapped[list[Lineup]] = relationship(
        "Lineup", back_populates="match", cascade="all, delete-orphan"
    )
    expected_value_bets: Mapped[list[ExpectedValueBet]] = relationship(
        "ExpectedValueBet", back_populates="match", cascade="all, delete-orphan"
    )
    closing_line_values: Mapped[list[ClosingLineValue]] = relationship(
        "ClosingLineValue", back_populates="match", cascade="all, delete-orphan"
    )
    betting_results: Mapped[list[BettingResult]] = relationship(
        "BettingResult", back_populates="match", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Match(id={self.id}, {self.home_team_id} vs {self.away_team_id}, "
            f"{self.match_date})>"
        )
