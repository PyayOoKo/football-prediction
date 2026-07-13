"""
Weather model — match-day weather conditions.

Weather is known to affect football outcomes (scoring rates,
playing styles, home advantage). Stored 1:1 with matches
for optional inclusion in feature engineering.

Only populated for outdoor stadiums and when historical
weather data is available.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import Base
from src.database.models.match import Match


class Weather(Base):
    __tablename__ = "weather"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── 1:1 link ───────────────────────────────────────
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), unique=True, nullable=False, index=True
    )

    # ── Conditions ─────────────────────────────────────
    temperature_celsius: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Temperature in Celsius"
    )
    humidity_pct: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Humidity percentage 0-100"
    )
    wind_speed_kmh: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Wind speed in km/h"
    )
    precipitation_mm: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Precipitation in mm"
    )
    condition: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
        comment="Clear, Cloudy, Rain, Snow, Fog, etc."
    )
    pitch_condition: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
        comment="Dry, Wet, Waterlogged, Frozen, etc."
    )

    # ── Metadata ───────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Constraints ────────────────────────────────────
    __table_args__ = (
        CheckConstraint(
            "temperature_celsius IS NULL OR "
            "(temperature_celsius >= -30 AND temperature_celsius <= 60)",
            name="ck_weather_temperature",
        ),
        CheckConstraint(
            "humidity_pct IS NULL OR "
            "(humidity_pct >= 0 AND humidity_pct <= 100)",
            name="ck_weather_humidity",
        ),
    )

    # ── Relationships ──────────────────────────────────
    match: Mapped[Match] = relationship(Match, back_populates="weather")

    def __repr__(self) -> str:
        return f"<Weather(match_id={self.match_id}, temp={self.temperature_celsius})>"
