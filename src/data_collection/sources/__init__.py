"""Data source adapters for football match data."""

from __future__ import annotations

from src.data_collection.sources import (
    referee_stats,
    statsbomb_open,
    transfermarkt,
    transfers,
    weather_api,
    worldcup,
)

__all__ = [
    "worldcup",
    "transfermarkt",
    "transfers",
    "weather_api",
    "referee_stats",
    "statsbomb_open",
]
