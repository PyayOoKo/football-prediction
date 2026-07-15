"""Data source adapters for football match data."""

from __future__ import annotations

from src.data_collection.sources import worldcup, transfermarkt
from src.data_collection.sources import transfers, weather_api, referee_stats, statsbomb_open

__all__ = [
    "worldcup",
    "transfermarkt",
    "transfers",
    "weather_api",
    "referee_stats",
    "statsbomb_open",
]
