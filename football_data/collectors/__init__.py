"""
Collectors for the football data pipeline.

Each collector handles a specific data source and returns
normalised dicts ready for database insertion.

Sources
-------
- ``FootballDataCollector`` — football-data.co.uk (results + odds)
- ``FBrefParser`` — FBref (match results only, no odds, manual-save mode)
- ``WeatherCollector`` — Open-Meteo (weather data)
"""

from football_data.collectors.football_data import FootballDataCollector
from football_data.collectors.weather import WeatherCollector
from football_data.collectors.fbref import FBrefParser

__all__ = [
    "FootballDataCollector",
    "FBrefParser",
    "WeatherCollector",
]
