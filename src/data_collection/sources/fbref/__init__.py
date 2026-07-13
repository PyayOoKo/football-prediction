"""
FBref — football statistics scraper for fbref.com.

Collects team and player statistics from FBref across 7 categories:
team stats, match stats, shooting, passing, possession, defense,
and goalkeeping.

Usage
-----
::

    from src.data_collection.sources.fbref import FBrefScraper

    async with FBrefScraper() as scraper:
        # Team-level stats for Premier League 2024-25
        df = await scraper.get_team_stats("9", "2024-2025")

        # Squad shooting stats
        shooting = await scraper.get_squad_stats("9", "2024-2025", "shooting")

        # Match-level data for a specific match
        match = await scraper.get_match_stats("https://fbref.com/en/matches/...")

Features
--------
- Respects robots.txt with cached policy checks
- Polite rate limiting (configurable delay between requests)
- Exponential backoff retry with jitter via RetryWithBackoff
- LRU response caching with configurable TTL
- Async HTTP via httpx for concurrent page fetching
- HTML comment extraction for FBref's hidden tables
- Incremental updates via checkpoint files
- Resume interrupted downloads
"""

from __future__ import annotations

from src.data_collection.sources.fbref.client import FBrefClient
from src.data_collection.sources.fbref.models import (
    FBrefTable,
    SquadStats,
    MatchStats,
    PlayerStats,
    StatCategory,
)
from src.data_collection.sources.fbref.parser import FBrefTableParser
from src.data_collection.sources.fbref.robots import RobotsChecker
from src.data_collection.sources.fbref.scraper import FBrefScraper

__all__ = [
    "FBrefScraper",
    "FBrefClient",
    "FBrefTableParser",
    "RobotsChecker",
    "FBrefTable",
    "SquadStats",
    "MatchStats",
    "PlayerStats",
    "StatCategory",
]
