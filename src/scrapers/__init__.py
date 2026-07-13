"""
Scrapers package — data collection from external sources.

Modules
-------
base
    Abstract base class defining the scraper interface.
football_data_co_uk
    Fetches match results and odds from football-data.co.uk.
football_data_org
    Fetches data from football-data.org API.
transfermarkt
    Scrapes player and team data from Transfermarkt.
worldcup
    Collects World Cup historical data.
"""

from src.scrapers.base import BaseScraper, ScrapeResult

__all__ = [
    "BaseScraper",
    "ScrapeResult",
]
