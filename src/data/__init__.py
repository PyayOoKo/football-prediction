"""
Data collection, preprocessing, loading, and odds processing package.

Provides:
    OddsCollector — multi-source odds aggregation with arbitrage detection
    find_best_odds — one-shot convenience for best odds lookup
    detect_arbitrage — one-shot convenience for arbitrage detection
    DataPreprocessor — configurable data cleaning and transformation pipeline
    DataLoader — loads match data from CSV, Parquet, or database sources
    DataCleaner — source-specific data cleaning (football-data.co.uk, etc.)
"""

from .odds_collector import OddsCollector, find_best_odds, detect_arbitrage
from .preprocessing import DataPreprocessor
from .loader import DataLoader
from .cleaners import DataCleaner

__all__ = [
    "OddsCollector",
    "find_best_odds",
    "detect_arbitrage",
    "DataPreprocessor",
    "DataLoader",
    "DataCleaner",
]
