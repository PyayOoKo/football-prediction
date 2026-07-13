"""
Data cleaners — source-specific data cleaning and normalisation.

Each function takes a raw DataFrame from a specific source
and returns a cleaned, standardised version.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Collection of data cleaning methods for different sources."""

    @staticmethod
    def football_data_co_uk(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame from football-data.co.uk.

        Handles column renaming, type coercion, and missing values.
        """
        logger.info("Cleaning football-data.co.uk data: %d rows", len(df))
        # TODO: Implement actual cleaning logic
        return df

    @staticmethod
    def football_data_org(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame from football-data.org API."""
        logger.info("Cleaning football-data.org data: %d rows", len(df))
        # TODO: Implement actual cleaning logic
        return df

    @staticmethod
    def transfermarkt(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame scraped from Transfermarkt."""
        logger.info("Cleaning Transfermarkt data: %d rows", len(df))
        # TODO: Implement actual cleaning logic
        return df
