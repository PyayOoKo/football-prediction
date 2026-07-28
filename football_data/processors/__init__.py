"""
Data processors — cleaning, validation, and team name normalisation.

Each processor takes raw collected data and returns cleaned,
validated data ready for database insertion.
"""

from football_data.processors.clean import DataCleaner
from football_data.processors.validation import DataValidator

__all__ = ["DataCleaner", "DataValidator"]
