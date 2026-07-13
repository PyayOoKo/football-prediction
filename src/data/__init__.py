"""
Data package — data processing and feature engineering.

Modules
-------
loader
    Loads data from CSV, database, or API sources.
preprocessing
    Cleaning, normalisation, and imputation pipelines.
feature_engineering
    Rolling averages, form features, Elo ratings, etc.
cleaners
    Source-specific data cleaning functions.
"""

from src.data.cleaners import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor

__all__ = [
    "DataCleaner",
    "DataLoader",
    "DataPreprocessor",
    "FeatureEngineer",
]
