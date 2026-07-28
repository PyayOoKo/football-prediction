"""
Utilities package — cross-cutting helpers and base classes.

Modules
-------
exceptions
    Custom exception hierarchy for domain-specific errors.
helpers
    General-purpose helper functions (date parsing, file I/O, etc.).
validators
    Input validation and data quality checks.

Note
----
Logging configuration lives in ``src.config.logging``, not here.
"""

from src.utils.exceptions import (
    BettingError,
    CacheError,
    CalibrationError,
    ConfigurationError,
    DatabaseError,
    DataIntegrityError,
    DataNotFoundError,
    DataSourceError,
    ExperimentError,
    FeatureEngineeringError,
    FootballPredictionError,
    ModelIncompatibleError,
    ModelNotFoundError,
    ModelNotTrainedError,
    OddsError,
    PipelineError,
    PredictionError,
    ScraperAuthError,
    ScraperError,
    ScraperRateLimitError,
    StakingError,
    TrainingError,
    ValidationError,
)
from src.utils.helpers import timer
from src.utils.logging_utils import get_logger

__all__ = [
    "get_logger",
    "FootballPredictionError",
    "DataNotFoundError",
    "DatabaseError",
    "DataIntegrityError",
    "DataSourceError",
    "ModelNotFoundError",
    "ModelNotTrainedError",
    "ModelIncompatibleError",
    "TrainingError",
    "PredictionError",
    "CalibrationError",
    "FeatureEngineeringError",
    "BettingError",
    "StakingError",
    "OddsError",
    "ScraperError",
    "ScraperRateLimitError",
    "ScraperAuthError",
    "ConfigurationError",
    "ValidationError",
    "PipelineError",
    "CacheError",
    "ExperimentError",
    "timer",
]
