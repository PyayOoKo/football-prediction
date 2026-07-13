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
    ConfigurationError,
    DataNotFoundError,
    DatabaseError,
    FootballPredictionError,
    ModelNotFoundError,
    PredictionError,
    ScraperError,
    ValidationError,
)
from src.utils.helpers import timer

__all__ = [
    "DataNotFoundError",
    "FootballPredictionError",
    "ModelNotFoundError",
    "PredictionError",
    "ScraperError",
    "timer",
]
