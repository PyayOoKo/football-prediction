"""
Custom exception hierarchy for the football prediction application.

All domain exceptions inherit from ``FootballPredictionError``,
which itself inherits from ``Exception``.
"""


class FootballPredictionError(Exception):
    """Base exception for all application-specific errors."""


# ── Data layer ──────────────────────────────────────────

class DataNotFoundError(FootballPredictionError):
    """Raised when requested data is not found (file, DB row, etc.)."""

    def __init__(self, message: str = "", resource: str = "") -> None:
        self.resource = resource
        super().__init__(message or f"Data not found: {resource}")


class DatabaseError(FootballPredictionError):
    """Raised on database connection or query failures."""


class DataIntegrityError(FootballPredictionError):
    """Raised when data fails integrity or consistency checks."""


class DataSourceError(FootballPredictionError):
    """Raised when an external data source is unreachable or returns unexpected data."""


# ── Model layer ─────────────────────────────────────────

class ModelNotFoundError(FootballPredictionError):
    """Raised when a trained model file cannot be located."""

    def __init__(self, message: str = "", model_name: str = "") -> None:
        self.model_name = model_name
        super().__init__(message or f"Model not found: {model_name}")


class ModelNotTrainedError(FootballPredictionError):
    """Raised when attempting to predict with an untrained model."""


class ModelIncompatibleError(FootballPredictionError):
    """Raised when a loaded model is incompatible with the expected interface."""


class TrainingError(FootballPredictionError):
    """Raised when model training fails."""


class PredictionError(FootballPredictionError):
    """Raised when prediction generation fails."""


class CalibrationError(FootballPredictionError):
    """Raised when probability calibration fails."""


# ── Feature / engineering layer ─────────────────────────

class FeatureEngineeringError(FootballPredictionError):
    """Raised when feature engineering or extraction fails."""


# ── Betting layer ───────────────────────────────────────

class BettingError(FootballPredictionError):
    """Raised on betting calculation errors."""


class StakingError(FootballPredictionError):
    """Raised when stake calculation fails."""


class OddsError(FootballPredictionError):
    """Raised when odds data is invalid or missing."""


# ── Scraper layer ───────────────────────────────────────

class ScraperError(FootballPredictionError):
    """Raised when a data scraper encounters a non-recoverable error."""


class ScraperRateLimitError(ScraperError):
    """Raised when a scraper is rate-limited."""

    def __init__(self, message: str = "", retry_after: int = 60) -> None:
        self.retry_after = retry_after
        super().__init__(message or f"Rate limited. Retry after {retry_after}s")


class ScraperAuthError(ScraperError):
    """Raised when API credentials are missing or invalid."""


# ── Config / validation layer ───────────────────────────

class ConfigurationError(FootballPredictionError):
    """Raised when application configuration is invalid or incomplete."""


class ValidationError(FootballPredictionError):
    """Raised when data fails validation checks."""


# ── Pipeline / orchestration layer ──────────────────────

class PipelineError(FootballPredictionError):
    """Raised when a data pipeline step fails."""


class CacheError(FootballPredictionError):
    """Raised on caching failures."""


class ExperimentError(FootballPredictionError):
    """Raised when experiment tracking fails."""
