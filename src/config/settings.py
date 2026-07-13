"""
Application settings — centralised configuration hierarchy.

All configuration is loaded from environment variables (via ``.env``)
and organised into typed dataclasses. Import the singleton ``config``
instance to access settings anywhere in the application.

Environment
-----------
``.env`` files are loaded automatically from the project root.
See ``.env.example`` for the full list of supported variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# ── Auto-load .env from project root ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# ── Helper ──────────────────────────────────────────────
def _env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# ── Paths ───────────────────────────────────────────────
@dataclass
class Paths:
    """Managed directory and file paths.

    Directories are created on instantiation if they don't exist.
    """

    root: Path = _PROJECT_ROOT

    # Data
    data: Path = root / "data"
    raw: Path = data / "raw"
    processed: Path = data / "processed"
    external: Path = data / "external"

    # Models
    models: Path = root / "models"

    # Logs
    logs: Path = root / "logs"

    # Reports
    reports: Path = root / "reports"

    def __post_init__(self) -> None:
        for d in (
            self.data,
            self.raw,
            self.processed,
            self.external,
            self.models,
            self.logs,
            self.reports,
        ):
            d.mkdir(parents=True, exist_ok=True)


# ── Database ────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    """PostgreSQL / SQLAlchemy connection settings.

    Attributes
    ----------
    url : str
        Full database URL (e.g. ``postgresql+psycopg2://user:pass@host:5432/db``).
        Builds from ``DATABASE_URL`` env var, or from individual components.
    host : str
        Database host (default ``localhost``).
    port : int
        Database port (default ``5432``).
    name : str
        Database name (default ``football_prediction``).
    user : str
        Database user (default ``postgres``).
    password : str
        Database password.
    pool_size : int
        Connection pool size (default ``10``).
    max_overflow : int
        Maximum overflow connections (default ``20``).
    pool_pre_ping : bool
        Verify connections before use (default ``True``).
    echo : bool
        Log all SQL statements (default ``False``).
    """

    url: str = field(default_factory=lambda: _env_str(
        "DATABASE_URL",
        f"postgresql+psycopg2://{_env_str('DB_USER', 'postgres')}:"
        f"{_env_str('DB_PASSWORD', 'postgres')}@"
        f"{_env_str('DB_HOST', 'localhost')}:"
        f"{_env_int('DB_PORT', 5432)}/"
        f"{_env_str('DB_NAME', 'football_prediction')}",
    ))
    host: str = field(default_factory=lambda: _env_str("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("DB_PORT", 5432))
    name: str = field(default_factory=lambda: _env_str("DB_NAME", "football_prediction"))
    user: str = field(default_factory=lambda: _env_str("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: _env_str("DB_PASSWORD", "postgres"))
    pool_size: int = field(default_factory=lambda: _env_int("DB_POOL_SIZE", 10))
    max_overflow: int = field(default_factory=lambda: _env_int("DB_MAX_OVERFLOW", 20))
    pool_pre_ping: bool = field(default_factory=lambda: _env_bool("DB_POOL_PRE_PING", True))
    echo: bool = field(default_factory=lambda: _env_bool("DB_ECHO", False))

    @property
    def sa_url(self) -> str:
        """Return the SQLAlchemy-compatible database URL."""
        return self.url


# ── Logging ─────────────────────────────────────────────
@dataclass
class LoggingConfig:
    """Logging configuration.

    Attributes
    ----------
    level : str
        Root logger level (default ``INFO``).
    format : str
        Log message format string.
    file : bool
        Whether to log to a file (default ``True``).
    file_path : Path
        Path to the log file.
    rotation : str
        Log rotation interval (default ``midnight``).
    retention : int
        Number of rotated logs to keep (default ``30``).
    """

    level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    format: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    file: bool = field(default_factory=lambda: _env_bool("LOG_FILE", True))
    file_path: Path = _PROJECT_ROOT / "logs" / "football_prediction.log"
    rotation: str = "midnight"
    retention: int = 30


# ── API Keys ────────────────────────────────────────────
@dataclass
class APIConfig:
    """External API configuration.

    Attributes
    ----------
    football_data_key : str
        API key for football-data.org (env: ``FOOTBALL_DATA_API_KEY``).
    odds_api_key : str
        API key for The Odds API (env: ``THE_ODDS_API_KEY``).
    """

    football_data_key: str = field(
        default_factory=lambda: _env_str("FOOTBALL_DATA_API_KEY", "")
    )
    odds_api_key: str = field(
        default_factory=lambda: _env_str("THE_ODDS_API_KEY", "")
    )


# ── Application ─────────────────────────────────────────
@dataclass
class AppConfig:
    """Application-level settings.

    Attributes
    ----------
    debug : bool
        Enable debug mode (env: ``APP_DEBUG``, default ``False``).
    environment : str
        Deployment environment: ``development``, ``staging``, or ``production``
        (env: ``APP_ENV``, default ``development``).
    secret_key : str
        Secret key for session signing, etc. (env: ``SECRET_KEY``).
    """

    debug: bool = field(default_factory=lambda: _env_bool("APP_DEBUG", False))
    environment: str = field(
        default_factory=lambda: _env_str("APP_ENV", "development")
    )
    secret_key: str = field(
        default_factory=lambda: _env_str("SECRET_KEY", "change-me-in-production")
    )


# ── Top-level Config ────────────────────────────────────
@dataclass
class Config:
    """Root configuration object aggregating all sub-configs."""

    app: AppConfig = field(default_factory=AppConfig)
    paths: Paths = field(default_factory=Paths)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)


# Singleton instance — import this anywhere in the application.
config = Config()
