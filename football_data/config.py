"""
Configuration for the football data collection system.

Centralises all settings — league codes, database paths, rate limits,
retry policies, and caching — in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── Project paths ───────────────────────────────────────

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


# ── League definitions ──────────────────────────────────

# Maps league codes (from football-data.co.uk) to human-readable names
# Maintained separately from the existing config.LEAGUE_NAMES to keep
# this module self-contained.
LEAGUE_NAMES: dict[str, str] = {
    # ═══════════════════════════════════════════════════════
    #  TOP 5 European Leagues (football-data.co.uk, with odds)
    # ═══════════════════════════════════════════════════════
    "E0":  "England Premier League",
    "SP1": "Spain La Liga",
    "D1":  "Germany Bundesliga",
    "I1":  "Italy Serie A",
    "F1":  "France Ligue 1",
    # ═══════════════════════════════════════════════════════
    #  Lower-tier leagues for value betting (via FBref, no odds)
    # ═══════════════════════════════════════════════════════
    "SE1": "Sweden Superettan",
    "NO2": "Norway OBOS-ligaen",
    "FI2": "Finland Ykkösliiga",
}

# Default leagues to collect when none specified
# Top 5 European leagues — lower-tier collected via FBref
DEFAULT_LEAGUES: tuple[str, ...] = ("E0", "SP1", "D1", "I1", "F1")


@dataclass
class FootballDataConfig:
    """Settings for football-data.co.uk collector."""

    # Base URL for CSV downloads
    base_url: str = "https://www.football-data.co.uk"
    mmz_path: str = "mmz4281"
    new_csv_url: str = "https://www.football-data.co.uk/new/{league}.csv"

    # How many recent seasons to collect per league
    max_seasons: int = 15

    # Whether to also fetch the current in-progress season
    include_current: bool = True

    # Request timeout in seconds
    request_timeout: int = 30

    # Retry settings
    max_retries: int = 3
    retry_backoff: float = 1.0  # seconds


@dataclass
class FBrefConfig:
    """Settings for FBref collector (manual browser-save mode)."""

    # Directory where users drop saved HTML pages from FBref
    saved_pages_dir: Path = PACKAGE_ROOT / "data" / "fbref_pages"

    # Supported leagues and their FBref URLs (for documentation)
    league_urls: dict[str, str] = field(
        default_factory=lambda: {
            "SE1": "https://fbref.com/en/comps/440/Superettan-Stats",
            "NO2": "https://fbref.com/en/comps/438/OBOS-ligaen-Stats",
            "FI2": "https://fbref.com/en/comps/441/Ykkösliiga-Stats",
            "IRL": "https://fbref.com/en/comps/433/LoI-Premier-Division-Stats",
            "D2": "https://fbref.com/en/comps/442/1st-Division-Stats",
            "P1": "https://fbref.com/en/comps/436/I-Liga-Stats",
        }
    )

    # Whether to attempt automated scraping (with cloudscraper)
    # This may violate FBref ToS — disabled by default
    enable_automated_scraping: bool = False


@dataclass
class TransfermarktConfig:
    """Settings for Transfermarkt data (manual CSV import only)."""

    # Directory where users place manually downloaded Transfermarkt CSVs
    import_dir: Path = PACKAGE_ROOT / "data" / "transfermarkt_imports"


@dataclass
class WeatherConfig:
    """Settings for Open-Meteo weather collector (free, no API key)."""

    # Open-Meteo API endpoint (free, no API key required)
    base_url: str = "https://archive-api.open-meteo.com/v1/archive"

    # Maximum requests per minute (free tier)
    rate_limit_per_minute: int = 30

    # Request timeout in seconds
    request_timeout: int = 15

    # Retry settings
    max_retries: int = 3
    retry_backoff: float = 1.0


@dataclass
class DatabaseConfig:
    """Settings for the SQLite database."""

    # Path to the SQLite database file
    db_path: Path = PROJECT_ROOT / "data" / "football_data.db"

    # Whether to enable WAL mode for better concurrent reads
    wal_mode: bool = True

    # Whether to create tables on first connection
    auto_create: bool = True


@dataclass
class CacheConfig:
    """Settings for the HTTP response cache."""

    # Cache directory
    cache_dir: Path = PACKAGE_ROOT / "data" / "cache"

    # Cache TTL in seconds (default 24 hours)
    ttl_seconds: int = 86400

    # Maximum cache size in MB (0 = unlimited)
    max_size_mb: int = 500


@dataclass
class LoggingConfig:
    """Settings for logging."""

    # Log level
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Log file path
    log_file: Path = PROJECT_ROOT / "logs" / "football_data.log"

    # Whether to also log to console
    console_logging: bool = True


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""

    # Which leagues to collect (league codes)
    leagues: tuple[str, ...] = DEFAULT_LEAGUES

    # Sub-configs
    football_data: FootballDataConfig = field(default_factory=FootballDataConfig)
    fbref: FBrefConfig = field(default_factory=FBrefConfig)
    transfermarkt: TransfermarktConfig = field(default_factory=TransfermarktConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Single importable instance
config = Config()
