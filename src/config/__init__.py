"""
Configuration package.

Provides application-wide settings, environment variable loading,
and logging configuration. All config is centralised here for
easy discovery and modification.

Sub-modules
-----------
settings
    Dataclass-based configuration hierarchy loaded from env vars + .env file.
logging
    Structured logging setup (console + file) with rotation.
"""

from src.config.settings import Config, config
from src.config.logging import configure_logging

__all__ = [
    "Config",
    "config",
    "configure_logging",
]
