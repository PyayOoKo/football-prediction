"""
Logging configuration — structured console and file logging.

Usage
-----
::

    from src.config import configure_logging

    configure_logging()   # call once at application startup

    import logging
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config.settings import config


def configure_logging(
    *,
    level: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure the root logger for both console and file output.

    Parameters
    ----------
    level : str, optional
        Override the log level from config. Defaults to ``logging.level``.
    log_file : str, optional
        Override the log file path. Defaults to ``logging.file_path``.
    """
    cfg = config.logging
    resolved_level = (level or cfg.level).upper()
    root_logger = logging.getLogger()

    # Prevent duplicate handlers on repeated calls
    root_logger.handlers.clear()
    root_logger.setLevel(resolved_level)

    # ── Formatter ───────────────────────────────────────
    formatter = logging.Formatter(cfg.format)

    # ── Console handler (stderr) ────────────────────────
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ── File handler (rotating) ─────────────────────────
    if cfg.file:
        log_path = str(log_file or cfg.file_path)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # ── Third-party noise reduction ─────────────────────
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    root_logger.info(
        "Logging configured: level=%s file=%s",
        resolved_level,
        cfg.file_path if cfg.file else "(console only)",
    )
