"""
Tests for logging configuration — configure_logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.logging import configure_logging


class TestConfigureLogging:
    def test_configure_console_handler(self) -> None:
        """configure_logging adds at least a console handler."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        root.handlers.clear()

        try:
            with patch("src.config.logging.config") as mock_config:
                mock_config.logging.level = "INFO"
                mock_config.logging.format = "%(message)s"
                mock_config.logging.file = False
                mock_config.logging.file_path = Path("/tmp/test.log")

                configure_logging()

                assert len(root.handlers) >= 1
                handler_types = [type(h).__name__ for h in root.handlers]
                assert "StreamHandler" in handler_types
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)

    def test_configure_with_file_handler(self, tmp_path: Path) -> None:
        """When cfg.file is True, a RotatingFileHandler is added."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        root.handlers.clear()

        try:
            with patch("src.config.logging.config") as mock_config:
                mock_config.logging.level = "DEBUG"
                mock_config.logging.format = "%(message)s"
                mock_config.logging.file = True
                mock_config.logging.file_path = tmp_path / "test.log"

                configure_logging()

                handler_types = [type(h).__name__ for h in root.handlers]
                assert "RotatingFileHandler" in handler_types
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)

    def test_custom_level_override(self) -> None:
        """Passing a level override uses it instead of config."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        original_level = root.level
        root.handlers.clear()

        try:
            with patch("src.config.logging.config") as mock_config:
                mock_config.logging.level = "INFO"
                mock_config.logging.format = "%(message)s"
                mock_config.logging.file = False

                configure_logging(level="ERROR")

                root_level = logging.getLogger().level
                # Root level should be ERROR
                assert root_level in (logging.ERROR, 40)
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)
            root.setLevel(original_level)

    def test_clears_previous_handlers(self) -> None:
        """Calling configure_logging twice clears old handlers."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        root.handlers.clear()

        try:
            with patch("src.config.logging.config") as mock_config:
                mock_config.logging.level = "INFO"
                mock_config.logging.format = "%(message)s"
                mock_config.logging.file = False

                configure_logging()
                handler_count_1 = len(root.handlers)

                configure_logging()
                handler_count_2 = len(root.handlers)

                # Should have the same number (cleared + re-added)
                assert handler_count_2 == handler_count_1
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)

    def test_third_party_noise_reduction(self) -> None:
        """httpx, urllib3, matplotlib are set to WARNING."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        root.handlers.clear()

        try:
            with patch("src.config.logging.config") as mock_config:
                mock_config.logging.level = "DEBUG"
                mock_config.logging.format = "%(message)s"
                mock_config.logging.file = False

                configure_logging()

                assert logging.getLogger("httpx").level == logging.WARNING
                assert logging.getLogger("urllib3").level == logging.WARNING
                assert logging.getLogger("matplotlib").level == logging.WARNING
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)

    def test_console_only_when_file_disabled(self) -> None:
        """When file logging is disabled, only console handler exists."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        root.handlers.clear()

        try:
            with patch("src.config.logging.config") as mock_config:
                mock_config.logging.level = "INFO"
                mock_config.logging.format = "%(message)s"
                mock_config.logging.file = False

                configure_logging()

                handler_types = [type(h).__name__ for h in root.handlers]
                assert "FileHandler" not in handler_types
                assert "RotatingFileHandler" not in handler_types
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)

    def test_custom_log_file(self, tmp_path: Path) -> None:
        """Passing a custom log_file path overrides config."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        root.handlers.clear()

        custom_log = tmp_path / "custom.log"

        try:
            with patch("src.config.logging.config") as mock_config:
                mock_config.logging.level = "INFO"
                mock_config.logging.format = "%(message)s"
                mock_config.logging.file = True
                mock_config.logging.file_path = tmp_path / "ignored.log"

                configure_logging(log_file=str(custom_log))

                handler_types = [type(h).__name__ for h in root.handlers]
                assert "RotatingFileHandler" in handler_types
        finally:
            root.handlers.clear()
            root.handlers.extend(original_handlers)
