"""
Tests for configuration settings — loading, defaults, and overrides.
"""

from __future__ import annotations

from src.config.settings import Config


class TestConfig:
    """Verify that the default configuration loads without errors."""

    def test_default_config_creates(self) -> None:
        cfg = Config()
        assert cfg.app.environment == "development"
        assert cfg.db.pool_size == 10

    def test_config_paths_exist(self) -> None:
        cfg = Config()
        assert cfg.paths.data is not None
        assert cfg.paths.models is not None
        assert cfg.paths.logs is not None
