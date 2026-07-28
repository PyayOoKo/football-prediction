"""
Tests for ``DatabaseConfig`` pool settings — ``pool_recycle``, ``sa_url``
PgBouncer query parameters, and related pool defaults.
"""

from __future__ import annotations

import pytest


class TestPoolRecycleDefaults:
    """Default values for pool_recycle and related pool settings."""

    def test_pool_recycle_default(self) -> None:
        """Default ``pool_recycle`` is 3600 seconds (1 hour)."""
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.pool_recycle == 3600

    def test_pool_size_default(self) -> None:
        """Default ``pool_size`` is 10 connections."""
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.pool_size == 10

    def test_max_overflow_default(self) -> None:
        """Default ``max_overflow`` is 20."""
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.max_overflow == 20

    def test_pool_pre_ping_default(self) -> None:
        """Default ``pool_pre_ping`` is ``True``."""
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.pool_pre_ping is True


class TestPoolRecycleOverride:
    """Environment variable overrides for pool_recycle."""

    def test_pool_recycle_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``DB_POOL_RECYCLE`` env var overrides the default."""
        monkeypatch.setenv("DB_POOL_RECYCLE", "1800")
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.pool_recycle == 1800

    def test_pool_recycle_disable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting ``DB_POOL_RECYCLE`` to ``-1`` disables recycling."""
        monkeypatch.setenv("DB_POOL_RECYCLE", "-1")
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.pool_recycle == -1

    def test_pool_recycle_invalid_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid ``DB_POOL_RECYCLE`` values fall back to default (3600)."""
        monkeypatch.setenv("DB_POOL_RECYCLE", "not-a-number")
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.pool_recycle == 3600

    def test_pool_size_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``DB_POOL_SIZE`` env var overrides the default."""
        monkeypatch.setenv("DB_POOL_SIZE", "25")
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert cfg.pool_size == 25


class TestSaUrl:
    """The ``sa_url`` property and PgBouncer query parameters."""

    def test_sa_url_no_pgbouncer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``USE_PGBOUNCER``, ``sa_url`` is unchanged."""
        monkeypatch.delenv("USE_PGBOUNCER", raising=False)
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        assert "prepared_statement_cache_size" not in cfg.sa_url
        assert "keepalives" not in cfg.sa_url

    def test_sa_url_appends_pgbouncer_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``USE_PGBOUNCER=true``, params are appended to a bare URL."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        assert "prepared_statement_cache_size=0" in result
        assert "keepalives=1" in result

    def test_sa_url_uses_ampersand_when_params_exist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the URL already has query params, appends with ``&``."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://user:pass@host:5432/db?sslmode=require",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        # Both params should be present
        assert "sslmode=require" in result
        assert "prepared_statement_cache_size=0" in result
        assert "keepalives=1" in result
        # Should use & separator, not ?
        assert "?sslmode=require&prepared_statement_cache_size" in result

    def test_sa_url_uses_question_mark_when_no_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the URL has no query params, appends with ``?``."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://user:pass@host:5432/db",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        assert "?prepared_statement_cache_size=0&keepalives=1" in result

    def test_sa_url_no_pgbouncer_when_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``USE_PGBOUNCER=false``, no params are appended."""
        monkeypatch.setenv("USE_PGBOUNCER", "false")
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        assert "prepared_statement_cache_size" not in result


class TestSaUrlWithPgBouncerPort:
    """PgBouncer port substitution in ``sa_url``."""

    def test_sa_url_with_pgbouncer_alias_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``USE_PGBOUNCER=true`` and no custom port, defaults to ``6432``."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://user:pass@db:5432/football_prediction",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        assert "@db:6432/" in result, f"Expected port 6432, got: {result}"

    def test_sa_url_uses_custom_pgbouncer_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``PGBOUNCER_PORT`` env var sets a custom PgBouncer port."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        monkeypatch.setenv("PGBOUNCER_PORT", "5000")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://user:pass@db:5432/football_prediction",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        assert "@db:5000/" in result, f"Expected port 5000, got: {result}"

    def test_sa_url_preserves_user_and_password_with_port_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """User/password credentials survive the port substitution."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        monkeypatch.setenv("PGBOUNCER_PORT", "7000")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://myuser:mypass@myhost:5432/mydb",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        assert "myuser:mypass@myhost:7000/" in result

    def test_sa_url_port_unchanged_without_pgbouncer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``USE_PGBOUNCER``, port stays as-is even if ``PGBOUNCER_PORT`` is set."""
        monkeypatch.delenv("USE_PGBOUNCER", raising=False)
        monkeypatch.setenv("PGBOUNCER_PORT", "6432")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://user:pass@db:5432/football_prediction",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        assert "@db:5432/" in result, f"Expected port 5432 preserved, got: {result}"
        assert "prepared_statement_cache_size" not in result

    def test_sa_url_pgbouncer_port_with_query_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Port substitution works correctly when URL already has query params."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        monkeypatch.setenv("PGBOUNCER_PORT", "6432")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://user:pass@db:5432/db?sslmode=require",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        # Port should be substituted AND query params appended
        assert "@db:6432/" in result
        assert "sslmode=require" in result
        assert "prepared_statement_cache_size=0" in result

    def test_sa_url_invalid_pgbouncer_port_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An invalid ``PGBOUNCER_PORT`` value falls back to the default (6432)."""
        monkeypatch.setenv("USE_PGBOUNCER", "true")
        monkeypatch.setenv("PGBOUNCER_PORT", "not-a-number")
        monkeypatch.setenv(
            "DATABASE_URL",
            "postgresql+psycopg2://user:pass@db:5432/football_prediction",
        )
        from src.config.settings import DatabaseConfig

        cfg = DatabaseConfig()
        result = cfg.sa_url
        # _env_int returns 6432 (the default) for invalid values
        assert "@db:6432/" in result, f"Expected fallback to 6432, got: {result}"


class TestPoolRecycleCreateEngine:
    """Verify pool_recycle is passed through to engine creation."""

    def test_create_engine_receives_pool_recycle(self) -> None:
        """create_engine_from_config passes pool_recycle to _create_engine."""
        from unittest.mock import patch, MagicMock
        from sqlalchemy import Engine
        from src.database.session import create_engine_from_config

        with patch("src.database.session._create_engine") as mock_create:
            mock_engine = MagicMock(spec=Engine)
            mock_create.return_value = mock_engine

            with patch("src.database.session.config") as mock_config:
                mock_config.db.sa_url = "sqlite:///:memory:"
                mock_config.db.pool_size = 5
                mock_config.db.pool_recycle = 3600
                mock_config.db.pool_pre_ping = True
                mock_config.db.echo = False

                create_engine_from_config()

                # Verify pool_recycle was passed to _create_engine
                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs.get("pool_recycle") == 3600
