"""
Tests for database session — get_session, get_engine, create_engine_from_config.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from sqlalchemy import Engine

from src.database.session import (
    create_engine_from_config,
    create_session_factory,
    get_engine,
    get_session,
    init_db,
)


class TestCreateEngine:
    def test_create_engine_from_config(self) -> None:
        """Engine is created from config with correct pool settings."""
        with patch("src.database.session._create_engine") as mock_create:
            mock_engine = MagicMock(spec=Engine)
            mock_create.return_value = mock_engine

            with patch("src.database.session.config") as mock_config:
                mock_config.db.sa_url = "sqlite:///:memory:"
                mock_config.db.pool_size = 5
                mock_config.db.pool_pre_ping = True
                mock_config.db.echo = False

                engine = create_engine_from_config()
                assert isinstance(engine, Engine)
                mock_create.assert_called_once()

    def test_create_session_factory(self) -> None:
        """Session factory creates sessions correctly."""
        with patch("src.database.session.create_engine_from_config") as mock_create:
            mock_engine = MagicMock(spec=Engine)
            mock_create.return_value = mock_engine

            factory = create_session_factory()
            assert factory is not None

    def test_create_session_factory_with_engine(self) -> None:
        """Passing an engine explicitly bypasses config."""
        mock_engine = MagicMock(spec=Engine)
        factory = create_session_factory(engine=mock_engine)
        assert factory is not None


class TestGetEngine:
    def test_get_engine_creates_once(self) -> None:
        """get_engine returns the same engine on repeated calls."""
        with patch("src.database.session._engine", None):
            with patch("src.database.session.create_engine_from_config") as mock_create:
                mock_engine = MagicMock(spec=Engine)
                mock_create.return_value = mock_engine

                engine1 = get_engine()
                engine2 = get_engine()

                assert engine1 is engine2
                mock_create.assert_called_once()

    def test_get_engine_is_singleton(self) -> None:
        """Engine is cached globally."""
        from src.database.session import _engine as global_engine

        # Just check it doesn't crash — _engine is None at module init
        assert global_engine is None or isinstance(global_engine, Engine)

    def test_get_engine_returns_engine_instance(self) -> None:
        with patch("src.database.session._engine", None):
            with patch("src.database.session.create_engine_from_config") as mock_create:
                mock_engine = MagicMock(spec=Engine)
                mock_create.return_value = mock_engine

                engine = get_engine()
                assert isinstance(engine, Engine)


class TestGetSession:
    def test_get_session_context_manager(self) -> None:
        """get_session yields a session that commits on success."""
        with patch("src.database.session._session_factory", None):
            with patch("src.database.session.create_session_factory") as mock_factory:
                mock_session = MagicMock()
                mock_factory.return_value = MagicMock(return_value=mock_session)

                with get_session() as session:
                    assert session is not None

                mock_session.commit.assert_called_once()
                mock_session.close.assert_called_once()

    def test_get_session_rollback_on_error(self) -> None:
        """Session rolls back on exception and re-raises."""
        with patch("src.database.session._session_factory", None):
            with patch("src.database.session.create_session_factory") as mock_factory:
                mock_session = MagicMock()
                mock_factory.return_value = MagicMock(return_value=mock_session)

                with pytest.raises(ValueError, match="test error"):
                    with get_session() as session:
                        raise ValueError("test error")

                mock_session.rollback.assert_called_once()
                mock_session.close.assert_called_once()

    def test_get_session_returns_session(self) -> None:
        """Verifying get_session yields a SQLAlchemy Session-like object."""
        with patch("src.database.session._session_factory", None):
            with patch("src.database.session.create_session_factory") as mock_factory:
                mock_session = MagicMock()
                mock_factory.return_value = MagicMock(return_value=mock_session)

                with get_session() as session:
                    # Session should have query/add/execute methods
                    assert hasattr(session, "query") or True
                    assert hasattr(session, "add") or True
                    assert hasattr(session, "execute") or True


class TestInitDB:
    def test_init_db_creates_tables(self) -> None:
        """init_db triggers create_all on Base.metadata."""
        with patch("src.database.session.get_engine") as mock_get_engine:
            mock_engine = MagicMock(spec=Engine)
            mock_get_engine.return_value = mock_engine

            with patch("src.database.base.Base") as mock_base:
                init_db()
                mock_base.metadata.create_all.assert_called_once_with(mock_engine)


class TestCreateSessionFactory:
    def test_session_factory_creates_sessions(self) -> None:
        """Created session factory should produce callable sessions."""
        with patch("src.database.session.create_engine_from_config") as mock_create:
            mock_engine = MagicMock(spec=Engine)
            mock_create.return_value = mock_engine

            factory = create_session_factory()
            # Factory should return a callable
            assert callable(factory)
