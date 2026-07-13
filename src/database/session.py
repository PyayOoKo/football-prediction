"""
Database engine and session management.

Provides a configured SQLAlchemy ``Engine`` and ``sessionmaker``
based on application configuration, plus a convenience ``get_session``
context manager for use in services and scripts.

Usage
-----
::

    from src.database import get_session

    with get_session() as session:
        teams = session.query(Team).all()
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import create_engine as _create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import config

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def create_engine_from_config() -> Engine:
    """Build a SQLAlchemy ``Engine`` from application config."""
    cfg = config.db
    engine = _create_engine(
        cfg.sa_url,
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        pool_pre_ping=cfg.pool_pre_ping,
        echo=cfg.echo,
        # Use psycopg2 for PostgreSQL; fall back to nullpool for SQLite
        # if the URL is an in-memory / file DB.
    )
    logger.info("Database engine created: %s", cfg.sa_url)
    return engine


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Create a ``sessionmaker`` bound to the given or default engine."""
    global _engine
    if engine is None:
        if _engine is None:
            _engine = create_engine_from_config()
        engine = _engine
    return sessionmaker(bind=engine)


def get_engine() -> Engine:
    """Return the global engine, creating it if necessary."""
    global _engine
    if _engine is None:
        _engine = create_engine_from_config()
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that yields a database session and commits/rolls back.

    Yields
    ------
    Session
        A SQLAlchemy ORM session.

    Examples
    --------
    ::

        with get_session() as session:
            session.add(my_object)
            # auto-committed on success, rolled back on exception
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory()

    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables defined by models that import ``Base``.

    **Intended for development / testing only.**
    In production, use Alembic migrations.
    """
    from src.database.base import Base

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("All database tables created (dev mode).")
