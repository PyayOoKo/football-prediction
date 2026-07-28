"""
Shared test fixtures and configuration.

Provides pytest fixtures for database sessions, sample data,
and mock objects used across test modules.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import Base
from src.database.models import Competition, Match, Team
from src.di_container import configure_container, reset_container, set_container


# ── DI container bootstrap ───────────────────────────────
@pytest.fixture(autouse=True)
def _bootstrap_global_di_container() -> None:
    """Register the root config as ConfigProvider so services can resolve it.

    Child conftest fixtures (e.g. ``tests/test_services/conftest.py``) run
    after this one and can override with a mock via ``set_container()``.
    """
    from src.config import config

    container = configure_container(config)
    set_container(container)
    yield
    reset_container()


# ── Sample data ─────────────────────────────────────────
@pytest.fixture
def sample_matches_df() -> pd.DataFrame:
    """Return a small DataFrame of synthetic match data for testing."""
    return pd.DataFrame(
        {
            "date": ["2024-01-07", "2024-01-08"],
            "home_team": ["Team A", "Team C"],
            "away_team": ["Team B", "Team D"],
            "home_goals": [2, 1],
            "away_goals": [1, 1],
            "result": ["H", "D"],
            "league": ["Test League", "Test League"],
        }
    )


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Return a minimal configuration dict for testing."""
    return {
        "app": {"debug": False, "environment": "test"},
        "paths": {"data": str(Path.cwd() / "data")},
        "db": {
            "url": "sqlite:///:memory:",
            "echo": False,
        },
    }


# ── Database fixtures ───────────────────────────────────
@pytest.fixture(scope="session")
def in_memory_db_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(in_memory_db_engine) -> Generator[Session, None, None]:
    """Provide a transactional database session.

    Rolls back after each test to ensure isolation.
    """
    connection = in_memory_db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    # Roll back before closing — session.close() would de-associate
    # the transaction from the connection, causing SAWarning.
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_team(db_session: Session) -> Team:
    """Create a sample team record."""
    team = Team(name="Arsenal", short_name="ARS")
    db_session.add(team)
    db_session.flush()
    return team


@pytest.fixture
def other_team(db_session: Session) -> Team:
    """Create a second sample team record for matches."""
    team = Team(name="Chelsea", short_name="CHE")
    db_session.add(team)
    db_session.flush()
    return team


@pytest.fixture
def sample_league(db_session: Session) -> Competition:
    """Create a sample competition (league) record."""
    comp = Competition(name="Premier League", code="E0", type="league")
    db_session.add(comp)
    db_session.flush()
    return comp
