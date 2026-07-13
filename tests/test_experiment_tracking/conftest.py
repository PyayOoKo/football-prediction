"""
Shared fixtures for experiment tracking tests.

Creates an in-memory SQLite database with all experiment tracking
tables created via ``Base.metadata.create_all``.

Fixtures
--------
session
    A clean SQLAlchemy ORM session for each test.
tracker
    ``ExperimentTracker`` instance bound to ``session``.
registry
    ``ModelRegistry`` instance bound to ``session``.
comparator
    ``ExperimentComparator`` instance bound to ``session``.
sample_experiment
    A persisted experiment with name ``test_exp``.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.experiment_tracking.comparator import ExperimentComparator
from src.experiment_tracking.models import Base, Experiment
from src.experiment_tracking.registry import ModelRegistry
from src.experiment_tracking.tracker import ExperimentTracker


@pytest.fixture
def session():
    """Create an in-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        # Enable FK enforcement for constraint testing
        sess.execute(text("PRAGMA foreign_keys = ON"))
        yield sess


@pytest.fixture
def tracker(session: Session) -> ExperimentTracker:
    """ExperimentTracker bound to the test session."""
    return ExperimentTracker(session)


@pytest.fixture
def registry(session: Session) -> ModelRegistry:
    """ModelRegistry bound to the test session."""
    return ModelRegistry(session)


@pytest.fixture
def comparator(session: Session) -> ExperimentComparator:
    """ExperimentComparator bound to the test session."""
    return ExperimentComparator(session)


@pytest.fixture
def sample_experiment(session: Session) -> Experiment:
    """A persisted experiment for tests."""
    exp = Experiment(name="test_exp", tags={"env": "test"})
    session.add(exp)
    session.flush()
    return exp
