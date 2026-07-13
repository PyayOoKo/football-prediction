"""Pytest configuration and fixtures for feature store tests."""

from __future__ import annotations

from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import Base
from src.feature_store import (
    FeatureDefinition,
    FeatureComputationBatch,
    FeatureCategory,
    FeatureStatus,
    EntityType,
)
from src.feature_store.registry import FeatureRegistry
from src.feature_store.store import FeatureStore
from src.feature_store.validation import FeatureValidator


@pytest.fixture
def session() -> Iterator[Session]:
    """Create an in-memory SQLite session with all tables."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    s = SessionLocal()
    yield s
    s.close()


@pytest.fixture
def registry(session: Session) -> FeatureRegistry:
    return FeatureRegistry(session)


@pytest.fixture
def store(session: Session) -> FeatureStore:
    return FeatureStore(session)


@pytest.fixture
def validator() -> FeatureValidator:
    return FeatureValidator()


@pytest.fixture
def sample_def(session: Session, registry: FeatureRegistry) -> FeatureDefinition:
    """Register and return a simple feature definition."""
    return registry.register(
        name="test_feature",
        feature_type="rolling_stat",
        category=FeatureCategory.ROLLING_STAT,
        entity_type="match",
        description="A test feature",
        computation_params={"window": 5},
        validation_rules={"min": 0.0, "max": 100.0},
        status=FeatureStatus.ACTIVE,
    )


@pytest.fixture
def sample_team_def(session: Session, registry: FeatureRegistry) -> FeatureDefinition:
    """Register and return a team-level feature definition."""
    return registry.register(
        name="team_elo_rating",
        feature_type="elo",
        category=FeatureCategory.ELO_RATING,
        entity_type="team",
        description="Team Elo rating",
        computation_params={"k": 32},
        validation_rules={"min": 1000.0, "max": 2500.0},
        status=FeatureStatus.ACTIVE,
    )
