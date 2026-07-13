"""Tests for Feature Store ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.feature_store.models import (
    FeatureCategory,
    FeatureComputationBatch,
    FeatureDefinition,
    FeatureDependency,
    FeatureStatus,
    FeatureValue,
    FeatureVersion,
    EntityType,
)


class TestFeatureDefinition:
    """Test FeatureDefinition ORM model."""

    def test_create(self, session) -> None:
        fd = FeatureDefinition(
            name="test_feature",
            version=1,
            feature_type="rolling_stat",
            category=FeatureCategory.ROLLING_STAT,
            entity_type=EntityType.MATCH,
            status=FeatureStatus.ACTIVE,
        )
        session.add(fd)
        session.flush()

        assert fd.id is not None
        assert isinstance(fd.id, str)
        assert len(fd.id) == 36  # UUID string format
        assert fd.name == "test_feature"
        assert fd.is_active is True

    def test_unique_constraint(self, session) -> None:
        fd1 = FeatureDefinition(name="dup", version=1, feature_type="elo",
                                category=FeatureCategory.ELO_RATING,
                                entity_type=EntityType.TEAM)
        session.add(fd1)
        session.flush()

        fd2 = FeatureDefinition(name="dup", version=1, feature_type="elo",
                                category=FeatureCategory.ELO_RATING,
                                entity_type=EntityType.TEAM)
        session.add(fd2)
        with pytest.raises(Exception):  # IntegrityError
            session.flush()

    def test_to_dict(self, session) -> None:
        fd = FeatureDefinition(
            name="test",
            version=1,
            feature_type="rolling_stat",
            category=FeatureCategory.ROLLING_STAT,
            entity_type=EntityType.MATCH,
            description="A test",
            status=FeatureStatus.ACTIVE,
        )
        session.add(fd)
        session.flush()

        d = fd.to_dict()
        assert d["name"] == "test"
        assert d["version"] == 1
        assert d["category"] == "rolling_stat"
        assert d["status"] == "active"
        assert d["is_active"] is True
        assert "id" in d

    def test_repr(self) -> None:
        fd = FeatureDefinition(name="test", version=2)
        assert repr(fd) == "<FeatureDefinition 'test' v2>"


class TestFeatureValue:
    """Test FeatureValue ORM model."""

    def test_create(self, session, sample_def) -> None:
        fv = FeatureValue(
            feature_definition_id=sample_def.id,
            match_id=1,
            numeric_value=0.85,
            computed_by="test_computer",
        )
        session.add(fv)
        session.flush()

        assert fv.id is not None
        assert isinstance(fv.id, str)
        assert len(fv.id) == 36  # UUID string format
        assert fv.numeric_value == 0.85
        assert fv.match_id == 1

    def test_text_value(self, session, sample_def) -> None:
        fv = FeatureValue(
            feature_definition_id=sample_def.id,
            team_id=42,
            text_value="strong",
            computed_by="test",
        )
        session.add(fv)
        session.flush()
        assert fv.text_value == "strong"

    def test_json_value(self, session, sample_def) -> None:
        fv = FeatureValue(
            feature_definition_id=sample_def.id,
            match_id=1,
            json_value={"attack": 1.2, "defense": 0.8},
            computed_by="test",
        )
        session.add(fv)
        session.flush()
        assert fv.json_value["attack"] == 1.2
        assert fv.json_value["defense"] == 0.8

    def test_unique_entity_constraint(self, session, sample_def) -> None:
        # Test: same definition + same match = duplicate (should be caught)
        # But SQLite treats NULL as distinct in UNIQUE constraints,
        # so we must set team_id to a non-NULL value to trigger the constraint.
        fv1 = FeatureValue(
            feature_definition_id=sample_def.id, match_id=1, team_id=10,
            numeric_value=0.5, computed_by="test",
        )
        session.add(fv1)
        session.flush()

        fv2 = FeatureValue(
            feature_definition_id=sample_def.id, match_id=1, team_id=10,
            numeric_value=0.9, computed_by="test",
        )
        session.add(fv2)
        with pytest.raises(Exception):
            session.flush()

    def test_repr(self, session, sample_def) -> None:
        fv = FeatureValue(
            feature_definition_id=sample_def.id, match_id=1,
            numeric_value=0.85, computed_by="test",
        )
        session.add(fv)
        session.flush()
        assert "FeatureValue" in repr(fv)
        assert "0.85" in repr(fv)


class TestFeatureDependency:
    """Test FeatureDependency ORM model."""

    def test_create(self, session, registry) -> None:
        def_a = registry.register(
            name="feature_a", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        def_b = registry.register(
            name="feature_b", feature_type="rolling_stat",
            category=FeatureCategory.ROLLING_STAT, entity_type="match",
            dependencies=["feature_a"],
        )

        # Get the edge that was created
        from sqlalchemy import select
        stmt = select(FeatureDependency).where(
            FeatureDependency.dependent_feature_id == def_b.id,
        )
        edge = session.execute(stmt).scalar_one()
        assert edge.dependency_feature_id == def_a.id
        assert edge.is_hard is True

    def test_repr(self) -> None:
        edge = FeatureDependency(
            dependent_feature_id=str(uuid.uuid4()),
            dependency_feature_id=str(uuid.uuid4()),
        )
        assert "FeatureDependency" in repr(edge)


class TestFeatureVersion:
    """Test FeatureVersion ORM model."""

    def test_create(self, session) -> None:
        # Create a standalone definition to avoid conflict with fixture's version
        from src.feature_store.models import FeatureDefinition, FeatureCategory, EntityType
        fd = FeatureDefinition(
            name="version_test", version=1,
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type=EntityType.TEAM,
        )
        session.add(fd)
        session.flush()

        fv = FeatureVersion(
            feature_definition_id=fd.id,
            version=99,
            is_current=True,
            changelog="Test version",
            snapshot=fd.to_dict(),
        )
        session.add(fv)
        session.flush()

        assert fv.id is not None
        assert fv.version == 99
        assert fv.is_current is True
        assert fv.changelog == "Test version"

    def test_repr(self, session) -> None:
        from src.feature_store.models import FeatureDefinition, FeatureCategory, EntityType
        fd = FeatureDefinition(
            name="repr_test", version=1,
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type=EntityType.TEAM,
        )
        session.add(fd)
        session.flush()

        fv = FeatureVersion(
            feature_definition_id=fd.id,
            version=99, is_current=True,
        )
        session.add(fv)
        session.flush()
        assert "FeatureVersion" in repr(fv)


class TestFeatureComputationBatch:
    """Test FeatureComputationBatch ORM model."""

    def test_create(self, session) -> None:
        batch = FeatureComputationBatch(
            batch_label="daily-2026-07-13",
            trigger="scheduled",
            features_computed=["elo", "form"],
            entity_count=500,
        )
        session.add(batch)
        session.flush()

        assert batch.id is not None
        assert batch.batch_label == "daily-2026-07-13"
        assert batch.success is True

    def test_complete(self, session) -> None:
        batch = FeatureComputationBatch(
            batch_label="test-batch", trigger="manual",
        )
        session.add(batch)
        session.flush()

        import time
        time.sleep(0.01)
        batch.complete(success=True)

        assert batch.completed_at is not None
        assert batch.duration_seconds is not None
        assert batch.duration_seconds > 0
        assert batch.success is True

    def test_complete_with_error(self, session) -> None:
        batch = FeatureComputationBatch(
            batch_label="test-error", trigger="manual",
        )
        session.add(batch)
        session.flush()

        batch.complete(success=False, error="Connection timeout")
        assert batch.success is False
        assert batch.error_message == "Connection timeout"

    def test_repr(self) -> None:
        batch = FeatureComputationBatch(
            batch_label="test-batch", trigger="manual",
        )
        assert "FeatureComputationBatch" in repr(batch)
