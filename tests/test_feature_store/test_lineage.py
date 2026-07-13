"""Tests for FeatureLineage — provenance tracking."""

from __future__ import annotations

import pytest

from src.feature_store.lineage import (
    FeatureLineage,
    FeatureLineageEntry,
    LineageProvenance,
)
from src.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    FeatureStatus,
)
from src.feature_store.registry import FeatureRegistry
from src.feature_store.store import FeatureStore


class TestFeatureLineageEntry:
    """Test FeatureLineageEntry ORM model."""

    def test_create_source_entry(self, session) -> None:
        entry = FeatureLineageEntry(
            source_type="source",
            source_name="football-data.co.uk",
            source_version="2026-07-13",
            source_metadata={"url": "https://www.football-data.co.uk/englandm"},
        )
        session.add(entry)
        session.flush()

        assert entry.id is not None
        assert entry.source_type == "source"
        assert entry.source_name == "football-data.co.uk"

    def test_create_transform_entry(self, session) -> None:
        entry = FeatureLineageEntry(
            source_type="transform",
            source_name="elo_computer",
            source_version="1.0.0",
        )
        session.add(entry)
        session.flush()
        assert entry.id is not None
        assert entry.source_type == "transform"

    def test_chained_entries(self, session) -> None:
        source = FeatureLineageEntry(
            source_type="source", source_name="understat",
        )
        session.add(source)
        session.flush()

        transform = FeatureLineageEntry(
            source_type="transform",
            source_name="rolling_xg",
            parent_entry_id=source.id,
        )
        session.add(transform)
        session.flush()

        assert transform.parent_entry_id == source.id


class TestFeatureLineage:
    """Test FeatureLineage service."""

    def test_record_source(self, session) -> None:
        lineage = FeatureLineage(session)
        entry = lineage.record_source(
            source_name="test_source",
            source_version="1.0",
            source_metadata={"type": "csv"},
        )
        assert entry.source_type == "source"
        assert entry.source_name == "test_source"

    def test_record_transform(self, session) -> None:
        lineage = FeatureLineage(session)
        source = lineage.record_source("test_source")
        transform = lineage.record_transform(
            "test_transform",
            transform_version="2.0",
            parent_entry=source,
        )
        assert transform.source_type == "transform"
        assert transform.parent_entry_id == source.id

    def test_record_feature_computation(
        self, session, registry: FeatureRegistry, store: FeatureStore,
    ) -> None:
        lineage = FeatureLineage(session)
        source = lineage.record_source("test_source")

        fd = registry.register(
            name="test_feature_for_lineage",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        fv = store.set(
            definition_id=fd.id, team_id=42,
            numeric_value=1500.0, computed_by="test",
        )

        feature_entry = lineage.record_feature_computation(
            fd, fv, computed_by="test_comp",
            parent_entries=[source],
        )
        assert feature_entry.source_type == "feature"
        assert feature_entry.feature_definition_id == fd.id
        assert feature_entry.feature_value_id == fv.id

    def test_get_provenance(
        self, session, registry: FeatureRegistry, store: FeatureStore,
    ) -> None:
        lineage = FeatureLineage(session)
        source = lineage.record_source("test_source")
        transform = lineage.record_transform("test_transform", parent_entry=source)

        fd = registry.register(
            name="provenance_test_feature",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        fv = store.set(
            definition_id=fd.id, team_id=42,
            numeric_value=1500.0, computed_by="test",
        )
        lineage.record_feature_computation(
            fd, fv, computed_by="test_comp",
            parent_entries=[transform],
        )

        provenance = lineage.get_provenance(fd, team_id=42)
        assert provenance.feature_name == "provenance_test_feature"
        assert provenance.feature_version == 1
        assert provenance.value == 1500.0
        assert provenance.computed_by == "test"
        assert len(provenance.source_chain) >= 2  # transform + source

    def test_get_provenance_no_value(
        self, session, registry: FeatureRegistry,
    ) -> None:
        lineage = FeatureLineage(session)
        fd = registry.register(
            name="no_value_feature",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        provenance = lineage.get_provenance(fd)
        assert provenance.feature_name == "no_value_feature"
        assert provenance.value is None

    def test_record_model_consumption(
        self, session, registry: FeatureRegistry,
    ) -> None:
        lineage = FeatureLineage(session)

        fd1 = registry.register(
            name="model_feat_1", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        fd2 = registry.register(
            name="model_feat_2", feature_type="rolling_stat",
            category=FeatureCategory.ROLLING_STAT, entity_type="match",
            status=FeatureStatus.ACTIVE,
        )

        entries = lineage.record_model_consumption(
            "xgboost_ensemble",
            model_version="v2.0",
            features_used=[fd1, fd2],
        )
        assert len(entries) == 2
        assert entries[0].source_type == "model"
        assert entries[0].source_name == "xgboost_ensemble"

    def test_get_downstream(
        self, session, registry: FeatureRegistry, store: FeatureStore,
    ) -> None:
        lineage = FeatureLineage(session)
        source = lineage.record_source("downstream_source")

        fd = registry.register(
            name="downstream_feat",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        fv = store.set(
            definition_id=fd.id, team_id=1, numeric_value=1000.0, computed_by="test",
        )
        lineage.record_feature_computation(
            fd, fv, computed_by="test", parent_entries=[source],
        )

        downstream = lineage.get_downstream("downstream_source")
        assert len(downstream) >= 1
        names = [d["name"] for d in downstream]
        assert "downstream_feat" in names or "pipeline" in names

    def test_get_source_summary_empty(
        self, session,
    ) -> None:
        lineage = FeatureLineage(session)
        summary = lineage.get_source_summary()
        assert summary == []

    def test_to_dict(
        self, session,
    ) -> None:
        lineage = FeatureLineage(session)
        lineage.record_source("source_a")
        lineage.record_source("source_b")

        d = lineage.to_dict()
        assert d["total_entries"] == 2
        assert d["sources"] == 2
        assert len(d["entries"]) == 2

    def test_get_upstream(
        self, session, registry: FeatureRegistry, store: FeatureStore,
    ) -> None:
        lineage = FeatureLineage(session)
        source = lineage.record_source("upstream_source")

        fd = registry.register(
            name="upstream_feat",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        fv = store.set(
            definition_id=fd.id, team_id=1, numeric_value=1000.0, computed_by="test",
        )
        lineage.record_feature_computation(
            fd, fv, computed_by="test", parent_entries=[source],
        )

        upstream = lineage.get_upstream(fd)
        assert len(upstream) >= 1
        types = [u["type"] for u in upstream]
        assert "feature" in types
