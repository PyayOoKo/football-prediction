"""Tests for FeatureStore CRUD operations."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.feature_store.models import (
    FeatureCategory,
    FeatureComputationBatch,
)
from src.feature_store.store import FeatureStore


class TestFeatureStore:
    """Test core store operations."""

    # ── Single value ──────────────────────────────────────

    def test_set_numeric(self, store: FeatureStore, sample_def) -> None:
        fv = store.set(
            definition_id=sample_def.id,
            match_id=1,
            numeric_value=0.85,
            computed_by="test",
        )
        assert fv.numeric_value == 0.85
        assert fv.match_id == 1
        assert fv.feature_definition_id == sample_def.id

    def test_set_text(self, store: FeatureStore, sample_def) -> None:
        fv = store.set(
            definition_id=sample_def.id,
            team_id=42,
            text_value="strong",
            computed_by="test",
        )
        assert fv.text_value == "strong"
        assert fv.team_id == 42

    def test_set_json(self, store: FeatureStore, sample_def) -> None:
        fv = store.set(
            definition_id=sample_def.id,
            match_id=1,
            json_value={"attack": 1.2, "defense": 0.8},
            computed_by="test",
        )
        assert fv.json_value["attack"] == 1.2
        assert fv.json_value["defense"] == 0.8

    def test_set_global_feature(self, store: FeatureStore, sample_def) -> None:
        """A global feature has no match_id, team_id, or league_id."""
        fv = store.set(
            definition_id=sample_def.id,
            numeric_value=1.0,
            computed_by="test",
        )
        assert fv.numeric_value == 1.0
        assert fv.match_id is None
        assert fv.team_id is None

    def test_set_updates_existing(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.5, computed_by="test")
        fv = store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.9, computed_by="test")
        assert fv.numeric_value == 0.9

    def test_get_nonexistent(self, store: FeatureStore, sample_def) -> None:
        assert store.get(definition_id=sample_def.id, match_id=999) is None

    def test_get_existing(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.5, computed_by="test")
        fv = store.get(definition_id=sample_def.id, match_id=1)
        assert fv is not None
        assert fv.numeric_value == 0.5

    def test_get_value(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.5, computed_by="test")
        val = store.get_value(definition_id=sample_def.id, match_id=1)
        assert val == 0.5

    def test_get_value_nonexistent(self, store: FeatureStore, sample_def) -> None:
        assert store.get_value(definition_id=sample_def.id, match_id=999) is None

    # ── Batch operations ──────────────────────────────────

    def test_set_many(self, store: FeatureStore, sample_def) -> None:
        values = [
            dict(definition_id=sample_def.id, match_id=1, numeric_value=0.1, computed_by="test"),
            dict(definition_id=sample_def.id, match_id=2, numeric_value=0.2, computed_by="test"),
            dict(definition_id=sample_def.id, match_id=3, numeric_value=0.3, computed_by="test"),
        ]
        count = store.set_many(values)
        assert count == 3

        fv1 = store.get(definition_id=sample_def.id, match_id=1)
        assert fv1 is not None
        assert fv1.numeric_value == 0.1

    def test_set_many_updates_existing(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.1, computed_by="test")
        store.set_many([
            dict(definition_id=sample_def.id, match_id=1, numeric_value=0.99, computed_by="test"),
        ])
        fv = store.get(definition_id=sample_def.id, match_id=1)
        assert fv is not None
        assert fv.numeric_value == 0.99

    def test_get_many(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.1, computed_by="test")
        store.set(definition_id=sample_def.id, match_id=2, numeric_value=0.2, computed_by="test")
        results = store.get_many(
            [sample_def.id], match_ids=[1, 2],
        )
        assert len(results) == 2

    def test_get_many_empty(self, store: FeatureStore, sample_def) -> None:
        results = store.get_many([sample_def.id], match_ids=[999])
        assert results == []

    # ── Delete ────────────────────────────────────────────

    def test_delete_existing(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.5, computed_by="test")
        assert store.delete(definition_id=sample_def.id, match_id=1) is True
        assert store.get(definition_id=sample_def.id, match_id=1) is None

    def test_delete_nonexistent(self, store: FeatureStore, sample_def) -> None:
        assert store.delete(definition_id=sample_def.id, match_id=999) is False

    def test_delete_all_for_definition(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.1, computed_by="test")
        store.set(definition_id=sample_def.id, match_id=2, numeric_value=0.2, computed_by="test")
        count = store.delete_all_for_definition(sample_def.id)
        assert count == 2
        assert store.get(definition_id=sample_def.id, match_id=1) is None
        assert store.get(definition_id=sample_def.id, match_id=2) is None

    # ── Batch tracking ────────────────────────────────────

    def test_start_batch(self, store: FeatureStore) -> None:
        batch = store.start_batch(
            batch_label="test-batch",
            trigger="manual",
            features_computed=["elo", "form"],
            entity_count=100,
        )
        assert batch.id is not None
        assert batch.batch_label == "test-batch"
        assert batch.success is True
        assert batch.started_at is not None

    def test_complete_batch(self, store: FeatureStore) -> None:
        batch = store.start_batch("test-batch")
        import time
        time.sleep(0.01)
        completed = store.complete_batch(batch.id, success=True)
        assert completed.completed_at is not None
        assert completed.duration_seconds is not None
        assert completed.duration_seconds > 0

    def test_complete_batch_with_error(self, store: FeatureStore) -> None:
        batch = store.start_batch("test-error")
        completed = store.complete_batch(batch.id, success=False, error="Failed")
        assert completed.success is False
        assert completed.error_message == "Failed"

    def test_complete_batch_nonexistent(self, store: FeatureStore) -> None:
        with pytest.raises(ValueError, match="not found"):
            store.complete_batch(str(uuid.uuid4()))

    def test_get_batch(self, store: FeatureStore) -> None:
        batch = store.start_batch("test-batch")
        found = store.get_batch(batch.id)
        assert found is not None
        assert found.batch_label == "test-batch"

    def test_list_batches(self, store: FeatureStore) -> None:
        store.start_batch("batch-1")
        store.start_batch("batch-2", trigger="scheduled")
        store.start_batch("batch-3")
        assert len(store.list_batches(limit=10)) == 3
        assert len(store.list_batches(trigger="scheduled")) == 1

    # ── Incremental updates ───────────────────────────────

    def test_needs_update_no_values(self, store: FeatureStore, sample_def) -> None:
        stale = store.needs_update(sample_def.id, [1, 2, 3], entity_type="match")
        assert stale == [1, 2, 3]

    def test_needs_update_all_fresh(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.5, computed_by="test")
        store.set(definition_id=sample_def.id, match_id=2, numeric_value=0.6, computed_by="test")
        stale = store.needs_update(
            sample_def.id, [1, 2],
            entity_type="match", max_age_hours=24,
        )
        assert stale == []

    def test_needs_update_some_stale(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.5, computed_by="test")
        stale = store.needs_update(sample_def.id, [1, 2, 3])
        assert stale == [2, 3]

    def test_needs_update_empty_ids(self, store: FeatureStore, sample_def) -> None:
        assert store.needs_update(sample_def.id, []) == []

    # ── Feature vector assembly ───────────────────────────

    def test_assemble_feature_vector(self, store: FeatureStore, registry, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.85, computed_by="test")
        vector = store.assemble_feature_vector([sample_def.id], match_id=1)
        assert "test_feature" in vector
        assert vector["test_feature"] == 0.85

    def test_assemble_feature_vector_no_match(self, store: FeatureStore, sample_def) -> None:
        store.set(definition_id=sample_def.id, match_id=1, numeric_value=0.85, computed_by="test")
        vector = store.assemble_feature_vector([sample_def.id], match_id=999)
        assert vector == {}

    def test_assemble_feature_vector_team_level(self, store: FeatureStore, registry, sample_team_def) -> None:
        store.set(definition_id=sample_team_def.id, team_id=42, numeric_value=1500.0, computed_by="test")
        vector = store.assemble_feature_vector(
            [sample_team_def.id],
            match_id=1,
            team_ids={"home": 42},
        )
        assert "team_elo_rating" in vector
        assert vector["team_elo_rating"] == 1500.0
