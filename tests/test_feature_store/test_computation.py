"""Tests for FeatureComputationEngine, LazyFeature, and LazyFeatureSet."""

from __future__ import annotations

from typing import Any

import pytest

from src.feature_store import (
    FeatureCategory,
    FeatureStatus,
)
from src.feature_store.computation import (
    ComputationReport,
    FeatureComputationEngine,
    LazyFeature,
    LazyFeatureSet,
)
from src.feature_store.computers import ComputerRegistry, FeatureComputer
from src.feature_store.registry import FeatureRegistry
from src.feature_store.store import FeatureStore


# ── Mock computer for testing ───────────────────────────

class MockEloComputer(FeatureComputer):
    """Mock Elo computer for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="elo", description="Mock Elo", **kwargs)

    def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
        return {"elo_rating": float(1500 + entity_id * 10)}

    def compute_batch(
        self, entity_ids: list[int], **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        return {eid: self.compute_one(eid, **kwargs) for eid in entity_ids}


class MockAttackComputer(FeatureComputer):
    """Mock attack strength computer for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="attack", description="Mock Attack", **kwargs)

    def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
        return {"attack_strength": 1.0 + entity_id * 0.1}


# ── Fixtures ────────────────────────────────────────────


@pytest.fixture
def computer_registry(session) -> ComputerRegistry:
    """Register mock computers."""
    reg = ComputerRegistry()
    reg.add("elo", MockEloComputer())
    reg.add("attack_strength", MockAttackComputer())
    return reg


@pytest.fixture
def engine(
    session,
    computer_registry: ComputerRegistry,
    registry: FeatureRegistry,
    store: FeatureStore,
) -> FeatureComputationEngine:
    """Create engine with registered computers."""
    return FeatureComputationEngine(
        registry=computer_registry,
        store=store,
        registry_service=registry,
        show_progress=False,
    )


@pytest.fixture
def elo_def(session, registry: FeatureRegistry) -> None:
    """Register elo_rating feature for tests."""
    registry.register(
        name="elo_rating",
        feature_type="elo",
        category=FeatureCategory.ELO_RATING,
        entity_type="team",
        status=FeatureStatus.ACTIVE,
    )


@pytest.fixture
def attack_def(session, registry: FeatureRegistry) -> None:
    """Register attack_strength feature for tests."""
    registry.register(
        name="attack_strength",
        feature_type="attack_strength",
        category=FeatureCategory.ATTACK_STRENGTH,
        entity_type="team",
        status=FeatureStatus.ACTIVE,
    )


# ═══════════════════════════════════════════════════════════
#  ComputationReport
# ═══════════════════════════════════════════════════════════


class TestComputationReport:
    """Test ComputationReport dataclass."""

    def test_defaults(self) -> None:
        report = ComputationReport()
        assert report.success is True
        assert report.computed_count == 0
        assert report.duration_seconds == 0.0

    def test_to_dict(self) -> None:
        report = ComputationReport(
            batch_id="abc123",
            batch_label="test-batch",
            feature_names=["elo"],
            entity_count=100,
            computed_count=80,
            skipped_count=15,
            failed_count=5,
            duration_seconds=12.5,
            success=True,
        )
        d = report.to_dict()
        assert d["batch_id"] == "abc123"
        assert d["computed_count"] == 80
        assert d["success"] is True
        assert d["duration_seconds"] == 12.5

    def test_failed_to_dict(self) -> None:
        report = ComputationReport(
            batch_id="failed-batch",
            success=False,
            error="Connection timeout",
        )
        d = report.to_dict()
        assert d["success"] is False
        assert d["error"] == "Connection timeout"


# ═══════════════════════════════════════════════════════════
#  FeatureComputationEngine
# ═══════════════════════════════════════════════════════════


class TestFeatureComputationEngine:
    """Test computation engine orchestration."""

    def test_compute_features_basic(
        self, engine: FeatureComputationEngine,
        elo_def, attack_def,
    ) -> None:
        """Basic batch computation with registered computers."""
        report = engine.compute_features(
            feature_names=["elo_rating", "attack_strength"],
            entity_ids=[1, 2, 3],
            entity_type="team",
        )
        assert report.success is True
        assert report.computed_count == 6  # 2 features x 3 entities
        assert report.failed_count == 0
        assert report.batch_id is not None

    def test_compute_features_with_nonexistent_computer(
        self, session,
        registry: FeatureRegistry,
        store: FeatureStore,
    ) -> None:
        """Missing computer should be skipped, not crash."""
        comp_reg = ComputerRegistry()
        # Only register 'elo', not 'rolling_stat'
        comp_reg.add("elo", MockEloComputer())

        engine = FeatureComputationEngine(
            registry=comp_reg, store=store,
            registry_service=registry, show_progress=False,
        )

        # Register a feature whose computer type is NOT in the registry
        registry.register(
            name="missing_computer_feat",
            feature_type="rolling_stat",  # Valid type, but no computer registered
            category=FeatureCategory.ROLLING_STAT,
            entity_type="match",
            status=FeatureStatus.ACTIVE,
        )

        report = engine.compute_features(
            feature_names=["missing_computer_feat"],
            entity_ids=[1],
            entity_type="match",
            force_recompute=True,
        )
        # Should report as skipped/failed since there's no computer
        assert report.computed_count == 0
        assert "missing_computer_feat" in report.per_feature_stats

    def test_compute_one(
        self, engine: FeatureComputationEngine,
        session,
        registry: FeatureRegistry,
    ) -> None:
        """Single feature, single entity computation."""
        registry.register(
            name="elo_rating",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        result = engine.compute_one(
            feature_name="elo_rating",
            entity_id=42,
            entity_type="team",
        )
        assert isinstance(result, dict)
        assert "elo_rating" in result or len(result) > 0

    def test_run_incremental(
        self, engine: FeatureComputationEngine,
        elo_def,
    ) -> None:
        """Incremental computation skips already-computed entities."""
        # First run computes everything
        report1 = engine.run_incremental(
            feature_names=["elo_rating"],
            entity_ids=[1, 2, 3],
            entity_type="team",
            max_age_hours=24,
        )
        assert report1.computed_count == 3

        # Second run should skip all (already fresh)
        report2 = engine.run_incremental(
            feature_names=["elo_rating"],
            entity_ids=[1, 2, 3],
            entity_type="team",
            max_age_hours=24,
        )
        assert report2.computed_count < 3  # Some or all were skipped

    def test_force_recompute(
        self, engine: FeatureComputationEngine,
        elo_def,
    ) -> None:
        """Force recompute processes all entities regardless of freshness."""
        engine.compute_features(
            feature_names=["elo_rating"],
            entity_ids=[1, 2, 3],
            entity_type="team",
            incremental=True,
        )

        # Force recompute should process all again
        report = engine.compute_features(
            feature_names=["elo_rating"],
            entity_ids=[1, 2, 3],
            entity_type="team",
            force_recompute=True,
        )
        assert report.computed_count == 3

    def test_compute_with_nonexistent_feature(
        self, engine: FeatureComputationEngine,
    ) -> None:
        """Unknown features should be skipped without crashing."""
        report = engine.compute_features(
            feature_names=["nonexistent_feature"],
            entity_ids=[1],
        )
        assert report.success is True  # No critical error

    def test_resume_nonexistent_batch(
        self, engine: FeatureComputationEngine,
    ) -> None:
        """Resuming a non-existent batch should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.resume("nonexistent-batch-id")

    def test_resume_completed_batch(
        self, engine: FeatureComputationEngine,
        elo_def,
    ) -> None:
        """Resuming a completed batch should be a no-op."""
        report = engine.compute_features(
            feature_names=["elo_rating"],
            entity_ids=[1],
            entity_type="team",
        )
        resume_report = engine.resume(report.batch_id)
        assert resume_report.success is True


# ═══════════════════════════════════════════════════════════
#  LazyFeature
# ═══════════════════════════════════════════════════════════


class TestLazyFeature:
    """Test lazy feature computation."""

    def test_not_computed_initially(self) -> None:
        feat = LazyFeature("elo_rating", entity_id=42, entity_type="team")
        assert feat.is_computed is False

    def test_compute_on_get(
        self, engine: FeatureComputationEngine,
        session, registry: FeatureRegistry,
    ) -> None:
        registry.register(
            name="elo_rating",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )

        feat = LazyFeature(
            "elo_rating", entity_id=42, entity_type="team",
            engine=engine,
        )
        value = feat.get()
        assert value is not None
        assert feat.is_computed is True

    def test_cache_returns_same_value(
        self, engine: FeatureComputationEngine,
        session, registry: FeatureRegistry,
    ) -> None:
        registry.register(
            name="elo_rating",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )

        feat = LazyFeature(
            "elo_rating", entity_id=1, entity_type="team",
            engine=engine, use_cache=True,
        )
        v1 = feat.get()
        v2 = feat.get()  # Should return cached
        assert v1 == v2

    def test_force_recompute(
        self, engine: FeatureComputationEngine,
        session, registry: FeatureRegistry,
    ) -> None:
        registry.register(
            name="elo_rating",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )

        feat = LazyFeature(
            "elo_rating", entity_id=1, entity_type="team",
            engine=engine,
        )
        v1 = feat.get()
        feat._is_computed = False  # Simulate cache clearing
        v2 = feat.get(force_recompute=True)
        assert v2 is not None

    def test_no_engine_raises(self) -> None:
        feat = LazyFeature("elo_rating", entity_id=1)
        with pytest.raises(ValueError, match="No engine provided"):
            feat.get()

    def test_set_engine_after_init(
        self, engine: FeatureComputationEngine,
    ) -> None:
        feat = LazyFeature("elo_rating", entity_id=1)
        feat.set_engine(engine)
        assert feat._engine is engine

    def test_repr(self) -> None:
        feat = LazyFeature("elo_rating", entity_id=42, entity_type="team")
        assert "LazyFeature" in repr(feat)
        assert "elo_rating" in repr(feat)
        assert "lazy" in repr(feat)

        feat._is_computed = True
        assert "computed" in repr(feat)


# ═══════════════════════════════════════════════════════════
#  LazyFeatureSet
# ═══════════════════════════════════════════════════════════


class TestLazyFeatureSet:
    """Test LazyFeatureSet collection."""

    def test_empty_set(self) -> None:
        features = LazyFeatureSet(entity_id=1)
        assert len(features) == 0

    def test_add_feature(self) -> None:
        features = LazyFeatureSet(entity_id=42, entity_type="team")
        lazy = features.add("elo_rating")
        assert lazy.feature_name == "elo_rating"
        assert lazy.is_computed is False
        assert len(features) == 1

    def test_getitem_lazy(
        self, engine: FeatureComputationEngine,
        session, registry: FeatureRegistry,
    ) -> None:
        registry.register(
            name="elo_rating",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )

        features = LazyFeatureSet(
            entity_id=1, entity_type="team", engine=engine,
        )
        features.add("elo_rating")

        value = features["elo_rating"]
        assert value is not None

        computed = features.get_computed()
        assert "elo_rating" in computed

    def test_contains(self) -> None:
        features = LazyFeatureSet(entity_id=1)
        features.add("test_feat")
        assert "test_feat" in features
        assert "other" not in features

    def test_compute_all(
        self, engine: FeatureComputationEngine,
        session, registry: FeatureRegistry,
    ) -> None:
        registry.register(
            name="feat_a",
            feature_type="elo",
            category=FeatureCategory.ELO_RATING,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        registry.register(
            name="feat_b",
            feature_type="attack_strength",
            category=FeatureCategory.ATTACK_STRENGTH,
            entity_type="team",
            status=FeatureStatus.ACTIVE,
        )

        features = LazyFeatureSet(
            entity_id=1, entity_type="team", engine=engine,
        )
        features.add("feat_a")
        features.add("feat_b")

        results = features.compute_all()
        assert len(results) == 2

    def test_repr(self) -> None:
        features = LazyFeatureSet(entity_id=1)
        assert "LazyFeatureSet" in repr(features)
