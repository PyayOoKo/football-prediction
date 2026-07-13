"""
Tests for ModelRegistry.

Covers
------
- Register best models
- Auto-rank computation (lower-is-better vs higher-is-better)
- Leaderboard queries with filters
- Promote and demote models
- Get promoted models
- Dataframe conversion
- Error handling
"""

from __future__ import annotations

import pytest

from src.experiment_tracking.models import BestModel, Run
from src.experiment_tracking.registry import ModelRegistry


@pytest.fixture
def populated_registry(
    tracker, registry: ModelRegistry,
):
    """Create experiments, runs, and register best models."""
    exp1 = tracker.create_experiment("exp_a")
    exp2 = tracker.create_experiment("exp_b")

    # Runs for exp1
    r1 = tracker.start_run(exp1.id, model_type="xgboost", run_name="xgb_v1")
    tracker.finish_run(r1.id, metrics={"val_log_loss": 0.58, "val_accuracy": 0.72})

    r2 = tracker.start_run(exp1.id, model_type="lr", run_name="lr_v1")
    tracker.finish_run(r2.id, metrics={"val_log_loss": 0.62, "val_accuracy": 0.68})

    # Runs for exp2
    r3 = tracker.start_run(exp2.id, model_type="xgboost", run_name="xgb_v2")
    tracker.finish_run(r3.id, metrics={"val_log_loss": 0.55, "val_accuracy": 0.75})

    # Register best models
    registry.register(exp1.id, r1.id, "val_log_loss", 0.58, rank=1)
    registry.register(exp1.id, r2.id, "val_log_loss", 0.62, rank=2)
    registry.register(exp1.id, r1.id, "val_accuracy", 0.72, rank=1)
    registry.register(exp1.id, r2.id, "val_accuracy", 0.68, rank=2)
    registry.register(exp2.id, r3.id, "val_log_loss", 0.55, rank=1)

    return registry, exp1, exp2, r1, r2, r3


class TestRegistration:
    """Best model registration."""

    def test_register(self, tracker, registry: ModelRegistry):
        """Register a best model."""
        exp = tracker.create_experiment("reg_test")
        run = tracker.start_run(exp.id, model_type="xgboost")
        tracker.finish_run(run.id, metrics={"loss": 0.5})

        entry = registry.register(exp.id, run.id, "val_log_loss", 0.58, rank=1)
        assert entry.metric_name == "val_log_loss"
        assert entry.metric_value == 0.58
        assert entry.rank == 1

    def test_register_auto_rank(self, tracker, registry: ModelRegistry):
        """Auto-rank inserts at correct position."""
        exp = tracker.create_experiment("auto_rank")
        r1 = tracker.start_run(exp.id, model_type="xgboost")
        tracker.finish_run(r1.id, metrics={"loss": 0.6})
        r2 = tracker.start_run(exp.id, model_type="lr")
        tracker.finish_run(r2.id, metrics={"loss": 0.5})

        # Register worse one first
        registry.register(exp.id, r1.id, "val_log_loss", 0.6)
        # Register better one — should become rank 1
        registry.register(exp.id, r2.id, "val_log_loss", 0.5)

        entries = registry.get_leaderboard(metric_name="val_log_loss")
        assert len(entries) == 2
        assert entries[0].rank == 1
        assert entries[0].metric_value == 0.5  # Best first
        assert entries[1].rank == 2

    def test_register_auto_rank_accuracy(self, tracker, registry: ModelRegistry):
        """Auto-rank with higher-is-better metric."""
        exp = tracker.create_experiment("auto_rank_acc")
        r1 = tracker.start_run(exp.id, model_type="xgboost")
        tracker.finish_run(r1.id, metrics={"accuracy": 0.8})
        r2 = tracker.start_run(exp.id, model_type="lr")
        tracker.finish_run(r2.id, metrics={"accuracy": 0.9})

        registry.register(exp.id, r1.id, "val_accuracy", 0.8)
        registry.register(exp.id, r2.id, "val_accuracy", 0.9)

        entries = registry.get_leaderboard(metric_name="val_accuracy")
        assert entries[0].rank == 1
        assert entries[0].metric_value == 0.9  # Higher is better

    def test_register_nonexistent_experiment(self, registry: ModelRegistry):
        """Register with nonexistent experiment raises IntegrityError (FK)."""
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            registry.register("nonexistent", "run_123", "loss", 0.5, rank=1)


class TestLeaderboard:
    """Leaderboard queries."""

    def test_get_leaderboard(self, populated_registry):
        """Get leaderboard for a specific metric."""
        registry, exp1, exp2, r1, r2, r3 = populated_registry
        entries = registry.get_leaderboard(metric_name="val_log_loss")
        assert len(entries) == 3  # Two from exp1 + one from exp2

    def test_get_leaderboard_by_experiment(self, populated_registry):
        """Filter leaderboard by experiment."""
        registry, exp1, exp2, r1, r2, r3 = populated_registry
        entries = registry.get_leaderboard(
            metric_name="val_log_loss",
            experiment_id=exp1.id,
        )
        assert len(entries) == 2
        assert all(e.experiment_id == exp1.id for e in entries)

    def test_get_leaderboard_by_model_type(self, populated_registry):
        """Filter leaderboard by model type."""
        registry, exp1, exp2, r1, r2, r3 = populated_registry
        entries = registry.get_leaderboard(
            metric_name="val_log_loss",
            model_type="lr",
        )
        assert len(entries) == 1

    def test_get_best(self, populated_registry):
        """Get single best model for a metric."""
        registry, exp1, exp2, r1, r2, r3 = populated_registry
        best = registry.get_best("val_log_loss")
        assert best is not None
        assert best.metric_value == 0.55  # r3 has 0.55 (best)

    def test_get_best_with_filters(self, populated_registry):
        """Get best model scoped to an experiment."""
        registry, exp1, exp2, r1, r2, r3 = populated_registry
        best = registry.get_best("val_log_loss", experiment_id=exp1.id)
        assert best is not None
        assert best.metric_value == 0.58  # Best in exp1

    def test_get_best_empty(self, registry: ModelRegistry):
        """Get best when no entries returns None."""
        assert registry.get_best("val_log_loss") is None


class TestPromotion:
    """Promotion lifecycle."""

    def test_promote(self, tracker, registry: ModelRegistry):
        """Promote a best model."""
        exp = tracker.create_experiment("promote_test")
        run = tracker.start_run(exp.id, model_type="xgboost")
        tracker.finish_run(run.id, metrics={"loss": 0.5})
        entry = registry.register(exp.id, run.id, "val_log_loss", 0.58, rank=1)

        promoted = registry.promote(entry.id, notes="Production candidate")
        assert promoted.is_promoted is True
        assert promoted.promoted_at is not None

    def test_promote_nonexistent(self, registry: ModelRegistry):
        """Promote nonexistent entry raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            registry.promote("nonexistent")

    def test_demote(self, tracker, registry: ModelRegistry):
        """Demote a promoted model."""
        exp = tracker.create_experiment("demote_test")
        run = tracker.start_run(exp.id, model_type="xgboost")
        tracker.finish_run(run.id, metrics={"loss": 0.5})
        entry = registry.register(exp.id, run.id, "val_log_loss", 0.58, rank=1)
        registry.promote(entry.id)

        demoted = registry.demote(entry.id)
        assert demoted.is_promoted is False
        assert demoted.promoted_at is None

    def test_get_promoted(self, tracker, registry: ModelRegistry):
        """Get all promoted models."""
        exp = tracker.create_experiment("get_promoted")
        r1 = tracker.start_run(exp.id, model_type="xgboost")
        tracker.finish_run(r1.id, metrics={"loss": 0.5})
        r2 = tracker.start_run(exp.id, model_type="lr")
        tracker.finish_run(r2.id, metrics={"loss": 0.6})

        e1 = registry.register(exp.id, r1.id, "val_log_loss", 0.5, rank=1)
        e2 = registry.register(exp.id, r2.id, "val_log_loss", 0.6, rank=2)

        registry.promote(e1.id)
        promoted = registry.get_promoted()
        assert len(promoted) == 1
        assert promoted[0].id == e1.id

    def test_to_dataframe(self, populated_registry):
        """to_dataframe returns pandas DataFrame."""
        registry, exp1, exp2, r1, r2, r3 = populated_registry
        df = registry.to_dataframe()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 5 total entries
        assert "rank" in df.columns
        assert "metric_name" in df.columns
