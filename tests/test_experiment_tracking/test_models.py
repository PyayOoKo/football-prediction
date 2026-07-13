"""
Tests for experiment tracking ORM models.

Covers
------
- Experiment creation and constraints
- Run creation, relationships, status lifecycle
- BestModel registration and uniqueness
- ModelArtifact logging
- ``to_dict()`` and ``__repr__`` methods
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from src.experiment_tracking.models import (
    BestModel,
    Experiment,
    ModelArtifact,
    Run,
)


class TestExperiment:
    """Experiment model tests."""

    def test_create(self, session):
        """Create a basic experiment."""
        exp = Experiment(name="test_create", tags={"env": "test"})
        session.add(exp)
        session.flush()
        assert exp.id is not None
        assert len(exp.id) == 36
        assert exp.name == "test_create"
        assert exp.tags == {"env": "test"}
        assert isinstance(exp.created_at, datetime)

    def test_default_tags(self, session):
        """Tags default to None."""
        exp = Experiment(name="no_tags")
        session.add(exp)
        session.flush()
        assert exp.tags is None

    def test_to_dict(self, session):
        """to_dict returns correct fields."""
        exp = Experiment(name="dict_test", description="A test")
        session.add(exp)
        session.flush()
        d = exp.to_dict()
        assert d["name"] == "dict_test"
        assert d["description"] == "A test"
        assert d["id"] == exp.id
        assert d["run_count"] == 0
        assert "created_at" in d

    def test_repr(self, session):
        """__repr__ includes name and run count."""
        exp = Experiment(name="repr_test")
        session.add(exp)
        session.flush()
        r = repr(exp)
        assert "repr_test" in r
        assert "0 runs" in r


class TestRun:
    """Run model tests."""

    def test_create(self, session, sample_experiment):
        """Create a basic run."""
        run = Run.create(
            experiment_id=sample_experiment.id,
            model_type="xgboost",
        )
        session.add(run)
        session.flush()
        assert run.id is not None
        assert run.experiment_id == sample_experiment.id
        assert run.model_type == "xgboost"
        assert run.status == "running"
        assert isinstance(run.started_at, datetime)

    def test_create_with_params(self, session, sample_experiment):
        """Create a run with full parameters."""
        run = Run.create(
            experiment_id=sample_experiment.id,
            model_type="logistic_regression",
            run_name="lr_baseline",
            hyperparameters={"C": 1.0, "max_iter": 1000},
            random_seed=42,
            git_commit="abc1234",
            notes="Baseline LR",
            tags={"set": "baseline"},
        )
        session.add(run)
        session.flush()
        assert run.run_name == "lr_baseline"
        assert run.hyperparameters == {"C": 1.0, "max_iter": 1000}
        assert run.random_seed == 42
        assert run.git_commit == "abc1234"

    def test_status_lifecycle(self, session, sample_experiment):
        """Run status transitions: running -> completed/failed."""
        run = Run.create(
            experiment_id=sample_experiment.id,
            model_type="xgboost",
        )
        session.add(run)
        session.flush()
        assert run.status == "running"

        # Complete
        run.status = "completed"
        run.metrics = {"val_log_loss": 0.58}
        run.finished_at = datetime.now(timezone.utc)
        session.flush()
        assert run.status == "completed"

        # Fail
        run2 = Run.create(
            experiment_id=sample_experiment.id,
            model_type="xgboost",
        )
        session.add(run2)
        session.flush()
        run2.status = "failed"
        run2.error_message = "Out of memory"
        run2.finished_at = datetime.now(timezone.utc)
        session.flush()
        assert run2.status == "failed"
        assert run2.error_message == "Out of memory"

    def test_unique_run_name(self, session, sample_experiment):
        """Run names must be unique within an experiment."""
        r1 = Run.create(
            experiment_id=sample_experiment.id,
            model_type="xgboost",
            run_name="same_name",
        )
        session.add(r1)
        session.flush()

        with pytest.raises(IntegrityError):
            r2 = Run.create(
                experiment_id=sample_experiment.id,
                model_type="lr",
                run_name="same_name",
            )
            session.add(r2)
            session.flush()

    def test_cascade_delete(self, session, sample_experiment):
        """Deleting an experiment cascades to its runs."""
        run = Run.create(
            experiment_id=sample_experiment.id,
            model_type="xgboost",
        )
        session.add(run)
        session.flush()
        run_id = run.id

        session.delete(sample_experiment)
        session.flush()

        assert session.get(Run, run_id) is None

    def test_to_dict(self, session, sample_experiment):
        """to_dict returns correct fields."""
        run = Run.create(
            experiment_id=sample_experiment.id,
            model_type="xgboost",
            hyperparameters={"n_estimators": 100},
        )
        run.status = "completed"
        run.metrics = {"val_log_loss": 0.58}
        session.add(run)
        session.flush()

        d = run.to_dict()
        assert d["model_type"] == "xgboost"
        assert d["status"] == "completed"
        assert d["metrics"] == {"val_log_loss": 0.58}
        assert d["hyperparameters"] == {"n_estimators": 100}

    def test_repr(self, session, sample_experiment):
        """__repr__ includes model type and status."""
        run = Run.create(
            experiment_id=sample_experiment.id,
            model_type="xgboost",
        )
        session.add(run)
        session.flush()
        r = repr(run)
        assert "xgboost" in r
        assert "running" in r


class TestBestModel:
    """BestModel model tests."""

    def test_create(self, session, sample_experiment, sample_run):
        """Create a best model entry."""
        entry = BestModel(
            experiment_id=sample_experiment.id,
            run_id=sample_run.id,
            metric_name="val_log_loss",
            metric_value=0.58,
            rank=1,
        )
        session.add(entry)
        session.flush()
        assert entry.id is not None
        assert entry.rank == 1
        assert entry.is_promoted is False

    def test_unique_rank_per_metric(self, session, sample_experiment, sample_run):
        """Cannot have two entries with same rank for same metric/experiment."""
        b1 = BestModel(
            experiment_id=sample_experiment.id,
            run_id=sample_run.id,
            metric_name="val_log_loss",
            metric_value=0.58,
            rank=1,
        )
        session.add(b1)
        session.flush()

        with pytest.raises(IntegrityError):
            b2 = BestModel(
                experiment_id=sample_experiment.id,
                run_id=sample_run.id,
                metric_name="val_log_loss",
                metric_value=0.60,
                rank=1,
            )
            session.add(b2)
            session.flush()

    def test_promotion(self, session, sample_experiment, sample_run):
        """Promote and demote a model."""
        entry = BestModel(
            experiment_id=sample_experiment.id,
            run_id=sample_run.id,
            metric_name="val_log_loss",
            metric_value=0.58,
            rank=1,
        )
        session.add(entry)
        session.flush()

        entry.is_promoted = True
        entry.promoted_at = datetime.now(timezone.utc)
        session.flush()
        assert entry.is_promoted is True

        entry.is_promoted = False
        entry.promoted_at = None
        session.flush()
        assert entry.is_promoted is False

    def test_to_dict(self, session, sample_experiment, sample_run):
        """to_dict returns correct fields."""
        entry = BestModel(
            experiment_id=sample_experiment.id,
            run_id=sample_run.id,
            metric_name="accuracy",
            metric_value=0.85,
            rank=1,
        )
        session.add(entry)
        session.flush()
        d = entry.to_dict()
        assert d["metric_name"] == "accuracy"
        assert d["metric_value"] == 0.85
        assert d["rank"] == 1
        assert d["is_promoted"] is False


class TestModelArtifact:
    """ModelArtifact model tests."""

    def test_create(self, session, sample_experiment, sample_run):
        """Create a model artifact."""
        artifact = ModelArtifact(
            run_id=sample_run.id,
            name="model.joblib",
            uri="/models/model.joblib",
            file_size_bytes=1024,
            artifact_type="model",
        )
        session.add(artifact)
        session.flush()
        assert artifact.id is not None
        assert artifact.name == "model.joblib"

    def test_cascade_delete(self, session, sample_experiment, sample_run):
        """Deleting a run cascades to its artifacts."""
        artifact = ModelArtifact(
            run_id=sample_run.id,
            name="model.joblib",
            uri="/models/model.joblib",
        )
        session.add(artifact)
        session.flush()
        art_id = artifact.id

        session.delete(sample_run)
        session.flush()
        assert session.get(ModelArtifact, art_id) is None

    def test_to_dict(self, session, sample_experiment, sample_run):
        """to_dict returns correct fields."""
        artifact = ModelArtifact(
            run_id=sample_run.id,
            name="preprocessor.pkl",
            uri="/models/preprocessor.pkl",
            artifact_type="preprocessor",
        )
        session.add(artifact)
        session.flush()
        d = artifact.to_dict()
        assert d["name"] == "preprocessor.pkl"
        assert d["artifact_type"] == "preprocessor"


# ── Helper fixture ────────────────────────────────────────


@pytest.fixture
def sample_run(session, sample_experiment):
    """Create a sample run for tests."""
    run = Run.create(
        experiment_id=sample_experiment.id,
        model_type="xgboost",
        run_name="sample_run",
    )
    session.add(run)
    session.flush()
    return run
