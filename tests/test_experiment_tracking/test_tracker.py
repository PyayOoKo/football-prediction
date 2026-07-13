"""
Tests for ExperimentTracker.

Covers
------
- Create, get, update, delete experiments
- Start, finish, fail runs
- Run context manager
- Resume runs
- Log artifacts
- List/filter runs and experiments
- Error handling
"""

from __future__ import annotations

import pytest

from src.experiment_tracking.models import Experiment, Run
from src.experiment_tracking.tracker import ExperimentTracker


class TestExperimentLifecycle:
    """Experiment CRUD and listing."""

    def test_create_experiment(self, tracker: ExperimentTracker):
        """Create a basic experiment."""
        exp = tracker.create_experiment("test_lifecycle")
        assert exp.name == "test_lifecycle"
        assert exp.id is not None
        assert exp.created_at is not None

    def test_create_experiment_empty_name(self, tracker: ExperimentTracker):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            tracker.create_experiment("")
        with pytest.raises(ValueError, match="cannot be empty"):
            tracker.create_experiment("   ")

    def test_create_experiment_with_all_params(self, tracker: ExperimentTracker):
        """Create experiment with all optional fields."""
        exp = tracker.create_experiment(
            "full_test",
            description="Full test experiment",
            dataset_version="v1.0",
            feature_version="v2.0",
            model_version="xgboost_v3",
            notes="Testing full creation",
            tags={"env": "test", "team": "ml"},
        )
        assert exp.description == "Full test experiment"
        assert exp.dataset_version == "v1.0"
        assert exp.feature_version == "v2.0"
        assert exp.model_version == "xgboost_v3"
        assert exp.notes == "Testing full creation"
        assert exp.tags == {"env": "test", "team": "ml"}

    def test_get_experiment(self, tracker: ExperimentTracker):
        """Get experiment by ID."""
        exp = tracker.create_experiment("get_test")
        fetched = tracker.get_experiment(exp.id)
        assert fetched is not None
        assert fetched.id == exp.id
        assert fetched.name == "get_test"

    def test_get_experiment_not_found(self, tracker: ExperimentTracker):
        """Get nonexistent experiment returns None."""
        assert tracker.get_experiment("nonexistent") is None

    def test_list_experiments(self, tracker: ExperimentTracker):
        """List experiments returns all in desc order."""
        tracker.create_experiment("first")
        tracker.create_experiment("second")
        exps = tracker.list_experiments(limit=10)
        assert len(exps) == 2
        assert exps[0].name == "second"  # Most recent first

    def test_list_experiments_empty(self, tracker: ExperimentTracker):
        """List with no experiments returns empty list."""
        assert tracker.list_experiments() == []

    def test_list_experiments_filter_by_tag(self, tracker: ExperimentTracker):
        """Filter experiments by tag key and value."""
        tracker.create_experiment("a", tags={"env": "prod"})
        tracker.create_experiment("b", tags={"env": "test"})

        prod_exps = tracker.list_experiments(tag_key="env", tag_value="prod")
        assert len(prod_exps) == 1
        assert prod_exps[0].name == "a"

    def test_update_experiment(self, tracker: ExperimentTracker):
        """Update experiment fields."""
        exp = tracker.create_experiment("update_test")
        updated = tracker.update_experiment(exp.id, notes="Updated notes")
        assert updated.notes == "Updated notes"

        # Get fresh copy
        fetched = tracker.get_experiment(exp.id)
        assert fetched.notes == "Updated notes"

    def test_update_experiment_not_found(self, tracker: ExperimentTracker):
        """Update nonexistent experiment raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tracker.update_experiment("nonexistent", notes="x")

    def test_delete_experiment(self, tracker: ExperimentTracker):
        """Delete an experiment."""
        exp = tracker.create_experiment("delete_test")
        assert tracker.delete_experiment(exp.id) is True
        assert tracker.get_experiment(exp.id) is None

    def test_delete_experiment_not_found(self, tracker: ExperimentTracker):
        """Delete nonexistent experiment returns False."""
        assert tracker.delete_experiment("nonexistent") is False


class TestRunLifecycle:
    """Run start/finish/fail lifecycle."""

    def test_start_run(self, tracker: ExperimentTracker):
        """Start a basic run."""
        exp = tracker.create_experiment("run_test")
        run = tracker.start_run(exp.id, model_type="xgboost")
        assert run.status == "running"
        assert run.model_type == "xgboost"
        assert run.hardware is not None  # Auto-captured

    def test_start_run_with_params(self, tracker: ExperimentTracker):
        """Start a run with all parameters."""
        exp = tracker.create_experiment("params_test")
        run = tracker.start_run(
            exp.id,
            model_type="logistic_regression",
            run_name="lr_v1",
            hyperparameters={"C": 1.0},
            random_seed=42,
            notes="First try",
            tags={"fold": "1"},
        )
        assert run.run_name == "lr_v1"
        assert run.hyperparameters == {"C": 1.0}
        assert run.random_seed == 42

    def test_finish_run(self, tracker: ExperimentTracker):
        """Finish a run with metrics."""
        exp = tracker.create_experiment("finish_test")
        run = tracker.start_run(exp.id, model_type="xgboost")
        finished = tracker.finish_run(
            run.id,
            metrics={"val_log_loss": 0.58, "val_accuracy": 0.72},
            duration_seconds=12.5,
            notes="Good results",
        )
        assert finished.status == "completed"
        assert finished.metrics == {"val_log_loss": 0.58, "val_accuracy": 0.72}
        assert finished.training_duration_seconds == 12.5
        assert "Good results" in (finished.notes or "")

    def test_finish_run_auto_duration(self, tracker: ExperimentTracker):
        """Auto-compute duration from started_at."""
        exp = tracker.create_experiment("auto_dur")
        run = tracker.start_run(exp.id, model_type="xgboost")
        import time
        time.sleep(0.05)
        finished = tracker.finish_run(run.id, metrics={"loss": 0.5})
        assert finished.training_duration_seconds is not None
        assert finished.training_duration_seconds > 0.01

    def test_fail_run(self, tracker: ExperimentTracker):
        """Fail a run with error message."""
        exp = tracker.create_experiment("fail_test")
        run = tracker.start_run(exp.id, model_type="xgboost")

        failed = tracker.fail_run(run.id, error="CUDA out of memory")
        assert failed.status == "failed"
        assert failed.error_message == "CUDA out of memory"

    def test_fail_run_auto_duration(self, tracker: ExperimentTracker):
        """Auto-compute duration on failure."""
        exp = tracker.create_experiment("fail_auto")
        run = tracker.start_run(exp.id, model_type="xgboost")
        import time
        time.sleep(0.05)
        failed = tracker.fail_run(run.id, error="Error")
        assert failed.training_duration_seconds is not None
        assert failed.training_duration_seconds > 0.01

    def test_get_run(self, tracker: ExperimentTracker):
        """Get a run by ID."""
        exp = tracker.create_experiment("get_run")
        run = tracker.start_run(exp.id, model_type="xgboost")
        fetched = tracker.get_run(run.id)
        assert fetched is not None
        assert fetched.id == run.id
        assert fetched.model_type == "xgboost"

    def test_get_run_not_found(self, tracker: ExperimentTracker):
        """Get nonexistent run returns None."""
        assert tracker.get_run("nonexistent") is None

    def test_list_runs(self, tracker: ExperimentTracker):
        """List runs with optional filters."""
        exp = tracker.create_experiment("list_runs")
        tracker.start_run(exp.id, model_type="xgboost")
        tracker.start_run(exp.id, model_type="lr")

        all_runs = tracker.list_runs(experiment_id=exp.id)
        assert len(all_runs) == 2

        xgb_runs = tracker.list_runs(experiment_id=exp.id, model_type="xgboost")
        assert len(xgb_runs) == 1
        assert xgb_runs[0].model_type == "xgboost"

    def test_resume_run(self, tracker: ExperimentTracker):
        """Resume a failed or running run."""
        exp = tracker.create_experiment("resume")
        run = tracker.start_run(exp.id, model_type="xgboost")
        tracker.fail_run(run.id, error="Timeout")

        resumed = tracker.resume_run(run.id)
        assert resumed.status == "running"
        assert resumed.error_message is None
        assert resumed.finished_at is None

    def test_resume_completed_run_raises(self, tracker: ExperimentTracker):
        """Cannot resume a completed run."""
        exp = tracker.create_experiment("resume_fail")
        run = tracker.start_run(exp.id, model_type="xgboost")
        tracker.finish_run(run.id, metrics={"loss": 0.5})

        with pytest.raises(ValueError, match="Cannot resume completed"):
            tracker.resume_run(run.id)

    def test_resume_nonexistent_run(self, tracker: ExperimentTracker):
        """Resume nonexistent run raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tracker.resume_run("nonexistent")


class TestRunContextManager:
    """Context manager for safe run lifecycle."""

    def test_context_manager_success(self, tracker: ExperimentTracker):
        """Context manager starts and finishes run on success."""
        exp = tracker.create_experiment("ctx_success")
        with tracker.run(exp.id, model_type="xgboost") as run:
            assert run.status == "running"
            tracker.finish_run(run.id, metrics={"loss": 0.5})

        # Verify run is still completed
        fetched = tracker.get_run(run.id)
        assert fetched.status == "completed"

    def test_context_manager_failure(self, tracker: ExperimentTracker):
        """Context manager auto-fails run on exception."""
        exp = tracker.create_experiment("ctx_fail")
        try:
            with tracker.run(exp.id, model_type="xgboost") as run:
                raise RuntimeError("Training crashed")
        except RuntimeError:
            pass

        fetched = tracker.get_run(run.id)
        assert fetched.status == "failed"
        assert "RuntimeError" in (fetched.error_message or "")

    def test_context_manager_with_params(self, tracker: ExperimentTracker):
        """Context manager passes parameters correctly."""
        exp = tracker.create_experiment("ctx_params")
        with tracker.run(
            exp.id,
            model_type="lr",
            run_name="ctx_lr",
            hyperparameters={"C": 0.1},
            random_seed=42,
        ) as run:
            assert run.run_name == "ctx_lr"
            assert run.hyperparameters == {"C": 0.1}
            assert run.random_seed == 42


class TestArtifacts:
    """Artifact logging."""

    def test_log_artifact(self, tracker: ExperimentTracker):
        """Log an artifact to a run."""
        exp = tracker.create_experiment("art_test")
        run = tracker.start_run(exp.id, model_type="xgboost")

        art = tracker.log_artifact(
            run.id,
            name="model.joblib",
            uri="/models/model.joblib",
            file_size_bytes=2048,
            artifact_type="model",
        )
        assert art.name == "model.joblib"
        assert art.uri == "/models/model.joblib"
        assert art.file_size_bytes == 2048

    def test_log_multiple_artifacts(self, tracker: ExperimentTracker):
        """Log multiple artifacts to the same run."""
        exp = tracker.create_experiment("multi_art")
        run = tracker.start_run(exp.id, model_type="xgboost")

        tracker.log_artifact(run.id, "model.joblib", "/models/model.joblib")
        tracker.log_artifact(run.id, "preprocessor.pkl", "/models/preprocessor.pkl")
        tracker.log_artifact(run.id, "encoder.pkl", "/models/encoder.pkl")

        arts = tracker.list_artifacts(run.id)
        assert len(arts) == 3

    def test_list_artifacts_empty(self, tracker: ExperimentTracker):
        """List artifacts for run with none."""
        exp = tracker.create_experiment("empty_art")
        run = tracker.start_run(exp.id, model_type="xgboost")
        assert tracker.list_artifacts(run.id) == []


class TestErrorHandling:
    """Error handling for edge cases."""

    def test_fail_nonexistent_run(self, tracker: ExperimentTracker):
        """Fail nonexistent run raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tracker.fail_run("nonexistent", error="Error")

    def test_finish_nonexistent_run(self, tracker: ExperimentTracker):
        """Finish nonexistent run raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tracker.finish_run("nonexistent", metrics={"loss": 0.5})

    def test_log_artifact_nonexistent_run(self, tracker: ExperimentTracker):
        """Log artifact to nonexistent run raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            tracker.log_artifact("nonexistent", "model.joblib", "/tmp/model.joblib")

    def test_start_run_nonexistent_experiment(self, tracker: ExperimentTracker):
        """Start run with nonexistent experiment raises IntegrityError."""
        with pytest.raises(Exception):
            tracker.start_run("nonexistent", model_type="xgboost")
