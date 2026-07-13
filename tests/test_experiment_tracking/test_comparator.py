"""
Tests for ExperimentComparator.

Covers
------
- Compare specific runs by ID
- Compare runs within an experiment
- Compare across experiments
- Rank by metric
- Best-per-metric summary
- Dataframe and summary extraction
- Edge cases (empty, missing metrics, etc.)
"""

from __future__ import annotations

import pytest

from src.experiment_tracking.comparator import ExperimentComparator
from src.experiment_tracking.tracker import ExperimentTracker


@pytest.fixture
def populated_experiments(tracker: ExperimentTracker):
    """Create experiments with multiple runs for comparison tests."""
    exp = tracker.create_experiment("comparison_test")

    # Create runs with different metrics
    runs = []
    configs = [
        ("xgboost", {"val_log_loss": 0.55, "val_accuracy": 0.75, "f1_score": 0.72}, 15.2),
        ("lr", {"val_log_loss": 0.62, "val_accuracy": 0.68, "f1_score": 0.65}, 8.1),
        ("random_forest", {"val_log_loss": 0.58, "val_accuracy": 0.71, "f1_score": 0.70}, 25.5),
    ]
    for model_type, metrics, duration in configs:
        run = tracker.start_run(
            exp.id,
            model_type=model_type,
            hyperparameters={"seed": 42},
            random_seed=42,
        )
        tracker.finish_run(run.id, metrics=metrics, duration_seconds=duration)
        runs.append(run)

    return tracker, exp, runs


class TestCompareRuns:
    """Compare specific runs by ID."""

    def test_compare_runs(self, comparator: ExperimentComparator, populated_experiments):
        """Compare specific runs."""
        _, _, runs = populated_experiments
        result = comparator.compare_runs([runs[0].id, runs[1].id])
        assert len(result["runs"]) == 2
        assert "best_by_metric" in result

    def test_compare_runs_best_by_metric(self, comparator: ExperimentComparator, populated_experiments):
        """Identify best run for each metric."""
        _, _, runs = populated_experiments
        result = comparator.compare_runs([r.id for r in runs])

        best = result["best_by_metric"]
        # val_log_loss: lower is better → xgboost (0.55)
        assert best["val_log_loss"]["model_type"] == "xgboost"
        # val_accuracy: higher is better → xgboost (0.75)
        assert best["val_accuracy"]["model_type"] == "xgboost"
        # f1_score: higher is better → xgboost (0.72)
        assert best["f1_score"]["model_type"] == "xgboost"

    def test_compare_runs_empty(self, comparator: ExperimentComparator):
        """Compare empty list returns empty result."""
        result = comparator.compare_runs([])
        assert result["best_by_metric"] == {}
        assert result["runs"] == {}

    def test_compare_runs_nonexistent(self, comparator: ExperimentComparator):
        """Compare nonexistent runs skips them."""
        result = comparator.compare_runs(["nonexistent"])
        assert result["runs"] == {}


class TestCompareInExperiment:
    """Compare runs within an experiment."""

    def test_compare_in_experiment(self, comparator: ExperimentComparator, populated_experiments):
        """Compare all completed runs in an experiment."""
        tracker, exp, runs = populated_experiments
        result = comparator.compare_runs_in_experiment(exp.id)
        assert len(result["runs"]) == 3

    def test_compare_in_experiment_filter_model(self, comparator: ExperimentComparator, populated_experiments):
        """Filter by model type within experiment."""
        tracker, exp, runs = populated_experiments
        result = comparator.compare_runs_in_experiment(exp.id, model_type="xgboost")
        assert len(result["runs"]) == 1
        rid = list(result["runs"].keys())[0]
        assert result["runs"][rid]["model_type"] == "xgboost"

    def test_compare_in_experiment_no_runs(self, comparator: ExperimentComparator, tracker: ExperimentTracker):
        """Experiment with no completed runs returns empty."""
        exp = tracker.create_experiment("empty")
        result = comparator.compare_runs_in_experiment(exp.id)
        assert result["runs"] == {}


class TestCompareAcrossExperiments:
    """Compare runs across experiments."""

    def test_compare_across(self, comparator: ExperimentComparator, tracker: ExperimentTracker):
        """Compare runs across multiple experiments."""
        exp1 = tracker.create_experiment("cross_a")
        exp2 = tracker.create_experiment("cross_b")

        r1 = tracker.start_run(exp1.id, model_type="xgboost")
        tracker.finish_run(r1.id, metrics={"loss": 0.55})

        r2 = tracker.start_run(exp2.id, model_type="lr")
        tracker.finish_run(r2.id, metrics={"loss": 0.60})

        result = comparator.compare_across_experiments([exp1.id, exp2.id])
        assert len(result["runs"]) == 2


class TestRankByMetric:
    """Rank runs by a specific metric."""

    def test_rank_by_metric(self, comparator: ExperimentComparator, populated_experiments):
        """Rank runs by val_log_loss."""
        tracker, exp, runs = populated_experiments
        ranked = comparator.rank_by_metric(experiment_id=exp.id, metric="val_log_loss")
        assert len(ranked) == 3
        assert ranked[0]["rank"] == 1
        assert ranked[0]["metric_value"] == 0.55  # Best (lowest loss)
        assert ranked[2]["rank"] == 3

    def test_rank_by_metric_accuracy(self, comparator: ExperimentComparator, populated_experiments):
        """Rank by higher-is-better metric."""
        tracker, exp, runs = populated_experiments
        ranked = comparator.rank_by_metric(experiment_id=exp.id, metric="val_accuracy")
        assert ranked[0]["metric_value"] == 0.75  # Best (highest accuracy)

    def test_rank_by_metric_no_results(self, comparator: ExperimentComparator, populated_experiments):
        """Rank returns empty for nonexistent metric."""
        tracker, exp, runs = populated_experiments
        ranked = comparator.rank_by_metric(experiment_id=exp.id, metric="nonexistent")
        assert ranked == []


class TestExportHelpers:
    """Comparison export helpers."""

    def test_to_dataframe(self, comparator: ExperimentComparator, populated_experiments):
        """Convert comparison result to DataFrame."""
        tracker, exp, runs = populated_experiments
        result = comparator.compare_runs([runs[0].id, runs[1].id])
        df = comparator.to_dataframe(result)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "model_type" in df.columns
        assert "val_log_loss" in df.columns

    def test_best_summary(self, comparator: ExperimentComparator, populated_experiments):
        """Extract best-per-metric summary."""
        tracker, exp, runs = populated_experiments
        result = comparator.compare_runs([r.id for r in runs])
        summary = comparator.best_summary(result)
        assert len(summary) == 3  # Three metrics
        # Should be sorted by metric name
        metric_names = [s["metric"] for s in summary]
        assert metric_names == sorted(metric_names)
