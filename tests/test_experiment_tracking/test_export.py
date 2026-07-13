"""
Tests for the experiment tracking export module.

Covers
------
- JSON export: structure, optional file output
- CSV export: table structure, metric columns
- HTML export: self-contained report, chart data
- Empty/no-experiments edge cases
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.experiment_tracking.export import (
    export_csv,
    export_html,
    export_json,
    html_escape,
)
from src.experiment_tracking.tracker import ExperimentTracker


@pytest.fixture
def populated_data(tracker: ExperimentTracker):
    """Create experiments with runs for export testing."""
    exp1 = tracker.create_experiment(
        "export_test_a",
        dataset_version="v1",
        feature_version="v2",
        tags={"env": "test"},
    )
    exp2 = tracker.create_experiment("export_test_b")

    # Runs for exp1
    r1 = tracker.start_run(exp1.id, model_type="xgboost", run_name="xgb_v1")
    tracker.finish_run(r1.id, metrics={
        "val_log_loss": 0.55, "val_accuracy": 0.75,
    }, duration_seconds=12.5)

    r2 = tracker.start_run(exp1.id, model_type="lr", run_name="lr_v1")
    tracker.finish_run(r2.id, metrics={
        "val_log_loss": 0.62, "val_accuracy": 0.68,
    }, duration_seconds=8.2)

    # Artifact for r1
    tracker.log_artifact(
        r1.id, "model.joblib", "/models/model.joblib",
        file_size_bytes=2048, artifact_type="model",
    )

    return tracker, exp1, exp2, r1, r2


class TestExportJSON:
    """JSON export tests."""

    def test_export_json_string(self, session, populated_data):
        """Export returns valid JSON string."""
        tracker, exp1, exp2, r1, r2 = populated_data
        json_str = export_json(session)
        data = json.loads(json_str)

        assert "exported_at" in data
        assert data["experiment_count"] == 2
        assert len(data["experiments"]) == 2

    def test_export_json_filter_experiment(self, session, populated_data):
        """Export filtered by experiment ID."""
        tracker, exp1, exp2, r1, r2 = populated_data
        json_str = export_json(session, experiment_id=exp1.id)
        data = json.loads(json_str)
        assert data["experiment_count"] == 1
        assert data["experiments"][0]["name"] == "export_test_a"

    def test_export_json_empty(self, session):
        """Export with no experiments."""
        json_str = export_json(session)
        data = json.loads(json_str)
        assert data["experiment_count"] == 0

    def test_export_json_to_file(self, session, populated_data):
        """Export JSON writes to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "experiments.json")
            export_json(session, output_path=output)
            assert os.path.exists(output)
            with open(output) as f:
                data = json.load(f)
            assert data["experiment_count"] == 2


class TestExportCSV:
    """CSV export tests."""

    def test_export_csv_structure(self, session, populated_data):
        """Export CSV returns correct tables."""
        tracker, exp1, exp2, r1, r2 = populated_data
        results = export_csv(session)

        assert "experiments.csv" in results
        assert "runs.csv" in results

        # Experiments CSV
        exp_lines = results["experiments.csv"].strip().split("\n")
        assert len(exp_lines) == 3  # Header + 2 experiments

        # Runs CSV
        run_lines = results["runs.csv"].strip().split("\n")
        assert len(run_lines) == 3  # Header + 2 runs

    def test_export_csv_metric_columns(self, session, populated_data):
        """CSV includes metric columns."""
        tracker, exp1, exp2, r1, r2 = populated_data
        results = export_csv(session)

        run_csv = results["runs.csv"]
        assert "val_log_loss" in run_csv
        assert "val_accuracy" in run_csv

    def test_export_csv_empty(self, session):
        """Export CSV with no experiments."""
        results = export_csv(session)
        exp_lines = results["experiments.csv"].strip().split("\n")
        assert len(exp_lines) == 1  # Just header

    def test_export_csv_to_directory(self, session, populated_data):
        """Export CSV writes to directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_csv(session, output_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "experiments.csv"))
            assert os.path.exists(os.path.join(tmpdir, "runs.csv"))


class TestExportHTML:
    """HTML export tests."""

    def test_export_html_structure(self, session, populated_data):
        """Export HTML contains expected sections."""
        tracker, exp1, exp2, r1, r2 = populated_data
        html = export_html(session)

        assert "<!DOCTYPE html>" in html
        assert "ML Experiment Report" in html
        assert "export_test_a" in html
        assert "export_test_b" in html
        assert "chart-container" in html
        assert "Leaderboard" in html
        assert "plotly" in html.lower() or "Plotly" in html

    def test_export_html_highlights_best(self, session, populated_data):
        """HTML highlights best metric values."""
        tracker, exp1, exp2, r1, r2 = populated_data
        html = export_html(session)
        # Best val_log_loss is 0.55 (xgboost)
        assert "0.5500" in html
        # Best val_accuracy is 0.75 (xgboost)
        assert "0.7500" in html

    def test_export_html_empty(self, session):
        """Export HTML with no experiments."""
        html = export_html(session)
        assert "0 experiment" in html or "0experiment" in html

    def test_export_html_to_file(self, session, populated_data):
        """Export HTML writes to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "report.html")
            export_html(session, output_path=output, title="Custom Report")
            assert os.path.exists(output)
            with open(output) as f:
                content = f.read()
            assert "Custom Report" in content

    def test_export_html_custom_title(self, session, populated_data):
        """Export HTML uses custom title."""
        tracker, exp1, exp2, r1, r2 = populated_data
        html = export_html(session, title="My Custom Report")
        assert "My Custom Report" in html


class TestHTMLEscape:
    """HTML escaping utility."""

    def test_escape_ampersand(self):
        assert html_escape("A & B") == "A &amp; B"

    def test_escape_angle_brackets(self):
        assert html_escape("<script>") == "&lt;script&gt;"

    def test_escape_quotes(self):
        text = 'He said "hello"'
        escaped = html_escape(text)
        assert "&quot;" in escaped

    def test_escape_none(self):
        assert html_escape(None) == ""

    def test_escape_empty(self):
        assert html_escape("") == ""
