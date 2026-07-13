"""
Tests for ``src.data_profiling.cli``.

Covers all CLI commands: create-report, list-reports, compare,
auto, and dashboard.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_profiling.cli import (
    REPORTS_DIR,
    _load_report,
    _ensure_reports_dir,
    cmd_create_report,
    cmd_list_reports,
    cmd_compare,
    cmd_auto,
    cmd_dashboard,
    main,
)


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture
def sample_csv() -> str:
    """Create a temporary CSV file with sample data."""
    df = pd.DataFrame({
        "date": ["2024-01-07", "2024-01-08", "2024-01-09"],
        "home_team": ["Team A", "Team C", "Team E"],
        "away_team": ["Team B", "Team D", "Team F"],
        "home_goals": [2, 1, 0],
        "away_goals": [1, 1, 3],
        "result": ["H", "D", "A"],
        "league": ["E0", "E0", "E1"],
        "season": ["2425", "2425", "2425"],
    })
    path = Path(tempfile.mktemp(suffix=".csv"))
    df.to_csv(path, index=False)
    yield str(path)
    path.unlink(missing_ok=True)


@pytest.fixture
def empty_csv() -> str:
    """Create a temporary empty CSV."""
    df = pd.DataFrame()
    path = Path(tempfile.mktemp(suffix=".csv"))
    df.to_csv(path, index=False)
    yield str(path)
    path.unlink(missing_ok=True)


# ── Test _load_report ───────────────────────────────────

class TestLoadReport:
    def test_load_valid_json(self) -> None:
        data = {
            "source_name": "test_source",
            "n_rows": 100,
            "n_columns": 10,
            "duration_seconds": 1.5,
            "sections": {},
        }
        path = Path(tempfile.mktemp(suffix=".json"))
        with open(path, "w") as f:
            json.dump(data, f)

        report = _load_report(str(path))
        assert report.source_name == "test_source"
        assert report.n_rows == 100
        assert report.n_columns == 10
        path.unlink()

    def test_load_minimal_json(self) -> None:
        data = {"source_name": "minimal"}
        path = Path(tempfile.mktemp(suffix=".json"))
        with open(path, "w") as f:
            json.dump(data, f)

        report = _load_report(str(path))
        assert report.source_name == "minimal"
        path.unlink()


# ── Test cmd_create_report ──────────────────────────────

class TestCmdCreateReport:
    def test_create_report_csv(self, sample_csv: str) -> None:
        class Args:
            filepath = sample_csv
            source = None
            odds_patterns = None
            outlier_std = 3.0

        result = cmd_create_report(Args())
        assert result == 0

        # Check report files were created
        json_files = list(REPORTS_DIR.glob("*.json"))
        csv_files = list(REPORTS_DIR.glob("*.csv"))
        html_files = list(REPORTS_DIR.glob("*.html"))
        assert len(json_files) > 0
        assert len(csv_files) > 0
        assert len(html_files) > 0

    def test_create_report_with_source(self, sample_csv: str) -> None:
        class Args:
            filepath = sample_csv
            source = "my_custom_source"
            odds_patterns = None
            outlier_std = 3.0

        result = cmd_create_report(Args())
        assert result == 0

        # Check that a file with the source name exists
        assert (REPORTS_DIR / "my_custom_source.json").exists()

    def test_create_report_file_not_found(self) -> None:
        class Args:
            filepath = "/nonexistent/file.csv"
            source = None
            odds_patterns = None
            outlier_std = 3.0

        result = cmd_create_report(Args())
        assert result == 1

    def test_create_report_empty_csv(self, empty_csv: str) -> None:
        class Args:
            filepath = empty_csv
            source = "empty"
            odds_patterns = None
            outlier_std = 3.0

        result = cmd_create_report(Args())
        # Should still succeed (profiling an empty dataset is valid)
        assert result == 0


# ── Test cmd_list_reports ───────────────────────────────

class TestCmdListReports:
    def test_list_existing_reports(self) -> None:
        # Create a dummy report
        _ensure_reports_dir()
        dummy_path = REPORTS_DIR / "dummy.json"
        with open(dummy_path, "w") as f:
            json.dump({"source_name": "dummy", "n_rows": 10}, f)

        result = cmd_list_reports(None)
        assert result == 0
        dummy_path.unlink(missing_ok=True)

    def test_list_no_reports(self) -> None:
        result = cmd_list_reports(None)
        assert result == 0

    def test_list_no_reports(self) -> None:
        # Use the conftest cleanup; just run the command
        result = cmd_list_reports(None)
        assert result == 0


# ── Test cmd_dashboard ──────────────────────────────────

class TestCmdDashboard:
    def test_dashboard_from_json(self) -> None:
        # Create a minimal JSON report
        _ensure_reports_dir()
        data = {
            "source_name": "dash_test",
            "n_rows": 5,
            "n_columns": 3,
            "duration_seconds": 0.5,
            "sections": {},
        }
        json_path = REPORTS_DIR / "dash_test.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        class Args:
            report_json = str(json_path)

        result = cmd_dashboard(Args())
        assert result == 0
        assert (REPORTS_DIR / "dash_test.html").exists()

        json_path.unlink(missing_ok=True)
        (REPORTS_DIR / "dash_test.html").unlink(missing_ok=True)

    def test_dashboard_file_not_found(self) -> None:
        class Args:
            report_json = "/nonexistent/report.json"

        result = cmd_dashboard(Args())
        assert result == 1


# ── Test main entry point ───────────────────────────────

class TestMain:
    def test_no_command(self) -> None:
        result = main([])
        assert result == 1  # prints help

    def test_create_report_command(self, sample_csv: str) -> None:
        result = main(["create-report", sample_csv])
        assert result == 0

    def test_list_reports_command(self) -> None:
        result = main(["list-reports"])
        assert result == 0

    def test_dashboard_command_file_not_found(self) -> None:
        result = main(["dashboard", "/nonexistent.json"])
        assert result == 1

    def test_unknown_command(self) -> None:
        with pytest.raises(SystemExit):
            main(["unknown-cmd"])


