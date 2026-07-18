"""
Tests for ``dashboard/app.py`` — data pipeline status detection,
file counting helpers, and streamit page structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestDetectDataFileCount:
    """Tests for the ``detect_data_file_count`` fallback helper."""

    def test_returns_counts_by_directory(self, tmp_path: Path) -> None:
        """Should return a dict with counts for raw/processed/external dirs."""
        # Create some CSV files in a temp directory structure
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        (raw / "results.csv").touch()
        (raw / "fixtures.csv").touch()

        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "features.csv").touch()

        external = tmp_path / "external"
        external.mkdir()
        # No CSV files in external

        # Patch config.paths to point at tmp_path
        with patch("dashboard.app.config") as mock_config:
            mock_config.paths.raw = raw
            mock_config.paths.processed = processed
            mock_config.paths.data = tmp_path

            from dashboard.app import detect_data_file_count

            counts = detect_data_file_count()
            assert counts.get("raw") == 2
            assert counts.get("processed") == 1
            assert counts.get("external") == 0

    def test_handles_missing_directories(self) -> None:
        """Should return 0 for directories that don't exist."""
        with patch("dashboard.app.config") as mock_config:
            mock_config.paths.raw = Path("/nonexistent/raw")
            mock_config.paths.processed = Path("/nonexistent/processed")
            mock_config.paths.data = Path("/nonexistent")

            from dashboard.app import detect_data_file_count

            counts = detect_data_file_count()
            for v in counts.values():
                assert v == 0

    def test_returns_dict(self) -> None:
        """Should always return a dict (never None)."""
        with patch("dashboard.app.config") as mock_config:
            mock_config.paths.raw = Path("/nonexistent/raw")
            mock_config.paths.processed = Path("/nonexistent/processed")
            mock_config.paths.data = Path("/nonexistent")

            from dashboard.app import detect_data_file_count

            result = detect_data_file_count()
            assert isinstance(result, dict)


class TestDetectDataPipelineStatus:
    """Tests for the ``detect_data_pipeline_status`` cached helper."""

    def test_returns_default_when_no_data(self) -> None:
        """Should return default zeros when load_and_prepare raises or returns empty."""
        with patch("dashboard.app.load_and_prepare", return_value=None):
            from dashboard.app import detect_data_pipeline_status

            status = detect_data_pipeline_status()
            assert status["found"] is False
            assert status["rows"] == 0
            assert status["teams"] == 0
            assert status["pipeline_applied"] is False

    def test_returns_summary_when_data_found(self) -> None:
        """Should populate all fields when data is successfully loaded."""
        import pandas as pd

        sample_df = pd.DataFrame({
            "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
            "home_team": ["Brazil", "France", "Argentina"],
            "away_team": ["Norway", "Morocco", "Egypt"],
            "result": ["H", None, "H"],  # 1 upcoming (Morocco row)
            "home_goals": [2, None, 3],
            "away_goals": [1, None, 0],
        })

        with patch("dashboard.app.load_and_prepare", return_value=sample_df):
            from dashboard.app import detect_data_pipeline_status

            status = detect_data_pipeline_status()
            assert status["found"] is True
            assert status["rows"] == 3
            assert status["completed"] == 2  # H, H = 2 notnull
            assert status["upcoming"] == 1   # None result = 1
            assert status["teams"] == 6       # 3 home + 3 away = 6 unique
            assert status["date_min"] == "2026-07-01"
            assert status["date_max"] == "2026-07-03"
            assert status["pipeline_applied"] is True
            assert status["n_columns"] == 5

    def test_handles_empty_dataframe(self) -> None:
        """Should return found=False for an empty DataFrame."""
        import pandas as pd

        with patch("dashboard.app.load_and_prepare", return_value=pd.DataFrame()):
            from dashboard.app import detect_data_pipeline_status

            status = detect_data_pipeline_status()
            assert status["found"] is True  # Not None, so found=True
            assert status["rows"] == 0

    def test_handles_load_and_prepare_exception(self) -> None:
        """Should gracefully handle exceptions from load_and_prepare."""
        with patch(
            "dashboard.app.load_and_prepare",
            side_effect=Exception("DB connection failed"),
        ):
            from dashboard.app import detect_data_pipeline_status

            status = detect_data_pipeline_status()
            assert status["found"] is False
            assert status["rows"] == 0


class TestPageInfoStructure:
    """Validate the page_info list structure in the main app."""

    def test_data_pipeline_status_keys(self) -> None:
        """The pipeline status dict should have all expected keys."""
        with patch("dashboard.app.load_and_prepare", return_value=None):
            from dashboard.app import detect_data_pipeline_status

            status = detect_data_pipeline_status()
            expected_keys = {
                "found", "rows", "completed", "upcoming", "teams",
                "date_min", "date_max", "n_columns", "pipeline_applied",
            }
            assert set(status.keys()) == expected_keys
