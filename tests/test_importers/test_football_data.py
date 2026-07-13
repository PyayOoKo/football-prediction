"""
Unit tests for FootballDataImporter — full orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

from src.importers.football_data import FootballDataImporter, ImportReport


# ── ImportReport ──────────────────────────────────────────


class TestImportReport:
    def test_success_property(self) -> None:
        report = ImportReport(status="success")
        assert report.success is True

        report.status = "failed"
        assert report.success is False

    def test_to_dict(self) -> None:
        report = ImportReport(
            league="E0",
            season="2425",
            season_name="2024/2025",
            status="success",
            rows_downloaded=100,
            rows_parsed=95,
            rows_imported=90,
            rows_skipped=5,
            errors=["Minor warning"],
            duration_seconds=12.345,
        )
        d = report.to_dict()
        assert d["league"] == "E0"
        assert d["rows_imported"] == 90
        assert d["duration_seconds"] == 12.35  # Rounded


# ── FootballDataImporter ──────────────────────────────────


class TestFootballDataImporter:
    def test_init_defaults(self) -> None:
        """Importer initialises with sensible defaults."""
        importer = FootballDataImporter()
        assert importer.raw_dir.name == "football-data"
        assert importer.incremental is True
        assert importer.db_batch_size == 500
        assert "E0" in importer.league_map

    @patch("src.importers.football_data.DownloadManager.download_season")
    @patch("src.importers.football_data.DownloadManager.verify_integrity")
    def test_import_single_skipped_on_download_fail(
        self,
        mock_verify,
        mock_download,
    ) -> None:
        """Import marks skipped when download fails."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "404 Not Found"
        mock_download.return_value = mock_result

        importer = FootballDataImporter()
        report = importer._import_single("E0", "2425")

        assert report.status == "skipped"
        assert "404" in report.errors[0]

    @patch("src.importers.football_data.DownloadManager.download_season")
    @patch("src.importers.football_data.DownloadManager.verify_integrity")
    def test_import_single_failed_on_integrity(
        self,
        mock_verify,
        mock_download,
    ) -> None:
        """Import marks failed when integrity check fails."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.row_count = 100
        mock_download.return_value = mock_result
        mock_verify.return_value = False

        importer = FootballDataImporter()
        report = importer._import_single("E0", "2425")

        assert report.status == "failed"
        assert "Integrity" in report.errors[0]

    @patch("src.importers.football_data.DownloadManager.download_current")
    @patch("src.importers.football_data.DownloadManager.verify_integrity")
    def test_import_current_failed_on_download(
        self,
        mock_verify,
        mock_download,
    ) -> None:
        """Current season import marks skipped on download failure."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Timeout"
        mock_download.return_value = mock_result

        importer = FootballDataImporter()
        report = importer._import_current("E0")

        assert report.status == "skipped"
        assert "Timeout" in report.errors[0]

    def test_season_name_conversion(self) -> None:
        """Season code conversion matches expected format."""
        importer = FootballDataImporter()
        assert importer._season_name("2425") == "2024/2025"
        assert importer._season_name("9394") == "1993/1994"
        assert importer._season_name("current") != "current"  # Resolves to real season

    def test_import_historical_empty_leagues(self) -> None:
        """Importing with no leagues returns empty list."""
        importer = FootballDataImporter()
        reports = importer.import_historical(leagues=[], max_seasons=0)
        assert isinstance(reports, list)


# ── ImportReport collected results ────────────────────────


class TestImportResultsCollection:
    def test_multiple_imports_combine(self) -> None:
        """Multiple import reports can be combined and summarised."""
        reports = [
            ImportReport(
                league="E0", season="2424", status="success",
                rows_imported=200,
            ),
            ImportReport(
                league="E0", season="2425", status="success",
                rows_imported=180,
            ),
            ImportReport(
                league="E1", season="2424", status="skipped",
                rows_imported=0,
            ),
        ]

        success_count = sum(1 for r in reports if r.success)
        total_rows = sum(r.rows_imported for r in reports)

        assert success_count == 2
        assert total_rows == 380

    def test_report_dict_summary(self) -> None:
        """Reports work as dict for structured logging."""
        reports = [
            ImportReport(league="E0", status="success", rows_imported=150),
        ]
        dicts = [r.to_dict() for r in reports]
        assert dicts[0]["league"] == "E0"
        assert dicts[0]["rows_imported"] == 150
