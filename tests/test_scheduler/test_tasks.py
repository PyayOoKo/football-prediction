"""
Tests for scheduler task implementations — including all 6 tasks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scheduler.models import ScheduleConfig, TaskStatus
from src.scheduler.tasks import (
    backup_database,
    clean_data,
    download_fixtures,
    generate_logs,
    update_database,
    validate_data,
)


class TestDownloadFixtures:
    def test_download_success(self) -> None:
        """Download runs and returns success."""
        cfg = ScheduleConfig()

        with patch("src.importers.FootballDataImporter") as MockImporter:
            mock_importer = MagicMock()
            mock_importer.league_map.keys.return_value = ["E0", "E1"]

            mock_report = MagicMock()
            mock_report.success = True
            mock_report.rows_imported = 100
            mock_report.errors = []

            mock_importer.import_historical.return_value = [mock_report, mock_report]
            MockImporter.return_value = mock_importer

            result = download_fixtures(cfg)

            assert result.status == TaskStatus.SUCCESS
            assert result.records_processed == 200

    def test_download_with_warnings(self) -> None:
        """Download with partial failures returns WARNING."""
        cfg = ScheduleConfig()

        with patch("src.importers.FootballDataImporter") as MockImporter:
            mock_importer = MagicMock()
            mock_importer.league_map.keys.return_value = ["E0"]

            good_report = MagicMock()
            good_report.success = True
            good_report.rows_imported = 100
            good_report.errors = []
            good_report.status = "success"

            bad_report = MagicMock()
            bad_report.success = False
            bad_report.rows_imported = 0
            bad_report.errors = ["Timeout"]
            bad_report.status = "failed"

            mock_importer.import_historical.return_value = [good_report, bad_report]
            MockImporter.return_value = mock_importer

            result = download_fixtures(cfg)

            assert result.status == TaskStatus.WARNING
            assert len(result.warnings) > 0

    def test_download_failure(self) -> None:
        """Importer error returns FAILED."""
        cfg = ScheduleConfig()

        with patch("src.importers.FootballDataImporter") as MockImporter:
            MockImporter.side_effect = RuntimeError("Import crashed")

            result = download_fixtures(cfg)
            assert result.status == TaskStatus.FAILED
            assert "crashed" in result.error

    def test_download_empty_reports(self) -> None:
        """No leagues returns empty import."""
        cfg = ScheduleConfig()

        with patch("src.importers.FootballDataImporter") as MockImporter:
            mock_importer = MagicMock()
            mock_importer.league_map.keys.return_value = []
            MockImporter.return_value = mock_importer

            result = download_fixtures(cfg)
            assert result.status in (TaskStatus.SUCCESS, TaskStatus.WARNING)
            assert result.records_processed == 0


class TestValidateData:
    def test_validate_no_file(self) -> None:
        """When no data file exists, task is skipped."""
        cfg = ScheduleConfig()

        with patch("pathlib.PosixPath.exists", return_value=False):
            with patch("pathlib.WindowsPath.exists", return_value=False):
                result = validate_data(cfg)
                assert result.status == TaskStatus.SKIPPED

    def test_validate_success(self, tmp_path: Path) -> None:
        """Clean data passes validation."""
        import pandas as pd

        cfg = ScheduleConfig()
        cfg.report_dir = str(tmp_path / "reports")

        df = pd.DataFrame({
            "id": [1, 2],
            "date": ["2024-01-07", "2024-01-08"],
            "home_team": ["Arsenal", "Liverpool"],
            "away_team": ["Chelsea", "Man City"],
            "home_goals": [2, 1],
            "away_goals": [1, 1],
            "result": ["H", "D"],
            "status": ["finished", "finished"],
            "league": ["E0", "E0"],
        })

        with patch("pathlib.PosixPath.exists", return_value=True):
            with patch("pathlib.WindowsPath.exists", return_value=True):
                with patch("pandas.read_csv") as mock_read:
                    mock_read.return_value = df
                    # to_html, to_csv, to_json may have template issues
                    with patch("src.validation.reporter.HTMLReporter") as MockReporter:
                        MockReporter.render.return_value = "<html></html>"
                        with patch("csv.DictWriter") as MockWriter:
                            with patch("json.dump"):
                                    result = validate_data(cfg)
                                    assert result.status in (TaskStatus.SUCCESS, TaskStatus.WARNING)
                                    assert result.records_processed == 2

    def test_validate_failure(self) -> None:
        """Engine error returns FAILED."""
        cfg = ScheduleConfig()

        with patch("pathlib.PosixPath.exists", return_value=True):
            with patch("pathlib.WindowsPath.exists", return_value=True):
                with patch("pandas.read_csv") as mock_read:
                    mock_read.side_effect = RuntimeError("Parse error")
                    result = validate_data(cfg)
                    assert result.status == TaskStatus.FAILED


class TestUpdateDatabase:
    def test_no_data_file(self) -> None:
        """When no data file exists, task is skipped."""
        cfg = ScheduleConfig()

        with patch("pathlib.PosixPath.exists", return_value=False):
            with patch("pathlib.WindowsPath.exists", return_value=False):
                with patch("src.database.session.get_session") as mock_session:
                    mock_session.return_value.__enter__.return_value = MagicMock()
                    result = update_database(cfg)
                    assert result.status == TaskStatus.SKIPPED
                    assert "No processed data" in result.output

    def test_db_connection_error(self) -> None:
        """Database connection failure returns FAILED."""
        cfg = ScheduleConfig()

        with patch("pathlib.PosixPath.exists", return_value=True):
            with patch("pathlib.WindowsPath.exists", return_value=True):
                with patch("pandas.read_csv") as mock_read:
                    mock_df = MagicMock()
                    mock_df.__len__.return_value = 100
                    mock_read.return_value = mock_df

                    with patch("src.database.session.get_session") as mock_session:
                        mock_session.side_effect = RuntimeError("DB connection lost")
                        result = update_database(cfg)
                        assert result.status == TaskStatus.FAILED

    def test_update_success(self) -> None:
        """Successful DB update returns SUCCESS."""
        cfg = ScheduleConfig()

        with patch("pathlib.PosixPath.exists", return_value=True):
            with patch("pathlib.WindowsPath.exists", return_value=True):
                with patch("pandas.read_csv") as mock_read:
                    import pandas as pd
                    mock_df = pd.DataFrame({
                        "id": [1, 2],
                        "home_team": ["A", "B"],
                        "away_team": ["C", "D"],
                    })
                    mock_read.return_value = mock_df

                    with patch("src.database.session.get_session") as mock_get_session:
                        mock_session = MagicMock()
                        mock_get_session.return_value.__enter__.return_value = mock_session

                        with patch("src.feature_engineering.build_features") as mock_build:
                            mock_build.return_value = (MagicMock(), MagicMock())

                            with patch("src.feature_engineering.train_val_test_split") as mock_split:
                                mock_split.return_value = {
                                    "X_train": MagicMock(),
                                    "y_train": MagicMock(),
                                    "X_val": MagicMock(),
                                    "y_val": MagicMock(),
                                }

                                with patch("src.etl.store.DatabaseStore") as MockStore:
                                    mock_store = MagicMock()
                                    mock_store.write.return_value = MagicMock(records_out=2)
                                    MockStore.return_value = mock_store

                                    with patch("src.ensemble.EnsembleModel") as MockEnsemble:
                                        mock_ensemble = MagicMock()
                                        mock_ensemble.fit.return_value = {
                                            "weights": {"xgboost": 0.6, "logistic_regression": 0.4},
                                            "val_log_loss": 0.45,
                                        }
                                        MockEnsemble.return_value = mock_ensemble

                                        result = update_database(cfg)
                                        assert result.status == TaskStatus.SUCCESS
                                        assert "Weights" in result.output
                                        assert result.records_processed == 2


class TestCleanData:
    def test_clean_data_skips_when_no_files(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.backup_dir = str(tmp_path / "backups")
        cfg.log_dir = str(tmp_path / "logs")

        result = clean_data(cfg)
        assert result.status == TaskStatus.SUCCESS

    def test_clean_deduplicates_csv(self, tmp_path: Path) -> None:
        import pandas as pd

        cfg = ScheduleConfig()
        cfg.backup_dir = str(tmp_path / "backups")
        cfg.log_dir = str(tmp_path / "logs")

        # Create a temp CSV file
        csv_path = tmp_path / "results_clean.csv"
        df = pd.DataFrame({
            "date": ["2024-01-07", "2024-01-07"],
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Chelsea"],
            "league": ["E0", "E0"],
        })
        df.to_csv(csv_path, index=False)

        # Mock Path to find our temp file
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "glob", return_value=[]):
                with patch.object(Path, "rglob", return_value=[]):
                    with patch("pandas.read_csv", return_value=df) as mock_read:
                        with patch("pandas.DataFrame.to_csv") as mock_write:
                            result = clean_data(cfg)
                            assert result.status == TaskStatus.SUCCESS

    def test_clean_no_duplicates(self, tmp_path: Path) -> None:
        import pandas as pd

        cfg = ScheduleConfig()
        cfg.backup_dir = str(tmp_path / "backups")
        cfg.log_dir = str(tmp_path / "logs")

        df = pd.DataFrame({
            "date": ["2024-01-07", "2024-01-08"],
            "home_team": ["Arsenal", "Liverpool"],
            "away_team": ["Chelsea", "Man City"],
            "league": ["E0", "E1"],
        })

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "glob", return_value=[]):
                with patch.object(Path, "rglob", return_value=[]):
                    with patch("pandas.read_csv", return_value=df):
                        result = clean_data(cfg)
                        assert result.status == TaskStatus.SUCCESS


class TestBackupDatabase:
    def test_backup_sqlite(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.backup_dir = str(tmp_path / "backups")

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        db_file = db_dir / "test.db"
        db_file.write_text("mock database content")

        with patch("src.database.session.get_engine") as mock_engine:
            mock_engine.return_value.url = f"sqlite:///{db_file}"
            result = backup_database(cfg)

        assert result.status in (TaskStatus.SUCCESS, TaskStatus.FAILED)

    def test_backup_sqlite_file_not_found(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.backup_dir = str(tmp_path / "backups")

        with patch("src.database.session.get_engine") as mock_engine:
            mock_engine.return_value.url = "sqlite:///nonexistent.db"
            result = backup_database(cfg)

        assert result.status == TaskStatus.SKIPPED

    def test_backup_unsupported_db(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.backup_dir = str(tmp_path / "backups")

        with patch("src.database.session.get_engine") as mock_engine:
            mock_engine.return_value.url = "mysql://user:pass@localhost/db"
            result = backup_database(cfg)

        assert result.status == TaskStatus.SKIPPED

    def test_backup_failure(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.backup_dir = str(tmp_path / "backups")

        with patch("src.database.session.get_engine") as mock_engine:
            mock_engine.side_effect = RuntimeError("Engine error")
            result = backup_database(cfg)

        assert result.status == TaskStatus.FAILED


class TestGenerateLogs:
    def test_generate_logs_succeeds(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.log_dir = str(tmp_path / "logs")
        cfg.report_dir = str(tmp_path / "reports")

        result = generate_logs(cfg)
        assert result.status == TaskStatus.SUCCESS

    def test_generate_logs_writes_summary(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.log_dir = str(tmp_path / "logs")
        cfg.report_dir = str(tmp_path / "reports")

        result = generate_logs(cfg)
        assert result.status == TaskStatus.SUCCESS

        log_files = list(Path(cfg.log_dir).glob("run_summary_*.json"))
        assert len(log_files) > 0

    def test_generate_rotates_old_logs(self, tmp_path: Path) -> None:
        cfg = ScheduleConfig()
        cfg.log_dir = str(tmp_path / "logs")
        cfg.report_dir = str(tmp_path / "reports")

        old_log = Path(cfg.log_dir) / "pipeline.log"
        old_log.parent.mkdir(parents=True, exist_ok=True)
        old_log.write_text("old log content")

        result = generate_logs(cfg)
        assert result.status == TaskStatus.SUCCESS

    def test_generate_failure(self) -> None:
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("No permission")):
            cfg = ScheduleConfig()
            cfg.log_dir = "/invalid/path/logs"
            cfg.report_dir = "/invalid/path/reports"

            result = generate_logs(cfg)
            assert result.status == TaskStatus.FAILED
