"""Integration tests for the Monitor orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.monitoring import Monitor
from src.monitoring.cli import main as cli_main
from tests.test_monitoring.conftest import force_close_connections


class TestMonitorRecord:
    """Test recording various metric types."""

    def test_record_etl(self, monitor: Monitor) -> None:
        row_id = monitor.record_etl(
            pipeline="collect_all",
            duration_seconds=45.2,
            rows_imported=15000,
            download_speed_mbps=12.5,
            source="test",
        )
        assert row_id > 0

    def test_record_etl_from_result(self, monitor: Monitor) -> None:
        row_id = monitor.record_etl_from_result({
            "pipeline": "download",
            "duration_seconds": 30.0,
            "rows": 5000,
            "success": True,
        })
        assert row_id > 0

    def test_record_system(self, monitor: Monitor) -> None:
        row_id = monitor.record_system()
        assert row_id > 0

    def test_record_data_quality(self, monitor: Monitor) -> None:
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        row_id = monitor.record_data_quality(df, source="test")
        assert row_id > 0

    def test_record_cache(self, monitor: Monitor) -> None:
        row_id = monitor.record_cache(hits=100, misses=20, hit_rate=0.8333, entries=500)
        assert row_id > 0

    def test_record_etl_with_error(self, monitor: Monitor) -> None:
        row_id = monitor.record_etl(
            pipeline="failing_pipeline",
            duration_seconds=10.0,
            rows_imported=0,
            success=False,
            error_message="Timeout error",
        )
        assert row_id > 0
        snap = monitor.get_latest_snapshot()
        assert snap["etl"]["success"] is False
        assert "Timeout error" in snap["etl"]["error_message"]


class TestMonitorReports:
    """Test report generation."""

    def test_generate_reports(self, monitor: Monitor) -> None:
        # Record some data first
        monitor.record_etl(pipeline="test", duration_seconds=10.0, rows_imported=100)
        monitor.record_system()
        import pandas as pd
        monitor.record_data_quality(pd.DataFrame({"a": [1, 2, 3]}), source="test")

        results = monitor.generate_reports(days=30)
        assert "html" in results
        assert "json" in results
        assert "csv" in results
        assert "summary" in results

        # Verify files exist
        for key in ["html", "json", "summary"]:
            assert Path(results[key]).exists(), f"{key} file not found: {results[key]}"

        for path_str in results.get("csv", []):
            assert Path(path_str).exists(), f"CSV file not found: {path_str}"

    def test_generate_html_dashboard(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test", duration_seconds=10.0, rows_imported=100)
        path = monitor.html_report.generate(days=30)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "ETL Monitoring Dashboard" in content
        assert "plotly" in content.lower()

    def test_generate_json_report(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test", duration_seconds=10.0)
        path = monitor.json_report.generate(days=30)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "generated_at" in data
        assert "etl_metrics" in data
        assert "trends" in data

    def test_generate_csv_report(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test", duration_seconds=10.0, rows_imported=100)
        monitor.record_system()
        paths = monitor.csv_report.generate(days=30)
        assert len(paths) > 0
        for p in paths:
            content = p.read_text(encoding="utf-8")
            assert "recorded_at" in content

    def test_daily_summary(self, monitor: Monitor) -> None:
        monitor.record_etl(
            pipeline="test",
            duration_seconds=45.2,
            rows_imported=15000,
            download_speed_mbps=12.5,
            success=True,
        )
        summary = monitor.daily_summary()
        assert "ETL Pipeline" in summary
        assert "45.2s" in summary
        assert "15,000" in summary

    def test_daily_summary_empty(self, monitor: Monitor) -> None:
        summary = monitor.daily_summary()
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestMonitorLifecycle:
    """Test monitor lifecycle operations."""

    def test_get_latest_snapshot(self, monitor: Monitor) -> None:
        snap = monitor.get_latest_snapshot()
        assert "timestamp" in snap
        assert snap.get("etl") is None  # No data yet
        assert snap.get("system") is None

    def test_cleanup(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test")
        deleted = monitor.cleanup(retention_days=0)
        assert sum(deleted.values()) > 0

    def test_context_manager(self) -> None:
        tmp = TemporaryDirectory()
        try:
            with Monitor(db_path=str(Path(tmp.name) / "m.db")) as m:
                m.record_etl(pipeline="test")
            force_close_connections()
        finally:
            try:
                tmp.cleanup()
            except PermissionError:
                pass

    def test_background_collection(self) -> None:
        tmp = TemporaryDirectory()
        try:
            m = Monitor(
                db_path=str(Path(tmp.name) / "m.db"),
                auto_collect_system_interval=0.5,
            )
            import time
            time.sleep(1.2)  # Allow 2 collections
            history = m.store.get_system_history(days=1)
            assert len(history) >= 1
            m.close()
            m.store.close()
            force_close_connections()
        finally:
            try:
                tmp.cleanup()
            except PermissionError:
                pass

    def test_background_collection_stop(self) -> None:
        """close() should stop the background thread."""
        tmp = TemporaryDirectory()
        try:
            m = Monitor(
                db_path=str(Path(tmp.name) / "m.db"),
                auto_collect_system_interval=0.2,
            )
            m.close()
            m.store.close()
            force_close_connections()
        finally:
            try:
                tmp.cleanup()
            except PermissionError:
                pass


class TestMonitorCLI:
    """Test CLI commands."""

    def test_cli_summary(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test", duration_seconds=10.0)
        exit_code = cli_main([
            "--db", str(monitor.db_path),
            "--output", str(monitor.output_dir),
            "summary",
        ])
        assert exit_code == 0

    def test_cli_report(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test", duration_seconds=10.0)
        exit_code = cli_main([
            "--db", str(monitor.db_path),
            "--output", str(monitor.output_dir),
            "report", "--days", "30",
        ])
        assert exit_code == 0

    def test_cli_collect(self, monitor: Monitor) -> None:
        exit_code = cli_main([
            "--db", str(monitor.db_path),
            "--output", str(monitor.output_dir),
            "collect",
        ])
        assert exit_code == 0

    def test_cli_cleanup(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test")
        exit_code = cli_main([
            "--db", str(monitor.db_path),
            "--output", str(monitor.output_dir),
            "cleanup", "--retention", "0",
        ])
        assert exit_code == 0

    def test_cli_stats(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test")
        exit_code = cli_main([
            "--db", str(monitor.db_path),
            "--output", str(monitor.output_dir),
            "stats",
        ])
        assert exit_code == 0

    def test_cli_dashboard(self, monitor: Monitor) -> None:
        monitor.record_etl(pipeline="test", duration_seconds=10.0)
        exit_code = cli_main([
            "--db", str(monitor.db_path),
            "--output", str(monitor.output_dir),
            "dashboard", "--days", "30",
        ])
        assert exit_code == 0
