"""Tests for the MonitoringStore — SQLite-backed metric persistence."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.monitoring.models import (
    CacheMetric,
    DataQualityMetric,
    ETLMetric,
    SystemMetric,
)
from src.monitoring.store import MonitoringStore
from tests.test_monitoring.conftest import force_close_connections


class TestMonitoringStore:
    """Test all store operations."""

    # ── ETL Metrics ────────────────────────────────────

    def test_record_etl(self, store: MonitoringStore) -> None:
        row_id = store.record_etl(ETLMetric(
            pipeline="collect_all",
            duration_seconds=45.2,
            rows_imported=15000,
            download_speed_mbps=12.5,
            processing_speed_rows_s=320.0,
        ))
        assert row_id > 0

    def test_get_etl_history(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=10.0))
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=20.0))
        history = store.get_etl_history(days=30)
        assert len(history) == 2

    def test_get_etl_history_filtered(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="pipeline_a", duration_seconds=10.0))
        store.record_etl(ETLMetric(pipeline="pipeline_b", duration_seconds=20.0))
        history = store.get_etl_history(days=30, pipeline="pipeline_a")
        assert len(history) == 1
        assert history[0]["pipeline"] == "pipeline_a"

    def test_get_etl_history_empty(self, store: MonitoringStore) -> None:
        history = store.get_etl_history(days=30)
        assert history == []

    # ── System Metrics ─────────────────────────────────

    def test_record_system(self, store: MonitoringStore) -> None:
        row_id = store.record_system(SystemMetric(
            cpu_percent=45.5,
            memory_percent=62.3,
            memory_used_mb=4096.0,
            disk_usage_pct=55.0,
            db_size_mb=128.5,
        ))
        assert row_id > 0

    def test_get_system_history(self, store: MonitoringStore) -> None:
        for i in range(3):
            store.record_system(SystemMetric(cpu_percent=float(i * 10)))
        history = store.get_system_history(days=30)
        assert len(history) == 3

    def test_get_system_history_empty(self, store: MonitoringStore) -> None:
        assert store.get_system_history(days=30) == []

    # ── Data Quality Metrics ───────────────────────────

    def test_record_dq(self, store: MonitoringStore) -> None:
        row_id = store.record_data_quality(DataQualityMetric(
            source="test_source",
            n_rows=1000,
            n_columns=20,
        ))
        assert row_id > 0

    def test_get_dq_history(self, store: MonitoringStore) -> None:
        for i in range(3):
            store.record_data_quality(DataQualityMetric(source="test", n_rows=i * 100))
        history = store.get_data_quality_history(days=30)
        assert len(history) == 3

    # ── Cache Metrics ──────────────────────────────────

    def test_record_cache(self, store: MonitoringStore) -> None:
        row_id = store.record_cache(CacheMetric(hits=100, misses=20, hit_rate=0.8333))
        assert row_id > 0

    def test_get_cache_history(self, store: MonitoringStore) -> None:
        for i in range(3):
            store.record_cache(CacheMetric(hits=i * 10))
        history = store.get_cache_history(days=30)
        assert len(history) == 3

    # ── Snapshot ───────────────────────────────────────

    def test_get_latest_empty(self, store: MonitoringStore) -> None:
        snap = store.get_latest()
        assert snap.etl is None
        assert snap.system is None

    def test_get_latest_with_data(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="p1"))
        store.record_system(SystemMetric(cpu_percent=50.0))
        snap = store.get_latest()
        assert snap.etl is not None
        assert snap.etl.pipeline == "p1"
        assert snap.system is not None
        assert snap.system.cpu_percent == 50.0

    # ── Trends ─────────────────────────────────────────

    def test_get_trends_empty(self, store: MonitoringStore) -> None:
        trends = store.get_trends(days=30)
        assert trends == []

    def test_get_trends(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=10.0, rows_imported=100))
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=20.0, rows_imported=200))
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=30.0, rows_imported=300))
        trends = store.get_trends(days=30)
        assert len(trends) > 0
        # Should have duration_seconds and rows_imported trends
        names = [t.metric_name for t in trends]
        assert "duration_seconds" in names
        assert "rows_imported" in names

    def test_trend_computation(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=10.0))
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=20.0))
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=30.0))
        trends = store.get_trends(days=30)
        dur_trend = [t for t in trends if t.metric_name == "duration_seconds"][0]
        assert dur_trend.direction == "up"
        assert dur_trend.change_pct > 0

    def test_trend_with_single_point(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="test", duration_seconds=10.0))
        trends = store.get_trends(days=30)
        dur_trend = next(t for t in trends if t.metric_name == "duration_seconds")
        assert dur_trend.direction == "stable"
        assert dur_trend.change_pct == 0.0

    # ── Cleanup ────────────────────────────────────────

    def test_cleanup(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="test"))
        deleted = store.cleanup(retention_days=0)
        assert deleted.get("etl_metrics", 0) > 0
        history = store.get_etl_history(days=30)
        assert len(history) == 0

    def test_cleanup_respects_retention(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="test"))
        deleted = store.cleanup(retention_days=90)
        assert deleted.get("etl_metrics", 0) == 0
        history = store.get_etl_history(days=30)
        assert len(history) == 1

    # ── Stats ──────────────────────────────────────────

    def test_get_stats_empty(self, store: MonitoringStore) -> None:
        stats = store.get_stats()
        assert stats["retention_days"] == 90
        assert "tables" in stats
        for table_info in stats["tables"].values():
            assert table_info["count"] == 0

    def test_get_stats_with_data(self, store: MonitoringStore) -> None:
        store.record_etl(ETLMetric(pipeline="p1"))
        store.record_etl(ETLMetric(pipeline="p2"))
        store.record_system(SystemMetric())
        stats = store.get_stats()
        assert stats["tables"]["etl_metrics"]["count"] == 2
        assert stats["tables"]["system_metrics"]["count"] == 1

    # ── Thread Safety ──────────────────────────────────

    def test_thread_safety(self, store: MonitoringStore) -> None:
        """Concurrent writes from multiple threads should not corrupt the DB."""

        def write_etl(i: int) -> int:
            return store.record_etl(ETLMetric(
                pipeline=f"thread_{i}",
                duration_seconds=float(i),
            ))

        n_threads = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(write_etl, i) for i in range(n_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == n_threads
        assert len(set(results)) == n_threads  # All unique IDs

        history = store.get_etl_history(days=30)
        assert len(history) == n_threads

    # ── Edge Cases ─────────────────────────────────────

    def test_empty_db_path_creates_file(self) -> None:
        tmp = TemporaryDirectory()
        try:
            db_path = Path(tmp.name) / "nested" / "monitor.db"
            s = MonitoringStore(db_path=db_path)
            assert db_path.exists()
            s.close()
            force_close_connections()
        finally:
            try:
                tmp.cleanup()
            except PermissionError:
                pass

    def test_multiple_stores_isolated(self) -> None:
        tmp = TemporaryDirectory()
        try:
            db1 = Path(tmp.name) / "db1.db"
            db2 = Path(tmp.name) / "db2.db"
            s1 = MonitoringStore(db_path=db1)
            s2 = MonitoringStore(db_path=db2)

            s1.record_etl(ETLMetric(pipeline="s1"))
            s2.record_etl(ETLMetric(pipeline="s2"))

            assert len(s1.get_etl_history(days=30)) == 1
            assert len(s2.get_etl_history(days=30)) == 1
            assert s1.get_etl_history(days=30)[0]["pipeline"] == "s1"
            assert s2.get_etl_history(days=30)[0]["pipeline"] == "s2"

            s1.close()
            s2.close()
            force_close_connections()
        finally:
            try:
                tmp.cleanup()
            except PermissionError:
                pass

    def test_reinitialization(self, store: MonitoringStore) -> None:
        """Initializing twice should be a no-op."""
        store.record_etl(ETLMetric(pipeline="test"))
        store._init_db()  # Re-init
        assert len(store.get_etl_history(days=30)) == 1
