"""Tests for monitoring data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.monitoring.models import (
    CacheMetric,
    DataQualityMetric,
    ETLMetric,
    MetricSnapshot,
    SystemMetric,
    TrendLine,
)


class TestETLMetric:
    """Test ETLMetric creation and serialization."""

    def test_defaults(self) -> None:
        m = ETLMetric()
        assert m.pipeline == ""
        assert m.duration_seconds == 0.0
        assert m.rows_imported == 0
        assert m.success is True
        assert isinstance(m.timestamp, datetime)

    def test_custom_values(self) -> None:
        m = ETLMetric(
            pipeline="collect_all",
            duration_seconds=45.2,
            rows_imported=15000,
            download_speed_mbps=12.5,
            processing_speed_rows_s=320.0,
            retry_count=2,
            duplicate_pct=0.3,
            missing_values_pct=1.2,
            validation_failures=0,
            source="football-data.co.uk",
            league="E0",
            season="2024-25",
            success=True,
        )
        assert m.pipeline == "collect_all"
        assert m.duration_seconds == 45.2
        assert m.rows_imported == 15000

    def test_to_dict(self) -> None:
        m = ETLMetric(pipeline="test", duration_seconds=10.5, rows_imported=100)
        d = m.to_dict()
        assert d["pipeline"] == "test"
        assert d["duration_seconds"] == 10.5
        assert d["rows_imported"] == 100
        assert "timestamp" in d
        assert d["success"] is True

    def test_to_dict_rounding(self) -> None:
        m = ETLMetric(duration_seconds=10.55555, duplicate_pct=0.33333)
        d = m.to_dict()
        assert d["duration_seconds"] == 10.56
        assert d["duplicate_pct"] == 0.33


class TestSystemMetric:
    """Test SystemMetric creation and serialization."""

    def test_defaults(self) -> None:
        m = SystemMetric()
        assert m.cpu_percent == 0.0
        assert m.memory_percent == 0.0
        assert m.db_size_mb == 0.0

    def test_custom_values(self) -> None:
        m = SystemMetric(
            cpu_percent=45.5,
            memory_percent=62.3,
            memory_used_mb=4096.0,
            disk_usage_pct=55.0,
            db_size_mb=128.5,
            cache_hit_rate=0.85,
            cache_entries=1000,
            cache_size_mb=5.0,
        )
        assert m.cpu_percent == 45.5
        assert m.memory_percent == 62.3
        assert m.db_size_mb == 128.5

    def test_to_dict(self) -> None:
        m = SystemMetric(cpu_percent=45.5, memory_percent=62.3)
        d = m.to_dict()
        assert d["cpu_percent"] == 45.5
        assert d["memory_percent"] == 62.3
        assert "timestamp" in d


class TestDataQualityMetric:
    """Test DataQualityMetric creation and serialization."""

    def test_defaults(self) -> None:
        m = DataQualityMetric()
        assert m.source == ""
        assert m.n_rows == 0
        assert m.validation_passed is True

    def test_custom_values(self) -> None:
        m = DataQualityMetric(
            source="test_source",
            n_rows=1000,
            n_columns=20,
            duplicate_pct=0.5,
            null_pct=2.3,
            columns_with_nulls=3,
            validation_passed=True,
            validation_errors=0,
        )
        assert m.source == "test_source"
        assert m.n_rows == 1000
        assert m.duplicate_pct == 0.5

    def test_to_dict(self) -> None:
        m = DataQualityMetric(source="test")
        d = m.to_dict()
        assert d["source"] == "test"
        assert d["validation_passed"] is True


class TestCacheMetric:
    """Test CacheMetric creation and serialization."""

    def test_defaults(self) -> None:
        m = CacheMetric()
        assert m.hits == 0
        assert m.misses == 0
        assert m.hit_rate == 0.0

    def test_custom_values(self) -> None:
        m = CacheMetric(hits=100, misses=20, hit_rate=0.8333, entries=500, size_bytes=1024 * 1024)
        assert m.hits == 100
        assert m.hit_rate == 0.8333

    def test_to_dict(self) -> None:
        m = CacheMetric(hits=100, misses=20, hit_rate=0.8333, entries=500, size_bytes=1048576)
        d = m.to_dict()
        assert d["hits"] == 100
        assert d["misses"] == 20
        assert d["size_mb"] == 1.0


class TestMetricSnapshot:
    """Test MetricSnapshot — composite of all metric types."""

    def test_empty_snapshot(self) -> None:
        snap = MetricSnapshot()
        assert snap.etl is None
        assert snap.system is None
        assert snap.data_quality is None
        assert snap.cache is None

    def test_full_snapshot(self) -> None:
        snap = MetricSnapshot(
            etl=ETLMetric(pipeline="test"),
            system=SystemMetric(cpu_percent=50.0),
            data_quality=DataQualityMetric(source="test"),
            cache=CacheMetric(hits=10),
        )
        assert snap.etl is not None
        assert snap.etl.pipeline == "test"
        assert snap.system is not None
        assert snap.system.cpu_percent == 50.0

    def test_to_dict_empty(self) -> None:
        d = MetricSnapshot().to_dict()
        assert "timestamp" in d
        assert "etl" not in d
        assert "system" not in d
        assert "data_quality" not in d
        assert "cache" not in d

    def test_to_dict_with_data(self) -> None:
        snap = MetricSnapshot(etl=ETLMetric(pipeline="test"))
        d = snap.to_dict()
        assert d["etl"]["pipeline"] == "test"


class TestTrendLine:
    """Test TrendLine model."""

    def test_defaults(self) -> None:
        t = TrendLine()
        assert t.metric_name == ""
        assert t.values == []
        assert t.direction == "stable"

    def test_custom_values(self) -> None:
        t = TrendLine(
            metric_name="rows_imported",
            values=[("2026-01-01", 100.0), ("2026-01-02", 200.0)],
            direction="up",
            change_pct=100.0,
        )
        assert t.metric_name == "rows_imported"
        assert len(t.values) == 2
        assert t.direction == "up"

    def test_to_dict(self) -> None:
        t = TrendLine(
            metric_name="cpu_percent",
            values=[("2026-01-01", 50.0), ("2026-01-02", 60.0)],
            direction="up",
            change_pct=20.0,
        )
        d = t.to_dict()
        assert d["metric_name"] == "cpu_percent"
        assert d["direction"] == "up"
        assert d["data_points"] == 2
        assert d["latest"] == 60.0
        assert d["earliest"] == 50.0

    def test_to_dict_empty_values(self) -> None:
        t = TrendLine()
        d = t.to_dict()
        assert d["latest"] is None
        assert d["earliest"] is None
        assert d["data_points"] == 0
