"""Tests for metric collectors."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.monitoring.collectors import (
    CacheMetricCollector,
    DataQualityCollector,
    ETLMetricCollector,
    SystemCollector,
)
from src.monitoring.models import CacheMetric, DataQualityMetric, ETLMetric


class TestETLMetricCollector:
    """Test ETL metric computation and creation."""

    def test_compute_download_speed(self) -> None:
        # 10 MB in 2 seconds = 40 Mbps
        speed = ETLMetricCollector.compute_download_speed(
            bytes_downloaded=10 * 1024 * 1024,
            duration_seconds=2.0,
        )
        assert speed == pytest.approx(40.0, rel=0.1)

    def test_compute_download_speed_zero_duration(self) -> None:
        speed = ETLMetricCollector.compute_download_speed(
            bytes_downloaded=1000, duration_seconds=0,
        )
        assert speed == 0.0

    def test_compute_processing_speed(self) -> None:
        speed = ETLMetricCollector.compute_processing_speed(
            rows=1000, duration_seconds=5.0,
        )
        assert speed == 200.0

    def test_compute_processing_speed_zero_rows(self) -> None:
        speed = ETLMetricCollector.compute_processing_speed(rows=0, duration_seconds=5.0)
        assert speed == 0.0

    def test_from_run(self) -> None:
        m = ETLMetricCollector.from_run(
            pipeline="collect_all",
            duration=45.2,
            rows=15000,
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
        assert isinstance(m, ETLMetric)
        assert m.pipeline == "collect_all"
        assert m.duration_seconds == 45.2
        assert m.rows_imported == 15000

    def test_from_run_failure(self) -> None:
        m = ETLMetricCollector.from_run(
            pipeline="collect_all",
            duration=10.0,
            rows=0,
            success=False,
            error_message="Connection timeout",
        )
        assert m.success is False
        assert m.error_message == "Connection timeout"

    def test_from_pipeline_result(self) -> None:
        m = ETLMetricCollector.from_pipeline_result({
            "pipeline": "download",
            "duration_seconds": 30.0,
            "rows": 5000,
            "retry_count": 1,
            "success": True,
            "source": "test",
        })
        assert m.pipeline == "download"
        assert m.duration_seconds == 30.0
        assert m.rows_imported == 5000
        assert m.success is True

    def test_from_pipeline_result_fallback_keys(self) -> None:
        m = ETLMetricCollector.from_pipeline_result({
            "elapsed": 15.0,
            "new_rows": 200,
            "error": "Partial failure",
            "success": False,
        })
        assert m.duration_seconds == 15.0
        assert m.rows_imported == 200
        assert m.success is False
        assert m.error_message == "Partial failure"


class TestSystemCollector:
    """Test system resource collection."""

    def test_collect_without_psutil(self) -> None:
        """Should work even without psutil."""
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            collector = SystemCollector(data_dir=tmp, db_path=db_path)
            metric = collector.collect()
            assert metric.cpu_percent >= 0
            assert metric.memory_percent >= 0
            assert metric.db_size_mb == 0.0  # Empty/non-existent db

    def test_collect_with_db_file(self) -> None:
        """Should report DB file size."""
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            db_path.write_text("x" * 1024 * 100)  # 100 KB
            collector = SystemCollector(data_dir=tmp, db_path=db_path)
            metric = collector.collect()
            assert metric.db_size_mb == pytest.approx(0.1, rel=0.1)


class TestDataQualityCollector:
    """Test data quality analysis."""

    def test_empty_dataframe(self) -> None:
        import pandas as pd
        m = DataQualityCollector.from_dataframe(pd.DataFrame(), source="test")
        assert m.n_rows == 0
        assert m.validation_passed is True

    def test_healthy_dataframe(self) -> None:
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        m = DataQualityCollector.from_dataframe(df, source="test")
        assert m.n_rows == 3
        assert m.n_columns == 2
        assert m.duplicate_pct == 0.0
        assert m.null_pct == 0.0
        assert m.columns_with_nulls == 0

    def test_with_nulls(self) -> None:
        import pandas as pd
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
        m = DataQualityCollector.from_dataframe(df, source="test")
        assert m.n_rows == 3
        assert m.columns_with_nulls == 2
        # 3 nulls out of 6 cells = 50%
        assert m.null_pct == 50.0

    def test_with_duplicates(self) -> None:
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 2, 3, 1], "b": [4, 5, 5, 6, 4]})
        m = DataQualityCollector.from_dataframe(df, source="test")
        assert m.n_rows == 5
        # 2 duplicate rows (rows 2 and 4)
        assert m.duplicate_pct == 40.0

    def test_non_dataframe_input(self) -> None:
        """Should return empty metric for non-DataFrame input."""
        m = DataQualityCollector.from_dataframe("not a dataframe", source="test")
        assert m.n_rows == 0
        assert m.n_columns == 0


class TestCacheMetricCollector:
    """Test cache metric collection."""

    def test_from_cache_manager(self) -> None:
        """Should handle missing cache manager gracefully."""
        m = CacheMetricCollector.from_cache_manager(None)
        assert isinstance(m, CacheMetric)
        assert m.hits == 0

    def test_from_fbref_client(self) -> None:
        """Should handle missing client gracefully."""
        m = CacheMetricCollector.from_fbref_client(None)
        assert isinstance(m, CacheMetric)
        assert m.hits == 0

    def test_from_cache_manager_with_stats(self) -> None:
        """Test with an object that has a stats() method."""

        class FakeCache:
            async def stats(self):
                return type("Stats", (), {
                    "hits": 100,
                    "misses": 20,
                    "hit_ratio": 0.8333,
                    "entries": 500,
                    "size_bytes": 1024 * 1024,
                })()

        m = CacheMetricCollector.from_cache_manager(FakeCache())
        assert m.hits == 100
        assert m.misses == 20
        assert m.hit_rate == pytest.approx(0.8333)
        assert m.entries == 500
        assert m.size_bytes == 1048576

    def test_from_cache_manager_with_to_dict_fallback(self) -> None:
        """Fallback to to_dict if stats() fails."""

        class FakeCache:
            def to_dict(self):
                return {"hits": 50, "misses": 10, "hit_ratio": 0.8333}

        m = CacheMetricCollector.from_cache_manager(FakeCache())
        assert m.hits == 50
        assert m.misses == 10
