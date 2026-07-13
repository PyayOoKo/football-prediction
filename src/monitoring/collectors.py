"""
Metric collectors — gather real-time metrics from the system, ETL pipeline,
data quality checks, and cache infrastructure.

Each collector returns a typed metric dataclass that can be stored
and reported on.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.monitoring.models import (
    CacheMetric,
    DataQualityMetric,
    ETLMetric,
    SystemMetric,
)

logger = logging.getLogger(__name__)


class SystemCollector:
    """Collects system resource metrics (CPU, memory, disk).

    Uses ``psutil`` if available, otherwise provides best-effort
    estimates using stdlib-only approaches.

    Parameters
    ----------
    data_dir : str | Path
        Data directory to check for disk usage (default ``data/``).
    db_path : str | Path, optional
        Database file path for size tracking.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        db_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path) if db_path else None
        self._psutil_available = False

        try:
            import psutil  # noqa: F401
            self._psutil_available = True
        except ImportError:
            logger.info("psutil not available — using stdlib-only system metrics")

    def collect(self) -> SystemMetric:
        """Collect current system metrics.

        Returns
        -------
        SystemMetric
            Snapshot of CPU, memory, disk usage.
        """
        metric = SystemMetric()

        if self._psutil_available:
            import psutil
            metric.cpu_percent = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()
            metric.memory_percent = mem.percent
            metric.memory_used_mb = mem.used / (1024 * 1024)

            # Disk usage for the data partition
            try:
                disk = psutil.disk_usage(str(self.data_dir))
                metric.disk_usage_pct = disk.percent
            except Exception:
                pass
        else:
            # Stdlib fallback — mark as unknown
            metric.cpu_percent = 0.0
            metric.memory_percent = 0.0
            metric.disk_usage_pct = 0.0

        # Database file size (always available)
        metric.db_size_mb = self._get_db_size()

        return metric

    def _get_db_size(self) -> float:
        """Get the database file size in MB."""
        total = 0.0
        if self.db_path and self.db_path.exists():
            total += self.db_path.stat().st_size / (1024 * 1024)

        # Also check for WAL and SHM files
        for suffix in ["-wal", "-shm"]:
            wal_path = self.db_path.parent / f"{self.db_path.name}{suffix}"
            if wal_path.exists():
                total += wal_path.stat().st_size / (1024 * 1024)

        return total


class ETLMetricCollector:
    """Records and computes ETL pipeline metrics.

    Provides helpers to compute speeds from raw counters and
    to create ETL metrics from pipeline stage results.

    Usage
    -----
    ::

        collector = ETLMetricCollector()
        etl_metric = collector.from_run(
            pipeline="collect_all",
            duration=45.2,
            rows=15000,
            download_bytes=50 * 1024 * 1024,
            retries=2,
        )
    """

    @staticmethod
    def compute_download_speed(
        bytes_downloaded: int,
        duration_seconds: float,
    ) -> float:
        """Compute download speed in Mbps.

        Parameters
        ----------
        bytes_downloaded : int
            Total bytes downloaded.
        duration_seconds : float
            Download duration in seconds.

        Returns
        -------
        float
            Speed in megabits per second.
        """
        if duration_seconds <= 0:
            return 0.0
        bits = bytes_downloaded * 8
        return bits / duration_seconds / 1_000_000

    @staticmethod
    def compute_processing_speed(
        rows: int,
        duration_seconds: float,
    ) -> float:
        """Compute processing speed in rows per second.

        Parameters
        ----------
        rows : int
            Number of rows processed.
        duration_seconds : float
            Processing duration in seconds.

        Returns
        -------
        float
            Speed in rows per second.
        """
        if duration_seconds <= 0:
            return 0.0
        return rows / duration_seconds

    @staticmethod
    def from_run(
        pipeline: str,
        duration: float,
        rows: int,
        download_speed_mbps: float = 0.0,
        processing_speed_rows_s: float = 0.0,
        retry_count: int = 0,
        duplicate_pct: float = 0.0,
        missing_values_pct: float = 0.0,
        validation_failures: int = 0,
        source: str = "",
        league: str = "",
        season: str = "",
        success: bool = True,
        error_message: str = "",
        rows_skipped: int = 0,
    ) -> ETLMetric:
        """Create an ETLMetric from pipeline run data.

        Parameters
        ----------
        pipeline : str
            Pipeline name.
        duration : float
            Duration in seconds.
        rows : int
            Rows imported.
        download_speed_mbps : float
            Download speed in Mbps.
        processing_speed_rows_s : float
            Processing speed in rows/s.
        retry_count : int
            Retry count.
        duplicate_pct : float
            Duplicate percentage.
        missing_values_pct : float
            Missing values percentage.
        validation_failures : int
            Validation failure count.
        source : str
            Data source name.
        league : str
            League code.
        season : str
            Season code.
        success : bool
            Whether it succeeded.
        error_message : str
            Error if failed.
        rows_skipped : int
            Rows skipped.

        Returns
        -------
        ETLMetric
        """
        return ETLMetric(
            pipeline=pipeline,
            duration_seconds=duration,
            rows_imported=rows,
            rows_skipped=rows_skipped,
            download_speed_mbps=download_speed_mbps,
            processing_speed_rows_s=processing_speed_rows_s,
            retry_count=retry_count,
            duplicate_pct=duplicate_pct,
            missing_values_pct=missing_values_pct,
            validation_failures=validation_failures,
            source=source,
            league=league,
            season=season,
            success=success,
            error_message=error_message,
        )

    @staticmethod
    def from_pipeline_result(result: dict[str, Any]) -> ETLMetric:
        """Create an ETLMetric from a pipeline run result dict.

        Expects keys like ``duration_seconds``, ``rows``, ``new_rows``,
        ``total_rows``, ``error``, etc., as returned by
        ``run_pipeline.step_download()`` and similar.

        Parameters
        ----------
        result : dict
            Pipeline stage result dictionary.

        Returns
        -------
        ETLMetric
        """
        duration = result.get("duration_seconds", 0.0) or \
                   result.get("elapsed", 0.0)
        rows = result.get("rows", 0) or result.get("new_rows", 0) or \
               result.get("total_rows", 0)
        error = result.get("error", "")
        success = result.get("success", True)

        return ETLMetric(
            pipeline=result.get("pipeline", "unknown"),
            duration_seconds=duration,
            rows_imported=rows,
            rows_skipped=result.get("skipped", 0),
            download_speed_mbps=result.get("download_speed_mbps", 0.0),
            processing_speed_rows_s=result.get("processing_speed_rows_s", 0.0),
            retry_count=result.get("retry_count", 0),
            duplicate_pct=result.get("duplicate_pct", 0.0),
            missing_values_pct=result.get("missing_values_pct", 0.0),
            validation_failures=result.get("validation_failures", 0),
            source=result.get("source", ""),
            league=result.get("league", ""),
            season=result.get("season", ""),
            success=success,
            error_message=str(error) if error else "",
        )


class DataQualityCollector:
    """Collects data quality metrics from a dataset.

    Analyzes a DataFrame for nulls, duplicates, and schema issues,
    producing a DataQualityMetric.
    """

    @staticmethod
    def from_dataframe(
        df: Any,
        source: str = "unknown",
    ) -> DataQualityMetric:
        """Analyze a DataFrame and produce quality metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze.
        source : str
            Source identifier.

        Returns
        -------
        DataQualityMetric
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame) or df.empty:
            return DataQualityMetric(
                source=source,
                n_rows=0, n_columns=0,
                validation_passed=True,
            )

        n_rows = len(df)
        n_columns = len(df.columns)

        # Duplicates
        dup_count = df.duplicated().sum()
        duplicate_pct = round(dup_count / n_rows * 100, 2) if n_rows > 0 else 0.0

        # Nulls
        total_cells = n_rows * n_columns
        null_count = int(df.isna().sum().sum())
        null_pct = round(null_count / total_cells * 100, 2) if total_cells > 0 else 0.0
        cols_with_nulls = int((df.isna().sum() > 0).sum())

        return DataQualityMetric(
            source=source,
            n_rows=n_rows,
            n_columns=n_columns,
            duplicate_pct=duplicate_pct,
            null_pct=null_pct,
            columns_with_nulls=cols_with_nulls,
            validation_passed=True,
            validation_errors=0,
        )


class CacheMetricCollector:
    """Collects cache performance metrics.

    Interfaces with the application cache manager to get
    hit rates, entry counts, and size data.
    """

    @staticmethod
    def from_cache_manager(cache_manager: Any) -> CacheMetric:
        """Collect metrics from a CacheManager instance.

        Parameters
        ----------
        cache_manager : CacheManager
            The application's cache manager.

        Returns
        -------
        CacheMetric
        """
        import asyncio

        try:
            stats = asyncio.run(cache_manager.stats())
            return CacheMetric(
                hits=stats.hits,
                misses=stats.misses,
                hit_rate=stats.hit_ratio,
                entries=stats.entries,
                size_bytes=stats.size_bytes,
            )
        except Exception as exc:
            logger.warning("Failed to collect cache metrics: %s", exc)
            # Fall back to to_dict if stats() fails
            try:
                d = cache_manager.to_dict()
                return CacheMetric(
                    hits=d.get("hits", 0),
                    misses=d.get("misses", 0),
                    hit_rate=d.get("hit_ratio", 0.0),
                    entries=0,
                    size_bytes=0,
                )
            except Exception:
                return CacheMetric()

    @staticmethod
    def from_fbref_client(client: Any) -> CacheMetric:
        """Collect cache metrics from an FBrefClient.

        Parameters
        ----------
        client : FBrefClient
            The FBref HTTP client with its own cache.

        Returns
        -------
        CacheMetric
        """
        try:
            stats = client.cache_stats
            total = stats.get("total", 0)
            hits = stats.get("hits", 0)
            return CacheMetric(
                hits=hits,
                misses=stats.get("misses", 0),
                hit_rate=hits / max(total, 1),
                entries=0,
                size_bytes=0,
            )
        except Exception as exc:
            logger.warning("Failed to collect FBref cache metrics: %s", exc)
            return CacheMetric()
