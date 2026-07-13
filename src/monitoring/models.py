"""
Monitoring data models — typed dataclasses for all tracked metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ETLMetric:
    """Metrics recorded during a single ETL pipeline run.

    Attributes
    ----------
    pipeline : str
        Pipeline name (e.g. ``collect_all``, ``import_current``).
    duration_seconds : float
        Total pipeline execution time.
    rows_imported : int
        Number of rows imported.
    rows_skipped : int
        Number of rows skipped (duplicates).
    download_speed_mbps : float
        Download throughput.
    processing_speed_rows_s : float
        Processing throughput.
    retry_count : int
        Number of retry attempts during this run.
    duplicate_pct : float
        Percentage of duplicate rows detected.
    missing_values_pct : float
        Percentage of missing values.
    validation_failures : int
        Number of validation failures.
    source : str
        Data source identifier.
    league : str
        League code.
    season : str
        Season identifier.
    success : bool
        Whether the pipeline completed successfully.
    error_message : str
        Error message if failed.
    timestamp : datetime
        When the metric was recorded.
    """

    pipeline: str = ""
    duration_seconds: float = 0.0
    rows_imported: int = 0
    rows_skipped: int = 0
    download_speed_mbps: float = 0.0
    processing_speed_rows_s: float = 0.0
    retry_count: int = 0
    duplicate_pct: float = 0.0
    missing_values_pct: float = 0.0
    validation_failures: int = 0
    source: str = ""
    league: str = ""
    season: str = ""
    success: bool = True
    error_message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "duration_seconds": round(self.duration_seconds, 2),
            "rows_imported": self.rows_imported,
            "rows_skipped": self.rows_skipped,
            "download_speed_mbps": round(self.download_speed_mbps, 2),
            "processing_speed_rows_s": round(self.processing_speed_rows_s, 2),
            "retry_count": self.retry_count,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "missing_values_pct": round(self.missing_values_pct, 2),
            "validation_failures": self.validation_failures,
            "source": self.source,
            "league": self.league,
            "season": self.season,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SystemMetric:
    """System resource snapshot at a point in time.

    Attributes
    ----------
    cpu_percent : float
        Overall CPU usage percentage.
    memory_percent : float
        Memory usage percentage.
    memory_used_mb : float
        Memory used in MB.
    disk_usage_pct : float
        Disk usage percentage of the data partition.
    db_size_mb : float
        Database file size in MB.
    cache_hit_rate : float
        Application cache hit rate (0.0–1.0).
    cache_entries : int
        Number of entries in the cache.
    cache_size_mb : float
        Approximate cache data size in MB.
    timestamp : datetime
        When the metric was recorded.
    """

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_pct: float = 0.0
    db_size_mb: float = 0.0
    cache_hit_rate: float = 0.0
    cache_entries: int = 0
    cache_size_mb: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "disk_usage_pct": round(self.disk_usage_pct, 1),
            "db_size_mb": round(self.db_size_mb, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cache_entries": self.cache_entries,
            "cache_size_mb": round(self.cache_size_mb, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DataQualityMetric:
    """Data quality assessment for a dataset.

    Attributes
    ----------
    source : str
        Dataset source identifier.
    n_rows : int
        Number of rows in the dataset.
    n_columns : int
        Number of columns.
    duplicate_pct : float
        Percentage of duplicate rows.
    null_pct : float
        Percentage of null values overall.
    columns_with_nulls : int
        Number of columns containing nulls.
    validation_passed : bool
        Whether validation checks passed.
    validation_errors : int
        Number of validation errors found.
    timestamp : datetime
        When the metric was recorded.
    """

    source: str = ""
    n_rows: int = 0
    n_columns: int = 0
    duplicate_pct: float = 0.0
    null_pct: float = 0.0
    columns_with_nulls: int = 0
    validation_passed: bool = True
    validation_errors: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_pct": round(self.duplicate_pct, 2),
            "null_pct": round(self.null_pct, 2),
            "columns_with_nulls": self.columns_with_nulls,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CacheMetric:
    """Cache performance snapshot.

    Attributes
    ----------
    hits : int
        Total cache hits.
    misses : int
        Total cache misses.
    hit_rate : float
        Cache hit rate (0.0–1.0).
    entries : int
        Number of cached entries.
    size_bytes : int
        Approximate cache data size in bytes.
    timestamp : datetime
        When the metric was recorded.
    """

    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    entries: int = 0
    size_bytes: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "entries": self.entries,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetricSnapshot:
    """Complete snapshot of all metrics at a point in time."""

    etl: ETLMetric | None = None
    system: SystemMetric | None = None
    data_quality: DataQualityMetric | None = None
    cache: CacheMetric | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"timestamp": self.timestamp.isoformat()}
        if self.etl:
            d["etl"] = self.etl.to_dict()
        if self.system:
            d["system"] = self.system.to_dict()
        if self.data_quality:
            d["data_quality"] = self.data_quality.to_dict()
        if self.cache:
            d["cache"] = self.cache.to_dict()
        return d


@dataclass
class TrendLine:
    """A trend line computed from historical metric data.

    Attributes
    ----------
    metric_name : str
        Name of the metric (e.g. ``rows_imported``, ``duration_seconds``).
    values : list[tuple[str, float]]
        List of (timestamp, value) pairs.
    direction : str
        Trend direction: ``up``, ``down``, ``stable``.
    change_pct : float
        Percentage change over the period.
    """

    metric_name: str = ""
    values: list[tuple[str, float]] = field(default_factory=list)
    direction: str = "stable"
    change_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "direction": self.direction,
            "change_pct": round(self.change_pct, 2),
            "data_points": len(self.values),
            "latest": self.values[-1][1] if self.values else None,
            "earliest": self.values[0][1] if self.values else None,
        }
