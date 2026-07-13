"""
Monitoring store — persists metrics to SQLite with historical trend support.

Stores all metric types in separate tables with timestamps for
time-series analysis and trend computation.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.monitoring.models import (
    CacheMetric,
    DataQualityMetric,
    ETLMetric,
    MetricSnapshot,
    SystemMetric,
    TrendLine,
)

logger = logging.getLogger(__name__)


class MonitoringStore:
    """SQLite-backed store for all monitoring metrics.

    Stores metrics in 4 tables:
    - ``etl_metrics`` — Pipeline execution metrics
    - ``system_metrics`` — System resource snapshots
    - ``data_quality_metrics`` — Data quality assessments
    - ``cache_metrics`` — Cache performance snapshots

    Each table has a ``recorded_at`` timestamp column for
    time-series analysis and trend computation.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file (default ``data/monitoring/monitor.db``).
    retention_days : int
        Number of days to retain metrics (default 90).
        Older metrics are purged on ``cleanup()``.
    """

    def __init__(
        self,
        db_path: str | Path = "data/monitoring/monitor.db",
        retention_days: int = 90,
    ) -> None:
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._local = threading.local()

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize tables
        self._init_db()

    # ── Connection management ──────────────────────────

    def close(self) -> None:
        """Close the thread-local SQLite connection and release file handles.

        Performs a WAL checkpoint to consolidate the WAL file, then
        switches journal mode to DELETE so the WAL file is removed on
        next connection close. This is important on Windows where
        file handles prevent temporary directory cleanup.

        After calling this method, the store will re-create the
        connection on the next operation.
        """
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            try:
                self._local.conn.execute("PRAGMA journal_mode=DELETE")
            except Exception:
                pass
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8MB
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Create all monitoring tables if they don't exist."""
        conn = self._get_conn()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS etl_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline TEXT NOT NULL,
                duration_seconds REAL NOT NULL DEFAULT 0,
                rows_imported INTEGER NOT NULL DEFAULT 0,
                rows_skipped INTEGER NOT NULL DEFAULT 0,
                download_speed_mbps REAL NOT NULL DEFAULT 0,
                processing_speed_rows_s REAL NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                duplicate_pct REAL NOT NULL DEFAULT 0,
                missing_values_pct REAL NOT NULL DEFAULT 0,
                validation_failures INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT '',
                league TEXT NOT NULL DEFAULT '',
                season TEXT NOT NULL DEFAULT '',
                success INTEGER NOT NULL DEFAULT 1,
                error_message TEXT NOT NULL DEFAULT '',
                recorded_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_percent REAL NOT NULL DEFAULT 0,
                memory_percent REAL NOT NULL DEFAULT 0,
                memory_used_mb REAL NOT NULL DEFAULT 0,
                disk_usage_pct REAL NOT NULL DEFAULT 0,
                db_size_mb REAL NOT NULL DEFAULT 0,
                cache_hit_rate REAL NOT NULL DEFAULT 0,
                cache_entries INTEGER NOT NULL DEFAULT 0,
                cache_size_mb REAL NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                n_rows INTEGER NOT NULL DEFAULT 0,
                n_columns INTEGER NOT NULL DEFAULT 0,
                duplicate_pct REAL NOT NULL DEFAULT 0,
                null_pct REAL NOT NULL DEFAULT 0,
                columns_with_nulls INTEGER NOT NULL DEFAULT 0,
                validation_passed INTEGER NOT NULL DEFAULT 1,
                validation_errors INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hits INTEGER NOT NULL DEFAULT 0,
                misses INTEGER NOT NULL DEFAULT 0,
                hit_rate REAL NOT NULL DEFAULT 0,
                entries INTEGER NOT NULL DEFAULT 0,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                recorded_at TEXT NOT NULL
            )
        """)

        # Indexes for time-range queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_etl_recorded_at
            ON etl_metrics(recorded_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_recorded_at
            ON system_metrics(recorded_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dq_recorded_at
            ON data_quality_metrics(recorded_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_recorded_at
            ON cache_metrics(recorded_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_etl_pipeline
            ON etl_metrics(pipeline)
        """)

    # ── Write methods ──────────────────────────────────

    def record_etl(self, metric: ETLMetric) -> int:
        """Store an ETL metric. Returns the row ID."""
        conn = self._get_conn()
        ts = metric.timestamp.isoformat()
        conn.execute(
            """
            INSERT INTO etl_metrics
                (pipeline, duration_seconds, rows_imported, rows_skipped,
                 download_speed_mbps, processing_speed_rows_s,
                 retry_count, duplicate_pct, missing_values_pct,
                 validation_failures, source, league, season,
                 success, error_message, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric.pipeline, metric.duration_seconds,
                metric.rows_imported, metric.rows_skipped,
                metric.download_speed_mbps, metric.processing_speed_rows_s,
                metric.retry_count, metric.duplicate_pct,
                metric.missing_values_pct, metric.validation_failures,
                metric.source, metric.league, metric.season,
                int(metric.success), metric.error_message, ts,
            ),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def record_system(self, metric: SystemMetric) -> int:
        """Store a system metric. Returns the row ID."""
        conn = self._get_conn()
        ts = metric.timestamp.isoformat()
        conn.execute(
            """
            INSERT INTO system_metrics
                (cpu_percent, memory_percent, memory_used_mb,
                 disk_usage_pct, db_size_mb, cache_hit_rate,
                 cache_entries, cache_size_mb, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric.cpu_percent, metric.memory_percent,
                metric.memory_used_mb, metric.disk_usage_pct,
                metric.db_size_mb, metric.cache_hit_rate,
                metric.cache_entries, metric.cache_size_mb, ts,
            ),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def record_data_quality(self, metric: DataQualityMetric) -> int:
        """Store a data quality metric. Returns the row ID."""
        conn = self._get_conn()
        ts = metric.timestamp.isoformat()
        conn.execute(
            """
            INSERT INTO data_quality_metrics
                (source, n_rows, n_columns, duplicate_pct, null_pct,
                 columns_with_nulls, validation_passed, validation_errors,
                 recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric.source, metric.n_rows, metric.n_columns,
                metric.duplicate_pct, metric.null_pct,
                metric.columns_with_nulls, int(metric.validation_passed),
                metric.validation_errors, ts,
            ),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def record_cache(self, metric: CacheMetric) -> int:
        """Store a cache metric. Returns the row ID."""
        conn = self._get_conn()
        ts = metric.timestamp.isoformat()
        conn.execute(
            """
            INSERT INTO cache_metrics
                (hits, misses, hit_rate, entries, size_bytes, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (metric.hits, metric.misses, metric.hit_rate,
             metric.entries, metric.size_bytes, ts),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # ── Read methods ───────────────────────────────────

    def get_etl_history(
        self,
        days: int = 30,
        pipeline: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get ETL metrics for the last N days.

        Parameters
        ----------
        days : int
            Number of days of history.
        pipeline : str, optional
            Filter by pipeline name.

        Returns
        -------
        list[dict[str, Any]]
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._get_conn()

        if pipeline:
            rows = conn.execute(
                """
                SELECT * FROM etl_metrics
                WHERE recorded_at >= ? AND pipeline = ?
                ORDER BY recorded_at DESC
                """,
                (cutoff, pipeline),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM etl_metrics
                WHERE recorded_at >= ?
                ORDER BY recorded_at DESC
                """,
                (cutoff,),
            ).fetchall()

        return [dict(r) for r in rows]

    def get_system_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get system metrics for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM system_metrics
            WHERE recorded_at >= ?
            ORDER BY recorded_at DESC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_data_quality_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get data quality metrics for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM data_quality_metrics
            WHERE recorded_at >= ?
            ORDER BY recorded_at DESC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_cache_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get cache metrics for the last N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM cache_metrics
            WHERE recorded_at >= ?
            ORDER BY recorded_at DESC
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest(self) -> MetricSnapshot:
        """Get the latest of each metric type as a snapshot."""
        conn = self._get_conn()

        # Latest ETL
        etl_row = conn.execute(
            "SELECT * FROM etl_metrics ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()
        etl = self._row_to_etl(etl_row) if etl_row else None

        # Latest system
        sys_row = conn.execute(
            "SELECT * FROM system_metrics ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()
        sys = self._row_to_system(sys_row) if sys_row else None

        # Latest data quality
        dq_row = conn.execute(
            "SELECT * FROM data_quality_metrics ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()
        dq = self._row_to_dq(dq_row) if dq_row else None

        # Latest cache
        cache_row = conn.execute(
            "SELECT * FROM cache_metrics ORDER BY recorded_at DESC LIMIT 1"
        ).fetchone()
        cache = self._row_to_cache(cache_row) if cache_row else None

        return MetricSnapshot(etl=etl, system=sys, data_quality=dq, cache=cache)

    # ── Trends ─────────────────────────────────────────

    def get_trends(self, days: int = 30) -> list[TrendLine]:
        """Compute trend lines for key metrics over the period.

        Parameters
        ----------
        days : int
            Lookback period in days.

        Returns
        -------
        list[TrendLine]
        """
        trends: list[TrendLine] = []

        # ETL trends
        etl_data = self.get_etl_history(days=days)
        if etl_data:
            trends.extend([
                self._compute_trend(etl_data, "duration_seconds"),
                self._compute_trend(etl_data, "rows_imported"),
                self._compute_trend(etl_data, "download_speed_mbps"),
                self._compute_trend(etl_data, "processing_speed_rows_s"),
                self._compute_trend(etl_data, "retry_count"),
                self._compute_trend(etl_data, "duplicate_pct"),
                self._compute_trend(etl_data, "missing_values_pct"),
            ])

        # System trends
        sys_data = self.get_system_history(days=days)
        if sys_data:
            trends.extend([
                self._compute_trend(sys_data, "cpu_percent"),
                self._compute_trend(sys_data, "memory_percent"),
                self._compute_trend(sys_data, "db_size_mb"),
            ])

        # Data quality trends
        dq_data = self.get_data_quality_history(days=days)
        if dq_data:
            trends.extend([
                self._compute_trend(dq_data, "n_rows"),
                self._compute_trend(dq_data, "null_pct"),
                self._compute_trend(dq_data, "duplicate_pct"),
            ])

        # Cache trends
        cache_data = self.get_cache_history(days=days)
        if cache_data:
            trends.extend([
                self._compute_trend(cache_data, "hit_rate"),
                self._compute_trend(cache_data, "entries"),
            ])

        return trends

    def _compute_trend(
        self,
        data: list[dict[str, Any]],
        metric_name: str,
    ) -> TrendLine:
        """Compute a trend line for a single metric from time-series data."""
        values: list[tuple[str, float]] = []
        for row in sorted(data, key=lambda x: x["recorded_at"]):
            val = row.get(metric_name, 0)
            if val is not None:
                values.append((row["recorded_at"], float(val)))

        # Determine direction and % change
        direction = "stable"
        change_pct = 0.0
        if len(values) >= 2:
            first = values[0][1]
            last = values[-1][1]
            if first > 0:
                change_pct = (last - first) / first * 100
            if abs(change_pct) < 5:
                direction = "stable"
            elif change_pct > 0:
                direction = "up"
            else:
                direction = "down"

        return TrendLine(
            metric_name=metric_name,
            values=values,
            direction=direction,
            change_pct=change_pct,
        )

    # ── Cleanup ────────────────────────────────────────

    def cleanup(self, retention_days: int | None = None) -> dict[str, int]:
        """Remove metrics older than retention_days.

        Parameters
        ----------
        retention_days : int, optional
            Override the instance's retention_days setting.

        Returns
        -------
        dict[str, int]
            Number of rows deleted per table.
        """
        days = retention_days if retention_days is not None else self.retention_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._get_conn()

        deleted: dict[str, int] = {}
        for table in [
            "etl_metrics", "system_metrics",
            "data_quality_metrics", "cache_metrics",
        ]:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE recorded_at <= ?",
                (cutoff,),
            )
            deleted[table] = cursor.rowcount

        logger.info("Cleaned up %d old monitoring entries", sum(deleted.values()))
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns
        -------
        dict[str, Any]
            Counts per table and oldest/newest timestamps.
        """
        conn = self._get_conn()
        stats: dict[str, Any] = {
            "retention_days": self.retention_days,
            "db_path": str(self.db_path),
            "tables": {},
        }

        for table in [
            "etl_metrics", "system_metrics",
            "data_quality_metrics", "cache_metrics",
        ]:
            row = conn.execute(
                f"""
                SELECT COUNT(*) as count,
                       MIN(recorded_at) as oldest,
                       MAX(recorded_at) as newest
                FROM {table}
                """
            ).fetchone()
            stats["tables"][table] = {
                "count": row["count"],
                "oldest": row["oldest"],
                "newest": row["newest"],
            }

        return stats

    # ── Row converters ─────────────────────────────────

    @staticmethod
    def _row_to_etl(row: sqlite3.Row) -> ETLMetric:
        return ETLMetric(
            pipeline=row["pipeline"],
            duration_seconds=row["duration_seconds"],
            rows_imported=row["rows_imported"],
            rows_skipped=row["rows_skipped"],
            download_speed_mbps=row["download_speed_mbps"],
            processing_speed_rows_s=row["processing_speed_rows_s"],
            retry_count=row["retry_count"],
            duplicate_pct=row["duplicate_pct"],
            missing_values_pct=row["missing_values_pct"],
            validation_failures=row["validation_failures"],
            source=row["source"],
            league=row["league"],
            season=row["season"],
            success=bool(row["success"]),
            error_message=row["error_message"],
            timestamp=datetime.fromisoformat(row["recorded_at"]),
        )

    @staticmethod
    def _row_to_system(row: sqlite3.Row) -> SystemMetric:
        return SystemMetric(
            cpu_percent=row["cpu_percent"],
            memory_percent=row["memory_percent"],
            memory_used_mb=row["memory_used_mb"],
            disk_usage_pct=row["disk_usage_pct"],
            db_size_mb=row["db_size_mb"],
            cache_hit_rate=row["cache_hit_rate"],
            cache_entries=row["cache_entries"],
            cache_size_mb=row["cache_size_mb"],
            timestamp=datetime.fromisoformat(row["recorded_at"]),
        )

    @staticmethod
    def _row_to_dq(row: sqlite3.Row) -> DataQualityMetric:
        return DataQualityMetric(
            source=row["source"],
            n_rows=row["n_rows"],
            n_columns=row["n_columns"],
            duplicate_pct=row["duplicate_pct"],
            null_pct=row["null_pct"],
            columns_with_nulls=row["columns_with_nulls"],
            validation_passed=bool(row["validation_passed"]),
            validation_errors=row["validation_errors"],
            timestamp=datetime.fromisoformat(row["recorded_at"]),
        )

    @staticmethod
    def _row_to_cache(row: sqlite3.Row) -> CacheMetric:
        return CacheMetric(
            hits=row["hits"],
            misses=row["misses"],
            hit_rate=row["hit_rate"],
            entries=row["entries"],
            size_bytes=row["size_bytes"],
            timestamp=datetime.fromisoformat(row["recorded_at"]),
        )
