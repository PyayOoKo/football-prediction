"""
Monitor — primary orchestrator for the monitoring framework.

Ties together collectors, storage, and report generators into
a clean, single-entry-point API.

Usage
-----
::

    from src.monitoring import Monitor

    monitor = Monitor()

    # Record an ETL run
    monitor.record_etl(
        pipeline="collect_all",
        duration=45.2,
        rows_imported=15000,
        ...
    )

    # Take a system snapshot
    monitor.record_system()

    # Generate all reports
    monitor.generate_reports()
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.monitoring.collectors import (
    CacheMetricCollector,
    DataQualityCollector,
    ETLMetricCollector,
    SystemCollector,
)
from src.monitoring.models import (
    CacheMetric,
    DataQualityMetric,
    ETLMetric,
    SystemMetric,
)
from src.monitoring.reports import (
    CSVReport,
    DailySummaryReport,
    HTMLReport,
    JSONReport,
)
from src.monitoring.store import MonitoringStore

logger = logging.getLogger(__name__)


class Monitor:
    """Central monitor for collecting, storing, and reporting metrics.

    Parameters
    ----------
    db_path : str | Path
        Path to the monitoring SQLite database
        (default ``data/monitoring/monitor.db``).
    output_dir : str | Path
        Output directory for generated reports
        (default ``reports/monitoring``).
    data_dir : str | Path
        Data directory for disk-usage checks
        (default ``data``).
    retention_days : int
        Days to retain metrics (default 90).
    auto_collect_system_interval : float, optional
        If set, automatically collect system metrics every N seconds
        in a background daemon thread.
    cache_manager : Any, optional
        Application cache manager for cache metrics.
    fbref_client : Any, optional
        FBref HTTP client for cache metrics.
    """

    def __init__(
        self,
        db_path: str | Path = "data/monitoring/monitor.db",
        output_dir: str | Path = "reports/monitoring",
        data_dir: str | Path = "data",
        retention_days: int = 90,
        auto_collect_system_interval: float | None = None,
        cache_manager: Any = None,
        fbref_client: Any = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Secondary DB path for system collector to measure DB size
        self.db_path = Path(db_path)

        # Sub-components
        self.store = MonitoringStore(db_path=db_path, retention_days=retention_days)
        self.system_collector = SystemCollector(
            data_dir=data_dir,
            db_path=db_path,
        )
        self.etl_collector = ETLMetricCollector()
        self.dq_collector = DataQualityCollector()
        self.cache_collector = CacheMetricCollector()

        self.cache_manager = cache_manager
        self.fbref_client = fbref_client

        # Report generators (lazily created for freshness)
        self._html_report: HTMLReport | None = None
        self._json_report: JSONReport | None = None
        self._csv_report: CSVReport | None = None
        self._daily_report: DailySummaryReport | None = None

        # Background collection
        self._bg_thread: threading.Thread | None = None
        self._bg_stop = threading.Event()
        if auto_collect_system_interval is not None and auto_collect_system_interval > 0:
            self._start_background_collection(auto_collect_system_interval)

    # ── Property-based report generators ──────────────

    @property
    def html_report(self) -> HTMLReport:
        if self._html_report is None:
            self._html_report = HTMLReport(self.store, self.output_dir)
        return self._html_report

    @property
    def json_report(self) -> JSONReport:
        if self._json_report is None:
            self._json_report = JSONReport(self.store, self.output_dir)
        return self._json_report

    @property
    def csv_report(self) -> CSVReport:
        if self._csv_report is None:
            self._csv_report = CSVReport(self.store, self.output_dir)
        return self._csv_report

    @property
    def daily_report(self) -> DailySummaryReport:
        if self._daily_report is None:
            self._daily_report = DailySummaryReport(self.store, self.output_dir)
        return self._daily_report

    # ── Recording methods ──────────────────────────────

    def record_etl(self, **kwargs: Any) -> int:
        """Record an ETL pipeline run.

        Accepts all ``ETLMetric`` fields as keyword arguments.
        See ``ETLMetric`` for the full list of parameters.

        Returns
        -------
        int
            Row ID of the stored metric.
        """
        metric = ETLMetric(**kwargs)
        row_id = self.store.record_etl(metric)
        logger.info(
            "Recorded ETL metric: pipeline=%s duration=%.1fs rows=%d id=%d",
            metric.pipeline, metric.duration_seconds, metric.rows_imported, row_id,
        )
        return row_id

    def record_etl_from_result(self, result: dict[str, Any]) -> int:
        """Record an ETL metric from a pipeline result dict.

        Parameters
        ----------
        result : dict
            Pipeline stage result dictionary.

        Returns
        -------
        int
            Row ID of the stored metric.
        """
        metric = self.etl_collector.from_pipeline_result(result)
        return self.store.record_etl(metric)

    def record_system(self) -> int:
        """Record a system resource snapshot.

        Returns
        -------
        int
            Row ID of the stored metric.
        """
        metric = self.system_collector.collect()

        # Add cache metrics if available
        if self.cache_manager is not None:
            cache_metric = self.cache_collector.from_cache_manager(self.cache_manager)
            metric.cache_hit_rate = cache_metric.hit_rate
            metric.cache_entries = cache_metric.entries
            metric.cache_size_mb = cache_metric.size_bytes / (1024 * 1024) if cache_metric.size_bytes > 0 else 0.0
        elif self.fbref_client is not None:
            cache_metric = self.cache_collector.from_fbref_client(self.fbref_client)
            metric.cache_hit_rate = cache_metric.hit_rate

        row_id = self.store.record_system(metric)
        logger.debug(
            "Recorded system metric: cpu=%.1f%% mem=%.1f%% db=%.2fMB id=%d",
            metric.cpu_percent, metric.memory_percent, metric.db_size_mb, row_id,
        )
        return row_id

    def record_data_quality(self, df: Any, source: str = "unknown") -> int:
        """Record data quality metrics from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze.
        source : str
            Source identifier.

        Returns
        -------
        int
            Row ID of the stored metric.
        """
        metric = self.dq_collector.from_dataframe(df, source=source)
        row_id = self.store.record_data_quality(metric)
        logger.info(
            "Recorded data quality: source=%s rows=%d null=%.1f%% dup=%.1f%% id=%d",
            source, metric.n_rows, metric.null_pct, metric.duplicate_pct, row_id,
        )
        return row_id

    def record_cache(self, hits: int = 0, misses: int = 0, hit_rate: float = 0.0,
                     entries: int = 0, size_bytes: int = 0) -> int:
        """Record a cache metric snapshot.

        Parameters
        ----------
        hits : int
            Cache hits.
        misses : int
            Cache misses.
        hit_rate : float
            Hit rate (0.0–1.0).
        entries : int
            Number of cached entries.
        size_bytes : int
            Approximate cache data size in bytes.

        Returns
        -------
        int
            Row ID of the stored metric.
        """
        metric = CacheMetric(
            hits=hits, misses=misses, hit_rate=hit_rate,
            entries=entries, size_bytes=size_bytes,
        )
        return self.store.record_cache(metric)

    # ── Report generation ──────────────────────────────

    def generate_reports(self, days: int = 30) -> dict[str, Any]:
        """Generate all report formats.

        Creates HTML dashboard, JSON export, CSV exports,
        and saves the daily summary to a text file.

        Parameters
        ----------
        days : int
            Lookback period for reports.

        Returns
        -------
        dict[str, Any]
            Paths to generated files:
            ``html``, ``json``, ``csv``, ``summary``.
        """
        results: dict[str, Any] = {}

        try:
            path = self.html_report.generate(days=days)
            results["html"] = str(path)
        except Exception as exc:
            logger.error("HTML report failed: %s", exc)
            results["html_error"] = str(exc)

        try:
            path = self.json_report.generate(days=days)
            results["json"] = str(path)
        except Exception as exc:
            logger.error("JSON report failed: %s", exc)
            results["json_error"] = str(exc)

        try:
            paths = self.csv_report.generate(days=days)
            results["csv"] = [str(p) for p in paths]
        except Exception as exc:
            logger.error("CSV report failed: %s", exc)
            results["csv_error"] = str(exc)

        try:
            text = self.daily_report.generate()
            summary_path = self.output_dir / "daily_summary.txt"
            summary_path.write_text(text, encoding="utf-8")
            results["summary"] = str(summary_path)
            results["summary_text"] = text
        except Exception as exc:
            logger.error("Daily summary failed: %s", exc)
            results["summary_error"] = str(exc)

        return results

    def daily_summary(self) -> str:
        """Get the daily summary as a formatted string.

        Returns
        -------
        str
            Plain text daily summary.
        """
        return self.daily_report.generate()

    def get_latest_snapshot(self) -> dict[str, Any]:
        """Get the latest metric snapshot as a dictionary.

        Returns
        -------
        dict[str, Any]
            Latest metrics per type.
        """
        return self.store.get_latest().to_dict()

    # ── Lifecycle ──────────────────────────────────────

    def cleanup(self, retention_days: int | None = None) -> dict[str, int]:
        """Purge metrics older than retention_days.

        Parameters
        ----------
        retention_days : int, optional
            Days to retain.

        Returns
        -------
        dict[str, int]
            Rows deleted per table.
        """
        return self.store.cleanup(retention_days=retention_days)

    def close(self) -> None:
        """Stop background collection and release resources."""
        if self._bg_thread is not None:
            self._bg_stop.set()
            self._bg_thread.join(timeout=5)
            self._bg_thread = None
        logger.info("Monitor closed")

    def __enter__(self) -> Monitor:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Background collection ──────────────────────────

    def _start_background_collection(self, interval: float) -> None:
        """Start a daemon thread that periodically collects system metrics.

        Parameters
        ----------
        interval : float
            Collection interval in seconds.
        """
        def _loop() -> None:
            logger.info("Background system collection started (interval=%.1fs)", interval)
            while not self._bg_stop.is_set():
                try:
                    self.record_system()
                except Exception as exc:
                    logger.warning("Background system collection failed: %s", exc)
                self._bg_stop.wait(timeout=interval)

        self._bg_thread = threading.Thread(
            target=_loop,
            name="monitor-bg",
            daemon=True,
        )
        self._bg_thread.start()
