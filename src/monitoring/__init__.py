"""
Monitoring framework for the football ETL platform.

Tracks key operational metrics:
- Download speed, processing speed, rows imported
- Duplicate percentage, missing values, validation failures
- Retry count, pipeline duration
- Memory usage, CPU usage, database size, cache hit rate

Generates:
- HTML dashboard (Plotly interactive)
- JSON metrics
- CSV metrics
- Daily summary reports
- Historical trend analysis

Usage
-----
::

    from src.monitoring import Monitor

    monitor = Monitor()

    # Record an ETL run
    monitor.record_etl({
        "pipeline": "collect_all",
        "duration": 45.2,
        "rows_imported": 15000,
        "download_speed_mbps": 12.5,
        "processing_speed_rows_s": 320,
        "retry_count": 2,
        "duplicate_pct": 0.3,
        "missing_values_pct": 1.2,
        "validation_failures": 0,
    })

    # Record system metrics
    monitor.record_system()

    # Generate reports
    monitor.generate_reports()

    # Print daily summary
    print(monitor.daily_summary())
"""

from __future__ import annotations

from src.monitoring.models import (
    ETLMetric, SystemMetric, DataQualityMetric, CacheMetric,
    MetricSnapshot, TrendLine,
)
from src.monitoring.collectors import (
    SystemCollector, ETLMetricCollector,
    DataQualityCollector, CacheMetricCollector,
)
from src.monitoring.store import MonitoringStore
from src.monitoring.reports import (
    ReportGenerator,
    HTMLReport, JSONReport, CSVReport, DailySummaryReport,
)
from src.monitoring.cli import main as cli_main
from src.monitoring.monitor import Monitor

__all__ = [
    "ETLMetric", "SystemMetric", "DataQualityMetric", "CacheMetric",
    "MetricSnapshot", "TrendLine",
    "SystemCollector", "ETLMetricCollector",
    "DataQualityCollector", "CacheMetricCollector",
    "MonitoringStore",
    "ReportGenerator",
    "HTMLReport", "JSONReport", "CSVReport", "DailySummaryReport",
    "Monitor",
    "cli_main",
]
