"""
Data Quality Dashboard — comprehensive monitoring and reporting for football data.

Combines monitoring metrics, profiling reports, validation results, and
coverage statistics into a single professional-quality HTML dashboard with
interactive Plotly trend charts, JSON/CSV exports, and automatic generation
after every ETL pipeline run.

Quick Start
-----------
::

    from src.data_quality import DataQualityDashboard

    # Build and generate a complete dashboard
    dq = DataQualityDashboard(df=df, source_name="my_dataset")
    dq.generate()

    # Or generate from existing monitoring store
    dq = DataQualityDashboard.from_monitor(monitor)
    dq.generate()

CLI
---
::

    # Generate dashboard from the latest data
    python -m src.data_quality.cli generate

    # Coverage analysis only
    python -m src.data_quality.cli coverage

    # Serve as a simple HTTP page
    python -m src.data_quality.cli serve
"""

from __future__ import annotations

from src.data_quality.models import (
    CoverageMetrics,
    DataQualitySnapshot,
    DataQualitySummary,
)
from src.data_quality.coverage import CoverageAnalyzer
from src.data_quality.dashboard import DataQualityDashboard

__all__ = [
    "CoverageMetrics",
    "DataQualitySnapshot",
    "DataQualitySummary",
    "CoverageAnalyzer",
    "DataQualityDashboard",
]
