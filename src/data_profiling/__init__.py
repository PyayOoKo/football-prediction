"""
Data Profiling — automatically analyze every imported dataset.

Generates comprehensive profiling reports covering missing values,
duplicates, null percentages, column distributions, goal/odds/league/
season/team distributions, outliers, data drift, and schema/type validation.

Reports are generated in three formats:
- **HTML dashboard** with interactive Plotly visualizations
- **JSON** structured report for programmatic consumption
- **CSV** for spreadsheet analysis

Every import automatically triggers a profiling run via ETL pipeline hooks.

Usage
-----
::

    from src.data_profiling import DataProfiler

    profiler = DataProfiler()
    report = profiler.profile(df)
    report.to_html("reports/profiling/report.html")
    report.to_json("reports/profiling/report.json")
    report.to_csv("reports/profiling/report.csv")

CLI
---
::

    # Profile a CSV and generate all report formats
    python -m src.data_profiling create-report data/raw/results.csv --source my_dataset

    # Compare two profiling runs
    python -m src.data_profiling compare --prev prev.json --curr curr.json

    # Auto-profile the latest dataset
    python -m src.data_profiling auto --source latest
"""

from __future__ import annotations

from src.data_profiling.profiler import DataProfiler, ProfilingReport, ProfileSection
from src.data_profiling.reports import ReportGenerator
from src.data_profiling.drift import DataDriftDetector, DriftReport, DriftMetric

__all__ = [
    "DataProfiler",
    "ProfilingReport",
    "ProfileSection",
    "ReportGenerator",
    "DataDriftDetector",
    "DriftReport",
    "DriftMetric",
]
