"""
ETL Pipeline Integration — auto-generates the Data Quality Dashboard
after every successful ETL pipeline run.

Usage
-----
In your pipeline or scheduler config::

    from src.data_quality.integration import run_data_quality_pipeline

    # After ETL run
    run_data_quality_pipeline()

Or integrate directly into the pipeline script::

    # In run_pipeline.py
    from src.data_quality.integration import DataQualityPipelineHook

    hook = DataQualityPipelineHook()
    results["data_quality"] = hook.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_quality.coverage import CoverageAnalyzer
from src.data_quality.dashboard import DataQualityDashboard
from src.monitoring.monitor import Monitor
from src.monitoring.store import MonitoringStore

logger = logging.getLogger(__name__)


def run_data_quality_pipeline(
    data_path: str | Path | None = None,
    previous_data_path: str | Path | None = None,
    output_dir: str | Path = "reports/data_quality",
    source_name: str = "daily_pipeline",
) -> dict[str, Any]:
    """Run the full data quality pipeline after an ETL run.

    Parameters
    ----------
    data_path : str | Path, optional
        Path to the current cleaned results CSV. If None, uses
        ``data/processed/results_clean.csv``.
    previous_data_path : str | Path, optional
        Path to the previous version of the dataset for drift detection.
    output_dir : str | Path
        Output directory for dashboard files.
    source_name : str
        Source identifier for the dashboard.

    Returns
    -------
    dict[str, Any]
        Paths to generated files: ``html``, ``json``, ``csv``, ``summary``.
    """
    if data_path is None:
        data_path = Path("data/processed/results_clean.csv")

    df = None
    df_previous = None

    # Load current dataset
    path = Path(data_path)
    if path.exists():
        try:
            df = pd.read_csv(path, low_memory=False)
            logger.info("Loaded %d rows from %s", len(df), path)
        except Exception as exc:
            logger.warning("Failed to load current data: %s", exc)
    else:
        logger.warning("Current data not found at %s", path)

    # Load previous dataset
    if previous_data_path:
        prev_path = Path(previous_data_path)
        if prev_path.exists():
            try:
                df_previous = pd.read_csv(prev_path, low_memory=False)
                logger.info("Loaded %d rows from previous data", len(df_previous))
            except Exception as exc:
                logger.warning("Failed to load previous data: %s", exc)

    # Connect to monitoring store
    monitor_store = MonitoringStore()

    # Build and generate dashboard
    dq = DataQualityDashboard(
        df=df,
        source_name=source_name,
        output_dir=output_dir,
        monitor_store=monitor_store,
        df_previous=df_previous,
    )

    return dq.generate()


class DataQualityPipelineHook:
    """Pipeline hook that can be registered with the ETL pipeline.

    When called, runs the full data quality analysis and generates
    all report formats. Designed to be called after the Store stage.

    Parameters
    ----------
    output_dir : str | Path
        Output directory for dashboard files.
    data_path_pattern : str
        Glob pattern to find the most recent processed dataset.
    source_name : str
        Source name for the dashboard.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/data_quality",
        data_path_pattern: str = "data/processed/results_clean.csv",
        source_name: str = "pipeline_hook",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = Path(data_path_pattern)
        self.source_name = source_name
        self.monitor_store = MonitoringStore()

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the hook. Compatible with ETL pipeline callbacks.

        Returns
        -------
        dict[str, Any]
            Results dictionary with paths to generated files.
        """
        return self.run(**kwargs)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the data quality check and dashboard generation.

        Parameters
        ----------
        **kwargs
            Override parameters (data_path, source_name, output_dir).

        Returns
        -------
        dict[str, Any]
        """
        data_path = Path(kwargs.get("data_path", self.data_path))
        source_name = kwargs.get("source_name", self.source_name)
        output_dir = Path(kwargs.get("output_dir", self.output_dir))

        df = None
        if data_path.exists():
            try:
                df = pd.read_csv(data_path, low_memory=False)
                logger.info("Hook loaded %d rows from %s", len(df), data_path)
            except Exception as exc:
                logger.warning("Hook failed to load data: %s", exc)

        dq = DataQualityDashboard(
            df=df,
            source_name=source_name,
            output_dir=output_dir,
            monitor_store=self.monitor_store,
        )

        return dq.generate()
