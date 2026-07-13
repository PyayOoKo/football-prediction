"""
Profiling Integration — automatically profile datasets after every ETL import.

Hooks into the data collection and import pipeline to run profiling after
every import completes and generate HTML, JSON, and CSV reports.

Usage
-----
::

    from src.data_profiling.integration import auto_profile_collect, enable_auto_profiling

    # Method 1: Wrap a collector result
    report = await collect_all()
    auto_profile_collect(report, "historical")

    # Method 2: Enable auto-profiling globally (monkey-patches collection functions)
    enable_auto_profiling()

    # Method 3: Run profiling explicitly
    from src.data_profiling import DataProfiler
    profiler = DataProfiler()
    profiler.profile(df, source_name="my_data")
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.data_profiling import DataProfiler, DataDriftDetector

logger = logging.getLogger(__name__)


# ── Helper: profile a file or DataFrame ────────────────

def profile_path(
    data_path: str | Path,
    source_name: str | None = None,
    reports_dir: str | Path = "reports/profiling",
    compare_previous: bool = True,
) -> dict[str, Any]:
    """Profile a CSV/Parquet file and save all report formats.

    Parameters
    ----------
    data_path : str | Path
        Path to the data file.
    source_name : str, optional
        Source identifier. Defaults to the file stem.
    reports_dir : str | Path
        Output directory for reports.
    compare_previous : bool
        If True, compare with the previous profile for drift detection.

    Returns
    -------
    dict[str, Any]
        Report with keys ``success``, ``path``, ``html_path``, ``json_path``,
        ``csv_path``, ``drift`` (optional), and ``error``.
    """
    data_path = Path(data_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source = source_name or data_path.stem

    try:
        # Load data
        if data_path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, low_memory=False)

        if df.empty:
            return {"success": False, "error": "Empty dataset", "path": str(data_path)}

        # Profile
        profiler = DataProfiler()
        report = profiler.profile(df, source_name=source)

        # Save reports
        report.to_json(str(reports_dir / f"{source}.json"))
        report.to_csv(str(reports_dir / f"{source}.csv"))
        report.to_html(str(reports_dir / f"{source}.html"))

        result: dict[str, Any] = {
            "success": True,
            "path": str(data_path),
            "html_path": str(reports_dir / f"{source}.html"),
            "json_path": str(reports_dir / f"{source}.json"),
            "csv_path": str(reports_dir / f"{source}.csv"),
            "n_rows": report.n_rows,
            "n_columns": report.n_columns,
            "duration": round(report.duration_seconds, 2),
        }

        # Compare with previous if available
        if compare_previous:
            prev_path = reports_dir / f"{source}.previous.json"
            if prev_path.exists():
                try:
                    from src.data_profiling.cli import _load_report
                    prev_report = _load_report(str(prev_path))
                    detector = DataDriftDetector()
                    drift = detector.detect(report, prev_report)
                    result["drift"] = drift.to_dict()
                    if not drift.passed:
                        logger.warning(
                            "Data drift detected: %d signal(s)",
                            drift.n_warnings,
                        )
                except Exception as exc:
                    logger.warning("Drift comparison failed: %s", exc)

        # Save as previous for next comparison
        import shutil
        current_path = reports_dir / f"{source}.json"
        if current_path.exists():
            shutil.copy2(current_path, reports_dir / f"{source}.previous.json")

        logger.info(
            "Profiling complete: %s → %s (HTML/JSON/CSV) in %.2fs",
            source, reports_dir, report.duration_seconds,
        )
        return result

    except Exception as exc:
        logger.exception("Profiling failed for %s: %s", data_path, exc)
        return {"success": False, "error": str(exc), "path": str(data_path)}


# ── Hook: profile after a collector result ─────────────

def auto_profile_collect(
    collect_result: dict[str, Any],
    source_name: str | None = None,
    reports_dir: str | Path = "reports/profiling",
) -> dict[str, Any]:
    """Profile the dataset produced by a collector function.

    Call this after ``collect_all()``, ``collect_worldcup()``,
    ``collect_league()``, or ``update()``.

    Parameters
    ----------
    collect_result : dict
        The dictionary returned by a collector function. Must contain
        ``"path"`` key pointing to the saved CSV file.
    source_name : str, optional
        Override the source name. Defaults to the file stem.
    reports_dir : str | Path
        Output directory for reports.

    Returns
    -------
    dict[str, Any]
        Profiling result dictionary.
    """
    data_path = collect_result.get("path")
    if not data_path:
        return {"success": False, "error": "No 'path' in collect result"}

    return profile_path(data_path, source_name=source_name, reports_dir=reports_dir)


# ── Hook: profile after an importer result ────────────

def auto_profile_import(
    import_reports: list[Any],
    reports_dir: str | Path = "reports/profiling",
) -> list[dict[str, Any]]:
    """Profile datasets imported by ``FootballDataImporter``.

    Profiles each successful league+season import that has a corresponding
    CSV file in the raw directory.

    Parameters
    ----------
    import_reports : list[ImportReport]
        List of ImportReport objects from ``FootballDataImporter``.
    reports_dir : str | Path
        Output directory for reports.

    Returns
    -------
    list[dict[str, Any]]
        One profiling result per successful imported dataset.
    """
    results: list[dict[str, Any]] = []
    for imp in import_reports:
        if not imp.success:
            continue

        # Build expected path: data/raw/football-data/{league}_{season}.csv
        league = getattr(imp, "league", "unknown")
        season = getattr(imp, "season", "unknown")
        csv_path = Path("data/raw/football-data") / f"{league}_{season}.csv"

        if csv_path.exists():
            src = f"{league}_{season}"
            result = profile_path(csv_path, source_name=src, reports_dir=reports_dir)
            result["league"] = league
            result["season"] = season
            results.append(result)

    return results


# ── Monkey-patching helpers ────────────────────────────

def _patch_collect_fn(
    fn: Callable[..., dict[str, Any]],
    source_name: str,
) -> Callable[..., dict[str, Any]]:
    """Wrap a collector function to auto-profile after execution."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = fn(*args, **kwargs)
        try:
            auto_profile_collect(result, source_name=source_name)
        except Exception as exc:
            logger.warning("Auto-profile failed for %s: %s", source_name, exc)
        return result
    return wrapper


def enable_auto_profiling() -> None:
    """Monkey-patch collector functions to auto-profile after every import.

    Call this once at application startup to automatically profile every
    dataset as it is imported.  This patches:
        - ``src.data_collection.collector.collect_all``
        - ``src.data_collection.collector.collect_worldcup``
        - ``src.data_collection.collector.collect_league``
        - ``src.data_collection.collector.update``

    Example::

        from src.data_profiling.integration import enable_auto_profiling
        enable_auto_profiling()
        # Now every collect_all() call also profiles the data automatically
    """
    import src.data_collection.collector as collector

    collector.collect_all = _patch_collect_fn(  # type: ignore[assignment]
        collector.collect_all, "historical"
    )
    collector.collect_worldcup = _patch_collect_fn(  # type: ignore[assignment]
        collector.collect_worldcup, "worldcup"
    )
    collector.collect_league = _patch_collect_fn(  # type: ignore[assignment]
        collector.collect_league, "league"
    )
    collector.update = _patch_collect_fn(  # type: ignore[assignment]
        collector.update, "incremental"
    )

    logger.info("Auto-profiling enabled for collector functions")


# ── ETL Pipeline profiling step ───────────────────────

class ProfilingStep:
    """A pipeline step that profiles data after the ETL Store stage.

    Attach this after the Store stage in an ETLPipeline to automatically
    profile the stored dataset.

    Parameters
    ----------
    reports_dir : str | Path
        Output directory for profiling reports.
    source_name : str
        Name identifier for the profiled dataset.
    """

    def __init__(
        self,
        reports_dir: str | Path = "reports/profiling",
        source_name: str = "etl_output",
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.source_name = source_name

    def run(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Profile data and generate reports.

        Parameters
        ----------
        data : list[dict[str, Any]]
            The processed data rows (as dicts) from the Store stage.

        Returns
        -------
        dict[str, Any]
            Profiling result.
        """
        if not data:
            return {"success": False, "error": "No data to profile"}

        df = pd.DataFrame(data)
        profiler = DataProfiler()
        report = profiler.profile(df, source_name=self.source_name)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(str(self.reports_dir / f"{self.source_name}.json"))
        report.to_csv(str(self.reports_dir / f"{self.source_name}.csv"))
        report.to_html(str(self.reports_dir / f"{self.source_name}.html"))

        logger.info("ETL profiling step complete: %s", self.source_name)
        return {
            "success": True,
            "n_rows": report.n_rows,
            "n_columns": report.n_columns,
            "duration": round(report.duration_seconds, 2),
            "html_path": str(self.reports_dir / f"{self.source_name}.html"),
        }
