"""
Integration — automatically create dataset versions when importing data.

Provides monkey-patching hooks for the existing ``collector`` and
``FootballDataImporter`` so that every import automatically creates a
versioned snapshot.

Usage
-----
::

    # Enable versioning for all imports
    from src.data_versioning.integration import patch_all
    patch_all()

    # Now every call to collect_all(), collect_worldcup(), etc.
    # will automatically create a versioned snapshot.

    # Or use the context manager:
    from src.data_versioning.integration import versioning_context
    with versioning_context(source="football-data"):
        from src.data_collection import collector
        report = collector.collect_all()

    # Or manually version a DataFrame:
    from src.data_versioning.integration import version_dataframe
    info = version_dataframe(df, source="worldcup", league="WC", season="2026")
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

import pandas as pd

from src.data_versioning import VersionManager

logger = logging.getLogger(__name__)

# ── Global version manager instance ─────────────────────
_manager: VersionManager | None = None


def get_manager() -> VersionManager:
    """Return the global VersionManager singleton."""
    global _manager
    if _manager is None:
        _manager = VersionManager(data_dir="data/versions")
    return _manager


def set_manager(mgr: VersionManager) -> None:
    """Override the global VersionManager (for testing or custom config)."""
    global _manager
    _manager = mgr


# ── Version a DataFrame ─────────────────────────────────


def version_dataframe(
    df: pd.DataFrame,
    source: str = "",
    league: str = "",
    season: str = "",
    user: str = "pipeline",
    notes: str = "",
) -> Any:
    """Create a versioned snapshot from a DataFrame.

    This is the core function called by all integration hooks.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to version.
    source : str
        Data source identifier.
    league : str
        League code.
    season : str
        Season identifier.
    user : str
        User/process creating the version.
    notes : str
        Optional notes.

    Returns
    -------
    VersionInfo
        The created version metadata, or None if df is empty.
    """
    if df is None or df.empty:
        logger.warning("Skipping versioning: empty DataFrame")
        return None

    mgr = get_manager()
    info = mgr.create_version(
        df=df,
        source=source,
        league=league,
        season=season,
        user=user,
        notes=notes,
    )
    logger.info("Versioned %d rows as %s (%s/%s)", len(df), info.version_id, source, league)
    return info


# ── Decorator / wrapper ─────────────────────────────────


def _wrap_collector_fn(fn: Callable, source: str, league_fn: Callable | None = None) -> Callable:
    """Wrap a collector function so its result is versioned automatically."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Determine league from args/kwargs if possible
        league = ""
        season = ""
        if league_fn:
            league = league_fn(*args, **kwargs)
        elif args and isinstance(args[0], str):
            league = args[0]

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if isinstance(result, dict):
            # Collector functions return dict reports
            path = result.get("path", "")
            n_rows = result.get("rows", result.get("total_matches", 0))
            league = league or result.get("league", "")
            season = result.get("season", str(result.get("season", "")))

            # Try to load CSV that was just saved
            try:
                df = pd.read_csv(path, low_memory=False) if path else pd.DataFrame()
                notes = f"Auto-versioned from {fn.__name__}"
                version_dataframe(
                    df, source=source, league=league,
                    season=season, notes=notes,
                )
            except Exception as exc:
                logger.warning("Failed to version result of %s: %s", fn.__name__, exc)

        return result

    return wrapper


# ── Patch functions ─────────────────────────────────────


def patch_collector() -> None:
    """Patch ``src.data_collection.collector`` functions to auto-version."""
    try:
        from src.data_collection import collector as c

        c.collect_all = _wrap_collector_fn(
            c.collect_all, source="football-data-co-uk",
        )
        c.collect_worldcup = _wrap_collector_fn(
            c.collect_worldcup, source="worldcup",
        )
        c.update = _wrap_collector_fn(c.update, source="football-data-co-uk")

        logger.info("Patched collector functions for auto-versioning")
    except ImportError as exc:
        logger.warning("Could not patch collector: %s", exc)


def patch_importer() -> None:
    """Patch ``FootballDataImporter`` to auto-version on import."""
    try:
        from src.importers.football_data import FootballDataImporter

        original_import = FootballDataImporter._import_single

        def _versioned_import(self, league: str, season: str) -> Any:
            start = time.perf_counter()
            report = original_import(self, league, season)
            elapsed = time.perf_counter() - start

            if report.success and report.rows_imported > 0:
                # Load the imported file and version it
                try:
                    raw_path = self.raw_dir / f"{league}_{season}.csv"
                    if raw_path.exists():
                        df = pd.read_csv(raw_path, low_memory=False)
                        notes = f"Imported {report.rows_imported} rows"
                        version_dataframe(
                            df, source="football-data-co-uk",
                            league=league, season=season,
                            notes=notes,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to version import %s/%s: %s",
                        league, season, exc,
                    )

            return report

        FootballDataImporter._import_single = _versioned_import
        logger.info("Patched FootballDataImporter for auto-versioning")

    except ImportError as exc:
        logger.warning("Could not patch importer: %s", exc)


def patch_all() -> None:
    """Apply all versioning patches to the existing pipeline."""
    patch_collector()
    patch_importer()
    logger.info("All versioning patches applied")


# ── Context manager ─────────────────────────────────────


class versioning_context:
    """Context manager that enables versioning for the wrapped code block.

    Usage::

        with versioning_context(source="worldcup", league="WC"):
            from src.data_collection import collector
            report = collector.collect_worldcup()
    """

    def __init__(
        self,
        source: str = "",
        league: str = "",
        season: str = "",
        user: str = "pipeline",
    ) -> None:
        self.source = source
        self.league = league
        self.season = season
        self.user = user

    def __enter__(self) -> VersionManager:
        # Always patch on enter
        patch_all()
        return get_manager()

    def __exit__(self, *args: Any) -> None:
        pass
