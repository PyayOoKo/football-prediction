"""
Collector — high-level orchestrator for downloading, cleaning, and storing data.

Coordinates the full data collection pipeline:

1. Download raw CSV from Football-Data.co.uk (or another source)
2. Clean: deduplicate → handle missing values → standardise schema
3. Validate the resulting dataset
4. Save to ``data/raw/`` as CSV
5. Optionally update incrementally (new matches only)

Typical usage::

    from src.data_collection import collect_all, update
    collect_all()   # Full historical download for all configured leagues
    update()        # Fetch only the latest matches (incremental)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import config
from src.data_collection.cleaners import (
    deduplicate,
    handle_missing_values,
    standardise_schema,
    validate_data,
)
from src.data_collection.sources import football_data_co_uk as fdc
from src.data_collection.sources import worldcup as wc_source


logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────


def collect_worldcup(
    save: bool = True,
    output_file: str = "worldcup_2026.csv",
) -> dict[str, Any]:
    """Download, clean, and save 2026 World Cup match data.

    Fetches the free public-domain dataset from openfootball/worldcup.json,
    converts to the project's standard schema, cleans it, and saves to
    ``data/raw/``.

    Parameters
    ----------
    save : bool
        Whether to persist the data to CSV (default ``True``).
    output_file : str
        Output file name (default ``"worldcup_2026.csv"``).

    Returns
    -------
    dict[str, Any]
        Report with ``path``, ``total_matches``, ``completed``,
        ``upcoming``, ``validation``, and ``teams``.
    """
    import time

    start = time.time()
    logger.info("=" * 50)
    logger.info("COLLECTING 2026 WORLD CUP DATA")
    logger.info("=" * 50)

    # 1. Download from openfootball
    raw = wc_source.download_worldcup()
    total = len(raw)
    completed = raw["result"].notna().sum()
    upcoming = total - completed

    logger.info(
        "Downloaded %d matches (%d completed, %d upcoming)",
        total, completed, upcoming,
    )

    # 2. Clean
    df = _clean_pipeline(raw)

    # 3. Validate
    validation = validate_data(df)

    # 4. Convert to CSV-friendly format
    # Keep metadata columns useful for downstream tasks
    csv_cols = [
        "season", "date", "league", "round", "group", "ground",
        "home_team", "away_team",
        "result", "home_goals", "away_goals",
        "home_goals_ht", "away_goals_ht",
        "source", "downloaded_at",
    ]
    df_csv = df[[c for c in csv_cols if c in df.columns]].copy()

    # 5. Save
    output_path = config.paths.raw / output_file
    _save_csv(df_csv, output_path)

    elapsed = round(time.time() - start, 2)

    # Count unique teams (excluding placeholders)
    all_teams = set()
    for col in ["home_team", "away_team"]:
        for name in df[col].dropna().unique():
            if not wc_source.is_placeholder_team(name):
                all_teams.add(name)

    report = {
        "path": str(output_path),
        "total_matches": total,
        "completed": int(completed),
        "upcoming": int(upcoming),
        "teams": sorted(all_teams),
        "n_teams": len(all_teams),
        "validation": validation,
        "duration_seconds": elapsed,
    }

    logger.info(
        "World Cup data collection complete — %d matches (%d completed) → %s (%.1f s)",
        total, completed, output_path, elapsed,
    )
    return report


def collect_all() -> dict[str, Any]:
    """Download, clean, and save historical data for all configured leagues.

    This is the primary entry point for the initial data collection.

    Returns
    -------
    dict[str, Any]
        Report dictionary with keys ``path``, ``rows``, ``leagues``,
        ``validation``, and ``duration_seconds``.
    """
    import time

    start = time.time()
    logger.info("Starting full data collection for leagues: %s", config.data_collection.leagues)

    # 1. Bulk download
    raw = fdc.download_bulk(
        leagues=config.data_collection.leagues,
        max_seasons=config.data_collection.max_seasons,
    )

    if raw.empty:
        logger.warning("No data downloaded — check your network or league codes")
        return {"rows": 0, "error": "no data downloaded"}

    # 2. Clean
    df = _clean_pipeline(raw)

    # 3. Validate
    validation = validate_data(df)

    # 4. Save
    output_path = config.paths.raw / config.data_collection.output_file
    _save_csv(df, output_path)

    elapsed = round(time.time() - start, 2)
    report = {
        "path": str(output_path),
        "rows": len(df),
        "leagues": df["league"].unique().tolist() if "league" in df.columns else [],
        "validation": validation,
        "duration_seconds": elapsed,
    }

    if validation["is_valid"]:
        logger.info(
            "Data collection complete — %d rows saved to %s (%.1f s)",
            len(df),
            output_path,
            elapsed,
        )
    else:
        logger.warning(
            "Data collected with %d validation warnings — see log",
            len(validation["warnings"]),
        )

    return report


def collect_league(
    league: str | None = None,
    seasons: int | None = None,
) -> dict[str, Any]:
    """Download data for a single league.

    Parameters
    ----------
    league : str, optional
        League code (e.g. ``"E0"``).  Defaults to the first configured league.
    seasons : int, optional
        Number of recent seasons.  Defaults to ``config.data_collection.max_seasons``.

    Returns
    -------
    dict[str, Any]
        Report dictionary.
    """
    if league is None:
        league = config.data_collection.leagues[0]
    if seasons is None:
        seasons = config.data_collection.max_seasons

    logger.info("Collecting data for league %s (%d seasons)", league, seasons)

    raw = fdc.download_bulk(leagues=[league], max_seasons=seasons)
    if raw.empty:
        return {"rows": 0, "error": "no data downloaded"}

    df = _clean_pipeline(raw)
    output_path = config.paths.raw / f"results_{league.lower()}.csv"
    _save_csv(df, output_path)

    validation = validate_data(df)

    report = {
        "path": str(output_path),
        "rows": len(df),
        "league": league,
        "validation": validation,
    }
    logger.info("League collection complete — %d rows", len(df))
    return report


def update() -> dict[str, Any]:
    """Incremental update: download only the newest season and merge.

    Merges new data with the existing CSV, deduplicates, and saves.

    Returns
    -------
    dict[str, Any]
        Report dictionary with ``new_rows``, ``total_rows``.
    """
    logger.info("Running incremental update")

    # 1. Load existing data if available
    existing_path = config.paths.raw / config.data_collection.output_file
    existing = pd.DataFrame()
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        logger.info("Loaded %d existing rows from %s", len(existing), existing_path)

    # 2. Download current season for each configured league
    new_parts: list[pd.DataFrame] = []
    for league in config.data_collection.leagues:
        try:
            df = fdc.download_season("current", league)
            new_parts.append(df)
        except Exception as exc:
            logger.warning("Failed to download current season for %s: %s", league, exc)

    if not new_parts:
        logger.info("No new data available")
        return {"new_rows": 0, "total_rows": len(existing)}

    new_data = pd.concat(new_parts, ignore_index=True)

    # 3. Merge: combine existing + new, then deduplicate
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = _clean_pipeline(combined)

    # 4. Save back
    _save_csv(combined, existing_path)

    new_rows = len(combined) - len(existing)
    logger.info("Update complete — %d new rows, %d total rows", new_rows, len(combined))

    return {
        "new_rows": new_rows,
        "total_rows": len(combined),
        "path": str(existing_path),
    }


# ── Internal helpers ────────────────────────────────────


def _clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline: dedup → missing values → schema."""
    df = deduplicate(df)
    df = handle_missing_values(df, strategy=config.data_collection.missing_strategy)
    df = standardise_schema(df)
    return df


# ═══════════════════════════════════════════════════════════
#  World Cup helpers (convenience wrappers)
# ═══════════════════════════════════════════════════════════


def list_worldcup_teams() -> list[str]:
    """Return the list of real (non-placeholder) teams in the 2026 World Cup.

    Downloads the dataset in-memory and extracts unique team names.

    Returns
    -------
    list[str]
        Alphabetically sorted team names (48 expected for 2026).
    """
    df = wc_source.download_worldcup()
    teams: set[str] = set()
    for col in ["home_team", "away_team"]:
        for name in df[col].dropna().unique():
            if not wc_source.is_placeholder_team(name):
                teams.add(name)
    return sorted(teams)


def list_worldcup_groups() -> dict[str, list[str]]:
    """Return group-stage team assignments for the 2026 World Cup.

    Downloads the dataset in-memory and extracts group → teams mapping.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping group name (e.g. ``"Group A"``) to a sorted list
        of team names.
    """
    df = wc_source.download_worldcup()
    group_stage = df[df["round"].str.startswith("Matchday", na=False)]

    groups: dict[str, set[str]] = {}
    for _, row in group_stage.iterrows():
        g = row.get("group", "Unknown")
        if g not in groups:
            groups[g] = set()
        for col in ["home_team", "away_team"]:
            name = row.get(col, "")
            if name and not wc_source.is_placeholder_team(name):
                groups[g].add(name)

    return {k: sorted(v) for k, v in sorted(groups.items())}


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.debug("Saved %d rows → %s", len(df), path)
