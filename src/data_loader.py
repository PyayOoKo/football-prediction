"""
Data Loader — ingest football match data from CSV files, APIs, or databases.

Supported sources (controlled via ``config.data.source``):
    - ``"local"``  : read CSV files from ``data/raw/``
    - ``"api"``    : fetch from a REST API
    - ``"db"``     : query a relational database

Typical usage::

    from src.data_loader import load_fixtures, load_results
    fixtures = load_fixtures()
    results = load_results()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import config

logger = logging.getLogger(__name__)


# ── Public helpers ──────────────────────────────────────


def load_results(
    file_name: str | None = None,
    **csv_kwargs: Any,
) -> pd.DataFrame:
    """Load historical match results.

    Parameters
    ----------
    file_name : str, optional
        Override the default results file name from ``config``.
    **csv_kwargs
        Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned results DataFrame.
    """
    path = _resolve_path(file_name or config.data.results_file)
    logger.info("Loading results from %s", path)
    df = pd.read_csv(path, **csv_kwargs)
    df = _standardise_columns(df)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


def load_fixtures(
    file_name: str | None = None,
    **csv_kwargs: Any,
) -> pd.DataFrame:
    """Load upcoming fixtures to predict.

    Parameters
    ----------
    file_name : str, optional
        Override the default fixtures file name from ``config``.
    **csv_kwargs
        Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned fixtures DataFrame.
    """
    path = _resolve_path(file_name or config.data.fixtures_file)
    logger.info("Loading fixtures from %s", path)
    df = pd.read_csv(path, **csv_kwargs)
    df = _standardise_columns(df)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


def load_teams(
    file_name: str | None = None,
    **csv_kwargs: Any,
) -> pd.DataFrame:
    """Load team metadata (name, league, stadium, etc.).

    Parameters
    ----------
    file_name : str, optional
        Override the default teams file name from ``config``.
    **csv_kwargs
        Extra keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned teams DataFrame.
    """
    path = _resolve_path(file_name or config.data.teams_file)
    logger.info("Loading teams from %s", path)
    df = pd.read_csv(path, **csv_kwargs)
    df = _standardise_columns(df)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


# ── Internal helpers ────────────────────────────────────


def _resolve_path(file_name: str) -> Path:
    """Return the full path for *file_name* inside ``data/raw``."""
    return config.paths.raw / file_name


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase with underscores."""
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


# ── Future: API / DB loaders (stubs) ────────────────────


def load_from_api(endpoint: str | None = None) -> pd.DataFrame:
    """Fetch match data from a REST API.  *Not yet implemented.*"""
    raise NotImplementedError("API loader is not implemented.")


def load_from_db(query: str) -> pd.DataFrame:
    """Query a database for match data.  *Not yet implemented.*"""
    raise NotImplementedError("DB loader is not implemented.")
