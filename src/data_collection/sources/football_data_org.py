"""
Football-Data.org — API client stub for live/upcoming fixture data.

This module is **optional** and requires a free API key from
https://www.football-data.org/ (set the ``FOOTBALL_DATA_API_KEY`` env var).

It provides:
- Upcoming fixtures for Premier League
- Live match scores
- League standings (useful for feature engineering)

The endpoint is rate-limited to 10 requests/minute on the free tier.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import requests

from config import config

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

API_BASE = "https://api.football-data.org/v4"

COMPETITION_IDS = {
    "PL": 2021,  # Premier League
    "EL1": 2016,  # Championship
    "BL1": 2002,  # Bundesliga
    "SA": 2019,   # Serie A
    "PD": 2014,   # La Liga
    "FL1": 2015,  # Ligue 1
}

# ── Session ─────────────────────────────────────────────


def _session() -> requests.Session:
    """Create an authenticated session for football-data.org."""
    api_key = os.environ.get(config.data.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{config.data.api_key_env} environment variable is not set. "
            "Get a free key from https://www.football-data.org/"
        )

    sess = requests.Session()
    sess.headers.update({"X-Auth-Token": api_key})
    return sess


# ── Public API (stubs — will be fleshed out when needed) ─


def fetch_fixtures(
    competition_code: str | None = None,
    matchday: int | None = None,
) -> pd.DataFrame:
    """Fetch upcoming fixtures for a competition.

    Parameters
    ----------
    competition_code : str, optional
        Competition code (e.g. ``"PL"`` for Premier League).
        Defaults to Premier League.
    matchday : int, optional
        Specific matchday to filter.  If ``None``, returns all upcoming.

    Returns
    -------
    pd.DataFrame
        Fixtures with home/away teams, date, and venue.

    Raises
    ------
    NotImplementedError
        Always raised — this is a stub for future implementation.
    """
    raise NotImplementedError(
        "football-data.org integration requires an API key and is not yet implemented. "
        "Use src.data_collection.sources.football_data_co_uk for free CSV-based data."
    )


def fetch_standings(
    competition_code: str | None = None,
) -> pd.DataFrame:
    """Fetch current league standings.

    Parameters
    ----------
    competition_code : str, optional
        Competition code (default Premier League).

    Returns
    -------
    pd.DataFrame
        Standings with position, team, played, points, etc.

    Raises
    ------
    NotImplementedError
        Always raised — this is a stub for future implementation.
    """
    raise NotImplementedError(
        "football-data.org standings are not yet implemented."
    )
