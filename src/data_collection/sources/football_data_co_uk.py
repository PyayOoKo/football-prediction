"""
Football-Data.co.uk — CSV-based downloader for historical match data.

Downloads clean CSV files from https://www.football-data.co.uk/.

Features
--------
- Zero registration, no API key needed
- Supports Premier League (E0) and dozens of other leagues
- Season-by-season or one-shot bulk download
- Auto-detects available seasons from the archive page
- Returns standardised DataFrames with consistent column names

URL conventions
---------------
- Base: ``https://www.football-data.co.uk/mmz4281/{season}/{league}.csv``
- Season encoding: first two digits of start year + last two of end year
  (e.g. ``2425`` for the 2024/25 season).
- League encoding: ``E0`` = Premier League, ``E1`` = Championship, etc.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import date, datetime
from typing import Any, Literal
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import config as _global_config
from src.data_collection.cleaners import MATCH_KEY_COLS

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

BASE_URL = "https://www.football-data.co.uk"
"""Root URL of football-data.co.uk."""

MMZ_PATH = "mmz4281"
"""Path prefix used for season-based CSV files."""

NEW_CSV_URL = "https://www.football-data.co.uk/new/{league}.csv"
"""Mirror URL for the *current* in-progress season (updated weekly)."""

ARCHIVE_URL = urljoin(BASE_URL, "downloadm.php")
"""Page listing all available season folders."""

# Default league code for Premier League
PREMIER_LEAGUE_CODE = "E0"

# Map of league codes to their readable names
LEAGUE_NAMES: dict[str, str] = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
    "SC0": "Scottish Premiership",
    "D1": "German Bundesliga",
    "D2": "German 2. Bundesliga",
    "I1": "Italian Serie A",
    "I2": "Italian Serie B",
    "SP1": "Spanish La Liga",
    "SP2": "Spanish Segunda Division",
    "F1": "French Ligue 1",
    "F2": "French Ligue 2",
    "N1": "Dutch Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Portuguese Primeira Liga",
    "T1": "Turkish Super Lig",
}



# ── Session ─────────────────────────────────────────────


def _session() -> requests.Session:
    """Create a requests session with retry logic."""
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; FootballPrediction/0.1.0)",
            "Accept": "text/csv, text/html, application/zip, */*",
        }
    )
    return sess


# ── Public API ──────────────────────────────────────────


def download_season(
    season: str,
    league: str = PREMIER_LEAGUE_CODE,
) -> pd.DataFrame:
    """Download a single season's match data as a DataFrame.

    Parameters
    ----------
    season : str
        Season code in ``YYYY`` short form, e.g. ``"2425"`` for 2024/25.
        Also accepts ``"current"`` to fetch the latest available season.
    league : str
        League code (default ``"E0"`` = Premier League).

    Returns
    -------
    pd.DataFrame
        Cleaned match results for the requested season.

    Raises
    ------
    requests.HTTPError
        If the CSV is not found (HTTP 404) or the server errors.
    """
    if season == "current":
        return _download_current(league)

    url = f"{BASE_URL}/{MMZ_PATH}/{season}/{league}.csv"
    logger.info("Downloading season %s (%s) from %s", season, league, url)

    resp = _session().get(url, timeout=30)
    resp.raise_for_status()

    df = _parse_csv(resp.text, season=season)
    logger.info("Downloaded %d rows for season %s", len(df), season)
    return df


def download_bulk(
    leagues: list[str] | None = None,
    max_seasons: int = 10,
    include_current: bool = True,
    config: Any | None = None,
) -> pd.DataFrame:
    """Download multiple recent seasons across one or more leagues.

    Parameters
    ----------
    leagues : list[str], optional
        League codes to download.  Defaults to ``[Premier League]``.
    max_seasons : int
        Number of most-recent seasons to fetch per league (default 10).
    include_current : bool
        If ``True``, also fetch the current in-progress season from the
        ``/new/`` mirror (which is not yet on the archive page).
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all downloaded seasons.
    """
    cfg = config or _global_config
    if leagues is None:
        leagues = [cfg.data_collection.leagues[0]]

    all_dfs: list[pd.DataFrame] = []
    seasons = _get_recent_seasons(max_seasons)

    for league in leagues:
        # Archived (completed) seasons
        for season in seasons:
            try:
                df = download_season(season, league)
                all_dfs.append(df)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.warning(
                        "Season %s not available for league %s — skipping",
                        season,
                        league,
                    )
                    continue
                raise

        # Current in-progress season (not yet in the archive)
        if include_current:
            try:
                df = _download_current(league)
                all_dfs.append(df)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.warning(
                        "Current season not available for league %s — skipping",
                        league,
                    )
                else:
                    logger.warning(
                        "Failed to download current season for %s: %s",
                        league,
                        exc,
                    )

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    logger.info("Bulk download complete — %d total rows", len(combined))
    return combined


def get_available_seasons() -> list[str]:
    """Scrape the archive page and return a list of all season codes.

    Returns
    -------
    list[str]
        Sorted list of season codes (e.g. ``["9394", "9495", ..., "2425"]``).
    """
    logger.info("Fetching available seasons from %s", ARCHIVE_URL)
    resp = _session().get(ARCHIVE_URL, timeout=30)
    resp.raise_for_status()

    # Extract all ``<a href="...">`` links containing season folders
    pattern = re.compile(rf'({MMZ_PATH})/\d{{4}}')
    matches = pattern.findall(resp.text)
    seasons = sorted({m.split("/")[1] for m in matches})
    logger.info("Found %d available seasons", len(seasons))
    return seasons


# ── Internal helpers ────────────────────────────────────


def _download_current(league: str) -> pd.DataFrame:
    """Download the current in-progress season from the ``/new/`` mirror."""
    url = NEW_CSV_URL.format(league=league)
    logger.info("Downloading current season from %s", url)

    resp = _session().get(url, timeout=30)
    resp.raise_for_status()

    df = _parse_csv(resp.text)
    # The "new" CSV doesn't have a season column by default — we tag it
    current_season = _guess_current_season()
    if "season" not in df.columns:
        df["season"] = current_season
    logger.info("Downloaded %d rows for current season (%s)", len(df), current_season)
    return df


def _parse_csv(text: str, season: str | None = None) -> pd.DataFrame:
    """Parse raw CSV text into a standardised DataFrame."""
    df = pd.read_csv(io.StringIO(text), na_values=["", "NA", "N/A", "NULL"])
    df = _standardise_columns(df)
    df = _add_metadata(df, season)
    df = _parse_dates(df)
    return df


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a consistent snake_case schema."""
    rename_map = {
        "div": "league",
        "date": "date",
        "hometeam": "home_team",
        "awayteam": "away_team",
        "fthg": "home_goals",
        "ftag": "away_goals",
        "ftr": "result",  # H / D / A
        "hthg": "home_goals_ht",
        "htag": "away_goals_ht",
        "htr": "result_ht",
        "hs": "home_shots",
        "as": "away_shots",
        "hst": "home_shots_target",
        "ast": "away_shots_target",
        "hc": "home_corners",
        "ac": "away_corners",
        "hf": "home_fouls",
        "af": "away_fouls",
        "hy": "home_yellow",
        "ay": "away_yellow",
        "hr": "home_red",
        "ar": "away_red",
    }

    df.columns = [col.strip().lower() for col in df.columns]
    df.rename(columns=rename_map, inplace=True)
    return df


def _add_metadata(df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    """Add season and source tracking columns."""
    if season and "season" not in df.columns:
        df["season"] = season
    df["source"] = "football-data.co.uk"
    df["downloaded_at"] = datetime.now().isoformat()
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the date column into a standard datetime."""
    if "date" in df.columns:
        # Try common formats: DD/MM/YY, DD/MM/YYYY, YYYY-MM-DD
        df["date"] = pd.to_datetime(
            df["date"],
            dayfirst=True,
            errors="coerce",
        )
    return df


def _get_recent_seasons(n: int) -> list[str]:
    """Return the *n* most recent season codes.

    Tries the archive page first; falls back to algorithmic generation
    (since season codes are predictable).
    """
    try:
        all_seasons = get_available_seasons()
        return all_seasons[-n:] if len(all_seasons) >= n else all_seasons
    except requests.RequestException as exc:
        logger.warning(
            "Could not fetch season archive (%s) — generating seasons algorithmically",
            exc,
        )
        return _generate_season_codes(n)


def _generate_season_codes(n: int) -> list[str]:
    """Generate the *n* most recent season codes algorithmically.

    Season codes are deterministic: the first two digits are the start year's
    last two digits, and the next two are the end year's last two digits.
    Example: 2024/25 → ``"2425"``.

    Parameters
    ----------
    n : int
        Number of season codes to generate.

    Returns
    -------
    list[str]
        List of season codes, oldest first.
    """
    from datetime import date

    today = date.today()
    # A season starts in August; if we're before August, the current season
    # started the previous calendar year.
    if today.month >= 8:
        end_year = today.year + 1
    else:
        end_year = today.year

    seasons: list[str] = []
    for i in range(n):
        ey = end_year - i
        sy = ey - 1
        seasons.append(f"{str(sy)[2:]}{str(ey)[2:]}")

    return list(reversed(seasons))


def _guess_current_season() -> str:
    """Return the season code for the current ongoing season (e.g. \"2425\")."""
    today = date.today()
    year = today.year
    if today.month >= 8:  # Northern hemisphere season starts Aug/Sep
        start = str(year)[2:]
        end = str(year + 1)[2:]
    else:
        start = str(year - 1)[2:]
        end = str(year)[2:]
    return f"{start}{end}"
