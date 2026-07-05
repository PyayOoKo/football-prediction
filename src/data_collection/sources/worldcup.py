"""
World Cup 2026 — match data from openfootball/worldcup.json.

Downloads the free, public-domain JSON dataset from the openfootball project
on GitHub and converts it to the project's standard schema so it flows through
the existing preprocessing → feature engineering pipeline.

Source
------
https://github.com/openfootball/worldcup.json
Licence: Public Domain

The endpoint returns a JSON object with a ``matches`` array.  Each match has:

.. code-block:: json

    {
      "round":   "Matchday 1",
      "date":    "2026-06-11",
      "time":    "17:00",
      "team1":   "Mexico",
      "team2":   "Canada",
      "group":   "Group A",
      "ground":  "Estadio Azteca",
      "score":   { "ft": [1, 0], "ht": [1, 0] },
      "goals1":  [{"name": "...", "minute": 35}],
      "goals2":  []
    }

Knockout rounds use placeholder team names (``W101``, ``R102``, etc.) before
the actual qualifying teams are known.

Typical usage::

    from src.data_collection.sources.worldcup import download_worldcup

    df = download_worldcup()
    print(f"{len(df)} matches loaded")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

WORLDCUP_JSON_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)
"""URL of the 2026 World Cup dataset (public-domain JSON)."""

SOURCE_NAME = "openfootball/worldcup.json"
"""Source identifier stored in the ``source`` column."""

SEASON = "2026"
"""Season code for the 2026 World Cup."""

LEAGUE = "WC"
"""League code used in the ``league`` column."""

_TEAM_PLACEHOLDER_PREFIXES = ("W", "R", "Q", "P", "L")
"""Prefixes used for placeholder team names in undecided knockout matches.

- W = Winner of match (e.g. W89)
- R = Runner-up/Repechage (e.g. R102)
- Q = Qualifier (e.g. Q3)
- P = Play-off winner (e.g. P1)
- L = Loser of match (e.g. L101 — used in third-place playoff)
"""


# ── Public API ──────────────────────────────────────────


def download_worldcup(
    url: str = WORLDCUP_JSON_URL,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download the full 2026 World Cup match dataset.

    Parameters
    ----------
    url : str
        URL of the openfootball JSON file (default the master branch 2026 file).
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        All 104 World Cup matches in the project's standard schema with
        columns: ``date``, ``season``, ``league``, ``home_team``,
        ``away_team``, ``result``, ``home_goals``, ``away_goals``,
        ``home_goals_ht``, ``away_goals_ht``, ``round``, ``group``,
        ``ground``, ``source``, ``downloaded_at``.

    Raises
    ------
    requests.RequestException
        If the download fails.
    ValueError
        If the JSON structure is unexpected.
    """
    logger.info("Downloading 2026 World Cup data from %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    data: dict[str, Any] = resp.json()

    if "matches" not in data:
        raise ValueError(
            f"Unexpected JSON structure — expected 'matches' key, "
            f"got top-level keys: {list(data.keys())}"
        )

    raw_matches: list[dict[str, Any]] = data["matches"]
    logger.info("Fetched %d match records", len(raw_matches))

    rows: list[dict[str, Any]] = []
    for m in raw_matches:
        rows.append(_convert_match(m))

    df = pd.DataFrame(rows)

    # Parse dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sort chronologically
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Tag metadata
    df["source"] = SOURCE_NAME
    df["downloaded_at"] = datetime.now().isoformat()

    logger.info(
        "World Cup data ready — %d matches, %d with scores",
        len(df),
        df["result"].notna().sum(),
    )
    return df


def is_placeholder_team(name: str) -> bool:
    """Return ``True`` if *name* is a knockout placeholder (e.g. ``W101``)."""
    return (
        isinstance(name, str)
        and len(name) > 0
        and name[0] in _TEAM_PLACEHOLDER_PREFIXES
        and name[1:].isdigit()
    )


def get_group_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to group-stage matches only."""
    return df[df["round"].str.startswith("Matchday", na=False)].copy()


def get_knockout_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to knockout-round matches only."""
    return df[~df["round"].str.startswith("Matchday", na=False)].copy()


def get_completed_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to matches that have a result (score available)."""
    return df[df["result"].notna()].copy()


def get_upcoming_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to matches still to be played (no result)."""
    return df[df["result"].isna()].copy()


# ── Internal helpers ────────────────────────────────────


def _convert_match(m: dict[str, Any]) -> dict[str, Any]:
    """Convert a single openfootball match dict to the project schema."""
    team1 = m.get("team1", "")
    team2 = m.get("team2", "")
    score = m.get("score") or {}
    ft = score.get("ft") if isinstance(score, dict) else None
    ht = score.get("ht") if isinstance(score, dict) else None

    # Determine result from full-time score
    result: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    home_goals_ht: int | None = None
    away_goals_ht: int | None = None

    if isinstance(ft, (list, tuple)) and len(ft) >= 2:
        try:
            home_goals = int(ft[0])
            away_goals = int(ft[1])
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"
        except (TypeError, ValueError):
            pass

    if isinstance(ht, (list, tuple)) and len(ht) >= 2:
        try:
            home_goals_ht = int(ht[0])
            away_goals_ht = int(ht[1])
        except (TypeError, ValueError):
            pass

    return {
        "season": SEASON,
        "date": m.get("date"),
        "league": LEAGUE,
        "round": m.get("round"),
        "group": m.get("group"),
        "ground": m.get("ground"),
        "home_team": team1,
        "away_team": team2,
        "result": result,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_goals_ht": home_goals_ht,
        "away_goals_ht": away_goals_ht,
        "is_knockout_placeholder": is_placeholder_team(team1) or is_placeholder_team(team2),
    }
