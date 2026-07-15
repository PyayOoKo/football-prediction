"""
StatsBomb Open Data — reader for the free public StatsBomb dataset.

Reads match event data directly from the StatsBomb open-data GitHub
repository at https://github.com/statsbomb/open-data.

The dataset includes:
- Match-level data (scores, teams, competition)
- Event-level data (passes, shots, tackles, etc.)
- Lineup data (starting XI + substitutes)
- 360.coordinates for selected competitions

Output formats
--------------
1. **Matches DataFrame** — per-match results with competition metadata
2. **Events DataFrame** — per-event granular data (shots, passes, etc.)
3. **Lineups DataFrame** — per-player appearance data
4. **Shot DataFrames** — shot-specific data with xG, location, and context

Usage
-----
    from src.data_collection.sources.statsbomb_open import (
        list_competitions,
        list_matches,
        get_match_events,
        get_match_lineups,
    )

    comps = list_competitions()
    matches = list_matches(competition_id=43, season_id=3)  # 2022 WC
    events = get_match_events(match_id=3869685)
    lineups = get_match_lineups(match_id=3869685)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
"""Base URL for the raw data files."""

MATCHES_URL = f"{RAW_BASE}/matches/{{competition_id}}/{{season_id}}.json"
"""URL pattern for match data (competition_id, season_id)."""

EVENTS_URL = f"{RAW_BASE}/events/{{match_id}}.json"
"""URL pattern for event data (match_id)."""

LINEUPS_URL = f"{RAW_BASE}/lineups/{{match_id}}.json"
"""URL pattern for lineup data (match_id)."""

COMP_URL = f"{RAW_BASE}/competitions.json"
"""URL for the competitions manifest."""

REQUEST_TIMEOUT = 30
"""HTTP request timeout in seconds."""

CACHE_DIR = "data/scrapers/statsbomb"
"""Local cache directory for JSON responses."""

# ── Well-known StatsBomb competition IDs ────────────────
# Source: https://github.com/statsbomb/open-data/blob/master/data/competitions.json
COMPETITION_IDS: dict[str, tuple[int, int]] = {
    # Note: StatsBomb open-data includes the 2018 World Cup (competition=43, season=3).
    # The 2022 World Cup is not yet in the open-data repository. Using
    # competition_name="World Cup 2022" will raise a ValueError with guidance.
    "World Cup 2018":             (43, 3),
    "UEFA Euro 2020":             (55, 5),
    "UEFA Euro 2024":             (55, 7),
    "FA Women's World Cup 2019":  (72, 2),
    "Premier League 2020/21":     (2, 44),
    "Premier League 2021/22":     (2, 93),
    "La Liga 2020/21":            (11, 45),
    "La Liga 2021/22":            (11, 94),
    "Serie A 2020/21":            (12, 46),
    "Serie A 2021/22":            (12, 95),
    "Bundesliga 2020/21":         (9, 47),
    "Bundesliga 2021/22":         (9, 96),
    "Ligue 1 2020/21":            (7, 48),
    "Ligue 1 2021/22":            (7, 97),
    "UEFA Champions League 2020/21": (16, 59),
    "UEFA Champions League 2021/22": (16, 98),
    "Major League Soccer 2021":   (217, 87),
}


# ── Data structures ─────────────────────────────────────


@dataclass
class StatsBombMatch:
    """Summary of a single match from the StatsBomb dataset."""

    match_id: int
    competition_id: int
    season_id: int
    match_date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    venue: str = ""
    referee: str = ""
    match_week: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatsBombMatch:
        """Create from a match JSON dict."""
        return cls(
            match_id=data.get("match_id", 0),
            competition_id=data.get("competition", {}).get("competition_id", 0),
            season_id=data.get("season", {}).get("season_id", 0),
            match_date=data.get("match_date", ""),
            home_team=data.get("home_team", {}).get("home_team_name", ""),
            away_team=data.get("away_team", {}).get("away_team_name", ""),
            home_score=data.get("home_score", 0),
            away_score=data.get("away_score", 0),
            venue=data.get("venue", {}).get("name", ""),
            referee=data.get("referee", {}).get("name", "") if isinstance(data.get("referee"), dict) else "",
            match_week=data.get("match_week", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "competition_id": self.competition_id,
            "season_id": self.season_id,
            "match_date": self.match_date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "venue": self.venue,
            "referee": self.referee,
            "match_week": self.match_week,
        }


# ── Session ─────────────────────────────────────────────


def _session() -> requests.Session:
    """Create a requests session with retry logic."""
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({
        "User-Agent": "FootballPrediction/2.0.0",
    })
    return sess


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def list_competitions(
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """List all competitions available in the StatsBomb open dataset.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached response (default True).

    Returns
    -------
    list[dict]
        List of competition metadata dicts with keys like
        ``competition_id``, ``competition_name``, ``season_id``, etc.
    """
    cache_path = Path(CACHE_DIR) / "competitions.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    logger.info("Fetching competitions list from StatsBomb open-data...")
    sess = _session()
    resp = sess.get(COMP_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def list_matches(
    competition_id: int | None = None,
    season_id: int | None = None,
    competition_name: str | None = None,
    use_cache: bool = True,
) -> list[StatsBombMatch]:
    """List matches for a given competition and season.

    Parameters
    ----------
    competition_id : int, optional
        StatsBomb competition ID.
    season_id : int, optional
        Season ID within the competition.
    competition_name : str, optional
        Human-readable name (e.g. ``"World Cup 2022"``).
        Preferred over (competition_id, season_id) for convenience.
    use_cache : bool
        Whether to use cached response (default True).

    Returns
    -------
    list[StatsBombMatch]
        Match objects for the requested competition/season.
    """
    if competition_name:
        comp_info = COMPETITION_IDS.get(competition_name)
        if comp_info is None:
            # Search the competitions list
            comps = list_competitions(use_cache=use_cache)
            for c in comps:
                cname = c.get("competition_name", "")
                sname = c.get("season_name", "")
                full = f"{cname} {sname}"
                if full.lower() == competition_name.lower():
                    competition_id = c["competition_id"]
                    season_id = c["season_id"]
                    break
            else:
                raise ValueError(
                    f"Competition '{competition_name}' not found. "
                    f"Use list_competitions() to see available options."
                )
        else:
            competition_id, season_id = comp_info

    if competition_id is None or season_id is None:
        raise ValueError("Must provide either competition_name or both competition_id and season_id")

    cache_path = Path(CACHE_DIR) / f"matches_{competition_id}_{season_id}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
    else:
        url = MATCHES_URL.format(competition_id=competition_id, season_id=season_id)
        logger.info("Fetching matches from %s", url)
        sess = _session()
        resp = sess.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(raw, f, indent=2)
        time.sleep(0.2)  # Polite delay

    return [StatsBombMatch.from_dict(m) for m in raw]


def get_match_events(
    match_id: int,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Fetch all events for a single match.

    Parameters
    ----------
    match_id : int
        StatsBomb match identifier.
    use_cache : bool
        Whether to use cached response (default True).

    Returns
    -------
    list[dict]
        Event data (passes, shots, tackles, etc.).
    """
    cache_path = Path(CACHE_DIR) / f"events_{match_id}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = EVENTS_URL.format(match_id=match_id)
    sess = _session()
    resp = sess.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def get_match_lineups(
    match_id: int,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Fetch lineup data for a single match.

    Parameters
    ----------
    match_id : int
        StatsBomb match identifier.
    use_cache : bool
        Whether to use cached response (default True).

    Returns
    -------
    list[dict]
        Lineup data with starting XI and substitutes per team.
    """
    cache_path = Path(CACHE_DIR) / f"lineups_{match_id}.json"

    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = LINEUPS_URL.format(match_id=match_id)
    sess = _session()
    resp = sess.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


# ═══════════════════════════════════════════════════════════
#  DataFrame constructors
# ═══════════════════════════════════════════════════════════


def matches_to_dataframe(matches: list[StatsBombMatch]) -> pd.DataFrame:
    """Convert a list of StatsBombMatch objects to a DataFrame.

    Parameters
    ----------
    matches : list[StatsBombMatch]
        Match objects from ``list_matches()``.

    Returns
    -------
    pd.DataFrame
    """
    if not matches:
        return pd.DataFrame()
    return pd.DataFrame([m.to_dict() for m in matches])


def events_to_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten StatsBomb events into a DataFrame.

    Extracts key event fields (type, player, team, location, timestamp,
    etc.) and flattens nested coordinates.

    Parameters
    ----------
    events : list[dict]
        Raw event data from ``get_match_events()``.

    Returns
    -------
    pd.DataFrame
        Flattened event data, one row per event.
    """
    flat: list[dict[str, Any]] = []

    for event in events:
        row = {
            "event_id": event.get("id", ""),
            "match_id": event.get("match_id", 0),
            "match_period": event.get("period", 0),
            "match_timestamp": event.get("timestamp", ""),
            "minute": event.get("minute", 0),
            "second": event.get("second", 0),
            "team": event.get("team", {}).get("name", ""),
            "team_id": event.get("team", {}).get("id", 0),
            "player": event.get("player", {}).get("name", ""),
            "player_id": event.get("player", {}).get("id", 0),
            "type": event.get("type", {}).get("name", ""),
            "subtype": event.get("sub_type", {}).get("name", "") if event.get("sub_type") else "",
            "possession": event.get("possession", 0),
        }

        # Location
        loc = event.get("location", [])
        row["location_x"] = loc[0] if len(loc) > 0 else None
        row["location_y"] = loc[1] if len(loc) > 1 else None

        # Shot-specific
        if event.get("shot"):
            shot = event["shot"]
            row["shot_type"] = shot.get("type", {}).get("name", "")
            row["shot_body_part"] = shot.get("body_part", {}).get("name", "")
            row["shot_outcome"] = shot.get("outcome", {}).get("name", "")
            row["xg"] = shot.get("statsbomb_xg", None)
            row["shot_end_location_x"] = (
                shot.get("end_location", [None])[0]
                if shot.get("end_location") else None
            )
            row["shot_end_location_y"] = (
                shot.get("end_location", [None, None])[1]
                if len(shot.get("end_location", [])) > 1 else None
            )

            # Freeze frame (defender positions) — too complex to flatten fully
            row["shot_freeze_frame_count"] = (
                len(shot.get("freeze_frame", []))
                if shot.get("freeze_frame") else 0
            )
        else:
            row["shot_type"] = None
            row["shot_body_part"] = None
            row["shot_outcome"] = None
            row["xg"] = None
            row["shot_end_location_x"] = None
            row["shot_end_location_y"] = None
            row["shot_freeze_frame_count"] = 0

        # Pass-specific
        if event.get("pass"):
            p = event["pass"]
            row["pass_length"] = p.get("length", None)
            row["pass_angle"] = p.get("angle", None)
            row["pass_height"] = p.get("height", {}).get("name", "") if p.get("height") else ""
            row["pass_recipient"] = p.get("recipient", {}).get("name", "") if p.get("recipient") else ""
            row["pass_goal_assist"] = p.get("goal_assist", False)
            row["pass_shot_assist"] = p.get("shot_assist", False)
            row["pass_cross"] = p.get("cross", False)
            row["pass_through_ball"] = p.get("through_ball", False)

        flat.append(row)

    return pd.DataFrame(flat)


def shots_to_dataframe(
    match_ids: list[int],
    use_cache: bool = True,
) -> pd.DataFrame:
    """Extract shot events from multiple matches into a single DataFrame.

    Convenience method that fetches events for all given match_ids
    and filters to shot-type events only.

    Parameters
    ----------
    match_ids : list[int]
        StatsBomb match identifiers.
    use_cache : bool
        Whether to use cached responses (default True).

    Returns
    -------
    pd.DataFrame
        Shot-level data with xG, location, outcome, body part.
    """
    all_shots: list[dict[str, Any]] = []

    for mid in match_ids:
        events = get_match_events(mid, use_cache=use_cache)
        for event in events:
            if event.get("type", {}).get("name") == "Shot":
                shot_data = event.get("shot", {})
                row = {
                    "match_id": mid,
                    "player": event.get("player", {}).get("name", ""),
                    "team": event.get("team", {}).get("name", ""),
                    "minute": event.get("minute", 0),
                    "xg": shot_data.get("statsbomb_xg"),
                    "shot_type": shot_data.get("type", {}).get("name", ""),
                    "shot_body_part": shot_data.get("body_part", {}).get("name", ""),
                    "shot_outcome": shot_data.get("outcome", {}).get("name", ""),
                }
                loc = event.get("location", [])
                row["location_x"] = loc[0] if len(loc) > 0 else None
                row["location_y"] = loc[1] if len(loc) > 1 else None

                # Freeze frame count
                row["shot_freeze_frame_count"] = (
                    len(shot_data.get("freeze_frame", []))
                    if shot_data.get("freeze_frame") else 0
                )
                all_shots.append(row)

    return pd.DataFrame(all_shots)


def lineups_to_dataframe(
    match_id: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Get lineup data for a match as a DataFrame.

    Parameters
    ----------
    match_id : int
        StatsBomb match identifier.
    use_cache : bool
        Whether to use cached response (default True).

    Returns
    -------
    pd.DataFrame
        Per-player lineup data with position, jersey number, starter status.
    """
    lineups = get_match_lineups(match_id, use_cache=use_cache)
    rows: list[dict[str, Any]] = []

    for team_lineup in lineups:
        team_name = team_lineup.get("team_name", "")
        for player in team_lineup.get("lineup", []):
            rows.append({
                "match_id": match_id,
                "team": team_name,
                "player_id": player.get("player_id", 0),
                "player_name": player.get("player_name", ""),
                "jersey_number": player.get("jersey_number", 0),
                "position": player.get("position", {}).get("name", ""),
                "position_id": player.get("position", {}).get("id", 0),
                "starting": player.get("starting", False),
                "substitute": not player.get("starting", False),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
#  Data saver helper
# ═══════════════════════════════════════════════════════════


def save_match_data(
    match: StatsBombMatch,
    output_dir: str | None = None,
    use_cache: bool = True,
) -> dict[str, str]:
    """Save all data for a single match to CSV files.

    Parameters
    ----------
    match : StatsBombMatch
        The match to save.
    output_dir : str, optional
        Output directory (default ``data/scrapers/statsbomb``).
    use_cache : bool
        Whether to use cached API responses (default True).

    Returns
    -------
    dict[str, str]
        Mapping of data type → file path.
    """
    if output_dir is None:
        output_dir = str(Path(CACHE_DIR) / f"match_{match.match_id}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Events
    events = get_match_events(match.match_id, use_cache=use_cache)
    events_df = events_to_dataframe(events)
    if not events_df.empty:
        epath = out / "events.csv"
        events_df.to_csv(epath, index=False)
        paths["events"] = str(epath)

    # Lineups
    lineups_df = lineups_to_dataframe(match.match_id, use_cache=use_cache)
    if not lineups_df.empty:
        lpath = out / "lineups.csv"
        lineups_df.to_csv(lpath, index=False)
        paths["lineups"] = str(lpath)

    # Shots
    shots_df = shots_to_dataframe([match.match_id], use_cache=use_cache)
    if not shots_df.empty:
        spath = out / "shots.csv"
        shots_df.to_csv(spath, index=False)
        paths["shots"] = str(spath)

    return paths


# ═══════════════════════════════════════════════════════════
#  CLI for testing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Test: list competitions
    print("\n  Fetching competitions list...")
    comps = list_competitions()
    print(f"  Found {len(comps)} competitions\n")

    # Test: list matches for World Cup 2022
    print("  Fetching 2022 World Cup matches...")
    try:
        matches = list_matches(competition_name="World Cup 2022")
        print(f"  Found {len(matches)} matches\n")
        for m in matches[:5]:
            print(f"    {m.match_date}: {m.home_team} {m.home_score}-{m.away_score} {m.away_team}")
    except Exception as exc:
        print(f"  [W] Could not fetch matches: {exc}")
