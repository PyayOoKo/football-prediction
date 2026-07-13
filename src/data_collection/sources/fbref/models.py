"""
Data models for FBref scraped statistics.

Each stat category has its own dataclass with typed fields matching
the columns in FBref's HTML tables.  Raw data from the parser is
converted into these models for type safety and downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StatCategory(str, Enum):
    """FBref stat category identifiers used in URLs and table IDs."""

    STANDARD = "stats_standard"
    SHOOTING = "stats_shooting"
    PASSING = "stats_passing"
    PASSING_TYPES = "stats_passing_types"
    GOAL_CREATION = "stats_gca"
    DEFENSE = "stats_defense"
    POSSESSION = "stats_possession"
    MISCELLANEOUS = "stats_misc"
    KEEPING = "stats_keeper"
    KEEPING_ADV = "stats_keeper_adv"
    PLAYING_TIME = "stats_playing_time"
    MATCH_STATS = "stats_match"


# Map category enums to URL path segments
CATEGORY_URL_MAP: dict[StatCategory, str] = {
    StatCategory.STANDARD: "",
    StatCategory.SHOOTING: "shooting",
    StatCategory.PASSING: "passing",
    StatCategory.PASSING_TYPES: "passing_types",
    StatCategory.GOAL_CREATION: "gca",
    StatCategory.DEFENSE: "defense",
    StatCategory.POSSESSION: "possession",
    StatCategory.MISCELLANEOUS: "misc",
    StatCategory.KEEPING: "keepers",
    StatCategory.KEEPING_ADV: "keepers_adv",
    StatCategory.PLAYING_TIME: "playing_time",
}


@dataclass
class FBrefTable:
    """A parsed FBref HTML table.

    Attributes
    ----------
    category : StatCategory
        Which stat category this table represents.
    team_name : str
        Team name extracted from the page context.
    competition : str
        Competition name (e.g. ``Premier League``).
    season : str
        Season identifier (e.g. ``2024-2025``).
    columns : list[str]
        Column names.
    rows : list[dict[str, Any]]
        Data rows as dicts keyed by column name.
    raw_html : str
        Original HTML string of the parsed table.
    """

    category: StatCategory
    team_name: str = ""
    competition: str = ""
    season: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    raw_html: str = ""


@dataclass
class PlayerStats:
    """Stats for a single player in a given category.

    Attributes
    ----------
    player_name : str
        Player's full name.
    position : str
        Position abbreviation (e.g. ``GK``, ``DF``, ``MF``, ``FW``).
    nationality : str
        Three-letter country code.
    age : int | None
        Player's age (years + days encoded as float).
    matches_played : int | None
        Total appearances.
    stats : dict[str, Any]
        Category-specific stat values.
    """

    player_name: str = ""
    position: str = ""
    nationality: str = ""
    age: int | None = None
    matches_played: int | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "position": self.position,
            "nationality": self.nationality,
            "age": self.age,
            "matches_played": self.matches_played,
            **self.stats,
        }


@dataclass
class SquadStats:
    """Aggregate squad stats for a team across a season.

    Attributes
    ----------
    team_id : str
        FBref squad ID (hex string).
    team_name : str
        Canonical team name.
    season : str
        Season identifier.
    competition : str
        Competition name.
    stat_tables : dict[StatCategory, FBrefTable]
        All scraped stat tables indexed by category.
    """

    team_id: str = ""
    team_name: str = ""
    season: str = ""
    competition: str = ""
    stat_tables: dict[StatCategory, FBrefTable] = field(default_factory=dict)

    def to_dataframe(self, category: StatCategory):
        """Convert a specific category's data to a DataFrame."""
        import pandas as pd

        table = self.stat_tables.get(category)
        if table is None or not table.rows:
            return None
        return pd.DataFrame(table.rows)


@dataclass
class MatchStats:
    """Statistics for a single match.

    Attributes
    ----------
    match_url : str
        FBref match URL.
    home_team : str
        Home team name.
    away_team : str
        Away team name.
    home_goals : int | None
        Home team goals.
    away_goals : int | None
        Away team goals.
    date : str
        Match date (YYYY-MM-DD).
    competition : str
        Competition name.
    stats : dict[str, Any]
        Match-level stat values (possession, shots, etc.).
    home_player_stats : list[PlayerStats]
        Per-player stats for the home team.
    away_player_stats : list[PlayerStats]
        Per-player stats for the away team.
    """

    match_url: str = ""
    home_team: str = ""
    away_team: str = ""
    home_goals: int | None = None
    away_goals: int | None = None
    date: str = ""
    competition: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    home_player_stats: list[PlayerStats] = field(default_factory=list)
    away_player_stats: list[PlayerStats] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_url": self.match_url,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "date": self.date,
            "competition": self.competition,
            **self.stats,
        }


# ── Known competition IDs ─────────────────────────────────

COMPETITION_IDS: dict[str, str] = {
    "Premier League": "9",
    "Championship": "10",
    "La Liga": "12",
    "Serie A": "11",
    "Bundesliga": "20",
    "Ligue 1": "13",
    "Eredivisie": "23",
    "Primeira Liga": "32",
    "Scottish Premiership": "40",
    "Champions League": "8",
    "Europa League": "19",
    "World Cup": "1",
    "European Championship": "15",
    "Copa America": "28",
    "Africa Cup of Nations": "57",
    "MLS": "22",
}

# Competition ID → display name
COMPETITION_NAMES: dict[str, str] = {v: k for k, v in COMPETITION_IDS.items()}
