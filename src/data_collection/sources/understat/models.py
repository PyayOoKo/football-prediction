"""
Data models for Understat scraped statistics.

Defines dataclasses for match-level xG, shot-level data, team xG,
and league identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── League identifiers ───────────────────────────────────

LEAGUE_NAMES: dict[str, str] = {
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
    "RFPL": "Russian Premier League",
}

# Reverse map: display name → Understat code
LEAGUE_CODES: dict[str, str] = {v: k for k, v in LEAGUE_NAMES.items()}


@dataclass
class ShotData:
    """A single shot event from Understat.

    Attributes
    ----------
    match_id : int
        Understat match identifier.
    shooter : str
        Player name who took the shot.
    team : str
        Team name (home or away).
    minute : int
        Minute of the shot.
    x : float
        X coordinate (0-1, normalised pitch width).
    y : float
        Y coordinate (0-1, normalised pitch height).
    xg : float
        Expected goals value for this shot.
    result : str
        Shot result (GOAL, MISSED, SAVED, BLOCKED, OWN_GOAL, etc.).
    situation : str
        Shot situation (open_play, direct_freekick, penalty, set_piece, etc.).
    last_action : str | None
        Last action before the shot (pass, dribble, etc.).
    season : str
        Season identifier (year).
    date : str
        Match date (YYYY-MM-DD).
    home_team : str
        Home team name.
    away_team : str
        Away team name.
    """

    match_id: int = 0
    shooter: str = ""
    team: str = ""
    minute: int = 0
    x: float = 0.0
    y: float = 0.0
    xg: float = 0.0
    result: str = ""
    situation: str = ""
    last_action: str | None = None
    season: str = ""
    date: str = ""
    home_team: str = ""
    away_team: str = ""

    @property
    def is_goal(self) -> bool:
        return self.result == "GOAL"

    @property
    def is_penalty(self) -> bool:
        return self.situation == "penalty"

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "shooter": self.shooter,
            "team": self.team,
            "minute": self.minute,
            "x": self.x,
            "y": self.y,
            "xg": self.xg,
            "result": self.result,
            "situation": self.situation,
            "last_action": self.last_action,
            "season": self.season,
            "date": self.date,
            "home_team": self.home_team,
            "away_team": self.away_team,
        }


@dataclass
class TeamXG:
    """Season-level xG data for a single team.

    Attributes
    ----------
    team_name : str
        Team name (Understat convention).
    season : str
        Season identifier (year).
    matches_played : int
        Matches played.
    xg : float
        Total expected goals scored.
    xga : float
        Total expected goals conceded.
    xg_per_match : float
        Average xG per match.
    xga_per_match : float
        Average xGA per match.
    scored : int
        Actual goals scored.
    conceded : int
        Actual goals conceded.
    wins : int
        Matches won.
    draws : int
        Matches drawn.
    losses : int
        Matches lost.
    pts : int
        Total points.
    npxg : float
        Non-penalty expected goals.
    npxga : float
        Non-penalty expected goals conceded.
    """

    team_name: str = ""
    season: str = ""
    matches_played: int = 0
    xg: float = 0.0
    xga: float = 0.0
    xg_per_match: float = 0.0
    xga_per_match: float = 0.0
    scored: int = 0
    conceded: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    pts: int = 0
    npxg: float = 0.0
    npxga: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_name": self.team_name,
            "season": self.season,
            "matches_played": self.matches_played,
            "xg": round(self.xg, 2),
            "xga": round(self.xga, 2),
            "xg_per_match": round(self.xg_per_match, 2),
            "xga_per_match": round(self.xga_per_match, 2),
            "scored": self.scored,
            "conceded": self.conceded,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "pts": self.pts,
            "npxg": round(self.npxg, 2),
            "npxga": round(self.npxga, 2),
        }


@dataclass
class MatchXG:
    """Match-level xG data for a single fixture.

    Attributes
    ----------
    match_id : int
        Understat match identifier.
    league : str
        League name.
    season : str
        Season identifier.
    date : str
        Match date (YYYY-MM-DD).
    home_team : str
        Home team name.
    away_team : str
        Away team name.
    home_xg : float
        Home team expected goals.
    away_xg : float
        Away team expected goals.
    home_goals : int
        Home team actual goals.
    away_goals : int
        Away team actual goals.
    home_shots : int
        Home team total shots.
    away_shots : int
        Away team total shots.
    home_shots_on_target : int
        Home team shots on target.
    away_shots_on_target : int
        Away team shots on target.
    is_result : bool
        Whether the match has a definitive result (not postponed).
    """

    match_id: int = 0
    league: str = ""
    season: str = ""
    date: str = ""
    home_team: str = ""
    away_team: str = ""
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_goals: int = 0
    away_goals: int = 0
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    is_result: bool = True

    @property
    def home_xg_diff(self) -> float:
        """xG difference (home - away)."""
        return round(self.home_xg - self.away_xg, 2)

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def total_xg(self) -> float:
        return round(self.home_xg + self.away_xg, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "league": self.league,
            "season": self.season,
            "date": self.date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_xg": self.home_xg,
            "away_xg": self.away_xg,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_shots": self.home_shots,
            "away_shots": self.away_shots,
            "home_shots_on_target": self.home_shots_on_target,
            "away_shots_on_target": self.away_shots_on_target,
            "home_xg_diff": self.home_xg_diff,
            "total_goals": self.total_goals,
            "total_xg": self.total_xg,
            "is_result": self.is_result,
        }


# ── Validation helpers ───────────────────────────────────


def validate_xg_shot(shot: ShotData) -> list[str]:
    """Validate a single ShotData record, returning a list of issues.

    Returns an empty list if the shot is valid.
    """
    issues: list[str] = []

    if shot.xg < 0 or shot.xg > 1.0:
        issues.append(f"xG out of range [0, 1.0]: {shot.xg}")
    if shot.x < 0 or shot.x > 1:
        issues.append(f"X coordinate out of range [0, 1]: {shot.x}")
    if shot.y < 0 or shot.y > 1:
        issues.append(f"Y coordinate out of range [0, 1]: {shot.y}")
    if shot.minute < 0 or shot.minute > 150:
        issues.append(f"Minute out of range [0, 150]: {shot.minute}")
    if shot.result not in ("GOAL", "MISSED", "SAVED", "BLOCKED", "OWN_GOAL", ""):
        issues.append(f"Invalid result: {shot.result}")
    if not shot.shooter:
        issues.append("Missing shooter name")
    if not shot.team:
        issues.append("Missing team name")

    return issues


def validate_match_xg(match: MatchXG) -> list[str]:
    """Validate a MatchXG record, returning a list of issues."""
    issues: list[str] = []

    if match.home_xg < 0 or match.home_xg > 20:
        issues.append(f"Home xG out of range [0, 20]: {match.home_xg}")
    if match.away_xg < 0 or match.away_xg > 20:
        issues.append(f"Away xG out of range [0, 20]: {match.away_xg}")
    if match.home_goals < 0 or match.home_goals > 30:
        issues.append(f"Home goals out of range: {match.home_goals}")
    if match.away_goals < 0 or match.away_goals > 30:
        issues.append(f"Away goals out of range: {match.away_goals}")
    if not match.home_team:
        issues.append("Missing home team")
    if not match.away_team:
        issues.append("Missing away team")
    if match.home_team == match.away_team:
        issues.append("Home and away teams are the same")

    return issues
