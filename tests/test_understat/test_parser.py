"""
Unit tests for UnderstatParser — JSON extraction and model conversion.
"""

from __future__ import annotations

import pytest

from src.data_collection.sources.understat.parser import UnderstatParser
from src.data_collection.sources.understat.models import TeamXG, MatchXG


# ── Sample Understat JSON data ────────────────────────────

_SAMPLE_TEAMS_DATA = {
    "42": {
        "id": 42,
        "title": "Arsenal",
        "history": [
            {
                "season": "2024",
                "played": 10,
                "xG": 20.5,
                "xGA": 8.3,
                "npxG": 18.2,
                "npxGA": 7.1,
                "scored": 22,
                "missed": 9,
                "wins": 7,
                "draws": 2,
                "loses": 1,
                "pts": 23,
            }
        ],
    },
    "55": {
        "id": 55,
        "title": "Chelsea",
        "history": [
            {
                "season": "2024",
                "played": 10,
                "xG": 15.0,
                "xGA": 12.0,
                "npxG": 14.0,
                "npxGA": 10.0,
                "scored": 18,
                "missed": 14,
                "wins": 5,
                "draws": 3,
                "loses": 2,
                "pts": 18,
            }
        ],
    },
}

_SAMPLE_TEAMS_DATA_NO_MATCH = {
    "42": {
        "id": 42,
        "title": "Arsenal",
        "history": [{"season": "2023", "played": 38, "xG": 70.0, "xGA": 30.0}],
    },
}

_SAMPLE_SHOTS_DATA = {
    "h_team": "Arsenal",
    "a_team": "Chelsea",
    "date": "2024-09-15",
    "h": [
        {
            "id": 1,
            "minute": 23,
            "player": "Bukayo Saka",
            "X": 0.85,
            "Y": 0.45,
            "xG": 0.15,
            "result": "GOAL",
            "situation": "open_play",
            "lastAction": "pass",
        },
        {
            "id": 2,
            "minute": 55,
            "player": "Martin Odegaard",
            "X": 0.72,
            "Y": 0.38,
            "xG": 0.08,
            "result": "SAVED",
            "situation": "open_play",
            "lastAction": "ball_recovery",
        },
    ],
    "a": [
        {
            "id": 3,
            "minute": 67,
            "player": "Cole Palmer",
            "X": 0.55,
            "Y": 0.30,
            "xG": 0.35,
            "result": "MISSED",
            "situation": "open_play",
            "lastAction": "through_ball",
        },
    ],
}

_SAMPLE_SHOTS_INVALID = {
    "h_team": "Team A",
    "a_team": "Team B",
    "h": [
        {
            "id": 99,
            "minute": 200,
            "player": "",
            "X": 2.0,
            "Y": -0.5,
            "xG": 5.0,
            "result": "INVALID",
        },
    ],
    "a": [],
}

_SAMPLE_DATES_DATA = {
    "42": [
        {
            "id": 12345,
            "isResult": True,
            "date": "2024-09-15",
            "h_team": {"id": "42", "title": "Arsenal"},
            "a_team": {"id": "55", "title": "Chelsea"},
            "goals": {"h": 3, "a": 1},
            "xG": {"h": 2.5, "a": 1.2},
            "h_shots": 15,
            "a_shots": 8,
            "h_sot": 7,
            "a_sot": 3,
        },
    ],
}


class TestUnderstatParser:
    def test_parse_league_teams(self) -> None:
        parser = UnderstatParser()
        teams = parser.parse_league_teams(_SAMPLE_TEAMS_DATA, "EPL", 2024)

        assert len(teams) == 2

        arsenal = teams[0]
        assert arsenal.team_name == "Arsenal"
        assert arsenal.matches_played == 10
        assert arsenal.xg == 20.5
        assert arsenal.xga == 8.3
        assert arsenal.scored == 22
        assert arsenal.conceded == 9
        assert arsenal.wins == 7
        assert arsenal.draws == 2
        assert arsenal.losses == 1
        assert arsenal.pts == 23
        assert round(arsenal.xg_per_match, 2) == 2.05
        assert round(arsenal.xga_per_match, 2) == 0.83

        chelsea = teams[1]
        assert chelsea.team_name == "Chelsea"
        assert chelsea.npxg == 14.0

    def test_parse_league_teams_no_match_season(self) -> None:
        """When the specified season is not found, use the most recent."""
        parser = UnderstatParser()
        teams = parser.parse_league_teams(_SAMPLE_TEAMS_DATA_NO_MATCH, "EPL", 2024)

        assert len(teams) == 1
        assert teams[0].xg == 70.0  # Uses 2023 as most recent

    def test_parse_match_shots(self) -> None:
        parser = UnderstatParser()
        shots = parser.parse_match_shots(_SAMPLE_SHOTS_DATA, 12345)

        assert len(shots) == 3

        # Home team first shot
        s0 = shots[0]
        assert s0.shooter == "Bukayo Saka"
        assert s0.team == "Arsenal"
        assert s0.minute == 23
        assert s0.xg == 0.15
        assert s0.x == 0.85
        assert s0.y == 0.45
        assert s0.result == "GOAL"
        assert s0.is_goal is True
        assert s0.situation == "open_play"
        assert s0.last_action == "pass"

        # Away team shot
        s2 = shots[2]
        assert s2.shooter == "Cole Palmer"
        assert s2.team == "Chelsea"
        assert s2.is_goal is False  # MISSED

    def test_parse_match_shots_validation_rejects_invalid(self) -> None:
        """Invalid shots are skipped when strict=False."""
        parser = UnderstatParser(validate=True, strict=False)
        shots = parser.parse_match_shots(_SAMPLE_SHOTS_INVALID, 99)

        assert len(shots) == 0  # Invalid shot was skipped

    def test_parse_league_matches(self) -> None:
        parser = UnderstatParser()
        matches = parser.parse_league_matches(
            _SAMPLE_DATES_DATA, _SAMPLE_TEAMS_DATA, "EPL", 2024,
        )

        assert len(matches) == 1
        match = matches[0]
        assert match.match_id == 12345
        assert match.home_team == "Arsenal"
        assert match.away_team == "Chelsea"
        assert match.home_xg == 2.5
        assert match.away_xg == 1.2
        assert match.home_goals == 3
        assert match.away_goals == 1
        assert match.home_shots == 15
        assert match.away_shots == 8
        assert match.home_shots_on_target == 7
        assert match.away_shots_on_target == 3
        assert match.is_result is True
