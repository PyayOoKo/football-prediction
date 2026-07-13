"""
Unit tests for Understat data models and validation.
"""

from __future__ import annotations

import pytest

from src.data_collection.sources.understat.models import (
    LEAGUE_NAMES,
    LEAGUE_CODES,
    MatchXG,
    ShotData,
    TeamXG,
    validate_match_xg,
    validate_xg_shot,
)


class TestLEAGUENAMES:
    def test_all_codes_present(self) -> None:
        assert LEAGUE_NAMES["EPL"] == "Premier League"
        assert LEAGUE_NAMES["La_liga"] == "La Liga"
        assert LEAGUE_NAMES["Bundesliga"] == "Bundesliga"
        assert LEAGUE_NAMES["Serie_A"] == "Serie A"
        assert LEAGUE_NAMES["Ligue_1"] == "Ligue 1"

    def test_reverse_map(self) -> None:
        assert LEAGUE_CODES["Premier League"] == "EPL"
        assert LEAGUE_CODES["La Liga"] == "La_liga"


class TestShotData:
    def test_minimal(self) -> None:
        shot = ShotData(
            match_id=12345,
            shooter="Erling Haaland",
            team="Manchester City",
            minute=35,
            x=0.5, y=0.3,
            xg=0.25,
            result="GOAL",
            situation="open_play",
        )
        assert shot.is_goal is True
        assert shot.is_penalty is False
        assert shot.to_dict()["match_id"] == 12345
        assert shot.to_dict()["xg"] == 0.25

    def test_penalty(self) -> None:
        shot = ShotData(
            match_id=1,
            shooter="Player",
            team="Team",
            minute=60, x=0.8, y=0.5,
            xg=0.79,
            result="GOAL",
            situation="penalty",
        )
        assert shot.is_penalty is True

    def test_missed_shot(self) -> None:
        shot = ShotData(
            match_id=2,
            shooter="Player",
            team="Team",
            minute=42, x=0.6, y=0.4,
            xg=0.12,
            result="MISSED",
            situation="open_play",
        )
        assert shot.is_goal is False


class TestTeamXG:
    def test_to_dict(self) -> None:
        team = TeamXG(
            team_name="Manchester City",
            season="2024",
            matches_played=38,
            xg=85.5,
            xga=32.1,
            scored=96,
            conceded=34,
            wins=28,
            draws=6,
            losses=4,
            pts=90,
        )
        d = team.to_dict()
        assert d["team_name"] == "Manchester City"
        assert d["xg"] == 85.5
        assert d["pts"] == 90

    def test_per_match_averages(self) -> None:
        team = TeamXG(matches_played=10, xg=20.0, xga=10.0)
        assert round(team.xg_per_match, 2) == 2.00
        assert round(team.xga_per_match, 2) == 1.00


class TestMatchXG:
    def test_properties(self) -> None:
        match = MatchXG(
            match_id=100,
            home_team="Arsenal",
            away_team="Chelsea",
            home_xg=2.5,
            away_xg=1.2,
            home_goals=3,
            away_goals=1,
        )
        assert match.home_xg_diff == 1.3
        assert match.total_goals == 4
        assert match.total_xg == 3.7

    def test_to_dict(self) -> None:
        match = MatchXG(
            match_id=100,
            league="Premier League",
            season="2024",
            date="2024-09-15",
            home_team="Arsenal",
            away_team="Chelsea",
            home_xg=2.5,
            away_xg=1.2,
            home_goals=3,
            away_goals=1,
        )
        d = match.to_dict()
        assert d["home_xg_diff"] == 1.3
        assert d["total_xg"] == 3.7


class TestValidateXGShot:
    def test_valid_shot(self) -> None:
        shot = ShotData(
            match_id=1, shooter="P", team="T",
            minute=30, x=0.5, y=0.5, xg=0.15,
            result="GOAL", situation="open_play",
        )
        assert validate_xg_shot(shot) == []

    def test_invalid_xg_range(self) -> None:
        shot = ShotData(
            match_id=1, shooter="P", team="T",
            minute=30, x=0.5, y=0.5, xg=2.5,
            result="GOAL", situation="open_play",
        )
        issues = validate_xg_shot(shot)
        assert len(issues) > 0
        assert any("xG" in i for i in issues)

    def test_invalid_coordinates(self) -> None:
        shot = ShotData(
            match_id=1, shooter="P", team="T",
            minute=30, x=1.5, y=-0.1, xg=0.5,
            result="GOAL", situation="open_play",
        )
        issues = validate_xg_shot(shot)
        assert len(issues) >= 2

    def test_invalid_minute(self) -> None:
        shot = ShotData(
            match_id=1, shooter="P", team="T",
            minute=200, x=0.5, y=0.5, xg=0.5,
            result="GOAL", situation="open_play",
        )
        issues = validate_xg_shot(shot)
        assert any("Minute" in i for i in issues)

    def test_invalid_result(self) -> None:
        shot = ShotData(
            match_id=1, shooter="P", team="T",
            minute=30, x=0.5, y=0.5, xg=0.5,
            result="INVALID", situation="open_play",
        )
        issues = validate_xg_shot(shot)
        assert any("Invalid result" in i for i in issues)

    def test_missing_shooter(self) -> None:
        shot = ShotData(
            match_id=1, shooter="", team="T",
            minute=30, x=0.5, y=0.5, xg=0.5,
            result="GOAL", situation="open_play",
        )
        issues = validate_xg_shot(shot)
        assert any("Missing shooter" in i for i in issues)


class TestValidateMatchXG:
    def test_valid_match(self) -> None:
        match = MatchXG(
            match_id=1,
            home_team="A", away_team="B",
            home_xg=1.5, away_xg=0.8,
            home_goals=2, away_goals=1,
        )
        assert validate_match_xg(match) == []

    def test_same_team(self) -> None:
        match = MatchXG(
            match_id=1,
            home_team="A", away_team="A",
            home_xg=1.0, away_xg=1.0,
            home_goals=0, away_goals=0,
        )
        issues = validate_match_xg(match)
        assert any("same" in i.lower() for i in issues)

    def test_xg_out_of_range(self) -> None:
        match = MatchXG(
            match_id=1,
            home_team="A", away_team="B",
            home_xg=-0.5, away_xg=25.0,
            home_goals=0, away_goals=0,
        )
        issues = validate_match_xg(match)
        assert len(issues) >= 2
