"""
Unit tests for FBref data models.
"""

from __future__ import annotations

import pytest

from src.data_collection.sources.fbref.models import (
    COMPETITION_IDS,
    COMPETITION_NAMES,
    CATEGORY_URL_MAP,
    FBrefTable,
    MatchStats,
    PlayerStats,
    SquadStats,
    StatCategory,
)


class TestStatCategory:
    def test_values(self) -> None:
        assert StatCategory.STANDARD.value == "stats_standard"
        assert StatCategory.SHOOTING.value == "stats_shooting"
        assert StatCategory.PASSING.value == "stats_passing"
        assert StatCategory.KEEPING.value == "stats_keeper"
        assert StatCategory.DEFENSE.value == "stats_defense"
        assert StatCategory.POSSESSION.value == "stats_possession"

    def test_category_url_map_has_all(self) -> None:
        """Every StatCategory except MATCH_STATS has a URL path."""
        for cat in StatCategory:
            if cat == StatCategory.MATCH_STATS:
                continue
            assert cat in CATEGORY_URL_MAP, f"Missing URL for {cat}"


class TestFBrefTable:
    def test_minimal(self) -> None:
        table = FBrefTable(
            category=StatCategory.STANDARD,
            columns=["player_name", "goals"],
            rows=[{"player_name": "Player A", "goals": 10}],
        )
        assert table.category == StatCategory.STANDARD
        assert len(table.rows) == 1
        assert table.rows[0]["player_name"] == "Player A"


class TestSquadStats:
    def test_empty(self) -> None:
        squad = SquadStats(team_id="abc123", team_name="Test FC", season="2024-2025")
        assert squad.team_id == "abc123"
        assert squad.stat_tables == {}
        assert squad.to_dataframe(StatCategory.STANDARD) is None

    def test_to_dataframe(self) -> None:
        import pandas as pd

        table = FBrefTable(
            category=StatCategory.STANDARD,
            columns=["player_name", "goals"],
            rows=[{"player_name": "P1", "goals": 5}, {"player_name": "P2", "goals": 3}],
        )
        squad = SquadStats(team_id="x", team_name="X", season="s")
        squad.stat_tables[StatCategory.STANDARD] = table

        df = squad.to_dataframe(StatCategory.STANDARD)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["player_name", "goals"]


class TestMatchStats:
    def test_empty(self) -> None:
        match = MatchStats()
        assert match.home_team == ""
        assert match.home_goals is None

    def test_to_dict(self) -> None:
        match = MatchStats(
            match_url="https://fbref.com/en/matches/abc",
            home_team="Arsenal",
            away_team="Chelsea",
            home_goals=2,
            away_goals=1,
            date="2024-01-07",
            competition="Premier League",
            stats={"possession": 55, "shots": 15},
        )
        d = match.to_dict()
        assert d["home_team"] == "Arsenal"
        assert d["possession"] == 55


class TestPlayerStats:
    def test_to_dict(self) -> None:
        player = PlayerStats(
            player_name="Erling Haaland",
            position="FW",
            matches_played=28,
            stats={"goals": 27, "assists": 5, "shots": 89},
        )
        d = player.to_dict()
        assert d["player_name"] == "Erling Haaland"
        assert d["goals"] == 27
        assert d["assists"] == 5


class TestCompetitionIds:
    def test_major_leagues(self) -> None:
        assert COMPETITION_IDS["Premier League"] == "9"
        assert COMPETITION_IDS["La Liga"] == "12"
        assert COMPETITION_IDS["Bundesliga"] == "20"
        assert COMPETITION_IDS["Serie A"] == "11"
        assert COMPETITION_IDS["Ligue 1"] == "13"

    def test_round_trip(self) -> None:
        for name, cid in COMPETITION_IDS.items():
            assert COMPETITION_NAMES[cid] == name
