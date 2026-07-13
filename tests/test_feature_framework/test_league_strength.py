"""
Tests for the League Strength module — LeagueStrengthEngine.

Covers:
- Core metric computation
- Cross-league normalization
- Promoted/relegated team tracking
- European competition adjustment
- History storage and retrieval
- Persistence (JSON save/load)
- Error handling and edge cases
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.feature_framework.league_strength import (
    LeagueStrengthEngine,
    LeagueStrengthRecord,
    create_league_strength_engine,
)


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """12 matches across 2 seasons and 2 leagues.

    Season 2024, League E0 (PL-style, lower scoring):
    - 6 matches, avg ~2.5 goals/match, ~45% home wins

    Season 2025, League E0:
    - 6 more matches

    Season 2024, League S1 (higher scoring):
    - 6 matches, avg ~3.5 goals/match
    """
    return pd.DataFrame({
        "season": ["2024"] * 6 + ["2024"] * 6 + ["2025"] * 6,
        "league": ["E0"] * 6 + ["S1"] * 6 + ["E0"] * 6,
        "date": pd.to_datetime(
            ["2024-01-07", "2024-01-14", "2024-01-21", "2024-01-28",
             "2024-02-04", "2024-02-11",
             "2024-01-07", "2024-01-14", "2024-01-21", "2024-01-28",
             "2024-02-04", "2024-02-11",
             "2025-01-05", "2025-01-12", "2025-01-19", "2025-01-26",
             "2025-02-02", "2025-02-09"] * 1
        ),
        "home_team": [
            "A", "C", "E", "G", "I", "K",
            "M", "O", "Q", "S", "U", "W",
            "A", "C", "E", "G", "I", "K",
        ],
        "away_team": [
            "B", "D", "F", "H", "J", "L",
            "N", "P", "R", "T", "V", "X",
            "B", "D", "F", "H", "J", "L",
        ],
        "home_goals": [2, 1, 0, 3, 1, 2, 3, 2, 4, 1, 2, 0, 1, 2, 0, 2, 3, 1],
        "away_goals": [0, 1, 2, 0, 1, 1, 1, 0, 2, 1, 0, 3, 0, 0, 1, 0, 0, 0],
        "result":   ["H", "D", "A", "H", "D", "H",
                     "H", "H", "H", "D", "H", "A",
                     "H", "H", "A", "H", "H", "H"],
    })


@pytest.fixture
def sample_with_xg(sample_matches: pd.DataFrame) -> pd.DataFrame:
    df = sample_matches.copy()
    df["home_xg"] = [1.8, 0.9, 0.4, 2.1, 0.8, 1.5,
                     2.2, 1.6, 2.8, 0.7, 1.3, 0.3,
                     1.2, 1.4, 0.5, 1.9, 2.0, 0.9]
    df["away_xg"] = [0.4, 0.8, 1.6, 0.3, 0.9, 0.7,
                     0.6, 0.3, 1.4, 0.8, 0.2, 2.1,
                     0.3, 0.2, 0.9, 0.1, 0.1, 0.2]
    return df


@pytest.fixture
def team_seasons() -> pd.DataFrame:
    """Team-season-league mapping for promotion/relegation detection."""
    return pd.DataFrame({
        "team": [
            "A", "B", "C", "D",  # 2024 E0
            "M", "N",            # 2024 E0 (different teams, same as S1 perspective)
            "A", "B", "C", "D", "X", "Y",  # 2025 E0 (X, Y promoted)
        ],
        "season": ["2024"] * 6 + ["2025"] * 6,
        "league": ["E0"] * 6 + ["E0"] * 6,
    })


# ═══════════════════════════════════════════════════════════════
#  Tests: Core Metric Computation
# ═══════════════════════════════════════════════════════════════


class TestLeagueCoreMetrics:
    def test_basic_computation(self, sample_matches):
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        assert len(results) == 3  # 2 seasons x E0 + 1 season x S1
        assert "2024/E0" in results
        assert "2024/S1" in results
        assert "2025/E0" in results

    def test_offensive_strength(self, sample_matches):
        """League E0 2024: 6 matches, home_goals = [2,1,0,3,1,2] → avg = 1.5."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        # Offensive strength = mean of all goals scored (home + away perspectives)
        # home_goals: [2,1,0,3,1,2] = total 9
        # away_goals: [0,1,2,0,1,1] = total 5
        # all scored = [9, 5] combined / 12 = 14/12 = 1.167
        assert rec.offensive_strength == pytest.approx(14.0 / 12.0, abs=1e-6)

    def test_defensive_strength(self, sample_matches):
        """Defensive = mean of all goals conceded."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        # conceded = away_goals for home teams, home_goals for away teams
        # = same pool: [0,1,2,0,1,1,2,1,0,3,1,2] / 12 = 14/12
        assert rec.defensive_strength == pytest.approx(14.0 / 12.0, abs=1e-6)

    def test_home_advantage(self, sample_matches):
        """Home advantage = avg home goals - avg away goals."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        # home_goals avg = 9/6 = 1.5
        # away_goals avg = 5/6 = 0.833
        expected_ha = 1.5 - (5.0 / 6.0)
        assert rec.home_adv == pytest.approx(expected_ha, abs=1e-6)

    def test_home_win_rate(self, sample_matches):
        """6 matches in E0 2024: results H,D,A,H,D,H → 3 home wins."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        assert rec.home_win_rate == pytest.approx(3.0 / 6.0, abs=1e-6)
        assert rec.draw_rate == pytest.approx(2.0 / 6.0, abs=1e-6)
        assert rec.away_win_rate == pytest.approx(1.0 / 6.0, abs=1e-6)

    def test_btts_rate(self, sample_matches):
        """E0 2024: results [2-0,1-1,0-2,3-0,1-1,2-1] → BTTS in 3."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        # BTTS in matches: 0-0? no. 1-1? yes. 0-2? no. 3-0? no. 1-1? yes. 2-1? yes.
        # That's 3 out of 6
        assert rec.btts_rate == pytest.approx(3.0 / 6.0, abs=1e-6)

    def test_over_2_5_rate(self, sample_matches):
        """E0 2024: total goals per match [2,2,2,3,2,3] → over 2.5 in 2."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        assert rec.over_2_5_rate == pytest.approx(2.0 / 6.0, abs=1e-6)

    def test_min_matches_filter(self, sample_matches):
        """With min_matches=10, 6-match leagues should be excluded."""
        engine = LeagueStrengthEngine(min_matches=10)
        results = engine.compute(sample_matches)
        assert len(results) == 0

    def test_competitive_balance(self, sample_matches):
        """E0 2024: goal diffs [2,0,-2,3,0,1]. Std = approx 1.47."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        goal_diffs = [2, 0, -2, 3, 0, 1]
        expected_std = float(np.std(goal_diffs, ddof=1))  # sample std (matches pandas default)
        assert rec.competitive_balance == pytest.approx(expected_std, abs=1e-6)

    def test_xg_computation(self, sample_with_xg):
        """With xG data, avg_xg should be computed."""
        engine = LeagueStrengthEngine(min_matches=1)
        results = engine.compute(sample_with_xg)

        rec = results["2024/E0"]
        assert rec.avg_xg is not None
        assert rec.avg_xg > 0

    def test_missing_required_columns(self, sample_matches):
        engine = LeagueStrengthEngine(min_matches=1)
        df = sample_matches.drop(columns=["result"])
        with pytest.raises(ValueError, match="Missing required columns"):
            engine.compute(df)


# ═══════════════════════════════════════════════════════════════
#  Tests: Cross-league Normalisation
# ═══════════════════════════════════════════════════════════════


class TestLeagueNormalisation:
    def test_reference_league_gets_factor_one(self, sample_matches):
        """Reference league (E0) should have attack/defence factor = 1.0."""
        engine = LeagueStrengthEngine(
            reference_league="E0", min_matches=1, auto_normalise=True,
        )
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        assert rec.attack_factor == pytest.approx(1.0, abs=1e-6)
        assert rec.defence_factor == pytest.approx(1.0, abs=1e-6)

    def test_stronger_league_higher_attack(self, sample_matches):
        """S1 scores more goals than E0 → higher attack factor."""
        engine = LeagueStrengthEngine(
            reference_league="E0", min_matches=1, auto_normalise=True,
        )
        results = engine.compute(sample_matches)

        rec_s1 = results["2024/S1"]
        rec_e0 = results["2024/E0"]
        # S1: home_goals=[3,2,4,1,2,0], avg=12/6=2.0 home, 7/6=1.167 away
        # S1 offensive = (12+7)/12 = 19/12 = 1.583
        # E0 offensive = 14/12 = 1.167
        # S1 attack_factor = 1.583/1.167 ≈ 1.357
        assert rec_s1.attack_factor > rec_e0.attack_factor

    def test_auto_normalise_disabled(self, sample_matches):
        """With auto_normalise=False, factors should remain at 1.0."""
        engine = LeagueStrengthEngine(
            min_matches=1, auto_normalise=False,
        )
        results = engine.compute(sample_matches)

        for rec in results.values():
            assert rec.attack_factor == 1.0
            assert rec.defence_factor == 1.0

    def test_normalise_across_leagues(self, sample_matches):
        engine = LeagueStrengthEngine(
            reference_league="E0", min_matches=1, auto_normalise=True,
        )
        engine.compute(sample_matches)
        df = engine.normalise_across_leagues(
            seasons=["2024"], leagues=["E0", "S1"],
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # E0 should have attack_factor=1.0
        e0_row = df[df["league"] == "E0"].iloc[0]
        assert e0_row["attack_factor"] == pytest.approx(1.0, abs=1e-6)

    def test_normalise_no_data(self):
        engine = LeagueStrengthEngine()
        df = engine.normalise_across_leagues()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ═══════════════════════════════════════════════════════════════
#  Tests: Promoted / Relegated Teams
# ═══════════════════════════════════════════════════════════════


class TestLeaguePromotionRelegation:
    def test_set_promoted(self):
        engine = LeagueStrengthEngine()
        engine.set_promoted("2025", "E0", {"Team_X", "Team_Y"})
        promoted = engine.get_promoted("2025", "E0")
        assert promoted == {"Team_X", "Team_Y"}

    def test_set_relegated(self):
        engine = LeagueStrengthEngine()
        engine.set_relegated("2024", "E0", {"Team_A", "Team_B"})
        relegated = engine.get_relegated("2024", "E0")
        assert relegated == {"Team_A", "Team_B"}

    def test_auto_detect_promoted_relegated(self, team_seasons):
        engine = LeagueStrengthEngine()
        engine.auto_detect_promoted_relegated(team_seasons)

        # Teams X and Y are new in E0 2025 → promoted
        promoted = engine.get_promoted("2025", "E0")
        assert "X" in promoted
        assert "Y" in promoted

        # Teams M and N are in E0 2024 but not 2025 → relegated
        relegated = engine.get_relegated("2024", "E0")
        assert "M" in relegated
        assert "N" in relegated

    def test_relegation_affects_computation(self, sample_matches, team_seasons):
        engine = LeagueStrengthEngine(min_matches=1)
        # Set relegation info
        engine.set_relegated("2024", "E0", {"A", "B"})
        results = engine.compute(sample_matches)

        rec = results["2024/E0"]
        assert rec.n_relegated_teams == 2

    def test_auto_detect_missing_columns(self):
        engine = LeagueStrengthEngine()
        df = pd.DataFrame({"team": ["A"], "season": ["2024"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            engine.auto_detect_promoted_relegated(df)


# ═══════════════════════════════════════════════════════════════
#  Tests: European Competition Tracking
# ═══════════════════════════════════════════════════════════════


class TestLeagueEuropean:
    def test_set_european(self):
        engine = LeagueStrengthEngine()
        engine.set_european("2024", "E0", {"Team_A", "Team_C"})
        euro = engine.get_european("2024", "E0")
        assert euro == {"Team_A", "Team_C"}

    def test_european_adjustment(self, sample_matches):
        """Teams A and C in European comps → their matches flagged."""
        engine = LeagueStrengthEngine(min_matches=1)
        engine.set_european("2024", "E0", {"A", "C"})
        engine.compute(sample_matches)

        result = engine.european_adjustment(sample_matches)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Should have both "with_european" and "without_european" rows
        types = set(result["type"].unique())
        assert "with_european" in types
        assert "without_european" in types

    def test_european_adjustment_no_data(self):
        engine = LeagueStrengthEngine()
        df = engine.european_adjustment(pd.DataFrame())
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ═══════════════════════════════════════════════════════════════
#  Tests: History Storage & Retrieval
# ═══════════════════════════════════════════════════════════════


class TestLeagueHistory:
    def test_store_and_retrieve(self):
        engine = LeagueStrengthEngine()
        rec = LeagueStrengthRecord(season="2024", league="E0", total_matches=100)
        engine.store_season("2024", "E0", rec)
        retrieved = engine.get_season("2024", "E0")
        assert retrieved is not None
        assert retrieved.season == "2024"
        assert retrieved.total_matches == 100

    def test_get_missing_season(self):
        engine = LeagueStrengthEngine()
        retrieved = engine.get_season("1900", "XX")
        assert retrieved is None

    def test_compute_stores_history(self, sample_matches):
        engine = LeagueStrengthEngine(min_matches=1, store_history=True)
        engine.compute(sample_matches)
        assert engine.get_season("2024", "E0") is not None

    def test_clear_history(self, sample_matches):
        engine = LeagueStrengthEngine(min_matches=1)
        engine.compute(sample_matches)
        assert len(engine._history) > 0
        engine.clear_history()
        assert len(engine._history) == 0

    def test_get_history_dataframe(self, sample_matches):
        engine = LeagueStrengthEngine(min_matches=1)
        engine.compute(sample_matches)
        df = engine.get_history_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 season-league combos

    def test_get_history_dataframe_empty(self):
        engine = LeagueStrengthEngine()
        df = engine.get_history_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ═══════════════════════════════════════════════════════════════
#  Tests: Persistence (JSON save/load)
# ═══════════════════════════════════════════════════════════════


class TestLeaguePersistence:
    def test_save_and_load_json(self, sample_matches):
        engine = LeagueStrengthEngine(min_matches=1)
        engine.compute(sample_matches)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            tmp_path = f.name

        try:
            engine.save_json(tmp_path)
            # Verify file exists and is valid JSON
            with open(tmp_path) as f:
                data = json.load(f)
            assert "metadata" in data
            assert "records" in data
            assert len(data["records"]) > 0

            # Load into a new engine
            engine2 = LeagueStrengthEngine()
            count = engine2.load_json(tmp_path)
            assert count == len(engine._history)

            # Verify records match
            original = engine.get_season("2024", "E0")
            loaded = engine2.get_season("2024", "E0")
            assert loaded is not None
            assert loaded.offensive_strength == pytest.approx(
                original.offensive_strength, abs=1e-6
            )
            assert loaded.avg_goals == pytest.approx(original.avg_goals, abs=1e-6)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Tests: LeagueStrengthRecord
# ═══════════════════════════════════════════════════════════════


class TestLeagueRecord:
    def test_to_dict(self):
        rec = LeagueStrengthRecord(season="2024", league="E0", total_matches=100)
        d = rec.to_dict()
        assert d["season"] == "2024"
        assert d["league"] == "E0"
        assert d["total_matches"] == 100

    def test_from_dict(self):
        data = {
            "season": "2024", "league": "E0", "total_matches": 100,
            "offensive_strength": 1.5, "defensive_strength": 0.8,
            "avg_goals": 2.3, "home_adv": 0.7, "competitive_balance": 1.2,
            "home_win_rate": 0.4, "draw_rate": 0.3, "away_win_rate": 0.3,
            "btts_rate": 0.5, "over_2_5_rate": 0.5,
            "attack_factor": 1.0, "defence_factor": 1.0,
            "avg_home_goals": 1.5, "avg_away_goals": 0.8,
            "std_goal_diff": 1.2,
        }
        rec = LeagueStrengthRecord.from_dict(data)
        assert rec.season == "2024"
        assert rec.offensive_strength == 1.5

    def test_from_dict_filters_unknown_keys(self):
        data = {"season": "2024", "league": "E0", "unknown_key": 42}
        rec = LeagueStrengthRecord.from_dict(data)
        assert rec.season == "2024"
        assert not hasattr(rec, "unknown_key")

    def test_goal_diff_std_alias(self):
        rec = LeagueStrengthRecord(
            season="2024", league="E0", competitive_balance=1.5,
        )
        assert rec.goal_diff_std == 1.5

    def test_repr(self):
        rec = LeagueStrengthRecord(season="2024", league="E0")
        assert "LeagueStrengthRecord" in repr(rec)


# ═══════════════════════════════════════════════════════════════
#  Tests: Summary
# ═══════════════════════════════════════════════════════════════


class TestLeagueSummary:
    def test_summary_with_data(self, sample_matches):
        engine = LeagueStrengthEngine(min_matches=1)
        engine.compute(sample_matches)
        summary = engine.summary()
        assert "LEAGUE STRENGTH REPORT" in summary
        assert "2024" in summary
        assert "E0" in summary

    def test_summary_empty(self):
        engine = LeagueStrengthEngine()
        summary = engine.summary()
        assert "No data" in summary or "No league strength" in summary


# ═══════════════════════════════════════════════════════════════
#  Tests: Factory Function
# ═══════════════════════════════════════════════════════════════


class TestLeagueFactory:
    def test_create_engine(self):
        engine = create_league_strength_engine(
            reference_league="S1", min_matches=5,
        )
        assert isinstance(engine, LeagueStrengthEngine)
        assert engine.reference_league == "S1"
        assert engine.min_matches == 5

    def test_default_params(self):
        engine = create_league_strength_engine()
        assert engine.reference_league == "E0"
        assert engine.min_matches == 10
        assert engine.auto_normalise is True
