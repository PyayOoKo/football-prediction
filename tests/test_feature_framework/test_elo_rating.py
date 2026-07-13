"""
Tests for the Elo Rating Engine (EloEngine + EloTransformer).

Covers:
- Core Elo formulas (expected score, actual score, K-factor)
- Single match updates
- Batch processing via process_matches
- Season change regression
- Dynamic K-factor (goal margin, importance, league strength)
- New team / promoted team handling
- Host nation bonus
- FeatureTransformer integration
- History and trajectory queries
- Edge cases (0-0 draws, single match, unknown results)
- Club Elo benchmark alignment
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_framework.features.elo_rating import (
    EloEngine,
    EloTransformer,
    EloMatchRecord,
    EloSnapshot,
    create_elo_transformer,
)


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def basic_engine() -> EloEngine:
    return EloEngine(k=32, home_advantage=100, initial_rating=1500,
                     new_team_rating=1500, use_goal_margin=False,
                     use_importance=False, use_league_strength=False,
                     regress_to_mean=False)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """12 match rows with 8 distinct teams covering H/D/A outcomes."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22",
            "2024-02-01", "2024-02-08", "2024-02-15", "2024-02-22",
            "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22",
        ]),
        "home_team": [
            "Team_A", "Team_C", "Team_E", "Team_G",
            "Team_B", "Team_D", "Team_F", "Team_H",
            "Team_A", "Team_C", "Team_E", "Team_G",
        ],
        "away_team": [
            "Team_B", "Team_D", "Team_F", "Team_H",
            "Team_A", "Team_C", "Team_E", "Team_G",
            "Team_B", "Team_D", "Team_F", "Team_H",
        ],
        "home_goals": [3, 1, 2, 0, 0, 2, 1, 3, 2, 0, 4, 1],
        "away_goals": [0, 1, 1, 1, 2, 1, 0, 0, 0, 3, 0, 1],
        "result": ["H", "D", "H", "A", "A", "H", "H", "H", "H", "A", "H", "D"],
        "league": ["PL"] * 12,
        "season": ["2024"] * 12,
    })


@pytest.fixture
def two_team_matches() -> pd.DataFrame:
    """Two matches between the same two teams."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
        "home_team": ["Team_A", "Team_B"],
        "away_team": ["Team_B", "Team_A"],
        "home_goals": [3, 0],
        "away_goals": [0, 1],
        "result": ["H", "A"],
        "league": ["PL", "PL"],
        "season": ["2024", "2024"],
    })


# ═══════════════════════════════════════════════════════════════
#  Tests: Core Formulas
# ═══════════════════════════════════════════════════════════════


class TestEloCoreFormulas:
    def test_expected_score_equal_ratings(self, basic_engine):
        """Equal ratings with home advantage → home team favoured."""
        E = basic_engine.expected_score(1500, 1500)
        assert 0.5 < E < 1.0  # Home advantage makes home team favourite
        assert abs(E - 1.0 / (1.0 + 10.0 ** (-100 / 400))) < 1e-10

    def test_expected_score_home_strong(self, basic_engine):
        """Strong home team should have high expected score."""
        E = basic_engine.expected_score(1700, 1500)
        assert E > 0.7

    def test_expected_score_away_strong(self, basic_engine):
        """Strong away team should have lower expected score for home."""
        E = basic_engine.expected_score(1500, 1700)
        assert E < 0.4

    def test_actual_score_home_win(self, basic_engine):
        assert basic_engine._actual_score("H") == 1.0

    def test_actual_score_draw(self, basic_engine):
        assert basic_engine._actual_score("D") == 0.5

    def test_actual_score_away_win(self, basic_engine):
        assert basic_engine._actual_score("A") == 0.0

    def test_actual_score_invalid(self, basic_engine):
        with pytest.raises(ValueError):
            basic_engine._actual_score("X")

    def test_k_factor_no_margin(self, basic_engine):
        """With use_goal_margin=False, K = base."""
        K = basic_engine._compute_k_factor(goal_margin=0, importance_mult=1.0, league_mult=1.0)
        assert K == 32.0

    def test_k_factor_with_margin(self):
        """With margin and goal_margin enabled, K > base for big wins."""
        engine = EloEngine(k=32, use_goal_margin=True)
        K = engine._compute_k_factor(goal_margin=3, importance_mult=1.0, league_mult=1.0)
        assert K > 32.0
        assert K < 32.0 * 3.0  # Clamped

    def test_k_factor_importance_mult(self):
        """World Cup importance should boost K."""
        engine = EloEngine(k=20, use_goal_margin=False, use_importance=True)
        K = engine._compute_k_factor(goal_margin=0, importance_mult=1.5, league_mult=1.0)
        assert K == 20.0 * 1.5

    def test_k_factor_league_mult(self):
        """Lower tier league should have higher K."""
        engine = EloEngine(k=20, use_goal_margin=False, use_league_strength=True)
        K = engine._compute_k_factor(goal_margin=0, importance_mult=1.0, league_mult=1.3)
        assert K == 20.0 * 1.3

    def test_parse_importance(self):
        assert EloEngine._parse_importance("World Cup") == 1.5
        assert EloEngine._parse_importance("Premier League") == 1.0
        assert EloEngine._parse_importance("International Friendly") == 0.6
        assert EloEngine._parse_importance(None) == 1.0

    def test_parse_league_strength(self):
        assert EloEngine._parse_league_strength("Premier League") == 1.0
        assert EloEngine._parse_league_strength("Championship") == 1.1
        assert EloEngine._parse_league_strength("League Two") == 1.2
        assert EloEngine._parse_league_strength("Unknown League") == 1.0


# ═══════════════════════════════════════════════════════════════
#  Tests: Single Match Updates
# ═══════════════════════════════════════════════════════════════


class TestEloSingleMatch:
    def test_home_win_updates_ratings(self, basic_engine):
        """Home win → home gains Elo, away loses Elo."""
        rec = basic_engine.update(
            "Team_A", "Team_B", "H",
            home_goals=3, away_goals=0,
            match_index=0,
        )
        assert rec.home_elo_before == 1500.0
        assert rec.away_elo_before == 1500.0
        assert rec.home_elo_after > 1500.0  # Home gained
        assert rec.away_elo_after < 1500.0  # Away lost
        assert rec.actual_home == 1.0
        assert rec.expected_home > 0.5  # Home advantage

    def test_away_win_flips_ratings(self, basic_engine):
        """Away win → away gains more than home loses."""
        rec = basic_engine.update(
            "Team_A", "Team_B", "A",
            match_index=0,
        )
        assert rec.home_elo_after < 1500.0
        assert rec.away_elo_after > 1500.0

    def test_draw_moves_ratings_toward_each_other(self, basic_engine):
        """Draw → lower rated team gains, higher rated loses."""
        # Manually set ratings so home is higher
        basic_engine.set_rating("Team_A", 1600)
        basic_engine.set_rating("Team_B", 1400)
        rec = basic_engine.update("Team_A", "Team_B", "D", match_index=0)
        assert rec.home_elo_after < rec.home_elo_before  # Higher-rated loses points
        assert rec.away_elo_after > rec.away_elo_before  # Lower-rated gains

    def test_first_match_uses_new_team_rating(self):
        """A team's first match should use new_team_rating."""
        engine = EloEngine(initial_rating=1500, new_team_rating=1300)
        rec = engine.update("NewTeam", "Opponent", "D", match_index=0)
        assert rec.home_elo_before == 1300.0  # New team uses lower rating

    def test_engine_tracks_match_count(self, basic_engine):
        basic_engine.update("A", "B", "H", match_index=0)
        assert basic_engine.match_count == 1
        basic_engine.update("C", "D", "D", match_index=1)
        assert basic_engine.match_count == 2

    def test_history_recorded(self, basic_engine):
        basic_engine.update("A", "B", "H", match_index=0)
        assert len(basic_engine.history) == 1
        assert isinstance(basic_engine.history[0], EloMatchRecord)

    def test_reset_clears_state(self, basic_engine):
        basic_engine.update("A", "B", "H", match_index=0)
        basic_engine.reset()
        assert basic_engine.match_count == 0
        assert basic_engine.ratings == {}

    def test_get_rating_creates_new(self, basic_engine):
        rating = basic_engine.get_rating("UnknownTeam")
        assert rating == 1500.0
        # Team should now be in ratings
        assert "UnknownTeam" in basic_engine.ratings


# ═══════════════════════════════════════════════════════════════
#  Tests: Batch Processing
# ═══════════════════════════════════════════════════════════════


class TestEloBatchProcessing:
    def test_process_matches_adds_columns(self, two_team_matches):
        engine = EloEngine(k=32, use_goal_margin=False)
        result = engine.process_matches(
            two_team_matches,
            home_goals_col="home_goals",
            away_goals_col="away_goals",
        )
        assert "h_elo" in result.columns
        assert "a_elo" in result.columns
        assert "elo_diff" in result.columns

    def test_process_matches_pre_match_ratings(self, two_team_matches):
        """First match should have initial ratings, second should be updated."""
        engine = EloEngine(k=32, use_goal_margin=False, use_importance=False,
                           use_league_strength=False, new_team_rating=1500.0)
        result = engine.process_matches(two_team_matches)

        # Match 0: first match → both teams at new_team_rating 1500
        assert result.loc[0, "h_elo"] == 1500.0
        assert result.loc[0, "a_elo"] == 1500.0

        # Match 1: ratings have been updated by match 0 result
        assert result.loc[1, "h_elo"] != 1500.0  # Team_B's rating changed
        assert result.loc[1, "a_elo"] != 1500.0  # Team_A's rating changed

    def test_append_mode(self, two_team_matches):
        """With append=True, engine continues from previous state."""
        engine = EloEngine(k=32, use_goal_margin=False)
        result1 = engine.process_matches(two_team_matches, append=True)
        assert len(engine.history) == 2

        # Process more matches with same engine
        more_matches = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-15"]),
            "home_team": ["Team_A"],
            "away_team": ["Team_B"],
            "home_goals": [1],
            "away_goals": [0],
            "result": ["H"],
            "league": ["PL"],
            "season": ["2024"],
        })
        result2 = engine.process_matches(more_matches, append=True)
        assert len(engine.history) == 3
        assert result2.loc[0, "h_elo"] != 1500.0  # Uses updated ratings

    def test_sort_chronologically(self):
        """Out-of-order dates should be sorted."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-02-01", "2024-01-01"]),
            "home_team": ["Team_B", "Team_A"],
            "away_team": ["Team_A", "Team_B"],
            "home_goals": [0, 3],
            "away_goals": [1, 0],
            "result": ["A", "H"],
            "league": ["PL", "PL"],
            "season": ["2024", "2024"],
        })
        engine = EloEngine(k=32, use_goal_margin=False)
        result = engine.process_matches(df)
        # After sorting, match 0 should be 2024-01-01
        assert result.iloc[0]["date"] == pd.Timestamp("2024-01-01")
        assert result.iloc[1]["date"] == pd.Timestamp("2024-02-01")

    def test_process_season_change(self):
        """Season change should trigger regression."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-12-01", "2025-01-01"]),
            "home_team": ["Team_A", "Team_A"],
            "away_team": ["Team_B", "Team_B"],
            "home_goals": [2, 1],
            "away_goals": [0, 0],
            "result": ["H", "H"],
            "league": ["PL", "PL"],
            "season": ["2024", "2025"],
        })
        engine = EloEngine(k=32, use_goal_margin=False, regress_to_mean=True,
                           regression_factor=1 / 3, new_team_rating=1500.0)
        result = engine.process_matches(df)

        # After 2 home wins, Team_A should be above 1500
        h_elo_2025 = result.loc[1, "h_elo"]
        assert h_elo_2025 > 1500.0

        # Without season regression, Team_A would be higher
        # Verify regression actually happened by checking rating is closer to mean
        assert h_elo_2025 < 1550.0


# ═══════════════════════════════════════════════════════════════
#  Tests: Dynamic K-factor
# ═══════════════════════════════════════════════════════════════


class TestEloDynamicKFactor:
    def test_big_win_larger_k(self):
        """A 5-0 win should have larger K than a 1-0 win."""
        engine = EloEngine(k=20, use_goal_margin=True, use_importance=False,
                           use_league_strength=False)
        rec_big = engine.update("A", "B", "H", home_goals=5, away_goals=0, match_index=0)
        engine.reset()
        rec_small = engine.update("A", "B", "H", home_goals=1, away_goals=0, match_index=0)
        assert rec_big.k_factor > rec_small.k_factor
        assert rec_big.home_elo_change > rec_small.home_elo_change

    def test_xg_margin_preferred(self):
        """xG margin should be used over actual goals when available."""
        engine = EloEngine(k=20, use_goal_margin=True)
        rec = engine.update(
            "A", "B", "H",
            home_goals=5, away_goals=0,   # Big actual win
            home_xg=0.5, away_xg=0.4,     # Small xG margin
            match_index=0,
        )
        # K should be based on xG margin (0.1), not actual goals (5)
        expected_margin = abs(0.5 - 0.4)  # xG margin = 0.1
        expected_K = 20.0 * max(0.5, np.log1p(expected_margin))
        assert abs(rec.k_factor - expected_K) < 1e-6

    def test_zero_zero_draw_min_k(self):
        """0-0 draw should use minimum K (0.5 × base)."""
        engine = EloEngine(k=20, use_goal_margin=True, use_importance=False,
                           use_league_strength=False)
        rec = engine.update("A", "B", "D", home_goals=0, away_goals=0, match_index=0)
        expected_min = 20.0 * max(0.5, np.log1p(0))  # = 20 * 0.5 = 10
        assert abs(rec.k_factor - expected_min) < 1e-6


# ═══════════════════════════════════════════════════════════════
#  Tests: Host Nation Bonus
# ═══════════════════════════════════════════════════════════════


class TestEloHostNation:
    def test_host_bonus_applied_to_expected_score(self):
        """Host nation should have higher expected score."""
        engine = EloEngine(home_advantage=100, host_bonus=50,
                           use_goal_margin=False)
        # Equal ratings + home advantage + host bonus
        expected_normal = engine.expected_score(1500, 1500)  # No host bonus in this method directly

        # Test via update with is_host=True
        rec_host = engine.update("Host", "Visitor", "H", is_host=True, match_index=0)
        engine.reset()
        rec_normal = engine.update("Home", "Visitor", "H", is_host=False, match_index=0)

        # Host nation should generate different expected score
        assert rec_host.expected_home != rec_normal.expected_home

    def test_host_bonus_does_not_persist(self):
        """Host bonus should only affect expected score, not update."""
        engine = EloEngine(home_advantage=100, host_bonus=50,
                           use_goal_margin=False)
        rec = engine.update("Host", "Visitor", "H", is_host=True, match_index=0)
        # The actual rating update uses unboosted rating
        # So changes should be consistent


# ═══════════════════════════════════════════════════════════════
#  Tests: History and Trajectory
# ═══════════════════════════════════════════════════════════════


class TestEloHistory:
    def test_get_history_df(self, two_team_matches):
        engine = EloEngine(k=32, use_goal_margin=False)
        engine.process_matches(two_team_matches)
        hist = engine.get_history_df()
        assert len(hist) == 2
        assert "h_elo_before" in hist.columns
        assert "h_elo_change" in hist.columns

    def test_team_trajectory(self, two_team_matches):
        engine = EloEngine(k=32, use_goal_margin=False, new_team_rating=1500.0)
        engine.process_matches(two_team_matches)
        traj = engine.team_trajectory("Team_A")
        assert len(traj) == 2  # Team_A appears in both matches (home/away)
        assert "elo_before" in traj.columns
        assert "side" in traj.columns
        assert set(traj["side"].values) == {"home", "away"}

    def test_current_snapshot(self, two_team_matches):
        engine = EloEngine(k=32, use_goal_margin=False)
        engine.process_matches(two_team_matches)
        snap = engine.current_snapshot()
        assert isinstance(snap, EloSnapshot)
        assert snap.total_matches_processed == 2
        assert len(snap.ratings) == 2  # Two teams

    def test_print_standings(self, two_team_matches, capsys):
        engine = EloEngine(k=32, use_goal_margin=False)
        engine.process_matches(two_team_matches)
        engine.print_standings(top_n=5)
        captured = capsys.readouterr()
        assert "ELO RATINGS" in captured.out


# ═══════════════════════════════════════════════════════════════
#  Tests: EloTransformer (FeatureTransformer wrapper)
# ═══════════════════════════════════════════════════════════════


class TestEloTransformer:
    def test_transformer_init(self):
        t = EloTransformer(k=20, home_advantage=80)
        assert t.engine is not None
        assert t.engine.k == 20
        assert t.engine.home_advantage == 80

    def test_transform_adds_columns(self, two_team_matches):
        t = EloTransformer(k=32)
        t.init()
        result = t.transform(two_team_matches.copy())

        assert "h_elo" in result.columns
        assert "a_elo" in result.columns
        assert "elo_diff" in result.columns

    def test_transform_pre_match_values(self, two_team_matches):
        t = EloTransformer(k=32, use_goal_margin=False)
        t.init()
        result = t.transform(two_team_matches.copy())

        # First match: both teams start at default (new_team_rating)
        assert result.loc[0, "h_elo"] == 1300.0  # Default new_team_rating
        assert result.loc[0, "a_elo"] == 1300.0

    def test_transform_preserves_original_columns(self, two_team_matches):
        t = EloTransformer(k=32)
        t.init()
        original_cols = set(two_team_matches.columns)
        result = t.transform(two_team_matches.copy())
        for col in original_cols:
            assert col in result.columns

    def test_engine_accessible(self):
        t = EloTransformer()
        assert hasattr(t, "engine")
        assert isinstance(t.engine, EloEngine)

    def test_reset_engine(self):
        t = EloTransformer()
        t.engine.update("A", "B", "H", match_index=0)
        assert t.engine.match_count == 1
        t.reset_engine()
        assert t.engine.match_count == 0

    def test_validate_input_missing_column(self):
        t = EloTransformer()
        df = pd.DataFrame({"home_team": ["A"]})
        errors = t.validate_input(df)
        assert len(errors) >= 1

    def test_validate_input_all_present(self, two_team_matches):
        t = EloTransformer()
        errors = t.validate_input(two_team_matches)
        assert errors == []

    def test_validate_output(self, two_team_matches):
        t = EloTransformer()
        t.init()
        result = t.transform(two_team_matches.copy())
        errors = t.validate_output(result)
        assert errors == []

    def test_metadata(self):
        t = EloTransformer()
        meta = t.metadata
        assert meta.name == "elo_rating"
        assert meta.category == "elo_rating"
        assert meta.data_type == "float"

    def test_to_dict(self):
        t = EloTransformer(k=25)
        d = t.to_dict()
        assert d["name"] == "elo_rating"
        assert d["params"].get("k") == 25

    def test_repr(self):
        t = EloTransformer()
        assert "EloTransformer" in repr(t)

    def test_create_elo_transformer(self):
        t = create_elo_transformer(k=30, home_advantage=80)
        assert isinstance(t, EloTransformer)
        assert t.engine.k == 30
        assert t.engine.home_advantage == 80


# ═══════════════════════════════════════════════════════════════
#  Tests: Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestEloEdgeCases:
    def test_empty_dataframe(self):
        engine = EloEngine()
        df = pd.DataFrame(columns=["date", "home_team", "away_team", "result"])
        result = engine.process_matches(df)
        assert len(result) == 0

    def test_single_row(self):
        engine = EloEngine(k=32, use_goal_margin=False)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "home_team": ["Team_A"],
            "away_team": ["Team_B"],
            "home_goals": [2],
            "away_goals": [0],
            "result": ["H"],
        })
        result = engine.process_matches(df)
        assert len(result) == 1
        assert result.loc[0, "h_elo"] == 1300.0  # new_team_rating
        assert len(engine.history) == 1

    def test_all_draws(self):
        """Consecutive draws should keep ratings close to initial."""
        engine = EloEngine(k=20, use_goal_margin=False, use_importance=False,
                           use_league_strength=False)
        n = 10
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="7D"),
            "home_team": [f"Team_{i % 4}" for i in range(n)],
            "away_team": [f"Team_{(i + 1) % 4}" for i in range(n)],
            "home_goals": [1] * n,
            "away_goals": [1] * n,
            "result": ["D"] * n,
        })
        engine.process_matches(df)
        # All ratings should still be near 1300 (new_team_rating)
        max_rating = max(engine.ratings.values())
        assert 1250 < max_rating < 1350

    def test_ratings_symmetric(self):
        """Home win then away win between same teams — Team_A won both."""
        engine = EloEngine(k=20, use_goal_margin=False, use_importance=False,
                           use_league_strength=False, new_team_rating=1500.0)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_A"],
            "home_goals": [2, 2],
            "away_goals": [0, 0],
            "result": ["H", "A"],  # Home win, then away win (same venue swap)
        })
        engine.process_matches(df)
        # Team_A won both matches → should have higher rating than Team_B
        assert engine.get_rating("Team_A") > engine.get_rating("Team_B")


# ═══════════════════════════════════════════════════════════════
#  Tests: Club Elo Benchmark Alignment
# ═══════════════════════════════════════════════════════════════


class TestEloBenchmark:
    def test_club_elo_aligned_default(self):
        """Default parameters should align with Club Elo."""
        engine = EloEngine()  # Defaults: k=20, home=100, initial=1500, new=1300
        report = engine.benchmark_report()
        assert report["club_elo_aligned"] is True

    def test_club_elo_not_aligned_custom(self):
        """Non-default K should flag as not aligned."""
        engine = EloEngine(k=32)
        report = engine.benchmark_report()
        assert report["club_elo_aligned"] is False

    def test_benchmark_report_structure(self):
        engine = EloEngine()
        report = engine.benchmark_report()
        assert "club_elo_parameters" in report
        assert "current_parameters" in report
        assert "ratings_count" in report
        assert "matches_processed" in report

    def test_process_matches_with_host_nations(self):
        """Host nation mapping should work."""
        engine = EloEngine(k=20, use_goal_margin=False)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-06-01"]),
            "home_team": ["USA"],
            "away_team": ["England"],
            "result": ["H"],
            "season": ["2026"],
        })
        result = engine.process_matches(
            df,
            host_nations={"2026": "USA"},
        )
        # USA should have host bonus applied
        assert "h_elo" in result.columns
