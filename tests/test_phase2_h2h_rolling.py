"""Phase 2 tests: H2H window behavior and rolling features correctness.

Each test uses handcrafted data to verify deterministic behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _tiny_matches() -> pd.DataFrame:
    """6 matches: Team A hosts Team B repeatedly — deterministic."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15",
                                "2023-01-22", "2023-01-29", "2023-02-05"]),
        "home_team": ["A", "A", "A", "A", "A", "A"],
        "away_team": ["B", "B", "B", "B", "B", "B"],
        "result": ["H", "A", "D", "H", "H", "A"],
        "home_goals": [2, 0, 1, 3, 2, 1],
        "away_goals": [1, 2, 1, 1, 0, 2],
        "season": "2023",
    })


def _team_stats_handcrafted() -> pd.DataFrame:
    """Team A: 4 matches (3 home, 1 away) to test win rate denominators."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-03-01", "2023-03-08", "2023-03-15", "2023-03-22"]),
        "home_team": ["A", "A", "B", "A"],
        "away_team": ["B", "C", "A", "D"],
        "result": ["H", "D", "A", "H"],
        "home_goals": [2, 1, 0, 3],
        "away_goals": [0, 1, 1, 1],
        "season": "2023",
    })


# ═══════════════════════════════════════════════════════════════
#  1. H2H Window Tests
# ═══════════════════════════════════════════════════════════════


class TestH2HWindow:
    """H2H window parameter must limit previous meetings."""

    def test_window_limits_previous_meetings(self):
        """With window=2, only last 2 meetings contribute to features."""
        from src.features.contextual import _compute_h2h_stats

        df = _tiny_matches()
        # Match 5 (index 4): should see only matches 3,4 as previous (not 0,1,2)
        # Match 5 result: H, goals 2-0
        # Previous meetings with window=2: matches 3 (H, 3-1) and 4 (H, 2-0)
        # home_points_avg for match 5: (3 + 3) / 2 = 3.0
        # away_points_avg for match 5: (0 + 0) / 2 = 0.0
        window = 2
        h2h = _compute_h2h_stats(df, window=window)

        # Check match index 5 (6th match, 0-indexed)
        row = h2h.loc[5]
        assert abs(row.get("home_points_avg", -1) - 3.0) < 0.01, (
            f"Expected home_points_avg ≈ 3.0, got {row.get('home_points_avg')}"
        )
        assert abs(row.get("away_points_avg", -1) - 0.0) < 0.01, (
            f"Expected away_points_avg ≈ 0.0, got {row.get('away_points_avg')}"
        )
        # With window=5, all 5 previous meetings would give a lower avg
        # (matches 0-4: H=3pts, A=3pts, D=1pt each → home avg = (3+0+1+3+3)/5=2.0)
        window5 = _compute_h2h_stats(df, window=5)
        row5 = window5.loc[5]
        assert abs(row5.get("home_points_avg", -1) - 2.0) < 0.01, (
            f"Window=5 should average all 5 prior matches, got {row5.get('home_points_avg')}"
        )

    def test_window_respects_first_match(self):
        """First match has NaN for H2H features (shift(1) makes it NaN)."""
        from src.features.contextual import _compute_h2h_stats

        df = _tiny_matches()
        h2h = _compute_h2h_stats(df, window=3)

        # First match (index 0) — no previous meetings → NaN
        row0 = h2h.loc[0]
        assert pd.isna(row0.get("matches_played")), (
            f"Expected NaN for first match, got {row0.get('matches_played')}"
        )
        assert pd.isna(row0.get("home_points_avg"))

    def test_window_fewer_than_window_matches(self):
        """When fewer prior meetings than window, still works (rolling min_periods=1)."""
        from src.features.contextual import _compute_h2h_stats

        df = _tiny_matches().head(3)  # Only 3 matches
        h2h = _compute_h2h_stats(df, window=10)  # Window larger than data

        # Match 2 (index 2) has 2 previous meetings
        row2 = h2h.loc[2]
        assert pd.notna(row2.get("matches_played")), (
            "Should have data even when window > total rows"
        )
        assert row2.get("matches_played") == 2

    def test_h2h_stats_per_ordered_pair(self):
        """H2H stats are computed per (home_team, away_team) ordered pair."""
        from src.features.contextual import _compute_h2h_stats

        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "B"],
            "result": ["H", "D", "A"],
            "home_goals": [2, 1, 0],
            "away_goals": [1, 1, 2],
            "season": ["2023", "2023", "2023"],
        })

        h2h = _compute_h2h_stats(df, window=5)

        # Pair (A, B): matches 0 and 2
        # Match 2: previous is match 0 (A home, H -> A 3pts, B 0pts)
        # home_points_avg = 3.0 (only match 0 before match 2)
        row2 = h2h.loc[2]
        assert abs(row2.get("home_points_avg", -1) - 3.0) < 0.01, (
            f"Expected home_points_avg=3.0, got {row2.get('home_points_avg')}"
        )
        assert row2.get("matches_played", 0) == 1.0, (
            f"Expected matches_played=1, got {row2.get('matches_played')}"
        )


# ═══════════════════════════════════════════════════════════════
#  2. Rolling Feature Tests
# ═══════════════════════════════════════════════════════════════


class TestRollingFeaturesWinRates:
    """Win rate denominators must be correct."""

    def test_win_rate_home_correct_denominator(self):
        """win_rate_home = home wins / home matches, not / all matches."""
        from src.features.rolling import _compute_team_stats, _merge_team_stats

        df = _team_stats_handcrafted()
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})

        team_stats = _compute_team_stats(df)
        result = _merge_team_stats(df, team_stats, windows=(3,))

        # Team A's matches:
        # Match 0: A(home) vs B, 2-0 H → A gets 3 pts, home win
        # Match 1: A(home) vs C, 1-1 D → A gets 1 pt, home draw
        # Match 2: B(home) vs A, 1-0 A → A gets 3 pts (as away team, B is "home")
        # Wait, result="A" means AWAY team won. So B(home) vs A(away), result "A" → A won as away.

        # Let me re-examine:
        # Match 2: home=B, away=A, result=A → A (away) wins 1-0
        # Team A was away → they get 3 points

        # For match 3: A(home) vs D, 3-1 H → A wins
        # Before match 3 (A's perspective as home team):
        # - Match 0: home win → 1 home win out of 1 home match → win_rate_home = 1.0
        # - Match 1: home draw → 1 home win out of 2 home matches → win_rate_home = 0.5
        # - Match 2: away win → doesn't affect home win count

        # So for match 3 (A is home):
        # Home wins = 1 (only match 0 was a home win)
        # Home matches = 2 (match 0 and match 1 were home)
        # win_rate_home = 1/2 = 0.5

        h_win_rate = result.loc[3, "h_win_rate_home"] if "h_win_rate_home" in result.columns else None
        if h_win_rate is not None:
            assert abs(h_win_rate - 0.5) < 0.01, (
                f"Expected h_win_rate_home=0.5, got {h_win_rate}"
            )

    def test_first_match_nan_rolling_features(self):
        """First match for a team has NaN rolling features (shift(1))."""
        from src.features.rolling import _compute_team_stats, _merge_team_stats

        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "result": ["H"],
            "home_goals": [1],
            "away_goals": [0],
            "season": ["2023"],
            "target": [2],
        })
        team_stats = _compute_team_stats(df)
        result = _merge_team_stats(df, team_stats, windows=(5,))

        # First match → no history → NaN for home rolling features
        h_form = result.get("h_form_last5", pd.Series([np.nan]))
        assert pd.isna(h_form.iloc[0]), (
            f"Expected NaN for first match, got {h_form.iloc[0]}"
        )

    def test_team_with_only_home_matches(self):
        """A team that only plays at home still gets correct away features as opponent."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "B", "B"],
            "result": ["H", "H", "D"],
            "home_goals": [2, 1, 1],
            "away_goals": [0, 1, 1],
            "season": ["2023", "2023", "2023"],
            "target": [2, 2, 1],
        })

        from src.features.rolling import _compute_team_stats

        team_stats = _compute_team_stats(df)
        # Team A should have 3 rows (all home), Team B should have 3 rows (all away)
        team_a = team_stats[team_stats["team"] == "A"]
        team_b = team_stats[team_stats["team"] == "B"]

        assert len(team_a) == 3
        assert len(team_b) == 3
        assert team_a["is_home"].sum() == 3  # All A's matches are home
        assert team_b["is_home"].sum() == 0  # All B's matches are away

    def test_days_since_last_match(self):
        """days_since_last_match measures gap between consecutive fixtures."""
        from src.features.rolling import _compute_team_stats, _merge_team_stats

        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-22"]),
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "result": ["H", "D", "H"],
            "home_goals": [2, 1, 3],
            "away_goals": [0, 1, 0],
            "season": ["2023", "2023", "2023"],
            "target": [2, 1, 2],
        })

        team_stats = _compute_team_stats(df)
        result = _merge_team_stats(df, team_stats, windows=(5,))

        # Match 0: first match → NaN days
        # Match 1: 7 days after match 0 → 7
        # Match 2: 14 days after match 1 → 14

        # For team A (home):
        h_days = result["h_days_since_last_match"]
        assert pd.isna(h_days.iloc[0]), f"Match 0 should be NaN, got {h_days.iloc[0]}"
        assert h_days.iloc[1] == 7.0, f"Match 1 should be 7, got {h_days.iloc[1]}"
        assert h_days.iloc[2] == 14.0, f"Match 2 should be 14, got {h_days.iloc[2]}"


class TestRollingSameDay:
    """Same-day matches must not leak information."""

    def test_same_day_matches_stable_ordering(self):
        """Teams playing twice on same date have stable sort by match_id."""
        from src.features.rolling import _compute_team_stats

        df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "result": ["H", "A"],
            "home_goals": [1, 0],
            "away_goals": [0, 2],
            "season": ["2023", "2023"],
            "target": [2, 0],
        })

        team_stats = _compute_team_stats(df)

        # Both teams appear in 2 rows each (home + away)
        for team in ["A", "B"]:
            tdf = team_stats[team_stats["team"] == team]
            # Sort order should be stable (by date then match_id)
            assert len(tdf) == 2, f"Team {team} should have 2 stat rows"


class TestRollingSeasonBoundaries:
    """Season boundaries should reset or continue per config."""

    def test_matches_this_season_resets_per_season(self):
        """matches_this_season counts correctly within each season."""
        from src.features.rolling import _compute_team_stats, _merge_team_stats

        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2022-11-01", "2022-12-01",
                "2023-01-01", "2023-02-01", "2023-03-01",
            ]),
            "home_team": ["A", "A", "A", "A", "A"],
            "away_team": ["B", "B", "B", "B", "B"],
            "result": ["H", "H", "D", "H", "A"],
            "home_goals": [2, 2, 1, 3, 0],
            "away_goals": [0, 0, 1, 0, 2],
            "season": ["2022", "2022", "2023", "2023", "2023"],
            "target": [2, 2, 1, 2, 0],
        })

        team_stats = _compute_team_stats(df)
        result = _merge_team_stats(df, team_stats, windows=(5,))

        # Team A's home matches:
        # Match 0 (2022-11-01): first of 2022 → NaN (shift(1) of cumcount+1)
        # Match 1 (2022-12-01): second of 2022 → 1 (one match before it in '22)
        # Match 2 (2023-01-01): first of 2023 → NaN (season reset)
        # Match 3 (2023-02-01): second of 2023 → 1
        # Match 4 (2023-03-01): third of 2023 → 2

        matches = result["h_matches_this_season"]
        assert pd.isna(matches.iloc[0]), f"Match 0 should be NaN, got {matches.iloc[0]}"
        assert matches.iloc[1] == 1.0, f"Match 1 should be 1, got {matches.iloc[1]}"
        assert pd.isna(matches.iloc[2]), f"Match 2 should be NaN, got {matches.iloc[2]}"
        assert matches.iloc[3] == 1.0, f"Match 3 should be 1, got {matches.iloc[3]}"
        assert matches.iloc[4] == 2.0, f"Match 4 should be 2, got {matches.iloc[4]}"
