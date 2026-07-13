"""
Tests for the ScheduleTransformer — fixture schedule and congestion features.

Covers:
- Required columns validation
- Rest days computation (days_since_last_match)
- Matches in last 7/14 days
- Consecutive home/away streaks
- Back-to-back opponent detection
- Travel distance (with lat/lon data)
- Days since competition
- League-specific reset
- Edge cases (empty DF, single row, all same teams)
- FeatureTransformer lifecycle
- Factory function
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.feature_framework.features.schedule import (
    ScheduleTransformer,
    create_schedule_transformer,
)

# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """12 match rows with 8 distinct teams, spanning 3 months.

    Creates clear schedule patterns:
    - Team_E plays matches at indices 2, 6, 10 (4-week gaps → plenty of rest)
    - Team_D plays matches at indices 1, 5 (1-week gap)
    - Team_A plays matches at indices 0, 4, 8 (tight schedule)
    """
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07",
            "2024-01-10", "2024-01-12", "2024-01-15", "2024-01-17",
            "2024-01-20", "2024-01-22", "2024-01-25", "2024-01-28",
        ]),
        "home_team": [
            "Team_A", "Team_C", "Team_E", "Team_G",
            "Team_A", "Team_D", "Team_F", "Team_H",
            "Team_A", "Team_C", "Team_E", "Team_G",
        ],
        "away_team": [
            "Team_B", "Team_D", "Team_F", "Team_H",
            "Team_B", "Team_C", "Team_E", "Team_G",
            "Team_B", "Team_D", "Team_F", "Team_H",
        ],
        "home_goals": [3, 1, 2, 0, 0, 2, 1, 3, 2, 0, 4, 1],
        "away_goals": [0, 1, 1, 1, 2, 1, 0, 0, 0, 3, 0, 1],
        "result": ["H", "D", "H", "A", "A", "H", "H", "H", "H", "A", "H", "D"],
        "league": ["PL"] * 12,
        "season": ["2024"] * 12,
    })


@pytest.fixture
def tight_schedule() -> pd.DataFrame:
    """A team playing very frequently (5 matches in 15 days)."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-04", "2024-01-07",
            "2024-01-10", "2024-01-13",
        ]),
        "home_team": [
            "Team_A", "Team_A", "Team_A", "Team_A", "Team_A",
        ],
        "away_team": [
            "Team_B", "Team_C", "Team_D", "Team_E", "Team_F",
        ],
        "result": ["H", "H", "H", "H", "H"],
    })


@pytest.fixture
def venue_streak_matches() -> pd.DataFrame:
    """A team with alternating/streaky venue patterns."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-08", "2024-01-15",
            "2024-01-22", "2024-01-29", "2024-02-05",
        ]),
        "home_team": [
            "Team_A", "Team_B", "Team_A",
            "Team_A", "Team_C", "Team_A",
        ],
        "away_team": [
            "Team_B", "Team_A", "Team_B",
            "Team_B", "Team_A", "Team_B",
        ],
        "result": ["H", "A", "H", "H", "A", "H"],
        # Home team:      H    A    H    H    A    H
        # Team_A venue:   H    A    H    H    A    H
        # Team_B venue:   A    H    A    A    H    A
    })


@pytest.fixture
def back_to_back_matches() -> pd.DataFrame:
    """Matches where Team_B faces the same opponent twice in a row."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-08", "2024-01-15",
            "2024-01-22", "2024-01-29",
        ]),
        "home_team": ["Team_A", "Team_B", "Team_C", "Team_B", "Team_D"],
        "away_team": ["Team_B", "Team_A", "Team_B", "Team_C", "Team_B"],
        "result": ["H", "A", "H", "H", "D"],
        # Team_B faces opponents: Team_A, Team_A, Team_C, Team_C
        # Opponent sequence for Team_B: A (match 0, away), A (match 1, home),
        #                               C (match 2, away), C (match 3, home)
        # So match 1 has same opponent as match 0 (Team_A)
        # And match 3 has same opponent as match 2 (Team_C)
    })


@pytest.fixture
def sample_with_travel() -> pd.DataFrame:
    """Add lat/lon coordinates to sample matches."""
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-08",
        ]),
        "home_team": ["Team_A", "Team_C"],
        "away_team": ["Team_B", "Team_D"],
        "result": ["H", "D"],
        # London (51.5, -0.1) vs Manchester (53.5, -2.2)
        "home_lat": [51.5, 53.5],
        "home_lon": [-0.1, -2.2],
        "away_lat": [53.5, 51.5],
        "away_lon": [-2.2, -0.1],
        "league": ["PL", "PL"],
    })
    return df


@pytest.fixture
def multi_league_matches() -> pd.DataFrame:
    """Team_A plays in two different competitions (league and cup)."""
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-04", "2024-01-08",
            "2024-01-11", "2024-01-15",
        ]),
        "home_team": [
            "Team_A", "Team_A", "Team_A",
            "Team_A", "Team_A",
        ],
        "away_team": [
            "Team_B", "Team_C", "Team_B",
            "Team_D", "Team_C",
        ],
        "result": ["H", "H", "D", "H", "H"],
        "league": [
            "PL", "FA_Cup", "PL",
            "PL", "FA_Cup",
        ],
    })
    return df


@pytest.fixture
def edge_matches() -> pd.DataFrame:
    """Minimal edge-case DataFrame: few matches, 0-0 draws."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        "home_team": ["Team_X", "Team_Y", "Team_X"],
        "away_team": ["Team_Y", "Team_X", "Team_Y"],
        "result": ["D", "D", "D"],
    })


# ═══════════════════════════════════════════════════════════════
#  Tests: Input Validation
# ═══════════════════════════════════════════════════════════════


class TestScheduleInputValidation:
    def test_missing_required_column(self):
        t = ScheduleTransformer()
        df = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
        errors = t.validate_input(df)
        assert len(errors) >= 1
        assert any("date" in e for e in errors)

    def test_all_required_columns_present(self, sample_matches):
        t = ScheduleTransformer()
        errors = t.validate_input(sample_matches)
        assert errors == []

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "home_team", "away_team"])
        t = ScheduleTransformer()
        errors = t.validate_input(df)
        assert errors == []


# ═══════════════════════════════════════════════════════════════
#  Tests: Rest Days
# ═══════════════════════════════════════════════════════════════


class TestScheduleRestDays:
    @staticmethod
    def _run_transform(
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        t = ScheduleTransformer(**params)
        t.init()
        return t.transform(df.copy())

    def test_output_columns_exist(self, sample_matches):
        result = self._run_transform(sample_matches)
        assert "h_rest_days" in result.columns
        assert "a_rest_days" in result.columns
        assert "h_days_since_last_match" in result.columns
        assert "a_days_since_last_match" in result.columns

    def test_first_match_no_history(self, sample_matches):
        """First match for each team → NaN rest days."""
        result = self._run_transform(sample_matches)
        # Match 0: Team_A home vs Team_B (both first match)
        assert pd.isna(result.loc[0, "h_rest_days"])
        assert pd.isna(result.loc[0, "a_rest_days"])

    def test_rest_days_computation(self, sample_matches):
        """Team_A plays matches at indices 0, 4, 8 (dates: Jan 1, Jan 10, Jan 20)."""
        result = self._run_transform(sample_matches)

        # Match 4 (idx=4): Team_A away at Team_B, date=Jan 10
        # Team_A's previous match: Jan 1 (9 days earlier)
        h_rest_4 = result.loc[4, "h_rest_days"]
        assert not pd.isna(h_rest_4)
        assert abs(h_rest_4 - 9.0) < 1e-6  # Jan 10 - Jan 1 = 9 days

        # Match 8 (idx=8): Team_A home vs Team_B, date=Jan 20
        # Team_A's previous match: Jan 10 (10 days earlier)
        h_rest_8 = result.loc[8, "h_rest_days"]
        assert not pd.isna(h_rest_8)
        assert abs(h_rest_8 - 10.0) < 1e-6  # Jan 20 - Jan 10 = 10 days

    def test_tight_schedule_rest_days(self, tight_schedule):
        """Team_A plays every 3 days → rest_days = 3."""
        result = self._run_transform(tight_schedule)
        for i in range(1, len(tight_schedule)):
            assert not pd.isna(result.loc[i, "h_rest_days"])
            assert abs(result.loc[i, "h_rest_days"] - 3.0) < 1e-6

    def test_rest_days_alias(self, tight_schedule):
        """days_since_last_match should equal rest_days."""
        result = self._run_transform(tight_schedule)
        for i in range(len(tight_schedule)):
            h_rest = result.loc[i, "h_rest_days"]
            h_dslm = result.loc[i, "h_days_since_last_match"]
            if pd.isna(h_rest):
                assert pd.isna(h_dslm)
            else:
                assert abs(h_rest - h_dslm) < 1e-6


# ═══════════════════════════════════════════════════════════════
#  Tests: Matches in Last 7/14 Days
# ═══════════════════════════════════════════════════════════════


class TestScheduleMatchDensity:
    @staticmethod
    def _run_transform(
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        t = ScheduleTransformer(**params)
        t.init()
        return t.transform(df.copy())

    def test_density_columns_exist(self, sample_matches):
        result = self._run_transform(sample_matches)
        assert "h_matches_last_7_days" in result.columns
        assert "a_matches_last_7_days" in result.columns
        assert "h_matches_last_14_days" in result.columns
        assert "a_matches_last_14_days" in result.columns

    def test_first_match_zero_density(self, sample_matches):
        """First match → 0 matches in last 7/14 days."""
        result = self._run_transform(sample_matches)
        assert result.loc[0, "h_matches_last_7_days"] == 0
        assert result.loc[0, "a_matches_last_7_days"] == 0
        assert result.loc[0, "h_matches_last_14_days"] == 0

    def test_density_tight_schedule(self, tight_schedule):
        """Team_A plays every 3 days → after match 2, 2 matches in last 7."""
        result = self._run_transform(tight_schedule)
        # Match 0 (Jan 1): 0 matches before
        assert result.loc[0, "h_matches_last_7_days"] == 0
        # Match 1 (Jan 4): 1 match (Jan 1) in last 7
        assert result.loc[1, "h_matches_last_7_days"] == 1
        # Match 2 (Jan 7): 2 matches (Jan 1, Jan 4) in last 7 → but Jan 1 is 6 days ago
        # Jan 7 - Jan 1 = 6 days ≤ 7 → yes, both Jan 1 and Jan 4 are within 7 days
        assert result.loc[2, "h_matches_last_7_days"] == 2

    def test_density_14_day_window(self, tight_schedule):
        """After 5 matches in 13 days, should have 4 in last 14."""
        result = self._run_transform(tight_schedule)
        # Match 4 (Jan 13): 4 previous matches (Jan 1, 4, 7, 10)
        # All within 14 days (Jan 13 - Jan 1 = 12 days)
        assert result.loc[4, "h_matches_last_14_days"] == 4

    def test_density_7_vs_14_difference(self, tight_schedule):
        """After match 4: 14-day window has 4, 7-day window has 3."""
        result = self._run_transform(tight_schedule)
        # Match 4 (Jan 13): Previous matches: Jan 1 (12d), Jan 4 (9d), Jan 7 (6d), Jan 10 (3d)
        # 7-day window: matches >= Jan 6 → Jan 7, Jan 10 → 2
        # Actually Jan 13 - 7 = Jan 6. So matches on Jan 7 and Jan 10 are within 7 days.
        assert result.loc[4, "h_matches_last_7_days"] == 2
        # 14-day window: matches >= Dec 30 → all 4
        assert result.loc[4, "h_matches_last_14_days"] == 4


# ═══════════════════════════════════════════════════════════════
#  Tests: Consecutive Home/Away Streaks
# ═══════════════════════════════════════════════════════════════


class TestScheduleConsecutiveStreaks:
    @staticmethod
    def _run_transform(
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        t = ScheduleTransformer(**params)
        t.init()
        return t.transform(df.copy())

    def test_streak_columns_exist(self, venue_streak_matches):
        result = self._run_transform(venue_streak_matches)
        assert "h_consec_home" in result.columns
        assert "a_consec_home" in result.columns
        assert "h_consec_away" in result.columns
        assert "a_consec_away" in result.columns

    def test_home_team_venue_streaks(self, venue_streak_matches):
        """Home team venue sequence: H, A, H, H, A, H."""
        result = self._run_transform(venue_streak_matches)

        # Match 0 (Team_A home, first match of all time → NaN)
        assert pd.isna(result.loc[0, "h_consec_home"])
        assert pd.isna(result.loc[0, "h_consec_away"])

        # Match 1 (Team_B home, first home match for Team_B → NaN)
        assert pd.isna(result.loc[1, "h_consec_home"])

        # Match 2 (Team_A home again, previous match was also home at idx 0)
        # Team_A's venue sequence: H (idx 0), A (idx 1), H (idx 2), H (idx 3), A (idx 4), H (idx 5)
        # For Team_A at match 2 (third Team_A match): venue = H
        # Previous Team_A match at idx 1: A (different venue, start of new streak)
        # So at match 2: streak of H just started → 0 consecutive H before
        # Let me check: match 1 is Team_B home, Team_A away.
        # Team_A's venue sequence for its 6 matches:
        #   idx 0: H (home vs Team_B)
        #   idx 1: A (away at Team_B)
        #   idx 2: H (home vs Team_B)
        #   idx 3: H (home vs Team_B)
        #   idx 4: A (away at Team_C)
        #   idx 5: H (home vs Team_B)
        # Streak groups: [H], [A], [H, H], [A], [H]
        # cumcount: [0], [0], [0, 1], [0], [0]
        # For h_consec (is_home=1): [0 at idx 0], [NaN at idx 1 (is_home=0)],
        #   [0 at idx 2], [1 at idx 3], [NaN at idx 4], [0 at idx 5]
        # After shift/masking first of streak: [NaN, NaN, NaN, 1, NaN, NaN]

        # But we need to check what the output actually is.

        # Match 3 (Team_A home, same venue streak as match 2)
        # For Team_A at match 3: venue = H, previous match was also H (idx 2)
        # cumcount at idx 3 in its H-streak = 1 (1 previous H in same streak)
        # So consec_home should be 1.0
        h_consec_home_3 = result.loc[3, "h_consec_home"]
        assert not pd.isna(h_consec_home_3)
        assert h_consec_home_3 > 0.5  # At least 1 consecutive home

    def test_away_team_streaks(self, venue_streak_matches):
        """Away team venue sequence from matches: A, H, A, A, H, A."""
        # Match 0: Team_B away (away venue) → first away match → NaN away streak
        # Match 1: Team_A away (away venue) → first away match for Team_A → NaN
        # Match 2: Team_B away (away venue) → Team_B's second away appearance
        # Match 3: Team_B away (away venue) → Team_B's third consecutive away
        result = self._run_transform(venue_streak_matches)

        # Match 0: Team_B is away, first away match → NaN
        assert pd.isna(result.loc[0, "a_consec_away"])

        # Match 2: Team_B is away again (second consec away for Team_B)
        # Wait, match 2 is Team_A vs Team_B. Team_B is away.
        # Team_B's venue sequence: A (idx 0), H (idx 1), A (idx 2), A (idx 3), H (idx 4), A (idx 5)
        # Streak groups: [A], [H], [A, A], [H], [A]
        # cumcount: [0], [0], [0, 1], [0], [0]
        # At match 2 (idx 2 in df, but 3rd Team_B appearance):
        #   Team_B's streak position = 0 in A-streak
        #   But first match of streak → NaN
        assert pd.isna(result.loc[2, "a_consec_away"])

    def test_no_leakage(self, venue_streak_matches):
        """The current match should not affect its own streak features.

        For Team_A playing consecutively at home (matches 2 and 3):
        Match 3 should show 1, NOT 2 (the current match is not counted).
        """
        result = self._run_transform(venue_streak_matches)
        # Match 3: Team_A at home, had 1 consecutive home before → value should be 1.0
        h_consec_3 = result.loc[3, "h_consec_home"]
        assert not pd.isna(h_consec_3)
        assert 0.5 <= h_consec_3 <= 1.5  # Should be ~1.0


# ═══════════════════════════════════════════════════════════════
#  Tests: Back-to-Back Opponent
# ═══════════════════════════════════════════════════════════════


class TestScheduleBackToBack:
    @staticmethod
    def _run_transform(
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        t = ScheduleTransformer(**params)
        t.init()
        return t.transform(df.copy())

    def test_column_exists(self, back_to_back_matches):
        result = self._run_transform(back_to_back_matches)
        assert "h_is_back_to_back" in result.columns
        assert "a_is_back_to_back" in result.columns

    def test_first_match_not_b2b(self, back_to_back_matches):
        """First match should have is_back_to_back = 0."""
        result = self._run_transform(back_to_back_matches)
        assert result.loc[0, "h_is_back_to_back"] == 0
        assert result.loc[0, "a_is_back_to_back"] == 0

    def test_back_to_back_detected(self, back_to_back_matches):
        """Team_B faces Team_A in matches 0 and 1 → is_back_to_back for Matches 1."""
        result = self._run_transform(back_to_back_matches)
        # Match 1: Team_B home vs Team_A (away), same opponent as match 0
        # Team_B's previous opponent was Team_A (in match 0) → b2b=1
        assert result.loc[1, "a_is_back_to_back"] == 1  # Team_A is away at Match 1

        # Match 0: Team_A home vs Team_B → Team_A's first match → 0
        # Actually wait, Team_A has a previous match... no, match 0 is first match for everyone
        assert result.loc[0, "h_is_back_to_back"] == 0

    def test_no_false_positives(self, back_to_back_matches):
        """Different opponent → is_back_to_back = 0."""
        result = self._run_transform(back_to_back_matches)
        # Match 2: Team_C home vs Team_B (away). Team_B's prev opponent was Team_A → not b2b
        assert result.loc[2, "a_is_back_to_back"] == 0  # Team_B is away, prev opp was Team_A


# ═══════════════════════════════════════════════════════════════
#  Tests: Travel Distance
# ═══════════════════════════════════════════════════════════════


class TestScheduleTravelDistance:
    @staticmethod
    def _run_transform(
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        t = ScheduleTransformer(**params)
        t.init()
        return t.transform(df.copy())

    def test_travel_column_exists(self, sample_with_travel):
        result = self._run_transform(sample_with_travel)
        assert "h_travel_distance" in result.columns
        assert "a_travel_distance" in result.columns

    def test_first_match_no_travel(self, sample_with_travel):
        """First match for a team → NaN travel distance (no previous venue)."""
        result = self._run_transform(sample_with_travel)
        assert pd.isna(result.loc[0, "h_travel_distance"])
        assert pd.isna(result.loc[0, "a_travel_distance"])

    def test_travel_distance_no_data(self, sample_matches):
        """Without lat/lon columns, travel_distance should be NaN."""
        result = self._run_transform(sample_matches)
        assert "h_travel_distance" in result.columns
        assert pd.isna(result.loc[0, "h_travel_distance"])

    def test_travel_distance_not_included(self, sample_with_travel):
        """When include_travel_distance=False, column should not exist."""
        result = self._run_transform(
            sample_with_travel.copy(), include_travel_distance=False,
        )
        assert "h_travel_distance" not in result.columns
        assert "a_travel_distance" not in result.columns

    def test_london_to_manchester_distance(self, sample_with_travel):
        """London to Manchester is ~262 km via Haversine."""
        result = self._run_transform(sample_with_travel)
        # Team_A is home at match 0 (London), Team_B is away at match 0 but we need
        # match 1 for Team_B's travel (from previous venue).
        # Actually: Team_B is away at match 0 (in London), then at match 1 they are
        # not playing (it's Team_C vs Team_D). Wait, Team_B is away at 0.
        # Let me check: match 0 = Team_A (home, London) vs Team_B (away, Manchester).
        # Team_B's first match is away at London (Team_A's venue).
        # match 1 = Team_C vs Team_D, different teams. Team_B doesn't play.
        # So Team_B only has 1 match → no travel distance for them (NaN).
        # Team_A plays match 0 only → no travel distance (NaN first match).

        # Let me use a fixture where a team plays TWO matches with different venues.
        travel_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_A"],
            "result": ["H", "A"],
            "home_lat": [51.5, 53.5],  # London, Manchester
            "home_lon": [-0.1, -2.2],
            "away_lat": [53.5, 51.5],  # Manchester, London
            "away_lon": [-2.2, -0.1],
            "league": ["PL", "PL"],
        })
        result = self._run_transform(travel_df)

        # Match 1: Team_A is away at Manchester (from London)
        # Travel distance for Team_A from London (match 0) to Manchester (match 1)
        a_travel = result.loc[1, "a_travel_distance"]
        assert not pd.isna(a_travel)
        # London (51.5, -0.1) to Manchester (53.5, -2.2) ≈ 262 km
        assert 200 < a_travel < 350

        # Match 1: Team_B is home at Manchester (from London)
        # Travel distance for Team_B from London (match 0, away) to Manchester (match 1, home)
        h_travel = result.loc[1, "h_travel_distance"]
        assert not pd.isna(h_travel)
        assert 200 < h_travel < 350


# ═══════════════════════════════════════════════════════════════
#  Tests: Days Since Competition
# ═══════════════════════════════════════════════════════════════


class TestScheduleDaysSinceCompetition:
    @staticmethod
    def _run_transform(
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        t = ScheduleTransformer(**params)
        t.init()
        return t.transform(df.copy())

    def test_column_exists(self, multi_league_matches):
        result = self._run_transform(multi_league_matches)
        assert "h_days_since_competition" in result.columns
        assert "a_days_since_competition" in result.columns

    def test_first_match_nan(self, multi_league_matches):
        """First match for a team → NaN."""
        result = self._run_transform(multi_league_matches)
        assert pd.isna(result.loc[0, "h_days_since_competition"])

    def test_same_competition_gap(self, multi_league_matches):
        """Team_A's PL matches at indices 0, 2, 3 (Jan 1, Jan 8, Jan 11)."""
        result = self._run_transform(multi_league_matches)
        # Match 2 (idx=2): Team_A vs Team_B, PL, date=Jan 8
        # Previous PL match for Team_A: match 0 (Jan 1) → 7 days
        h_comp_2 = result.loc[2, "h_days_since_competition"]
        assert not pd.isna(h_comp_2)
        assert abs(h_comp_2 - 7.0) < 1e-6

    def test_different_competition_gap(self, multi_league_matches):
        """Cup matches track separately from league."""
        result = self._run_transform(multi_league_matches)
        # Match 1 (idx=1): Team_A vs Team_C, FA_Cup, date=Jan 4
        # First FA_Cup match for Team_A → NaN
        assert pd.isna(result.loc[1, "h_days_since_competition"])

        # Match 4 (idx=4): Team_A vs Team_C, FA_Cup, date=Jan 15
        # Previous FA_Cup match for Team_A: match 1 (Jan 4) → 11 days
        h_comp_4 = result.loc[4, "h_days_since_competition"]
        assert not pd.isna(h_comp_4)
        assert abs(h_comp_4 - 11.0) < 1e-6

    def test_no_league_column_fallback(self, edge_matches):
        """Without league column, falls back to rest_days."""
        result = self._run_transform(edge_matches)
        # Should still have the column
        assert "h_days_since_competition" in result.columns


# ═══════════════════════════════════════════════════════════════
#  Tests: League-Specific Reset
# ═══════════════════════════════════════════════════════════════


class TestScheduleLeagueSpecific:
    @staticmethod
    def _run_transform(
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        t = ScheduleTransformer(**params)
        t.init()
        return t.transform(df.copy())

    def test_league_specific_feature_computation(self, multi_league_matches):
        """League-specific computation groups by (team, league)."""
        result = self._run_transform(multi_league_matches)

        # All required columns should exist regardless
        assert "h_rest_days" in result.columns
        assert "h_matches_last_7_days" in result.columns

    def test_league_specific_false(self, multi_league_matches):
        """When league_specific=False, all matches for a team are grouped together."""
        result = self._run_transform(multi_league_matches, league_specific=False)
        # Features should still be computed (just without league grouping)
        assert "h_rest_days" in result.columns
        assert not pd.isna(result.loc[1, "h_rest_days"])


# ═══════════════════════════════════════════════════════════════
#  Tests: Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestScheduleEdgeCases:
    def test_empty_dataframe(self):
        """Empty DataFrame should not crash and should still get output columns."""
        t = ScheduleTransformer()
        t.init()
        df = pd.DataFrame(columns=["date", "home_team", "away_team"])
        result = t.transform(df)
        assert isinstance(result, pd.DataFrame)
        # All output columns should exist, filled with NaN
        for col in t.output_columns:
            assert col in result.columns

    def test_single_row(self):
        """Single row → all features should be NaN or 0."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "home_team": ["Team_A"],
            "away_team": ["Team_B"],
            "result": ["H"],
        })
        t = ScheduleTransformer()
        t.init()
        result = t.transform(df)

        assert pd.isna(result.loc[0, "h_rest_days"])
        assert result.loc[0, "h_matches_last_7_days"] == 0
        assert result.loc[0, "h_matches_last_14_days"] == 0
        assert pd.isna(result.loc[0, "h_consec_home"])
        assert pd.isna(result.loc[0, "h_consec_away"])
        assert result.loc[0, "h_is_back_to_back"] == 0
        assert pd.isna(result.loc[0, "h_travel_distance"])

    def test_two_matches_same_teams(self):
        """Two matches between same teams."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "home_team": ["Team_A", "Team_B"],
            "away_team": ["Team_B", "Team_A"],
            "result": ["H", "A"],
        })
        t = ScheduleTransformer()
        t.init()
        result = t.transform(df)

        # Match 0: first match for all → NaN rest
        assert pd.isna(result.loc[0, "h_rest_days"])
        assert pd.isna(result.loc[0, "a_rest_days"])

        # Match 1: Team_B's second match → 7 days rest
        assert not pd.isna(result.loc[1, "h_rest_days"])
        assert abs(result.loc[1, "h_rest_days"] - 7.0) < 1e-6

        # Team_A's second match (away at Team_B) → 7 days rest
        assert not pd.isna(result.loc[1, "a_rest_days"])
        assert abs(result.loc[1, "a_rest_days"] - 7.0) < 1e-6

    def test_preserves_original_columns(self, sample_matches):
        """All original columns should survive transformation."""
        t = ScheduleTransformer()
        t.init()
        original_cols = set(sample_matches.columns)
        result = t.transform(sample_matches.copy())
        for col in original_cols:
            assert col in result.columns

    def test_no_duplicate_columns(self, sample_matches):
        """Running transform twice should not duplicate columns."""
        t = ScheduleTransformer()
        t.init()
        result1 = t.transform(sample_matches.copy())
        result2 = t.transform(result1.copy())

        col_counts = result2.columns.value_counts()
        assert col_counts.max() == 1, "Duplicate columns detected"


# ═══════════════════════════════════════════════════════════════
#  Tests: Validation & Metadata
# ═══════════════════════════════════════════════════════════════


class TestScheduleValidation:
    def test_validate_output_passes(self, sample_matches):
        t = ScheduleTransformer()
        t.init()
        result = t.transform(sample_matches.copy())
        errors = t.validate_output(result)
        assert errors == []

    def test_validate_output_fails_empty_df(self):
        t = ScheduleTransformer()
        t.init()
        df = pd.DataFrame(columns=["date", "home_team"])
        errors = t.validate_output(df)
        assert len(errors) > 0

    def test_validate_output_all_present(self):
        """After transform with non-empty data, validation should pass."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "home_team": ["Team_A"],
            "away_team": ["Team_B"],
            "result": ["H"],
        })
        t = ScheduleTransformer()
        t.init()
        result = t.transform(df)
        errors = t.validate_output(result)
        assert errors == []

    def test_validate_output_missing_columns(self):
        """Validate on empty DataFrame should report missing columns."""
        t = ScheduleTransformer()
        t.init()
        df = pd.DataFrame()
        errors = t.validate_output(df)
        assert len(errors) > 0
        assert all("Missing output column" in e for e in errors)


class TestScheduleMetadata:
    def test_metadata(self):
        t = ScheduleTransformer()
        meta = t.metadata
        assert meta.name == "schedule"
        assert meta.version == 1
        assert meta.data_type == "float"
        assert meta.category == "schedule"
        assert meta.computation_time == "fast"

    def test_repr(self):
        t = ScheduleTransformer()
        assert "ScheduleTransformer" in repr(t) or "schedule" in repr(t)

    def test_to_dict(self):
        t = ScheduleTransformer(league_specific=False)
        d = t.to_dict()
        assert d["name"] == "schedule"
        assert d["params"].get("league_specific") is False


class TestScheduleFactory:
    def test_create_schedule_transformer(self):
        t = create_schedule_transformer(
            league_specific=False,
            include_travel_distance=False,
        )
        assert isinstance(t, ScheduleTransformer)
        assert t.params.get("league_specific") is False
        assert t.params.get("include_travel_distance") is False

    def test_default_params(self):
        t = create_schedule_transformer()
        assert t.params.get("league_specific") is True
        assert t.params.get("include_travel_distance") is True

    def test_custom_params(self):
        t = create_schedule_transformer(
            sort_by_date=False,
            custom_extra="hello",
        )
        assert t.params.get("sort_by_date") is False
        assert t.params.get("custom_extra") == "hello"
