"""
Tests for all 9 football-specific validation checks.

Each check is tested with:
- Clean data that should pass
- Dirty data that should trigger violations
- Edge cases (empty list, missing columns)
"""

from __future__ import annotations

from src.validation.checks import (
    check_duplicate_ids,
    check_duplicate_matches,
    check_impossible_scores,
    check_incorrect_leagues,
    check_invalid_dates,
    check_invalid_odds,
    check_invalid_statistics,
    check_missing_goals,
    check_missing_teams,
)

# ── Sample data ────────────────────────────────────────

CLEAN_MATCHES = [
    {"id": 1, "date": "2024-01-07", "home_team": "Arsenal", "away_team": "Chelsea",
     "home_goals": 2, "away_goals": 1, "result": "H", "status": "finished",
     "league": "E0", "possession_home": 55, "BbAvH": 2.10, "BbAvD": 3.40, "BbAvA": 3.80},
    {"id": 2, "date": "2024-01-08", "home_team": "Liverpool", "away_team": "Man City",
     "home_goals": 1, "away_goals": 1, "result": "D", "status": "finished",
     "league": "E0", "possession_home": 48, "BbAvH": 2.50, "BbAvD": 3.30, "BbAvA": 2.80},
    {"id": 3, "date": "2024-01-14", "home_team": "Man Utd", "away_team": "Tottenham",
     "home_goals": None, "away_goals": None, "result": None, "status": "scheduled",
     "league": "E0"},
]


# ═══════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════

class TestCheckDuplicateMatches:
    def test_clean_data_passes(self) -> None:
        result = check_duplicate_matches(CLEAN_MATCHES)
        assert result.passed is True
        assert result.violation_count == 0

    def test_duplicate_detected(self) -> None:
        data = CLEAN_MATCHES + [
            {"id": 4, "date": "2024-01-07", "home_team": "Arsenal",
             "away_team": "Chelsea", "home_goals": 3, "away_goals": 0},
        ]
        result = check_duplicate_matches(data)
        assert result.passed is False
        assert result.violation_count == 1

    def test_empty_data(self) -> None:
        result = check_duplicate_matches([])
        assert result.passed is True


class TestCheckInvalidDates:
    def test_clean_dates_pass(self) -> None:
        result = check_invalid_dates(CLEAN_MATCHES)
        assert result.passed is True

    def test_null_date_detected(self) -> None:
        data = [{"id": 1, "date": None}]
        result = check_invalid_dates(data)
        assert result.passed is False
        assert result.violation_count == 1

    def test_empty_string_date(self) -> None:
        data = [{"id": 1, "date": ""}]
        result = check_invalid_dates(data)
        assert result.passed is False

    def test_garbled_date(self) -> None:
        data = [{"id": 1, "date": "not-a-date"}]
        result = check_invalid_dates(data)
        assert result.passed is False

    def test_date_object(self) -> None:
        from datetime import date
        data = [{"id": 1, "date": date(2024, 1, 7)}]
        result = check_invalid_dates(data)
        assert result.passed is True


class TestCheckInvalidOdds:
    def test_clean_odds_pass(self) -> None:
        result = check_invalid_odds(CLEAN_MATCHES)
        assert result.passed is True

    def test_odds_less_than_one(self) -> None:
        data = [{"id": 1, "BbAvH": 0.5, "BbAvD": 1.2, "BbAvA": 3.0}]
        result = check_invalid_odds(data)
        assert result.passed is False
        assert result.violation_count == 1

    def test_no_odds_columns(self) -> None:
        data = [{"id": 1, "home_team": "A", "away_team": "B"}]
        result = check_invalid_odds(data)
        assert result.passed is True  # No odds columns = nothing to validate

    def test_non_numeric_odds(self) -> None:
        data = [{"id": 1, "BbAvH": "abc", "BbAvD": "1.5", "BbAvA": "3.0"}]
        result = check_invalid_odds(data)
        assert result.passed is False
        assert result.violation_count == 1


class TestCheckMissingGoals:
    def test_clean_goals_pass(self) -> None:
        result = check_missing_goals(CLEAN_MATCHES)
        assert result.passed is True

    def test_missing_goals_on_finished(self) -> None:
        data = [{"id": 1, "home_goals": None, "away_goals": None,
                 "result": "H", "status": "finished"}]
        result = check_missing_goals(data)
        assert result.passed is False
        assert result.violation_count == 1

    def test_scheduled_match_skipped(self) -> None:
        data = [{"id": 1, "home_goals": None, "away_goals": None,
                 "result": None, "status": "scheduled"}]
        result = check_missing_goals(data)
        assert result.passed is True  # scheduled matches skipped


class TestCheckMissingTeams:
    def test_clean_teams_pass(self) -> None:
        result = check_missing_teams(CLEAN_MATCHES)
        assert result.passed is True

    def test_null_home_team(self) -> None:
        data = [{"id": 1, "home_team": None, "away_team": "Chelsea"}]
        result = check_missing_teams(data)
        assert result.passed is False
        assert result.violation_count == 1

    def test_null_away_team(self) -> None:
        data = [{"id": 1, "home_team": "Arsenal", "away_team": ""}]
        result = check_missing_teams(data)
        assert result.passed is False

    def test_identical_teams(self) -> None:
        data = [{"id": 1, "home_team": "Arsenal", "away_team": "arsenal"}]
        result = check_missing_teams(data)
        assert result.passed is False
        assert result.violation_count == 1


class TestCheckIncorrectLeagues:
    def test_clean_leagues_pass(self) -> None:
        result = check_incorrect_leagues(CLEAN_MATCHES)
        assert result.passed is True

    def test_unknown_league(self) -> None:
        data = [{"id": 1, "league": "NONEXISTENT_SUPER_LIGA"}]
        result = check_incorrect_leagues(data)
        assert result.passed is False

    def test_null_league(self) -> None:
        data = [{"id": 1, "league": None}]
        result = check_incorrect_leagues(data)
        assert result.passed is False

    def test_empty_league(self) -> None:
        data = [{"id": 1, "league": ""}]
        result = check_incorrect_leagues(data)
        assert result.passed is False


class TestCheckInvalidStatistics:
    def test_clean_stats_pass(self) -> None:
        result = check_invalid_statistics(CLEAN_MATCHES)
        assert result.passed is True

    def test_negative_possession(self) -> None:
        data = [{"id": 1, "possession_home": -10}]
        result = check_invalid_statistics(data)
        assert result.passed is False

    def test_possession_over_100(self) -> None:
        data = [{"id": 1, "possession": 150}]
        result = check_invalid_statistics(data)
        assert result.passed is False

    def test_negative_corners(self) -> None:
        data = [{"id": 1, "corners_away": -5}]
        result = check_invalid_statistics(data)
        assert result.passed is False


class TestCheckDuplicateIDs:
    def test_clean_ids_pass(self) -> None:
        result = check_duplicate_ids(CLEAN_MATCHES)
        assert result.passed is True

    def test_duplicate_id_detected(self) -> None:
        data = [{"id": 1}, {"id": 1}, {"id": 2}]
        result = check_duplicate_ids(data)
        assert result.passed is False
        assert result.violation_count == 1

    def test_null_ids_skipped(self) -> None:
        data = [{"id": None}, {"id": None}]
        result = check_duplicate_ids(data)
        assert result.passed is True


class TestCheckImpossibleScores:
    def test_clean_scores_pass(self) -> None:
        result = check_impossible_scores(CLEAN_MATCHES)
        assert result.passed is True

    def test_negative_home_goals(self) -> None:
        data = [{"id": 1, "home_goals": -1, "away_goals": 2, "result": "A"}]
        result = check_impossible_scores(data)
        assert result.passed is False
        assert result.violation_count == 1

    def test_negative_away_goals(self) -> None:
        data = [{"id": 1, "home_goals": 2, "away_goals": -3, "result": "H"}]
        result = check_impossible_scores(data)
        assert result.passed is False

    def test_excessive_score(self) -> None:
        data = [{"id": 1, "home_goals": 25, "away_goals": 0, "result": "H"}]
        result = check_impossible_scores(data, max_score=20)
        assert result.passed is False

    def test_result_mismatch(self) -> None:
        data = [{"id": 1, "home_goals": 3, "away_goals": 0, "result": "A"}]
        result = check_impossible_scores(data)
        assert result.passed is False

    def test_non_integer_goals(self) -> None:
        data = [{"id": 1, "home_goals": "two", "away_goals": 1, "result": None}]
        result = check_impossible_scores(data)
        assert result.passed is False

    def test_upcoming_match_skipped(self) -> None:
        data = [{"id": 1, "home_goals": None, "away_goals": None, "result": None}]
        result = check_impossible_scores(data)
        assert result.passed is True
