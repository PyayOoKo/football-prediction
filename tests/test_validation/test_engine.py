"""
Tests for ValidationEngine — orchestration and result aggregation.
"""

from __future__ import annotations

from src.validation.engine import ValidationEngine


class TestValidationEngine:
    CLEAN_DATA = [
        {"id": 1, "date": "2024-01-07", "home_team": "Arsenal",
         "away_team": "Chelsea", "home_goals": 2, "away_goals": 1,
         "result": "H", "status": "finished", "league": "E0"},
        {"id": 2, "date": "2024-01-14", "home_team": "Liverpool",
         "away_team": "Man City", "home_goals": None, "away_goals": None,
         "result": None, "status": "scheduled", "league": "E0"},
    ]

    def test_run_all_checks(self) -> None:
        engine = ValidationEngine()
        result = engine.run(self.CLEAN_DATA, source_name="test")
        assert result.total_checks == 9
        assert result.source_name == "test"
        assert result.total_rows == 2

    def test_run_with_empty_data(self) -> None:
        engine = ValidationEngine()
        result = engine.run([], source_name="empty")
        assert result.total_rows == 0
        assert result.total_checks == 0  # No checks run on empty data

    def test_run_selected_checks(self) -> None:
        engine = ValidationEngine()
        result = engine.run_selected(
            self.CLEAN_DATA,
            check_names=["Duplicate Matches", "Missing Teams"],
            source_name="test",
        )
        assert result.total_checks == 2

    def test_all_checks_pass_on_clean_data(self) -> None:
        engine = ValidationEngine(verbose=False)
        result = engine.run(self.CLEAN_DATA)
        assert result.passed is True
        assert result.passed_checks == 9
        assert result.failed_checks == 0

    def test_check_that_fails(self) -> None:
        dirty = [
            {"id": 1, "date": "2024-01-07", "home_team": "Arsenal",
             "away_team": "Chelsea", "home_goals": None, "away_goals": None,
             "result": "H", "status": "finished", "league": "E0"},
            {"id": 1, "date": "2024-01-07", "home_team": "Arsenal",
             "away_team": "Chelsea", "home_goals": 2, "away_goals": 0,
             "result": "H", "status": "finished", "league": "E0"},
        ]
        engine = ValidationEngine(verbose=False)
        result = engine.run(dirty)
        assert result.passed is False
        assert result.failed_checks > 0
        assert result.total_violations > 0

    def test_result_has_all_properties(self) -> None:
        engine = ValidationEngine(verbose=False)
        result = engine.run(self.CLEAN_DATA, source_name="test")

        assert hasattr(result, "to_dict")
        d = result.to_dict()
        assert "total_checks" in d
        assert "passed_checks" in d
        assert "failed_checks" in d
        assert "checks" in d
        assert len(d["checks"]) == 9
