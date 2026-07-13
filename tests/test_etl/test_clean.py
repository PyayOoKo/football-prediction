"""
Tests for the ETL cleaning stage — DataCleaner.
"""

from __future__ import annotations

from src.etl.clean import DataCleaner


class TestDataCleaner:
    def test_trim_strings(self) -> None:
        data = [{"name": "  Arsenal  ", "goals": 2}]
        result = DataCleaner().run(data)
        assert result.data[0]["name"] == "Arsenal"

    def test_type_coercion(self) -> None:
        data = [{"goals": "2", "rating": "7.5"}]
        result = DataCleaner(
            type_coercions={"goals": int, "rating": float}
        ).run(data)
        assert isinstance(result.data[0]["goals"], int)
        assert isinstance(result.data[0]["rating"], float)

    def test_drop_null_keys(self) -> None:
        """Should drop rows where key_columns are null."""
        data = [
            {"match_id": 1, "home_team": "Arsenal", "away_team": "Chelsea"},
            {"match_id": None, "home_team": "Liverpool", "away_team": None},
            {"match_id": 3, "home_team": "Man City", "away_team": "Man Utd"},
        ]
        result = DataCleaner(
            fill_strategy="drop",
            key_columns=["match_id"],
        ).run(data)
        # Row 2 dropped because match_id is None
        assert len(result.data) == 2

    def test_keep_null_optional(self) -> None:
        """Should keep rows where non-key columns are null."""
        data = [
            {"match_id": 1, "home_goals": 2, "away_goals": 1, "possession": None},
            {"match_id": 2, "home_goals": 0, "away_goals": 0, "possession": 55.0},
        ]
        result = DataCleaner(
            fill_strategy="drop",
            key_columns=["match_id", "home_goals", "away_goals"],
        ).run(data)
        # Both rows kept because key columns are non-null
        assert len(result.data) == 2

    def test_fill_zero(self) -> None:
        data = [
            {"goals": None, "possession": 60.0},
            {"goals": 2, "possession": None},
        ]
        result = DataCleaner(fill_strategy="fill_zero").run(data)
        assert result.data[0]["goals"] == 0
        assert result.data[1]["possession"] == 0

    def test_dedup(self) -> None:
        data = [
            {"id": 1, "name": "A"},
            {"id": 2, "name": "B"},
            {"id": 1, "name": "A"},  # duplicate
        ]
        result = DataCleaner(
            drop_duplicates=True,
            duplicate_keys=["id"],
        ).run(data)
        assert len(result.data) == 2

    def test_clip_ranges(self) -> None:
        data = [
            {"possession": 105.0, "goals": -1},
            {"possession": 50.0, "goals": 5},
        ]
        result = DataCleaner(
            clip_ranges={"possession": (0.0, 100.0), "goals": (0, 20)}
        ).run(data)
        assert result.data[0]["possession"] == 100.0
        assert result.data[0]["goals"] == 0

    def test_empty_data(self) -> None:
        result = DataCleaner().run([])
        assert result.status.value == "warning"
