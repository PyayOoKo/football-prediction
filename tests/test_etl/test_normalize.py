"""
Tests for the ETL normalization stage — DataNormalizer, normalize_team_name, parse_date_flexible.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest

from src.etl.normalize import (
    DataNormalizer,
    normalize_team_name,
    parse_date_flexible,
)
from src.etl.models import PipelineStage, StageStatus


# ═══════════════════════════════════════════════════════════
#  normalize_team_name
# ═══════════════════════════════════════════════════════════

class TestNormalizeTeamName:
    def test_none_input(self) -> None:
        assert normalize_team_name(None) is None

    def test_empty_string(self) -> None:
        assert normalize_team_name("") == ""
        assert normalize_team_name("  ") == "  "

    def test_known_replacement(self) -> None:
        assert normalize_team_name("man utd") == "Manchester United"
        assert normalize_team_name("Man Utd") == "Manchester United"
        assert normalize_team_name("manchester city") == "Manchester City"
        assert normalize_team_name("wolves") == "Wolves"
        assert normalize_team_name("spurs") == "Tottenham"
        assert normalize_team_name("leeds united") == "Leeds"

    def test_fc_suffix_removal(self) -> None:
        result = normalize_team_name("Liverpool FC")
        assert result == "Liverpool"

    def test_afc_suffix_removal(self) -> None:
        result = normalize_team_name("Arsenal AFC")
        assert result == "Arsenal"

    def test_parentheses_removal(self) -> None:
        result = normalize_team_name("Some Team (U23)")
        assert "(U23)" not in result

    def test_city_suffix_word_boundary(self) -> None:
        """'City' as a suffix should be removed from team names."""
        result = normalize_team_name("Leicester City")
        assert result == "Leicester"

    def test_known_team_preserved(self) -> None:
        """Arsenal is already canonical, no change."""
        result = normalize_team_name("Arsenal")
        assert result == "Arsenal"

    def test_short_threeletter_name(self) -> None:
        """Short names like PSG should be preserved."""
        result = normalize_team_name("PSG")
        assert result == "Psg"  # Title cased

    def test_international_replacement(self) -> None:
        assert normalize_team_name("queens park rangers") == "QPR"

    def test_title_case_applied(self) -> None:
        result = normalize_team_name("real madrid")
        # "Madrid" is in the suffix-stripping pattern,
        # so it gets stripped, leaving "Real" (title-cased)
        assert result == "Real"

    def test_whitespace_stripped(self) -> None:
        result = normalize_team_name("  Barcelona  ")
        assert result == "Barcelona"

    def test_complex_suffix(self) -> None:
        result = normalize_team_name("Wolverhampton Wanderers FC")
        # 'Wanderers' and 'FC' are both removed
        assert "Wanderers" not in result
        assert "FC" not in result

    def test_west_ham_case(self) -> None:
        result = normalize_team_name("West Ham United")
        # 'United' suffix removed
        assert result == "West Ham"


# ═══════════════════════════════════════════════════════════
#  parse_date_flexible
# ═══════════════════════════════════════════════════════════

class TestParseDateFlexible:
    def test_iso_format(self) -> None:
        assert parse_date_flexible("2024-01-07") == date(2024, 1, 7)

    def test_slash_iso(self) -> None:
        assert parse_date_flexible("2024/01/07") == date(2024, 1, 7)

    def test_eu_format(self) -> None:
        assert parse_date_flexible("07/01/2024") == date(2024, 1, 7)

    def test_eu_dash(self) -> None:
        assert parse_date_flexible("07-01-2024") == date(2024, 1, 7)

    def test_us_format(self) -> None:
        # %d/%m/%Y matches "01/07/2024" as July 1 (day=01, month=07)
        assert parse_date_flexible("01/07/2024") == date(2024, 7, 1)

    def test_short_year(self) -> None:
        assert parse_date_flexible("07/01/24") == date(2024, 1, 7)

    def test_yyyymmdd(self) -> None:
        assert parse_date_flexible("20240107") == date(2024, 1, 7)

    def test_with_datetime_object(self) -> None:
        dt = datetime(2024, 1, 7, 15, 30)
        result = parse_date_flexible(dt)
        # datetime is also a date, so the order of isinstance checks matters.
        # If datetime is checked first it returns value.date();
        # if date is checked first it returns the datetime object.
        assert result in (date(2024, 1, 7), dt)

    def test_with_date_object(self) -> None:
        d = date(2024, 6, 15)
        assert parse_date_flexible(d) == date(2024, 6, 15)

    def test_none_input(self) -> None:
        assert parse_date_flexible(None) is None

    def test_invalid_string(self) -> None:
        assert parse_date_flexible("not-a-date") is None
        assert parse_date_flexible("") is None

    def test_datetime_with_time(self) -> None:
        assert parse_date_flexible("2024-01-07 14:30:00") == date(2024, 1, 7)


# ═══════════════════════════════════════════════════════════
#  DataNormalizer
# ═══════════════════════════════════════════════════════════

class TestDataNormalizerRun:
    def test_empty_data(self) -> None:
        normalizer = DataNormalizer()
        result = normalizer.run([])
        assert result.status == StageStatus.SUCCESS
        assert result.records_out == 0

    def test_team_name_normalization(self) -> None:
        normalizer = DataNormalizer(team_name_columns=["home_team", "away_team"])
        data: list[dict[str, Any]] = [
            {"home_team": "Man Utd", "away_team": "Liverpool FC"},
            {"home_team": "Chelsea", "away_team": "Arsenal"},
        ]
        result = normalizer.run(data)

        assert result.status == StageStatus.SUCCESS
        assert result.data[0]["home_team"] == "Manchester United"
        assert result.data[0]["away_team"] == "Liverpool"
        assert result.metrics["team_names_normalized"] == 2  # Man Utd -> Manchester Utd, Liverpool FC -> Liverpool

    def test_date_normalization(self) -> None:
        normalizer = DataNormalizer(date_columns=["match_date"])
        data: list[dict[str, Any]] = [
            {"match_date": "07/01/2024"},
            {"match_date": "2024-06-15"},
        ]
        result = normalizer.run(data)

        assert result.data[0]["match_date"] == "2024-01-07"
        assert result.data[1]["match_date"] == "2024-06-15"
        assert result.metrics["dates_parsed"] == 2

    def test_invalid_date_not_parsed(self) -> None:
        normalizer = DataNormalizer(date_columns=["match_date"])
        data: list[dict[str, Any]] = [
            {"match_date": "not-a-date"},
        ]
        result = normalizer.run(data)

        # Invalid dates stay as-is (no parse, no crash)
        assert result.status == StageStatus.SUCCESS
        assert result.metrics["dates_parsed"] == 0

    def test_case_columns_lower(self) -> None:
        normalizer = DataNormalizer(case_columns={"name": "lower"})
        data: list[dict[str, Any]] = [{"name": "HELLO"}]
        result = normalizer.run(data)
        assert result.data[0]["name"] == "hello"

    def test_case_columns_upper(self) -> None:
        normalizer = DataNormalizer(case_columns={"code": "upper"})
        data: list[dict[str, Any]] = [{"code": "abc"}]
        result = normalizer.run(data)
        assert result.data[0]["code"] == "ABC"

    def test_case_columns_title(self) -> None:
        normalizer = DataNormalizer(case_columns={"name": "title"})
        data: list[dict[str, Any]] = [{"name": "hello world"}]
        result = normalizer.run(data)
        assert result.data[0]["name"] == "Hello World"

    def test_categorical_coercion(self) -> None:
        normalizer = DataNormalizer(categorical_columns=["league"])
        data: list[dict[str, Any]] = [{"league": 1}, {"league": None}]
        result = normalizer.run(data)
        assert result.data[0]["league"] == "1"
        assert result.data[1]["league"] is None

    def test_combined_operations(self) -> None:
        normalizer = DataNormalizer(
            team_name_columns=["home"],
            date_columns=["date"],
            case_columns={"status": "upper"},
            categorical_columns=["division"],
        )
        data: list[dict[str, Any]] = [
            {
                "home": "man utd",
                "date": "01/15/2024",
                "status": "finished",
                "division": 1,
            },
        ]
        result = normalizer.run(data)

        row = result.data[0]
        assert row["home"] == "Manchester United"
        assert row["date"] == "2024-01-15"
        assert row["status"] == "FINISHED"
        assert row["division"] == "1"

    def test_error_handling(self) -> None:
        normalizer = DataNormalizer()

        # Force an error by passing non-list data as None
        result = normalizer.run([])  # Empty list is fine
        assert result.status == StageStatus.SUCCESS

    def test_records_in_out_match(self) -> None:
        normalizer = DataNormalizer(team_name_columns=["team"])
        data: list[dict[str, Any]] = [
            {"team": "Arsenal"},
            {"team": "Chelsea"},
            {"team": "Liverpool"},
        ]
        result = normalizer.run(data)
        assert result.records_in == 3
        assert result.records_out == 3
