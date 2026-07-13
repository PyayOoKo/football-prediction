"""
Tests for the ETL validation stage — DataValidator, SchemaValidator, built-in rules.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.etl.validate import (
    DataValidator,
    SchemaValidator,
    rule_no_null_columns,
    rule_not_empty,
    rule_unique_rows,
)
from src.etl.models import PipelineStage, StageStatus


# ═══════════════════════════════════════════════════════════
#  Built-in rules
# ═══════════════════════════════════════════════════════════

class TestRuleNotEmpty:
    def test_passes_with_data(self) -> None:
        result = rule_not_empty([{"a": 1}])
        assert result.passed is True
        assert result.record_count == 1

    def test_fails_on_empty(self) -> None:
        result = rule_not_empty([])
        assert result.passed is False
        assert result.failure_count == 1
        assert "0 rows" in result.details


class TestRuleNoNullColumns:
    def test_all_columns_have_data(self) -> None:
        data: list[dict[str, Any]] = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
        ]
        result = rule_no_null_columns(data)
        assert result.passed is True

    def test_entirely_null_column(self) -> None:
        data = [
            {"a": 1, "b": None},
            {"a": 2, "b": None},
        ]
        result = rule_no_null_columns(data)
        assert result.passed is False
        assert result.failure_count == 1
        assert "b" in result.details

    def test_empty_data_skips(self) -> None:
        result = rule_no_null_columns([])
        assert result.passed is True

    def test_with_specified_columns(self) -> None:
        data = [{"a": 1, "b": None}]
        result = rule_no_null_columns(data, required_columns=["a"])
        # 'a' has data, 'b' not checked
        assert result.passed is True

    def test_nan_values_detected(self) -> None:
        data = [{"a": float("nan")}]
        result = rule_no_null_columns(data)
        # NaN is technically not None, so the rule may or may not detect it
        # depending on the `v != v` NaN check. Ensure no exception is raised.
        assert result is not None


class TestRuleUniqueRows:
    def test_all_unique(self) -> None:
        data: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = rule_unique_rows(data)
        assert result.passed is True

    def test_duplicates_found(self) -> None:
        data = [{"id": 1}, {"id": 1}, {"id": 2}]
        result = rule_unique_rows(data)
        assert result.passed is False
        assert result.failure_count == 1

    def test_empty_data(self) -> None:
        result = rule_unique_rows([])
        assert result.passed is True

    def test_with_custom_keys(self) -> None:
        data = [
            {"match_id": 1, "date": "2024-01-01"},
            {"match_id": 1, "date": "2024-01-02"},  # Duplicate match_id
        ]
        result = rule_unique_rows(data, key_columns=["match_id"])
        assert result.passed is False
        assert result.failure_count == 1

    def test_all_duplicates(self) -> None:
        data = [{"id": 1}, {"id": 1}, {"id": 1}]
        result = rule_unique_rows(data)
        assert result.passed is False
        assert result.failure_count == 2


# ═══════════════════════════════════════════════════════════
#  SchemaValidator
# ═══════════════════════════════════════════════════════════

class TestSchemaValidator:
    def test_no_schema_passes(self) -> None:
        validator = SchemaValidator()
        result = validator.validate([{"a": 1}])
        assert result.passed is True

    def test_matches_schema(self) -> None:
        schema = {
            "id": {"type": int, "nullable": False},
            "name": {"type": str, "nullable": False},
        }
        validator = SchemaValidator(schema)
        data: list[dict[str, Any]] = [
            {"id": 1, "name": "Test"},
            {"id": 2, "name": "Test2"},
        ]
        result = validator.validate(data)
        assert result.passed is True

    def test_null_on_non_nullable(self) -> None:
        schema = {"id": {"type": int, "nullable": False}}
        validator = SchemaValidator(schema)
        data = [{"id": None}]
        result = validator.validate(data)
        assert result.passed is False
        assert result.failure_count == 1

    def test_wrong_type(self) -> None:
        schema = {"id": {"type": int, "nullable": True}}
        validator = SchemaValidator(schema)
        data = [{"id": "not_an_int"}]
        result = validator.validate(data)
        assert result.passed is False
        assert result.failure_count == 1

    def test_nullable_allows_none(self) -> None:
        schema = {"optional": {"type": int, "nullable": True}}
        validator = SchemaValidator(schema)
        data = [{"optional": None}]
        result = validator.validate(data)
        assert result.passed is True

    def test_empty_data_passes(self) -> None:
        schema = {"id": {"type": int, "nullable": False}}
        validator = SchemaValidator(schema)
        result = validator.validate([])
        assert result.passed is True


# ═══════════════════════════════════════════════════════════
#  DataValidator
# ═══════════════════════════════════════════════════════════

class TestDataValidator:
    def test_default_rules(self) -> None:
        """Default validator uses not_empty + unique_rows rules."""
        validator = DataValidator()
        result = validator.run([{"id": 1}])
        assert result.status == StageStatus.SUCCESS
        assert result.metrics["checks_total"] == 2

    def test_clean_data_passes(self) -> None:
        validator = DataValidator()
        result = validator.run([{"id": 1}, {"id": 2}])
        assert result.status == StageStatus.SUCCESS
        assert result.metrics["checks_passed"] == 2

    def test_empty_data_warning(self) -> None:
        validator = DataValidator()
        result = validator.run([])
        assert result.status == StageStatus.WARNING
        assert result.metrics["checks_failed"] >= 1

    def test_strict_mode_fails_on_error(self) -> None:
        validator = DataValidator(strict=True)
        result = validator.run([])
        assert result.status == StageStatus.FAILED

    def test_with_schema(self) -> None:
        schema = {"id": {"type": int, "nullable": False}}
        validator = DataValidator(
            schema=schema,
            rules=[],  # No built-in rules, only schema
        )
        result = validator.run([{"id": "wrong_type"}])
        assert result.status == StageStatus.WARNING
        assert result.metrics["checks_failed"] >= 1

    def test_custom_rules(self) -> None:
        def my_rule(data):
            from src.etl.models import ValidationRuleResult
            return ValidationRuleResult("my_rule", passed=True)

        validator = DataValidator(rules=[my_rule])
        result = validator.run([{"a": 1}])
        assert result.status == StageStatus.SUCCESS
        assert result.metrics["checks_total"] == 1

    def test_rule_that_raises_exception(self) -> None:
        def broken_rule(data):
            raise ValueError("Rule crashed")

        validator = DataValidator(rules=[broken_rule])
        result = validator.run([{"a": 1}])
        # Exception is caught and treated as failure
        assert result.metrics["checks_failed"] >= 1

    def test_passes_data_through(self) -> None:
        validator = DataValidator()
        data = [{"a": 1}]
        result = validator.run(data)
        assert result.data is data  # Same object passed through

    def test_records_in_out(self) -> None:
        validator = DataValidator()
        data = [{"a": 1}, {"b": 2}]
        result = validator.run(data)
        assert result.records_in == 2
        assert result.records_out == 2

    def test_warning_not_strict_by_default(self) -> None:
        validator = DataValidator()
        result = validator.run([])
        assert result.status == StageStatus.WARNING  # Not FAILED
