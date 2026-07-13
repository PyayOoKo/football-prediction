"""
Tests for validation data models — CheckResult, ValidationResult, Severity.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.validation.models import (
    CheckResult,
    Severity,
    ValidationResult,
)


class TestSeverity:
    def test_values(self) -> None:
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"


class TestCheckResult:
    def test_minimal(self) -> None:
        result = CheckResult(
            check_name="Duplicate Matches",
            description="Check for duplicate match entries",
            severity=Severity.ERROR,
            passed=True,
        )
        assert result.check_name == "Duplicate Matches"
        assert result.passed is True
        assert result.violations == []

    def test_with_violations(self) -> None:
        result = CheckResult(
            check_name="Duplicate Matches",
            description="Check for duplicate match entries",
            severity=Severity.ERROR,
            passed=False,
            total_rows=100,
            violation_count=2,
            violations=[
                {"row_index": 10, "message": "Duplicate match_id=123"},
                {"row_index": 25, "message": "Duplicate match_id=456"},
            ],
        )
        assert result.passed is False
        assert result.violation_count == 2
        assert len(result.violations) == 2

    def test_to_dict(self) -> None:
        result = CheckResult(
            check_name="Missing Teams",
            description="Teams must exist",
            severity=Severity.ERROR,
            passed=False,
            total_rows=50,
            violation_count=3,
            violations=[{"row_index": 1, "value": None}],
        )
        d = result.to_dict()
        assert d["check_name"] == "Missing Teams"
        assert d["severity"] == "error"
        assert d["passed"] is False
        assert "sample_violations" in d
        assert len(d["sample_violations"]) == 1

    def test_no_violations_in_to_dict(self) -> None:
        result = CheckResult(
            check_name="Test Check",
            description="Test",
            severity=Severity.INFO,
            passed=True,
        )
        d = result.to_dict()
        assert d["sample_violations"] == []

    def test_severity_warning(self) -> None:
        result = CheckResult(
            check_name="Suspicious Score",
            description="Check for unusual scores",
            severity=Severity.WARNING,
            passed=True,
        )
        assert result.severity == Severity.WARNING
        assert result.passed is True

    def test_info_passed_no_violations(self) -> None:
        result = CheckResult(
            check_name="Stat Observation",
            description="Just observing",
            severity=Severity.INFO,
            passed=True,
        )
        assert result.passed is True
        assert result.violation_count == 0


class TestValidationResult:
    def test_all_passed(self) -> None:
        result = ValidationResult(
            source_name="test_data",
            total_rows=100,
            checks=[
                CheckResult("C1", "Check 1", Severity.ERROR, passed=True),
                CheckResult("C2", "Check 2", Severity.ERROR, passed=True),
            ],
        )
        assert result.passed is True
        assert result.total_checks == 2
        assert result.passed_checks == 2
        assert result.failed_checks == 0
        assert result.total_violations == 0

    def test_some_failed(self) -> None:
        result = ValidationResult(
            checks=[
                CheckResult("C1", "", Severity.ERROR, passed=True),
                CheckResult("C2", "", Severity.ERROR, passed=False, violation_count=5),
                CheckResult("C3", "", Severity.WARNING, passed=True),
            ],
        )
        assert result.passed is False
        assert result.total_checks == 3
        assert result.passed_checks == 2
        assert result.failed_checks == 1
        assert result.total_violations == 5

    def test_empty_checks(self) -> None:
        result = ValidationResult()
        assert result.passed is True
        assert result.total_checks == 0
        assert result.passed_checks == 0
        assert result.failed_checks == 0

    def test_to_dict_structure(self) -> None:
        result = ValidationResult(
            source_name="test_source",
            total_rows=100,
            checks=[
                CheckResult("C1", "Desc", Severity.ERROR, passed=True),
            ],
        )
        d = result.to_dict()
        assert d["source_name"] == "test_source"
        assert d["total_rows"] == 100
        assert d["total_checks"] == 1
        assert d["passed_checks"] == 1
        assert "checks" in d
        assert "timestamp" in d

    def test_to_json_string(self) -> None:
        result = ValidationResult(
            source_name="json_test",
            checks=[
                CheckResult("C1", "Desc", Severity.ERROR, passed=True),
            ],
        )
        json_str = result.to_json()
        assert isinstance(json_str, str)
        loaded = json.loads(json_str)
        assert loaded["source_name"] == "json_test"
        assert loaded["total_checks"] == 1

    def test_to_json_file(self) -> None:
        result = ValidationResult(
            source_name="file_test",
            checks=[
                CheckResult("C1", "Desc", Severity.ERROR, passed=True),
            ],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output = result.to_json(f.name)
            assert output == ""  # file mode returns empty string
            with open(f.name) as f_read:
                loaded = json.load(f_read)
            assert loaded["source_name"] == "file_test"
        os.unlink(f.name)

    def test_to_csv_export(self) -> None:
        result = ValidationResult(
            source_name="csv_test",
            checks=[
                CheckResult(
                    "Missing Teams", "Desc", Severity.ERROR,
                    passed=False, violation_count=1,
                    violations=[{"row_index": 5, "field": "home_team", "value": None, "message": "Null home team"}],
                ),
            ],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            result.to_csv(f.name)
            with open(f.name) as f_read:
                content = f_read.read()
            assert "check_name" in content
            assert "Missing Teams" in content
            assert "error" in content
        os.unlink(f.name)

    def test_to_html_export(self) -> None:
        result = ValidationResult(
            source_name="html_test",
            checks=[
                CheckResult("Pass Check", "Desc", Severity.ERROR, passed=True),
            ],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            result.to_html(f.name)
            with open(f.name, encoding="utf-8") as f_read:
                content = f_read.read()
            assert "<!DOCTYPE html>" in content
            assert "html_test" in content
        os.unlink(f.name)

    def test_timestamp_generated(self) -> None:
        result = ValidationResult()
        assert result.timestamp != ""

    def test_to_csv_no_violations(self) -> None:
        """Exporting CSV with no violations produces header only."""
        result = ValidationResult(
            checks=[
                CheckResult("C1", "Desc", Severity.INFO, passed=True),
            ],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            result.to_csv(f.name)
            with open(f.name) as f_read:
                content = f_read.read()
            assert "check_name" in content
            assert "severity" in content
        os.unlink(f.name)
