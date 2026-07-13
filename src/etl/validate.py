"""
Validation stage — schema compliance and data quality checks.

Produces a ``ValidationReport`` with per-rule pass/fail/warn results.
Validation is **optional** by default — warnings don't block the
pipeline unless ``validate_strict=True``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from src.etl.models import (
    PipelineStage,
    StageResult,
    StageStatus,
    ValidationReport,
    ValidationRuleResult,
)

logger = logging.getLogger(__name__)

# ── Built-in rules ─────────────────────────────────────

RuleFn = Callable[[list[dict[str, Any]]], ValidationRuleResult]


def rule_not_empty(data: list[dict[str, Any]]) -> ValidationRuleResult:
    """Fail if the dataset has zero rows."""
    n = len(data)
    return ValidationRuleResult(
        rule_name="not_empty",
        passed=n > 0,
        record_count=n,
        failure_count=1 if n == 0 else 0,
        details=f"Dataset has {n} rows",
    )


def rule_no_null_columns(
    data: list[dict[str, Any]],
    required_columns: list[str] | None = None,
) -> ValidationRuleResult:
    """Check that required columns exist and are not entirely null."""
    if not data:
        return ValidationRuleResult("no_null_columns", passed=True)

    cols = required_columns or list(data[0].keys())
    null_cols = []
    for col in cols:
        values = [row.get(col) for row in data]
        if all(v is None or (isinstance(v, float) and v != v) for v in values):
            null_cols.append(col)

    return ValidationRuleResult(
        rule_name="no_null_columns",
        passed=len(null_cols) == 0,
        failure_count=len(null_cols),
        details=f"Fully null columns: {null_cols}" if null_cols else "All columns have data",
    )


def rule_unique_rows(
    data: list[dict[str, Any]],
    key_columns: list[str] | None = None,
) -> ValidationRuleResult:
    """Check for duplicate rows by key columns."""
    if not data:
        return ValidationRuleResult("unique_rows", passed=True)

    keys = key_columns or list(data[0].keys())
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0

    for row in data:
        key = tuple(row.get(k) for k in keys)
        if key in seen:
            duplicates += 1
        seen.add(key)

    return ValidationRuleResult(
        rule_name="unique_rows",
        passed=duplicates == 0,
        failure_count=duplicates,
        details=f"{duplicates} duplicate rows found (keys: {keys})",
    )


# ── Validator ──────────────────────────────────────────


class SchemaValidator:
    """Column-level schema validation.

    Checks that a dataset matches expected column names,
    types, and nullability constraints.
    """

    def __init__(self, schema: dict[str, dict[str, Any]] | None = None) -> None:
        """
        Parameters
        ----------
        schema : dict, optional
            Expected schema dict: ``{col_name: {type: type, nullable: bool}}``.
            Example::

                {
                    "match_id": {"type": int, "nullable": False},
                    "home_goals": {"type": int, "nullable": True},
                    "match_date": {"type": str, "nullable": False},
                }
        """
        self.schema = schema or {}

    def validate(self, data: list[dict[str, Any]]) -> ValidationRuleResult:
        """Validate data against the schema.

        Parameters
        ----------
        data : list[dict]
            Data to validate.

        Returns
        -------
        ValidationRuleResult
        """
        if not self.schema:
            return ValidationRuleResult("schema_validator", passed=True)

        if not data:
            return ValidationRuleResult(
                "schema_validator", passed=True,
                details="No data to validate against schema",
            )

        failures = 0
        details: list[str] = []

        for col, constraints in self.schema.items():
            expected_type = constraints.get("type")
            nullable = constraints.get("nullable", True)

            for i, row in enumerate(data):
                val = row.get(col)

                if val is None:
                    if not nullable:
                        failures += 1
                        if failures <= 5:
                            details.append(f"Row {i}: '{col}' is null but not nullable")
                    continue

                if expected_type is not None and not isinstance(val, expected_type):
                    failures += 1
                    if failures <= 5:
                        details.append(
                            f"Row {i}: '{col}' expected {expected_type.__name__}, "
                            f"got {type(val).__name__} ({val!r})"
                        )

        return ValidationRuleResult(
            rule_name="schema_validator",
            passed=failures == 0,
            failure_count=failures,
            details="; ".join(details[:10]),
        )


class DataValidator:
    """Pluggable validation engine.

    Runs a configurable list of rule functions against the
    extracted data and produces a ``ValidationReport``.

    Parameters
    ----------
    rules : list[RuleFn]
        Validation rule functions. Each takes ``(data,)`` and
        returns a ``ValidationRuleResult``.
    schema : dict, optional
        Optional schema for column-level validation.
    strict : bool
        If True, failures cause the stage to FAIL (default False).
    """

    def __init__(
        self,
        rules: list[RuleFn] | None = None,
        schema: dict[str, dict[str, Any]] | None = None,
        strict: bool = False,
    ) -> None:
        self.rules = rules or [
            rule_not_empty,
            rule_unique_rows,
        ]
        self.schema_validator = SchemaValidator(schema) if schema else None
        self.strict = strict

    def run(self, data: list[dict[str, Any]]) -> StageResult:
        """Execute all validation rules.

        Parameters
        ----------
        data : list[dict]
            Raw extracted data.

        Returns
        -------
        StageResult
            Stage result with ``data`` containing the ``ValidationReport``.
        """
        stage = PipelineStage.VALIDATE
        result = StageResult(stage=stage, status=StageStatus.RUNNING)
        start = time.perf_counter()

        result.records_in = len(data)
        report = ValidationReport()

        # Run each rule
        for rule_fn in self.rules:
            try:
                rule_result = rule_fn(data)
            except Exception as exc:
                rule_result = ValidationRuleResult(
                    rule_name=rule_fn.__name__,
                    passed=False,
                    failure_count=1,
                    details=f"Rule raised exception: {exc}",
                )

            report.rules.append(rule_result)
            report.total_checks += 1

            if rule_result.passed:
                report.passed += 1
            elif not rule_result.passed and rule_result.failure_count == 0:
                report.warnings += 1
            else:
                report.failures += 1

        # Schema validation (if configured)
        if self.schema_validator is not None:
            schema_result = self.schema_validator.validate(data)
            report.rules.append(schema_result)
            report.total_checks += 1
            if schema_result.passed:
                report.passed += 1
            else:
                report.failures += 1

        # Determine status
        result.metrics["checks_passed"] = report.passed
        result.metrics["checks_failed"] = report.failures
        result.metrics["checks_total"] = report.total_checks

        if report.failures > 0 and self.strict:
            result.status = StageStatus.FAILED
            result.errors.append(
                f"Validation strict mode: {report.failures} check(s) failed"
            )
        elif report.failures > 0:
            result.status = StageStatus.WARNING
            result.errors.append(f"{report.failures} validation check(s) failed")
        elif report.warnings > 0:
            result.status = StageStatus.WARNING
        else:
            result.status = StageStatus.SUCCESS

        result.data = data  # pass data through unchanged
        result.records_out = len(data)
        result.duration_seconds = time.perf_counter() - start

        logger.info(
            "Validation: %d passed, %d failed, %d warnings in %.1fs",
            report.passed,
            report.failures,
            report.warnings,
            result.duration_seconds,
        )

        return result
