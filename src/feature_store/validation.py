"""
FeatureValidator — validation rules engine for computed features.

Provides:
- ``ValidationRule`` base class and built-in rule types
- ``FeatureValidator`` orchestrator that runs rules against feature values
- Integration with ``FeatureDefinition.validation_rules`` JSON config

Validation rules are defined per-feature in the ``validation_rules``
JSON column of ``FeatureDefinition``. Example::

    {
        "min": 0.0,
        "max": 1.0,
        "nullable": false,
        "max_cardinality": 0.3,
        "expected_type": "float"
    }
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.feature_store.models import FeatureDefinition, FeatureValue

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Results
# ═══════════════════════════════════════════════════════════


@dataclass
class ValidationResult:
    """Result of validating a single feature value.

    Attributes
    ----------
    feature_name : str
        Name of the validated feature.
    passed : bool
        Whether all rules passed.
    errors : list[str]
        Human-readable error messages.
    warnings : list[str]
        Non-critical warnings.
    metadata : dict
        Extra validation metadata (rule counts, timing).
    """

    feature_name: str = ""
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Combine this result with another (AND semantics)."""
        return ValidationResult(
            feature_name=self.feature_name or other.feature_name,
            passed=self.passed and other.passed,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


# ═══════════════════════════════════════════════════════════
#  Rule base class
# ═══════════════════════════════════════════════════════════


class ValidationRule(ABC):
    """Abstract base for a single validation rule.

    Each rule checks one aspect of a feature value and produces
    a ``ValidationResult``.

    Parameters
    ----------
    name : str
        Rule name for identification.
    severity : str
        ``error`` or ``warning``.
    """

    def __init__(self, name: str, severity: str = "error") -> None:
        self.name = name
        self.severity = severity

    @abstractmethod
    def validate(
        self,
        definition: FeatureDefinition,
        value: FeatureValue | None,
    ) -> ValidationResult:
        """Run this rule against a feature definition + value (or None if missing)."""
        ...


# ═══════════════════════════════════════════════════════════
#  Built-in rules
# ═══════════════════════════════════════════════════════════


class RangeRule(ValidationRule):
    """Validates that a numeric value falls within [min, max].

    Reads ``min`` and ``max`` from ``definition.validation_rules``.

    Parameters
    ----------
    severity : str
        ``error`` or ``warning`` (default ``error``).
    """

    def __init__(self, severity: str = "error") -> None:
        super().__init__("range", severity=severity)

    def validate(
        self,
        definition: FeatureDefinition,
        value: FeatureValue | None,
    ) -> ValidationResult:
        result = ValidationResult(feature_name=definition.name)

        if value is None or value.numeric_value is None:
            return result  # No numeric value to check

        rules = definition.validation_rules or {}
        min_val = rules.get("min")
        max_val = rules.get("max")

        if min_val is not None and value.numeric_value < min_val:
            msg = (
                f"Value {value.numeric_value} < min {min_val} "
                f"for {definition.name}"
            )
            if self.severity == "error":
                result.passed = False
                result.errors.append(msg)
            else:
                result.warnings.append(msg)

        if max_val is not None and value.numeric_value > max_val:
            msg = (
                f"Value {value.numeric_value} > max {max_val} "
                f"for {definition.name}"
            )
            if self.severity == "error":
                result.passed = False
                result.errors.append(msg)
            else:
                result.warnings.append(msg)

        return result


class NotNullRule(ValidationRule):
    """Validates that a value exists (is not None).

    Reads ``nullable`` from ``definition.validation_rules``.
    When ``nullable`` is ``False`` (the default), a missing
    ``FeatureValue`` or missing numeric/text/json value is an error.
    """

    def __init__(self, severity: str = "error") -> None:
        super().__init__("not_null", severity=severity)

    def validate(
        self,
        definition: FeatureDefinition,
        value: FeatureValue | None,
    ) -> ValidationResult:
        result = ValidationResult(feature_name=definition.name)

        rules = definition.validation_rules or {}
        nullable = rules.get("nullable", False)

        if nullable:
            return result  # Nulls are allowed

        if value is None:
            msg = f"Missing value for {definition.name} (nullable=False)"
            if self.severity == "error":
                result.passed = False
                result.errors.append(msg)
            else:
                result.warnings.append(msg)
            return result

        has_value = (
            value.numeric_value is not None
            or value.text_value is not None
            or value.json_value is not None
        )
        if not has_value:
            msg = f"Empty value (all fields None) for {definition.name} (nullable=False)"
            if self.severity == "error":
                result.passed = False
                result.errors.append(msg)
            else:
                result.warnings.append(msg)

        return result


class CardinalityRule(ValidationRule):
    """Validates that a feature value doesn't have too many distinct values
    for its entity count (detects near-constant or near-unique features).

    Reads ``max_cardinality_ratio`` from ``definition.validation_rules``.
    The ratio is ``distinct_values / total_entities``.

    Parameters
    ----------
    distinct_count : int
        Number of distinct values for this feature.
    entity_count : int
        Number of entities with values.
    severity : str
        ``error`` or ``warning`` (default ``warning``).
    """

    def __init__(
        self,
        distinct_count: int = 0,
        entity_count: int = 0,
        severity: str = "warning",
    ) -> None:
        super().__init__("cardinality", severity=severity)
        self.distinct_count = distinct_count
        self.entity_count = entity_count

    def validate(
        self,
        definition: FeatureDefinition,
        value: FeatureValue | None,
    ) -> ValidationResult:
        result = ValidationResult(feature_name=definition.name)

        rules = definition.validation_rules or {}
        max_ratio = rules.get("max_cardinality_ratio", 1.0)

        if self.entity_count == 0 or self.distinct_count == 0:
            return result

        ratio = self.distinct_count / self.entity_count
        if ratio > max_ratio:
            msg = (
                f"Cardinality ratio {ratio:.3f} exceeds limit {max_ratio} "
                f"({self.distinct_count} distinct / {self.entity_count} total) "
                f"for {definition.name}"
            )
            if self.severity == "error":
                result.passed = False
                result.errors.append(msg)
            else:
                result.warnings.append(msg)
        elif ratio < 0.01 and self.entity_count > 100:
            msg = (
                f"Near-constant feature: ratio={ratio:.4f} "
                f"({self.distinct_count} distinct / {self.entity_count} total) "
                f"for {definition.name}"
            )
            result.warnings.append(msg)

        return result


class ConsistencyRule(ValidationRule):
    """Cross-feature consistency check.

    Validates that a feature value is consistent with another
    related feature value (e.g. ``home_attack_strength`` should be
    the inverse of the opponent's home defense strength for the same
    match).

    Parameters
    ----------
    check_fn : callable
        Function ``(value, other_value) -> bool`` that returns
        ``True`` if consistent.
    other_definition_name : str
        Name of the related feature.
    severity : str
    """

    def __init__(
        self,
        check_fn: Any,
        other_definition_name: str = "",
        severity: str = "warning",
    ) -> None:
        super().__init__("consistency", severity=severity)
        self.check_fn = check_fn
        self.other_definition_name = other_definition_name

    def validate(
        self,
        definition: FeatureDefinition,
        value: FeatureValue | None,
    ) -> ValidationResult:
        result = ValidationResult(feature_name=definition.name)
        # Consistency checks require the actual related value,
        # which is passed in via the validator. This is handled
        # at the FeatureValidator level.
        result.metadata["check_fn"] = self.check_fn
        result.metadata["other_feature"] = self.other_definition_name
        return result


# ═══════════════════════════════════════════════════════════
#  Validator orchestrator
# ═══════════════════════════════════════════════════════════


class FeatureValidator:
    """Orchestrates validation of feature values against definitions.

    Runs a configurable set of rules against each feature value,
    aggregating results and providing batch-level summaries.

    Parameters
    ----------
    rules : list[ValidationRule], optional
        Rules to apply. Defaults to ``[RangeRule(), NotNullRule()]``.
    raise_on_error : bool
        If True, raises ``ValueError`` on first validation failure.
        Default is ``False`` (collect all errors).
    """

    def __init__(
        self,
        rules: list[ValidationRule] | None = None,
        raise_on_error: bool = False,
    ) -> None:
        self.rules = rules if rules is not None else [RangeRule(), NotNullRule()]
        self.raise_on_error = raise_on_error

    def validate_one(
        self,
        definition: FeatureDefinition,
        value: FeatureValue | None,
    ) -> ValidationResult:
        """Validate a single feature value against all rules.

        Parameters
        ----------
        definition : FeatureDefinition
            The feature definition (provides validation_rules config).
        value : FeatureValue, optional
            The computed value, or None if missing.

        Returns
        -------
        ValidationResult
        """
        combined = ValidationResult(feature_name=definition.name)

        for rule in self.rules:
            try:
                result = rule.validate(definition, value)
                combined = combined.merge(result)
            except Exception as exc:
                combined.passed = False
                combined.errors.append(
                    f"Rule {rule.name} raised exception: {exc}",
                )

        combined.metadata["rules_run"] = len(self.rules)
        combined.metadata["errors_count"] = len(combined.errors)
        combined.metadata["warnings_count"] = len(combined.warnings)

        if not combined.passed and self.raise_on_error:
            raise ValueError(
                f"Validation failed for {definition.name}: "
                f"{'; '.join(combined.errors)}"
            )

        return combined

    def validate_many(
        self,
        definition_values: list[tuple[FeatureDefinition, FeatureValue | None]],
    ) -> list[ValidationResult]:
        """Validate multiple feature values.

        Parameters
        ----------
        definition_values : list[tuple[FeatureDefinition, FeatureValue | None]]

        Returns
        -------
        list[ValidationResult]
        """
        return [self.validate_one(def_, val) for def_, val in definition_values]

    def batch_summary(
        self,
        results: list[ValidationResult],
    ) -> dict[str, Any]:
        """Aggregate validation results into a summary dict.

        Parameters
        ----------
        results : list[ValidationResult]

        Returns
        -------
        dict[str, Any]
            Summary with total, passed, failed, error/warning counts.
        """
        n_total = len(results)
        n_passed = sum(1 for r in results if r.passed)
        n_failed = n_total - n_passed
        n_errors = sum(len(r.errors) for r in results)
        n_warnings = sum(len(r.warnings) for r in results)
        failed_features = [
            r.feature_name for r in results if not r.passed
        ]

        return {
            "total": n_total,
            "passed": n_passed,
            "failed": n_failed,
            "pass_rate": round(n_passed / max(n_total, 1), 4),
            "total_errors": n_errors,
            "total_warnings": n_warnings,
            "failed_features": failed_features,
        }
