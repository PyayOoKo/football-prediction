"""Tests for the FeatureValidator."""

from __future__ import annotations

import pytest

from src.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    FeatureStatus,
    FeatureValue,
    EntityType,
)
from src.feature_store.validation import (
    CardinalityRule,
    ConsistencyRule,
    FeatureValidator,
    NotNullRule,
    RangeRule,
    ValidationResult,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_defaults(self) -> None:
        r = ValidationResult()
        assert r.passed is True
        assert r.errors == []
        assert r.warnings == []

    def test_merge_both_pass(self) -> None:
        a = ValidationResult(passed=True)
        b = ValidationResult(passed=True)
        merged = a.merge(b)
        assert merged.passed is True

    def test_merge_one_fails(self) -> None:
        a = ValidationResult(passed=True)
        b = ValidationResult(passed=False, errors=["error 1"])
        merged = a.merge(b)
        assert merged.passed is False
        assert merged.errors == ["error 1"]

    def test_merge_combines_errors(self) -> None:
        a = ValidationResult(passed=False, errors=["error a"])
        b = ValidationResult(passed=False, errors=["error b"])
        merged = a.merge(b)
        assert merged.errors == ["error a", "error b"]


class TestRangeRule:
    """Test RangeRule validation."""

    def test_value_within_range(self) -> None:
        rule = RangeRule()
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0, "max": 1.0})
        value = FeatureValue(numeric_value=0.5)
        result = rule.validate(defn, value)
        assert result.passed is True

    def test_value_below_min(self) -> None:
        rule = RangeRule()
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0})
        value = FeatureValue(numeric_value=-1.0)
        result = rule.validate(defn, value)
        assert result.passed is False
        assert len(result.errors) == 1
        assert "min" in result.errors[0]

    def test_value_above_max(self) -> None:
        rule = RangeRule()
        defn = FeatureDefinition(name="test", validation_rules={"max": 100.0})
        value = FeatureValue(numeric_value=150.0)
        result = rule.validate(defn, value)
        assert result.passed is False
        assert "max" in result.errors[0]

    def test_no_rules_defined(self) -> None:
        rule = RangeRule()
        defn = FeatureDefinition(name="test")
        value = FeatureValue(numeric_value=999.0)
        result = rule.validate(defn, value)
        assert result.passed is True

    def test_missing_value(self) -> None:
        rule = RangeRule()
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0})
        result = rule.validate(defn, None)
        assert result.passed is True  # Nothing to check

    def test_warning_severity(self) -> None:
        rule = RangeRule(severity="warning")
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0})
        value = FeatureValue(numeric_value=-5.0)
        result = rule.validate(defn, value)
        assert result.passed is True  # Warning doesn't fail
        assert len(result.warnings) == 1


class TestNotNullRule:
    """Test NotNullRule validation."""

    def test_nullable_allowed(self) -> None:
        rule = NotNullRule()
        defn = FeatureDefinition(name="test", validation_rules={"nullable": True})
        result = rule.validate(defn, None)
        assert result.passed is True

    def test_nullable_disallowed_missing_value(self) -> None:
        rule = NotNullRule()
        defn = FeatureDefinition(name="test", validation_rules={"nullable": False})
        result = rule.validate(defn, None)
        assert result.passed is False
        assert "Missing" in result.errors[0]

    def test_nullable_disallowed_empty_value(self) -> None:
        rule = NotNullRule()
        defn = FeatureDefinition(name="test", validation_rules={"nullable": False})
        value = FeatureValue()  # All value fields are None
        result = rule.validate(defn, value)
        assert result.passed is False
        assert "Empty" in result.errors[0]

    def test_nullable_disallowed_with_value(self) -> None:
        rule = NotNullRule()
        defn = FeatureDefinition(name="test", validation_rules={"nullable": False})
        value = FeatureValue(numeric_value=42.0)
        result = rule.validate(defn, value)
        assert result.passed is True

    def test_default_not_nullable(self) -> None:
        """Default is nullable=False."""
        rule = NotNullRule()
        defn = FeatureDefinition(name="test")
        result = rule.validate(defn, None)
        assert result.passed is False


class TestCardinalityRule:
    """Test CardinalityRule validation."""

    def test_within_limit(self) -> None:
        rule = CardinalityRule(distinct_count=10, entity_count=100)
        defn = FeatureDefinition(name="test", validation_rules={"max_cardinality_ratio": 0.5})
        result = rule.validate(defn, None)
        assert result.passed is True

    def test_exceeds_limit(self) -> None:
        rule = CardinalityRule(distinct_count=80, entity_count=100, severity="error")
        defn = FeatureDefinition(name="test", validation_rules={"max_cardinality_ratio": 0.5})
        result = rule.validate(defn, None)
        assert result.passed is False
        assert "0.800" in result.errors[0]

    def test_near_constant_warning(self) -> None:
        rule = CardinalityRule(distinct_count=2, entity_count=1000)
        defn = FeatureDefinition(name="test")
        result = rule.validate(defn, None)
        assert result.passed is True  # Warnings don't fail
        assert len(result.warnings) == 1
        assert "Near-constant" in result.warnings[0]

    def test_zero_entity_count(self) -> None:
        rule = CardinalityRule(distinct_count=0, entity_count=0)
        defn = FeatureDefinition(name="test")
        result = rule.validate(defn, None)
        assert result.passed is True


class TestConsistencyRule:
    """Test ConsistencyRule validation."""

    def test_metadata(self) -> None:
        def check_fn(a, b): return a == b
        rule = ConsistencyRule(check_fn=check_fn, other_definition_name="other_feature")
        defn = FeatureDefinition(name="test")
        result = rule.validate(defn, None)
        assert result.passed is True
        assert result.metadata["check_fn"] is check_fn
        assert result.metadata["other_feature"] == "other_feature"


class TestFeatureValidator:
    """Test the FeatureValidator orchestrator."""

    def test_empty_rules(self) -> None:
        validator = FeatureValidator(rules=[])
        defn = FeatureDefinition(name="test")
        result = validator.validate_one(defn, None)
        assert result.passed is True

    def test_single_rule_pass(self) -> None:
        validator = FeatureValidator(rules=[RangeRule()])
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0, "max": 1.0})
        value = FeatureValue(numeric_value=0.5)
        result = validator.validate_one(defn, value)
        assert result.passed is True

    def test_single_rule_fail(self) -> None:
        validator = FeatureValidator(rules=[RangeRule()])
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0})
        value = FeatureValue(numeric_value=-1.0)
        result = validator.validate_one(defn, value)
        assert result.passed is False

    def test_multiple_rules(self) -> None:
        validator = FeatureValidator(rules=[RangeRule(), NotNullRule()])
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0, "nullable": False})
        result = validator.validate_one(defn, None)
        assert result.passed is False  # NotNullRule fails (None value)
        assert len(result.errors) >= 1

    def test_validate_many(self) -> None:
        validator = FeatureValidator()
        def1 = FeatureDefinition(name="a", validation_rules={"min": 0.0})
        def2 = FeatureDefinition(name="b", validation_rules={"min": 0.0})
        val1 = FeatureValue(numeric_value=1.0)
        val2 = FeatureValue(numeric_value=-1.0)
        results = validator.validate_many([(def1, val1), (def2, val2)])
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    def test_raise_on_error(self) -> None:
        validator = FeatureValidator(raise_on_error=True)
        defn = FeatureDefinition(name="test", validation_rules={"min": 0.0})
        value = FeatureValue(numeric_value=-1.0)
        with pytest.raises(ValueError, match="Validation failed"):
            validator.validate_one(defn, value)

    def test_rule_exception_handling(self) -> None:
        class BrokenRule(RangeRule):
            def validate(self, defn, value):
                raise RuntimeError("Broken!")
        validator = FeatureValidator(rules=[BrokenRule()])
        defn = FeatureDefinition(name="test")
        result = validator.validate_one(defn, None)
        assert result.passed is False
        assert "Broken" in result.errors[0]

    def test_batch_summary_all_pass(self) -> None:
        validator = FeatureValidator()
        results = [
            ValidationResult(feature_name="a", passed=True),
            ValidationResult(feature_name="b", passed=True),
        ]
        summary = validator.batch_summary(results)
        assert summary["total"] == 2
        assert summary["passed"] == 2
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 1.0

    def test_batch_summary_some_fail(self) -> None:
        validator = FeatureValidator()
        results = [
            ValidationResult(feature_name="a", passed=True),
            ValidationResult(feature_name="b", passed=False, errors=["bad value"]),
        ]
        summary = validator.batch_summary(results)
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["failed_features"] == ["b"]
