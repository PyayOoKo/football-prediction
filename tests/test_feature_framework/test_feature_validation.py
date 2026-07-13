"""
Tests for the Feature Validation Framework — FeatureValidator, checks, reports.

Covers all 10 detection checks, 5 report types, edge cases, and
pipeline integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_framework.validation import FeatureValidator
from src.feature_framework.validation.checks import (
    check_constant_features,
    check_data_leakage,
    check_duplicate_features,
    check_feature_drift,
    check_highly_correlated,
    check_infinite_values,
    check_invalid_ranges,
    check_low_variance,
    check_missing_values,
    check_nan_values,
)
from src.feature_framework.validation.report import (
    CorrelationReport,
    DriftReport,
    FeatureImportancePlaceholder,
    MissingValueReport,
    ValidationReport,
)


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def clean_features() -> pd.DataFrame:
    """Well-formed feature DataFrame with no issues."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "h_overall_points_avg5": rng.uniform(0.5, 2.5, n),
        "a_overall_points_avg5": rng.uniform(0.3, 2.0, n),
        "h_overall_goals_scored_avg5": rng.uniform(0.5, 3.0, n),
        "a_overall_goals_conceded_avg5": rng.uniform(0.3, 2.5, n),
        "elo_diff": rng.normal(0, 100, n),
        "home_adv": rng.uniform(-0.5, 1.5, n),
        "odds_home_opening": rng.uniform(1.2, 10.0, n),
        "fair_prob_home_closing": rng.uniform(0.05, 0.95, n),
        "clv_home": rng.normal(0, 0.5, n),
    })


@pytest.fixture
def dirty_features() -> pd.DataFrame:
    """Feature DataFrame with known issues."""
    rng = np.random.RandomState(1)
    n = 50
    dup_vals = rng.uniform(0, 1, n)
    return pd.DataFrame({
        "date": pd.date_range("2024-06-01", periods=n, freq="D"),
        "constant_feat": np.ones(n),  # constant (1 unique)
        "has_nan": np.concatenate([rng.uniform(0, 1, 40), [np.nan] * 10]),  # 20% NaN
        "has_inf": np.concatenate([rng.uniform(0, 1, 48), [np.inf, -np.inf]]),  # Inf
        "low_var": rng.uniform(1.0, 1.001, n),  # low variance
        "duplicate_a": dup_vals.copy(),  # will have identical copy
        "duplicate_b": dup_vals.copy(),  # identical to duplicate_a
        "odds_col": rng.uniform(1.5, 15.0, n),
    })
    return dirty_features


@pytest.fixture
def reference_features() -> pd.DataFrame:
    """Reference dataset for drift detection.

    Uses the same random state as ``clean_features`` for the common
    columns to ensure distributions match.
    """
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "h_overall_points_avg5": rng.uniform(0.5, 2.5, n),
        "a_overall_points_avg5": rng.uniform(0.3, 2.0, n),
        "elo_diff": rng.normal(0, 100, n),
        "fair_prob_home_closing": rng.uniform(0.05, 0.95, n),
    })


@pytest.fixture
def drifted_features() -> pd.DataFrame:
    """Feature DataFrame with distribution drift."""
    rng = np.random.RandomState(99)
    n = 100
    df = pd.DataFrame({
        "h_overall_points_avg5": rng.uniform(1.5, 3.5, n),  # shifted up from [0.5, 2.5]
        "a_overall_points_avg5": rng.uniform(0.3, 2.0, n),
        "elo_diff": rng.normal(50, 150, n),  # shifted mean + higher std
        "fair_prob_home_closing": rng.uniform(0.05, 0.95, n),
    })
    return df


@pytest.fixture
def unsorted_features() -> pd.DataFrame:
    """Feature DataFrame with dates out of order."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"]),
        "h_points_avg5": [1.5, 2.0, 1.8],
        "a_points_avg5": [1.0, 1.2, 1.1],
    })
    return df


# ═══════════════════════════════════════════════════════════════
#  Tests: FeatureValidator Orchestration
# ═══════════════════════════════════════════════════════════════


class TestFeatureValidatorOrchestration:
    def test_validate_clean(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(clean_features)
        assert isinstance(report, ValidationReport)
        assert report.passed
        assert report.total_checks > 0

    def test_validate_dirty(self, dirty_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(dirty_features)
        assert not report.passed
        assert report.failed_checks > 0
        assert report.total_violations > 0

    def test_validate_empty(self):
        validator = FeatureValidator(verbose=False)
        df = pd.DataFrame()
        report = validator.validate(df)
        # Most checks should pass gracefully on empty DF
        assert isinstance(report, ValidationReport)

    def test_validate_single_column(self):
        validator = FeatureValidator(verbose=False)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        report = validator.validate(df)
        assert isinstance(report, ValidationReport)

    def test_validate_with_reference(self, clean_features, reference_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(clean_features, reference_df=reference_features)
        assert isinstance(report, ValidationReport)

    def test_validate_all_checks_present(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(clean_features)
        expected_checks = {
            "data_leakage", "constant_features", "highly_correlated",
            "missing_values", "invalid_ranges", "infinite_values",
            "nan_values", "duplicate_features", "low_variance",
            "feature_drift",
        }
        assert expected_checks.issubset(set(report.checks.keys()))

    def test_validate_for_pipeline(self, clean_features):
        validator = FeatureValidator(verbose=False)
        result = validator.validate_for_pipeline(clean_features, step_name="test_step")
        assert result["step"] == "test_step"
        assert "passed" in result
        assert "summary" in result

    def test_last_report_property(self, clean_features):
        validator = FeatureValidator(verbose=False)
        assert validator.last_report is None
        validator.validate(clean_features)
        assert validator.last_report is not None

    def test_repr(self):
        validator = FeatureValidator()
        r = repr(validator)
        assert "FeatureValidator" in r
        assert "10 checks" in r


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 1 — Data Leakage
# ═══════════════════════════════════════════════════════════════


class TestCheckDataLeakage:
    def test_sorted_passes(self, clean_features):
        result = check_data_leakage(clean_features)
        assert result["passed"]

    def test_unsorted_fails(self, unsorted_features):
        result = check_data_leakage(unsorted_features)
        assert not result["passed"]
        assert len(result["violations"]) > 0

    def test_no_date_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = check_data_leakage(df)
        assert result["passed"]  # Graceful skip

    def test_empty_dataframe(self):
        df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        result = check_data_leakage(df)
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 2 — Constant Features
# ═══════════════════════════════════════════════════════════════


class TestCheckConstantFeatures:
    def test_clean_passes(self, clean_features):
        result = check_constant_features(clean_features)
        assert result["passed"]

    def test_constant_detected(self, dirty_features):
        result = check_constant_features(dirty_features)
        assert not result["passed"]
        violations = result["violations"]
        cols_found = [v["column"] for v in violations]
        assert "constant_feat" in cols_found

    def test_near_constant_detected(self):
        """A near-constant column with very few unique values."""
        df = pd.DataFrame({
            "a": np.concatenate([np.ones(90), np.ones(10) * 1.001]),  # 2 unique / 100
            "b": np.random.uniform(0, 1, 100),
        })
        result = check_constant_features(df, min_unique_ratio=0.03)  # 2/100 = 2% < 3% → fails
        assert not result["passed"]
        cols_found = [v["column"] for v in result["violations"]]
        assert "a" in cols_found

    def test_empty(self):
        result = check_constant_features(pd.DataFrame())
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 3 — Highly Correlated Features
# ═══════════════════════════════════════════════════════════════


class TestCheckHighlyCorrelated:
    def test_clean_passes(self, clean_features):
        result = check_highly_correlated(clean_features)
        assert result["passed"]

    def test_perfect_correlation_detected(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],  # a * 2
            "c": np.random.uniform(0, 1, 5),
        })
        result = check_highly_correlated(df, correlation_threshold=0.95)
        assert not result["passed"]
        assert len(result["violations"]) >= 1

    def test_single_column(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = check_highly_correlated(df)
        assert result["passed"]

    def test_empty(self):
        result = check_highly_correlated(pd.DataFrame())
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 4 — Missing Values
# ═══════════════════════════════════════════════════════════════


class TestCheckMissingValues:
    def test_clean_passes(self, clean_features):
        result = check_missing_values(clean_features)
        assert result["passed"]

    def test_missing_detected(self, dirty_features):
        result = check_missing_values(dirty_features)
        assert not result["passed"]
        cols_found = [v["column"] for v in result["violations"]]
        assert "has_nan" in cols_found

    def test_empty(self):
        result = check_missing_values(pd.DataFrame())
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 5 — Invalid Ranges
# ═══════════════════════════════════════════════════════════════


class TestCheckInvalidRanges:
    def test_clean_passes(self, clean_features):
        result = check_invalid_ranges(clean_features, range_bounds={})
        assert result["passed"]

    def test_out_of_range_detected(self):
        df = pd.DataFrame({
            "probability": [0.5, 1.2, -0.1, 0.8],
            "odds": [2.0, 0.5, 3.0, 1.5],
        })
        result = check_invalid_ranges(df, range_bounds={
            "probability": (0.0, 1.0),
            "odds": (1.0, 100.0),
        })
        assert not result["passed"]
        assert len(result["violations"]) >= 2  # 1 above + 1 below for prob, 1 below for odds

    def test_wildcard_pattern(self):
        df = pd.DataFrame({
            "odds_home": [2.0, 0.8, 3.0],
            "odds_away": [1.5, 2.5, 4.0],
            "other_col": [10, 20, 30],
        })
        result = check_invalid_ranges(df, range_bounds={
            "odds_*": (1.0, 100.0),
        })
        assert not result["passed"]

    def test_empty_bounds(self):
        result = check_invalid_ranges(pd.DataFrame(), range_bounds={})
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 6 — Infinite Values
# ═══════════════════════════════════════════════════════════════


class TestCheckInfiniteValues:
    def test_clean_passes(self, clean_features):
        result = check_infinite_values(clean_features)
        assert result["passed"]

    def test_inf_detected(self, dirty_features):
        result = check_infinite_values(dirty_features)
        assert not result["passed"]
        cols_found = [v["column"] for v in result["violations"]]
        assert "has_inf" in cols_found

    def test_empty(self):
        result = check_infinite_values(pd.DataFrame())
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 7 — NaN Values
# ═══════════════════════════════════════════════════════════════


class TestCheckNanValues:
    def test_clean_passes(self, clean_features):
        result = check_nan_values(clean_features)
        assert result["passed"]

    def test_nan_detected(self, dirty_features):
        result = check_nan_values(dirty_features)
        assert not result["passed"]
        cols_found = [v["column"] for v in result["violations"]]
        assert "has_nan" in cols_found

    def test_empty(self):
        result = check_nan_values(pd.DataFrame())
        assert result["passed"]

    def test_string_column_ignored(self):
        df = pd.DataFrame({"text": ["hello", None, "world"]})
        result = check_nan_values(df)
        assert result["passed"]  # Non-numeric columns are ignored


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 8 — Duplicate Features
# ═══════════════════════════════════════════════════════════════


class TestCheckDuplicateFeatures:
    def test_clean_passes(self, clean_features):
        result = check_duplicate_features(clean_features)
        assert result["passed"]

    def test_identical_detected(self, dirty_features):
        result = check_duplicate_features(dirty_features)
        assert not result["passed"]
        violations = result["violations"]
        cols = [(v["feature_1"], v["feature_2"]) for v in violations]
        assert ("duplicate_a", "duplicate_b") in cols

    def test_empty(self):
        result = check_duplicate_features(pd.DataFrame())
        assert result["passed"]

    def test_single_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = check_duplicate_features(df)
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 9 — Low Variance Features
# ═══════════════════════════════════════════════════════════════


class TestCheckLowVariance:
    def test_clean_passes(self, clean_features):
        result = check_low_variance(clean_features)
        assert result["passed"]

    def test_low_variance_detected(self, dirty_features):
        result = check_low_variance(dirty_features, variance_threshold=0.01)
        assert not result["passed"]
        cols_found = [v["column"] for v in result["violations"]]
        assert "low_var" in cols_found

    def test_empty(self):
        result = check_low_variance(pd.DataFrame())
        assert result["passed"]

    def test_string_column_ignored(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = check_low_variance(df)
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Check 10 — Feature Drift
# ═══════════════════════════════════════════════════════════════


class TestCheckFeatureDrift:
    def test_no_reference_passes(self, clean_features):
        result = check_feature_drift(clean_features)
        assert result["passed"]

    def test_similar_data_passes(self):
        """Two datasets with same distribution should pass drift check."""
        rng = np.random.RandomState(42)
        curr = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        result = check_feature_drift(curr, reference_df=ref, drift_threshold=0.2)
        assert result["passed"]

    def test_drift_detected(self, drifted_features, reference_features):
        result = check_feature_drift(
            drifted_features,
            reference_df=reference_features,
            drift_threshold=0.1,
        )
        assert not result["passed"]
        assert len(result["violations"]) > 0

    def test_empty_data(self):
        result = check_feature_drift(pd.DataFrame(), reference_df=pd.DataFrame({"a": [1, 2]}))
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════
#  Tests: Correlation Matrix Report
# ═══════════════════════════════════════════════════════════════


class TestCorrelationMatrix:
    def test_correlation_matrix(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.correlation_matrix(clean_features)
        assert isinstance(report, CorrelationReport)
        assert report.n_features > 0

    def test_single_column(self):
        validator = FeatureValidator(verbose=False)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        report = validator.correlation_matrix(df)
        assert isinstance(report, CorrelationReport)

    def test_summary(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.correlation_matrix(clean_features)
        assert "CORRELATION" in report.summary()

    def test_to_dict(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.correlation_matrix(clean_features)
        d = report.to_dict()
        assert "n_features" in d
        assert "n_high_pairs" in d


# ═══════════════════════════════════════════════════════════════
#  Tests: Missing Value Report
# ═══════════════════════════════════════════════════════════════


class TestMissingValueReport:
    def test_missing_value_report(self, dirty_features):
        validator = FeatureValidator(verbose=False)
        report = validator.missing_value_report(dirty_features)
        assert isinstance(report, MissingValueReport)
        assert report.n_missing_cells > 0

    def test_clean_data(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.missing_value_report(clean_features)
        assert isinstance(report, MissingValueReport)

    def test_summary(self, dirty_features):
        validator = FeatureValidator(verbose=False)
        report = validator.missing_value_report(dirty_features)
        assert "MISSING" in report.summary()

    def test_to_dataframe(self, dirty_features):
        validator = FeatureValidator(verbose=False)
        report = validator.missing_value_report(dirty_features)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ═══════════════════════════════════════════════════════════════
#  Tests: Drift Report
# ═══════════════════════════════════════════════════════════════


class TestDriftReport:
    def test_drift_report_no_reference(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.drift_report(clean_features, reference_df=None)
        assert isinstance(report, DriftReport)

    def test_drift_report_similar(self):
        """Two datasets with same distribution should produce passing DriftReport."""
        rng = np.random.RandomState(42)
        curr = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        validator = FeatureValidator(verbose=False, drift_threshold=0.2)
        report = validator.drift_report(curr, ref)
        assert isinstance(report, DriftReport)
        assert report.passed

    def test_drift_report_drifted(self, drifted_features, reference_features):
        validator = FeatureValidator(verbose=False)
        report = validator.drift_report(drifted_features, reference_features)
        assert not report.passed
        assert report.n_drifted > 0

    def test_summary(self, drifted_features, reference_features):
        validator = FeatureValidator(verbose=False)
        report = validator.drift_report(drifted_features, reference_features)
        assert "DRIFT" in report.summary()


# ═══════════════════════════════════════════════════════════════
#  Tests: Feature Importance Placeholder
# ═══════════════════════════════════════════════════════════════


class TestFeatureImportancePlaceholder:
    def test_placeholder_creation(self):
        validator = FeatureValidator(verbose=False)
        placeholder = validator.feature_importance_placeholder(
            feature_names=["a", "b", "c"],
            model_type="xgboost",
        )
        assert isinstance(placeholder, FeatureImportancePlaceholder)
        assert placeholder.n_features == 3
        assert placeholder.model_type == "xgboost"

    def test_summary(self):
        validator = FeatureValidator(verbose=False)
        placeholder = validator.feature_importance_placeholder(
            feature_names=["elo_diff", "home_adv"],
            model_type="logistic_regression",
        )
        assert "FEATURE IMPORTANCE" in placeholder.summary()

    def test_to_dict(self):
        validator = FeatureValidator(verbose=False)
        placeholder = validator.feature_importance_placeholder(
            feature_names=["x"],
        )
        d = placeholder.to_dict()
        assert "n_features" in d
        assert "feature_names" in d


# ═══════════════════════════════════════════════════════════════
#  Tests: Validation Report
# ═══════════════════════════════════════════════════════════════


class TestValidationReport:
    def test_report_properties(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(clean_features)
        assert report.passed
        assert report.passed_checks > 0
        assert report.total_checks == report.passed_checks + report.failed_checks

    def test_summary(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(clean_features)
        assert "FEATURE VALIDATION" in report.summary()

    def test_to_dict(self, clean_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(clean_features)
        d = report.to_dict()
        assert "passed" in d
        assert "n_rows" in d
        assert "check_details" in d

    def test_violations_dataframe(self, dirty_features):
        validator = FeatureValidator(verbose=False)
        report = validator.validate(dirty_features)
        df = report.violations_dataframe
        assert isinstance(df, pd.DataFrame)
        assert "check" in df.columns

    def test_empty_report(self):
        report = ValidationReport()
        # An empty report has no failed checks, so passed=True
        assert report.passed
        assert report.passed_checks == 0
        assert report.failed_checks == 0


# ═══════════════════════════════════════════════════════════════
#  Tests: PSI Computation
# ═══════════════════════════════════════════════════════════════


class TestPSIComputation:
    def test_identical_distributions(self):
        validator = FeatureValidator(verbose=False)
        a = np.random.normal(0, 1, 1000)
        psi = validator._compute_psi(a, a)
        assert psi < 0.01  # Near-zero for identical distributions

    def test_different_distributions(self):
        validator = FeatureValidator(verbose=False)
        a = np.random.normal(0, 1, 1000)
        b = np.random.normal(5, 1, 1000)  # Shifted
        psi = validator._compute_psi(a, b)
        assert psi > 0.1  # Meaningful drift

    def test_small_samples(self):
        validator = FeatureValidator(verbose=False)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5])
        psi = validator._compute_psi(a, b, n_bins=10)
        assert psi == 0.0  # Too few samples, returns 0


# ═══════════════════════════════════════════════════════════════
#  Tests: Custom Configuration
# ═══════════════════════════════════════════════════════════════


class TestCustomConfiguration:
    def test_selected_checks(self, clean_features):
        validator = FeatureValidator(
            checks=["constant_features", "missing_values"],
            verbose=False,
        )
        report = validator.validate(clean_features)
        assert report.total_checks == 2
        assert "constant_features" in report.checks
        assert "missing_values" in report.checks

    def test_custom_thresholds(self, dirty_features):
        validator = FeatureValidator(
            variance_threshold=0.5,  # Very high — more features will be flagged
            verbose=False,
        )
        report = validator.validate(dirty_features)
        # More low-variance features should be caught
        low_var_check = report.checks.get("low_variance", {})
        assert len(low_var_check.get("violations", [])) > 0

    def test_custom_range_bounds(self, clean_features):
        validator = FeatureValidator(
            range_bounds={"fair_prob_*": (0.0, 1.0)},
            verbose=False,
        )
        report = validator.validate(clean_features)
        # fair_prob_home_closing should pass range check
        range_check = report.checks.get("invalid_ranges", {})
        assert range_check.get("passed", True)


# ═══════════════════════════════════════════════════════════════
#  Tests: Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_all_nan_columns(self):
        df = pd.DataFrame({
            "a": [np.nan, np.nan, np.nan],
            "b": [1.0, 2.0, 3.0],
        })
        validator = FeatureValidator(verbose=False)
        report = validator.validate(df)
        # NaN check should catch col 'a'
        assert not report.passed

    def test_all_constant(self):
        df = pd.DataFrame({
            "a": [1.0, 1.0, 1.0],
            "b": [2.0, 2.0, 2.0],
        })
        validator = FeatureValidator(verbose=False)
        report = validator.validate(df)
        # Constant check should flag both
        const_check = report.checks.get("constant_features", {})
        assert len(const_check.get("violations", [])) >= 2

    def test_mixed_types(self):
        df = pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0],
            "text": ["a", "b", "c"],
            "bool": [True, False, True],
        })
        validator = FeatureValidator(verbose=False)
        report = validator.validate(df)
        assert isinstance(report, ValidationReport)

    def test_large_dataframe(self):
        """Ensure performance is acceptable with many columns."""
        n = 1000
        cols = {f"feat_{i}": np.random.uniform(0, 1, n) for i in range(50)}
        cols["date"] = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(cols)
        validator = FeatureValidator(verbose=False)
        report = validator.validate(df)
        assert isinstance(report, ValidationReport)
