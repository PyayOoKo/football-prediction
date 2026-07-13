"""
Feature Validation Framework — production-grade validation for computed features.

Automatically detects data leakage, constant features, highly correlated
features, missing values, invalid ranges, infinite values, NaN values,
duplicate features, low variance features, and feature drift.

Generates validation reports, correlation matrices, missing value reports,
drift reports, and feature importance placeholders.

Integrates with :class:`~src.feature_framework.pipeline.FeaturePipeline`
so every pipeline run automatically validates its output.

Usage
-----
::

    from src.feature_framework.validation import FeatureValidator

    validator = FeatureValidator()

    # Run all checks on a feature DataFrame
    report = validator.validate(features_df, reference_df=training_df)

    # Generate reports
    print(report.summary())
    correlation = validator.correlation_matrix(features_df)
    missing = validator.missing_value_report(features_df)
    drift = validator.drift_report(features_df, reference_df)

    # Integration with pipeline
    report = validator.validate_for_pipeline(features_df, step_name="team_form")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

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
    compute_psi,
)
from src.feature_framework.validation.report import (
    CorrelationReport,
    DriftReport,
    FeatureImportancePlaceholder,
    MissingValueReport,
    ValidationReport,
)

logger = logging.getLogger(__name__)


_DEFAULT_CHECKS: list[str] = [
    "data_leakage",
    "constant_features",
    "highly_correlated",
    "missing_values",
    "invalid_ranges",
    "infinite_values",
    "nan_values",
    "duplicate_features",
    "low_variance",
    "feature_drift",
]

_CHECK_FUNCTIONS = {
    "data_leakage": check_data_leakage,
    "constant_features": check_constant_features,
    "highly_correlated": check_highly_correlated,
    "missing_values": check_missing_values,
    "invalid_ranges": check_invalid_ranges,
    "infinite_values": check_infinite_values,
    "nan_values": check_nan_values,
    "duplicate_features": check_duplicate_features,
    "low_variance": check_low_variance,
    "feature_drift": check_feature_drift,
}


class FeatureValidator:
    """Orchestrates all feature validation checks and report generation.

    Parameters
    ----------
    checks : list[str], optional
        Names of checks to run. Defaults to all 10 checks.
    correlation_threshold : float
        Threshold for high correlation detection (default 0.95).
    variance_threshold : float
        Threshold for low variance detection (default 0.01).
    drift_threshold : float
        Threshold for drift detection in PSI/KL divergence (default 0.1).
    min_unique_ratio : float
        Minimum ratio of unique values / total rows for constant check
        (default 0.01).
    range_bounds : dict[str, tuple[float, float]], optional
        Custom range bounds per column. E.g. ``{"odds_*": (1.0, 100.0)}``.
    date_column : str, optional
        Column name for temporal leakage check (default ``"date"``).
    verbose : bool
        Log check progress (default True).
    """

    def __init__(
        self,
        checks: list[str] | None = None,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
        drift_threshold: float = 0.1,
        min_unique_ratio: float = 0.01,
        range_bounds: dict[str, tuple[float, float]] | None = None,
        date_column: str = "date",
        verbose: bool = True,
    ) -> None:
        self.checks = checks or _DEFAULT_CHECKS
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.drift_threshold = drift_threshold
        self.min_unique_ratio = min_unique_ratio
        self.range_bounds = range_bounds or {}
        self.date_column = date_column
        self.verbose = verbose

        # Storage for computed reports
        self._last_report: ValidationReport | None = None
        self._last_correlation: CorrelationReport | None = None
        self._last_missing: MissingValueReport | None = None
        self._last_drift: DriftReport | None = None
        self._last_importance: FeatureImportancePlaceholder | None = None

    # ══════════════════════════════════════════════════════
    #  Main validation entry point
    # ══════════════════════════════════════════════════════

    def validate(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame | None = None,
        column_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> ValidationReport:
        """Run all configured checks on a feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix to validate.
        reference_df : pd.DataFrame, optional
            Reference data for drift detection (e.g. training set).
        column_bounds : dict[str, tuple[float, float]], optional
            Per-column valid range bounds. Supports ``*`` wildcard.
            E.g. ``{"odds_*": (1.0, 100.0), "prob_*": (0.0, 1.0)}``.

        Returns
        -------
        ValidationReport
            Aggregated results from all checks.
        """
        report = ValidationReport(
            n_rows=len(df),
            n_columns=len(df.columns),
            column_names=list(df.columns),
        )

        combined_bounds = {**self.range_bounds, **(column_bounds or {})}

        for check_name in self.checks:
            check_fn = _CHECK_FUNCTIONS.get(check_name)
            if check_fn is None:
                logger.warning("Unknown check: %s — skipping", check_name)
                continue

            try:
                result = check_fn(
                    df,
                    reference_df=reference_df,
                    correlation_threshold=self.correlation_threshold,
                    variance_threshold=self.variance_threshold,
                    drift_threshold=self.drift_threshold,
                    min_unique_ratio=self.min_unique_ratio,
                    range_bounds=combined_bounds,
                    date_column=self.date_column,
                )
                report.checks[check_name] = result
                report.total_checks += 1

                if result.get("passed", True):
                    report.passed_checks += 1
                else:
                    report.failed_checks += 1
                    report.total_violations += len(result.get("violations", []))

                if self.verbose:
                    status = "PASS" if result.get("passed", True) else f"FAIL ({len(result.get('violations', []))} violations)"
                    logger.debug("  [%s] %s", status, check_name)

            except Exception as exc:
                logger.exception("Check '%s' raised an exception: %s", check_name, exc)
                report.checks[check_name] = {
                    "check_name": check_name,
                    "passed": False,
                    "violations": [{"error": str(exc)}],
                }
                report.failed_checks += 1

        self._last_report = report

        if self.verbose:
            logger.info(
                "Validation complete: %d/%d checks passed, %d violations",
                report.passed_checks, report.total_checks, report.total_violations,
            )

        return report

    def validate_for_pipeline(
        self,
        df: pd.DataFrame,
        step_name: str = "unknown",
        reference_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run validation suitable for pipeline integration.

        Returns a dict that the pipeline can log or add to its report.
        """
        report = self.validate(df, reference_df=reference_df)
        return {
            "step": step_name,
            "passed": report.passed,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "total_violations": report.total_violations,
            "summary": report.summary(),
        }

    # ══════════════════════════════════════════════════════
    #  Specialised report generators
    # ══════════════════════════════════════════════════════

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        max_features: int = 50,
    ) -> CorrelationReport:
        """Compute correlation matrix for numeric features.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame.
        max_features : int
            Maximum features to include (default 50).

        Returns
        -------
        CorrelationReport
            Correlation matrix data and highly-correlated pairs.
        """
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            report = CorrelationReport()
            report.message = "Need at least 2 numeric columns"
            self._last_correlation = report
            return report

        # Select features (limit to max_features for performance)
        cols = numeric.columns[:max_features].tolist()
        corr = numeric[cols].corr(method="pearson")

        # Find highly correlated pairs
        threshold = self.correlation_threshold
        high_pairs: list[dict[str, Any]] = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if not pd.isna(val) and abs(val) >= threshold:
                    high_pairs.append({
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": round(float(val), 4),
                    })

        report = CorrelationReport(
            n_features=len(cols),
            correlation_matrix=corr,
            high_correlation_pairs=high_pairs,
            n_high_pairs=len(high_pairs),
        )
        self._last_correlation = report
        return report

    def missing_value_report(self, df: pd.DataFrame) -> MissingValueReport:
        """Generate a missing value analysis report.

        Returns
        -------
        MissingValueReport
            Per-column missing value counts and rates.
        """
        total = len(df)
        missing: list[dict[str, Any]] = []

        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            n_inf = int(np.isinf(df[col]).sum()) if pd.api.types.is_float_dtype(df[col]) else 0
            if n_missing > 0 or n_inf > 0:
                missing.append({
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "n_missing": n_missing,
                    "missing_rate": round(n_missing / total, 4) if total > 0 else 0.0,
                    "n_infinite": n_inf,
                })

        total_cells = total * len(df.columns)
        n_missing_cells = sum(m["n_missing"] for m in missing)
        n_inf_cells = sum(m["n_infinite"] for m in missing)

        report = MissingValueReport(
            total_rows=total,
            total_columns=len(df.columns),
            total_cells=total_cells,
            n_missing_cells=n_missing_cells,
            n_infinite_cells=n_inf_cells,
            missing_rate=round(n_missing_cells / total_cells, 4) if total_cells > 0 else 0.0,
            columns_with_missing=len(missing),
            details=missing,
        )
        self._last_missing = report
        return report

    def drift_report(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> DriftReport:
        """Compare feature distributions between current and reference data.

        Parameters
        ----------
        current_df : pd.DataFrame
            Current (new) feature data.
        reference_df : pd.DataFrame
            Reference (baseline) feature data (e.g. training set).

        Returns
        -------
        DriftReport
            Per-feature drift scores and warnings.
        """
        if reference_df is None or reference_df.empty:
            report = DriftReport()
            report.message = "No reference data provided for drift detection"
            self._last_drift = report
            return report

        # Only compare common numeric columns
        common_cols = [c for c in current_df.columns if c in reference_df.columns]
        numeric_cols = [
            c for c in common_cols
            if pd.api.types.is_float_dtype(current_df[c])
            or pd.api.types.is_integer_dtype(current_df[c])
        ]

        if not numeric_cols:
            report = DriftReport()
            report.message = "No common numeric columns to compare"
            self._last_drift = report
            return report

        drift_features: list[dict[str, Any]] = []
        n_warnings = 0

        for col in numeric_cols:
            curr = current_df[col].dropna().values
            ref = reference_df[col].dropna().values

            if len(curr) == 0 or len(ref) == 0:
                continue

            # Use Population Stability Index (PSI) for drift
            psi = self._compute_psi(curr, ref)
            is_drifted = bool(psi > self.drift_threshold)
            if is_drifted:
                n_warnings += 1

            drift_features.append({
                "column": col,
                "psi": round(psi, 4),
                "drifted": is_drifted,
                "current_mean": round(float(np.mean(curr)), 4),
                "reference_mean": round(float(np.mean(ref)), 4),
                "current_std": round(float(np.std(curr, ddof=1)), 4),
                "reference_std": round(float(np.std(ref, ddof=1)), 4),
            })

        report = DriftReport(
            n_features=len(numeric_cols),
            n_drifted=n_warnings,
            drift_threshold=self.drift_threshold,
            passed=n_warnings == 0,
            details=drift_features,
        )
        self._last_drift = report
        return report

    def feature_importance_placeholder(
        self,
        feature_names: list[str],
        model_type: str = "unknown",
    ) -> FeatureImportancePlaceholder:
        """Create a feature importance placeholder for pipeline reports.

        Actual feature importance requires a trained model. This placeholder
        provides the structure for future integration with model evaluation.

        Parameters
        ----------
        feature_names : list[str]
            Names of features used by the model.
        model_type : str
            Type of model (e.g. ``xgboost``, ``logistic_regression``).

        Returns
        -------
        FeatureImportancePlaceholder
        """
        placeholder = FeatureImportancePlaceholder(
            n_features=len(feature_names),
            feature_names=sorted(feature_names),
            model_type=model_type,
            message=(
                "Feature importance scores are not available until a model "
                "is trained. Use `model.feature_importances_` or "
                "`permutation_importance()` after training."
            ),
        )
        self._last_importance = placeholder
        return placeholder

    # ══════════════════════════════════════════════════════
    #  Internal: Drift metric (delegates to shared compute_psi)
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _compute_psi(
        current: np.ndarray,
        reference: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Population Stability Index (delegates to ``compute_psi``)."""
        return compute_psi(current, reference, n_bins=n_bins)

    # ── Properties ──────────────────────────────────────

    @property
    def last_report(self) -> ValidationReport | None:
        return self._last_report

    def __repr__(self) -> str:
        return (
            f"<FeatureValidator {len(self.checks)} checks, "
            f"corr>{self.correlation_threshold}, "
            f"drift>{self.drift_threshold}>"
        )
