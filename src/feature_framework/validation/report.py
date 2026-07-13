"""
Feature Validation Reports — structured output types for all checks.

Each report type is a dataclass that can be serialised to dict,
formatted as text, or converted to a pandas DataFrame for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd


# ═══════════════════════════════════════════════════════════════
#  Validation Report
# ═══════════════════════════════════════════════════════════════


@dataclass
class ValidationReport:
    """Aggregated results from all feature validation checks.

    Parameters
    ----------
    n_rows : int
        Number of rows in the validated DataFrame.
    n_columns : int
        Number of columns.
    column_names : list[str]
        Names of all columns.
    total_checks : int
        Number of checks that ran.
    passed_checks : int
        Number of checks that passed.
    failed_checks : int
        Number of checks that failed.
    total_violations : int
        Total violation count across all failed checks.
    checks : dict[str, dict]
        Per-check results, keyed by check name.
    """

    n_rows: int = 0
    n_columns: int = 0
    column_names: list[str] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    total_violations: int = 0
    checks: dict[str, dict] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if all checks passed."""
        return self.failed_checks == 0

    def summary(self) -> str:
        """Return a concise human-readable summary."""
        lines = [
            "FEATURE VALIDATION REPORT",
            "=" * 60,
            f"  Data:         {self.n_rows} rows × {self.n_columns} columns",
            f"  Checks:       {self.passed_checks}/{self.total_checks} passed",
            f"  Failures:     {self.failed_checks}",
            f"  Violations:   {self.total_violations}",
            f"  Result:       {'✅ PASS' if self.passed else '❌ FAIL'}",
            "",
        ]

        if self.failed_checks > 0 and self.checks:
            lines.append("  Failed checks:")
            for name, result in sorted(self.checks.items()):
                if not result.get("passed", True):
                    n_v = len(result.get("violations", []))
                    lines.append(f"    • {name}: {n_v} violation(s)")
                    # Show first 3 violations
                    for v in result.get("violations", [])[:3]:
                        msg = v.get("message", v.get("column", str(v)))
                        lines.append(f"      - {msg}")
                    if n_v > 3:
                        lines.append(f"      ... and {n_v - 3} more")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (excludes the full check details by default)."""
        return {
            "passed": self.passed,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "total_violations": self.total_violations,
            "check_details": {
                k: {"passed": v.get("passed"), "n_violations": len(v.get("violations", []))}
                for k, v in self.checks.items()
            },
        }

    @property
    def violations_dataframe(self) -> pd.DataFrame:
        """Flatten all violations into a DataFrame for analysis."""
        rows: list[dict[str, Any]] = []
        for check_name, result in self.checks.items():
            for v in result.get("violations", []):
                row = {"check": check_name, **v}
                rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"<ValidationReport {self.passed_checks}/{self.total_checks} passed, "
            f"{self.total_violations} violations>"
        )


# ═══════════════════════════════════════════════════════════════
#  Correlation Report
# ═══════════════════════════════════════════════════════════════


@dataclass
class CorrelationReport:
    """Correlation matrix analysis for numeric features.

    Parameters
    ----------
    n_features : int
        Number of features in the matrix.
    correlation_matrix : pd.DataFrame, optional
        Full correlation matrix.
    high_correlation_pairs : list[dict]
        Pairs with |r| above threshold.
    n_high_pairs : int
        Count of high-correlation pairs.
    message : str, optional
        Status message.
    """

    n_features: int = 0
    correlation_matrix: pd.DataFrame | None = None
    high_correlation_pairs: list[dict] = field(default_factory=list)
    n_high_pairs: int = 0
    message: str = ""

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "CORRELATION MATRIX ANALYSIS",
            "=" * 50,
            f"  Features:       {self.n_features}",
            f"  High pairs      {self.n_high_pairs} (|r| > threshold)",
            "",
        ]
        if self.high_correlation_pairs:
            lines.append("  Top correlated pairs:")
            for pair in sorted(
                self.high_correlation_pairs,
                key=lambda x: abs(x.get("correlation", 0)),
                reverse=True,
            )[:10]:
                f1 = pair.get("feature_1", "?")
                f2 = pair.get("feature_2", "?")
                r = pair.get("correlation", 0)
                lines.append(f"    • {f1} ↔ {f2}: r={r:.4f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_high_pairs": self.n_high_pairs,
            "high_correlation_pairs": self.high_correlation_pairs[:20],
            "message": self.message,
        }

    def __repr__(self) -> str:
        return f"<CorrelationReport {self.n_features} features, {self.n_high_pairs} high pairs>"


# ═══════════════════════════════════════════════════════════════
#  Missing Value Report
# ═══════════════════════════════════════════════════════════════


@dataclass
class MissingValueReport:
    """Missing value analysis for all columns.

    Parameters
    ----------
    total_rows : int
    total_columns : int
    total_cells : int
        Total cells count (rows × columns).
    n_missing_cells : int
        Count of NaN cells.
    n_infinite_cells : int
        Count of Inf/-Inf cells.
    missing_rate : float
        Proportion of all cells that are missing.
    columns_with_missing : int
        Number of columns with at least one missing value.
    details : list[dict]
        Per-column missing value details.
    """

    total_rows: int = 0
    total_columns: int = 0
    total_cells: int = 0
    n_missing_cells: int = 0
    n_infinite_cells: int = 0
    missing_rate: float = 0.0
    columns_with_missing: int = 0
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        pct = self.missing_rate * 100
        lines = [
            "MISSING VALUE REPORT",
            "=" * 50,
            f"  Total cells:     {self.total_cells}",
            f"  Missing cells:   {self.n_missing_cells} ({pct:.2f}%)",
            f"  Inf cells:       {self.n_infinite_cells}",
            f"  Columns affected: {self.columns_with_missing}/{self.total_columns}",
            "",
        ]
        if self.details:
            lines.append("  Top columns by missing rate:")
            for d in sorted(self.details, key=lambda x: x["missing_rate"], reverse=True)[:10]:
                col = d["column"]
                n = d["n_missing"]
                rate = d["missing_rate"] * 100
                lines.append(f"    • {col}: {n} missing ({rate:.1f}%)")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cells": self.total_cells,
            "n_missing_cells": self.n_missing_cells,
            "n_infinite_cells": self.n_infinite_cells,
            "missing_rate": self.missing_rate,
            "columns_with_missing": self.columns_with_missing,
            "details": self.details,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return missing value details as a DataFrame."""
        return pd.DataFrame(self.details)

    def __repr__(self) -> str:
        return (
            f"<MissingValueReport {self.n_missing_cells}/{self.total_cells} "
            f"({self.missing_rate:.1%}) missing>"
        )


# ═══════════════════════════════════════════════════════════════
#  Drift Report
# ═══════════════════════════════════════════════════════════════


@dataclass
class DriftReport:
    """Feature distribution drift analysis.

    Parameters
    ----------
    n_features : int
        Number of features compared.
    n_drifted : int
        Number of features with PSI > threshold.
    drift_threshold : float
        PSI threshold used.
    passed : bool
        True if no features drifted.
    details : list[dict]
        Per-feature drift metrics.
    message : str, optional
        Status message.
    """

    n_features: int = 0
    n_drifted: int = 0
    drift_threshold: float = 0.1
    passed: bool = True
    details: list[dict] = field(default_factory=list)
    message: str = ""

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "✅ PASS" if self.passed else f"❌ FAIL ({self.n_drifted} drifted)"
        lines = [
            "DRIFT REPORT",
            "=" * 50,
            f"  Features compared: {self.n_features}",
            f"  Drifted:           {self.n_drifted} (PSI > {self.drift_threshold})",
            f"  Result:            {status}",
            "",
        ]
        if self.details:
            drifted = [d for d in self.details if d.get("drifted", False)]
            if drifted:
                lines.append("  Drifted features:")
                for d in sorted(drifted, key=lambda x: x.get("psi", 0), reverse=True)[:10]:
                    col = d["column"]
                    psi = d["psi"]
                    cur_mean = d.get("current_mean", "?")
                    ref_mean = d.get("reference_mean", "?")
                    lines.append(
                        f"    • {col}: PSI={psi:.4f} "
                        f"(ref μ={ref_mean} → cur μ={cur_mean})"
                    )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_drifted": self.n_drifted,
            "threshold": self.drift_threshold,
            "passed": self.passed,
            "details": self.details,
            "message": self.message,
        }

    def __repr__(self) -> str:
        return f"<DriftReport {self.n_drifted}/{self.n_features} drifted>"


# ═══════════════════════════════════════════════════════════════
#  Feature Importance Placeholder
# ═══════════════════════════════════════════════════════════════


@dataclass
class FeatureImportancePlaceholder:
    """Placeholder for feature importance data.

    Actual feature importance requires a trained model. This placeholder
    provides the schema and messaging until model training is integrated.

    Parameters
    ----------
    n_features : int
        Number of features.
    feature_names : list[str]
        Sorted list of feature names.
    model_type : str
        Model type (e.g. ``xgboost``).
    importance_scores : dict, optional
        Future: actual importance scores.
    message : str
        Explanation text.
    """

    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    model_type: str = "unknown"
    importance_scores: dict[str, float] | None = None
    message: str = (
        "Feature importance not available. Train a model and "
        "call ``model.feature_importances_`` or "
        "``sklearn.inspection.permutation_importance()``."
    )

    def summary(self) -> str:
        """Return placeholder summary."""
        lines = [
            "FEATURE IMPORTANCE",
            "=" * 50,
            f"  Features:  {self.n_features}",
            f"  Model:     {self.model_type}",
            f"  Status:    ⏳ Placeholder — no model trained yet",
            f"  {self.message}",
        ]
        if self.feature_names:
            lines.append("")
            lines.append("  Feature names (top 20):")
            for name in self.feature_names[:20]:
                lines.append(f"    • {name}")
            if len(self.feature_names) > 20:
                lines.append(f"    ... and {len(self.feature_names) - 20} more")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "importance_scores": self.importance_scores,
            "message": self.message,
        }

    def __repr__(self) -> str:
        return f"<FeatureImportancePlaceholder {self.n_features} features, model={self.model_type}>"
