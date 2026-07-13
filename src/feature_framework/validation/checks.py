"""
Feature Validation Checks — 10 detection functions for computed features.

Each check is a standalone function that operates on a feature DataFrame
and returns a dict with keys: ``check_name``, ``passed``, ``violations``.

All checks handle edge cases (empty DataFrames, single columns, all-NaN).
"""

from __future__ import annotations

import fnmatch
from typing import Any

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
#  Check 1: Data Leakage
# ═══════════════════════════════════════════════════════════════


def check_data_leakage(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect potential temporal data leakage.

    Checks that the DataFrame is sorted chronologically (if a ``date``
    column exists) and that no future data is leaking into past rows.
    Also flags any obviously impossible temporal patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    **kwargs
        ``date_column`` — column name for dates (default ``\"date\"``).

    Returns
    -------
    dict
        ``check_name``, ``passed``, ``violations``.
    """
    date_col = kwargs.get("date_column", "date")
    violations: list[dict[str, Any]] = []

    if date_col not in df.columns:
        return {
            "check_name": "data_leakage",
            "passed": True,
            "violations": [],
            "message": f"No '{date_col}' column — skipping temporal check",
        }

    # Check if date column is sorted
    dates = pd.to_datetime(df[date_col], errors="coerce")
    non_null = dates.dropna()

    if len(non_null) < 2:
        return {
            "check_name": "data_leakage",
            "passed": True,
            "violations": [],
            "message": "Insufficient dates for temporal check",
        }

    # Check for out-of-order dates
    diffs = non_null.diff()
    n_negative = int((diffs < pd.Timedelta(0)).sum())
    n_zero = int((diffs == pd.Timedelta(0)).sum())

    if n_negative > 0:
        violations.append({
            "field": date_col,
            "value": f"{n_negative} out-of-order date(s)",
            "message": (
                f"DataFrame is not sorted chronologically. "
                f"{n_negative} row(s) have dates before the previous row. "
                f"This could cause temporal leakage if rolling features "
                f"are computed on unsorted data."
            ),
        })

    if n_zero > 0:
        violations.append({
            "field": date_col,
            "value": f"{n_zero} duplicate date(s)",
            "message": (
                f"{n_zero} row(s) share the same date as the previous row. "
                f"Ensure multiple matches on the same date are ordered "
                f"consistently (e.g. by home team)."
            ),
        })

    return {
        "check_name": "data_leakage",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"Temporal check: {n_negative} out-of-order, {n_zero} duplicate dates" if violations else "Dates are chronologically sorted",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 2: Constant Features
# ═══════════════════════════════════════════════════════════════


def check_constant_features(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect columns with zero variance (constant values).

    Constant features provide no predictive signal and should be dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    **kwargs
        ``min_unique_ratio`` — min unique/total ratio (default 0.01).

    Returns
    -------
    dict
    """
    min_ratio = kwargs.get("min_unique_ratio", 0.01)
    violations: list[dict[str, Any]] = []
    total = len(df)

    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        ratio = nunique / max(total, 1)

        if ratio < min_ratio or nunique <= 1:
            violations.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(nunique),
                "unique_ratio": round(ratio, 4),
                "message": (
                    f"Feature '{col}' has only {nunique} unique value(s) "
                    f"({ratio:.2%} of rows) — {'zero' if nunique <= 1 else 'near-zero'} variance."
                ),
            })

    return {
        "check_name": "constant_features",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} constant or near-constant feature(s) detected",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 3: Highly Correlated Features
# ═══════════════════════════════════════════════════════════════


def check_highly_correlated(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect pairs of features with high Pearson correlation.

    Highly correlated features cause multicollinearity in linear models
    and inflate feature importance in tree-based models.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    **kwargs
        ``correlation_threshold`` — |r| threshold (default 0.95).

    Returns
    -------
    dict
    """
    threshold = kwargs.get("correlation_threshold", 0.95)
    violations: list[dict[str, Any]] = []

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return {
            "check_name": "highly_correlated",
            "passed": True,
            "violations": [],
            "message": "Need at least 2 numeric columns for correlation check",
        }

    corr = numeric.corr(method="pearson").values
    cols = numeric.columns.tolist()
    n = len(cols)

    checked_pairs: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in checked_pairs:
                continue
            checked_pairs.add((i, j))

            val = corr[i, j]
            if not pd.isna(val) and abs(val) >= threshold:
                violations.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": round(float(val), 4),
                    "message": (
                        f"Features '{cols[i]}' and '{cols[j]}' have "
                        f"|r| = {abs(val):.4f} (threshold={threshold}). "
                        f"Consider removing one to reduce multicollinearity."
                    ),
                })

    return {
        "check_name": "highly_correlated",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} highly correlated pair(s) found",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 4: Missing Values
# ═══════════════════════════════════════════════════════════════


def check_missing_values(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect columns with missing (NaN) values.

    Reports the count and percentage of missing values per column.
    Flags columns with > 50% missing as potentially unusable.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.

    Returns
    -------
    dict
    """
    violations: list[dict[str, Any]] = []
    total = len(df)

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue

        rate = n_missing / max(total, 1)
        severity = "ERROR" if rate > 0.5 else "WARNING" if rate > 0.1 else "INFO"

        violations.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": n_missing,
            "missing_rate": round(rate, 4),
            "severity": severity,
            "message": (
                f"Feature '{col}' has {n_missing}/{total} missing values "
                f"({rate:.1%})."
            ),
        })

    return {
        "check_name": "missing_values",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} column(s) with missing values",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 5: Invalid Ranges
# ═══════════════════════════════════════════════════════════════


def check_invalid_ranges(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect values outside expected range bounds.

    Supports wildcard patterns in ``range_bounds`` (e.g. ``\"odds_*\"``).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    **kwargs
        ``range_bounds`` — dict mapping column patterns to ``(min, max)``.
        Supports ``*`` wildcard. E.g. ``{\"prob_*\": (0.0, 1.0)}``.

    Returns
    -------
    dict
    """
    range_bounds: dict[str, tuple[float, float]] = kwargs.get("range_bounds", {})
    violations: list[dict[str, Any]] = []

    # Resolve wildcard patterns to actual column names
    resolved: dict[str, tuple[float, float]] = {}
    for pattern, bounds in range_bounds.items():
        if "*" in pattern:
            matched = fnmatch.filter(df.columns, pattern)
            for col in matched:
                resolved[col] = bounds
        else:
            resolved[pattern] = bounds

    for col, (lo, hi) in resolved.items():
        if col not in df.columns:
            continue

        values = df[col]
        if not pd.api.types.is_float_dtype(values) and not pd.api.types.is_integer_dtype(values):
            continue

        below = values < lo
        above = values > hi
        n_below = int(below.sum())
        n_above = int(above.sum())

        if n_below > 0:
            violations.append({
                "column": col,
                "range": f"[{lo}, {hi}]",
                "n_below": n_below,
                "message": f"{n_below} value(s) below minimum ({lo}) in '{col}'",
            })
        if n_above > 0:
            violations.append({
                "column": col,
                "range": f"[{lo}, {hi}]",
                "n_above": n_above,
                "message": f"{n_above} value(s) above maximum ({hi}) in '{col}'",
            })

    return {
        "check_name": "invalid_ranges",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} out-of-range violation(s)",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 6: Infinite Values
# ═══════════════════════════════════════════════════════════════


def check_infinite_values(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect columns containing Inf or -Inf values.

    Infinite values cause numerical instability in most ML models
    and must be handled before training.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.

    Returns
    -------
    dict
    """
    violations: list[dict[str, Any]] = []

    for col in df.columns:
        if not pd.api.types.is_float_dtype(df[col]):
            continue

        n_inf = int(np.isinf(df[col]).sum())
        if n_inf > 0:
            violations.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "n_infinite": n_inf,
                "message": (
                    f"Feature '{col}' has {n_inf} infinite value(s). "
                    f"Replace with NaN or impute before training."
                ),
            })

    return {
        "check_name": "infinite_values",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} column(s) with infinite values",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 7: NaN Values (alias for missing values in numeric cols)
# ═══════════════════════════════════════════════════════════════


def check_nan_values(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect NaN values in numeric columns specifically.

    This is similar to ``check_missing_values`` but focuses on numeric
    columns, which are the primary input to ML models and cannot contain
    NaN without special handling.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.

    Returns
    -------
    dict
    """
    violations: list[dict[str, Any]] = []

    for col in df.columns:
        if not pd.api.types.is_float_dtype(df[col]) and not pd.api.types.is_integer_dtype(df[col]):
            continue

        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            violations.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "n_nan": n_nan,
                "message": (
                    f"Numeric feature '{col}' has {n_nan} NaN value(s). "
                    f"ML models cannot handle NaN — impute or drop."
                ),
            })

    return {
        "check_name": "nan_values",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} column(s) with NaN values",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 8: Duplicate Features
# ═══════════════════════════════════════════════════════════════


def check_duplicate_features(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect columns with identical or nearly identical values.

    Duplicate features provide no additional information and should
    be dropped to reduce dimensionality and multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.

    Returns
    -------
    dict
    """
    violations: list[dict[str, Any]] = []
    cols = list(df.columns)
    n = len(cols)

    checked_pairs: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in checked_pairs:
                continue
            checked_pairs.add((i, j))

            col_a = cols[i]
            col_b = cols[j]

            # Quick check: if dtypes differ, they're not duplicates
            if df[col_a].dtype != df[col_b].dtype:
                continue

            # Check if values are identical
            if df[col_a].equals(df[col_b]):
                violations.append({
                    "feature_1": col_a,
                    "feature_2": col_b,
                    "identical": True,
                    "message": (
                        f"Features '{col_a}' and '{col_b}' are identical. "
                        f"Keep one and drop the other."
                    ),
                })
            else:
                # Check for near-identical (e.g. scaled version)
                numeric_a = pd.to_numeric(df[col_a], errors="coerce")
                numeric_b = pd.to_numeric(df[col_b], errors="coerce")
                both_finite = np.isfinite(numeric_a) & np.isfinite(numeric_b)
                if both_finite.sum() > 0:
                    corr_val = float(
                        pd.Series(numeric_a[both_finite]).corr(
                            pd.Series(numeric_b[both_finite])
                        )
                    )
                    if not pd.isna(corr_val) and abs(corr_val) > 0.999:
                        violations.append({
                            "feature_1": col_a,
                            "feature_2": col_b,
                            "identical": False,
                            "correlation": round(corr_val, 4),
                            "message": (
                                f"Features '{col_a}' and '{col_b}' are "
                                f"near-identical (r={corr_val:.4f}). "
                                f"They may represent the same signal."
                            ),
                        })

    return {
        "check_name": "duplicate_features",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} duplicate/near-identical pair(s)",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 9: Low Variance Features
# ═══════════════════════════════════════════════════════════════


def check_low_variance(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect columns with very low variance (near-constant).

    Low-variance features contribute little to model predictions and
    can increase overfitting risk.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    **kwargs
        ``variance_threshold`` — minimum variance (default 0.01).

    Returns
    -------
    dict
    """
    threshold = kwargs.get("variance_threshold", 0.01)
    violations: list[dict[str, Any]] = []

    for col in df.columns:
        if not pd.api.types.is_float_dtype(df[col]) and not pd.api.types.is_integer_dtype(df[col]):
            continue

        vals = df[col].dropna()
        if len(vals) < 2:
            continue

        variance = float(vals.var(ddof=1))

        if variance < threshold:
            violations.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "variance": round(variance, 6),
                "std_dev": round(float(vals.std(ddof=1)), 4),
                "message": (
                    f"Feature '{col}' has variance {variance:.6f} "
                    f"(threshold={threshold}). Low-variance features "
                    f"provide little predictive signal."
                ),
            })

    return {
        "check_name": "low_variance",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} low-variance feature(s)",
    }


# ═══════════════════════════════════════════════════════════════
#  Check 10: Feature Drift
# ═══════════════════════════════════════════════════════════════


def check_feature_drift(
    df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect feature distribution drift compared to a reference dataset.

    Uses Population Stability Index (PSI) to quantify distribution
    changes. PSI > 0.1 typically indicates meaningful drift.

    Parameters
    ----------
    df : pd.DataFrame
        Current (new) feature data.
    **kwargs
        ``reference_df`` — reference/baseline DataFrame (required).
        ``drift_threshold`` — PSI threshold (default 0.1).

    Returns
    -------
    dict
    """
    reference_df = kwargs.get("reference_df")
    threshold = kwargs.get("drift_threshold", 0.1)
    violations: list[dict[str, Any]] = []

    if reference_df is None or reference_df.empty:
        return {
            "check_name": "feature_drift",
            "passed": True,
            "violations": [],
            "message": "No reference data provided for drift detection — skipping",
        }

    # Find common numeric columns
    common = set(df.columns) & set(reference_df.columns)
    numeric_cols = [
        c for c in common
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])
    ]

    if not numeric_cols:
        return {
            "check_name": "feature_drift",
            "passed": True,
            "violations": [],
            "message": "No common numeric columns for drift comparison",
        }

    for col in numeric_cols:
        curr = df[col].dropna().values
        ref = reference_df[col].dropna().values

        if len(curr) < 10 or len(ref) < 10:
            continue

        # Compute PSI
        psi = compute_psi(curr, ref)

        if psi > threshold:
            violations.append({
                "column": col,
                "psi": round(psi, 4),
                "current_mean": round(float(np.mean(curr)), 4),
                "reference_mean": round(float(np.mean(ref)), 4),
                "message": (
                    f"Feature '{col}' shows drift (PSI={psi:.4f}). "
                    f"Mean changed from {np.mean(ref):.3f} → {np.mean(curr):.3f}. "
                    f"Model performance may degrade."
                ),
            })

    return {
        "check_name": "feature_drift",
        "passed": len(violations) == 0,
        "violations": violations,
        "message": f"{len(violations)} drifted feature(s) (PSI > {threshold})",
    }


def compute_psi(
    current: np.ndarray,
    reference: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Parameters
    ----------
    current : np.ndarray
        Current (new) distribution.
    reference : np.ndarray
        Reference (expected) distribution.
    n_bins : int
        Number of bins to discretise into (default 10).

    Returns
    -------
    float
        PSI value. > 0.1 typically indicates meaningful drift.
    """
    if len(current) < n_bins or len(reference) < n_bins:
        return 0.0

    combined = np.concatenate([current, reference])
    lo = float(np.min(combined))
    hi = float(np.max(combined))

    if hi - lo < 1e-10:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)

    expected_counts, _ = np.histogram(reference, bins=bins)
    actual_counts, _ = np.histogram(current, bins=bins)

    expected_pct = expected_counts / max(len(reference), 1) + 1e-6
    actual_pct = actual_counts / max(len(current), 1) + 1e-6

    psi = float(np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    ))
    return psi
