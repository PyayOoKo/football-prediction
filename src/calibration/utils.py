"""Shared helper functions for the calibration subpackage.

Provides input validation and probability renormalisation used by
all calibrator classes.
"""

from __future__ import annotations

import numpy as np


def validate_probs_input(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert calibrator input arrays.

    Parameters
    ----------
    X : probability array of shape (n, n_classes)
    y : label array of shape (n,)
    n_classes : expected number of columns

    Returns
    -------
    tuple of (X, y) as float64 and int64 arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if X.shape[1] != n_classes:
        raise ValueError(f"X must have {n_classes} columns, got {X.shape[1]}")
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)} samples) and y ({len(y)} samples) must match")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf values")

    return X, y


def renormalise_probs(probs: np.ndarray) -> np.ndarray:
    """Normalise probability rows to sum to 1.0.

    Rows that sum to zero (degenerate case) are replaced with a uniform
    distribution to keep downstream consumers from seeing NaN or zero rows.
    """
    probs = np.asarray(probs, dtype=np.float64)
    row_sums = probs.sum(axis=1)
    zero_mask = row_sums <= 0
    if zero_mask.any():
        probs = probs.copy()
        probs[zero_mask] = 1.0 / probs.shape[1]
        row_sums = row_sums.copy()
        row_sums[zero_mask] = 1.0
    return probs / row_sums[:, np.newaxis]


# Legacy aliases for backward compatibility
_validate_probs_input = validate_probs_input
_renormalise_probs = renormalise_probs
