"""
Dixon-Coles tau correction function for low-scoring matches.

The correction adjusts for the empirical observation that low-scoring
results (0-0, 1-0, 0-1, 1-1) occur more frequently than the independent
Poisson model predicts.
"""

from __future__ import annotations

import numpy as np


def dixon_coles_tau(
    x: int | np.ndarray,
    y: int | np.ndarray,
    lam: float | np.ndarray,
    mu: float | np.ndarray,
    rho: float,
) -> float | np.ndarray:
    """Dixon-Coles tau correction factor for low-scoring matches.

    The correction adjusts for the empirical observation that low-scoring
    results (0-0, 1-0, 0-1, 1-1) occur more frequently than the independent
    Poisson model predicts.

    Parameters
    ----------
    x : int or array-like
        Home goals (can be array for vectorised computation).
    y : int or array-like
        Away goals.
    lam : float or array-like
        Expected home goals.
    mu : float or array-like
        Expected away goals.
    rho : float
        Dependence parameter. rho > 0 means low scores are more likely
        than independent model; rho < 0 means less likely; rho = 0 means
        independent Poisson.

    Returns
    -------
    float or np.ndarray
        Tau correction factor.
    """
    # Handle scalar case
    if np.isscalar(x):
        if x == 0 and y == 0:
            return 1.0 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1.0 + lam * rho
        elif x == 1 and y == 0:
            return 1.0 + mu * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        else:
            return 1.0

    # Vectorised case
    tau = np.ones_like(lam, dtype=float)
    tau[(x == 0) & (y == 0)] = 1.0 - lam[(x == 0) & (y == 0)] * mu[(x == 0) & (y == 0)] * rho
    tau[(x == 0) & (y == 1)] = 1.0 + lam[(x == 0) & (y == 1)] * rho
    tau[(x == 1) & (y == 0)] = 1.0 + mu[(x == 1) & (y == 0)] * rho
    tau[(x == 1) & (y == 1)] = 1.0 - rho
    return tau
