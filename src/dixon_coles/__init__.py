"""
Dixon-Coles Model Package

This package provides the Dixon-Coles model for football match prediction,
split into modular components for better maintainability.
"""

from src.dixon_coles.fit import fit_dixon_coles_predict
from src.dixon_coles.model import DixonColesModel, DixonColesResult
from src.dixon_coles.tau import dixon_coles_tau
from src.dixon_coles.weights import (
    TOURNAMENT_IMPORTANCE,
    compute_recency_weight,
    get_tournament_importance,
)

__all__ = [
    "DixonColesModel",
    "DixonColesResult",
    "get_tournament_importance",
    "compute_recency_weight",
    "TOURNAMENT_IMPORTANCE",
    "dixon_coles_tau",
    "fit_dixon_coles_predict",
]
