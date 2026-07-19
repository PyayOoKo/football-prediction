"""
Dixon-Coles Model Package

This package provides the Dixon-Coles model for football match prediction,
split into modular components for better maintainability.
"""

from src.dixon_coles.model import DixonColesModel
from src.dixon_coles.weights import (
    get_tournament_importance,
    compute_recency_weight,
    TOURNAMENT_IMPORTANCE,
)
from src.dixon_coles.tau import dixon_coles_tau
from src.dixon_coles.fit import fit_dixon_coles_predict

__all__ = [
    "DixonColesModel",
    "get_tournament_importance",
    "compute_recency_weight",
    "TOURNAMENT_IMPORTANCE",
    "dixon_coles_tau",
    "fit_dixon_coles_predict",
]
