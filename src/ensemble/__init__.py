"""Ensemble Models Package.

This package provides ensemble methods for football match prediction,
combining multiple models to improve accuracy and robustness.

Modules
-------
weighted : WeightedEnsemble class for fixed/optimized weighted averaging
stacking : StackingEnsemble class with meta-learner
training : EnsembleModel class for full training pipeline
utils : Utility functions for ensemble operations
"""

from src.ensemble.weighted import WeightedEnsemble
from src.ensemble.stacking import StackingEnsemble
from src.ensemble.training import EnsembleModel
from src.ensemble.utils import train_ensemble

__all__ = [
    "WeightedEnsemble",
    "StackingEnsemble",
    "EnsembleModel",
    "train_ensemble",
]
