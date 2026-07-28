"""
Hyper-parameter Tuning — optimise every ML model, compare before/after, and save.

Usage::

    from src.hyperparameter_tuning import HyperTuner

    tuner = HyperTuner()
    results = tuner.run(X_train, y_train, X_val, y_val, X_test, y_test)
"""

from src.hyperparameter_tuning.tuner import HyperTuner, ModelResult
from src.hyperparameter_tuning.optuna_tuner import OptunaTuner
from src.hyperparameter_tuning.optimisers import tune_hyperparameters

__all__ = [
    "HyperTuner",
    "ModelResult",
    "OptunaTuner",
    "tune_hyperparameters",
]
