"""Ensemble Utilities - helper functions for ensemble operations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from src.ensemble.training import EnsembleModel

logger = logging.getLogger(__name__)


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_train: pd.DataFrame | None = None,
    df_val: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a complete ensemble end-to-end and return all results.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data (used for weight optimisation)
    X_test, y_test : test data (held-out evaluation)
    df_train, df_val, df_test : raw match DataFrames for Poisson model
    verbose : bool
        Print summary to console if True.

    Returns
    -------
    dict[str, Any]
        ``{ensemble, test_report, weights, ensemble_probs}``
    """
    ensemble = EnsembleModel()

    # Train & optimise weights
    fit_report = ensemble.fit(
        X_train, y_train, X_val, y_val,
        df_train=df_train, df_val=df_val,
    )

    # Evaluate on test
    test_report = ensemble.evaluate(X_test, y_test, df_test)

    # Predictions
    ensemble_probs = ensemble.predict_proba(X_test, df_test)

    if verbose:
        print("\n" + "=" * 90)
        print("  ENSEMBLE TRAINING RESULTS".center(88))
        print("=" * 90)

        print(f"\n  Validation log-loss: {fit_report['val_log_loss']:.4f}")
        print(f"  Test log-loss:       {test_report['ensemble_log_loss']:.4f}")
        print(f"  Test accuracy:       {test_report['ensemble_accuracy']:.2%}")
        print(f"\n  Best single model:   {test_report['best_single_model']} "
              f"({test_report['individual_log_losses'][test_report['best_single_model']]:.4f})")
        print(f"  Improvement:         Delta = {test_report['improvement_over_best_single']:+.4f}")
        print(f"\n  {ensemble.weight_summary}")
        print(f"\n  {'=' * 30}  LOG-LOSS BREAKDOWN {'=' * 30}")
        for name, loss in sorted(test_report["individual_log_losses"].items()):
            marker = " <- BEST" if abs(loss - min(test_report["individual_log_losses"].values())) < 1e-6 else ""
            print(f"    {name:<30s}  {loss:.4f}{marker}")
        print("=" * 90)
        print()

    return {
        "ensemble": ensemble,
        "test_report": test_report,
        "weights": fit_report["weights"],
        "ensemble_probs": ensemble_probs,
    }
