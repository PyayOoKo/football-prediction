"""Ensemble Utilities - helper functions for ensemble operations."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

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
        X_train,
        y_train,
        X_val,
        y_val,
        df_train=df_train,
        df_val=df_val,
    )

    # Evaluate on test
    test_report = ensemble.evaluate(X_test, y_test, df_test)

    # Predictions
    ensemble_probs = ensemble.predict_proba(X_test, df_test)

    if verbose:
        logger.info("")
        logger.info("=" * 90)
        logger.info("  ENSEMBLE TRAINING RESULTS".center(88))
        logger.info("=" * 90)

        logger.info("  Validation log-loss: %.4f", fit_report["val_log_loss"])
        logger.info("  Test log-loss:       %.4f", test_report["ensemble_log_loss"])
        logger.info(
            "  Test accuracy:       %.2f%%", test_report["ensemble_accuracy"] * 100
        )
        best_model = test_report["best_single_model"]
        logger.info(
            "  Best single model:   %s (%.4f)",
            best_model,
            test_report["individual_log_losses"][best_model],
        )
        logger.info(
            "  Improvement:         Delta = %+.4f",
            test_report["improvement_over_best_single"],
        )
        logger.info("  %s", ensemble.weight_summary)
        logger.info("  %s", "=" * 30 + "  LOG-LOSS BREAKDOWN " + "=" * 30)
        best_loss = min(test_report["individual_log_losses"].values())
        for name, loss in sorted(test_report["individual_log_losses"].items()):
            marker = " <- BEST" if abs(loss - best_loss) < 1e-6 else ""
            logger.info("    %-30s  %.4f%s", name, loss, marker)
        logger.info("=" * 90)
        logger.info("")

    return {
        "ensemble": ensemble,
        "test_report": test_report,
        "weights": fit_report["weights"],
        "ensemble_probs": ensemble_probs,
    }
