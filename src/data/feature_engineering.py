"""
Feature engineering — builds model input features from raw match data.

Generates rolling averages, form indicators, Elo ratings,
head-to-head stats, and optional xG-derived features.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Builds feature matrices for model training and prediction.

    Parameters
    ----------
    form_window : int
        Rolling window for recent form features (default ``5``).
    include_elo : bool
        Whether to compute Elo ratings (default ``True``).
    include_h2h : bool
        Whether to include head-to-head features (default ``True``).
    """

    def __init__(
        self,
        form_window: int = 5,
        include_elo: bool = True,
        include_h2h: bool = True,
    ) -> None:
        self.form_window = form_window
        self.include_elo = include_elo
        self.include_h2h = include_h2h

    def build_features(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Build feature matrix and optional target.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned match data with date, teams, goals, and odds.
        is_training : bool
            If ``True``, expect a target column and return ``(X, y)``.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series | None]
            Feature matrix ``X`` and optionally target series ``y``.
        """
        logger.info("Building features: %d rows", len(df))
        # TODO: Implement actual feature engineering logic
        X = df
        y = None
        if is_training and "result" in df.columns:
            y = df["result"]
            X = df.drop(columns=["result"])
        return X, y
