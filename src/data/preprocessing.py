"""
Data preprocessor — cleaning and transformation pipeline.

Applies a configurable sequence of transformations:
- Column renaming and type casting
- Missing value imputation
- Categorical encoding
- Temporal feature extraction (year, month, day of week)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Configurable data preprocessing pipeline.

    Parameters
    ----------
    normalise_teams : bool
        Whether to normalise team name spelling (default ``True``).
    add_temporal_features : bool
        Whether to add year/month/day-of-week columns (default ``True``).
    """

    def __init__(
        self,
        normalise_teams: bool = True,
        add_temporal_features: bool = True,
    ) -> None:
        self.normalise_teams = normalise_teams
        self.add_temporal_features = add_temporal_features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw match data.

        Returns
        -------
        pd.DataFrame
            Cleaned and transformed data.
        """
        logger.info("Preprocessing dataset: %d rows, %d cols", len(df), len(df.columns))
        # TODO: Implement actual preprocessing logic
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same transformation to new data."""
        return self.fit_transform(df)
