"""
Categorical encoding — transforms team names into numeric feature vectors.

Strategies (configurable via ``config.features.categorical_encoding``):
- ``label`` — simple integer label encoding
- ``onehot`` — one-hot encoding
- ``target`` — target encoding (expanding mean, shifted to avoid leakage)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config import config as _global_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  SafeTargetEncoder — leakage-free target encoder
# ═══════════════════════════════════════════════════════════════


class SafeTargetEncoder:
    """Target encoder that stores training-only statistics for inference.

    Unlike the ad-hoc ``_target_encode`` function (which computes the
    prior from all data including future rows), this class stores
    category-level means and a global prior that are **only** derived
    from the data passed to ``fit()``.

    Parameters
    ----------
    cols : list[str]
        Categorical columns to encode (e.g. ``["home_team", "away_team"]``).
    prior : float | None
        Global mean to use for unseen categories.  If ``None``, set from
        ``fit()`` data.
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        prior: float | None = None,
    ) -> None:
        self.cols = cols or ["home_team", "away_team"]
        self._prior = prior
        self._category_means: dict[str, dict[str, float]] = {}
        self._fitted = False

    @property
    def prior(self) -> float:
        if self._prior is None:
            return 0.5
        return self._prior

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SafeTargetEncoder:
        """Compute category-level target means and a global prior.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix containing ``cols``.
        y : pd.Series, optional
            Target vector.  If ``None``, looks for ``"target"`` column in *X*.

        Returns
        -------
        SafeTargetEncoder
            Fitted instance.
        """
        if y is None:
            if "target" in X.columns:
                y = X["target"]
            else:
                raise ValueError(
                    "SafeTargetEncoder.fit() requires y or a 'target' column in X."
                )

        # Global prior from training data ONLY (handle None y.name)
        y_clean = y.dropna() if hasattr(y, "dropna") else y
        self._prior = float(y_clean.mean())

        # Add y as a named column for groupby (avoids KeyError when y.name is None)
        _y_col = y.name if (hasattr(y, "name") and y.name is not None) else "_target_"
        X_with_y = X.copy()
        X_with_y[_y_col] = y_clean.values if hasattr(y_clean, "values") else y_clean

        for col in self.cols:
            if col not in X_with_y.columns:
                continue
            means = X_with_y.groupby(col)[_y_col].mean()
            self._category_means[col] = means.to_dict()

        self._fitted = True
        logger.debug(
            "SafeTargetEncoder fitted on %d rows: prior=%.4f, categories=%s",
            len(X),
            self._prior,
            {col: len(v) for col, v in self._category_means.items()},
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored encoding — unseen categories get the global prior.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to transform (train, val, test, or inference).

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with encoded columns replacing originals.
        """
        if not self._fitted:
            raise RuntimeError(
                "SafeTargetEncoder.transform() called before fit(). "
                "Call .fit(X, y) first."
            )

        X = X.copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[f"{col}_encoded"] = (
                X[col].map(self._category_means.get(col, {})).fillna(self.prior)
            )
            X.drop(columns=[col], inplace=True)

        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def get_state(self) -> dict[str, Any]:
        """Return serialisable state for persistence (e.g. in ModelArtifact)."""
        return {
            "cols": self.cols,
            "prior": self._prior,
            "category_means": self._category_means,
            "fitted": self._fitted,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> SafeTargetEncoder:
        """Restore from a state dict (e.g. loaded from ModelArtifact)."""
        encoder = cls(cols=state.get("cols"), prior=state.get("prior"))
        encoder._category_means = state.get("category_means", {})
        encoder._fitted = state.get("fitted", False)
        return encoder


# ═══════════════════════════════════════════════════════════════
#  Public encoding dispatcher
# ═══════════════════════════════════════════════════════════════


def _encode_categoricals(
    df: pd.DataFrame,
    config: Any | None = None,
    encoder: SafeTargetEncoder | None = None,
) -> pd.DataFrame:
    """Encode categorical columns per configured strategy.

    When *encoder* is provided (pre-fitted ``SafeTargetEncoder``), it is
    used for inference — the global prior comes from training data only.

    Strategies:
    - ``"label"`` — simple integer label encoding
    - ``"onehot"`` — one-hot encoding
    - ``"target"`` — target encoding (expanding mean, leakage-free)

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    encoder : SafeTargetEncoder, optional
        Pre-fitted target encoder for inference.  If provided, its
        stored statistics are used instead of recomputing.
    """
    cfg = config or _global_config
    cat_cols = ["home_team", "away_team"]
    existing_cats = [c for c in cat_cols if c in df.columns]

    if not existing_cats:
        return df

    strategy = cfg.features.categorical_encoding
    logger.debug("Encoding categoricals via '%s'", strategy)

    if encoder is not None:
        # Use pre-fitted encoder for inference — no leakage
        return encoder.transform(df)

    if strategy == "label":
        df = _label_encode(df, existing_cats)
    elif strategy == "onehot":
        df = _onehot_encode(df, existing_cats)
    elif strategy == "target":
        df = _target_encode(df, existing_cats)
    else:
        logger.warning(
            "Unknown encoding strategy '%s' — falling back to label", strategy
        )
        df = _label_encode(df, existing_cats)

    return df


def _label_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Label-encode categorical columns in place."""
    for col in cols:
        df[col] = df[col].astype("category").cat.codes
    for col in ["h_opponent", "a_opponent"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def _onehot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns and drop the originals."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, dtype="int8")
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)
    return df


def _target_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Target-encode categorical columns with expanding mean (shifted to avoid leakage).

    .. caution::

        The prior (``global_mean``) is computed from **all** data in *df*,
        which may include validation/test rows.  For production use, pass a
        pre-fitted ``SafeTargetEncoder`` via *encoder* to avoid leaking
        future label information into the prior.
    """
    if "target" not in df.columns:
        logger.warning(
            "Target encoding requires 'target' column — falling back to label"
        )
        return _label_encode(df, cols)

    df_sorted = df.sort_values("date").copy()
    for col in cols:
        encoded = (
            df_sorted.groupby(col)["target"]
            .expanding()
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_encoded"] = encoded
        global_mean = df_sorted["target"].mean()
        df[f"{col}_encoded"].fillna(global_mean, inplace=True)
        df.drop(columns=[col], inplace=True)

    return df
