"""
Time Series Cross-Validation — train only on past matches, never the future.

Why chronological validation is required for football prediction
================================================================

**1. Temporal dependence of features.** Every feature in this pipeline —
rolling averages, Elo ratings, Poisson strengths, league positions — is
computed from *past* matches using `.shift(1)`.  If a random shuffle
places a future match in the training set and a past match in the
validation set, the validation metrics will be **optimistically biased**:
the model was trained on data that, in reality, wouldn't exist yet.

**2. Non-stationary distributions.** Football evolves.  Tactics, fitness
science, financial disparities, and rule changes shift the underlying
distribution over time.  A model trained on random slices of history
cannot honestly measure how it will perform on *unseen future* matches —
which is the only question that matters for betting or prediction.

**3. Injury / transfer windows.** Player availability is seasonal and
time-dependent.  Random CV can place a match from before a team's
star player was injured into the training set while validating on a
match *after* the injury, creating a false sense of accuracy.

**4. Standard k-fold CV is invalid for time-series.**
::

    ┌─────────────────────────────────────────────────┐
    │  Standard k-fold (random shuffle):              │
    │  Fold 1: Train [M₁₉₈₀ ... M₂₀₂₀]  Val [M₁₉₉₀] │  ← FUTURE leaks into training
    │  Fold 2: Train [M₁₉₈₅ ... M₂₀₂₃]  Val [M₂₀₀₅] │  ← same problem
    │                                                 │
    │  TimeSeriesSplit (expanding window):             │
    │  Fold 1: Train [M₁₉₈₀ ... M₁₉₉₅]  Val [M₁₉₉₆] │
    │  Fold 2: Train [M₁₉₈₀ ... M₂₀₀₂]  Val [M₂₀₀₃] │  ← always past → future!
    │  Fold 3: Train [M₁₉₈₀ ... M₂₀₁₀]  Val [M₂₀₁₁] │
    └─────────────────────────────────────────────────┘

Usage
-----
::

    from src.time_series_cv import (
        TimeSeriesCrossValidator,
        time_series_train_val_test_split,
    )

    # 1. Get TimeSeriesSplit folds for hyper-parameter tuning
    cv = TimeSeriesCrossValidator.get_cv(n_splits=5)

    # 2. Use with GridSearchCV / RandomizedSearchCV
    searcher = GridSearchCV(model, param_grid, cv=cv)

    # 3. Simple chronological train/val/test split
    splits = time_series_train_val_test_split(X, y)
"""

from __future__ import annotations

import logging
from typing import Any, Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

from config import config as _global_config

logger = logging.getLogger(__name__)

# ── Default settings ────────────────────────────────────
_DEFAULT_N_SPLITS = 5
_DEFAULT_GAP = 0          # No gap between train and test by default
_DEFAULT_TEST_SIZE = None  # auto: last (1 / (n_splits + 1)) of data


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


class TimeSeriesCrossValidator:
    """Time series cross-validation for football prediction.

    Wraps scikit-learn's ``TimeSeriesSplit`` with an expanding window
    strategy: each fold's training set is a strict superset of the
    previous fold's training set, and validation data always comes
    *after* training data chronologically.

    Parameters
    ----------
    n_splits : int
        Number of CV folds (default 5).
    gap : int
        Number of samples to exclude from the **end** of each training
        set before the validation set begins.  This creates a "buffer"
        to avoid autocorrelation leakage between consecutive matches
        (default 0).
    max_train_size : int, optional
        Maximum size of the training set in each fold.  If set, the
        window becomes **sliding** instead of **expanding** (default
        None = expanding).
    test_size : int, optional
        Number of samples in each validation fold.  If None, the folds
        partition the data equally (default None).
    """

    def __init__(
        self,
        n_splits: int = _DEFAULT_N_SPLITS,
        gap: int = _DEFAULT_GAP,
        max_train_size: int | None = None,
        test_size: int | None = _DEFAULT_TEST_SIZE,
    ) -> None:
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.test_size = test_size
        self._splitter = SklearnTimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
            test_size=test_size,
        )

    # ── Public interface ─────────────────────────────────

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate chronological train/val index pairs.

        Yields
        ------
        train_idx : np.ndarray
            Indices for the training set (past data).
        val_idx : np.ndarray
            Indices for the validation set (future data).
        """
        return self._splitter.split(X, y, groups)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of CV folds."""
        return self._splitter.get_n_splits(X, y, groups)

    # ── Static constructors ──────────────────────────────

    @staticmethod
    def get_cv(
        n_splits: int | None = None,
        gap: int | None = None,
        config: Any | None = None,
    ) -> SklearnTimeSeriesSplit:
        """Get a ``TimeSeriesSplit`` instance for use with scikit-learn CV.

        Reads defaults from ``config`` if parameters are omitted.

        Parameters
        ----------
        n_splits : int, optional
            Number of CV folds.  Defaults to ``config.hyper_tune.cv_folds``.
        gap : int, optional
            Gap between train and val.  Defaults to 0.
        config : Any, optional
            Injected config object.  Falls back to global ``config`` when
            ``None`` (default).

        Returns
        -------
        TimeSeriesSplit
            Ready to pass as ``cv=`` to ``GridSearchCV`` or
            ``RandomizedSearchCV``.
        """
        cfg = config or _global_config
        if n_splits is None:
            n_splits = cfg.hyper_tune.cv_folds
        if gap is None:
            gap = _DEFAULT_GAP
        return SklearnTimeSeriesSplit(n_splits=n_splits, gap=gap)

    @staticmethod
    def expanding_window_split(
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = _DEFAULT_N_SPLITS,
        gap: int = _DEFAULT_GAP,
    ) -> list[dict[str, Any]]:
        """Generate expanding-window train/val splits as DataFrames.

        Unlike ``split()`` which yields index arrays, this returns actual
        DataFrames — useful for inspecting the composition of each fold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (chronologically sorted).
        y : pd.Series
            Target vector.
        n_splits : int
            Number of folds (default 5).
        gap : int
            Gap between train and val (default 0).

        Returns
        -------
        list[dict[str, Any]]
            Each element has keys ``fold``, ``X_train``, ``y_train``,
            ``X_val``, ``y_val``, ``train_size``, ``val_size``.
        """
        cv = SklearnTimeSeriesSplit(n_splits=n_splits, gap=gap)
        folds: list[dict[str, Any]] = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            folds.append({
                "fold": fold,
                "X_train": X.iloc[train_idx],
                "y_train": y.iloc[train_idx],
                "X_val": X.iloc[val_idx],
                "y_val": y.iloc[val_idx],
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            })

        return folds

    @staticmethod
    def sliding_window_split(
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int,
        horizon: int,
        step: int = 1,
    ) -> list[dict[str, Any]]:
        """Generate **sliding** window train/val splits.

        Unlike ``TimeSeriesSplit`` (which expands), a sliding window keeps
        the training set at a fixed size.  This is useful for very long
        time series where old data becomes irrelevant.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (chronologically sorted).
        y : pd.Series
            Target vector.
        window_size : int
            Fixed number of most-recent training samples per fold.
        horizon : int
            Number of validation samples per fold (must be > 0).
        step : int
            Step between folds (default 1).

        Returns
        -------
        list[dict[str, Any]]
            Each element has keys ``fold``, ``X_train``, ``y_train``,
            ``X_val``, ``y_val``, ``train_size``, ``val_size``.
        """
        n = len(X)
        folds: list[dict[str, Any]] = []
        fold = 1

        start = 0
        while start + window_size + horizon <= n:
            train_start = start
            train_end = start + window_size
            val_start = train_end
            val_end = min(val_start + horizon, n)

            folds.append({
                "fold": fold,
                "X_train": X.iloc[train_start:train_end],
                "y_train": y.iloc[train_start:train_end],
                "X_val": X.iloc[val_start:val_end],
                "y_val": y.iloc[val_start:val_end],
                "train_size": train_end - train_start,
                "val_size": val_end - val_start,
            })

            start += step
            fold += 1

        return folds

    @staticmethod
    def walk_forward_split(
        X: pd.DataFrame,
        y: pd.Series,
        initial_train_size: int,
        val_size: int,
    ) -> list[dict[str, Any]]:
        """Walk-forward validation (expanding, one validation block per fold).

        This is the most realistic simulation of a production prediction
        system: train on all data up to point *t*, predict the next
        ``val_size`` matches, move forward, repeat.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (chronologically sorted).
        y : pd.Series
            Target vector.
        initial_train_size : int
            Size of the first training set.
        val_size : int
            Number of validation samples per fold.

        Returns
        -------
        list[dict[str, Any]]
            Each element has keys ``fold``, ``X_train``, ``y_train``,
            ``X_val``, ``y_val``, ``train_size``, ``val_size``.
        """
        n = len(X)
        folds: list[dict[str, Any]] = []
        fold = 1

        train_end = initial_train_size
        while train_end + val_size <= n:
            val_end = train_end + val_size

            folds.append({
                "fold": fold,
                "X_train": X.iloc[:train_end],
                "y_train": y.iloc[:train_end],
                "X_val": X.iloc[train_end:val_end],
                "y_val": y.iloc[train_end:val_end],
                "train_size": train_end,
                "val_size": val_end - train_end,
            })

            train_end = val_end
            fold += 1

        return folds

    # ── Info ─────────────────────────────────────────────

    @staticmethod
    def summary() -> str:
        """Return a plain-text explanation of time series CV."""
        return """\
TIME SERIES CROSS-VALIDATION — SUMMARY
======================================

What it does
  Replaces standard k-fold CV (which shuffles data randomly) with
  expanding-window chronological folds.  Each fold:
    - Train: matches 0 ... t
    - Valid: matches t+1 ... t+k
  The training set grows with each fold; validation is always the
  *next* unseen block of matches.

Why it matters for football prediction
  1. Feature leakage prevention — rolling stats, Elo, and league
     positions all depend on temporal ordering.  Random CV would
     train on future data and validate on past data, giving
     unrealistically optimistic metrics.
  2. Realistic simulation — production models predict upcoming
     matches, not randomly sampled ones.  Time series CV mirrors
     this deployment reality.
  3. Detects overfitting to temporal patterns — if a model performs
     well on random folds but poorly on chronological folds, it has
     learned time-specific noise (e.g., a particular season's quirks)
     rather than generalisable patterns.

Strategies available
  - Expanding window (TimeSeriesSplit): train grows, val is fixed-size.
  - Sliding window: train is fixed-size (most recent N matches).
  - Walk-forward: train grows, val is a contiguous block.

When to use each
  - Expanding window:  Default.  Good for most situations.
  - Sliding window:    When old data becomes irrelevant (e.g., 20+
                       years of data where tactics have changed).
  - Walk-forward:      Production simulation.  Most realistic but
                       most computationally expensive.
"""


# ═══════════════════════════════════════════════════════════
#  Drop-in replacement for feature_engineering.train_val_test_split
# ═══════════════════════════════════════════════════════════


def time_series_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratios: tuple[float, float, float] | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """Chronological train / validation / test split (no shuffle, no leakage).

    This is a drop-in replacement for
    ``feature_engineering.train_val_test_split`` that uses strict
    chronological ordering and returns the same dict schema.

    **Important:** The split is purely chronological — the oldest
    ``train_ratio`` fraction goes to training, the next ``val_ratio``
    to validation, and the most recent ``test_ratio`` to testing.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (assumed chronologically sorted).
    y : pd.Series
        Target vector.
    ratios : tuple[float, float, float], optional
        (train, val, test) ratios.  Defaults to ``config.data.split_ratios``.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).

    Returns
    -------
    dict[str, Any]
        ``X_train``, ``X_val``, ``X_test``, ``y_train``, ``y_val``, ``y_test``.
    """
    cfg = config or _global_config
    if ratios is None:
        ratios = cfg.data.split_ratios

    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    n = len(X)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
    }


# ═══════════════════════════════════════════════════════════
#  CV-aware hyper-parameter tuning helpers
# ═══════════════════════════════════════════════════════════


def create_time_series_folds(
    n_splits: int | None = None,
    gap: int | None = None,
    config: Any | None = None,
) -> SklearnTimeSeriesSplit:
    """Shortcut to create a ``TimeSeriesSplit`` for use with CV.

    Example::

        from src.time_series_cv import create_time_series_folds
        from sklearn.model_selection import GridSearchCV

        cv = create_time_series_folds(n_splits=5)
        searcher = GridSearchCV(model, param_grid, cv=cv)

    Parameters
    ----------
    n_splits : int, optional
        Number of CV folds.  Defaults to ``config.hyper_tune.cv_folds``.
    gap : int, optional
        Gap between train and val.  Defaults to 0.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    return TimeSeriesCrossValidator.get_cv(
        n_splits=n_splits or cfg.hyper_tune.cv_folds,
        gap=gap or _DEFAULT_GAP,
        config=cfg,
    )
