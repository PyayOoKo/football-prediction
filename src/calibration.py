"""
Calibration — Platt scaling & isotonic regression for probability calibration.

Models (especially tree-based ones like XGBoost) often produce miscalibrated
probabilities — they may be overconfident (predicting 80% when accuracy is 65%)
or underconfident.  Calibration fixes this by learning a monotonic transform
from raw model output to well-calibrated probabilities.

Three calibrator classes are provided:

1. **PlattScalingCalibrator:**  Fits a logistic regression on the model's
   raw predict_proba output.  Works well when the miscalibration pattern is
   sigmoid-shaped.  Best for small validation sets.

2. **IsotonicRegressionCalibrator:**  Fits a non-parametric monotonic function.
   More flexible than Platt — captures any monotonic miscalibration pattern.
   Requires more data to avoid overfitting.

3. **TemperatureScalingCalibrator:**  Fits a single temperature parameter on
   neural network logits.  The simplest multi-class calibration method.

Usage
-----
::

    from src.calibration import IsotonicRegressionCalibrator

    calibrator = IsotonicRegressionCalibrator(n_classes=3)
    calibrator.fit(val_probs, y_val)
    cal_probs = calibrator.transform(raw_probs)

Or wrap an existing sklearn model with the ``CalibratedModel`` wrapper::

    from src.calibration import CalibratedModel
    model = CalibratedModel(xgb_model, method="platt")
    model.fit(X_train, y_train, X_val, y_val)
    probs = model.predict_proba(X_test)
    metrics = model.evaluate_calibration(X_test, y_test)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import scipy.optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Shared helpers for calibrator classes
# ═══════════════════════════════════════════════════════════


def _validate_probs_input(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert calibrator input arrays.

    Parameters
    ----------
    X : probability array of shape (n, n_classes)
    y : label array of shape (n,)
    n_classes : expected number of columns

    Returns
    -------
    tuple of (X, y) as float64 and int64 arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if X.shape[1] != n_classes:
        raise ValueError(f"X must have {n_classes} columns, got {X.shape[1]}")
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)} samples) and y ({len(y)} samples) must match")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf values")

    return X, y


def _renormalise_probs(probs: np.ndarray) -> np.ndarray:
    """Normalise probability rows to sum to 1.0.

    Rows that sum to zero (degenerate case) are replaced with a uniform
    distribution to keep downstream consumers from seeing NaN or zero rows.
    """
    probs = np.asarray(probs, dtype=np.float64)
    row_sums = probs.sum(axis=1)
    zero_mask = row_sums <= 0
    if zero_mask.any():
        probs = probs.copy()
        probs[zero_mask] = 1.0 / probs.shape[1]
        row_sums = row_sums.copy()
        row_sums[zero_mask] = 1.0
    return probs / row_sums[:, np.newaxis]


# ═══════════════════════════════════════════════════════════
#  Public Calibrator Classes
# ═══════════════════════════════════════════════════════════


class IsotonicRegressionCalibrator:
    """Isotonic regression calibrator for probability calibration.

    Fits a separate isotonic regression per class in a one-vs-rest fashion.
    Isotonic regression learns a non-parametric monotonic function, making it
    flexible enough to capture any monotonic miscalibration pattern.

    Parameters
    ----------
    n_classes : int
        Number of classes (default 3: Home Win, Draw, Away Win).
    out_of_bounds : str
        How to handle out-of-bounds values in ``transform``.
        ``"clip"`` clamps to [min(y), max(y)] seen during fit.
        ``"nan"`` returns NaN for out-of-bounds values.
        Default ``"clip"``.

    Examples
    --------
    ::

        calibrator = IsotonicRegressionCalibrator(n_classes=3)
        calibrator.fit(val_probs, y_val)
        cal_probs = calibrator.transform(raw_probs)
    """

    def __init__(
        self,
        n_classes: int = 3,
        out_of_bounds: str = "clip",
    ) -> None:
        self.n_classes = n_classes
        self.out_of_bounds = out_of_bounds
        self._calibrators: list[IsotonicRegression | None] = [None] * n_classes
        self._fitted: bool = False
        self._classes_seen: set[int] = set()
        self._constant_values: np.ndarray = np.full(n_classes, np.nan)

    # ── Properties ────────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    # ── Fit ──────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> IsotonicRegressionCalibrator:
        """Fit isotonic regression calibrators for each class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities from the base model.
        y : np.ndarray of shape (n_samples,)
            True class labels (0, 1, 2).

        Returns
        -------
        IsotonicRegressionCalibrator
            Self, fitted.
        """
        X, y = _validate_probs_input(X, y, self.n_classes)
        self._classes_seen = set(np.unique(y))

        for c in range(self.n_classes):
            y_binary = (y == c).astype(np.float64)

            # Edge case: only one class present in y for this calibrator
            unique_vals = np.unique(y_binary)
            if len(unique_vals) == 1:
                # All predictions are either 0 or 1 for this class
                self._calibrators[c] = None
                self._constant_values[c] = float(unique_vals[0])
                continue

            # Clip probabilities to avoid numerical instability
            p = np.clip(X[:, c], 1e-7, 1 - 1e-7)

            calibrator = IsotonicRegression(
                out_of_bounds=self.out_of_bounds,
                y_min=0.0,
                y_max=1.0,
            )
            calibrator.fit(p, y_binary)
            self._calibrators[c] = calibrator

        self._fitted = True
        logger.debug(
            "IsotonicRegressionCalibrator fitted — %d classes, %d samples",
            self.n_classes, len(X),
        )
        return self

    # ── Transform ─────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            Raw predicted probabilities from the base model.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Calibrated probabilities that sum to 1.0 per row.
        """
        if not self._fitted:
            raise RuntimeError(
                "IsotonicRegressionCalibrator must be fitted before transforming."
            )

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected X of shape (n, {self.n_classes}), got {X.shape}"
            )

        calibrated = np.zeros_like(X)

        for c in range(self.n_classes):
            calibrator = self._calibrators[c]
            if calibrator is None:
                # Edge case: single class seen during fit
                const_val = self._constant_values[c]
                if np.isfinite(const_val):
                    calibrated[:, c] = const_val
                else:
                    calibrated[:, c] = X[:, c]
            else:
                p = np.clip(X[:, c], 1e-7, 1 - 1e-7)
                calibrated[:, c] = calibrator.transform(p)

        # Renormalise to sum to 1.0
        calibrated = _renormalise_probs(calibrated)
        return calibrated


class PlattScalingCalibrator:
    """Platt scaling (sigmoid) calibrator for probability calibration.

    Fits a separate logistic regression per class in a one-vs-rest fashion.
    The raw probabilities are logit-transformed before fitting, so the
    calibrator learns a sigmoid-shaped mapping.  Works well when the
    miscalibration pattern is roughly sigmoid-shaped and requires less data
    than isotonic regression.

    Parameters
    ----------
    n_classes : int
        Number of classes (default 3: Home Win, Draw, Away Win).
    max_iter : int
        Maximum iterations for the logistic regression solver (default 1000).
    random_state : int | None
        Random state for reproducibility (default None).

    Examples
    --------
    ::

        calibrator = PlattScalingCalibrator(n_classes=3)
        calibrator.fit(val_probs, y_val)
        cal_probs = calibrator.transform(raw_probs)
    """

    def __init__(
        self,
        n_classes: int = 3,
        max_iter: int = 1000,
        random_state: int | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.random_state = random_state
        self._calibrators: list[LogisticRegression | None] = [None] * n_classes
        self._fitted: bool = False
        self._classes_seen: set[int] = set()
        self._constant_values: np.ndarray = np.full(n_classes, np.nan)

    # ── Properties ────────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    # ── Fit ──────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> PlattScalingCalibrator:
        """Fit Platt scaling calibrators for each class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities from the base model.
        y : np.ndarray of shape (n_samples,)
            True class labels (0, 1, 2).

        Returns
        -------
        PlattScalingCalibrator
            Self, fitted.
        """
        X, y = _validate_probs_input(X, y, self.n_classes)
        self._classes_seen = set(np.unique(y))

        for c in range(self.n_classes):
            y_binary = (y == c).astype(int)

            # Edge case: only one class present
            unique_vals = np.unique(y_binary)
            if len(unique_vals) == 1:
                self._calibrators[c] = None
                self._constant_values[c] = float(unique_vals[0])
                continue

            # Logit transform with clipping for numerical stability
            p = np.clip(X[:, c], 1e-7, 1 - 1e-7)
            X_logit = np.log(p / (1.0 - p)).reshape(-1, 1)

            calibrator = LogisticRegression(
                penalty=None,
                solver="lbfgs",
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            calibrator.fit(X_logit, y_binary)
            self._calibrators[c] = calibrator

        self._fitted = True
        logger.debug(
            "PlattScalingCalibrator fitted — %d classes, %d samples",
            self.n_classes, len(X),
        )
        return self

    # ── Transform ─────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            Raw predicted probabilities from the base model.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Calibrated probabilities that sum to 1.0 per row.
        """
        if not self._fitted:
            raise RuntimeError(
                "PlattScalingCalibrator must be fitted before transforming."
            )

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected X of shape (n, {self.n_classes}), got {X.shape}"
            )

        calibrated = np.zeros_like(X)

        for c in range(self.n_classes):
            calibrator = self._calibrators[c]
            if calibrator is None:
                const_val = self._constant_values[c]
                if np.isfinite(const_val):
                    calibrated[:, c] = const_val
                else:
                    calibrated[:, c] = X[:, c]
            else:
                p = np.clip(X[:, c], 1e-7, 1 - 1e-7)
                X_logit = np.log(p / (1.0 - p)).reshape(-1, 1)
                calibrated[:, c] = calibrator.predict_proba(X_logit)[:, 1]

        # Renormalise to sum to 1.0
        calibrated = _renormalise_probs(calibrated)
        return calibrated


class TemperatureScalingCalibrator:
    """Temperature scaling calibrator for neural network outputs.

    Temperature scaling is the simplest extension of Platt scaling to
    multi-class settings.  It uses a single scalar parameter T > 0 (the
    temperature) that scales the logits (pre-softmax outputs) before the
    softmax function::

        calibrated_probs = softmax(logits / T)

    A higher temperature (> 1) produces softer, more uniform probabilities
    (reducing overconfidence), while a lower temperature (< 1) produces
    sharper probabilities (increasing confidence).

    The temperature is optimised to minimise the negative log-likelihood
    on a held-out validation set using L-BFGS.

    Parameters
    ----------
    n_classes : int
        Number of classes (default 3: Home Win, Draw, Away Win).
    max_iter : int
        Maximum iterations for the L-BFGS optimiser (default 100).
    lr : float
        Learning rate for gradient-based optimisation (not used with L-BFGS,
        kept for API compatibility).
    init_temp : float
        Initial temperature value (default 1.0).

    Examples
    --------
    ::

        calibrator = TemperatureScalingCalibrator(n_classes=3)
        calibrator.fit(val_logits, y_val)
        cal_probs = calibrator.transform(raw_logits)
    """

    def __init__(
        self,
        n_classes: int = 3,
        max_iter: int = 100,
        lr: float = 0.01,
        init_temp: float = 1.0,
    ) -> None:
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.lr = lr
        self.init_temp = init_temp

        self.temperature_: float = init_temp
        self._fitted: bool = False

    # ── Properties ────────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    @property
    def temperature(self) -> float:
        """The learned temperature parameter."""
        return self.temperature_

    # ── Fit ──────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TemperatureScalingCalibrator:
        """Fit the temperature parameter using L-BFGS optimisation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            **Logits** (pre-softmax outputs) from the neural network.
            These are the raw model outputs *before* softmax is applied.
        y : np.ndarray of shape (n_samples,)
            True class labels (0, 1, 2).

        Returns
        -------
        TemperatureScalingCalibrator
            Self, fitted.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if X.shape[1] != self.n_classes:
            raise ValueError(
                f"X must have {self.n_classes} columns, got {X.shape[1]}"
            )
        if len(X) != len(y):
            raise ValueError(
                f"X ({len(X)} samples) and y ({len(y)} samples) must match"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values")

        unique_classes = np.unique(y)

        # Edge case: only a single class present
        if len(unique_classes) <= 1:
            logger.warning(
                "Only %d class(es) present in y — setting temperature to 1.0",
                len(unique_classes),
            )
            self.temperature_ = self.init_temp
            self._fitted = True
            return self

        # ── Define loss and gradient ──────────────────────

        def _nll(log_t: float) -> float:
            """Negative log-likelihood given log(temperature)."""
            t = np.exp(log_t)
            scaled_logits = X / t

            # Numerically stable softmax
            s_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_s = np.exp(s_logits)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)

            # Clip to avoid log(0)
            probs = np.clip(probs, 1e-15, 1.0)

            nll = -np.mean(np.log(probs[np.arange(len(y)), y]))
            return nll

        def _grad(log_t: float) -> float:
            """Gradient of NLL w.r.t. log(temperature).

            Derivation::

                NLL  = -log(softmax(z/T)_y)
                     = -z_y/T + log(sum(exp(z_j/T)))

                d(NLL)/d(log T)  =  z_y/T - sum(p_j * z_j/T)
                                 =  true_scaled - sum_p_scaled
            """
            t = np.exp(log_t)
            scaled_logits = X / t

            # Softmax probabilities
            s_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_s = np.exp(s_logits)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)

            sum_p_scaled = (probs * scaled_logits).sum(axis=1)
            true_scaled = scaled_logits[np.arange(len(y)), y]
            return float(np.mean(true_scaled - sum_p_scaled))

        # ── Optimise in log-space to enforce T > 0 ────────
        result = scipy.optimize.minimize(
            _nll,
            x0=np.log(self.init_temp),
            method="L-BFGS-B",
            jac=_grad,
            options={"maxiter": self.max_iter, "ftol": 1e-7},
        )

        self.temperature_ = float(np.exp(result.x[0]))

        # Edge case: temperature exploded (all predictions same class)
        if not np.isfinite(self.temperature_) or self.temperature_ > 1e6:
            logger.warning(
                "Temperature exploded to %s — resetting to init_temp=%.2f",
                self.temperature_, self.init_temp,
            )
            self.temperature_ = self.init_temp

        self._fitted = True
        logger.info(
            "TemperatureScalingCalibrator fitted — T=%.4f (nll=%.4f, %d samples)",
            self.temperature_, result.fun, len(X),
        )
        return self

    # ── Transform ─────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform logits into calibrated probabilities using temperature scaling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            **Logits** (pre-softmax outputs) from the neural network.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Calibrated probabilities that sum to 1.0 per row.
        """
        if not self._fitted:
            raise RuntimeError(
                "TemperatureScalingCalibrator must be fitted before transforming."
            )

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected X of shape (n, {self.n_classes}), got {X.shape}"
            )

        if not np.all(np.isfinite(X)):
            # Replace NaN/Inf in logits
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            logger.warning("X contains NaN or Inf values — replaced with finite values")

        # Edge case: single class — probability is always 1.0 for that class
        if self.n_classes == 1:
            return np.ones((len(X), 1), dtype=np.float64)

        # Apply temperature scaling: softmax(logits / T)
        scaled = X / self.temperature_
        scaled = scaled - scaled.max(axis=1, keepdims=True)  # numerical stability
        exp_s = np.exp(scaled)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)

        return probs

    # ── Evaluate ─────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate the temperature scaling calibration.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            Logits (pre-softmax outputs).
        y : np.ndarray of shape (n_samples,)
            True class labels.

        Returns
        -------
        dict with ``temperature``, ``nll_before``, ``nll_after``.
        """
        # Before calibration (T=1.0)
        before = self._softmax(X / 1.0)
        nll_before = float(log_loss(y, before))

        # After calibration
        after = self.transform(X)
        nll_after = float(log_loss(y, after))

        return {
            "temperature": round(self.temperature_, 4),
            "nll_before": round(nll_before, 4),
            "nll_after": round(nll_after, 4),
            "nll_improvement": round(nll_before - nll_after, 4),
        }

    @staticmethod
    def _softmax(X: np.ndarray) -> np.ndarray:
        """Compute softmax along axis=1 with numerical stability."""
        X = X - X.max(axis=1, keepdims=True)
        exp_X = np.exp(X)
        return exp_X / exp_X.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════
#  Temperature Wrapper (Phase 4)
# ═══════════════════════════════════════════════════════════


class CalibratedTemperatureWrapper:
    """Wrap a Phase 4 ML model with temperature scaling calibration.

    Temperature scaling divides logits by T before softmax. This wrapper
    stores both the original model and the temperature calibrator, and
    applies temperature scaling in ``predict_proba``.

    Parameters
    ----------
    base_model : Any
        Sklearn-compatible model with ``predict`` and ``predict_proba``.
    calibrator : TemperatureScalingCalibrator
        Fitted temperature calibrator.
    """

    def __init__(self, base_model: Any, calibrator: TemperatureScalingCalibrator) -> None:
        self.base_model = base_model
        self._calibrator = calibrator

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict hard class labels using calibrated probabilities."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict calibrated probabilities via temperature scaling.

        Converts raw probabilities to logits, applies temperature scaling,
        and returns softmax-calibrated probabilities.
        """
        raw_probs = self.base_model.predict_proba(X)
        # Convert probs to logits
        p = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1.0 - p))
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return self._calibrator.transform(logits)


# ═══════════════════════════════════════════════════════════
#  Stats Model Wrapper (Phase 3)
# ═══════════════════════════════════════════════════════════


class CalibratedStatsModel:
    """Wrap a Phase 3 statistical model with probability calibration.

    Phase 3 models (Poisson, Dixon-Coles, Elo) use ``predict_matches()``
    which returns a DataFrame with probability columns. This wrapper stores
    the original model and a fitted calibrator, then transforms the
    probability columns on every call.

    NOTE: This wrapper does NOT use ``__getattr__`` delegation to the base
    model to avoid recursion issues with joblib serialization. All forwarded
    methods must be explicitly listed in ``_forwarded_methods``.

    Parameters
    ----------
    base_model : Any
        The original Phase 3 model.
    calibrator : Any
        Fitted calibrator instance.
    method : str
        Display name of the calibration method.
    """

    _forwarded_methods = {"add_features", "fit", "process_matches"}

    def __init__(
        self,
        base_model: Any,
        calibrator: Any,
        method: str,
    ) -> None:
        self._base_model = base_model
        self._calibrator = calibrator
        self._method = method

    def predict_matches(self, df: Any) -> Any:
        """Predict with calibrated probabilities.

        Calls the original ``predict_matches()``, then transforms
        the probability columns using the fitted calibrator.

        Returns
        -------
        pd.DataFrame with updated ``away_win_prob``, ``draw_prob``,
        ``home_win_prob`` columns.
        """
        import pandas as pd

        preds = self._base_model.predict_matches(df)

        # Build probability matrix from the raw predictions
        raw_probs = np.column_stack([
            preds["away_win_prob"].values,
            preds["draw_prob"].values,
            preds["home_win_prob"].values,
        ])

        # Apply calibration
        if isinstance(self._calibrator, TemperatureScalingCalibrator):
            p = np.clip(raw_probs, 1e-7, 1 - 1e-7)
            logits = np.log(p / (1.0 - p))
            logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            cal_probs = self._calibrator.transform(logits)
        else:
            cal_probs = self._calibrator.transform(raw_probs)

        # Update the prediction DataFrame
        preds["away_win_prob"] = cal_probs[:, 0]
        preds["draw_prob"] = cal_probs[:, 1]
        preds["home_win_prob"] = cal_probs[:, 2]

        return preds

    def __getattr__(self, name: str) -> Any:
        """Forward whitelisted methods to the base model."""
        if name in self._forwarded_methods and hasattr(self, "_base_model"):
            return getattr(self._base_model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# ═══════════════════════════════════════════════════════════
#  Calibrated Model Wrapper
# ═══════════════════════════════════════════════════════════


class CalibratedModel:
    """Wrap a model with probability calibration.

    Parameters
    ----------
    base_model : Any
        Any sklearn-compatible model with ``predict`` and ``predict_proba``.
    method : Literal["platt", "isotonic"]
        Calibration method.  ``"platt"`` uses logistic regression (sigmoid
        mapping).  ``"isotonic"`` uses isotonic regression (non-parametric).
        Default ``"platt"``.
    n_classes : int
        Number of classes.  Default 3 (Away, Draw, Home).
    """

    def __init__(
        self,
        base_model: Any,
        method: Literal["platt", "isotonic"] = "platt",
        n_classes: int = 3,
    ) -> None:
        self.base_model = base_model
        self.method = method
        self.n_classes = n_classes
        self._calibrators: list[Any | None] = [None] * n_classes
        self._fitted: bool = False

    # ── Properties ────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the model + calibrators have been fitted."""
        return self._fitted

    # ── Fit ──────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray | Any,
        y_train: np.ndarray,
        X_val: np.ndarray | Any | None = None,
        y_val: np.ndarray | None = None,
    ) -> CalibratedModel:
        """Fit the base model and calibrate on validation data.

        If ``X_val`` and ``y_val`` are provided, calibration is learned on
        the validation set.  Otherwise, a train/val split is performed on the
        training data (last 20% used for calibration).

        Parameters
        ----------
        X_train, y_train : training data for base model.
        X_val, y_val : optional validation data for calibration.

        Returns
        -------
        CalibratedModel
            Self, fitted.
        """
        # ── 1. Fit base model ────────────────────────────
        logger.info(
            "Fitting base model with calibration method='%s'", self.method,
        )
        self.base_model.fit(X_train, y_train)

        # ── 2. Get validation predictions for calibration ─
        if X_val is not None and y_val is not None:
            val_probs = self.base_model.predict_proba(X_val)
            val_true = y_val
        else:
            # Hold out last 20% of training data for calibration
            split = int(len(X_train) * 0.8)
            X_cal = X_train[split:] if hasattr(X_train, "iloc") else X_train[split:]
            y_cal = y_train[split:] if hasattr(y_train, "iloc") else y_train[split:]
            X_fit = X_train[:split] if hasattr(X_train, "iloc") else X_train[:split]
            y_fit = y_train[:split] if hasattr(y_train, "iloc") else y_train[:split]

            self.base_model.fit(X_fit, y_fit)
            val_probs = self.base_model.predict_proba(X_cal)
            val_true = y_cal

        # ── 3. Fit calibrators using shared helper ─────
        self._calibrators = _fit_calibrators(val_probs, val_true, self.n_classes, self.method)

        self._fitted = True
        logger.info("Calibration fitted — %s, %d classes", self.method, self.n_classes)
        return self

    # ── Predict ──────────────────────────────────────────

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict hard class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict calibrated probabilities.

        Returns
        -------
        np.ndarray of shape (n, n_classes)
            Calibrated probabilities that sum to 1.0 per row.
        """
        if not self._fitted:
            raise RuntimeError("CalibratedModel must be fitted before predicting.")

        raw_probs = self.base_model.predict_proba(X)
        calibrated = np.zeros_like(raw_probs)

        for c in range(self.n_classes):
            calibrator = self._calibrators[c]
            if calibrator is None:
                calibrated[:, c] = raw_probs[:, c]
                continue

            if self.method == "platt":
                p = np.clip(raw_probs[:, c], 1e-7, 1 - 1e-7)
                X_calib = np.log(p / (1.0 - p)).reshape(-1, 1)
                calibrated[:, c] = calibrator.predict_proba(X_calib)[:, 1]
            else:
                calibrated[:, c] = calibrator.transform(raw_probs[:, c])

        # Renormalise to sum to 1.0
        row_sums = calibrated.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        calibrated = calibrated / row_sums[:, np.newaxis]

        return calibrated

    # ── Evaluation ───────────────────────────────────────

    def evaluate_calibration(
        self,
        X_test: np.ndarray | Any,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Compare raw vs calibrated probabilities.

        Parameters
        ----------
        X_test, y_test : test data.

        Returns
        -------
        dict with ``raw_log_loss``, ``calibrated_log_loss``,
        ``raw_brier``, ``calibrated_brier``, and ``improvement``.
        """
        raw_probs = self.base_model.predict_proba(X_test)
        cal_probs = self.predict_proba(X_test)

        y_true_onehot = np.zeros((len(y_test), self.n_classes))
        for i, v in enumerate(y_test):
            y_true_onehot[i, int(v)] = 1

        raw_ll = log_loss(y_test, raw_probs)
        cal_ll = log_loss(y_test, cal_probs)

        raw_brier = float(np.mean(np.sum((raw_probs - y_true_onehot) ** 2, axis=1)))
        cal_brier = float(np.mean(np.sum((cal_probs - y_true_onehot) ** 2, axis=1)))

        result = {
            "raw_log_loss": round(raw_ll, 4),
            "calibrated_log_loss": round(cal_ll, 4),
            "log_loss_improvement": round(raw_ll - cal_ll, 4),
            "raw_brier": round(raw_brier, 4),
            "calibrated_brier": round(cal_brier, 4),
            "brier_improvement": round(raw_brier - cal_brier, 4),
        }

        logger.info(
            "Calibration eval — log-loss: %.4f → %.4f (Δ=%.4f)  |  "
            "Brier: %.4f → %.4f (Δ=%.4f)",
            raw_ll, cal_ll, raw_ll - cal_ll,
            raw_brier, cal_brier, raw_brier - cal_brier,
        )
        return result


# ═══════════════════════════════════════════════════════════
#  Convenience: calibrate any trained model
# ═══════════════════════════════════════════════════════════


def _fit_calibrators(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    method: str,
) -> list[Any]:
    """Fit calibrators for each class."""
    calibrators: list[Any] = []
    for c in range(n_classes):
        if method == "platt":
            cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            p = np.clip(val_probs[:, c], 1e-7, 1 - 1e-7)
            X_c = np.log(p / (1.0 - p)).reshape(-1, 1)
            cal.fit(X_c, (y_val == c).astype(int))
        elif method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(val_probs[:, c], (y_val == c).astype(int))
        else:
            logger.warning("Unknown calibration method '%s' — falling back to Platt", method)
            cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            p = np.clip(val_probs[:, c], 1e-7, 1 - 1e-7)
            X_c = np.log(p / (1.0 - p)).reshape(-1, 1)
            cal.fit(X_c, (y_val == c).astype(int))
        calibrators.append(cal)
    return calibrators


def calibrate_model(
    model: Any,
    X_val: np.ndarray | Any,
    y_val: np.ndarray,
    method: Literal["platt", "isotonic"] = "platt",
    n_classes: int = 3,
) -> CalibratedModel:
    """Wrap an already-trained model with calibration in one call.

    Unlike ``CalibratedModel.fit()``, this does **not** refit the base model.
    Use this when you already have a trained model and just need to add
    calibration on top.

    Parameters
    ----------
    model : Any
        Already-trained sklearn-compatible model.
    X_val, y_val : validation data for fitting the calibrator.
    method : str
        ``"platt"`` or ``"isotonic"``.
    n_classes : int
        Number of classes (default 3).

    Returns
    -------
    CalibratedModel
        The wrapped model with calibrators fitted.
    """
    calibrated = CalibratedModel(base_model=model, method=method, n_classes=n_classes)
    val_probs = model.predict_proba(X_val)
    calibrated._calibrators = _fit_calibrators(val_probs, y_val, n_classes, method)
    calibrated._fitted = True
    logger.info("Model calibrated via %s on %d validation samples", method, len(y_val))
    return calibrated


# ═══════════════════════════════════════════════════════════
#  Calibration curve utilities
# ═══════════════════════════════════════════════════════════


def calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute a reliability (calibration) curve.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        True class labels (0, 1, 2).
    y_prob : np.ndarray of shape (n, 3)
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    dict with ``bin_centers``, ``accuracies``, ``confidences``, ``counts``.
    """
    pred_class = np.argmax(y_prob, axis=1)
    pred_conf = np.max(y_prob, axis=1)
    correct = (pred_class == y_true).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = np.zeros(n_bins)
    confidences = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (pred_conf >= bins[i]) & (pred_conf < bins[i + 1])
        count = in_bin.sum()
        counts[i] = count
        if count > 0:
            accuracies[i] = correct[in_bin].mean()
            confidences[i] = pred_conf[in_bin].mean()

    return {
        "bin_centers": bin_centers,
        "accuracies": accuracies,
        "confidences": confidences,
        "counts": counts,
    }


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> str:
    """Generate a human-readable calibration report.

    Parameters
    ----------
    y_true : true labels
    y_prob : predicted probabilities
    model_name : optional label

    Returns
    -------
    str — formatted report.
    """
    from sklearn.metrics import brier_score_loss, log_loss

    ll = log_loss(y_true, y_prob)
    y_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

    curve = calibration_curve(y_true, y_prob)
    ece = float(np.mean(
        curve["counts"] / curve["counts"].sum() * np.abs(curve["accuracies"] - curve["confidences"])
    ))

    lines = [f"Calibration Report — {model_name}",
             f"{'=' * 50}",
             f"  Log-loss:      {ll:.4f}",
             f"  Brier score:   {brier:.4f}",
             f"  ECE:           {ece:.4f}  (Expected Calibration Error)",
             "",
             f"  {'Bin':<8} {'Count':<8} {'Accuracy':<10} {'Confidence':<12} {'Gap':<8}",
             f"  {'-' * 46}"]
    for i in range(len(curve["bin_centers"])):
        if curve["counts"][i] > 0:
            gap = abs(curve["accuracies"][i] - curve["confidences"][i])
            lines.append(
                f"  {curve['bin_centers'][i]:<8.2f} {int(curve['counts'][i]):<8} "
                f"{curve['accuracies'][i]:<10.3f} {curve['confidences'][i]:<12.3f} {gap:<8.3f}"
            )

    return "\n".join(lines)
