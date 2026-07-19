"""Calibrator classes — probability calibration methods.

Provides four calibrator implementations:

1. **IsotonicRegressionCalibrator** — non-parametric monotonic transform.
2. **PlattScalingCalibrator** — logistic regression on logit-transformed probs.
3. **HybridTailCalibrator** — blends Platt (mid-range) + Isotonic (tails).
4. **TemperatureScalingCalibrator** — single temperature on logits (NNs).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.calibration.utils import validate_probs_input, renormalise_probs

logger = logging.getLogger(__name__)


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

    # ── Properties ────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    # ── Fit ──────────────────────────────────────────────

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
        X, y = validate_probs_input(X, y, self.n_classes)
        self._classes_seen = set(np.unique(y))

        for c in range(self.n_classes):
            y_binary = (y == c).astype(np.float64)

            unique_vals = np.unique(y_binary)
            if len(unique_vals) == 1:
                self._calibrators[c] = None
                self._constant_values[c] = float(unique_vals[0])
                continue

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

    # ── Transform ─────────────────────────────────────────

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
                const_val = self._constant_values[c]
                if np.isfinite(const_val):
                    calibrated[:, c] = const_val
                else:
                    calibrated[:, c] = X[:, c]
            else:
                p = np.clip(X[:, c], 1e-7, 1 - 1e-7)
                calibrated[:, c] = calibrator.transform(p)

        calibrated = renormalise_probs(calibrated)
        return calibrated


class PlattScalingCalibrator:
    """Platt scaling (sigmoid) calibrator for probability calibration.

    Fits a separate logistic regression per class in a one-vs-rest fashion.
    Works well when the miscalibration pattern is roughly sigmoid-shaped.

    Parameters
    ----------
    n_classes : int
        Number of classes (default 3: Home Win, Draw, Away Win).
    max_iter : int
        Maximum iterations for the logistic regression solver (default 1000).
    random_state : int | None
        Random state for reproducibility (default None).
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

    # ── Properties ────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    # ── Fit ──────────────────────────────────────────────

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
        X, y = validate_probs_input(X, y, self.n_classes)
        self._classes_seen = set(np.unique(y))

        for c in range(self.n_classes):
            y_binary = (y == c).astype(int)

            unique_vals = np.unique(y_binary)
            if len(unique_vals) == 1:
                self._calibrators[c] = None
                self._constant_values[c] = float(unique_vals[0])
                continue

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

    # ── Transform ─────────────────────────────────────────

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

        calibrated = renormalise_probs(calibrated)
        return calibrated


class HybridTailCalibrator:
    """Tail-aware hybrid calibrator combining Platt scaling & isotonic regression.

    Standard calibrators (Platt, Isotonic) fit a single transform across the
    entire probability range.  This is suboptimal at the tails (0-10%, 90-100%).

    ``HybridTailCalibrator`` fits **both** calibrators per class and blends them:

    * **Tails (p < 0.10 or p > 0.90):**  Isotonic-weighted
    * **Mid-range (0.10 <= p <= 0.90):**  Platt-weighted
    * Smooth linear transition between the two regimes

    Parameters
    ----------
    n_classes : int
        Number of classes (default 3).
    tail_threshold : float
        Probability threshold defining the tail regions (default 0.10).
    mid_isotonic_weight : float
        Weight given to isotonic regression in the mid-range (default 0.30).
    max_iter : int
        Max iterations for Platt scaling (default 1000).
    random_state : int | None
        Random state for reproducibility (default None).
    """

    def __init__(
        self,
        n_classes: int = 3,
        tail_threshold: float = 0.10,
        mid_isotonic_weight: float = 0.30,
        max_iter: int = 1000,
        random_state: int | None = None,
    ) -> None:
        if not 0.0 < tail_threshold < 0.5:
            raise ValueError(f"tail_threshold must be in (0, 0.5), got {tail_threshold}")
        if not 0.0 <= mid_isotonic_weight <= 1.0:
            raise ValueError(f"mid_isotonic_weight must be in [0, 1], got {mid_isotonic_weight}")

        self.n_classes = n_classes
        self.tail_threshold = tail_threshold
        self.mid_isotonic_weight = mid_isotonic_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self._platt_calibrators: list[LogisticRegression | None] = [None] * n_classes
        self._iso_calibrators: list[IsotonicRegression | None] = [None] * n_classes
        self._constant_values: np.ndarray = np.full(n_classes, np.nan)
        self._fitted: bool = False

    # ── Properties ────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    # ── Fit ──────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> HybridTailCalibrator:
        """Fit both Platt and Isotonic calibrators per class."""
        X, y = validate_probs_input(X, y, self.n_classes)

        for c in range(self.n_classes):
            y_binary = (y == c).astype(int)

            unique_vals = np.unique(y_binary)
            if len(unique_vals) == 1:
                self._platt_calibrators[c] = None
                self._iso_calibrators[c] = None
                self._constant_values[c] = float(unique_vals[0])
                continue

            p = np.clip(X[:, c], 1e-7, 1 - 1e-7)

            # Platt scaling
            X_logit = np.log(p / (1.0 - p)).reshape(-1, 1)
            platt_cal = LogisticRegression(
                penalty=None, solver="lbfgs",
                max_iter=self.max_iter, random_state=self.random_state,
            )
            platt_cal.fit(X_logit, y_binary)
            self._platt_calibrators[c] = platt_cal

            # Isotonic regression
            iso_cal = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0,
            )
            iso_cal.fit(p, y_binary.astype(np.float64))
            self._iso_calibrators[c] = iso_cal

        self._fitted = True
        logger.info(
            "HybridTailCalibrator fitted — %d classes, %d samples, "
            "tail_threshold=%.2f, mid_isotonic_weight=%.2f",
            self.n_classes, len(X), self.tail_threshold, self.mid_isotonic_weight,
        )
        return self

    # ── Transform ─────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities.

        Blends Platt and Isotonic outputs based on probability region.
        """
        if not self._fitted:
            raise RuntimeError(
                "HybridTailCalibrator must be fitted before transforming."
            )

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected X of shape (n, {self.n_classes}), got {X.shape}"
            )

        tail = self.tail_threshold
        mid_iso_w = self.mid_isotonic_weight
        calibrated = np.zeros_like(X)

        for c in range(self.n_classes):
            platt_cal = self._platt_calibrators[c]
            iso_cal = self._iso_calibrators[c]

            if platt_cal is None and iso_cal is None:
                const_val = self._constant_values[c]
                calibrated[:, c] = const_val if np.isfinite(const_val) else X[:, c]
                continue

            p = np.clip(X[:, c], 1e-7, 1 - 1e-7)

            # Platt output
            X_logit = np.log(p / (1.0 - p)).reshape(-1, 1)
            platt_out = platt_cal.predict_proba(X_logit)[:, 1]  # type: ignore[union-attr]

            # Isotonic output
            iso_out = iso_cal.transform(p)  # type: ignore[union-attr]

            # Compute blend weights
            iso_weight = np.full_like(p, mid_iso_w)

            # Lower tail: p < tail
            lower_mask = p < tail
            if lower_mask.any():
                iso_weight[lower_mask] = 1.0 - (p[lower_mask] / tail) * (1.0 - mid_iso_w)

            # Upper tail: p > 1 - tail
            upper_mask = p > (1.0 - tail)
            if upper_mask.any():
                iso_weight[upper_mask] = (
                    ((p[upper_mask] - (1.0 - tail)) / tail) * (1.0 - mid_iso_w) + mid_iso_w
                )

            calibrated[:, c] = iso_weight * iso_out + (1.0 - iso_weight) * platt_out

        calibrated = renormalise_probs(calibrated)
        return calibrated

    # ── Evaluate ──────────────────────────────────────────

    def evaluate_calibration(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Compare raw vs calibrated probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_classes)
            Raw predicted probabilities.
        y : np.ndarray of shape (n,)
            True class labels.

        Returns
        -------
        dict with raw/calibrated log-loss, Brier, ECE, and per-region ECE.
        """
        raw_probs = X
        cal_probs = self.transform(X)
        y_true_int = np.asarray(y, dtype=np.int64)

        raw_ll = float(log_loss(y, raw_probs))
        cal_ll = float(log_loss(y, cal_probs))
        y_onehot = np.eye(self.n_classes)[y_true_int]
        raw_brier = float(np.mean(np.sum((raw_probs - y_onehot) ** 2, axis=1)))
        cal_brier = float(np.mean(np.sum((cal_probs - y_onehot) ** 2, axis=1)))

        # Per-region tail ECE
        pred_class = np.argmax(cal_probs, axis=1).astype(np.int64)
        pred_conf = np.max(cal_probs, axis=1)
        correct = (pred_class == y).astype(float)

        tail = self.tail_threshold
        lower_conf = pred_conf < tail
        upper_conf = pred_conf > (1.0 - tail)
        mid_conf = ~lower_conf & ~upper_conf

        def _ece(mask: np.ndarray) -> float:
            if mask.sum() < 2:
                return 0.0
            return float(np.abs(correct[mask] - pred_conf[mask]).mean())

        return {
            "raw_log_loss": round(raw_ll, 4),
            "calibrated_log_loss": round(cal_ll, 4),
            "log_loss_improvement": round(raw_ll - cal_ll, 4),
            "raw_brier": round(raw_brier, 4),
            "calibrated_brier": round(cal_brier, 4),
            "brier_improvement": round(raw_brier - cal_brier, 4),
            "tail_threshold": self.tail_threshold,
            "mid_isotonic_weight": self.mid_isotonic_weight,
            "low_tail_threshold": tail,
            "high_tail_threshold": 1.0 - tail,
            "low_tail_samples": int(lower_conf.sum()),
            "mid_samples": int(mid_conf.sum()),
            "high_tail_samples": int(upper_conf.sum()),
            "low_tail_ece": round(_ece(lower_conf), 4),
            "mid_ece": round(_ece(mid_conf), 4),
            "high_tail_ece": round(_ece(upper_conf), 4),
        }


class TemperatureScalingCalibrator:
    """Temperature scaling calibrator for neural network outputs.

    Uses a single scalar parameter T > 0 that scales logits before softmax::

        calibrated_probs = softmax(logits / T)

    The temperature is optimised to minimise NLL on a held-out validation set.

    Parameters
    ----------
    n_classes : int
        Number of classes (default 3).
    max_iter : int
        Maximum iterations for L-BFGS optimiser (default 100).
    lr : float
        Learning rate (not used with L-BFGS, kept for API compat).
    init_temp : float
        Initial temperature value (default 1.0).
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

    # ── Properties ────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._fitted

    @property
    def temperature(self) -> float:
        """The learned temperature parameter."""
        return self.temperature_

    # ── Fit ──────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TemperatureScalingCalibrator:
        """Fit the temperature parameter using L-BFGS optimisation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_classes)
            Logits (pre-softmax outputs) from the neural network.
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
            raise ValueError(f"X must have {self.n_classes} columns, got {X.shape[1]}")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)} samples) and y ({len(y)} samples) must match")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values")

        unique_classes = np.unique(y)
        if len(unique_classes) <= 1:
            logger.warning(
                "Only %d class(es) present in y — setting temperature to 1.0",
                len(unique_classes),
            )
            self.temperature_ = self.init_temp
            self._fitted = True
            return self

        def _nll(log_t: float) -> float:
            t = np.exp(log_t)
            scaled_logits = X / t
            s_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_s = np.exp(s_logits)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs = np.clip(probs, 1e-15, 1.0)
            return float(-np.mean(np.log(probs[np.arange(len(y)), y])))

        def _grad(log_t: float) -> float:
            t = np.exp(log_t)
            scaled_logits = X / t
            s_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            exp_s = np.exp(s_logits)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            sum_p_scaled = (probs * scaled_logits).sum(axis=1)
            true_scaled = scaled_logits[np.arange(len(y)), y]
            return float(np.mean(true_scaled - sum_p_scaled))

        result = scipy.optimize.minimize(
            _nll,
            x0=np.log(self.init_temp),
            method="L-BFGS-B",
            jac=_grad,
            options={"maxiter": self.max_iter, "ftol": 1e-7},
        )

        self.temperature_ = float(np.exp(result.x[0]))

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

    # ── Transform ─────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform logits into calibrated probabilities using temperature scaling."""
        if not self._fitted:
            raise RuntimeError(
                "TemperatureScalingCalibrator must be fitted before transforming."
            )

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_classes:
            raise ValueError(f"Expected X of shape (n, {self.n_classes}), got {X.shape}")

        if not np.all(np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            logger.warning("X contains NaN or Inf values — replaced with finite values")

        if self.n_classes == 1:
            return np.ones((len(X), 1), dtype=np.float64)

        scaled = X / self.temperature_
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    # ── Evaluate ──────────────────────────────────────────

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
        before = self._softmax(X / 1.0)
        nll_before = float(log_loss(y, before))
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
