"""Wrapper classes and convenience functions for model calibration.

Provides:
  - CalibratedTemperatureWrapper (Phase 4)
  - CalibratedStatsModel (Phase 3)
  - CalibratedModel (main wrapper)
  - _fit_calibrators (shared helper)
  - calibrate_model (one-shot convenience)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.calibration.calibrators import (
    HybridTailCalibrator,
    IsotonicRegressionCalibrator,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
)
from src.calibration.utils import renormalise_probs

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Temperature Wrapper (Phase 4)
# ═══════════════════════════════════════════════════════════


class CalibratedTemperatureWrapper:
    """Wrap a Phase 4 ML model with temperature scaling calibration.

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
        """Predict calibrated probabilities via temperature scaling."""
        raw_probs = self.base_model.predict_proba(X)
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
    which returns a DataFrame with probability columns.

    NOTE: Does NOT use ``__getattr__`` delegation to avoid recursion issues
    with joblib serialization.

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
        """Predict with calibrated probabilities."""
        import pandas as pd  # noqa: PLC0415

        preds = self._base_model.predict_matches(df)

        raw_probs = np.column_stack([
            preds["away_win_prob"].values,
            preds["draw_prob"].values,
            preds["home_win_prob"].values,
        ])

        if isinstance(self._calibrator, TemperatureScalingCalibrator):
            p = np.clip(raw_probs, 1e-7, 1 - 1e-7)
            logits = np.log(p / (1.0 - p))
            logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            cal_probs = self._calibrator.transform(logits)
        else:
            cal_probs = self._calibrator.transform(raw_probs)

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
    method : Literal["platt", "isotonic", "hybrid"]
        Calibration method.
    n_classes : int
        Number of classes.  Default 3 (Away, Draw, Home).
    """

    def __init__(
        self,
        base_model: Any,
        method: Literal["platt", "isotonic", "hybrid"] = "platt",
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
        """
        logger.info("Fitting base model with calibration method='%s'", self.method)
        self.base_model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            val_probs = self.base_model.predict_proba(X_val)
            val_true = y_val
        else:
            split = int(len(X_train) * 0.8)
            X_cal = X_train[split:] if hasattr(X_train, "iloc") else X_train[split:]
            y_cal = y_train[split:] if hasattr(y_train, "iloc") else y_train[split:]
            X_fit = X_train[:split] if hasattr(X_train, "iloc") else X_train[:split]
            y_fit = y_train[:split] if hasattr(y_train, "iloc") else y_train[:split]

            self.base_model.fit(X_fit, y_fit)
            val_probs = self.base_model.predict_proba(X_cal)
            val_true = y_cal

        self._calibrators = _fit_calibrators(val_probs, val_true, self.n_classes, self.method)
        self._fitted = True
        logger.info("Calibration fitted — %s, %d classes", self.method, self.n_classes)
        return self

    # ── Predict ──────────────────────────────────────────

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """Predict hard class labels."""
        return np.argmax(self.predict_proba(X), axis=1)

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

        if self.method == "hybrid":
            hybrid_cal = self._calibrators[0]
            if hybrid_cal is not None:
                return hybrid_cal.transform(raw_probs)

        # Legacy per-class calibrator logic (platt / isotonic)
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
            "Calibration eval — log-loss: %.4f -> %.4f (Δ=%.4f)  |  "
            "Brier: %.4f -> %.4f (Δ=%.4f)",
            raw_ll, cal_ll, raw_ll - cal_ll,
            raw_brier, cal_brier, raw_brier - cal_brier,
        )
        return result


# ═══════════════════════════════════════════════════════════
#  Convenience functions
# ═══════════════════════════════════════════════════════════


def _fit_calibrators(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    method: str,
) -> list[Any]:
    """Fit calibrators for each class (or a single HybridTailCalibrator).

    For ``method="hybrid"``, returns a single-element list containing the
    ``HybridTailCalibrator`` instance.

    For ``"platt"`` and ``"isotonic"``, returns per-class calibrators.
    """
    if method == "hybrid":
        calibrator = HybridTailCalibrator(n_classes=n_classes)
        calibrator.fit(val_probs, y_val)
        return [calibrator]

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
