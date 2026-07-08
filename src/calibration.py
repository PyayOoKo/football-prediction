"""
Calibration — Platt scaling & isotonic regression for probability calibration.

Models (especially tree-based ones like XGBoost) often produce miscalibrated
probabilities — they may be overconfident (predicting 80% when accuracy is 65%)
or underconfident.  Calibration fixes this by learning a monotonic transform
from raw model output to well-calibrated probabilities.

Two methods are supported:

1. **Platt scaling (sigmoid):**  Fits a logistic regression on the model's
   raw predict_proba output.  Works well when the miscalibration pattern is
   sigmoid-shaped.  Best for small validation sets.

2. **Isotonic regression:**  Fits a non-parametric monotonic function.
   More flexible than Platt — captures any monotonic miscalibration pattern.
   Requires more data to avoid overfitting.

Usage
-----
::

    from src.calibration import CalibratedModel

    # Wrap any sklearn-compatible model
    model = CalibratedModel(xgb_model, method="platt")
    model.fit(X_train, y_train, X_val, y_val)

    # Now model.predict_proba() returns calibrated probabilities
    probs = model.predict_proba(X_test)

    # Evaluate calibration
    metrics = model.evaluate_calibration(X_test, y_test)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


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
