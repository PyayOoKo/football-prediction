"""Calibration analysis and reporting utilities.

Provides:
  - calibration_curve() — compute reliability curve bin metrics
  - calibration_report() — human-readable calibration report
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


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
    ll = log_loss(y_true, y_prob)
    y_onehot = np.eye(3)[y_true.astype(np.int64)]
    brier = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

    curve = calibration_curve(y_true, y_prob)
    ece = float(np.mean(
        curve["counts"] / curve["counts"].sum() * np.abs(curve["accuracies"] - curve["confidences"])
    ))

    lines = [
        f"Calibration Report — {model_name}",
        f"{'=' * 50}",
        f"  Log-loss:      {ll:.4f}",
        f"  Brier score:   {brier:.4f}",
        f"  ECE:           {ece:.4f}  (Expected Calibration Error)",
        "",
        f"  {'Bin':<8} {'Count':<8} {'Accuracy':<10} {'Confidence':<12} {'Gap':<8}",
        f"  {'-' * 46}",
    ]
    for i in range(len(curve["bin_centers"])):
        if curve["counts"][i] > 0:
            gap = abs(curve["accuracies"][i] - curve["confidences"][i])
            lines.append(
                f"  {curve['bin_centers'][i]:<8.2f} {int(curve['counts'][i]):<8} "
                f"{curve['accuracies'][i]:<10.3f} {curve['confidences'][i]:<12.3f} {gap:<8.3f}"
            )

    return "\n".join(lines)
