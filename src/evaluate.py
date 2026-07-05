"""
Evaluate — compute performance metrics and generate visualisation reports.

Typical usage::

    from src.evaluate import evaluate_model
    report = evaluate_model(model, X_test, y_test)
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import config

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute all configured metrics and optionally generate plots.

    Parameters
    ----------
    model : Any
        Trained model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True labels.

    Returns
    -------
    dict[str, Any]
        Dictionary of metric names to values, plus file paths to any saved plots.
    """
    logger.info("Evaluating model on %d test samples", len(X_test))
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    report: dict[str, Any] = {}

    # ── Metrics ──────────────────────────────────
    metrics = config.eval.metrics
    if "accuracy" in metrics:
        report["accuracy"] = accuracy_score(y_test, y_pred)
    if "precision" in metrics:
        report["precision"] = precision_score(y_test, y_pred, average="weighted")
    if "recall" in metrics:
        report["recall"] = recall_score(y_test, y_pred, average="weighted")
    if "f1" in metrics:
        report["f1"] = f1_score(y_test, y_pred, average="weighted")
    if "log_loss" in metrics and y_proba is not None:
        report["log_loss"] = log_loss(y_test, y_proba)
    if "roc_auc" in metrics and y_proba is not None:
        report["roc_auc"] = _compute_roc_auc(y_test, y_proba)

    # ── Classification report (text) ────────────
    report["classification_report"] = classification_report(y_test, y_pred)

    # ── Plots ────────────────────────────────────
    plot_paths: dict[str, str] = {}
    sns.set_theme(style="whitegrid")

    if config.eval.plot_confusion_matrix:
        path = _save_confusion_matrix(y_test, y_pred)
        plot_paths["confusion_matrix"] = path

    if config.eval.plot_roc_curve and y_proba is not None:
        path = _save_roc_curve(y_test, y_proba)
        plot_paths["roc_curve"] = path

    if config.eval.plot_feature_importance:
        path = _save_feature_importance(model, X_test.columns)
        if path:
            plot_paths["feature_importance"] = path

    report["plots"] = plot_paths

    logger.info("Evaluation complete.")
    return report


# ── Internal helpers ────────────────────────────────────


def _compute_roc_auc(y_test: pd.Series, y_proba: np.ndarray) -> float:
    """Compute ROC-AUC, handling multi-class via macro-average."""
    n_classes = y_proba.shape[1]
    if n_classes == 2:
        return roc_auc_score(y_test, y_proba[:, 1])
    return roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")


def _save_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> str:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test) | set(y_pred))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    path = str(config.eval.output_dir / "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)
    return path


def _save_roc_curve(y_test: pd.Series, y_proba: np.ndarray) -> str:
    """Plot and save ROC curves (one-vs-rest for multi-class)."""
    n_classes = y_proba.shape[1]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        auc = roc_auc_score(y_test == i, y_proba[:, i])
        ax.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    path = str(config.eval.output_dir / "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s", path)
    return path


def _save_feature_importance(
    model: Any,
    feature_names: pd.Index,
) -> str | None:
    """Plot and save feature importance if the model exposes one."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        logger.info("Model does not expose feature importance; skipping plot.")
        return None

    indices = np.argsort(importances)[::-1][:20]  # top 20
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(indices)), importances[indices][::-1])
    ax.set(yticks=range(len(indices)),
           yticklabels=feature_names[indices][::-1],
           xlabel="Importance", title="Top 20 Feature Importances")
    plt.tight_layout()
    path = str(config.eval.output_dir / "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance saved to %s", path)
    return path
