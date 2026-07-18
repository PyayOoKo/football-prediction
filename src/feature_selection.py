"""
Feature Selection — reusable methods for reducing the feature set.

Provides:
    - Recursive Feature Elimination (RFE)            via :func:`select_rfe`
    - Feature Importance Threshold                   via :func:`select_by_threshold`
    - Mutual Information                             via :func:`select_mutual_info`
    - Sequential Feature Selection (SFS)             via :func:`select_sfs`
    - High-correlation pair detection                via :func:`find_redundant_pairs`
    - Partial dependence plots                       via :func:`plot_partial_dependence`
    - Full comparison runner                         via :func:`run_all_selections`

Usage::

    from src.feature_selection import run_all_selections

    results = run_all_selections(X_train, y_train, X_val, y_val, feature_names)
    best = results["best_minimal"]
    print(f"Optimal: {best['n_features']} features, brier={best['brier_score']:.4f}")
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    matplotlib.use("Agg")
except Exception as exc:
    logger.debug("matplotlib Agg backend unavailable: %s — using default", exc)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SequentialFeatureSelector,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

logger = logging.getLogger(__name__)
SEED = 42
N_JOBS = -1


# ═══════════════════════════════════════════════════════════
#  Evaluation helper
# ═══════════════════════════════════════════════════════════


def _evaluate(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    model=None,
) -> dict[str, Any]:
    """Train a quick model and return validation metrics.

    Parameters
    ----------
    model : estimator or None
        If None, uses ``LogisticRegression`` (fast for feature selection).
    """
    if model is None:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
            class_weight="balanced",
            C=1.0,
            n_jobs=1,
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    # Brier score (multi-class)
    y_onehot = np.eye(3)[np.asarray(y_val)]
    brier = float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))
    n_feats = X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
    return {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "log_loss": round(float(log_loss(y_val, y_proba)), 4),
        "brier_score": round(brier, 4),
        "n_features": n_feats,
    }


# ═══════════════════════════════════════════════════════════
#  1. Recursive Feature Elimination
# ═══════════════════════════════════════════════════════════


def select_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_values: tuple[int, ...] = (10, 20, 30, 50),
    estimator=None,
) -> list[dict[str, Any]]:
    """Recursive Feature Elimination at multiple target sizes.

    Parameters
    ----------
    estimator : estimator or None
        If None, uses ``LogisticRegression(solver='lbfgs', C=1.0)``.
    """
    results = []
    if estimator is None:
        estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
            class_weight="balanced",
            C=1.0,
            n_jobs=1,
        )
    for n in n_values:
        n = min(n, X_train.shape[1])
        try:
            rfe = RFE(estimator, n_features_to_select=n, step=0.1)
            X_tr = rfe.fit_transform(X_train, y_train)
            X_v = rfe.transform(X_val)
            sel = [f for f, s in zip(feature_names, rfe.support_) if s]
            metrics = _evaluate(X_tr, y_train, X_v, y_val)
            metrics["feature_set"] = f"rfe_n{n}"
            metrics["selected_features"] = sel
            results.append(metrics)
            logger.info(
                "  RFE n=%d: acc=%.4f, brier=%.4f (%d features)",
                n, metrics["accuracy"], metrics["brier_score"], len(sel),
            )
        except Exception as e:
            logger.warning("  RFE n=%d failed: %s", n, e)
    return results


# ═══════════════════════════════════════════════════════════
#  2. Feature Importance Threshold
# ═══════════════════════════════════════════════════════════


def select_by_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    importance_df: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
) -> list[dict[str, Any]]:
    """Select features whose aggregated importance exceeds each threshold.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must contain an ``avg_importance`` column, indexed by feature name.
        Produced by :func:`~scripts.feature_importance_analysis.aggregate_importance`.
    """
    results = []
    for thresh in thresholds:
        try:
            keep = importance_df[importance_df["avg_importance"] >= thresh].index.tolist()
            keep_in = [c for c in keep if c in X_train.columns]
            if not keep_in or len(keep_in) == X_train.shape[1]:
                continue
            metrics = _evaluate(
                X_train[keep_in], y_train, X_val[keep_in], y_val,
            )
            metrics["feature_set"] = f"threshold_{thresh}"
            metrics["selected_features"] = keep_in
            results.append(metrics)
            logger.info(
                "  Threshold %.2f: acc=%.4f, brier=%.4f (%d features)",
                thresh, metrics["accuracy"], metrics["brier_score"], len(keep_in),
            )
        except Exception as e:
            logger.warning("  Threshold %.2f failed: %s", thresh, e)
    return results


# ═══════════════════════════════════════════════════════════
#  3. Mutual Information (Univariate)
# ═══════════════════════════════════════════════════════════


def select_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    k_values: tuple[int, ...] = (10, 20, 30, 50),
) -> list[dict[str, Any]]:
    """Select top-k features via mutual information.
    
    Uses ``SelectKBest(mutual_info_classif)``.
    """
    results = []
    for k in k_values:
        k = min(k, X_train.shape[1])
        try:
            selector = SelectKBest(mutual_info_classif, k=k)
            X_tr = selector.fit_transform(X_train, y_train)
            X_v = selector.transform(X_val)
            mask = selector.get_support()
            sel = [f for f, s in zip(feature_names, mask) if s]
            metrics = _evaluate(X_tr, y_train, X_v, y_val)
            metrics["feature_set"] = f"mutual_info_k{k}"
            metrics["selected_features"] = sel
            results.append(metrics)
            logger.info(
                "  MutualInfo k=%d: acc=%.4f, brier=%.4f (%d features)",
                k, metrics["accuracy"], metrics["brier_score"], len(sel),
            )
        except Exception as e:
            logger.warning("  MutualInfo k=%d failed: %s", k, e)
    return results


# ═══════════════════════════════════════════════════════════
#  4. Sequential Feature Selection
# ═══════════════════════════════════════════════════════════


def select_sfs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_values: tuple[int, ...] = (10, 20, 30),
    direction: str = "forward",
    estimator=None,
) -> list[dict[str, Any]]:
    """Sequential Feature Selection (forward or backward).

    Parameters
    ----------
    direction : ``"forward"`` or ``"backward"``.
    estimator : estimator or None
        If None, uses ``LogisticRegression``.
    """
    results = []
    if estimator is None:
        estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
            class_weight="balanced",
            C=1.0,
            n_jobs=1,
        )
    for n in n_values:
        n = min(n, X_train.shape[1])
        try:
            sfs = SequentialFeatureSelector(
                estimator,
                n_features_to_select=n,
                direction=direction,
                scoring="neg_log_loss",
                cv=3,
                n_jobs=1,
            )
            sfs.fit(X_train, y_train)
            sel = [f for f, s in zip(feature_names, sfs.get_support()) if s]
            X_tr = X_train[sel]
            X_v = X_val[sel]
            metrics = _evaluate(X_tr, y_train, X_v, y_val)
            metrics["feature_set"] = f"sfs_{direction}_n{n}"
            metrics["selected_features"] = sel
            results.append(metrics)
            logger.info(
                "  SFS %s n=%d: acc=%.4f, brier=%.4f (%d features)",
                direction, n, metrics["accuracy"], metrics["brier_score"], len(sel),
            )
        except Exception as e:
            logger.warning("  SFS %s n=%d failed: %s", direction, n, e)
    return results


# ═══════════════════════════════════════════════════════════
#  5. High-Correlation Pair Detection
# ═══════════════════════════════════════════════════════════


def find_redundant_pairs(
    X: pd.DataFrame,
    threshold: float = 0.80,
) -> list[dict[str, Any]]:
    """Find feature pairs with absolute Pearson correlation > *threshold*.

    Returns a list of dicts with keys ``feature_1``, ``feature_2``, ``correlation``.
    """
    numeric = X.select_dtypes(include=[np.number])
    corr = numeric.corr()
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):
            if i >= j:
                continue
            r = abs(corr.iloc[i, j])
            if r > threshold:
                key = tuple(sorted([col1, col2]))
                if key not in seen:
                    seen.add(key)
                    pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(float(corr.iloc[i, j]), 4),
                    })
    logger.info(
        "  Found %d feature pairs with |r| > %.2f", len(pairs), threshold,
    )
    return pairs


def drop_redundant(
    pairs: list[dict[str, Any]],
    importance_ranking: list[str],
) -> list[str]:
    """Given redundant pairs and an importance ranking, drop the less important feature.

    Parameters
    ----------
    pairs : list[dict]
        Output from :func:`find_redundant_pairs`.
    importance_ranking : list[str]
        Feature names in descending order of importance (most important first).

    Returns
    -------
    list[str]
        Features to *drop*.
    """
    rank = {f: i for i, f in enumerate(importance_ranking)}
    to_drop: set[str] = set()
    for p in pairs:
        f1, f2 = p["feature_1"], p["feature_2"]
        r1 = rank.get(f1, len(rank))
        r2 = rank.get(f2, len(rank))
        if r1 <= r2:
            to_drop.add(f2)
        else:
            to_drop.add(f1)
    return list(to_drop)


# ═══════════════════════════════════════════════════════════
#  6. Partial Dependence Plots
# ═══════════════════════════════════════════════════════════


def plot_partial_dependence(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
    n_top: int = 10,
    output_dir: str | Path = "reports",
    timestamp: str = "",
) -> str | None:
    """Plot partial dependence for the top *n_top* most important features.

    Parameters
    ----------
    model : trained sklearn-compatible estimator
    X : pd.DataFrame
        Feature set to compute partial dependence over (typically X_val or X_test).
    feature_names : list[str]
        Ordered list of feature names (most important first).
    n_top : int
        Number of top features to plot (default 10).
    output_dir : str | Path
        Directory to save the plot to.

    Returns
    -------
    str | None
        Path to saved figure, or None if plotting failed.
    """
    try:
        top = feature_names[:n_top]
        existing = [c for c in top if c in X.columns]
        if not existing:
            logger.warning("  No top features found in X for partial dependence")
            return None
        # Use a subset for speed
        X_sample = X[existing].iloc[: min(500, len(X))]

        fig, ax = plt.subplots(figsize=(14, 10))
        PartialDependenceDisplay.from_estimator(
            model,
            X_sample,
            features=existing,
            grid_resolution=20,
            random_state=SEED,
            ax=ax,
            kind="average",
        )
        fig.suptitle("Partial Dependence — Top Features", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = Path(output_dir) / f"partial_dependence_{timestamp}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Partial dependence plot saved to %s", path)
        return str(path)
    except Exception as e:
        logger.warning("  Partial dependence plot failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════
#  7. L1 Regularisation (bonus selection method)
# ═══════════════════════════════════════════════════════════


def select_l1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    C_values: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5),
) -> list[dict[str, Any]]:
    """Select features via L1-regularised logistic regression.

    Features whose absolute coefficient is below 1e-6 are considered eliminated.
    """
    results = []
    for C_val in C_values:
        try:
            l1 = LogisticRegression(
                solver="saga",
                penalty="l1",
                C=C_val,
                max_iter=1000,
                random_state=SEED,
                class_weight="balanced",
                n_jobs=N_JOBS,
            )
            l1.fit(X_train, y_train)
            retained = np.abs(l1.coef_).max(axis=0) > 1e-6
            sel = [f for f, s in zip(feature_names, retained) if s]
            if not sel:
                continue
            X_tr = X_train.loc[:, retained]
            X_v = X_val.loc[:, retained]
            metrics = _evaluate(X_tr, y_train, X_v, y_val)
            metrics["feature_set"] = f"l1_C{C_val}"
            metrics["selected_features"] = sel
            results.append(metrics)
            logger.info(
                "  L1 C=%.2f: acc=%.4f, brier=%.4f (%d features)",
                C_val, metrics["accuracy"], metrics["brier_score"], len(sel),
            )
        except Exception as e:
            logger.warning("  L1 C=%.2f failed: %s", C_val, e)
    return results


# ═══════════════════════════════════════════════════════════
#  8. Run All & Find Optimal
# ═══════════════════════════════════════════════════════════


def run_all_selections(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    importance_df: pd.DataFrame | None = None,
    *,
    run_rfe: bool = True,
    run_threshold: bool = True,
    run_mutual_info: bool = True,
    run_sfs: bool = True,
    run_l1: bool = True,
    rfe_n_values: tuple[int, ...] = (10, 20, 30, 50),
    mi_k_values: tuple[int, ...] = (10, 20, 30, 50),
    sfs_n_values: tuple[int, ...] = (10, 20, 30),
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
    l1_C_values: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5),
    max_drop_pct: float = 1.0,
) -> dict[str, Any]:
    """Run multiple feature selection methods and find the optimal minimal subset.

    Returns
    -------
    dict with keys:
        - ``baseline`` — full-feature metrics
        - ``all_selections`` — list of all method results
        - ``best_by_brier`` — selection with lowest Brier score
        - ``best_minimal`` — smallest feature set within *max_drop_pct*% of baseline
    """
    # Baseline
    baseline = _evaluate(X_train, y_train, X_val, y_val)
    baseline["feature_set"] = "full"
    baseline["n_features"] = X_train.shape[1]
    logger.info(
        "  Baseline (%d features): acc=%.4f, brier=%.4f",
        baseline["n_features"], baseline["accuracy"], baseline["brier_score"],
    )

    all_selections: list[dict[str, Any]] = [baseline]

    if run_rfe:
        logger.info("  Running RFE ...")
        all_selections.extend(
            select_rfe(X_train, y_train, X_val, y_val, feature_names, rfe_n_values)
        )

    if run_threshold and importance_df is not None:
        logger.info("  Running importance threshold ...")
        all_selections.extend(
            select_by_threshold(X_train, y_train, X_val, y_val, importance_df, thresholds)
        )

    if run_mutual_info:
        logger.info("  Running mutual information ...")
        all_selections.extend(
            select_mutual_info(X_train, y_train, X_val, y_val, feature_names, mi_k_values)
        )

    if run_sfs:
        logger.info("  Running SFS forward ...")
        all_selections.extend(
            select_sfs(X_train, y_train, X_val, y_val, feature_names, sfs_n_values, "forward")
        )
        logger.info("  Running SFS backward ...")
        all_selections.extend(
            select_sfs(X_train, y_train, X_val, y_val, feature_names, sfs_n_values, "backward")
        )

    if run_l1:
        logger.info("  Running L1 regularisation ...")
        all_selections.extend(
            select_l1(X_train, y_train, X_val, y_val, feature_names, l1_C_values)
        )

    # Find best
    best_by_brier = min(all_selections, key=lambda x: x["brier_score"])
    threshold_brier = baseline["brier_score"] * (1.0 + max_drop_pct / 100.0)
    candidates = [s for s in all_selections if s["brier_score"] <= threshold_brier]
    if candidates:
        best_minimal = min(candidates, key=lambda x: x["n_features"])
    else:
        best_minimal = baseline

    logger.info(
        "  Best by Brier: %s (brier=%.4f, %d features)",
        best_by_brier["feature_set"], best_by_brier["brier_score"],
        best_by_brier["n_features"],
    )
    logger.info(
        "  Best minimal (within %.1f%% of baseline): %s (%d features, brier=%.4f)",
        max_drop_pct, best_minimal["feature_set"], best_minimal["n_features"],
        best_minimal["brier_score"],
    )

    return {
        "baseline": baseline,
        "all_selections": all_selections,
        "best_by_brier": best_by_brier,
        "best_minimal": best_minimal,
    }


# ═══════════════════════════════════════════════════════════
#  9. Save optimal feature set to JSON
# ═══════════════════════════════════════════════════════════


def save_optimal_features(
    best_minimal: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the optimal feature set to a JSON file.

    The file contains:
        - ``selected_features`` — list of feature names to keep
        - ``n_features`` — count
        - ``metadata`` — method, metrics, timestamp
    """
    data = {
        "selected_features": best_minimal.get("selected_features", []),
        "n_features": best_minimal["n_features"],
        "method": best_minimal["feature_set"],
        "accuracy": best_minimal["accuracy"],
        "brier_score": best_minimal["brier_score"],
        "log_loss": best_minimal["log_loss"],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(data, indent=2))
    logger.info("  Optimal features saved to %s", output_path)
