"""
Feature Importance Analysis — multi-model importance, correlation, selection, validation.

Usage::

    python scripts/feature_importance_analysis.py

Outputs::

    reports/feature_importance_{timestamp}.json
    reports/feature_correlation_analysis_{timestamp}.csv
    reports/feature_selection_recommendations_{timestamp}.md
    reports/feature_importance_shap_{timestamp}.png
    reports/feature_importance_permutation_{timestamp}.png
    reports/feature_importance_convergence_{timestamp}.png
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import shap

from src.time_series_cv import time_series_train_val_test_split
from src.feature_engineering import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("importance")

for name in ("matplotlib", "PIL", "urllib3", "sklearn", "xgboost", "lightgbm"):
    logging.getLogger(name).setLevel(logging.WARNING)

DATA_PATH = ROOT / "data" / "processed" / "results_clean.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
SEED = 42
N_JOBS = -1

CLASS_LABELS = ["away", "draw", "home"]
TARGET_MAP = {0: "away", 1: "draw", 2: "home"}

# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════


def load_and_split():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    log.info("Loaded %d rows x %d cols", len(df), len(df.columns))
    X, y = build_features(df, is_training=True)
    log.info("Feature matrix: %s, target: %s", X.shape, y.shape)
    splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))
    col_means = splits["X_train"].mean().fillna(0)
    return {
        "X_train": splits["X_train"].fillna(col_means),
        "y_train": splits["y_train"],
        "X_val": splits["X_val"].fillna(col_means),
        "y_val": splits["y_val"],
        "X_test": splits["X_test"].fillna(col_means),
        "y_test": splits["y_test"],
        "feature_names": X.columns.tolist(),
    }


# ═══════════════════════════════════════════════════════════
#  1A: Train Multiple Models
# ═══════════════════════════════════════════════════════════


def train_models(data: dict) -> dict:
    log.info("Training 4 models ...")
    models = {}

    t0 = time.time()
    lr = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=SEED,
        class_weight="balanced", C=1.0, n_jobs=N_JOBS,
    )
    lr.fit(data["X_train"], data["y_train"])
    models["logistic_regression"] = lr
    log.info("  LR trained in %.1fs", time.time() - t0)

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        random_state=SEED, class_weight="balanced", n_jobs=N_JOBS,
    )
    rf.fit(data["X_train"], data["y_train"])
    models["random_forest"] = rf
    log.info("  RF trained in %.1fs", time.time() - t0)

    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        reg_alpha=0.1, random_state=SEED, eval_metric="mlogloss",
        tree_method="hist", n_jobs=N_JOBS,
    )
    xgb_model.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )
    models["xgboost"] = xgb_model
    log.info("  XGB trained in %.1fs", time.time() - t0)

    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        reg_alpha=0.1, random_state=SEED, class_weight="balanced",
        verbose=-1, n_jobs=N_JOBS,
    )
    lgb_model.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        eval_metric="multi_logloss",
    )
    models["lightgbm"] = lgb_model
    log.info("  LGBM trained in %.1fs", time.time() - t0)

    gc.collect()
    return models


# ═══════════════════════════════════════════════════════════
#  1B: Extract Importance Metrics
# ═══════════════════════════════════════════════════════════


def extract_importance(data: dict, models: dict) -> dict:
    log.info("Extracting importance metrics ...")
    feature_names = data["feature_names"]
    n_features = len(feature_names)
    result: dict[str, Any] = {}

    # ── Logistic Regression coefficients ──────────────────
    lr = models["logistic_regression"]
    coef = lr.coef_  # shape (3, n_features)
    avg_abs_coef = np.mean(np.abs(coef), axis=0)
    result["logistic_regression"] = {
        "type": "coefficient",
        "importance": {
            fname: float(avg_abs_coef[i])
            for i, fname in enumerate(feature_names)
        },
        "coef_home": {fname: float(coef[2][i]) for i, fname in enumerate(feature_names)},
        "coef_draw": {fname: float(coef[1][i]) for i, fname in enumerate(feature_names)},
        "coef_away": {fname: float(coef[0][i]) for i, fname in enumerate(feature_names)},
    }

    # ── Random Forest ─────────────────────────────────────
    rf = models["random_forest"]
    rf_imp = rf.feature_importances_
    result["random_forest"] = {
        "type": "feature_importance",
        "importance": {
            fname: float(rf_imp[i]) for i, fname in enumerate(feature_names)
        },
    }

    # ── XGBoost ────────────────────────────────────────────
    xgb_model = models["xgboost"]
    result["xgboost"] = {
        "type": "feature_importance",
        "importance": {
            fname: float(xgb_model.feature_importances_[i])
            for i, fname in enumerate(feature_names)
        },
        "importance_gain": {
            fname: float(v)
            for fname, v in zip(feature_names, xgb_model.feature_importances_)
        },
    }
    # XGBoost gain/weight/cover
    for imp_type in ("weight", "cover", "total_gain", "total_cover"):
        try:
            scores = xgb_model.get_booster().get_score(importance_type=imp_type)
            result["xgboost"][f"importance_{imp_type}"] = {
                feature_names[int(k[1:])]: float(v)
                for k, v in scores.items()
            }
        except Exception:
            pass

    # ── LightGBM ──────────────────────────────────────────
    lgb_model = models["lightgbm"]
    result["lightgbm"] = {
        "type": "feature_importance",
        "importance": {
            fname: float(lgb_model.feature_importances_[i])
            for i, fname in enumerate(feature_names)
        },
        "importance_gain": {
            fname: float(v)
            for fname, v in zip(feature_names, lgb_model.booster_.feature_importance(importance_type="gain"))
        },
        "importance_split": {
            fname: float(v)
            for fname, v in zip(feature_names, lgb_model.booster_.feature_importance(importance_type="split"))
        },
    }

    # ── Permutation Importance ────────────────────────────
    X_test_small = data["X_test"].iloc[: min(1000, len(data["X_test"]))]
    y_test_small = data["y_test"].iloc[: min(1000, len(data["y_test"]))]
    log.info("  Computing permutation importance (sample=%d) ...", len(X_test_small))
    for name, model in models.items():
        t0 = time.time()
        pi = permutation_importance(
            model, X_test_small, y_test_small,
            n_repeats=5, random_state=SEED, n_jobs=N_JOBS,
        )
        result[name]["permutation_importance"] = {
            fname: {
                "mean": float(pi.importances_mean[i]),
                "std": float(pi.importances_std[i]),
            }
            for i, fname in enumerate(feature_names)
        }
        log.info("    %s permutation importance done (%.1fs)", name, time.time() - t0)
    gc.collect()

    # ── SHAP Values (sampled for speed) ──────────────────
    X_shap = data["X_test"].iloc[: min(200, len(data["X_test"]))]
    log.info("  Computing SHAP (sample=%d) ...", len(X_shap))
    for name, model in models.items():
        t0 = time.time()
        try:
            if name == "logistic_regression":
                explainer = shap.LinearExplainer(model, data["X_train"])
            elif name == "random_forest":
                explainer = shap.TreeExplainer(model)
            elif name == "xgboost":
                explainer = shap.TreeExplainer(model)
            elif name == "lightgbm":
                explainer = shap.TreeExplainer(model)
            else:
                continue
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):
                shap_agg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                if shap_agg.ndim > 1:
                    shap_agg = np.mean(shap_agg, axis=0)
            else:
                shap_agg = np.mean(np.abs(shap_values), axis=0)
                if shap_agg.ndim > 1:
                    shap_agg = np.mean(shap_agg, axis=0)
            result[name]["shap"] = {
                fname: float(shap_agg[i]) if i < len(shap_agg) else 0.0
                for i, fname in enumerate(feature_names)
            }
            log.info("    %s SHAP done (%.1fs)", name, time.time() - t0)
        except Exception as e:
            log.warning("    %s SHAP skipped: %s", name, e)
            result[name]["shap"] = {}

    # SHAP summary plot
    _plot_shap_summary(models, X_shap, feature_names)

    return result


def _plot_shap_summary(models: dict, X_shap: pd.DataFrame, feature_names: list[str]):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx // 2][idx % 2]
        try:
            if name == "logistic_regression":
                explainer = shap.LinearExplainer(model, X_shap)
            else:
                explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            sv = shap_values
            if isinstance(sv, list):
                sv = np.array(sv)
            if sv.ndim == 3:
                sv = sv.reshape(-1, sv.shape[-1])
            elif sv.ndim != 2:
                sv = sv.reshape(-1, sv.shape[-1]) if sv.ndim > 2 else sv
            shap.summary_plot(
                sv, X_shap, feature_names=feature_names,
                show=False, max_display=15, ax=ax,
            )
            ax.set_title(name.replace("_", " ").title())
        except Exception:
            ax.text(0.5, 0.5, "SHAP not available", ha="center", va="center")
            ax.set_title(name.replace("_", " ").title())
    plt.tight_layout()
    path = REPORTS_DIR / f"feature_importance_shap_{TIMESTAMP}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  SHAP plot saved to %s", path)
    gc.collect()


# ═══════════════════════════════════════════════════════════
#  1C: Aggregate Importance
# ═══════════════════════════════════════════════════════════


def aggregate_importance(importance_results: dict) -> dict:
    log.info("Aggregating importance across models ...")
    feature_names = set()
    model_scores = defaultdict(dict)

    for model_name, metrics in importance_results.items():
        for metric_key in ("importance", "permutation_importance", "shap"):
            scores = metrics.get(metric_key, {})
            if not scores:
                continue
            # Extract scalar values (permutation_importance stores {"mean": ..., "std": ...})
            vals_list = []
            for v in scores.values():
                if isinstance(v, dict):
                    vals_list.append(v.get("mean", 0))
                else:
                    vals_list.append(v)
            vals = np.array(vals_list, dtype=float)
            if vals.max() == 0:
                continue
            norm = vals / vals.max()
            for i, fname in enumerate(scores.keys()):
                feature_names.add(fname)
                metric_name = f"{model_name}_{metric_key}"
                model_scores[fname][metric_name] = float(norm[i])

    # Add coefficient-based from LR
    lr_coef = importance_results["logistic_regression"]["importance"]
    vals = np.array(list(lr_coef.values()))
    if vals.max() > 0:
        norm = vals / vals.max()
        for i, fname in enumerate(lr_coef.keys()):
            model_scores[fname]["logistic_regression_coef"] = float(norm[i])

    # Build dataframe
    fnames = sorted(feature_names)
    df = pd.DataFrame(index=fnames)
    for fname in fnames:
        for metric_name, score in model_scores.get(fname, {}).items():
            df.loc[fname, metric_name] = score

    df = df.fillna(0)
    df["avg_importance"] = df.mean(axis=1)
    df["std_importance"] = df.std(axis=1)
    df["cv_importance"] = (df["std_importance"] / (df["avg_importance"] + 1e-10)).clip(0, 10)
    df = df.sort_values("avg_importance", ascending=False)

    top20 = df.head(20)
    near_zero = df[df["avg_importance"] < 0.01]
    high_variance = df[df["cv_importance"] > 1.5]

    aggregate = {
        "ranking": {
            fname: {
                "avg_importance": float(row["avg_importance"]),
                "std_importance": float(row["std_importance"]),
                "cv_importance": float(row["cv_importance"]),
                "rank": int(rank + 1),
            }
            for rank, (fname, row) in enumerate(df.iterrows())
        },
        "top_20": [
            {"feature": fname, "avg_importance": float(row["avg_importance"])}
            for fname, row in top20.iterrows()
        ],
        "near_zero": [
            {"feature": fname, "avg_importance": float(row["avg_importance"])}
            for fname, row in near_zero.iterrows()
        ],
        "high_variance": [
            {"feature": fname, "avg_importance": float(row["avg_importance"]),
             "cv": float(row["cv_importance"])}
            for fname, row in high_variance.iterrows()
        ],
    }

    log.info("  Top 20 features identified")
    log.info("  Near-zero features: %d", len(near_zero))
    log.info("  High-variance features: %d", len(high_variance))

    # Plot convergence
    _plot_convergence(df)

    return aggregate, df


def _plot_convergence(imp_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(range(1, len(imp_df) + 1), imp_df["avg_importance"].cumsum() / imp_df["avg_importance"].sum() * 100)
    ax.axhline(90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    ax.axvline(20, color="green", linestyle="--", alpha=0.5, label="Top 20")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Cumulative importance (%)")
    ax.set_title("Importance Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    top = imp_df.head(20)
    ax.barh(range(len(top)), top["avg_importance"].values[::-1], color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=9)
    ax.set_xlabel("Mean normalized importance")
    ax.set_title("Top 20 — Aggregated Importance")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = REPORTS_DIR / f"feature_importance_convergence_{TIMESTAMP}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Convergence plot saved to %s", path)


# ═══════════════════════════════════════════════════════════
#  2: Feature Correlation Analysis
# ═══════════════════════════════════════════════════════════


def correlation_analysis(data: dict) -> tuple[pd.DataFrame, list[dict]]:
    log.info("Computing feature correlations ...")
    X = data["X_train"]
    numeric = X.select_dtypes(include=[np.number])
    corr = numeric.corr()

    high_pairs = []
    seen = set()
    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):
            if i >= j:
                continue
            r = abs(corr.iloc[i, j])
            if r > 0.8:
                key = tuple(sorted([col1, col2]))
                if key not in seen:
                    seen.add(key)
                    high_pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": round(corr.iloc[i, j], 4),
                    })

    corr_path = REPORTS_DIR / f"feature_correlation_analysis_{TIMESTAMP}.csv"
    corr_df = corr.reset_index()
    corr_df.columns = ["feature"] + list(corr.columns)
    corr_df.to_csv(corr_path, index=False)
    log.info("  Correlation matrix saved to %s", corr_path)
    log.info("  Highly correlated pairs (|r|>0.8): %d", len(high_pairs))

    return corr, high_pairs


# ═══════════════════════════════════════════════════════════
#  3: Feature Selection Methods
# ═══════════════════════════════════════════════════════════


def _evaluate(data: dict, X_train, y_train, X_val, y_val, model=None) -> dict:
    """Train and evaluate a model on given data, return metrics."""
    if model is None:
        model = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=SEED,
            class_weight="balanced", C=1.0, n_jobs=N_JOBS,
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    y_onehot = np.eye(3)[y_val.values if hasattr(y_val, "values") else y_val]
    brier = float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))
    return {
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "log_loss": round(log_loss(y_val, y_proba), 4),
        "brier_score": round(brier, 4),
        "n_features": X_train.shape[1],
    }


def run_feature_selection(data: dict, imp_df: pd.DataFrame) -> dict:
    log.info("Running feature selection methods ...")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]

    # Full set baseline
    baseline = _evaluate(data, X_train, y_train, X_val, y_val)
    baseline["feature_set"] = "full"
    baseline["n_features"] = X_train.shape[1]
    log.info("  Baseline (all %d features): acc=%.4f, brier=%.4f",
             baseline["n_features"], baseline["accuracy"], baseline["brier_score"])

    selections = [baseline]

    # ── Method 1: Univariate (SelectKBest) ────────────────
    for k in [10, 20, 30, 50]:
        try:
            selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
            X_train_k = selector.fit_transform(X_train, y_train)
            X_val_k = selector.transform(X_val)
            mask = selector.get_support()
            sel_features = [f for f, s in zip(feature_names, mask) if s]
            metrics = _evaluate(data, X_train_k, y_train, X_val_k, y_val)
            metrics["feature_set"] = f"univariate_k{k}"
            metrics["selected_features"] = sel_features
            selections.append(metrics)
            log.info("  Univariate k=%d: acc=%.4f, brier=%.4f (%d features)",
                     k, metrics["accuracy"], metrics["brier_score"], len(sel_features))
        except Exception as e:
            log.warning("  Univariate k=%d failed: %s", k, e)

    # ── Method 2: RFE ─────────────────────────────────────
    for n in [10, 20, 30]:
        try:
            estimator = LogisticRegression(
                solver="lbfgs", max_iter=1000, random_state=SEED,
                class_weight="balanced", C=1.0, n_jobs=1,
            )
            rfe = RFE(estimator, n_features_to_select=n, step=0.1)
            X_train_rfe = rfe.fit_transform(X_train, y_train)
            X_val_rfe = rfe.transform(X_val)
            sel_features = [f for f, s in zip(feature_names, rfe.support_) if s]
            metrics = _evaluate(data, X_train_rfe, y_train, X_val_rfe, y_val)
            metrics["feature_set"] = f"rfe_n{n}"
            metrics["selected_features"] = sel_features
            selections.append(metrics)
            log.info("  RFE n=%d: acc=%.4f, brier=%.4f (%d features)",
                     n, metrics["accuracy"], metrics["brier_score"], len(sel_features))
        except Exception as e:
            log.warning("  RFE n=%d failed: %s", n, e)

    # ── Method 3: L1 Regularization ───────────────────────
    for C_val in [0.01, 0.05, 0.1, 0.5]:
        try:
            l1 = LogisticRegression(
                solver="saga", penalty="l1", C=C_val, max_iter=1000,
                random_state=SEED, class_weight="balanced", n_jobs=N_JOBS,
            )
            l1.fit(X_train, y_train)
            retained = np.abs(l1.coef_).max(axis=0) > 1e-6
            sel_features = [f for f, s in zip(feature_names, retained) if s]
            if len(sel_features) == 0:
                continue
            X_train_l1 = X_train.loc[:, retained]
            X_val_l1 = X_val.loc[:, retained]
            metrics = _evaluate(data, X_train_l1, y_train, X_val_l1, y_val)
            metrics["feature_set"] = f"l1_C{C_val}"
            metrics["selected_features"] = sel_features
            selections.append(metrics)
            log.info("  L1 C=%.2f: acc=%.4f, brier=%.4f (%d features, C=%.2f)",
                     C_val, metrics["accuracy"], metrics["brier_score"],
                     len(sel_features), C_val)
        except Exception as e:
            log.warning("  L1 C=%.2f failed: %s", C_val, e)

    # ── Method 4: Importance Threshold ────────────────────
    for threshold in [0.01, 0.02, 0.05, 0.10]:
        try:
            keep = imp_df[imp_df["avg_importance"] >= threshold].index.tolist()
            keep_in_data = [c for c in keep if c in X_train.columns]
            if len(keep_in_data) == 0 or len(keep_in_data) == X_train.shape[1]:
                continue
            X_train_th = X_train[keep_in_data]
            X_val_th = X_val[keep_in_data]
            metrics = _evaluate(data, X_train_th, y_train, X_val_th, y_val)
            metrics["feature_set"] = f"threshold_{threshold}"
            metrics["selected_features"] = keep_in_data
            selections.append(metrics)
            log.info("  Threshold %.2f: acc=%.4f, brier=%.4f (%d features)",
                     threshold, metrics["accuracy"], metrics["brier_score"],
                     len(keep_in_data))
        except Exception as e:
            log.warning("  Threshold %.2f failed: %s", threshold, e)

    # Find best
    best = min(selections, key=lambda x: x["brier_score"])
    best_min_features = min(
        (s for s in selections if s["brier_score"] <= baseline["brier_score"] * 1.01),
        key=lambda x: x["n_features"],
    )

    log.info("  Best by Brier: %s (brier=%.4f, %d features)",
             best["feature_set"], best["brier_score"], best["n_features"])
    log.info("  Best minimal (within 1%% of baseline): %s (%d features, brier=%.4f)",
             best_min_features["feature_set"], best_min_features["n_features"],
             best_min_features["brier_score"])

    return {
        "baseline": baseline,
        "all_selections": selections,
        "best_by_brier": best,
        "best_minimal": best_min_features,
    }


# ═══════════════════════════════════════════════════════════
#  5: Generate Recommendations
# ═══════════════════════════════════════════════════════════


def generate_recommendations(
    data: dict,
    importance: dict,
    aggregate: dict,
    corr_high_pairs: list[dict],
    selection_results: dict,
    imp_df: pd.DataFrame,
):
    log.info("Generating recommendations ...")

    best = selection_results["best_minimal"]
    baseline = selection_results["baseline"]

    # Determine features to keep (from best subset)
    keep_features = best.get("selected_features", [])
    # Features to investigate: high variance + important
    high_var = {f["feature"] for f in aggregate["high_variance"]}
    near_zero = {f["feature"] for f in aggregate["near_zero"]}
    redundant_from_corr = {p["feature_2"] for p in corr_high_pairs}

    gps = "\n".join(
        f"- `{p['feature_1']}` ↔ `{p['feature_2']}` (r={p['correlation']})"
        for p in corr_high_pairs[:20]
    )

    top20_str = "\n".join(
        f"  {i+1}. `{f['feature']}` — importance={f['avg_importance']:.4f}"
        for i, f in enumerate(aggregate["top_20"])
    )

    keep_str = "\n".join(f"- `{f}`" for f in keep_features[:30])
    if len(keep_features) > 30:
        keep_str += f"\n  ... and {len(keep_features) - 30} more"

    remove_str = "\n".join(f"- `{f}`" for f in list(near_zero)[:30])
    if len(near_zero) > 30:
        remove_str += f"\n  ... and {len(near_zero) - 30} more"

    investigate_str = "\n".join(f"- `{h['feature']}` (cv={h['cv']:.2f})" for h in aggregate["high_variance"][:20])

    redundant_str = "\n".join(
        f"- `{p['feature_1']}` ↔ `{p['feature_2']}` (r={p['correlation']:.2f})"
        for p in corr_high_pairs[:20]
    )
    if len(corr_high_pairs) > 20:
        redundant_str += f"\n  ... and {len(corr_high_pairs) - 20} more pairs"

    md = f"""# Feature Selection Recommendations

> **Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
> **Dataset:** {data['X_train'].shape[0] + data['X_val'].shape[0] + data['X_test'].shape[0]} samples
> **Total features:** {data['X_train'].shape[1]}

---

## Executive Summary

| Metric | Full Set | Optimal Reduced Set | Change |
|--------|----------|-------------------|--------|
| **N features** | {baseline['n_features']} | {best['n_features']} | **-{baseline['n_features'] - best['n_features']} ({-((baseline['n_features'] - best['n_features']) / baseline['n_features'] * 100):.0f}%)** |
| **Accuracy** | {baseline['accuracy']:.4f} | {best['accuracy']:.4f} | {best['accuracy'] - baseline['accuracy']:+.4f} |
| **Brier Score** | {baseline['brier_score']:.4f} | {best['brier_score']:.4f} | {baseline['brier_score'] - best['brier_score']:+.4f} |
| **Log Loss** | {baseline['log_loss']:.4f} | {best['log_loss']:.4f} | {baseline['log_loss'] - best['log_loss']:+.4f} |

**Recommendation:** Use **`{best['feature_set']}`** ({best['n_features']} features).

---

## 1. Top 20 Most Important Features

{top20_str}

---

## 2. Features to Remove (Near-Zero Importance)

{remove_str if remove_str else "None found."}

---

## 3. Features to Investigate (High Variance)

These features show inconsistent importance across models:

{investigate_str if investigate_str else "None found."}

---

## 4. Redundant Feature Pairs (|r| > 0.8)

Consider removing one feature from each pair:

{redundant_str if redundant_str else "None found."}

---

## 5. Selection Methods Comparison

| Method | Accuracy | Brier Score | Log Loss | N Features |
|--------|----------|-------------|----------|-----------|
"""

    for sel in sorted(selection_results["all_selections"], key=lambda x: x["brier_score"]):
        md += f"| {sel['feature_set']} | {sel['accuracy']:.4f} | {sel['brier_score']:.4f} | {sel['log_loss']:.4f} | {sel['n_features']} |\n"

    md += f"""
---

## 6. How to Use the Reduced Feature Set

```python
from src.feature_engineering import build_features
import pandas as pd

df = pd.read_csv("data/processed/results_clean.csv")
X, y = build_features(df, is_training=True)

# Select the optimal subset
optimal_features = {json.dumps(keep_features[:50])}
X_reduced = X[[c for c in optimal_features if c in X.columns]]
```

---

*Generated automatically by `scripts/feature_importance_analysis.py`*
"""

    path = REPORTS_DIR / f"feature_selection_recommendations_{TIMESTAMP}.md"
    with open(path, "w") as f:
        f.write(md)
    log.info("  Recommendations saved to %s", path)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    print("=" * 72)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("=" * 72)

    t_total = time.time()

    # Step 1: Load data
    print("\n[1/6] Loading data and building features ...")
    t0 = time.time()
    data = load_and_split()
    print(f"      {data['X_train'].shape[0]} train, {data['X_val'].shape[0]} val, "
          f"{data['X_test'].shape[0]} test, {data['X_train'].shape[1]} features  "
          f"({time.time() - t0:.1f}s)")

    # Step 2: Train models
    print("\n[2/6] Training multiple models ...")
    t0 = time.time()
    models = train_models(data)
    print(f"      Done ({time.time() - t0:.1f}s)")

    # Step 3: Extract importance
    print("\n[3/6] Extracting importance metrics ...")
    t0 = time.time()
    importance_results = extract_importance(data, models)
    print(f"      Done ({time.time() - t0:.1f}s)")

    # Step 4: Aggregate importance
    print("\n[4/6] Aggregating importance across models ...")
    t0 = time.time()
    aggregate, imp_df = aggregate_importance(importance_results)
    print(f"      Done ({time.time() - t0:.1f}s)")

    # Step 5: Correlation analysis
    print("\n[5/6] Correlation analysis ...")
    t0 = time.time()
    corr_matrix, corr_high_pairs = correlation_analysis(data)
    print(f"      {len(corr_high_pairs)} highly correlated pairs found "
          f"({time.time() - t0:.1f}s)")

    # Step 6: Feature selection + validation
    print("\n[6/6] Feature selection and validation ...")
    t0 = time.time()
    selection_results = run_feature_selection(data, imp_df)
    print(f"      Done ({time.time() - t0:.1f}s)")

    # Save main report JSON
    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "train_size": data["X_train"].shape[0],
            "val_size": data["X_val"].shape[0],
            "test_size": data["X_test"].shape[0],
            "n_features": data["X_train"].shape[1],
        },
        "model_importance": importance_results,
        "aggregated_importance": aggregate,
        "correlation_analysis": {
            "high_correlation_pairs": corr_high_pairs[:50],
            "n_high_pairs": len(corr_high_pairs),
            "correlation_matrix_file": f"feature_correlation_analysis_{TIMESTAMP}.csv",
        },
        "feature_selection": {
            "baseline": selection_results["baseline"],
            "best_by_brier": selection_results["best_by_brier"],
            "best_minimal": selection_results["best_minimal"],
            "all_selections": selection_results["all_selections"],
        },
    }

    report_path = REPORTS_DIR / f"feature_importance_{TIMESTAMP}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    # Generate recommendations
    print("\n  Generating recommendations report ...")
    generate_recommendations(
        data, importance_results, aggregate,
        corr_high_pairs, selection_results, imp_df,
    )

    total = time.time() - t_total
    print(f"\n  Total duration: {total:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
