"""
train_over_under.py — Train specialist Over/Under 2.5 prediction models.

Trains 5 model types on preprocessed data, compares performance,
generates SHAP feature importance, and saves the best models.

Usage:
    python scripts/train_over_under.py
    python scripts/train_over_under.py --target over_2_5
    python scripts/train_over_under.py --lightweight    # Skip NN, fewer estimators
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Filter noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_over_under")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Primary target
TARGET = "over_2_5"  # Predict Over 2.5 goals

# XGBoost starting params (from user spec)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

LGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "min_samples_split": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 0,
}

LR_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}

NN_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.001,
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.2,
    "verbose": False,
}


# ═══════════════════════════════════════════════════════════
#  1. Load and prepare data
# ═══════════════════════════════════════════════════════════


def find_latest_parquet() -> Path:
    """Find the most recent processed parquet file."""
    files = sorted(PROCESSED_DIR.glob("over_under_data_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No over_under_data_*.parquet found in {PROCESSED_DIR}. "
            "Run scripts/preprocess_over_under.py first."
        )
    return files[-1]


def load_data(
    path: Path, target_col: str = TARGET,
    exclude_leaky: bool = False,
    only_rolling: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Load parquet, separate splits, extract features and target.

    Parameters
    ----------
    exclude_leaky : bool
        If True, exclude post-match features (xG, shots, corners) that would
        leak match outcome information into the model.
    only_rolling : bool
        If True, only include rolling team features (h_*/a_* rolling + cumavg + diff).
        Excludes odds, H2H, league features — tests if pure team stats can beat market.
    """
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)

    # Validate target exists
    assert target_col in df.columns, f"Target column '{target_col}' not found!"

    # Identify split: train/val/test by the parquet composition
    # The data is already concatenated — we need the split metadata.
    # We stored all rows together, so infer from date ranges.
    df["date"] = pd.to_datetime(df["date"])

    train_mask = (df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2022)
    test_mask = (df["date"].dt.year >= 2023) & (df["date"].dt.year <= 2024)

    df_train_val = df[train_mask].copy().sort_values("date")
    df_test = df[test_mask].copy()

    # Validation: last 20% of train_val by date
    split_idx = int(len(df_train_val) * 0.8)
    df_train = df_train_val.iloc[:split_idx].copy()
    df_val = df_train_val.iloc[split_idx:].copy()

    logger.info(
        "Split: train=%d (2016-2022), val=%d (last 20%%), test=%d (2023-2024)",
        len(df_train), len(df_val), len(df_test),
    )

    # Identify feature columns
    id_cols = {
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "total_goals", "result",
        "btts", "over_2_5", "over35",
    }

    # Features that leak match outcome (post-match statistics)
    leaky_cols = {
        "home_xg", "away_xg",
        "home_shots", "away_shots",
        "home_shots_target", "away_shots_target",
        "home_corners", "away_corners",
        "home_fouls", "away_fouls",
        "home_yellow", "away_yellow",
        "home_red", "away_red",
    }
    if exclude_leaky:
        id_cols = id_cols | leaky_cols
        logger.info("Excluding %d leaky post-match features", len(leaky_cols & set(df.columns)))

    feature_cols = sorted([
        c for c in df.columns
        if c not in id_cols
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
        and df[c].notna().sum() > 0
    ])

    # Rolling-only: only team rolling features (exclude odds, H2H, league)
    if only_rolling:
        rolling_prefixes = ("h_rolling", "a_rolling", "h_cumavg", "a_cumavg", "diff_", "expected_total")
        feature_cols = [c for c in feature_cols if c.startswith(rolling_prefixes)]
        logger.info("Rolling-only mode: %d features (excluded odds/H2H/league)", len(feature_cols))

    # Handle remaining NaN
    for col in feature_cols:
        na_count = df_train[col].isna().sum()
        if na_count > 0:
            fill_val = df_train[col].median()
            for subset in [df_train, df_val, df_test]:
                subset[col] = subset[col].fillna(fill_val)

    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values
    X_test = df_test[feature_cols].values
    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values

    logger.info(
        "Features: %d | Train: %d, Val: %d, Test: %d",
        len(feature_cols), len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "Target distribution — train: %.1f%%, val: %.1f%%, test: %.1f%%",
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )

    return df_train, df_val, df_test, y_train, y_val, y_test, feature_cols


# ═══════════════════════════════════════════════════════════
#  2. Train helper
# ═══════════════════════════════════════════════════════════


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict | None = None,
) -> Any:
    """Train XGBoost with early stopping."""
    import xgboost as xgb

    p = {**(params or XGB_PARAMS)}
    use_early_stopping = p.pop("n_estimators", 500)

    model = xgb.XGBClassifier(**p, n_estimators=use_early_stopping, early_stopping_rounds=15)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict | None = None,
) -> Any:
    """Train LightGBM with early stopping."""
    import lightgbm as lgb

    p = {**(params or LGBM_PARAMS)}
    n_est = p.pop("n_estimators", 500)

    model = lgb.LGBMClassifier(**p, n_estimators=n_est)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )
    return model


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    params: dict | None = None,
) -> Any:
    """Train Random Forest (no early stopping needed)."""
    from sklearn.ensemble import RandomForestClassifier

    p = {**(params or RF_PARAMS)}
    model = RandomForestClassifier(**p)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    params: dict | None = None,
) -> Any:
    """Train Logistic Regression baseline."""
    from sklearn.linear_model import LogisticRegression

    p = {**(params or LR_PARAMS)}
    model = LogisticRegression(**p)
    model.fit(X_train, y_train)
    return model


def train_neural_network(
    X_train: np.ndarray, y_train: np.ndarray,
    params: dict | None = None,
) -> Any:
    """Train Neural Network (MLPClassifier)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    # NN needs scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    p = {**(params or NN_PARAMS)}
    model = MLPClassifier(**p)
    model.fit(X_train_scaled, y_train)
    return model, scaler


# ═══════════════════════════════════════════════════════════
#  3. Evaluation
# ═══════════════════════════════════════════════════════════


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Compute all evaluation metrics for a model."""
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, classification_report,
        confusion_matrix, f1_score, log_loss, precision_score,
        recall_score, roc_auc_score,
    )

    # Predict probabilities and classes
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "brier_score": round(brier_score_loss(y_test, y_prob), 5),
        "log_loss": round(log_loss(y_test, y_prob), 5),
        "accuracy": round(accuracy_score(y_test, y_pred), 5),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 5),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 5),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 5),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 5),
    }

    # Classification report as dict
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = {
        str(k): dict(v) if isinstance(v, dict) else float(v)
        for k, v in report.items()
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }
    metrics["n_test"] = int(len(y_test))

    return metrics


def compute_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
    n_repeats: int = 5,
    scaler: Any | None = None,
) -> list[dict[str, Any]]:
    """Compute permutation feature importance."""
    from sklearn.inspection import permutation_importance

    X = scaler.transform(X_val) if scaler is not None else X_val
    result = permutation_importance(
        model, X, y_val,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="neg_brier_score",
    )

    importances = []
    for i, col in enumerate(feature_cols):
        importances.append({
            "feature": col,
            "importance_mean": round(result.importances_mean[i], 6),
            "importance_std": round(result.importances_std[i], 6),
        })

    return sorted(importances, key=lambda x: abs(x["importance_mean"]), reverse=True)


def compute_shap_values(
    model: Any,
    X_val: np.ndarray,
    feature_cols: list[str],
    n_samples: int = 100,
    scaler: Any | None = None,
) -> list[dict[str, Any]] | None:
    """Compute SHAP values for feature importance (sample for speed)."""
    import shap

    try:
        X = scaler.transform(X_val) if scaler is not None else X_val

        # Use a random subset for speed
        if len(X) > n_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), n_samples, replace=False)
            X_subset = X[idx]
        else:
            X_subset = X

        explainer = shap.Explainer(model, X_subset, check_additivity=False)
        shap_values = explainer(X_subset)

        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_importances = [
            {"feature": feature_cols[i], "shap_value": round(float(mean_abs_shap[i]), 6)}
            for i in range(len(feature_cols))
        ]
        return sorted(shap_importances, key=lambda x: x["shap_value"], reverse=True)

    except Exception as exc:
        logger.warning("SHAP computation failed for %s: %s", type(model).__name__, exc)
        return None


def generate_feature_importance_plot(
    importances: list[dict[str, Any]],
    model_name: str,
    top_n: int = 20,
) -> str | None:
    """Generate and save feature importance bar plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        top = importances[:top_n]
        names = [x["feature"] for x in top][::-1]
        values = [x.get("importance_mean", x.get("shap_value", 0)) for x in top][::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(names)), values, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Features — {model_name} (Over/Under 2.5)")

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"over_under_feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved feature importance plot: %s", path)
        return str(path)
    except Exception as exc:
        logger.warning("Failed to generate plot: %s", exc)
        return None


def generate_confusion_matrix_plot(
    cm: dict[str, int], model_name: str,
) -> str | None:
    """Generate and save confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred No", "Pred Yes"],
                    yticklabels=["Actual No", "Actual Yes"],
                    ax=ax)
        ax.set_title(f"Confusion Matrix — {model_name}")

        path = FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return str(path)
    except Exception as exc:
        logger.warning("Failed to generate confusion matrix plot: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Over/Under prediction models")
    parser.add_argument("--input", default=None, help="Path to parquet file")
    parser.add_argument("--target", default=TARGET, help=f"Target column (default: {TARGET})")
    parser.add_argument("--lightweight", action="store_true", help="Skip NN, fewer estimators")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP (saves time)")
    parser.add_argument("--exclude-leaky", action="store_true",
                        help="Exclude post-match leaky features (xG, shots, corners)")
    parser.add_argument("--only-rolling", action="store_true",
                        help="Only use rolling team features (no odds/H2H/league)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve input path
    input_path = Path(args.input) if args.input else find_latest_parquet()
    target_col = args.target

    print("=" * 70)
    print(f"  OVER/UNDER MODEL TRAINING — Target: {target_col}")
    if args.exclude_leaky:
        print(f"  Mode: CLEAN (excluding leaky post-match features)")
    if args.only_rolling:
        print(f"  Mode: ROLLING-ONLY (team stats only, no odds/H2H/league)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_start = time.time()

    # ── Step 1: Load data ─────────────────────────────
    print("\n--- Step 1: Loading data ---")
    df_train, df_val, df_test, y_train, y_val, y_test, feature_cols = load_data(
        input_path, target_col=target_col,
        exclude_leaky=args.exclude_leaky,
        only_rolling=args.only_rolling,
    )
    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values
    X_test = df_test[feature_cols].values
    print(f"  {len(feature_cols)} features for {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # ── Step 2: Train all models ──────────────────────
    print("\n--- Step 2: Training models ---")
    models: dict[str, Any] = {}
    scalers: dict[str, Any] = {}
    training_times: dict[str, float] = {}

    # 2a. XGBoost (primary)
    print("  [1/5] XGBoost (primary)...")
    start = time.time()
    models["XGBoost"] = train_xgboost(X_train, y_train, X_val, y_val)
    training_times["XGBoost"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['XGBoost']:.1f}s")

    # 2b. LightGBM
    print("  [2/5] LightGBM...")
    start = time.time()
    models["LightGBM"] = train_lightgbm(X_train, y_train, X_val, y_val)
    training_times["LightGBM"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['LightGBM']:.1f}s")

    # 2c. Random Forest
    print("  [3/5] Random Forest...")
    start = time.time()
    # Use fewer trees for speed
    rf_params = {**RF_PARAMS, "n_estimators": 150} if args.lightweight else RF_PARAMS
    models["Random Forest"] = train_random_forest(X_train, y_train, rf_params)
    training_times["Random Forest"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['Random Forest']:.1f}s")

    # 2d. Logistic Regression
    print("  [4/5] Logistic Regression (baseline)...")
    start = time.time()
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)
    training_times["Logistic Regression"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['Logistic Regression']:.1f}s")

    # 2e. Neural Network
    nn_trained = False
    if not args.lightweight:
        print("  [5/5] Neural Network (MLP)...")
        start = time.time()
        try:
            models["Neural Network"], scalers["Neural Network"] = train_neural_network(X_train, y_train)
            training_times["Neural Network"] = round(time.time() - start, 1)
            nn_trained = True
            print(f"    Done in {training_times['Neural Network']:.1f}s")
        except Exception as exc:
            print(f"    SKIPPED — {exc}")
    else:
        print("  [5/5] Neural Network — SKIPPED (--lightweight)")

    # ── Step 3: Evaluate ──────────────────────────────
    print("\n--- Step 3: Evaluating models ---")
    all_metrics: list[dict[str, Any]] = []
    best_model_name = None
    best_brier = 1.0

    for model_name, model in models.items():
        scaler = scalers.get(model_name)
        metrics = evaluate_model(model, X_test, y_test, model_name, scaler)
        metrics["training_time_s"] = training_times.get(model_name, 0)
        all_metrics.append(metrics)

        brier = metrics["brier_score"]
        is_best = " [BEST]" if brier < best_brier else ""
        if brier < best_brier:
            best_brier = brier
            best_model_name = model_name

        print(f"  {model_name:25s} | Brier={metrics['brier_score']:.4f} | "
              f"LogLoss={metrics['log_loss']:.4f} | Acc={metrics['accuracy']:.4f} | "
              f"AUC={metrics['roc_auc']:.4f}{is_best}")

    print(f"\n  Best model: {best_model_name} (Brier={best_brier:.4f})")

    # ── Step 4: Feature importance ────────────────────
    print("\n--- Step 4: Feature importance ---")
    best_model = models[best_model_name]
    best_scaler = scalers.get(best_model_name)

    # Permutation importance (on val set)
    print("  Computing permutation importance...")
    perm_imp = compute_permutation_importance(
        best_model, X_val, y_val, feature_cols,
        n_repeats=3 if args.lightweight else 5,
        scaler=best_scaler,
    )

    if perm_imp:
        print("  Top 10 features:")
        for item in perm_imp[:10]:
            print(f"    {item['feature']:45s} {item['importance_mean']:.5f}")

    # SHAP values
    shap_imp = None
    if not args.skip_shap and best_model_name not in ("Logistic Regression", "Random Forest"):
        print("  Computing SHAP values (sample=200)...")
        shap_imp = compute_shap_values(
            best_model, X_val, feature_cols, n_samples=200, scaler=best_scaler,
        )
        if shap_imp:
            print("  Top 10 SHAP features:")
            for item in shap_imp[:10]:
                print(f"    {item['feature']:45s} {item['shap_value']:.5f}")

    # Generate plots
    imp_for_plot = shap_imp if shap_imp else perm_imp
    if imp_for_plot:
        plot_path = generate_feature_importance_plot(imp_for_plot, best_model_name)
        if plot_path:
            print(f"  Feature importance plot: {plot_path}")

    # Confusion matrix for best model
    best_metrics = [m for m in all_metrics if m["model"] == best_model_name][0]
    cm_path = generate_confusion_matrix_plot(best_metrics["confusion_matrix"], best_model_name)
    if cm_path:
        print(f"  Confusion matrix plot: {cm_path}")

    # ── Step 5: Save models ───────────────────────────
    print("\n--- Step 5: Saving models ---")
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved_models = {}

    for model_name, model in models.items():
        safe_name = model_name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"over_under_{safe_name}_{timestamp}.joblib"
        joblib.dump(model, model_path)
        saved_models[model_name] = str(model_path)
        print(f"  {model_name:25s} -> {model_path.name}")

    # Also save as the primary 'over_under_xgboost' for easy reference
    if "XGBoost" in models:
        xgb_path = MODELS_DIR / f"over_under_xgboost_{timestamp}.joblib"
        joblib.dump(models["XGBoost"], xgb_path)
        saved_models["XGBoost_primary"] = str(xgb_path)
        print(f"  {'XGBoost (primary)':25s} -> {xgb_path.name}")

    # ── Step 6: Save metrics report ───────────────────
    print("\n--- Step 6: Saving reports ---")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-model metrics
    metrics_path = REPORTS_DIR / f"over_under_xgboost_metrics_{timestamp}.json"
    best_metrics_clean = {
        k: v for k, v in best_metrics.items()
        if k not in ("classification_report",)
    }
    best_metrics_clean["permutation_importance_top20"] = perm_imp[:20] if perm_imp else []
    best_metrics_clean["shap_importance_top20"] = shap_imp[:20] if shap_imp else []
    best_metrics_clean["feature_importance_plot"] = str(plot_path) if imp_for_plot else None
    best_metrics_clean["confusion_matrix_plot"] = str(cm_path) if cm_path else None
    best_metrics_clean["saved_model"] = saved_models.get(best_model_name)
    best_metrics_clean["training_time_s"] = training_times.get(best_model_name, 0)
    best_metrics_clean["n_features"] = len(feature_cols)
    best_metrics_clean["n_train"] = len(X_train)
    best_metrics_clean["n_val"] = len(X_val)
    best_metrics_clean["n_test"] = len(X_test)

    with open(metrics_path, "w") as f:
        json.dump(best_metrics_clean, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Model comparison
    comparison = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": target_col,
        "source": str(input_path),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "best_model": best_model_name,
        "models": [],
    }
    for m in all_metrics:
        comparison["models"].append({
            "model": m["model"],
            "brier_score": m["brier_score"],
            "log_loss": m["log_loss"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "roc_auc": m["roc_auc"],
            "training_time_s": m.get("training_time_s", 0),
        })

    comparison_path = REPORTS_DIR / f"over_under_model_comparison_{timestamp}.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison: {comparison_path}")

    total_elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print(f"  [OK] TRAINING COMPLETE ({total_elapsed:.1f}s)")
    print("=" * 70)
    print(f"  Best model: {best_model_name} (Brier={best_brier:.4f})")
    print(f"  Models saved in: {MODELS_DIR}")
    print(f"  Reports in: {REPORTS_DIR}")
    print()


if __name__ == "__main__":
    main()
