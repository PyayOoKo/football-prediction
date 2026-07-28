"""
train_btts.py — Train specialist BTTS (Both Teams to Score) prediction models.

Trains 5 model types on preprocessed BTTS data:
1. Poisson Regression (primary — derived from Poisson goal distributions)
2. Logistic Regression (baseline)
3. XGBoost
4. Random Forest
5. Neural Network

Usage:
    python scripts/train_btts.py
    python scripts/train_btts.py --lightweight   # Skip NN, fewer iterations
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_btts")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "btts"

XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
    "early_stopping_rounds": 15,
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
#  1. Load data
# ═══════════════════════════════════════════════════════════


def find_latest_parquet() -> Path:
    files = sorted(PROCESSED_DIR.glob("btts_data_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No btts_data_*.parquet found. Run scripts/preprocess_btts.py first."
        )
    return files[-1]


def load_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           list[str]]:
    """Load BTTS parquet, split chronologically, return X/y splits + goals.

    Returns X_train, X_val, X_test, y_train, y_val, y_test,
            hg_train, hg_val, hg_test,  # home goals for Poisson model
            ag_train, ag_val, ag_test,  # away goals for Poisson model
            feature_cols
    """
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)

    assert "btts" in df.columns, "Target 'btts' not found!"
    assert "home_goals" in df.columns, "home_goals not found (needed for Poisson)!"
    assert "away_goals" in df.columns, "away_goals not found (needed for Poisson)!"

    df["date"] = pd.to_datetime(df["date"])

    # Chronological split
    train_mask = (df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2022)
    test_mask = (df["date"].dt.year >= 2023) & (df["date"].dt.year <= 2024)

    df_train_val = df[train_mask].copy().sort_values("date")
    df_test = df[test_mask].copy()

    split_idx = int(len(df_train_val) * 0.8)
    df_train = df_train_val.iloc[:split_idx].copy()
    df_val = df_train_val.iloc[split_idx:].copy()

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(df_train), len(df_val), len(df_test),
    )

    # Feature columns
    id_cols = {
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "total_goals", "result",
        "over_2_5", "btts",
    }
    feature_cols = sorted([
        c for c in df.columns
        if c not in id_cols
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
        and df[c].notna().sum() > 0
    ])

    # Impute remaining NaNs with training median
    for col in feature_cols:
        na_count = df_train[col].isna().sum()
        if na_count > 0:
            fill_val = df_train[col].median()
            for subset in [df_train, df_val, df_test]:
                subset[col] = subset[col].fillna(fill_val)

    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    y_train = df_train["btts"].values
    y_val = df_val["btts"].values
    y_test = df_test["btts"].values

    # Goals (for Poisson model)
    hg_train = df_train["home_goals"].values
    hg_val = df_val["home_goals"].values
    hg_test = df_test["home_goals"].values
    ag_train = df_train["away_goals"].values
    ag_val = df_val["away_goals"].values
    ag_test = df_test["away_goals"].values

    logger.info(
        "Features: %d | Train: %d, Val: %d, Test: %d",
        len(feature_cols), len(X_train), len(X_val), len(X_test),
    )
    logger.info("BTTS distribution — train: %.1f%%", y_train.mean() * 100)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        hg_train, hg_val, hg_test,
        ag_train, ag_val, ag_test,
        feature_cols,
    )


# ═══════════════════════════════════════════════════════════
#  2. Poisson Regression (Primary — BTTS via goal distribution)
# ═══════════════════════════════════════════════════════════


def train_poisson_btts(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    hg_train: np.ndarray, ag_train: np.ndarray,
    hg_val: np.ndarray, ag_val: np.ndarray,
) -> tuple[Any, Any, Any]:
    """Train two Poisson GLMs (home goals, away goals) and derive BTTS.

    BTTS probability = 1 - P(home=0) * P(away=0)
                     = 1 - exp(-lambda_home) * exp(-lambda_away)
                     = 1 - exp(-(lambda_home + lambda_away))

    Returns (model_home, model_away, scaler).
    """
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm

    # Poisson GLM for home goals
    logger.info("  Fitting Poisson GLM for home goals...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Add constant for intercept
    X_train_const = sm.add_constant(X_train_scaled)
    X_val_const = sm.add_constant(X_val_scaled)

    poisson_home = sm.GLM(
        hg_train, X_train_const,
        family=sm.families.Poisson(),
    ).fit(disp=False)

    # Poisson GLM for away goals
    logger.info("  Fitting Poisson GLM for away goals...")
    poisson_away = sm.GLM(
        ag_train, X_train_const,
        family=sm.families.Poisson(),
    ).fit(disp=False)

    return poisson_home, poisson_away, scaler


def predict_poisson_btts(
    model_home: Any, model_away: Any,
    X: np.ndarray, scaler: Any,
) -> np.ndarray:
    """Predict BTTS probability from Poisson goal expectations."""
    import statsmodels.api as sm
    X_scaled = scaler.transform(X)
    X_const = sm.add_constant(X_scaled)

    lambda_home = model_home.predict(X_const)
    lambda_away = model_away.predict(X_const)

    # P(BTTS) = 1 - P(home=0) * P(away=0)
    # P(team=0) = exp(-lambda)
    prob_btts = 1.0 - np.exp(-lambda_home) * np.exp(-lambda_away)
    return np.clip(prob_btts, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════
#  3. Standard classifiers
# ═══════════════════════════════════════════════════════════


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> Any:
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    n_est = p.pop("n_estimators")
    esr = p.pop("early_stopping_rounds")
    model = xgb.XGBClassifier(**p, n_estimators=n_est, early_stopping_rounds=esr)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_neural_network(
    X_train: np.ndarray, y_train: np.ndarray,
) -> tuple[Any, Any]:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = MLPClassifier(**NN_PARAMS)
    model.fit(X_train_scaled, y_train)
    return model, scaler


# ═══════════════════════════════════════════════════════════
#  4. Evaluation
# ═══════════════════════════════════════════════════════════


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray,
    model_name: str, scaler: Any = None,
    predict_fn: Any = None,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, classification_report,
        confusion_matrix, f1_score, log_loss, precision_score,
        recall_score, roc_auc_score,
    )

    if predict_fn is not None:
        y_prob = predict_fn(model, X_test, scaler)
    elif scaler is not None:
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

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = {
        str(k): dict(v) if isinstance(v, dict) else float(v)
        for k, v in report.items()
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    metrics["n_test"] = int(len(y_test))
    return metrics


def compute_permutation_importance(
    model: Any, X_val: np.ndarray, y_val: np.ndarray,
    feature_cols: list[str], n_repeats: int = 5,
    scaler: Any = None, predict_fn: Any = None,
) -> list[dict[str, Any]]:
    """Compute permutation feature importance using negative Brier score."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import brier_score_loss

    # For Poisson/custom models: pass raw X (scaling happens inside predict_fn)
    if predict_fn is not None:
        # predict_fn expects unscaled X and does scaling internally
        X_use = X_val

        def score_func(estimator, X_scrambled, y):
            prob = predict_fn(estimator, X_scrambled, scaler)
            return -brier_score_loss(y, prob)

        result = permutation_importance(
            model, X_use, y_val,
            n_repeats=n_repeats, random_state=42, n_jobs=-1,
            scoring=score_func,
        )
    else:
        # Standard sklearn models: scale if needed
        X_use = scaler.transform(X_val) if scaler is not None else X_val
        result = permutation_importance(
            model, X_use, y_val,
            n_repeats=n_repeats, random_state=42, n_jobs=-1,
            scoring="neg_brier_score",
        )

    importances = []
    for i, col in enumerate(feature_cols):
        importances.append({
            "feature": col,
            "importance_mean": round(float(result.importances_mean[i]), 6),
            "importance_std": round(float(result.importances_std[i]), 6),
        })
    return sorted(importances, key=lambda x: abs(x["importance_mean"]), reverse=True)


def compute_shap_values(
    model: Any, X_val: np.ndarray,
    feature_cols: list[str], n_samples: int = 200,
    scaler: Any = None, predict_fn: Any = None,
) -> list[dict[str, Any]] | None:
    """Compute SHAP values for feature importance (tree models only)."""
    try:
        import shap

        X = scaler.transform(X_val) if scaler is not None else X_val
        if len(X) > n_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), n_samples, replace=False)
            X_subset = X[idx]
        else:
            X_subset = X

        explainer = shap.Explainer(model, X_subset, check_additivity=False)
        shap_values = explainer(X_subset)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_importances = [
            {"feature": feature_cols[i], "shap_value": round(float(mean_abs_shap[i]), 6)}
            for i in range(len(feature_cols))
        ]
        return sorted(shap_importances, key=lambda x: x["shap_value"], reverse=True)

    except Exception as exc:
        logger.warning("SHAP computation failed for %s: %s", type(model).__name__, exc)
        return None


def get_linear_coefficients(
    model: Any, feature_cols: list[str], model_name: str,
) -> list[dict[str, Any]]:
    """Extract feature coefficients for linear models."""
    if hasattr(model, "coef_"):
        coefs = model.coef_.flatten()
        return sorted([
            {"feature": col, "coefficient": round(float(coef), 6)}
            for col, coef in zip(feature_cols, coefs)
        ], key=lambda x: abs(x["coefficient"]), reverse=True)
    return []


def generate_feature_importance_plot(
    importances: list[dict[str, Any]],
    model_name: str, top_n: int = 20,
) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        top = importances[:top_n]
        names = [x["feature"] for x in top][::-1]
        values = [    x.get("importance_mean", x.get("coefficient", x.get("shap_value", 0))) for x in top][::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(names)), values, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Features — {model_name} (BTTS)")

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"btts_feature_importance_{model_name.lower().replace(' ', '_')}.png"
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False

    try:
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4))

        if has_sns:
            import seaborn as sns
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred No", "Pred Yes"],
                        yticklabels=["Actual No", "Actual Yes"], ax=ax)
        else:
            ax.imshow(matrix, cmap="Blues", interpolation="nearest")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred No", "Pred Yes"])
            ax.set_yticklabels(["Actual No", "Actual Yes"])

        ax.set_title(f"Confusion Matrix — {model_name}")
        path = FIGURES_DIR / f"btts_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
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

    parser = argparse.ArgumentParser(description="Train BTTS prediction models")
    parser.add_argument("--input", default=None, help="Path to parquet file")
    parser.add_argument("--lightweight", action="store_true", help="Skip NN, fewer estimators")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = Path(args.input) if args.input else find_latest_parquet()

    print("=" * 70)
    print(f"  BTTS MODEL TRAINING")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    total_start = time.time()

    # ── Step 1: Load data ─────────────────────────────
    print("\n--- Step 1: Loading data ---")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     hg_train, hg_val, hg_test,
     ag_train, ag_val, ag_test,
     feature_cols) = load_data(input_path)
    print(f"  {len(feature_cols)} features, {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # ── Step 2: Train models ──────────────────────────
    print("\n--- Step 2: Training models ---")
    models: dict[str, Any] = {}
    scalers: dict[str, Any] = {}
    predict_fns: dict[str, Any] = {}
    training_times: dict[str, float] = {}

    # 2a. Poisson Regression (primary)
    print("  [1/5] Poisson Regression (home+away GLM -> BTTS)...")
    start = time.time()
    poisson_home, poisson_away, poisson_scaler = train_poisson_btts(
        X_train, y_train, X_val, y_val,
        hg_train, ag_train, hg_val, ag_val,
    )
    models["Poisson GLM"] = (poisson_home, poisson_away)
    scalers["Poisson GLM"] = poisson_scaler
    # Wrap predict_fn so evaluate_model can call it with (model, X, scaler) signature
    def _poisson_predict(model_tuple, X, scaler):
        return predict_poisson_btts(model_tuple[0], model_tuple[1], X, scaler)
    predict_fns["Poisson GLM"] = _poisson_predict
    training_times["Poisson GLM"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['Poisson GLM']:.1f}s")

    # 2b. Logistic Regression
    print("  [2/5] Logistic Regression...")
    start = time.time()
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)
    training_times["Logistic Regression"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['Logistic Regression']:.1f}s")

    # 2c. XGBoost
    print("  [3/5] XGBoost...")
    start = time.time()
    models["XGBoost"] = train_xgboost(X_train, y_train, X_val, y_val)
    training_times["XGBoost"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['XGBoost']:.1f}s")

    # 2d. Random Forest
    print("  [4/5] Random Forest...")
    start = time.time()
    rf_params = {**RF_PARAMS, "n_estimators": 150} if args.lightweight else RF_PARAMS
    models["Random Forest"] = train_random_forest(X_train, y_train)
    training_times["Random Forest"] = round(time.time() - start, 1)
    print(f"    Done in {training_times['Random Forest']:.1f}s")

    # 2e. Neural Network
    if not args.lightweight:
        print("  [5/5] Neural Network (MLP)...")
        start = time.time()
        try:
            models["Neural Network"], scalers["Neural Network"] = train_neural_network(X_train, y_train)
            training_times["Neural Network"] = round(time.time() - start, 1)
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
        predict_fn = predict_fns.get(model_name)

        if model_name == "Poisson GLM":
            # Poisson has custom predict function
            metrics = evaluate_model(
                model, X_test, y_test, model_name,
                scaler=scaler, predict_fn=predict_fn,
            )
        else:
            metrics = evaluate_model(model, X_test, y_test, model_name, scaler=scaler)

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
    best_predict_fn = predict_fns.get(best_model_name)

    imp_data: list[dict[str, Any]] | None = None
    shap_data: list[dict[str, Any]] | None = None

    if best_model_name == "Poisson GLM":
        # For Poisson: show coefficients from both GLMs
        coefs = []
        for goal_type, glm_model in [("home_goals", best_model[0]), ("away_goals", best_model[1])]:
            params = glm_model.params[1:]  # skip intercept
            for i, col in enumerate(feature_cols):
                coefs.append({
                    "feature": col,
                    f"coefficient_{goal_type}": round(float(params[i]), 6),
                })
        # Aggregate by mean absolute coefficient
        agg = {}
        for c in coefs:
            name = c["feature"]
            agg.setdefault(name, [])
            agg[name].append(abs(c.get(f"coefficient_home_goals", 0)))
            agg[name].append(abs(c.get(f"coefficient_away_goals", 0)))
        imp_data = [
            {"feature": k, "importance_mean": round(sum(v) / len(v), 6)}
            for k, v in agg.items()
        ]
        imp_data.sort(key=lambda x: x["importance_mean"], reverse=True)
        print("  Top 10 Poisson features:")
        for item in imp_data[:10]:
            print(f"    {item['feature']:45s} {item['importance_mean']:.5f}")

    elif best_model_name == "Logistic Regression":
        imp_data = get_linear_coefficients(best_model, feature_cols, best_model_name)
        print("  Top 10 logistic coefficients:")
        for item in imp_data[:10]:
            print(f"    {item['feature']:45s} {item['coefficient']:.5f}")
    else:
        # Tree-based: permutation importance + SHAP
        print("  Computing permutation importance...")
        imp_data = compute_permutation_importance(
            best_model, X_val, y_val, feature_cols,
            n_repeats=3, scaler=best_scaler, predict_fn=best_predict_fn,
        )
        if imp_data:
            print("  Top 10 features:")
            for item in imp_data[:10]:
                print(f"    {item['feature']:45s} {item['importance_mean']:.5f}")

        # SHAP for tree models
        if best_model_name in ("XGBoost", "Random Forest"):
            print("  Computing SHAP values...")
            shap_data = compute_shap_values(
                best_model, X_val, feature_cols,
                n_samples=200, scaler=best_scaler,
            )
            if shap_data:
                print("  Top 10 SHAP features:")
                for item in shap_data[:10]:
                    print(f"    {item['feature']:45s} {item['shap_value']:.5f}")

    # Generate importance plot (prefer SHAP if available, else permutation/coefficients)
    plot_data = shap_data if shap_data else imp_data
    plot_path: str | None = None
    if plot_data:
        plot_path = generate_feature_importance_plot(plot_data, best_model_name)
        if plot_path:
            print(f"  Feature importance plot: {plot_path}")

    # Confusion matrix
    best_metrics = [m for m in all_metrics if m["model"] == best_model_name][0]
    cm_path = generate_confusion_matrix_plot(best_metrics["confusion_matrix"], best_model_name)
    if cm_path:
        print(f"  Confusion matrix plot: {cm_path}")

    # ── Step 5: Save models ───────────────────────────
    print("\n--- Step 5: Saving models ---")
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved_models = {}

    # Save Poisson as tuple
    poisson_path = MODELS_DIR / f"btts_poisson_{timestamp}.joblib"
    joblib.dump((models["Poisson GLM"], poisson_scaler), poisson_path)
    saved_models["Poisson GLM"] = str(poisson_path)
    print(f"  {'Poisson GLM (primary)':25s} -> {poisson_path.name}")

    # Save other models
    for model_name, model in models.items():
        if model_name == "Poisson GLM":
            continue
        safe_name = model_name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"btts_{safe_name}_{timestamp}.joblib"
        joblib.dump(model, model_path)
        saved_models[model_name] = str(model_path)
        print(f"  {model_name:25s} -> {model_path.name}")

    # ── Step 6: Save reports ──────────────────────────
    print("\n--- Step 6: Saving reports ---")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORTS_DIR / f"btts_metrics_{timestamp}.json"
    best_clean = {
        k: v for k, v in best_metrics.items()
        if k not in ("classification_report",)
    }
    if imp_data:
        best_clean["feature_importance_top20"] = imp_data[:20]
    if shap_data:
        best_clean["shap_importance_top20"] = shap_data[:20]
    best_clean["feature_importance_plot"] = str(plot_path) if plot_path else None
    best_clean["confusion_matrix_plot"] = str(cm_path) if cm_path else None
    best_clean["saved_model"] = saved_models.get(best_model_name)
    best_clean["n_features"] = len(feature_cols)
    best_clean["n_train"] = len(X_train)
    best_clean["n_val"] = len(X_val)
    best_clean["n_test"] = len(X_test)

    with open(metrics_path, "w") as f:
        json.dump(best_clean, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    comparison = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": "btts",
        "source": str(input_path),
        "n_features": len(feature_cols),
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

    comparison_path = REPORTS_DIR / f"btts_model_comparison_{timestamp}.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison: {comparison_path}")

    total_elapsed = time.time() - total_start

    print()
    print("=" * 70)
    print(f"  [OK] BTTS TRAINING COMPLETE ({total_elapsed:.1f}s)")
    print("=" * 70)
    print(f"  Best model: {best_model_name} (Brier={best_brier:.4f})")
    print(f"  Models saved in: {MODELS_DIR}")
    print(f"  Reports in: {REPORTS_DIR}")
    print()


if __name__ == "__main__":
    main()
