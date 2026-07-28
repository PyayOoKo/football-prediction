"""
derive_btts_implied.py — Derive BTTS implied odds from 1X2 + O/U markets.

The-Odds-Api and football-data.co.uk do not provide BTTS market odds.
This script trains a model to estimate what the market "thinks" about
BTTS probability based on the 1X2 and O/U odds we DO have.

The trained model can then be used to generate BTTS implied probabilities
for matches where we have 1X2 and O/U odds but no direct BTTS odds.

Approach
--------
1. Load historical matches with 1X2 odds, O/U odds, and actual BTTS outcomes
2. Train a classifier: P(BTTS=Yes) = f(home_odds, draw_odds, away_odds,
   over25_odds, under25_odds, league, date features)
3. The model learns the market's implicit BTTS pricing from correlated markets
4. Save model for use in backtesting and value betting

Usage
-----
    python scripts/derive_btts_implied.py
    python scripts/derive_btts_implied.py --force  # Retrain even if model exists
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
logger = logging.getLogger("derive_btts_implied")

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

# Input data (must contain 1X2 odds, O/U odds, and btts outcome)
INPUT_DATA = PROCESSED_DIR / "over_under_data_20260725_222214.parquet"
OUTPUT_MODEL = MODELS_DIR / "btts_implied_from_markets.joblib"


def load_training_data() -> pd.DataFrame:
    """Load and prepare training data.

    Features from market odds that correlate with BTTS probability:
    - 1X2 odds (home, draw, away) → implied probabilities
    - O/U 2.5 odds → implied probabilities
    - League (categorical — BTTS rates vary by league)
    - Interaction: high home odds + low O/U odds → lower BTTS probability
    - Derived: 1X2 implied probabilities sum (margin)
    """
    df = pd.read_parquet(INPUT_DATA)

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])

    # Filter to matches with all required odds
    required = ["home_odds", "draw_odds", "away_odds", "over25_odds",
                 "under25_odds", "btts", "date", "league"]
    df = df.dropna(subset=required).copy()

    logger.info("Loaded %d matches with full odds + BTTS outcome", len(df))

    # ── Feature engineering ────────────────────────────

    # Implied probabilities (1/odds)
    df["home_imp"] = 1.0 / df["home_odds"]
    df["draw_imp"] = 1.0 / df["draw_odds"]
    df["away_imp"] = 1.0 / df["away_odds"]
    df["over25_imp"] = 1.0 / df["over25_odds"]
    df["under25_imp"] = 1.0 / df["under25_odds"]

    # Bookmaker margin (overround)
    df["margin_1x2"] = df["home_imp"] + df["draw_imp"] + df["away_imp"]
    df["margin_ou"] = df["over25_imp"] + df["under25_imp"]

    # Normalised probabilities (vig-free)
    df["home_prob"] = df["home_imp"] / df["margin_1x2"]
    df["draw_prob"] = df["draw_imp"] / df["margin_1x2"]
    df["away_prob"] = df["away_imp"] / df["margin_1x2"]
    df["over25_prob"] = df["over25_imp"] / df["margin_ou"]
    df["under25_prob"] = df["under25_imp"] / df["margin_ou"]

    # 📊 Key insight: BTTS is correlated with total goals expectation
    # High over25_prob → higher BTTS probability
    # High home_prob + low away_prob → lower BTTS (dominant team)
    # High draw_prob → moderate BTTS (competitive match)

    # Interaction features
    df["favorite_imp"] = df[["home_imp", "away_imp"]].min(axis=1)
    df["underdog_imp"] = df[["home_imp", "away_imp"]].max(axis=1)
    df["favorite_dominance"] = df["underdog_imp"] / df["favorite_imp"].clip(lower=0.001)

    # Expected total goals implied from O/U
    # Simple approximation: more accurate method uses Poisson, but good enough
    df["ou_ratio"] = df["over25_prob"] / df["under25_prob"].clip(lower=0.001)

    # Time features (BTTS rates change over time)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Rolling BTTS rate per league (last 100 matches) — uses ONLY past data
    df = df.sort_values(["league", "date"])
    df["league_btts_rolling"] = df.groupby("league")["btts"].transform(
        lambda x: x.rolling(100, min_periods=10).mean().shift(1)
    )

    # League BTC rate — computed after split to avoid leakage
    # (will be done inside train_model on training data only)
    df["league_btts_rate"] = 0.5  # placeholder

    return df


def train_model(df: pd.DataFrame) -> tuple[Any, list[str], dict[str, Any]]:
    """Train a model to predict BTTS from market odds.

    Returns: (model, feature_cols, metrics)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        brier_score_loss, log_loss, accuracy_score, roc_auc_score,
    )

    # Feature columns (league_btts_rate omitted — league_btts_rolling already captures it safely)
    feature_cols = [
        "home_imp", "draw_imp", "away_imp",
        "over25_imp", "under25_imp",
        "margin_1x2", "margin_ou",
        "home_prob", "draw_prob", "away_prob", "over25_prob",
        "favorite_imp", "underdog_imp", "favorite_dominance",
        "ou_ratio",
        "league_btts_rolling",
        "year", "month",
    ]

    # Split by time (temporal) FIRST
    split_date = df["date"].quantile(0.8)
    train_mask = df["date"] <= split_date
    test_mask = df["date"] > split_date

    # Build feature matrix AFTER any feature updates
    X = df[feature_cols].values.astype(np.float32)
    y = df["btts"].values

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Handle any remaining NaN values
    train_medians = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), train_medians, X_train)
    X_test = np.where(np.isnan(X_test), train_medians, X_test)

    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # ── Train models ──────────────────────────────────

    metrics_list = []

    # Logistic Regression (baseline — interpretable)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    metrics_list.append({
        "model": "LogisticRegression",
        "brier": float(brier_score_loss(y_test, lr_prob)),
        "log_loss": float(log_loss(y_test, lr_prob)),
        "accuracy": float(accuracy_score(y_test, lr_prob > 0.5)),
        "auc": float(roc_auc_score(y_test, lr_prob)),
    })

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=50,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    metrics_list.append({
        "model": "RandomForest",
        "brier": float(brier_score_loss(y_test, rf_prob)),
        "log_loss": float(log_loss(y_test, rf_prob)),
        "accuracy": float(accuracy_score(y_test, rf_prob > 0.5)),
        "auc": float(roc_auc_score(y_test, rf_prob)),
    })

    # XGBoost
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=4, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        metrics_list.append({
            "model": "XGBoost",
            "brier": float(brier_score_loss(y_test, xgb_prob)),
            "log_loss": float(log_loss(y_test, xgb_prob)),
            "accuracy": float(accuracy_score(y_test, xgb_prob > 0.5)),
            "auc": float(roc_auc_score(y_test, xgb_prob)),
        })
    except ImportError:
        xgb_model = None
        logger.info("XGBoost not available, skipping")

    # Select best model (lowest Brier)
    best = min(metrics_list, key=lambda m: m["brier"])
    logger.info("Best model: %s (Brier=%.4f)", best["model"], best["brier"])

    # Return the best model
    if best["model"] == "RandomForest":
        model = rf
    elif best["model"] == "XGBoost":
        model = xgb_model
    else:
        model = lr

    # Feature importance for Random Forest
    if best["model"] in ("RandomForest",):
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logger.info("Top features:")
        for _, row in importance.head(10).iterrows():
            logger.info("  %s: %.4f", row["feature"], row["importance"])
    elif best["model"] == "LogisticRegression":
        importance = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": model.coef_[0],
        }).sort_values("coefficient", ascending=False)
        logger.info("Top positive coefficients:")
        for _, row in importance.head(5).iterrows():
            logger.info("  %s: %.4f", row["feature"], row["coefficient"])

    # Calibration check
    pred_bins = np.clip((rf_prob * 10).astype(int), 0, 9)
    calibration = []
    for bin_idx in range(10):
        mask = pred_bins == bin_idx
        if mask.sum() >= 50:
            predicted = rf_prob[mask].mean()
            actual = y_test[mask].mean()
            calibration.append({
                "bin": f"{bin_idx*10}-{(bin_idx+1)*10}%",
                "n": int(mask.sum()),
                "predicted": round(float(predicted), 3),
                "actual": round(float(actual), 3),
            })

    metrics = {
        "best_model": best["model"],
        "all_models": metrics_list,
        "calibration": calibration,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "btts_rate_train": round(float(y_train.mean()), 4),
        "btts_rate_test": round(float(y_test.mean()), 4),
    }

    return model, feature_cols, metrics


def save_model(model: Any, feature_cols: list[str], metrics: dict[str, Any]):
    """Save model with metadata."""
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(model_data, OUTPUT_MODEL)
    logger.info("Saved model: %s", OUTPUT_MODEL)

    # Save metrics separately
    metrics_path = MODELS_DIR / "btts_implied_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics: %s", metrics_path)


def load_model() -> dict[str, Any] | None:
    """Load saved model if it exists."""
    import joblib

    if not OUTPUT_MODEL.exists():
        logger.info("No existing model found at %s", OUTPUT_MODEL)
        return None

    data = joblib.load(OUTPUT_MODEL)
    logger.info("Loaded existing model (generated: %s)", data.get("generated", "?"))
    return data


def predict_btts_implied(
    home_odds: float, draw_odds: float, away_odds: float,
    over25_odds: float, under25_odds: float,
    league: str = "E0",
    year: int = 2024, month: int = 6,
    league_btts_rate: float = 0.50,
    league_btts_rolling: float = 0.50,
) -> dict[str, Any]:
    """Generate BTTS implied probability for a single match.

    Uses the trained model to estimate what the market "thinks" about
    BTTS probability based on correlated market odds.

    Returns dict with btts_yes_prob, btts_no_prob, and derived odds.
    """
    model_data = load_model()
    if model_data is None:
        return {"error": "No trained model available. Run derive_btts_implied.py first."}

    model = model_data["model"]
    feature_cols = model_data["feature_cols"]

    # Build feature vector
    home_imp = 1.0 / home_odds
    draw_imp = 1.0 / draw_odds
    away_imp = 1.0 / away_odds
    over25_imp = 1.0 / over25_odds
    under25_imp = 1.0 / under25_odds
    margin_1x2 = home_imp + draw_imp + away_imp
    margin_ou = over25_imp + under25_imp
    home_prob = home_imp / margin_1x2
    draw_prob = draw_imp / margin_1x2
    away_prob = away_imp / margin_1x2
    over25_prob = over25_imp / margin_ou
    favorite_imp = min(home_imp, away_imp)
    underdog_imp = max(home_imp, away_imp)
    favorite_dominance = underdog_imp / max(favorite_imp, 0.001)
    ou_ratio = over25_prob / max(under25_imp / margin_ou, 0.001)

    features = {
        "home_imp": home_imp,
        "draw_imp": draw_imp,
        "away_imp": away_imp,
        "over25_imp": over25_imp,
        "under25_imp": under25_imp,
        "margin_1x2": margin_1x2,
        "margin_ou": margin_ou,
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "over25_prob": over25_prob,
        "favorite_imp": favorite_imp,
        "underdog_imp": underdog_imp,
        "favorite_dominance": favorite_dominance,
        "ou_ratio": ou_ratio,
        "league_btts_rate": league_btts_rate,
        "league_btts_rolling": league_btts_rolling,
        "year": year,
        "month": month,
    }

    X = np.array([[features[c] for c in feature_cols]], dtype=np.float32)

    if hasattr(model, "predict_proba"):
        btts_prob = model.predict_proba(X)[0, 1]
    else:
        btts_prob = float(model.predict(X)[0])

    btts_no_prob = 1.0 - btts_prob

    # Vig-free derived odds
    margin = 0.05  # Assume 5% margin
    btts_yes_odds = 1.0 / (btts_prob * (1 + margin))
    btts_no_odds = 1.0 / (btts_no_prob * (1 + margin))

    return {
        "btts_yes_prob": round(float(btts_prob), 4),
        "btts_no_prob": round(float(btts_no_prob), 4),
        "btts_yes_odds": round(float(btts_yes_odds), 2),
        "btts_no_odds": round(float(btts_no_odds), 2),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Derive BTTS implied odds from 1X2 + O/U markets"
    )
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if model exists")
    args = parser.parse_args()

    print("=" * 70)
    print("  BTTS IMPLIED ODDS DERIVATION")
    print("  Train model to estimate BTTS odds from 1X2 + O/U markets")
    print("=" * 70)

    # Check if model already exists
    if OUTPUT_MODEL.exists() and not args.force:
        print(f"\n  Model already exists: {OUTPUT_MODEL}")
        print("  Use --force to retrain.")
        model_data = load_model()
        if model_data:
            metrics = model_data.get("metrics", {})
            best = metrics.get("best_model", "?")
            print(f"  Best model: {best}")
            for m in metrics.get("all_models", []):
                print(f"    {m['model']:>20s}: Brier={m['brier']:.4f}, AUC={m['auc']:.4f}")
        return

    # Step 1: Load data
    print("\n--- Step 1: Loading training data ---")
    start = time.time()
    df = load_training_data()
    print(f"  {len(df)} matches loaded ({time.time()-start:.1f}s)")

    # Step 2: Train model
    print("\n--- Step 2: Training BTTS implied model ---")
    start = time.time()
    model, feature_cols, metrics = train_model(df)
    print(f"  Training complete ({time.time()-start:.1f}s)")

    # Step 3: Results
    print("\n--- Step 3: Results ---")
    print(f"  Best model: {metrics['best_model']}")
    print(f"  Train size: {metrics['n_train']} | Test size: {metrics['n_test']}")
    print(f"  BTTS rate: train={metrics['btts_rate_train']*100:.1f}%, "
          f"test={metrics['btts_rate_test']*100:.1f}%")
    print()
    for m in metrics["all_models"]:
        print(f"  {m['model']:>20s}: Brier={m['brier']:.4f}  "
              f"LogLoss={m['log_loss']:.4f}  "
              f"Acc={m['accuracy']:.3f}  "
              f"AUC={m['auc']:.3f}")

    # Calibration
    if metrics.get("calibration"):
        print(f"\n  {'Bin':>12s} | {'N':>5s} | {'Pred':>6s} | {'Actual':>6s}")
        print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}")
        for c in metrics["calibration"]:
            print(f"  {c['bin']:>12s} | {c['n']:>5d} | {c['predicted']:.3f} | {c['actual']:.3f}")

    # Step 4: Save
    print("\n--- Step 4: Saving model ---")
    save_model(model, feature_cols, metrics)

    # Example predictions
    print("\n--- Example Predictions ---")
    examples = [
        ("Manchester City vs Everton", 1.10, 8.50, 21.00, 1.30, 3.50),
        ("Crystal Palace vs West Ham", 2.30, 3.30, 3.20, 1.85, 1.95),
        ("Wolves vs Newcastle", 3.10, 3.40, 2.25, 1.80, 2.00),
        ("Chemnitzer vs ZFC Meuselwitz", 3.40, 3.60, 2.00, 2.10, 1.70),
    ]
    for label, h, d, a, ov, un in examples:
        result = predict_btts_implied(h, d, a, ov, un, league="E0",
                                      league_btts_rate=0.53, league_btts_rolling=0.52)
        if "error" not in result:
            print(f"  {label:<45s} BTTS: Yes={result['btts_yes_prob']:.1%} "
                  f"(~{result['btts_yes_odds']:.2f}) / "
                  f"No={result['btts_no_prob']:.1%}")

    print(f"\n  Done! Model saved: {OUTPUT_MODEL}")
    print()


if __name__ == "__main__":
    main()
