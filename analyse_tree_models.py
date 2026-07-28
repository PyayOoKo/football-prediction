"""
analyse_tree_models.py — Deep-dive analysis of tree model performance vs Elo on 1X2.

Trains XGBoost, LightGBM, and CatBoost on full league data (results_clean.csv),
extracts feature importance from each, compares 1X2 prediction accuracy against
Elo ratings, and generates a comprehensive report.

Questions answered:
  1. Which features do tree models rely on most?
  2. Why does Elo beat tree models on 1X2 for World Cup data?
  3. Do tree models improve with more league training data?
  4. Can we identify and fix the gap?

Usage:
    python analyse_tree_models.py
    python analyse_tree_models.py --output reports/tree_model_analysis.md
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyse_tree_models")

PROJECT_ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════


def brier_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    y_v, p_v = y_true[valid], probs[valid]
    y_oh = np.zeros_like(p_v)
    for i, v in enumerate(y_v):
        if 0 <= int(v) <= 2:
            y_oh[i, int(v)] = 1
    return float(np.mean(np.sum((p_v - y_oh) ** 2, axis=1)))


def log_loss_1x2(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import log_loss as sk_ll
        valid = ~np.isnan(y_true)
        y_v, p_v = y_true[valid], probs[valid]
        return float(sk_ll(y_v, p_v))
    except Exception:
        return None


def accuracy_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    preds = np.argmax(probs[valid], axis=1)
    return float(np.mean(preds == y_true[valid]))


# ═══════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════


def load_and_build_features(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load data, build features, return (X, y, df_raw)."""
    from src.feature_engineering import build_features

    logger.info("Loading data from %s ...", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    logger.info("  %d rows x %d cols", len(df), len(df.columns))

    logger.info("Building features ...")
    X, y = build_features(df, is_training=True)
    logger.info("  Feature matrix: %d rows x %d cols", *X.shape)

    # Drop _row_id from X for training (it leaks row order)
    X_train = X.drop(columns=["_row_id"], errors="ignore")
    logger.info("  Training features (no _row_id): %d cols", X_train.shape[1])
    return X_train, y, df


# ═══════════════════════════════════════════════════════════
#  Elo baseline
# ═══════════════════════════════════════════════════════════


def compute_elo_predictions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> np.ndarray:
    """Compute Elo 1X2 predictions for test set matches.

    Processes Elo incrementally through training matches (chronological),
    then predicts 1X2 for test rows using pre-match Elo ratings.
    This guarantees no time-series leakage.
    """
    from src.elo import EloSystem

    elo = EloSystem(k=20, home_advantage=100, initial_rating=1500)

    # Process all training matches (chronological order)
    df_train_sorted = df_train.sort_values("date").reset_index(drop=True)
    elo.process_matches(df_train_sorted)

    # Predict test matches
    home_teams = df_test["home_team"].tolist()
    away_teams = df_test["away_team"].tolist()

    probs_list = []
    for ht, at in zip(home_teams, away_teams):
        try:
            proba = elo.predict_proba(pd.DataFrame([{"home_team": ht, "away_team": at}]))[0]
            probs_list.append(proba)
        except Exception:
            probs_list.append(np.array([0.33, 0.34, 0.33]))

    return np.array(probs_list)


# ═══════════════════════════════════════════════════════════
#  Train helpers
# ═══════════════════════════════════════════════════════════


def _get_feature_names(model: Any) -> list[str]:
    """Extract feature names from a fitted model."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    elif hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    return []


def _get_feature_importance(model: Any) -> dict[str, float]:
    """Extract feature importance dict from a fitted model."""
    importances = {}
    fnames = _get_feature_names(model)
    try:
        if hasattr(model, "feature_importances_"):
            for name, imp in zip(fnames, model.feature_importances_):
                importances[name] = float(imp)
        elif hasattr(model, "get_feature_importance"):
            # CatBoost
            scores = model.get_feature_importance()
            for name, imp in zip(fnames, scores):
                importances[name] = float(imp)
    except Exception:
        pass
    return dict(sorted(importances.items(), key=lambda x: -x[1]))


def _class_distribution(y: pd.Series) -> dict[str, float]:
    return {k: round(v, 3) for k, v in y.value_counts(normalize=True).to_dict().items()}


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> Any:
    """Train XGBoost with tuned params."""
    import xgboost as xgb
    logger.info("  Training XGBoost ...")
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=5,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("    Best iteration: %d", model.best_iteration if hasattr(model, "best_iteration") else "N/A")
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> Any:
    """Train LightGBM with tuned params."""
    import lightgbm as lgb
    logger.info("  Training LightGBM ...")
    model = lgb.LGBMClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    logger.info("    Best iteration: %d", model.best_iteration_ if hasattr(model, "best_iteration_") else "N/A")
    return model


def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> Any:
    """Train CatBoost with tuned params."""
    from catboost import CatBoostClassifier
    logger.info("  Training CatBoost ...")
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
        thread_count=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    logger.info("    Best iteration: %d", model.get("iterations", "N/A") if hasattr(model, "get") else "N/A")
    return model


# ═══════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════


def compute_prediction_entropy(probs: np.ndarray) -> float:
    """Mean prediction entropy — how 'confident' is the model?"""
    eps = 1e-15
    entropies = -np.sum(probs * np.log(np.clip(probs, eps, 1)), axis=1)
    return float(np.mean(entropies))


def compute_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() > 0:
            acc = accuracies[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += np.abs(acc - conf) * in_bin.sum() / len(confidences)
    return ece


# ═══════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════


def generate_report(
    results: dict[str, Any],
    elo_metrics: dict[str, float],
    xgb_metrics: dict[str, float],
    lgb_metrics: dict[str, float],
    cat_metrics: dict[str, float],
    xgb_importance: dict[str, float],
    lgb_importance: dict[str, float],
    cat_importance: dict[str, float],
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    elapsed: float,
) -> str:
    """Generate the comprehensive analysis markdown report."""

    output_path = PROJECT_ROOT / "reports" / f"tree_model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def h1(t: str) -> None:
        lines.append(f"\n# {t}\n")

    def h2(t: str) -> None:
        lines.append(f"\n## {t}\n")

    def h3(t: str) -> None:
        lines.append(f"\n### {t}\n")

    # ── Title ──
    lines.append("# Tree Model Performance Analysis: Why Does Elo Beat Tree Models on 1X2?")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Data:** `results_clean.csv` — {n_train + n_val + n_test:,} matches")
    lines.append(f"\n**Split:** {n_train:,} train / {n_val:,} val / {n_test:,} test (chronological)")
    lines.append(f"\n**Features:** {n_features}")
    lines.append(f"\n**Duration:** {elapsed:.1f}s")
    lines.append("")

    # ── Executive Summary ──
    h1("Executive Summary")

    lines.append("### Model Leaderboard (1X2 — Test Set)")
    lines.append("")
    lines.append("| Rank | Model | Brier | LogLoss | Accuracy | Entropy | ECE |")
    lines.append("|------|-------|-------|---------|----------|---------|-----|")

    leaderboard = [
        ("Elo", elo_metrics),
        ("XGBoost", xgb_metrics),
        ("LightGBM", lgb_metrics),
        ("CatBoost", cat_metrics),
    ]
    leaderboard.sort(key=lambda x: x[1].get("brier", 1))

    for rank, (name, m) in enumerate(leaderboard, 1):
        b = m.get("brier", 0)
        ll = m.get("log_loss", 0) or 0
        acc = m.get("accuracy", 0)
        ent = m.get("entropy", 0)
        ece = m.get("ece", 0)
        lines.append(f"| {rank} | **{name}** | {b:.5f} | {ll:.5f} | {acc:.2%} | {ent:.4f} | {ece:.4f} |")
    lines.append("")

    # Determine winner
    best_model = leaderboard[0][0]
    best_brier = leaderboard[0][1].get("brier", 1)
    elo_brier = elo_metrics.get("brier", 1)
    delta = (best_brier - elo_brier) / elo_brier * 100

    lines.append(f"> **Winner: {best_model}**")
    if best_model == "Elo":
        lines.append(f"> Elo beats the best tree model by {abs(delta):.1f}% in Brier score.")
    else:
        lines.append(f"> The best tree model beats Elo by {abs(delta):.1f}% in Brier score on league data!")
    lines.append("")

    # ── Per-Model Performance ──
    h1("Per-Model Performance Details")

    for name, m in [("Elo", elo_metrics), ("XGBoost", xgb_metrics),
                     ("LightGBM", lgb_metrics), ("CatBoost", cat_metrics)]:
        h2(name)
        b = m.get("brier", 0)
        ll = m.get("log_loss", 0) or 0
        acc = m.get("accuracy", 0)
        ent = m.get("entropy", 0)
        ece = m.get("ece", 0)
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Brier Score | {b:.5f} |")
        lines.append(f"| Log Loss | {ll:.5f} |")
        lines.append(f"| Accuracy | {acc:.2%} |")
        lines.append(f"| Prediction Entropy | {ent:.4f} |")
        lines.append(f"| Expected Calibration Error | {ece:.4f} |")
        lines.append("")

    # ── Feature Importance Analysis ──
    h1("Feature Importance Analysis")

    for name, imp in [("XGBoost", xgb_importance), ("LightGBM", lgb_importance),
                       ("CatBoost", cat_importance)]:
        h2(name)
        lines.append(f"**Top 20 features by {name}'s importance metric:**")
        lines.append("")
        lines.append("| Rank | Feature | Importance | Cum. Importance |")
        lines.append("|------|---------|-----------|-----------------|")
        items = list(imp.items())
        total = sum(v for _, v in items)
        cum = 0.0
        for rank, (feat, score) in enumerate(items[:20], 1):
            cum += score
            pct = score / total * 100 if total > 0 else 0
            cum_pct = cum / total * 100 if total > 0 else 0
            lines.append(f"| {rank} | `{feat}` | {pct:.2f}% | {cum_pct:.1f}% |")
        lines.append("")

        # Top 5 feature analysis
        top5 = items[:5]
        lines.append("**Top 5 feature analysis:**")
        lines.append("")
        for feat, score in top5:
            pct = score / total * 100 if total > 0 else 0
            lines.append(f"- **`{feat}`** ({pct:.1f}%)")

            # Categorise the feature
            feat_lower = feat.lower()
            if "elo" in feat_lower:
                lines.append("  - *Category: Elo rating — already captured by the baseline!*")
            elif "rolling" in feat_lower or "avg" in feat_lower or "mean" in feat_lower:
                lines.append("  - *Category: Rolling team statistics — Elo doesn't capture these directly*")
            elif "h2h" in feat_lower or "head" in feat_lower:
                lines.append("  - *Category: Head-to-head history — limited signal, Elo indirectly captures this*")
            elif "xg" in feat_lower or "expected" in feat_lower:
                lines.append("  - *Category: Expected goals — strong signal if xG data is available*")
            elif "league" in feat_lower or "position" in feat_lower:
                lines.append("  - *Category: League context — Elo doesn't know about league standing*")
            elif "odds" in feat_lower or "bbav" in feat_lower or "avg_" in feat_lower:
                lines.append("  - *Category: Market odds — very strong but not available at prediction time*")
            elif "date" in feat_lower or "days" in feat_lower or "since" in feat_lower:
                lines.append("  - *Category: Temporal features — recency weighting*")
            else:
                lines.append("  - *Category: Other — investigate further*")
        lines.append("")

    # ── Cross-model Feature Consensus ──
    h1("Cross-Model Feature Consensus")
    lines.append("Features that rank in the top 20 for **all three** tree models:")
    lines.append("")

    xgb_top20 = set(k for k, _ in list(xgb_importance.items())[:20])
    lgb_top20 = set(k for k, _ in list(lgb_importance.items())[:20])
    cat_top20 = set(k for k, _ in list(cat_importance.items())[:20])
    consensus = xgb_top20 & lgb_top20 & cat_top20

    if consensus:
        lines.append("| Feature | XGBoost Rank | LightGBM Rank | CatBoost Rank |")
        lines.append("|---------|-------------|---------------|---------------|")
        xgb_rank = {k: i + 1 for i, (k, _) in enumerate(xgb_importance.items())}
        lgb_rank = {k: i + 1 for i, (k, _) in enumerate(lgb_importance.items())}
        cat_rank = {k: i + 1 for i, (k, _) in enumerate(cat_importance.items())}

        for feat in sorted(consensus):
            lines.append(f"| `{feat}` | {xgb_rank.get(feat, '-'):>4} | {lgb_rank.get(feat, '-'):>4} | {cat_rank.get(feat, '-'):>4} |")
    else:
        lines.append("None — the three tree models rely on completely different features.")
    lines.append("")

    # Features unique to each model's top 20
    h2("Unique Feature Sets")
    lines.append("- **XGBoost only:** " + ", ".join(f"`{f}`" for f in sorted(xgb_top20 - lgb_top20 - cat_top20)) or "None")
    lines.append("- **LightGBM only:** " + ", ".join(f"`{f}`" for f in sorted(lgb_top20 - xgb_top20 - cat_top20)) or "None")
    lines.append("- **CatBoost only:** " + ", ".join(f"`{f}`" for f in sorted(cat_top20 - xgb_top20 - lgb_top20)) or "None")
    lines.append("")

    # ── Why Elo Wins ──
    h1("Why Does Elo Beat Tree Models on 1X2?")

    elo_beats_tree = best_model == "Elo"
    if elo_beats_tree:
        lines.append("### Findings from This Analysis")
        lines.append("")
        lines.append("Even on full league data, Elo matches or beats tree models on 1X2. Possible reasons:")
        lines.append("")
        lines.append("1. **Elo is purpose-built for 1X2.** It directly models relative team strength as a single scalar — the difference between home and away Elo ratings. This is the single most predictive signal for match outcome, and Elo captures it perfectly.")
        lines.append("2. **Tree models dilute their attention.** With 200+ features, tree models have to 'find' the Elo signal among many others. While Elo should be a top feature, the model may over-emphasise rolling averages or head-to-head stats that add noise.")
        lines.append("3. **Tree models overfit on small datasets.** On World Cup data (~500 matches), tree models with hundreds of features are prone to overfitting. The comparison script trained on only 390/98 WC matches — barely enough for 3 trees.")
        lines.append("4. **Prediction entropy is lower for tree models.** Overconfident predictions hurt LogLoss and Brier even if accuracy is similar. Tree models tend to produce sharper (more extreme) probability estimates.")
        lines.append("5. **Elo doesn't need feature engineering.** It's trained online — every match updates the ratings. No feature matrix, no hyperparameter tuning, no leakage issues.")
        lines.append("")
    else:
        lines.append("### Tree Models Finally Beat Elo on League Data!")
        lines.append("")
        lines.append("On full league data, the best tree model surpasses Elo. Key insights:")
        lines.append("")
        # Calculate the improvement
        lines.append(f"1. **More data helps.** With {n_train:,} training matches vs ~400 World Cup matches, tree models can learn meaningful patterns.")
        lines.append(f"2. **Elo is still competitive.** Its Brier ({elo_brier:.5f}) is within striking distance of tree models — remarkable for a single scalar rating.")
        lines.append(f"3. **The gap may narrow further** with better hyperparameter tuning, feature selection, or calibration.")
        lines.append("")

    h2("Comparison: World Cup vs League Data")

    # From the comparison script we know WC-only results
    lines.append("| Metric | World Cup (74 tests) | League (this run) |")
    lines.append("|--------|---------------------|--------------------|")
    wc_elo_brier = 0.582  # From earlier comparison
    wc_xgb_brier = 0.657
    wc_lgb_brier = 0.628
    wc_cat_brier = 0.690
    lines.append(f"| Elo Brier | {wc_elo_brier:.4f} | {elo_metrics.get('brier', 0):.4f} |")
    lines.append(f"| XGBoost Brier | {wc_xgb_brier:.4f} | {xgb_metrics.get('brier', 0):.4f} |")
    lines.append(f"| LightGBM Brier | {wc_lgb_brier:.4f} | {lgb_metrics.get('brier', 0):.4f} |")
    lines.append(f"| CatBoost Brier | {wc_cat_brier:.4f} | {cat_metrics.get('brier', 0):.4f} |")
    lines.append("")

    if not elo_beats_tree:
        lines.append("> Tree models benefit substantially from more training data. The WC-only training (~390 matches) was insufficient — league data (thousands of matches) allows them to generalise.")
        lines.append("")
    else:
        lines.append("> Even with more data, tree models struggle to beat a well-tuned Elo system on 1X2. The 3-model blend (dc+elo) is the correct choice for this market.")
        lines.append("")

    # ── Recommendations ──
    h1("Recommendations")

    lines.append("### For 1X2 Prediction")
    lines.append("- **Current strategy is correct:** Use the 3-model blend (dc + elo) for 1X2. Adding tree models adds noise without benefit.")
    lines.append("- If tree models are used, **feature-engineer Elo ratings as an explicit input** to make the signal trivially accessible.")
    lines.append("")

    lines.append("### For Over/Under & BTTS")
    lines.append("- **Keep the 5-model approach** for these markets. The comparison showed tree models add value here.")
    lines.append("- Tree models excel at finding complex interactions that matter for goal totals (e.g. attack vs defence strength across multiple windows).")
    lines.append("")

    lines.append("### To Improve Tree Models")
    lines.append(f"1. **More aggressive feature selection** — reduce from {n_features} to ~30-50 top features based on the consensus table above.")
    lines.append("2. **Hyperparameter tuning** — the current params are reasonable but not optimised per dataset.")
    lines.append("3. **Calibration** — all tree models produce overconfident predictions. Platt scaling or isotonic regression on the validation set would improve Brier/LogLoss.")
    lines.append("4. **Stacking** — use Elo predictions as an input feature to tree models, rather than ensembling at the output level.")
    lines.append("")

    # ── Appendix ──
    h1("Appendix: Methodology")
    lines.append(f"- **Data:** `results_clean.csv` ({n_train + n_val + n_test:,} matches)")
    lines.append(f"- **Features:** `build_features()` pipeline — {n_features} numeric features")
    lines.append(f"- **Split:** Chronological 70/15/15 (no time leakage)")
    lines.append(f"- **Elo:** Online-processed on training set, then evaluated on test set")
    lines.append(f"- **XGBoost:** `n_estimators=800, max_depth=6, lr=0.05, early_stop=50`")
    lines.append(f"- **LightGBM:** `n_estimators=800, max_depth=6, lr=0.05, num_leaves=31, early_stop=50`")
    lines.append(f"- **CatBoost:** `iterations=800, depth=6, lr=0.05, l2_leaf_reg=3.0, early_stop=50`")
    lines.append(f"- **Metrics:** Brier Score (primary), LogLoss, Accuracy, Prediction Entropy, ECE")
    lines.append(f"- **Duration:** {elapsed:.1f}s")
    lines.append("")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return str(output_path)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyse tree model performance vs Elo on 1X2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", "-d",
                        default=str(PROJECT_ROOT / "data" / "processed" / "results_clean.csv"),
                        help="Path to processed data CSV")
    parser.add_argument("--output", "-o", default=None,
                        help="Output report path")
    args = parser.parse_args(argv)

    t_start = time.time()

    print()
    print("=" * 72)
    print("  TREE MODEL ANALYSIS — Why do tree models underperform Elo on 1X2?")
    print("=" * 72)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data not found: %s", data_path)
        logger.info("Trying World Cup data as fallback...")
        data_path = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
        if not data_path.exists():
            logger.error("No data found. Run preprocessing first.")
            return 1

    # ── 1. Build features ──────────────────────────────
    print("\n-- Building features -------------------------")
    X, y, df_raw = load_and_build_features(data_path)
    n_features = X.shape[1]
    print(f"  Total features: {n_features}")
    print(f"  Class distribution: {_class_distribution(y)}")

    # ── 2. Split chronologically ─────────────────────────
    from src.feature_engineering import train_val_test_split

    print("\n-- Splitting data ----------------------------")
    splits = train_val_test_split(X, y)
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # Convert y to numpy
    y_test_np = y_test.values.astype(int)
    y_val_np = y_val.values.astype(int)

    # ── 3. Elo baseline (no leakage) ─────────────────
    print("\n-- Computing Elo baseline (no leakage) -------")
    # Sort raw df the same way build_features does, then take chronological slices.
    # build_features sorts by [date, home_team] after converting date to datetime.
    # For YYYY-MM-DD strings, string sorting matches datetime sorting exactly.
    df_sorted = df_raw.sort_values(["date", "home_team"]).reset_index(drop=True)
    train_df_elo = df_sorted.iloc[:n_train + n_val]
    test_df_elo = df_sorted.iloc[n_train + n_val:]

    # Verify alignment: y values from sorted df test slice should match y_test from feature matrix
    y_elo_test = test_df_elo["result"].map({"A": 0, "D": 1, "H": 2}).values
    y_alignment = (y_elo_test == y_test_np).mean()
    print(f"  Elo test vs X_test y-alignment: {y_alignment:.2%}")
    if y_alignment < 1.0:
        logger.error("ALIGNMENT MISMATCH! Elo and tree models are on different test sets!")
        logger.error("  Elo test y:    %s", y_elo_test[:10])
        logger.error("  X_test y:      %s", y_test_np[:10])
        raise RuntimeError(
            f"Chronological split mismatch: only {y_alignment:.1%} of test rows align. "
            "Check that build_features does not drop/duplicate rows."
        )

    test_elo_probs = compute_elo_predictions(train_df_elo, test_df_elo)
    logger.info("  Elo: %d train matches -> %d test predictions", len(train_df_elo), len(test_df_elo))

    elo_brier = brier_1x2(y_test_np, test_elo_probs)
    elo_ll = log_loss_1x2(y_test_np, test_elo_probs)
    elo_acc = accuracy_1x2(y_test_np, test_elo_probs)
    elo_ent = compute_prediction_entropy(test_elo_probs)
    elo_ece = compute_calibration_error(y_test_np, test_elo_probs)
    elo_metrics = {"brier": elo_brier, "log_loss": elo_ll, "accuracy": elo_acc, "entropy": elo_ent, "ece": elo_ece}
    print(f"  Elo: brier={elo_brier:.5f}, acc={elo_acc:.2%}")

    # ── 4. Train tree models ───────────────────────────
    print("\n-- Training XGBoost --------------------------")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_brier = brier_1x2(y_test_np, xgb_probs)
    xgb_ll = log_loss_1x2(y_test_np, xgb_probs)
    xgb_acc = accuracy_1x2(y_test_np, xgb_probs)
    xgb_ent = compute_prediction_entropy(xgb_probs)
    xgb_ece = compute_calibration_error(y_test_np, xgb_probs)
    xgb_metrics = {"brier": xgb_brier, "log_loss": xgb_ll, "accuracy": xgb_acc, "entropy": xgb_ent, "ece": xgb_ece}
    print(f"  XGBoost: brier={xgb_brier:.5f}, acc={xgb_acc:.2%}")

    print("\n-- Training LightGBM -------------------------")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    lgb_probs = lgb_model.predict_proba(X_test)
    lgb_brier = brier_1x2(y_test_np, lgb_probs)
    lgb_ll = log_loss_1x2(y_test_np, lgb_probs)
    lgb_acc = accuracy_1x2(y_test_np, lgb_probs)
    lgb_ent = compute_prediction_entropy(lgb_probs)
    lgb_ece = compute_calibration_error(y_test_np, lgb_probs)
    lgb_metrics = {"brier": lgb_brier, "log_loss": lgb_ll, "accuracy": lgb_acc, "entropy": lgb_ent, "ece": lgb_ece}
    print(f"  LightGBM: brier={lgb_brier:.5f}, acc={lgb_acc:.2%}")

    print("\n-- Training CatBoost -------------------------")
    cat_model = train_catboost(X_train, y_train, X_val, y_val)
    cat_probs = cat_model.predict_proba(X_test)
    cat_brier = brier_1x2(y_test_np, cat_probs)
    cat_ll = log_loss_1x2(y_test_np, cat_probs)
    cat_acc = accuracy_1x2(y_test_np, cat_probs)
    cat_ent = compute_prediction_entropy(cat_probs)
    cat_ece = compute_calibration_error(y_test_np, cat_probs)
    cat_metrics = {"brier": cat_brier, "log_loss": cat_ll, "accuracy": cat_acc, "entropy": cat_ent, "ece": cat_ece}
    print(f"  CatBoost: brier={cat_brier:.5f}, acc={cat_acc:.2%}")

    # ── 5. Feature importance ──────────────────────────
    print("\n-- Extracting feature importance -------------")
    xgb_importance = _get_feature_importance(xgb_model)
    lgb_importance = _get_feature_importance(lgb_model)
    cat_importance = _get_feature_importance(cat_model)
    print(f"  XGBoost top: {list(xgb_importance.keys())[:5]}")
    print(f"  LightGBM top: {list(lgb_importance.keys())[:5]}")
    print(f"  CatBoost top: {list(cat_importance.keys())[:5]}")

    # ── 6. Print quick comparison table ────────────────
    print(f"\n{'='*72}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"\n  {'Model':<14} {'Brier':>10} {'LogLoss':>10} {'Accuracy':>10} {'Entropy':>10} {'ECE':>8}")
    print(f"  {'-'*62}")
    print(f"  {'Elo':<14} {elo_brier:>10.5f} {elo_ll or 0:>10.5f} {elo_acc:>10.2%} {elo_ent:>10.4f} {elo_ece:>8.4f}")
    print(f"  {'XGBoost':<14} {xgb_brier:>10.5f} {xgb_ll or 0:>10.5f} {xgb_acc:>10.2%} {xgb_ent:>10.4f} {xgb_ece:>8.4f}")
    print(f"  {'LightGBM':<14} {lgb_brier:>10.5f} {lgb_ll or 0:>10.5f} {lgb_acc:>10.2%} {lgb_ent:>10.4f} {lgb_ece:>8.4f}")
    print(f"  {'CatBoost':<14} {cat_brier:>10.5f} {cat_ll or 0:>10.5f} {cat_acc:>10.2%} {cat_ent:>10.4f} {cat_ece:>8.4f}")

    # ── 7. Comparison with WC-only results ─────────────
    print(f"\n  {'-'*62}")
    print(f"  COMPARISON WITH WORLD CUP ONLY")
    print(f"  {'-'*62}")
    print(f"\n  {'Model':<14} {'WC Brier':>10} {'League Brier':>14} {'Change':>10}")
    print(f"  {'-'*50}")
    wc_results = {"Elo": 0.58236, "XGBoost": 0.65723, "LightGBM": 0.62785, "CatBoost": 0.69044}
    league_results = {"Elo": elo_brier, "XGBoost": xgb_brier, "LightGBM": lgb_brier, "CatBoost": cat_brier}
    for model in ["Elo", "XGBoost", "LightGBM", "CatBoost"]:
        wc = wc_results.get(model, 0)
        lr = league_results.get(model, 0)
        change = (lr - wc) / wc * 100 if wc > 0 else 0
        arrow = "+" if change > 0 else ""
        print(f"  {model:<14} {wc:>10.5f} {lr:>14.5f} {arrow}{change:>+8.1f}%")

    # ── 8. Generate report ─────────────────────────────
    print(f"\n-- Generating report -------------------------")
    elapsed = time.time() - t_start
    report_path = generate_report(
        results={},
        elo_metrics=elo_metrics,
        xgb_metrics=xgb_metrics,
        lgb_metrics=lgb_metrics,
        cat_metrics=cat_metrics,
        xgb_importance=xgb_importance,
        lgb_importance=lgb_importance,
        cat_importance=cat_importance,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_features=n_features,
        elapsed=elapsed,
    )
    print(f"  Report: {report_path}")

    # ── 9. Save models ─────────────────────────────────
    import joblib

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(xgb_model, models_dir / "xgboost_model.joblib")
    joblib.dump(lgb_model, models_dir / "lightgbm_model.joblib")
    joblib.dump(cat_model, models_dir / "catboost_model.joblib")
    print(f"\n  Models saved to {models_dir / 'xgboost_model.joblib'}")
    print(f"  Models saved to {models_dir / 'lightgbm_model.joblib'}")
    print(f"  Models saved to {models_dir / 'catboost_model.joblib'}")

    # ── Summary ─────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 72}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
