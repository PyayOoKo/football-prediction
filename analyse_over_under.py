"""
analyse_over_under.py — Deep dive into Over/Under market performance.

Compares Poisson-only, XGBoost-only, 3-model blend, and current ensemble
on Over2.5 and Over3.5 markets. Includes team strength analysis and error
distribution visualizations.

Usage:
    python analyse_over_under.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Only try to import matplotlib if available ───────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    logger.warning("matplotlib not available — skipping charts")


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════

def brier_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    return round(float(np.mean((probs[valid] - y_true[valid]) ** 2)), 5)


def log_loss_binary(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import log_loss as sk_ll
        valid = ~np.isnan(y_true)
        p_v = np.clip(probs[valid], 1e-15, 1 - 1e-15)
        y_v = y_true[valid]
        return round(float(sk_ll(y_v, np.column_stack([1 - p_v, p_v]))), 5)
    except Exception:
        return None


def accuracy_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    preds = (probs[valid] > 0.5).astype(float)
    return round(float(np.mean(preds == y_true[valid])), 5)


# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════

def load_data(test_split: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    from src.models.three_model_blend import ConditionalRates

    data_path = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
    df = pd.read_csv(data_path, low_memory=False).sort_values("date").reset_index(drop=True)

    n = len(df)
    val_split = int(n * (1 - 2 * test_split))
    test_split_idx = int(n * (1 - test_split))

    train_df = df.iloc[:val_split].copy()
    val_df = df.iloc[val_split:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()
    cond_rates = ConditionalRates.from_data(train_df)

    logger.info("Data: %d train + %d val + %d test", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df, cond_rates


# ═══════════════════════════════════════════════════════════
#  Model Setup
# ═══════════════════════════════════════════════════════════

def setup_models(train_df: pd.DataFrame, val_df: pd.DataFrame,
                 cond_rates: Any) -> dict[str, Any]:
    """Fit/setup all models and return predictions for analysis."""
    import joblib
    from src.poisson_model import PoissonModel
    from src.elo import EloSystem
    from src.models.three_model_blend import ThreeModelBlend, _FeatureBuilder

    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    # Poisson
    logger.info("Fitting Poisson...")
    poisson = PoissonModel(min_matches=0)
    poisson.fit(fit_df)

    # Elo
    logger.info("Processing Elo...")
    elo = EloSystem()
    elo.process_matches(fit_df)

    # XGBoost
    xgb_path = PROJECT_ROOT / "models" / "xgboost_model.joblib"
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None
    if xgb is None:
        raise RuntimeError("XGBoost model not found")

    # Optimised weights
    weights_path = PROJECT_ROOT / "config" / "three_model_weights.json"
    weights = None
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f).get("weights")

    # ThreeModelBlend
    blend = ThreeModelBlend(
        poisson_model=poisson,
        elo_model=elo,
        xgb_model=xgb,
        weights=weights,
        conditional_rates=cond_rates,
        historical_df=fit_df,
    )

    # Ensemble
    ensemble_path = PROJECT_ROOT / "models" / "ensemble_model.joblib"
    payload = joblib.load(ensemble_path) if ensemble_path.exists() else None
    if payload and isinstance(payload, dict):
        from src.ensemble import WeightedEnsemble
        models_dict = payload.get("models", {})
        wts = payload.get("weights", {})
        pw = payload.get("poisson_model")
        tuples = [(m, wts.get(n, 1.0)) for n, m in models_dict.items()]
        if pw:
            tuples.append((pw, wts.get("poisson", 0.0)))
        ensemble = WeightedEnsemble(tuples, name="Ensemble")
    else:
        ensemble = payload

    # Feature builder for XGBoost batch
    fb = _FeatureBuilder(fit_df)

    return {
        "poisson": poisson,
        "elo": elo,
        "xgb": xgb,
        "blend": blend,
        "ensemble": ensemble,
        "fb": fb,
        "cond_rates": cond_rates,
    }


# ═══════════════════════════════════════════════════════════
#  Per-Model O/U Predictions
# ═══════════════════════════════════════════════════════════

def compute_ou_predictions(
    test_df: pd.DataFrame,
    models: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute Over/Under predictions for each model at both thresholds.

    Returns nested dict: {threshold: {model_name: np.ndarray}}
    """
    from src.feature_engineering import build_features

    # Shared predictions
    ppm = models["blend"].precompute(test_df, cache_key="ou_analysis")

    results: dict[str, dict[str, np.ndarray]] = {
        "Over2.5": {},
        "Over3.5": {},
    }

    # Poisson (from scoreline table — exact)
    results["Over2.5"]["Poisson"] = ppm.pois_over_25
    results["Over3.5"]["Poisson"] = ppm.pois_over_35

    # XGBoost (from Poisson CDF via expected total goals)
    fb = models["fb"]
    cr = models["cond_rates"]
    n = len(test_df)
    home_teams = test_df["home_team"].tolist()
    away_teams = test_df["away_team"].tolist()

    # XGBoost 1X2 probs
    xgb_1x2_list = []
    try:
        X = fb.build(home_teams, away_teams)
        if X is not None and len(X) > 0:
            xgb_raw = models["xgb"].predict_proba(X)
            xgb_1x2_list = [xgb_raw[i] for i in range(len(X))]
        else:
            xgb_1x2_list = [np.array([0.33, 0.34, 0.33])] * n
    except Exception as exc:
        logger.warning("XGBoost batch failed: %s", exc)
        xgb_1x2_list = [np.array([0.33, 0.34, 0.33])] * n

    xgb_1x2 = np.array(xgb_1x2_list)

    # Compute expected total goals → Poisson CDF → P(Over)
    for label, threshold in [("Over2.5", 2.5), ("Over3.5", 3.5)]:
        from src.models.three_model_blend import _poisson_cdf
        xgb_over = np.zeros(n)
        for i in range(n):
            exp_total = (
                xgb_1x2[i, 2] * cr.mean_total_given_home_win
                + xgb_1x2[i, 1] * cr.mean_total_given_draw
                + xgb_1x2[i, 0] * cr.mean_total_given_away_win
            )
            xgb_over[i] = 1.0 - _poisson_cdf(threshold, exp_total) if exp_total > 0 else 0.5
        results[label]["XGBoost"] = xgb_over

    # 3-Model Blend
    w_ou25 = models["blend"].weights.get("Over2.5", {})
    w_ou35 = models["blend"].weights.get("Over3.5", {})
    results["Over2.5"]["3-Model Blend"] = models["blend"]._blend_binary(ppm, w_ou25, "Over2.5")
    results["Over3.5"]["3-Model Blend"] = models["blend"]._blend_binary(ppm, w_ou35, "Over3.5")

    # Current Ensemble (from 1X2 → conditional rates)
    try:
        X_ens, _ = build_features(test_df, is_training=False)
        ens_1x2 = models["ensemble"].predict_proba(X_ens, df_raw=test_df)
    except Exception:
        ens_1x2 = np.full((n, 3), 1.0 / 3.0)

    results["Over2.5"]["Ensemble"] = cr.ou_from_1x2(ens_1x2, 2.5)
    results["Over3.5"]["Ensemble"] = cr.ou_from_1x2(ens_1x2, 3.5)

    # Elo-only (for reference)
    elo_1x2 = ppm.elo_1x2
    results["Over2.5"]["Elo"] = cr.ou_from_1x2(elo_1x2, 2.5)
    results["Over3.5"]["Elo"] = cr.ou_from_1x2(elo_1x2, 3.5)

    return results


# ═══════════════════════════════════════════════════════════
#  Team Strength Analysis
# ═══════════════════════════════════════════════════════════

def compute_team_strength_bins(test_df: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    """Bin test matches by home/away team strength from Elo ratings."""
    elo = models["elo"]
    bins_result = []

    for _, row in test_df.iterrows():
        ht, at = row["home_team"], row["away_team"]

        home_elo = elo.get_rating(ht) if hasattr(elo, "get_rating") else 1500
        away_elo = elo.get_rating(at) if hasattr(elo, "get_rating") else 1500
        try:
            home_elo = float(home_elo)
            away_elo = float(away_elo)
        except (TypeError, ValueError):
            home_elo, away_elo = 1500, 1500

        # Bin by combined strength
        avg_elo = (home_elo + away_elo) / 2
        if avg_elo >= 1800:
            strength_bin = "Elite (1800+)"
        elif avg_elo >= 1650:
            strength_bin = "Strong (1650-1800)"
        elif avg_elo >= 1500:
            strength_bin = "Average (1500-1650)"
        else:
            strength_bin = "Weak (<1500)"

        elo_diff = home_elo - away_elo
        if elo_diff > 100:
            matchup = "Big favourite"
        elif elo_diff > 30:
            matchup = "Slight favourite"
        elif elo_diff > -30:
            matchup = "Even match"
        elif elo_diff > -100:
            matchup = "Slight underdog"
        else:
            matchup = "Big underdog"

        bins_result.append({
            "home_team": ht,
            "away_team": at,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "avg_elo": avg_elo,
            "strength_bin": strength_bin,
            "matchup": matchup,
            "total_goals": float(row.get("home_goals", 0)) + float(row.get("away_goals", 0)),
            "over_2_5": float(row.get("home_goals", 0)) + float(row.get("away_goals", 0)) > 2.5,
            "over_3_5": float(row.get("home_goals", 0)) + float(row.get("away_goals", 0)) > 3.5,
        })

    return pd.DataFrame(bins_result)


# ═══════════════════════════════════════════════════════════
#  Visualizations
# ═══════════════════════════════════════════════════════════

def generate_charts(
    results: dict[str, dict[str, np.ndarray]],
    test_df: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    """Generate performance comparison and error distribution charts.

    Returns list of saved chart filenames.
    """
    if not MATPLOTLIB_OK:
        return []

    saved = []
    hg = test_df["home_goals"].values.astype(float)
    ag = test_df["away_goals"].values.astype(float)
    actual_ou25 = ((hg + ag) > 2.5).astype(float)
    actual_ou35 = ((hg + ag) > 3.5).astype(float)
    actuals = {"Over2.5": actual_ou25, "Over3.5": actual_ou35}

    model_colors = {
        "Poisson": "#2E86AB",
        "XGBoost": "#A23B72",
        "3-Model Blend": "#F18F01",
        "Ensemble": "#C73E1D",
        "Elo": "#6A994E",
    }

    chart_style = {
        "figsize": (10, 6),
        "dpi": 120,
        "fontsize": 11,
    }

    for threshold_label in ["Over2.5", "Over3.5"]:
        y_true = actuals[threshold_label]
        model_names = list(results[threshold_label].keys())

        # ── 1. Performance comparison (Brier bar chart) ────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Bar chart: Brier score (lower is better)
        briers = []
        names_bar = []
        colors_bar = []
        for mn in model_names:
            p = results[threshold_label][mn]
            br = brier_binary(y_true, p)
            briers.append(br)
            names_bar.append(mn)
            colors_bar.append(model_colors.get(mn, "#888888"))

        bars = ax1.barh(range(len(names_bar)), briers, color=colors_bar, edgecolor="white", height=0.6)
        ax1.set_yticks(range(len(names_bar)))
        ax1.set_yticklabels(names_bar, fontsize=chart_style["fontsize"])
        ax1.set_xlabel("Brier Score (lower is better)", fontsize=chart_style["fontsize"])
        ax1.set_title(f"{threshold_label} — Brier Score by Model", fontsize=13, fontweight="bold")
        ax1.invert_yaxis()
        ax1.tick_params(labelsize=10)
        for bar, val in zip(bars, briers):
            ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

        # Bar chart: Accuracy
        accs = []
        for mn in model_names:
            p = results[threshold_label][mn]
            acc = accuracy_binary(y_true, p)
            accs.append(acc)

        bars2 = ax2.barh(range(len(names_bar)), accs, color=colors_bar, edgecolor="white", height=0.6)
        ax2.set_yticks(range(len(names_bar)))
        ax2.set_yticklabels(names_bar, fontsize=chart_style["fontsize"])
        ax2.set_xlabel("Accuracy (higher is better)", fontsize=chart_style["fontsize"])
        ax2.set_title(f"{threshold_label} — Accuracy by Model", fontsize=13, fontweight="bold")
        ax2.invert_yaxis()
        ax2.tick_params(labelsize=10)
        for bar, val in zip(bars2, accs):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", fontsize=9)

        base_rate = y_true.mean()
        ax2.axvline(x=base_rate, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax2.text(base_rate + 0.01, len(names_bar) - 0.5, f"Base rate: {base_rate:.1%}",
                 fontsize=9, color="gray", va="center")

        plt.tight_layout()
        chart_path = output_dir / f"ou_{threshold_label.lower()}_comparison.png"
        fig.savefig(chart_path, dpi=chart_style["dpi"], bbox_inches="tight")
        plt.close(fig)
        saved.append(str(chart_path.name))
        logger.info("Saved chart: %s", chart_path.name)

        # ── 2. Error distribution ─────────────────────────
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for idx, mn in enumerate(["Poisson", "XGBoost", "3-Model Blend"]):
            if mn not in results[threshold_label]:
                continue
            p = results[threshold_label][mn]
            errors = p - y_true
            ax = axes[idx]
            ax.hist(errors, bins=20, color=model_colors.get(mn, "#888888"),
                    edgecolor="white", alpha=0.8)
            ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
            ax.set_title(f"{mn}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Prediction Error (predicted - actual)", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.tick_params(labelsize=8)
            # Add mean error and std dev annotations
            mean_err = errors.mean()
            std_err = errors.std()
            ax.axvline(x=mean_err, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(0.05, 0.95, f"Mean err: {mean_err:.3f}", transform=ax.transAxes,
                    fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.text(0.05, 0.85, f"Std:  {std_err:.3f}", transform=ax.transAxes,
                    fontsize=8, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        err_path = output_dir / f"ou_{threshold_label.lower()}_errors.png"
        fig2.savefig(err_path, dpi=chart_style["dpi"], bbox_inches="tight")
        plt.close(fig2)
        saved.append(str(err_path.name))
        logger.info("Saved chart: %s", err_path.name)

    # ── 3. Calibration curve ──────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, threshold_label in enumerate(["Over2.5", "Over3.5"]):
        ax = axes3[idx]
        y_true = actuals[threshold_label]

        for mn in ["Poisson", "XGBoost", "3-Model Blend", "Ensemble"]:
            if mn not in results[threshold_label]:
                continue
            p = results[threshold_label][mn]
            # Bin by decile
            bins = np.linspace(0, 1, 11)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            observed = []
            predicted = []
            for i in range(len(bins) - 1):
                mask = (p >= bins[i]) & (p < bins[i + 1])
                if mask.sum() > 0:
                    predicted.append(bin_centers[i])
                    observed.append(y_true[mask].mean())
            if len(predicted) > 1:
                ax.plot(predicted, observed, "o-", label=mn, color=model_colors.get(mn),
                        markersize=5, linewidth=1.5, alpha=0.8)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Perfect")
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Frequency", fontsize=10)
        ax.set_title(f"{threshold_label} — Calibration", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=9)
        ax.set_aspect("equal")

    plt.tight_layout()
    cal_path = output_dir / "ou_calibration.png"
    fig3.savefig(cal_path, dpi=chart_style["dpi"], bbox_inches="tight")
    plt.close(fig3)
    saved.append(str(cal_path.name))
    logger.info("Saved chart: %s", cal_path.name)

    return saved


# ═══════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════

def generate_report(
    results: dict[str, dict[str, np.ndarray]],
    test_df: pd.DataFrame,
    strength_df: pd.DataFrame,
    charts: list[str],
    models: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generate markdown report with full analysis."""
    hg = test_df["home_goals"].values.astype(float)
    ag = test_df["away_goals"].values.astype(float)
    actuals = {
        "Over2.5": ((hg + ag) > 2.5).astype(float),
        "Over3.5": ((hg + ag) > 3.5).astype(float),
    }

    lines: list[str] = []
    lines.append("# Over/Under Market — Deep Performance Analysis")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Test set:** {len(test_df)} matches "
                 f"({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")
    lines.append("")

    # ── 1. Overall Performance Table ──
    lines.append("## 1. Overall Performance by Model")
    lines.append("")
    lines.append("| Threshold | Model | Brier Score | Log Loss | Accuracy | n |")
    lines.append("|-----------|-------|-------------|----------|----------|---|")

    for threshold_label in ["Over2.5", "Over3.5"]:
        y_true = actuals[threshold_label]
        model_order = ["Poisson", "XGBoost", "3-Model Blend", "Ensemble", "Elo"]
        best_brier = float("inf")
        best_model = ""
        for mn in model_order:
            if mn not in results[threshold_label]:
                continue
            p = results[threshold_label][mn]
            br = brier_binary(y_true, p)
            ll = log_loss_binary(y_true, p) or 0
            acc = accuracy_binary(y_true, p)
            marker = " **(best)**" if br < best_brier else ""
            if br < best_brier:
                best_brier = br
                best_model = mn
            lines.append(f"| {threshold_label} | {mn}{marker} | {br:.5f} | {ll:.5f} | {acc:.2%} | {len(y_true)} |")
        lines.append("")

    # ── 2. Best model summary ──
    lines.append("## 2. Best Model Analysis")
    lines.append("")
    for threshold_label in ["Over2.5", "Over3.5"]:
        y_true = actuals[threshold_label]
        best_br, best_mn = float("inf"), ""
        for mn in ["Poisson", "XGBoost", "3-Model Blend", "Ensemble"]:
            if mn not in results[threshold_label]:
                continue
            br = brier_binary(y_true, results[threshold_label][mn])
            if br < best_br:
                best_br = br
                best_mn = mn

        # Delta vs worst
        worst_br = float("-inf")
        worst_mn = ""
        for mn in ["Poisson", "XGBoost", "3-Model Blend", "Ensemble"]:
            if mn not in results[threshold_label]:
                continue
            br = brier_binary(y_true, results[threshold_label][mn])
            if br > worst_br:
                worst_br = br
                worst_mn = mn

        improvement = (worst_br - best_br) / worst_br * 100 if worst_br > 0 else 0
        lines.append(f"- **{threshold_label}**: **{best_mn}** is best (Brier={best_br:.5f}). "
                     f"{improvement:.1f}% improvement over worst ({worst_mn}, Brier={worst_br:.5f}).")

        # Pairwise deltas
        lines.append(f"  - Pairwise vs Ensemble:")
        for mn in ["Poisson", "XGBoost", "3-Model Blend"]:
            if mn not in results[threshold_label]:
                continue
            br_mn = brier_binary(y_true, results[threshold_label][mn])
            ens_br = brier_binary(y_true, results[threshold_label]["Ensemble"])
            delta = (br_mn - ens_br) / ens_br * 100
            direction = "better" if br_mn < ens_br else "worse"
            lines.append(f"    - {mn}: {abs(delta):.1f}% {direction} than Ensemble "
                         f"(Brier: {br_mn:.5f} vs {ens_br:.5f})")

    lines.append("")

    # ── 3. Team Strength Analysis ──
    lines.append("## 3. Performance by Team Strength")
    lines.append("")
    if len(strength_df) > 0:
        for threshold_label in ["Over2.5", "Over3.5"]:
            lines.append(f"### {threshold_label}")
            lines.append("")
            lines.append("| Strength Bin | n | Base Rate | Poisson | XGBoost | 3-Blend | Ensemble |")
            lines.append("|-------------|---|-----------|---------|---------|---------|---------|")
            col = "over_2_5" if threshold_label == "Over2.5" else "over_3_5"

            for bin_name in ["Elite (1800+)", "Strong (1650-1800)", "Average (1500-1650)", "Weak (<1500)"]:
                sub = strength_df[strength_df["strength_bin"] == bin_name]
                if len(sub) == 0:
                    continue
                y_true_bin = sub[col].values.astype(float)
                base_rate = y_true_bin.mean()
                cells = [bin_name, str(len(sub)), f"{base_rate:.1%}"]

                for mn in ["Poisson", "XGBoost", "3-Model Blend", "Ensemble"]:
                    if mn in results.get(threshold_label, {}):
                        # Need to align predictions with subset rows
                        preds_all = results[threshold_label][mn]
                        # Subset by index (crude but effective if sorted)
                        idxs = sub.index.values
                        preds_sub = preds_all[idxs] if len(preds_all) >= max(idxs) + 1 else preds_all[:len(sub)]
                        br = brier_binary(y_true_bin, preds_sub[:len(y_true_bin)])
                        cells.append(f"{br:.4f}")
                    else:
                        cells.append("N/A")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    # ── 4. Matchup Analysis ──
    if len(strength_df) > 0:
        lines.append("## 4. Performance by Matchup Type")
        lines.append("")
        for threshold_label in ["Over2.5", "Over3.5"]:
            lines.append(f"### {threshold_label}")
            lines.append("")
            lines.append("| Matchup | n | Base Rate | Poisson | XGBoost | 3-Blend | Ensemble |")
            lines.append("|---------|---|-----------|---------|---------|---------|---------|")
            col = "over_2_5" if threshold_label == "Over2.5" else "over_3_5"

            for mtype in ["Big favourite", "Slight favourite", "Even match", "Slight underdog", "Big underdog"]:
                sub = strength_df[strength_df["matchup"] == mtype]
                if len(sub) == 0:
                    continue
                y_true_bin = sub[col].values.astype(float)
                base_rate = y_true_bin.mean()
                cells = [mtype, str(len(sub)), f"{base_rate:.1%}"]

                for mn in ["Poisson", "XGBoost", "3-Model Blend", "Ensemble"]:
                    if mn in results.get(threshold_label, {}):
                        preds_all = results[threshold_label][mn]
                        idxs = sub.index.values
                        preds_sub = preds_all[idxs] if len(preds_all) >= max(idxs) + 1 else preds_all[:len(sub)]
                        br = brier_binary(y_true_bin, preds_sub[:len(y_true_bin)])
                        cells.append(f"{br:.4f}")
                    else:
                        cells.append("N/A")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    # ── 5. Calibration ──
    lines.append("## 5. Calibration Analysis")
    lines.append("")
    for threshold_label in ["Over2.5", "Over3.5"]:
        y_true = actuals[threshold_label]
        lines.append(f"**{threshold_label}:**")
        for mn in ["Poisson", "XGBoost", "3-Model Blend", "Ensemble"]:
            if mn not in results[threshold_label]:
                continue
            p = results[threshold_label][mn]
            ll = log_loss_binary(y_true, p)
            # ECE estimate: mean absolute calibration error
            bins = np.linspace(0, 1, 11)
            ece = 0.0
            n_total = len(p)
            for i in range(len(bins) - 1):
                mask = (p >= bins[i]) & (p < bins[i + 1])
                if mask.sum() > 0:
                    ece += abs(y_true[mask].mean() - ((bins[i] + bins[i + 1]) / 2)) * (mask.sum() / n_total)
            lines.append(f"- {mn}: Log Loss={ll:.4f}, ECE={ece:.4f}")
        lines.append("")

    # ── 6. Conclusion ──
    lines.append("## 6. Conclusions")
    lines.append("")

    for threshold_label in ["Over2.5", "Over3.5"]:
        y_true = actuals[threshold_label]
        # Best model
        best_br, best_mn = float("inf"), ""
        for mn in ["Poisson", "XGBoost", "3-Model Blend", "Ensemble"]:
            if mn not in results[threshold_label]:
                continue
            br = brier_binary(y_true, results[threshold_label][mn])
            if br < best_br:
                best_br = br
                best_mn = mn

        # Base rate
        base_rate = y_true.mean()
        lines.append(f"**{threshold_label}:** Best model is **{best_mn}** with Brier={best_br:.4f}")
        lines.append(f"- Base rate: {base_rate:.1%} ({int(base_rate * len(y_true))}/{len(y_true)} matches)")
        lines.append(f"- {best_mn} Brier of {best_br:.4f}")

        # Check if blend helps
        blend_br = brier_binary(y_true, results[threshold_label].get("3-Model Blend", np.array([0.5])))
        pois_br = brier_binary(y_true, results[threshold_label].get("Poisson", np.array([0.5])))
        xgb_br = brier_binary(y_true, results[threshold_label].get("XGBoost", np.array([0.5])))

        if blend_br <= min(pois_br, xgb_br):
            lines.append("- **Blend improves upon individual models.**")
        else:
            if pois_br < xgb_br:
                lines.append(f"- Poisson alone ({pois_br:.4f}) outperforms blend ({blend_br:.4f}). "
                             "Consider increasing Poisson weight for this market.")
            else:
                lines.append(f"- XGBoost alone ({xgb_br:.4f}) outperforms blend ({blend_br:.4f}). "
                             "Consider increasing XGBoost weight for this market.")
        lines.append("")

    # ── 7. League Analysis Note ──
    lines.append("## 7. League Analysis")
    lines.append("")
    lines.append("League-specific performance analysis is not applicable because the dataset "
                 "contains only World Cup ('WC') matches. Performance may differ for league "
                 "competitions (Premier League, La Liga, etc.) where:")
    lines.append("- Home advantage is stronger (neutral venues in World Cup)")
    lines.append("- Team strength distributions are wider")
    lines.append("- Match frequency is higher (weekly vs tournament format)")
    lines.append("- Relegation/promotion dynamics affect motivation")
    lines.append("")
    lines.append("To extend this analysis to leagues, train the ThreeModelBlend on "
                 "league data and re-run this script.")
    lines.append("")

    # ── 8. Visualizations ──
    if charts:
        lines.append("## 8. Visualizations")
        lines.append("")
        for chart in charts:
            chart_path = output_dir / chart
            if chart_path.exists():
                # Use relative path for markdown
                rel = chart_path.relative_to(PROJECT_ROOT)
                lines.append(f"![{chart}]({rel})")
                lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / f"over_under_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    t_start = time.time()

    print()
    print("-" * 60)
    print("  OVER/UNDER PERFORMANCE ANALYSIS")
    print("-" * 60)

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────
    print("\n-- Loading data ---------------------------------")
    train_df, val_df, test_df, cond_rates = load_data()

    # ── 2. Setup models ──────────────────────────────────
    print("\n-- Setting up models ---------------------------")
    models = setup_models(train_df, val_df, cond_rates)

    # ── 3. Compute O/U predictions ───────────────────────
    print("\n-- Computing predictions -----------------------")
    results = compute_ou_predictions(test_df, models)
    print(f"  Models evaluated: {list(results['Over2.5'].keys())}")

    # ── 4. Team strength analysis ────────────────────────
    print("\n-- Team strength analysis ----------------------")
    strength_df = compute_team_strength_bins(test_df, models)
    print(f"  Strength bins: {strength_df['strength_bin'].value_counts().to_dict()}")

    # ── 5. Generate visualizations ──────────────────────
    print("\n-- Generating visualizations -------------------")
    charts = generate_charts(results, test_df, output_dir)
    if charts:
        print(f"  Saved {len(charts)} charts")
    else:
        print("  matplotlib not available — skipping charts")

    # ── 6. Generate report ──────────────────────────────
    print("\n-- Generating report ---------------------------")
    report_path = generate_report(results, test_df, strength_df, charts, models, output_dir)
    print(f"  Report saved: {report_path.name}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
