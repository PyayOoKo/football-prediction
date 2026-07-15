#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Generate Reliability Diagrams for All Models                              ║
║                                                                             ║
║  For each model (original and calibrated):                                  ║
║    - Bin predictions into 10 equal-width bins (0-0.1, ..., 0.9-1.0)        ║
║    - Calculate average prediction and actual frequency per bin              ║
║    - Plot reliability diagram (predicted vs actual)                         ║
║    - Calculate ECE (Expected Calibration Error)                             ║
║                                                                             ║
║  Outputs:                                                                   ║
║    - reports/figures/calibration_{model_name}_{variant}.png                 ║
║    - reports/calibration_diagrams_{timestamp}.json                          ║
║                                                                             ║
║  Usage:                                                                     ║
║      python scripts/reliability_diagrams.py                                  ║
║      python scripts/reliability_diagrams.py --models xgboost,lightgbm       ║
║      python scripts/reliability_diagrams.py --skip-phase3                   ║
║      python scripts/reliability_diagrams.py --skip-calibrated               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless
import matplotlib.pyplot as plt

from src.calibration import (
    CalibratedModel,
    CalibratedStatsModel,
    CalibratedTemperatureWrapper,
    calibration_curve,
    calibration_report,
)
from src.data_loader import load_results
from src.feature_engineering import build_features
from src.time_series_cv import time_series_train_val_test_split

logger = logging.getLogger(__name__)

N_CLASSES = 3
N_BINS = 10
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = ["Away Win", "Draw", "Home Win"]

# Phase 4 model stems
PHASE4_MODELS: dict[str, str] = {
    "XGBoost": "xgboost_model",
    "LightGBM": "lightgbm_model",
    "Random Forest": "random_forest_model",
    "Neural Network": "neural_network_model",
    "Logistic Regression": "logistic_regression_model",
}

# Phase 3 model filenames
PHASE3_MODELS: dict[str, str] = {
    "Poisson": "poisson_model.joblib",
    "Dixon-Coles": "dixon_coles_model.joblib",
    "Elo": "elo_model.joblib",
}


# ═══════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════


def load_data() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load data and return Phase 4 splits + Phase 3 raw data.

    Returns
    -------
    splits : dict with X_train, X_val, X_test, y_train, y_val, y_test
    df_train_p3, df_test_p3 : raw DataFrames for Phase 3 models
    """
    print("  [Data] Loading results and building features...")
    df_raw = load_results(low_memory=False)
    if "target" not in df_raw.columns and "result" in df_raw.columns:
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["date"])
    df_raw.sort_values(["date", "home_team"], inplace=True)
    df_raw.reset_index(drop=True, inplace=True)
    df_raw = df_raw[df_raw["target"] >= 0].copy()
    X, y = build_features(df_raw, is_training=True)
    splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))

    # Phase 3 raw data (aligned split)
    df_p3 = load_results(low_memory=False)
    df_p3["date"] = pd.to_datetime(df_p3["date"], errors="coerce")
    df_p3 = df_p3.dropna(subset=["date"])
    df_p3.sort_values(["date", "home_team"], inplace=True)
    df_p3.reset_index(drop=True, inplace=True)
    df_p3 = df_p3[df_p3["result"].notna() & df_p3["result"].isin(["H", "D", "A"])].copy()
    df_p3["target"] = df_p3["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    n = len(df_p3)
    train_end = int(n * 0.6)
    df_train_p3 = df_p3.iloc[:train_end].copy()
    df_test_p3 = df_p3.iloc[train_end:].copy()

    print(f"      Phase 4: train={len(splits['X_train'])}, val={len(splits['X_val'])}, test={len(splits['X_test'])}")
    print(f"      Phase 3: train={len(df_train_p3)}, test={len(df_test_p3)}")
    return splits, df_train_p3, df_test_p3


# ═══════════════════════════════════════════════════════════
#  Prediction helpers
# ═══════════════════════════════════════════════════════════


def find_model_file(stem: str) -> Path | None:
    """Find a model file by stem, trying multiple extensions."""
    candidates = [
        MODEL_DIR / stem,
        MODEL_DIR / f"{stem}.joblib",
        MODEL_DIR / f"{stem.replace('_model', '')}_tuned_model",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def predict_phase4(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Get predict_proba from a Phase 4 model.

    GBDT models (XGBoost, LightGBM) handle NaN natively.
    Other models (LR, RF, NN) require NaN imputation.
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        # First attempt: try with raw data (works for XGBoost/LightGBM)
        probs = model.predict_proba(X)
        return np.asarray(probs, dtype=np.float64)
    except Exception:
        # Second attempt: fill NaN with column mean
        try:
            X_filled = X.fillna(X.mean().fillna(0))
            probs = model.predict_proba(X_filled)
            return np.asarray(probs, dtype=np.float64)
        except Exception:
            return None


def predict_phase3(model: Any, df: pd.DataFrame) -> np.ndarray | None:
    """Get probability matrix from a Phase 3 model via predict_matches."""
    if not hasattr(model, "predict_matches"):
        return None
    try:
        preds = model.predict_matches(df)
        probs = np.column_stack([
            preds["away_win_prob"].values,
            preds["draw_prob"].values,
            preds["home_win_prob"].values,
        ])
        nan_mask = np.isnan(probs).any(axis=1)
        if nan_mask.any():
            probs[nan_mask] = 1.0 / N_CLASSES
        return np.asarray(probs, dtype=np.float64)
    except Exception:
        return None


def load_model_variants(display_name: str) -> list[dict[str, Any]]:
    """Load a model and its calibrated variant(s).

    Returns a list of dicts with keys: name, model, is_calibrated, method.
    """
    variants: list[dict[str, Any]] = []

    # Determine stem/slug
    stem = PHASE4_MODELS.get(display_name)
    is_phase4 = stem is not None

    # Load original model
    if is_phase4:
        path = find_model_file(stem)
    else:
        filename = PHASE3_MODELS.get(display_name, "")
        p = MODEL_DIR / filename
        if not p.exists():
            alt = MODEL_DIR / filename.replace(".joblib", "")
            p = alt if alt.exists() else None
        path = p

    if path is None:
        return variants

    try:
        original_model = joblib.load(path)
        variants.append({
            "name": display_name,
            "variant": "original",
            "model": original_model,
            "is_calibrated": False,
            "method": "none",
            "calibrated_brier": None,
        })
    except Exception as e:
        print(f"      [SKIP] Could not load original {display_name}: {e}")
        return variants

    # Load calibrated variant if it exists
    cal_stem = f"calibrated_{stem.replace('_model', '')}" if is_phase4 else f"calibrated_{display_name.lower().replace(' ', '_')}"
    if not is_phase4:
        filename = PHASE3_MODELS.get(display_name, "")
        cal_stem = f"calibrated_{filename.replace('_model.joblib', '').replace('.joblib', '')}"

    cal_path = MODEL_DIR / f"{cal_stem}.joblib"
    if cal_path.exists():
        try:
            cal_data = joblib.load(cal_path)
            cal_model = cal_data.get("model")
            meta = cal_data.get("metadata", {})
            variants.append({
                "name": display_name,
                "variant": f"calibrated_{meta.get('calibration_method', 'unknown')}",
                "model": cal_model,
                "is_calibrated": True,
                "method": meta.get("calibration_method", "unknown"),
                "calibrated_brier": meta.get("calibrated_brier"),
            })
        except Exception as e:
            print(f"      [SKIP] Could not load calibrated {display_name}: {e}")

    return variants


# ═══════════════════════════════════════════════════════════
#  Reliability diagram computation & plotting
# ═══════════════════════════════════════════════════════════


def compute_reliability_data(
    y_true: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    variant: str,
    n_bins: int = N_BINS,
) -> dict[str, Any]:
    """Compute reliability diagram data and ECE for a model.

    For each confidence bin (0-0.1, 0.1-0.2, ..., 0.9-1.0):
    - Average predicted confidence
    - Actual fraction positive
    - Count of samples in bin

    Also computes per-class calibration curves.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        True class labels (0, 1, 2).
    probs : np.ndarray of shape (n, 3)
        Predicted probabilities.
    model_name : str
        Display name.
    variant : str
        "original" or "calibrated_Method".

    Returns
    -------
    dict with overall and per-class calibration data.
    """
    n = len(y_true)

    # Guard against NaN in probabilities (MUST happen before any max/argmax)
    probs = np.nan_to_num(probs, nan=1.0 / N_CLASSES)

    pred_class = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)
    correct = (pred_class == y_true).astype(float)

    # ── Overall reliability ────────────────────────────
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    overall_bins: list[dict[str, Any]] = []

    for i in range(n_bins):
        # Last bin includes the upper bound (for predictions exactly = 1.0)
        if i == n_bins - 1:
            in_bin = (pred_conf >= bins[i]) & (pred_conf <= bins[i + 1])
        else:
            in_bin = (pred_conf >= bins[i]) & (pred_conf < bins[i + 1])
        count = int(in_bin.sum())
        if count > 0:
            avg_pred = float(pred_conf[in_bin].mean())
            actual = float(correct[in_bin].mean())
        else:
            avg_pred = float(bin_centers[i])
            actual = 0.0
        overall_bins.append({
            "bin_start": round(bins[i], 4),
            "bin_end": round(bins[i + 1], 4),
            "bin_center": round(bin_centers[i], 4),
            "count": count,
            "avg_prediction": round(avg_pred, 6),
            "actual_frequency": round(actual, 6),
            "gap": round(abs(avg_pred - actual), 6),
        })

    # ECE
    total = float(sum(b["count"] for b in overall_bins))
    ece = float(
        sum(b["count"] / total * b["gap"] for b in overall_bins if total > 0)
    ) if total > 0 else 0.0

    # ── Per-class reliability ─────────────────────────
    per_class: dict[str, Any] = {}
    for c in range(N_CLASSES):
        class_probs = probs[:, c]
        y_binary = (y_true == c).astype(float)
        class_bins: list[dict[str, Any]] = []
        for i in range(n_bins):
            in_bin = (class_probs >= bins[i]) & (class_probs < bins[i + 1])
            count = int(in_bin.sum())
            if count > 0:
                avg_pred = float(class_probs[in_bin].mean())
                actual = float(y_binary[in_bin].mean())
            else:
                avg_pred = float(bin_centers[i])
                actual = 0.0
            class_bins.append({
                "bin_start": round(bins[i], 4),
                "bin_end": round(bins[i + 1], 4),
                "bin_center": round(bin_centers[i], 4),
                "count": count,
                "avg_prediction": round(avg_pred, 6),
                "actual_frequency": round(actual, 6),
                "gap": round(abs(avg_pred - actual), 6),
            })
        class_total = float(sum(b["count"] for b in class_bins))
        class_ece = float(
            sum(b["count"] / class_total * b["gap"] for b in class_bins if class_total > 0)
        ) if class_total > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "bins": class_bins,
            "ece": round(class_ece, 6),
        }

    return {
        "model": model_name,
        "variant": variant,
        "n_samples": n,
        "ece": round(ece, 6),
        "bins": overall_bins,
        "per_class": per_class,
    }


def plot_reliability_diagram(
    data: dict[str, Any],
    save_path: Path,
    show_per_class: bool = True,
) -> None:
    """Generate a reliability diagram (calibration curve) plot.

    Parameters
    ----------
    data : dict
        Reliability data from ``compute_reliability_data()``.
    save_path : Path
        Path to save the PNG.
    show_per_class : bool
        Whether to include per-class subplots.
    """
    model_name = data["model"]
    variant = data["variant"]
    ece = data["ece"]
    bins_data = data["bins"]

    is_calibrated = "calibrated" in variant
    variant_label = variant.replace("calibrated_", "").title() if is_calibrated else "Original"

    if show_per_class and data.get("per_class"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax_main = axes[0, 0]
        per_class_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    else:
        fig, ax_main = plt.subplots(figsize=(8, 7))
        per_class_axes = []

    # ── Main reliability diagram ──────────────────────
    valid = [b for b in bins_data if b["count"] > 0]
    centers = [b["bin_center"] for b in valid]
    actuals = [b["actual_frequency"] for b in valid]
    counts = [b["count"] for b in bins_data]

    # Perfect calibration line
    ax_main.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.5, label="Perfect Calibration")

    # Reliability bars
    ax_main.bar(
        centers, actuals, width=0.09, alpha=0.7, color="#2ecc71" if is_calibrated else "#3498db",
        edgecolor="white", linewidth=0.8, label=f"{variant_label} (ECE={ece:.4f})",
    )

    # Gap fill (red for overconfident, blue for underconfident)
    for b in valid:
        if b["actual_frequency"] < b["avg_prediction"]:
            ax_main.plot(
                [b["bin_center"], b["bin_center"]],
                [b["actual_frequency"], b["avg_prediction"]],
                color="#e74c3c", alpha=0.3, linewidth=2,
            )
        else:
            ax_main.plot(
                [b["bin_center"], b["bin_center"]],
                [b["avg_prediction"], b["actual_frequency"]],
                color="#2ecc71", alpha=0.3, linewidth=2,
            )

    # Histogram of confidence distribution in background
    ax2 = ax_main.twinx()
    ax2.bar(
        [b["bin_center"] for b in bins_data],
        counts,
        width=0.09, alpha=0.15, color="gray",
        label="Count",
    )
    ax2.set_ylabel("Sample Count", fontsize=9, alpha=0.6)
    ax2.tick_params(axis="y", labelsize=8, colors="gray")

    # Styling
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Predicted Probability", fontsize=11)
    ax_main.set_ylabel("Actual Frequency", fontsize=11)
    ax_main.set_title(
        f"Reliability Diagram — {model_name} ({variant_label})\nECE = {ece:.4f}",
        fontsize=12, fontweight="bold",
    )
    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.spines[["top", "right"]].set_visible(False)

    # Add text annotations for key metrics
    n_total = data["n_samples"]
    n_overconf = sum(1 for b in valid if b["actual_frequency"] < b["avg_prediction"])
    n_underconf = sum(1 for b in valid if b["actual_frequency"] > b["avg_prediction"])
    info_text = (
        f"Samples: {n_total}\n"
        f"Bins: {len(valid)}/{N_BINS} populated\n"
        f"Overconfident bins: {n_overconf}\n"
        f"Underconfident bins: {n_underconf}"
    )
    ax_main.text(
        0.98, 0.05, info_text, transform=ax_main.transAxes,
        fontsize=8, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.3),
    )

    # ── Per-class calibration subplots ────────────────
    if per_class_axes and data.get("per_class"):
        colors_per_class = ["#e74c3c", "#f39c12", "#2ecc71"]
        for ax_pc, (class_name, class_data), color in zip(
            per_class_axes, data["per_class"].items(), colors_per_class
        ):
            class_bins = class_data["bins"]
            class_ece = class_data["ece"]

            valid_cb = [b for b in class_bins if b["count"] > 0]
            ax_pc.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
            ax_pc.bar(
                [b["bin_center"] for b in valid_cb],
                [b["actual_frequency"] for b in valid_cb],
                width=0.09, alpha=0.6, color=color, edgecolor="white", linewidth=0.8,
            )
            # Histogram
            ax2_pc = ax_pc.twinx()
            ax2_pc.bar(
                [b["bin_center"] for b in class_bins],
                [b["count"] for b in class_bins],
                width=0.09, alpha=0.1, color="gray",
            )
            ax2_pc.set_ylabel("Count", fontsize=7, alpha=0.5)
            ax2_pc.tick_params(axis="y", labelsize=6, colors="gray")

            ax_pc.set_xlim(0, 1)
            ax_pc.set_ylim(0, 1)
            ax_pc.set_xlabel("Predicted", fontsize=8)
            ax_pc.set_ylabel("Actual", fontsize=8)
            ax_pc.set_title(f"{class_name} (ECE={class_ece:.4f})", fontsize=9, fontweight="bold")
            ax_pc.spines[["top", "right"]].set_visible(False)
            ax_pc.tick_params(labelsize=7)

        # Hide unused subplot if any
        if len(per_class_axes) > len(data["per_class"]):
            for ax_extra in per_class_axes[len(data["per_class"]):]:
                ax_extra.set_visible(False)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════


def run_reliability_diagrams(
    model_filter: list[str] | None = None,
    skip_phase3: bool = False,
    skip_calibrated: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Generate reliability diagrams for all models."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_pretty = datetime.now().isoformat()

    print("\n" + "=" * 75)
    print("  RELIABILITY DIAGRAMS — All Models")
    print("=" * 75)

    # ── [1/3] Load data ────────────────────────────────
    print(f"\n[1/3] Loading data...")
    splits, df_train_p3, df_test_p3 = load_data()

    # ── [2/3] Process models ──────────────────────────
    print(f"\n[2/3] Computing reliability diagrams...")
    all_results: list[dict[str, Any]] = []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 4 models
    print(f"\n  ── Phase 4: ML Models ──")
    for display_name, stem in PHASE4_MODELS.items():
        if model_filter and display_name not in model_filter:
            print(f"  [SKIP] {display_name} (filtered)")
            continue

        print(f"\n  {display_name}...")
        variants = load_model_variants(display_name)
        if not variants:
            print(f"      [SKIP] No model files found")
            continue

        X_mean = splits["X_train"].mean().fillna(0)
        X_test = splits["X_test"].fillna(X_mean)
        y_test = splits["y_test"].values if hasattr(splits["y_test"], "values") else np.asarray(splits["y_test"])

        for var in variants:
            if skip_calibrated and var["is_calibrated"]:
                print(f"      [SKIP] {var['variant']} (--skip-calibrated)")
                continue

            probs = predict_phase4(var["model"], X_test)
            if probs is None:
                print(f"      [SKIP] {var['variant']}: predict_proba failed")
                continue

            rel_data = compute_reliability_data(y_test, probs, display_name, var["variant"])

            # Save plot
            fname = f"calibration_{stem.replace('_model', '')}_{var['variant']}.png"
            save_path = FIGURE_DIR / fname
            plot_reliability_diagram(rel_data, save_path)
            rel_data["plot_path"] = str(save_path.relative_to(PROJECT_ROOT))
            print(f"      {var['variant']:<30s} ECE={rel_data['ece']:.4f}  → {save_path.name}")

            all_results.append(rel_data)

    # Phase 3 models
    if not skip_phase3:
        print(f"\n  ── Phase 3: Statistical Models ──")
        for display_name, filename in PHASE3_MODELS.items():
            if model_filter and display_name not in model_filter:
                print(f"  [SKIP] {display_name} (filtered)")
                continue

            print(f"\n  {display_name}...")
            variants = load_model_variants(display_name)
            if not variants:
                print(f"      [SKIP] No model files found")
                continue

            for var in variants:
                if skip_calibrated and var["is_calibrated"]:
                    print(f"      [SKIP] {var['variant']} (--skip-calibrated)")
                    continue

                probs = predict_phase3(var["model"], df_test_p3)
                if probs is None:
                    print(f"      [SKIP] {var['variant']}: predict_matches failed")
                    continue

                y_test_p3 = df_test_p3["target"].values.astype(int)
                rel_data = compute_reliability_data(y_test_p3, probs, display_name, var["variant"])

                slug = filename.replace("_model.joblib", "").replace(".joblib", "")
                fname = f"calibration_{slug}_{var['variant']}.png"
                save_path = FIGURE_DIR / fname
                plot_reliability_diagram(rel_data, save_path)
                rel_data["plot_path"] = str(save_path.relative_to(PROJECT_ROOT))
                print(f"      {var['variant']:<30s} ECE={rel_data['ece']:.4f}  → {save_path.name}")

                all_results.append(rel_data)

    # ── [3/3] Save & print summary ────────────────────
    print(f"\n[3/3] Saving results...")

    # Build output JSON
    output = {
        "timestamp": timestamp_pretty,
        "n_bins": N_BINS,
        "results": all_results,
        "summary": {
            "total_diagrams": len(all_results),
            "models_covered": len(set(r["model"] for r in all_results)),
        },
    }

    # Best/worst ECE
    if all_results:
        best = min(all_results, key=lambda r: r["ece"])
        worst = max(all_results, key=lambda r: r["ece"])
        output["summary"]["best_calibrated_model"] = {
            "name": f"{best['model']} ({best['variant']})",
            "ece": best["ece"],
        }
        output["summary"]["worst_calibrated_model"] = {
            "name": f"{worst['model']} ({worst['variant']})",
            "ece": worst["ece"],
        }

        # ECE comparison: original vs calibrated per model
        ece_comparison: dict[str, dict[str, Any]] = {}
        for r in all_results:
            mn = r["model"]
            if mn not in ece_comparison:
                ece_comparison[mn] = {"original_ece": None, "calibrated_ece": None}
            if "calibrated" in r["variant"]:
                ece_comparison[mn]["calibrated_ece"] = r["ece"]
                ece_comparison[mn]["calibrated_method"] = r["variant"].replace("calibrated_", "")
            else:
                ece_comparison[mn]["original_ece"] = r["ece"]

        output["summary"]["ece_comparison"] = ece_comparison

        # Print summary table
        print(f"\n  {'Model':<22s} {'Variant':<25s} {'ECE':<10s}")
        print(f"  {'-' * 57}")
        for r in sorted(all_results, key=lambda x: (x["model"], x["variant"])):
            print(f"  {r['model']:<22s} {r['variant']:<25s} {r['ece']:<10.4f}")
        print(f"\n  Best ECE: {best['model']} ({best['variant']}) = {best['ece']:.4f}")
        print(f"  Worst ECE: {worst['model']} ({worst['variant']}) = {worst['ece']:.4f}")

    # Save JSON
    json_path = REPORT_DIR / f"calibration_diagrams_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data: {json_path}")

    duration = time.time() - start_time
    output["duration_seconds"] = round(duration, 2)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n{'=' * 75}")
    print(f"  RELIABILITY DIAGRAMS COMPLETE — {len(all_results)} diagrams generated")
    print(f"{'=' * 75}")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reliability diagrams for all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        default=None,
        help="Comma-separated list of models (default: all)",
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip Phase 3 statistical models",
    )
    parser.add_argument(
        "--skip-calibrated",
        action="store_true",
        help="Skip calibrated model variants (original only)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed logging",
    )
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    try:
        run_reliability_diagrams(
            model_filter=model_filter,
            skip_phase3=args.skip_phase3,
            skip_calibrated=args.skip_calibrated,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        print(f"\n[FAIL] Reliability diagram generation failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
