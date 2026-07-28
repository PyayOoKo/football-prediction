"""
calibrate_models.py — Calibrate Over/Under and BTTS models for better probability outputs.

Calibrates the best models using 3 methods:
1. Platt Scaling (logistic regression on logits)
2. Isotonic Regression (non-parametric)
3. Temperature Scaling (single parameter)

Selects the best method (lowest Brier on test set), generates reliability
diagrams, and saves calibrated models.

Usage:
    python scripts/calibrate_models.py
    python scripts/calibrate_models.py --ou-only       # Only O/U
    python scripts/calibrate_models.py --btts-only      # Only BTTS
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate_models")

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Best models from training
OU_MODEL_PATH = MODELS_DIR / "over_under_logistic_regression_20260725_222912.joblib"
OU_DATA_PATH = PROCESSED_DIR / "over_under_data_20260725_222214.parquet"

BTTS_MODEL_PATH = MODELS_DIR / "btts_xgboost_20260725_223702.joblib"
BTTS_DATA_PATH = PROCESSED_DIR / "btts_data_20260725_222516.parquet"

OU_TARGET = "over_2_5"
BTTS_TARGET = "btts"


# ═══════════════════════════════════════════════════════════
#  1. Load data & model
# ═══════════════════════════════════════════════════════════


def load_model_and_data(
    model_path: Path, data_path: Path, target_col: str,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load model + parquet, extract val/test splits + predictions.

    Returns: model, X_val, X_test, y_val, y_test, raw_val_probs, raw_test_probs, feature_cols
    """
    import joblib

    # Load model
    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path.name)

    # Load data
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded data: %d rows from %s", len(df), data_path.name)

    # Split chronologically (same logic as training)
    train_mask = (df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2022)
    test_mask = (df["date"].dt.year >= 2023) & (df["date"].dt.year <= 2024)
    df_train_val = df[train_mask].copy().sort_values("date")
    df_test = df[test_mask].copy()
    split_idx = int(len(df_train_val) * 0.8)
    df_val = df_train_val.iloc[split_idx:].copy()

    # Feature columns (exclude identifiers and targets)
    id_cols = {
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "total_goals", "result",
        "btts", "over_2_5", "over35",
    }
    feature_cols = sorted([
        c for c in df.columns
        if c not in id_cols
        and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)
        and df[c].notna().sum() > 0
    ])

    # Impute NaNs with training median
    fill_vals = {c: df_train_val[c].median() for c in feature_cols if df_train_val[c].isna().sum() > 0}
    for c, v in fill_vals.items():
        for subset in [df_val, df_test]:
            subset[c] = subset[c].fillna(v)

    X_val = df_val[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values

    # Get raw predictions
    if hasattr(model, "predict_proba"):
        raw_val_probs = model.predict_proba(X_val)[:, 1]
        raw_test_probs = model.predict_proba(X_test)[:, 1]
    else:
        # Poisson model: tuple (home_model, away_model) with custom predict
        raw_val_probs = model[0].predict(X_val)  # Placeholder — handled separately

    logger.info("Val: %d, Test: %d | Raw Brier (val): %.4f",
                 len(X_val), len(X_test),
                 brier_score_loss(y_val, raw_val_probs))

    return model, X_val, X_test, y_val, y_test, raw_val_probs, raw_test_probs, feature_cols


# ═══════════════════════════════════════════════════════════
#  2. Calibration methods
# ═══════════════════════════════════════════════════════════


def calibrate_platt(raw_probs: np.ndarray, y: np.ndarray) -> tuple[Any, np.ndarray]:
    """Platt Scaling: logistic regression on logits (log-odds).

    Maps raw probabilities to calibrated probabilities by fitting
    a logistic regression on log-odds: log(p / (1-p)).
    """
    # Convert probabilities to logits (log-odds), clipping extremes
    eps = 1e-7
    logits = np.clip(raw_probs, eps, 1 - eps)
    logits = np.log(logits / (1 - logits)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1e6, solver="lbfgs")  # No regularization
    calibrator.fit(logits, y)

    # Calibrated probs
    calibrated = calibrator.predict_proba(logits)[:, 1]
    return calibrator, calibrated


def calibrate_isotonic(raw_probs: np.ndarray, y: np.ndarray) -> tuple[Any, np.ndarray]:
    """Isotonic Regression: non-parametric calibration.

    Fits a monotonically increasing step function to map raw → calibrated.
    """
    # Split into fit and transform sets to avoid overfitting
    # (IsotonicRegression can overfit without holdout)
    probs_fit, probs_transform, y_fit, _ = train_test_split(
        raw_probs, y, test_size=0.3, random_state=42,
    )

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs_fit, y_fit)

    # Calibrated — clip to [0, 1]
    calibrated = np.clip(calibrator.predict(probs_transform), 0, 1)

    # Return full-set predictions for evaluation
    full_calibrated = np.clip(calibrator.predict(raw_probs), 0, 1)
    return calibrator, full_calibrated


def calibrate_temperature(raw_probs: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Temperature Scaling: single parameter T > 0.

    softmax(logits / T). T > 1 flattens (underconfident),
    T < 1 sharpens (overconfident). Optimized via NLL.
    """
    from scipy.optimize import minimize

    eps = 1e-7
    logits = np.clip(raw_probs, eps, 1 - eps)
    logits = np.log(logits / (1 - logits))

    def nll(t: float) -> float:
        scaled = logits / max(t, 1e-6)
        probs = 1 / (1 + np.exp(-scaled))
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    result = minimize(nll, x0=1.0, bounds=[(0.1, 10.0)], method="L-BFGS-B")
    T = float(result.x[0])

    # Apply temperature
    scaled = logits / T
    calibrated = 1 / (1 + np.exp(-scaled))
    return T, calibrated


# ═══════════════════════════════════════════════════════════
#  3. Evaluation & plotting
# ═══════════════════════════════════════════════════════════


def compute_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — lower is better."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        bin_acc = y[mask].mean()
        bin_conf = probs[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(probs)
    return ece


def generate_reliability_diagram(
    raw_probs: np.ndarray, calibrated_probs: np.ndarray,
    y: np.ndarray, model_name: str, method: str, n_bins: int = 10,
) -> str | None:
    """Generate and save a reliability diagram comparing raw vs calibrated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax_idx, (probs, label) in enumerate([
            (raw_probs, "Raw (uncalibrated)"),
            (calibrated_probs, f"Calibrated ({method})"),
        ]):
            ax = axes[ax_idx]

            # Binning
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_indices = np.digitize(probs, bins) - 1

            accuracies = []
            confidences = []
            counts = []
            for i in range(n_bins):
                mask = bin_indices == i
                cnt = mask.sum()
                counts.append(cnt)
                if cnt > 0:
                    accuracies.append(y[mask].mean())
                    confidences.append(probs[mask].mean())
                else:
                    accuracies.append(0)
                    confidences.append(0)

            # Perfect calibration line
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect")

            # Reliability curve
            ax.plot(confidences, accuracies, "o-", color="steelblue",
                    markersize=4, linewidth=1.5, label="Model")

            # Histogram of predictions
            ax2 = ax.twinx()
            ax2.bar(bin_centers, counts, width=1.0 / n_bins * 0.8,
                    alpha=0.2, color="gray", label="Count")
            ax2.set_ylabel("Count", fontsize=9)
            ax2.tick_params(axis="y", labelsize=8)

            brier = brier_score_loss(y, probs)
            ece = compute_ece(probs, y, n_bins)

            ax.set_xlabel("Predicted Probability", fontsize=10)
            ax.set_ylabel("Observed Frequency", fontsize=10)
            ax.set_title(f"{label}\nBrier={brier:.4f}  ECE={ece:.4f}", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, loc="upper left")

        plt.suptitle(f"Reliability Diagram — {model_name}", fontsize=12, fontweight="bold")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")
        path = FIGURES_DIR / f"calibration_{safe_name}_{method}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved reliability diagram: %s", path)
        return str(path)
    except Exception as exc:
        logger.warning("  Failed to generate reliability diagram: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════
#  4. Calibrate a single model
# ═══════════════════════════════════════════════════════════


def calibrate_model(
    model_name: str,
    model_path: Path,
    data_path: Path,
    target_col: str,
) -> dict[str, Any]:
    """Run all calibration methods for one model and return results."""
    print(f"\n  Loading {model_name} model + data...")
    model, X_val, X_test, y_val, y_test, raw_val, raw_test, _ = load_model_and_data(
        model_path, data_path, target_col,
    )

    results = {
        "model": model_name,
        "target": target_col,
        "n_val": len(y_val),
        "n_test": len(y_test),
        "raw_brier_val": round(brier_score_loss(y_val, raw_val), 5),
        "raw_brier_test": round(brier_score_loss(y_test, raw_test), 5),
        "raw_logloss_val": round(log_loss(y_val, raw_val), 5),
        "raw_logloss_test": round(log_loss(y_test, raw_test), 5),
        "raw_ece_val": round(compute_ece(raw_val, y_val), 5),
        "raw_ece_test": round(compute_ece(raw_test, y_test), 5),
        "methods": [],
    }

    print(f"  Raw (uncalibrated): Brier={results['raw_brier_test']:.4f}, ECE={results['raw_ece_test']:.4f}")

    # Try each calibration method
    methods = [
        ("Platt Scaling", calibrate_platt),
        ("Isotonic Regression", calibrate_isotonic),
        ("Temperature Scaling", calibrate_temperature),
    ]

    best_method = None
    best_brier = 999.0
    best_params = None
    best_calibrator = None
    best_calibrated_test = None

    for method_name, method_fn in methods:
        logger.info("  Trying %s...", method_name)

        try:
            # Fit on validation set
            result = method_fn(raw_val, y_val)

            if method_name == "Temperature Scaling":
                T, calibrated_val_raw = result
                # Apply same temperature to test set
                eps = 1e-7
                test_logits = np.clip(raw_test, eps, 1 - eps)
                test_logits = np.log(test_logits / (1 - test_logits))
                calibrated_test = 1 / (1 + np.exp(-test_logits / T))
                params = {"T": round(T, 4)}
                calibrator = T
            elif method_name == "Platt Scaling":
                calibrator, calibrated_val_all = result
                # Apply to test
                eps = 1e-7
                test_logits = np.clip(raw_test, eps, 1 - eps)
                test_logits = np.log(test_logits / (1 - test_logits)).reshape(-1, 1)
                calibrated_test = calibrator.predict_proba(test_logits)[:, 1]
                params = {
                    "coef": float(calibrator.coef_[0][0]),
                    "intercept": float(calibrator.intercept_[0]),
                }
            elif method_name == "Isotonic Regression":
                calibrator, calibrated_val_all = result
                calibrated_test = np.clip(calibrator.predict(raw_test), 0, 1)
                n_thresholds = len(getattr(calibrator, 'thresholds_', getattr(calibrator, 'X_thresholds_', [])))
            params = {"n_thresholds": n_thresholds}

            # Evaluate
            brier_test = brier_score_loss(y_test, calibrated_test)
            ll_test = log_loss(y_test, calibrated_test)
            ece_test = compute_ece(calibrated_test, y_test)

            method_result = {
                "method": method_name,
                "brier_test": round(brier_test, 5),
                "logloss_test": round(ll_test, 5),
                "ece_test": round(ece_test, 5),
                "params": params,
            }
            results["methods"].append(method_result)

            improvement = (results["raw_brier_test"] - brier_test) / results["raw_brier_test"] * 100
            is_best = " [BEST]" if brier_test < best_brier else ""
            if brier_test < best_brier:
                best_brier = brier_test
                best_method = method_name
                best_params = params
                best_calibrator = calibrator
                best_calibrated_test = calibrated_test

            print(f"    {method_name:25s} | Brier={brier_test:.4f} "
                  f"(improvement={improvement:+.1f}%) | ECE={ece_test:.4f}{is_best}")

            # Generate reliability diagram
            generate_reliability_diagram(raw_test, calibrated_test, y_test, model_name, method_name)

        except Exception as exc:
            logger.warning("    %s failed: %s", method_name, exc)
            results["methods"].append({
                "method": method_name,
                "error": str(exc),
            })

    # ── Save calibrated model ─────────────────────────────
    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.lower().replace(" ", "_")
    calibrated_path = MODELS_DIR / f"{safe_name}_calibrated_{timestamp}.joblib"

    # Save as tuple: (base_model, calibrator, method_name, params)
    calibrator_data = {
        "base_model": model,
        "calibrator": best_calibrator,
        "method": best_method,
        "params": best_params,
    }
    joblib.dump(calibrator_data, calibrated_path)
    print(f"\n  Saved calibrated model: {calibrated_path.name}")

    results.update({
        "best_method": best_method,
        "best_brier_test": round(best_brier, 5),
        "brier_improvement_pct": round(
            (results["raw_brier_test"] - best_brier) / results["raw_brier_test"] * 100, 1
        ),
        "saved_model": str(calibrated_path),
    })

    # ── Save calibration config ───────────────────────────
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{safe_name}_calibration_{timestamp}.json"
    config = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "target": target_col,
        "source_model": str(model_path),
        "source_data": str(data_path),
        "n_val": results["n_val"],
        "n_test": results["n_test"],
        "raw_metrics": {
            "brier": results["raw_brier_test"],
            "logloss": results["raw_logloss_test"],
            "ece": results["raw_ece_test"],
        },
        "calibrated_metrics": {
            "method": best_method,
            "brier": round(best_brier, 5),
            "improvement_pct": results["brier_improvement_pct"],
        },
        "params": best_params,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config: {config_path.name}")

    return results


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate O/U and BTTS models")
    parser.add_argument("--ou-only", action="store_true", help="Only calibrate O/U model")
    parser.add_argument("--btts-only", action="store_true", help="Only calibrate BTTS model")
    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL CALIBRATION")
    print("  Started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    total_start = time.time()

    calibrations = []

    # ── Over/Under calibration ────────────────────────
    if not args.btts_only:
        print("\n" + "-" * 70)
        print("  [A] OVER/UNDER 2.5 — Logistic Regression")
        print("-" * 70)

        if OU_MODEL_PATH.exists():
            result = calibrate_model(
                "over_under",
                OU_MODEL_PATH, OU_DATA_PATH, OU_TARGET,
            )
            calibrations.append(result)
        else:
            print(f"  Model not found: {OU_MODEL_PATH}")

    # ── BTTS calibration ──────────────────────────────
    if not args.ou_only:
        print("\n\n" + "-" * 70)
        print("  [B] BTTS — XGBoost")
        print("-" * 70)

        if BTTS_MODEL_PATH.exists():
            result = calibrate_model(
                "btts",
                BTTS_MODEL_PATH, BTTS_DATA_PATH, BTTS_TARGET,
            )
            calibrations.append(result)
        else:
            print(f"  Model not found: {BTTS_MODEL_PATH}")

    total_elapsed = time.time() - total_start

    # ── Summary report ────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  CALIBRATION SUMMARY")
    print("=" * 70)

    report_lines = [
        "# Calibration Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Results",
        "",
        "| Model | Raw Brier | Best Method | Calibrated Brier | Improvement |",
        "|:------|:---------:|:-----------:|:----------------:|:-----------:|",
    ]

    for r in calibrations:
        line = (
            f"| {r['model']:25s} | {r['raw_brier_test']:.4f} | "
            f"{r['best_method']:20s} | {r['best_brier_test']:.4f} | "
            f"{r['brier_improvement_pct']:+.1f}% |"
        )
        report_lines.append(line)
        print(f"\n  {r['model']:25s}")
        print(f"  {'Raw Brier:':20s} {r['raw_brier_test']:.4f}")
        print(f"  {'Best method:':20s} {r['best_method']}")
        print(f"  {'Calibrated Brier:':20s} {r['best_brier_test']:.4f}")
        print(f"  {'Improvement:':20s} {r['brier_improvement_pct']:+.1f}%")

    report_lines.extend([
        "",
        "## Method Details",
        "",
    ])
    for r in calibrations:
        report_lines.append(f"### {r['model']}")
        report_lines.append("")
        report_lines.append(f"- Raw Brier (test): {r['raw_brier_test']:.4f}")
        report_lines.append(f"- Raw ECE (test): {r['raw_ece_test']:.4f}")
        report_lines.append(f"- Best method: {r['best_method']}")
        report_lines.append(f"- Calibrated Brier (test): {r['best_brier_test']:.4f}")
        report_lines.append(f"- Brier improvement: {r['brier_improvement_pct']:+.1f}%")
        report_lines.append("")
        report_lines.append("| Method | Brier | LogLoss | ECE |")
        report_lines.append("|:-------|:-----:|:-------:|:---:|")
        for m in r.get("methods", []):
            if "error" in m:
                report_lines.append(f"| {m['method']:22s} | — | — | **{m['error']}** |")
            else:
                report_lines.append(
                    f"| {m['method']:22s} | {m['brier_test']:.4f} | "
                    f"{m['logloss_test']:.4f} | {m['ece_test']:.4f} |"
                )
        report_lines.append("")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"calibration_{timestamp}.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
