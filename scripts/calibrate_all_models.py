#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Calibrate All Phase 3 & Phase 4 Models                                    ║
║                                                                             ║
║  Loads all 7 models, applies 3 calibration methods (Platt, Isotonic,       ║
║  Temperature Scaling), evaluates Brier Score before/after, and saves       ║
║  results to reports/calibration_results_{timestamp}.json.                   ║
║                                                                             ║
║  Models:                                                                    ║
║    Phase 4 (ML):       XGBoost, LightGBM, Random Forest,                   ║
║                        Neural Network, Logistic Regression                 ║
║    Phase 3 (Stats):    Poisson, Dixon-Coles, Elo                           ║
║                                                                             ║
║  Calibration methods:                                                       ║
║    - PlattScalingCalibrator    (sigmoid / logistic regression)             ║
║    - IsotonicRegressionCalibrator  (non-parametric monotonic)             ║
║    - TemperatureScalingCalibrator  (single-parameter for logits)          ║
║                                                                             ║
║  Usage:                                                                     ║
║      python scripts/calibrate_all_models.py                                  ║
║      python scripts/calibrate_all_models.py --quiet                         ║
║      python scripts/calibrate_all_models.py --skip-phase3                  ║
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

from sklearn.metrics import log_loss

from config import config
from src.calibration import (
    IsotonicRegressionCalibrator,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
)
from src.data_loader import load_results
from src.feature_engineering import build_features
from src.time_series_cv import time_series_train_val_test_split
from src.poisson_model import PoissonModel
from src.dixon_coles import DixonColesModel
from src.elo import EloSystem

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

N_CLASSES = 3  # Away Win (0), Draw (1), Home Win (2)

PHASE4_MODELS: dict[str, str] = {
    "XGBoost": "xgboost_model",
    "LightGBM": "lightgbm_model",
    "Random Forest": "random_forest_model",
    "Neural Network": "neural_network_model",
    "Logistic Regression": "logistic_regression_model",
}

PHASE3_MODELS: dict[str, str] = {
    "Poisson": "poisson_model.joblib",
    "Dixon-Coles": "dixon_coles_model.joblib",
    "Elo": "elo_model.joblib",
}

CALIBRATION_METHODS = [
    "platt",
    "isotonic",
]

REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════════
#  Brier Score helper
# ═══════════════════════════════════════════════════════════


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the multi-class Brier score.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,)
        True class labels (0, 1, 2).
    y_prob : np.ndarray of shape (n, 3)
        Predicted probabilities.

    Returns
    -------
    float
        Brier score (0 = perfect, higher = worse).
    """
    y_onehot = np.zeros((len(y_true), y_prob.shape[1]))
    for i, v in enumerate(y_true):
        if 0 <= int(v) < y_prob.shape[1]:
            y_onehot[i, int(v)] = 1
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


# ═══════════════════════════════════════════════════════════
#  Data loading — Phase 4 uses feature matrix, Phase 3 uses raw
# ═══════════════════════════════════════════════════════════


def load_phase4_data() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load data, build features, split for Phase 4 ML models."""
    print("\n  [Phase 4] Building feature matrix...")
    df_raw = load_results(low_memory=False)

    # Add target column from result if missing
    if "target" not in df_raw.columns and "result" in df_raw.columns:
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)

    # Parse dates
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["date"])
    df_raw.sort_values(["date", "home_team"], inplace=True)
    df_raw.reset_index(drop=True, inplace=True)

    # Filter out rows with unknown target (-1)
    df_raw = df_raw[df_raw["target"] >= 0].copy()

    t0 = time.time()
    X, y = build_features(df_raw, is_training=True)
    elapsed = time.time() - t0
    print(f"    Feature matrix: {X.shape} ({elapsed:.1f}s)")

    splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))
    for k in ("X_train", "X_val", "X_test"):
        print(f"    {k}: {splits[k].shape[0]} rows")

    return X, y, splits


def load_phase3_data(df_fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load raw match data for Phase 3 statistical models."""
    print("\n  [Phase 3] Loading raw match data...")
    if df_fallback is not None:
        df = df_fallback.copy()
    else:
        df = load_results(low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df["result"].notna() & df["result"].isin(["H", "D", "A"])].copy()

    # Add target
    df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df = df[df["target"] >= 0].copy()

    print(f"    {len(df)} matches loaded")
    return df


def phase3_chronological_split(
    df_raw: pd.DataFrame,
    splits: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align Phase 3 split with Phase 4 using row-count based cutoffs."""
    n_raw = len(df_raw)
    n_test = len(splits["X_test"])
    n_val = len(splits["X_val"])
    n_train_val = n_raw - n_test

    df_train = df_raw.iloc[:n_train_val].copy()
    df_test = df_raw.iloc[n_train_val:].copy()

    # Further split train into train + val
    n_train = len(splits["X_train"])
    df_val = df_train.iloc[n_train:].copy()
    df_train = df_train.iloc[:n_train].copy()

    print(f"    Phase 3 train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    print(f"    Train period:  {df_train['date'].min().date()} to {df_train['date'].max().date()}")
    print(f"    Val period:    {df_val['date'].min().date()} to {df_val['date'].max().date()}")
    print(f"    Test period:   {df_test['date'].min().date()} to {df_test['date'].max().date()}")

    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════
#  Load Phase 4 ML models
# ═══════════════════════════════════════════════════════════


def load_phase4_model(display_name: str, stem: str) -> Any | None:
    """Load a Phase 4 ML model from the models/ directory.

    Tries multiple filename variants:
      - {stem}              (extensionless, e.g. xgboost_model)
      - {stem}.joblib       (with .joblib extension)
      - {stem.replace('_model', '')}_tuned_model  (tuned variant)
    """
    candidates = [
        MODEL_DIR / stem,
        MODEL_DIR / f"{stem}.joblib",
        MODEL_DIR / f"{stem.replace('_model', '')}_tuned_model",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            try:
                model = joblib.load(path)
                print(f"      Loaded {display_name} from {path.name} ({path.stat().st_size:,} bytes)")
                return model
            except Exception as e:
                logger.warning("Failed to load %s from %s: %s", display_name, path, e)
                continue

    print(f"      [SKIP] {display_name} — no model file found (tried: {[str(p.relative_to(PROJECT_ROOT)) for p in candidates]})")
    return None


# ═══════════════════════════════════════════════════════════
#  Load Phase 3 statistical models
# ═══════════════════════════════════════════════════════════


def load_phase3_model(display_name: str, filename: str) -> Any | None:
    """Load a Phase 3 statistical model from the models/ directory."""
    path = MODEL_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        try:
            model = joblib.load(path)
            print(f"      Loaded {display_name} from {path.name} ({path.stat().st_size:,} bytes)")
            return model
        except Exception as e:
            logger.warning("Failed to load %s from %s: %s", display_name, path, e)
            return None

    # Try without .joblib extension
    alt = MODEL_DIR / filename.replace(".joblib", "")
    if alt.exists() and alt.stat().st_size > 0:
        try:
            model = joblib.load(alt)
            print(f"      Loaded {display_name} from {alt.name} ({alt.stat().st_size:,} bytes)")
            return model
        except Exception as e:
            logger.warning("Failed to load %s from %s: %s", display_name, alt, e)
            return None

    print(f"      [SKIP] {display_name} — no model file found (tried: {path.name}, {alt.name})")
    return None


# ═══════════════════════════════════════════════════════════
#  Get predictions from a Phase 4 model
# ═══════════════════════════════════════════════════════════


def get_phase4_probs(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
) -> np.ndarray | None:
    """Get predict_proba from an ML model, with NaN handling."""
    if not hasattr(model, "predict_proba"):
        print(f"      [WARN] {model_name} has no predict_proba — skipping")
        return None

    try:
        # Handle NaN based on model type
        if model_name in ("LightGBM", "XGBoost"):
            probs = model.predict_proba(X)
        else:
            probs = model.predict_proba(X.fillna(X.mean().fillna(0)))

        if probs.shape[1] != 3:
            print(f"      [WARN] {model_name} predict_proba shape {probs.shape} — expected (n, 3)")
            return None

        return np.asarray(probs, dtype=np.float64)
    except Exception as e:
        print(f"      [WARN] {model_name} predict_proba failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  Get predictions from a Phase 3 model (Poisson, Dixon-Coles, Elo)
# ═══════════════════════════════════════════════════════════


def get_phase3_probs(
    model: Any,
    df_matches: pd.DataFrame,
    model_name: str,
) -> np.ndarray | None:
    """Get probability matrix from a Phase 3 statistical model.

    These models use predict_matches() which returns a DataFrame with
    away_win_prob, draw_prob, home_win_prob columns.
    """
    if not hasattr(model, "predict_matches"):
        print(f"      [WARN] {model_name} has no predict_matches — skipping")
        return None

    try:
        preds = model.predict_matches(df_matches)
        required = {"away_win_prob", "draw_prob", "home_win_prob"}
        if not required.issubset(preds.columns):
            print(f"      [WARN] {model_name} predict_matches missing columns: {required - set(preds.columns)}")
            return None

        probs = np.column_stack([
            preds["away_win_prob"].values,
            preds["draw_prob"].values,
            preds["home_win_prob"].values,
        ])

        # Handle NaN in predictions
        nan_mask = np.isnan(probs).any(axis=1)
        if nan_mask.any():
            print(f"      [WARN] {model_name}: {nan_mask.sum()} rows have NaN probs — filling with uniform")
            probs[nan_mask] = 1.0 / 3

        return np.asarray(probs, dtype=np.float64)
    except Exception as e:
        print(f"      [WARN] {model_name} predict_matches failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  Calibrate and evaluate a single model
# ═══════════════════════════════════════════════════════════


def calibrate_and_evaluate(
    model_name: str,
    phase: str,
    val_probs: np.ndarray,
    y_val: np.ndarray,
    test_probs: np.ndarray | None,
    y_test: np.ndarray | None,
    val_logits: np.ndarray | None = None,
    test_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply all calibration methods and evaluate Brier score.

    Parameters
    ----------
    model_name : str
        Display name of the model.
    phase : str
        "Phase 4 (ML)" or "Phase 3 (Stats)".
    val_probs : np.ndarray of shape (n_val, 3)
        Raw probabilities on validation set (used to fit calibrators).
    y_val : np.ndarray of shape (n_val,)
        True labels for validation set.
    test_probs : np.ndarray of shape (n_test, 3) or None
        Raw probabilities on test set (used to evaluate calibration).
    y_test : np.ndarray of shape (n_test,) or None
        True labels for test set.
    val_logits : np.ndarray of shape (n_val, 3) or None
        Logits for temperature scaling (if different from probabilities).
    test_logits : np.ndarray of shape (n_test, 3) or None
        Logits for temperature scaling on test set.

    Returns
    -------
    dict with model_name, phase, raw_brier, and results per calibration method.
    """
    result: dict[str, Any] = {
        "model": model_name,
        "phase": phase,
        "n_val": len(val_probs),
        "n_test": len(test_probs) if test_probs is not None else 0,
    }

    # ── Raw (uncalibrated) metrics ──────────────────────
    raw_brier_val = brier_score(y_val, val_probs)
    result["raw_brier_val"] = round(raw_brier_val, 6)
    result["raw_log_loss_val"] = round(float(log_loss(y_val, val_probs)), 6)

    if test_probs is not None and y_test is not None:
        result["raw_brier_test"] = round(brier_score(y_test, test_probs), 6)
        result["raw_log_loss_test"] = round(float(log_loss(y_test, test_probs)), 6)
    else:
        result["raw_brier_test"] = None
        result["raw_log_loss_test"] = None

    # ── Calibration results ─────────────────────────────
    result["calibration_results"] = {}

    # 1. Platt Scaling
    for method_name, CalibratorClass, use_logits in [
        ("platt", PlattScalingCalibrator, False),
        ("isotonic", IsotonicRegressionCalibrator, False),
    ]:
        cal_tag = f"{method_name}_scaling"
        try:
            calibrator = CalibratorClass(n_classes=N_CLASSES)
            calibrator.fit(val_probs, y_val)

            cal_val = calibrator.transform(val_probs)
            cal_brier_val = brier_score(y_val, cal_val)

            cal_test = None
            cal_brier_test = None
            cal_log_loss_test = None
            if test_probs is not None and y_test is not None:
                try:
                    cal_test = calibrator.transform(test_probs)
                    cal_brier_test = brier_score(y_test, cal_test)
                    cal_log_loss_test = float(log_loss(y_test, cal_test))
                except Exception as e:
                    logger.warning("%s transform on test set failed: %s", cal_tag, e)

            result["calibration_results"][cal_tag] = {
                "method": method_name,
                "brier_val": round(cal_brier_val, 6),
                "brier_improvement_val": round(raw_brier_val - cal_brier_val, 6),
                "brier_test": round(cal_brier_test, 6) if cal_brier_test is not None else None,
                "brier_improvement_test": (
                    round(result.get("raw_brier_test", 0) - cal_brier_test, 6)
                    if cal_brier_test is not None and result.get("raw_brier_test") is not None
                    else None
                ),
                "log_loss_test": round(cal_log_loss_test, 6) if cal_log_loss_test is not None else None,
                "fitted": calibrator.fitted,
            }
        except Exception as e:
            logger.error("%s calibration failed for %s: %s", method_name, model_name, e)
            result["calibration_results"][cal_tag] = {
                "method": method_name,
                "error": str(e),
                "fitted": False,
            }

    # 2. Temperature Scaling (uses logits if provided, else logit of probs)
    cal_tag = "temperature_scaling"
    try:
        # Temperature scaling works on logits. If no logits provided,
        # convert probabilities to logits for a pseudo-input.
        fit_logits: np.ndarray
        if val_logits is not None:
            fit_logits = val_logits
        else:
            # Convert probs to logits (with clipping)
            p = np.clip(val_probs, 1e-7, 1 - 1e-7)
            fit_logits = np.log(p / (1.0 - p))

        # Check for Inf after logit transform (can happen with extreme probs)
        fit_logits = np.nan_to_num(fit_logits, nan=0.0, posinf=1e6, neginf=-1e6)

        temp_cal = TemperatureScalingCalibrator(n_classes=N_CLASSES)
        temp_cal.fit(fit_logits, y_val)

        # Evaluate on val set using probability transform
        cal_val_temp = temp_cal.transform(fit_logits)
        cal_brier_val_temp = brier_score(y_val, cal_val_temp)

        cal_test_temp = None
        cal_brier_test_temp = None
        cal_log_loss_test_temp = None
        if test_probs is not None and y_test is not None:
            try:
                if test_logits is not None:
                    t_logits = test_logits
                else:
                    p_test = np.clip(test_probs, 1e-7, 1 - 1e-7)
                    t_logits = np.log(p_test / (1.0 - p_test))
                    t_logits = np.nan_to_num(t_logits, nan=0.0, posinf=1e6, neginf=-1e6)
                cal_test_temp = temp_cal.transform(t_logits)
                cal_brier_test_temp = brier_score(y_test, cal_test_temp)
                cal_log_loss_test_temp = float(log_loss(y_test, cal_test_temp))
            except Exception as e:
                logger.warning("Temperature transform on test set failed: %s", e)

        result["calibration_results"][cal_tag] = {
            "method": "temperature",
            "temperature": round(temp_cal.temperature_, 6),
            "brier_val": round(cal_brier_val_temp, 6),
            "brier_improvement_val": round(raw_brier_val - cal_brier_val_temp, 6),
            "brier_test": round(cal_brier_test_temp, 6) if cal_brier_test_temp is not None else None,
            "brier_improvement_test": (
                round(result.get("raw_brier_test", 0) - cal_brier_test_temp, 6)
                if cal_brier_test_temp is not None and result.get("raw_brier_test") is not None
                else None
            ),
            "log_loss_test": round(cal_log_loss_test_temp, 6) if cal_log_loss_test_temp is not None else None,
            "fitted": temp_cal.fitted,
        }
    except Exception as e:
        logger.error("Temperature calibration failed for %s: %s", model_name, e)
        result["calibration_results"][cal_tag] = {
            "method": "temperature",
            "error": str(e),
            "fitted": False,
        }

    return result


# ═══════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════


def run_calibration(quiet: bool = False, skip_phase3: bool = False) -> dict[str, Any]:
    """Run the full calibration pipeline."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "n_classes": N_CLASSES,
        "calibration_methods": CALIBRATION_METHODS + ["temperature"],
    }

    print("\n" + "=" * 75)
    print("  CALIBRATE ALL MODELS — Phase 3 & Phase 4")
    print("=" * 75)

    # ── [1/4] Load data ────────────────────────────────
    print("\n[1/4] Loading and preparing data...")
    X, y, splits = load_phase4_data()
    df_raw = load_phase3_data()

    df_train_p3, df_val_p3, df_test_p3 = phase3_chronological_split(df_raw, splits)

    y_val = splits["y_val"].values if hasattr(splits["y_val"], "values") else np.asarray(splits["y_val"])
    y_test = splits["y_test"].values if hasattr(splits["y_test"], "values") else np.asarray(splits["y_test"])

    report["split_sizes"] = {
        "train": int(splits["X_train"].shape[0]),
        "val": int(splits["X_val"].shape[0]),
        "test": int(splits["X_test"].shape[0]),
    }

    # ── [2/4] Process Phase 4 ML models ────────────────
    print(f"\n[2/4] Calibrating {len(PHASE4_MODELS)} Phase 4 ML models...")
    phase4_results: list[dict[str, Any]] = []

    for display_name, stem in PHASE4_MODELS.items():
        print(f"\n  {display_name}...")
        model = load_phase4_model(display_name, stem)
        if model is None:
            continue

        # Get predictions
        val_probs = get_phase4_probs(model, splits["X_val"], display_name)
        if val_probs is None:
            continue

        test_probs = get_phase4_probs(model, splits["X_test"], display_name)
        if test_probs is None:
            print(f"      [WARN] {display_name}: test predictions failed — skipping test eval")
            test_probs = None

        # For Neural Network, try to get logits for temperature scaling
        val_logits = None
        test_logits = None
        if display_name == "Neural Network" and hasattr(model, "predict_proba"):
            try:
                # NN outputs softmax directly — we can use the raw logits
                # if the model exposes them. Otherwise, we use probs -> logits.
                val_logits = None  # Will be derived from probs
            except Exception:
                pass

        # Calibrate
        result = calibrate_and_evaluate(
            model_name=display_name,
            phase="Phase 4 (ML)",
            val_probs=val_probs,
            y_val=y_val,
            test_probs=test_probs,
            y_test=y_test,
            val_logits=val_logits,
            test_logits=test_logits,
        )
        phase4_results.append(result)

        # Print summary
        raw_b = result["raw_brier_test"] if result.get("raw_brier_test") is not None else result["raw_brier_val"]
        print(f"      Raw Brier: {raw_b:.4f}")
        for cal_tag, cal_res in result["calibration_results"].items():
            if cal_res.get("fitted") and cal_res.get("error") is None:
                brier = cal_res.get("brier_test") if cal_res.get("brier_test") is not None else cal_res.get("brier_val")
                imp = cal_res.get("brier_improvement_test") or cal_res.get("brier_improvement_val") or 0
                print(f"      {cal_tag:<25s} Brier: {brier:.4f} (Δ={imp:+.4f})")
            else:
                print(f"      {cal_tag:<25s} FAILED: {cal_res.get('error', 'unknown')}")

    report["phase4"] = phase4_results

    # ── [3/4] Process Phase 3 statistical models ───────
    if not skip_phase3:
        print(f"\n[3/4] Calibrating {len(PHASE3_MODELS)} Phase 3 statistical models...")
        phase3_results: list[dict[str, Any]] = []

        for display_name, filename in PHASE3_MODELS.items():
            print(f"\n  {display_name}...")
            model = load_phase3_model(display_name, filename)
            if model is None:
                continue

            # Get predictions
            val_probs = get_phase3_probs(model, df_val_p3, display_name)
            if val_probs is None:
                continue

            test_probs = get_phase3_probs(model, df_test_p3, display_name)
            if test_probs is None:
                print(f"      [WARN] {display_name}: test predictions failed — skipping test eval")
                test_probs = None

            # For Phase 3, y_true comes from the raw data
            y_val_p3 = df_val_p3["target"].values.astype(int)
            y_test_p3 = df_test_p3["target"].values.astype(int)

            # Calibrate
            result = calibrate_and_evaluate(
                model_name=display_name,
                phase="Phase 3 (Stats)",
                val_probs=val_probs,
                y_val=y_val_p3,
                test_probs=test_probs,
                y_test=y_test_p3,
            )
            phase3_results.append(result)

            # Print summary
            raw_b = result["raw_brier_test"] if result.get("raw_brier_test") is not None else result["raw_brier_val"]
            print(f"      Raw Brier: {raw_b:.4f}")
            for cal_tag, cal_res in result["calibration_results"].items():
                if cal_res.get("fitted") and cal_res.get("error") is None:
                    brier = cal_res.get("brier_test") if cal_res.get("brier_test") is not None else cal_res.get("brier_val")
                    imp = cal_res.get("brier_improvement_test") or cal_res.get("brier_improvement_val") or 0
                    print(f"      {cal_tag:<25s} Brier: {brier:.4f} (Δ={imp:+.4f})")
                else:
                    print(f"      {cal_tag:<25s} FAILED: {cal_res.get('error', 'unknown')}")

        report["phase3"] = phase3_results
    else:
        print("\n[3/4] Skipping Phase 3 models (--skip-phase3)")
        report["phase3"] = []

    # ── [4/4] Save results ─────────────────────────────
    print(f"\n[4/4] Saving calibration results...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Build summary table
    summary_rows = []
    for result in phase4_results + report.get("phase3", []):
        row = {
            "model": result["model"],
            "phase": result["phase"],
            "raw_brier": result.get("raw_brier_test") or result.get("raw_brier_val"),
            "raw_log_loss": result.get("raw_log_loss_test") or result.get("raw_log_loss_val"),
        }
        for cal_tag in ["platt_scaling", "isotonic_scaling", "temperature_scaling"]:
            cal_res = result.get("calibration_results", {}).get(cal_tag, {})
            if cal_res.get("fitted"):
                row[f"{cal_tag}_brier"] = cal_res.get("brier_test") or cal_res.get("brier_val")
                row[f"{cal_tag}_improvement"] = cal_res.get("brier_improvement_test") or cal_res.get("brier_improvement_val") or 0
            else:
                row[f"{cal_tag}_brier"] = None
                row[f"{cal_tag}_improvement"] = None
        summary_rows.append(row)

    report["summary"] = summary_rows

    # Best calibration per model
    best_cal: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        model_name = row["model"]
        best_method = "none"
        best_brier = row["raw_brier"]
        for cal_tag in ["platt_scaling", "isotonic_scaling", "temperature_scaling"]:
            brier_val = row.get(f"{cal_tag}_brier")
            if brier_val is not None and (best_brier is None or brier_val < best_brier):
                best_brier = brier_val
                best_method = cal_tag
        best_cal[model_name] = {
            "best_method": best_method,
            "best_brier": best_brier,
            "improvement": round((row["raw_brier"] - best_brier), 6) if best_brier is not None and row["raw_brier"] is not None else None,
        }
    report["best_calibration_per_model"] = best_cal

    # Save JSON
    json_path = REPORT_DIR / f"calibration_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved: {json_path}")

    # ── Print final summary ─────────────────────────────
    duration = time.time() - start_time
    report["duration_seconds"] = round(duration, 2)

    print("\n" + "=" * 75)
    print("  CALIBRATION SUMMARY".center(73))
    print("=" * 75)
    print(f"  Duration: {duration:.1f}s")
    print(f"  Models processed: {len(phase4_results)} Phase 4 + {len(report.get('phase3', []))} Phase 3")
    print()

    # Print best per model
    header = f"  {'Model':<22s} {'Phase':<18s} {'Raw Brier':<12s} {'Best Cal':<20s} {'Best Brier':<12s} {'Δ':<10s}"
    print(header)
    print(f"  {'-' * 94}")
    for row in summary_rows:
        model_name = row["model"]
        bc = best_cal.get(model_name, {})
        raw_b = row["raw_brier"]
        best_m = bc.get("best_method", "none")
        best_b = bc.get("best_brier", raw_b)
        imp = bc.get("improvement", 0)
        imp_str = f"{imp:+.4f}" if imp is not None and imp != 0 else "—"
        print(f"  {model_name:<22s} {row['phase']:<18s} "
              f"{raw_b:<12.4f} {best_m:<20s} "
              f"{best_b:<12.4f} {imp_str:<10s}")

    # Most improved by calibration
    print(f"\n  Most improved models:")
    sorted_by_imp = sorted(
        [(m, bc["improvement"], bc["best_method"]) for m, bc in best_cal.items()
         if bc.get("improvement") and bc["improvement"] > 0],
        key=lambda x: x[1], reverse=True,
    )
    if sorted_by_imp:
        for model_name, imp, method in sorted_by_imp[:5]:
            print(f"    {model_name:<22s} {method:<20s} Δ={imp:.4f}")
    else:
        print("    (no improvement from calibration)")

    print("\n" + "=" * 75)
    print("  CALIBRATION COMPLETE".center(73))
    print("=" * 75)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate all Phase 3 & Phase 4 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed logging",
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip Phase 3 statistical models (Poisson, Dixon-Coles, Elo)",
    )
    args = parser.parse_args()

    try:
        run_calibration(quiet=args.quiet, skip_phase3=args.skip_phase3)
        return 0
    except Exception as e:
        logger.error("Calibration pipeline failed: %s", e, exc_info=True)
        print(f"\n[FAIL] Calibration pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    # Wrap stdout to handle Unicode on Windows terminals
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
