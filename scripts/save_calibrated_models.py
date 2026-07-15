#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Save Calibrated Models                                                     ║
║                                                                             ║
║  For each model, applies the best calibration method, saves the calibrated  ║
║  model to models/calibrated_{model_name}.joblib with metadata, and verifies ║
║  that saved models can be loaded and produce valid predictions.             ║
║                                                                             ║
║  Usage:                                                                     ║
║      python scripts/save_calibrated_models.py                               ║
║      python scripts/save_calibrated_models.py --selection reports/....json  ║
║      python scripts/save_calibrated_models.py --skip-verify                 ║
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
    CalibratedModel,
    CalibratedStatsModel,
    CalibratedTemperatureWrapper,
    IsotonicRegressionCalibrator,
    PlattScalingCalibrator,
    TemperatureScalingCalibrator,
    calibrate_model,
)
from src.data_loader import load_results
from src.feature_engineering import build_features, train_val_test_split
from src.time_series_cv import time_series_train_val_test_split
from src.poisson_model import PoissonModel
from src.dixon_coles import DixonColesModel
from src.elo import EloSystem

logger = logging.getLogger(__name__)

N_CLASSES = 3
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

# Mapping from selection method names to calibrator classes
CALIBRATOR_CLASSES: dict[str, type] = {
    "Platt": PlattScalingCalibrator,
    "Isotonic": IsotonicRegressionCalibrator,
    "Temperature": TemperatureScalingCalibrator,
}

# Phase 4 model stems (used for file lookup)
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
#  Wrappers for calibrated models (imported from src.calibration)
# ═══════════════════════════════════════════════════════════
#  CalibratedTemperatureWrapper and CalibratedStatsModel are
#  defined in src/calibration.py for joblib importability.
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
#  Data loading helpers
# ═══════════════════════════════════════════════════════════


def load_phase4_splits() -> dict[str, Any]:
    """Load data and build feature matrix, return chronological splits."""
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

    print(f"      Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
    return splits


def load_phase3_data() -> pd.DataFrame:
    """Load raw match data for Phase 3 statistical models."""
    print("  [Data] Loading raw match data...")
    df = load_results(low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df["result"].notna() & df["result"].isin(["H", "D", "A"])].copy()
    df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df = df[df["target"] >= 0].copy()

    # Chronological split using same ratio
    n = len(df)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    print(f"      Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════
#  Model loading
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


def load_phase4_model(display_name: str) -> Any | None:
    """Load a Phase 4 ML model."""
    stem = PHASE4_MODELS.get(display_name)
    if stem is None:
        return None
    path = find_model_file(stem)
    if path is None:
        print(f"      [SKIP] {display_name}: model file not found")
        return None
    try:
        model = joblib.load(path)
        print(f"      Loaded {display_name} from {path.name}")
        return model
    except Exception as e:
        print(f"      [FAIL] {display_name}: load error — {e}")
        return None


def load_phase3_model(display_name: str) -> Any | None:
    """Load a Phase 3 statistical model."""
    filename = PHASE3_MODELS.get(display_name)
    if filename is None:
        return None
    path = MODEL_DIR / filename
    if not path.exists():
        # Try extensionless
        alt = MODEL_DIR / filename.replace(".joblib", "")
        if alt.exists():
            path = alt
        else:
            print(f"      [SKIP] {display_name}: model file not found")
            return None
    try:
        model = joblib.load(path)
        print(f"      Loaded {display_name} from {path.name}")
        return model
    except Exception as e:
        print(f"      [FAIL] {display_name}: load error — {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  Calibrator fitting
# ═══════════════════════════════════════════════════════════


def fit_calibrator(
    method: str,
    val_probs: np.ndarray,
    y_val: np.ndarray,
) -> Any | None:
    """Fit a calibrator for the given method on validation data.

    Parameters
    ----------
    method : str
        One of "Platt", "Isotonic", "Temperature".
    val_probs : np.ndarray of shape (n, 3)
        Raw probabilities on validation set.
    y_val : np.ndarray of shape (n,)
        True labels.

    Returns
    -------
    Fitted calibrator instance, or None on failure.
    """
    try:
        if method == "Platt":
            cal = PlattScalingCalibrator(n_classes=N_CLASSES)
            cal.fit(val_probs, y_val)
            return cal
        elif method == "Isotonic":
            cal = IsotonicRegressionCalibrator(n_classes=N_CLASSES)
            cal.fit(val_probs, y_val)
            return cal
        elif method == "Temperature":
            # Convert probs to logits for temperature scaling
            p = np.clip(val_probs, 1e-7, 1 - 1e-7)
            logits = np.log(p / (1.0 - p))
            logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            cal = TemperatureScalingCalibrator(n_classes=N_CLASSES)
            cal.fit(logits, y_val)
            return cal
        else:
            print(f"      [WARN] Unknown calibration method: {method}")
            return None
    except Exception as e:
        print(f"      [FAIL] Calibrator fit error ({method}): {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  Save and verify
# ═══════════════════════════════════════════════════════════


def save_calibrated_model(
    payload: dict[str, Any],
    model_name_slug: str,
) -> Path:
    """Save the model payload to models/calibrated_{slug}.joblib."""
    path = MODEL_DIR / f"calibrated_{model_name_slug}.joblib"
    joblib.dump(payload, path)
    return path


def verify_model(
    payload: dict[str, Any],
    display_name: str,
    model_name_slug: str,
    test_X: Any | None = None,
    test_df: pd.DataFrame | None = None,
) -> bool:
    """Verify a saved model loads and produces predictions."""
    saved_path = MODEL_DIR / f"calibrated_{model_name_slug}.joblib"

    # 1. Load
    try:
        loaded = joblib.load(saved_path)
        print(f"      Verified: load OK")
    except Exception as e:
        print(f"      [FAIL] Verify load failed: {e}")
        return False

    # 2. Check structure
    if not isinstance(loaded, dict) or "model" not in loaded or "metadata" not in loaded:
        print(f"      [FAIL] Verify: payload missing required keys")
        return False

    metadata = loaded["metadata"]
    model = loaded["model"]

    # 3. Check metadata
    for key in ["calibration_method", "calibration_date", "original_model_name"]:
        if key not in metadata:
            print(f"      [FAIL] Verify: metadata missing '{key}'")
            return False

    # 4. Try to predict
    try:
        if hasattr(model, "predict_proba"):
            # Phase 4 model
            if test_X is not None:
                probs = model.predict_proba(test_X)
                assert probs.shape[1] == N_CLASSES, f"Expected {N_CLASSES} classes, got {probs.shape[1]}"
                assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"
                print(f"      Verified: predict_proba OK (shape={probs.shape}, sum≈1.0)")
        elif hasattr(model, "predict_matches"):
            # Phase 3 model
            if test_df is not None:
                preds = model.predict_matches(test_df)
                for col in ["away_win_prob", "draw_prob", "home_win_prob"]:
                    assert col in preds.columns, f"Missing column: {col}"
                probs = np.column_stack([
                    preds["away_win_prob"], preds["draw_prob"], preds["home_win_prob"],
                ])
                assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Probabilities don't sum to 1"
                print(f"      Verified: predict_matches OK (rows={len(preds)}, sum≈1.0)")
        else:
            print(f"      [WARN] Verify: model has no predict_proba or predict_matches — checking type only")
            print(f"      Verified: model type={type(model).__name__}")
    except Exception as e:
        print(f"      [FAIL] Verify prediction failed: {e}")
        return False

    return True


# ═══════════════════════════════════════════════════════════
#  Process Phase 4 models
# ═══════════════════════════════════════════════════════════


def process_phase4_models(
    selection: dict[str, Any],
    splits: dict[str, Any],
    skip_verify: bool = False,
) -> list[dict[str, Any]]:
    """Process all Phase 4 models: calibrate, save, verify."""
    results: list[dict[str, Any]] = []

    for display_name, stem in PHASE4_MODELS.items():
        model_name_slug = stem.replace("_model", "")
        sel = selection.get(display_name)

        print(f"\n  ── {display_name} ──")
        print(f"      Best method: {sel.get('best_method', 'none') if sel else 'N/A'}")

        if sel is None:
            print(f"      [SKIP] No selection data")
            continue

        model = load_phase4_model(display_name)
        if model is None:
            continue

        best_method = sel["best_method"]
        metadata = {
            "original_model_name": display_name,
            "original_model_type": "Phase 4 (ML)",
            "calibration_method": best_method,
            "calibration_date": datetime.now().isoformat(),
            "original_brier": sel.get("original_brier"),
            "calibrated_brier": sel.get("calibrated_brier"),
            "improvement": sel.get("improvement"),
        }

        if best_method == "none":
            # Save original model with metadata only
            payload = {"model": model, "metadata": metadata}
            path = save_calibrated_model(payload, model_name_slug)
            print(f"      Saved (uncalibrated): {path.name}")

            # Verify
            if not skip_verify:
                X_mean = splits["X_test"].mean().fillna(0)
                verify_model(payload, display_name, model_name_slug,
                             test_X=splits["X_test"].fillna(X_mean))

            results.append({"model": display_name, "method": "none", "path": str(path)})
            continue

        # Fit calibrator
        X_mean = splits["X_train"].mean().fillna(0)
        if display_name == "Neural Network":
            X_val_clean = splits["X_val"].fillna(0)
        else:
            X_val_clean = splits["X_val"].fillna(X_mean)

        val_probs = model.predict_proba(X_val_clean)
        y_val = splits["y_val"].values if hasattr(splits["y_val"], "values") else np.asarray(splits["y_val"])

        calibrator = fit_calibrator(best_method, val_probs, y_val)
        if calibrator is None:
            print(f"      [FAIL] Could not fit calibrator — saving uncalibrated")
            payload = {"model": model, "metadata": {**metadata, "calibration_method": "none (failed)"}}
            path = save_calibrated_model(payload, model_name_slug)
            results.append({"model": display_name, "method": "none (failed)", "path": str(path)})
            continue

        # Wrap and save
        if best_method == "Temperature":
            wrapped = CalibratedTemperatureWrapper(base_model=model, calibrator=calibrator)
        else:
            wrapped = CalibratedModel(base_model=model, method=best_method.lower(), n_classes=N_CLASSES)
            wrapped._calibrators = calibrator._calibrators
            wrapped._fitted = True

        payload = {"model": wrapped, "metadata": metadata}
        path = save_calibrated_model(payload, model_name_slug)
        print(f"      Saved (calibrated): {path.name}")

        # Verify
        if not skip_verify:
            test_X_clean = splits["X_test"].fillna(X_mean)
            verify_model(payload, display_name, model_name_slug, test_X=test_X_clean)

        results.append({"model": display_name, "method": best_method, "path": str(path)})

    return results


# ═══════════════════════════════════════════════════════════
#  Process Phase 3 models
# ═══════════════════════════════════════════════════════════


def process_phase3_models(
    selection: dict[str, Any],
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    skip_verify: bool = False,
) -> list[dict[str, Any]]:
    """Process all Phase 3 models: calibrate, save, verify."""
    results: list[dict[str, Any]] = []

    for display_name, filename in PHASE3_MODELS.items():
        model_name_slug = filename.replace("_model.joblib", "").replace(".joblib", "")
        sel = selection.get(display_name)

        print(f"\n  ── {display_name} ──")
        print(f"      Best method: {sel.get('best_method', 'none') if sel else 'N/A'}")

        if sel is None:
            print(f"      [SKIP] No selection data")
            continue

        model = load_phase3_model(display_name)
        if model is None:
            continue

        best_method = sel["best_method"]
        metadata = {
            "original_model_name": display_name,
            "original_model_type": "Phase 3 (Stats)",
            "calibration_method": best_method,
            "calibration_date": datetime.now().isoformat(),
            "original_brier": sel.get("original_brier"),
            "calibrated_brier": sel.get("calibrated_brier"),
            "improvement": sel.get("improvement"),
        }

        if best_method == "none":
            payload = {"model": model, "metadata": metadata}
            path = save_calibrated_model(payload, model_name_slug)
            print(f"      Saved (uncalibrated): {path.name}")

            if not skip_verify:
                verify_model(payload, display_name, model_name_slug, test_df=df_test)

            results.append({"model": display_name, "method": "none", "path": str(path)})
            continue

        # Fit calibrator on validation predictions
        val_preds = model.predict_matches(df_val)
        val_probs = np.column_stack([
            val_preds["away_win_prob"].values,
            val_preds["draw_prob"].values,
            val_preds["home_win_prob"].values,
        ])
        y_val = df_val["target"].values.astype(int)

        calibrator = fit_calibrator(best_method, val_probs, y_val)
        if calibrator is None:
            print(f"      [FAIL] Could not fit calibrator — saving uncalibrated")
            payload = {"model": model, "metadata": {**metadata, "calibration_method": "none (failed)"}}
            path = save_calibrated_model(payload, model_name_slug)
            results.append({"model": display_name, "method": "none (failed)", "path": str(path)})
            continue

        # Wrap and save
        wrapped = CalibratedStatsModel(base_model=model, calibrator=calibrator, method=best_method)
        payload = {"model": wrapped, "metadata": metadata}
        path = save_calibrated_model(payload, model_name_slug)
        print(f"      Saved (calibrated): {path.name}")

        if not skip_verify:
            verify_model(payload, display_name, model_name_slug, test_df=df_test)

        results.append({"model": display_name, "method": best_method, "path": str(path)})

    return results


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def run_save(
    selection_path: Path | None = None,
    skip_verify: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the save pipeline."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    start_time = time.time()

    print("\n" + "=" * 75)
    print("  SAVE CALIBRATED MODELS")
    print("=" * 75)

    # ── Load selection ─────────────────────────────────
    if selection_path is None:
        sel_files = sorted(REPORT_DIR.glob("calibration_selection_*.json"))
        if not sel_files:
            print("[FAIL] No calibration_selection_*.json found. Run select_best_calibration.py first.")
            sys.exit(1)
        selection_path = sel_files[-1]

    print(f"\n  Selection: {selection_path.name}")
    with open(selection_path, encoding="utf-8") as f:
        selection_data = json.load(f)
    selection = selection_data.get("models", {})

    print(f"  Models to process: {len(selection)}")
    print(f"  Skip verify: {skip_verify}")

    # ── [1/3] Load data ────────────────────────────────
    print(f"\n[1/3] Loading data...")
    splits = load_phase4_splits()
    df_train_p3, df_val_p3, df_test_p3 = load_phase3_data()

    # ── [2/3] Process models ──────────────────────────
    print(f"\n[2/3] Processing models...")

    print(f"\n  ── Phase 4: ML Models ──")
    phase4_results = process_phase4_models(selection, splits, skip_verify)

    print(f"\n  ── Phase 3: Statistical Models ──")
    phase3_results = process_phase3_models(selection, df_val_p3, df_test_p3, skip_verify)

    # ── [3/3] Summary ─────────────────────────────────
    duration = time.time() - start_time
    all_results = phase4_results + phase3_results
    n_saved = len(all_results)
    n_calibrated = sum(1 for r in all_results if r["method"] not in ("none", "none (failed)"))
    n_failed = sum(1 for r in all_results if "failed" in r["method"])

    print(f"\n[3/3] Summary")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Total models saved: {n_saved}")
    print(f"  Calibrated: {n_calibrated}")
    print(f"  Uncalibrated (no improvement): {n_saved - n_calibrated - n_failed}")
    if n_failed:
        print(f"  Failed calibrations: {n_failed}")

    print(f"\n  Saved files:")
    for r in all_results:
        cal_tag = f" → {r['method']}" if r["method"] != "none" else " (uncalibrated)"
        print(f"    {Path(r['path']).name}{cal_tag}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "selection_source": str(selection_path.name),
        "duration_seconds": round(duration, 2),
        "models_processed": n_saved,
        "models_calibrated": n_calibrated,
        "models_uncalibrated": n_saved - n_calibrated - n_failed,
        "models_failed": n_failed,
        "results": all_results,
    }
    report_path = REPORT_DIR / f"save_calibrated_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")
    print("\n" + "=" * 75)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save calibrated models with metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--selection", "-s",
        type=str,
        default=None,
        help="Path to calibration_selection JSON (default: latest in reports/)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip prediction verification after save",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output",
    )
    args = parser.parse_args()

    sel_path = Path(args.selection) if args.selection else None
    try:
        run_save(selection_path=sel_path, skip_verify=args.skip_verify, quiet=args.quiet)
        return 0
    except Exception as e:
        print(f"\n[FAIL] Save pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
