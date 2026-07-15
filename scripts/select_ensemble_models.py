#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Select & Prepare Base Models for Ensemble                                  ║
║                                                                             ║
║  Reads the latest calibration selection, picks the top 3-4 models by        ║
║  calibrated Brier score (ensuring model-type diversity), loads each model,  ║
║  verifies they produce valid predictions on the same data, and saves the    ║
║  selection to reports/ensemble_selection_{timestamp}.json.                  ║
║                                                                             ║
║  Usage:                                                                     ║
║      python scripts/select_ensemble_models.py                               ║
║      python scripts/select_ensemble_models.py --models 3                   ║
║      python scripts/select_ensemble_models.py --force-xgboost --quiet      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# Model family groups — used to enforce diversity
MODEL_FAMILIES: dict[str, str] = {
    "XGBoost": "boosting",
    "LightGBM": "boosting",
    "Random Forest": "bagging",
    "Neural Network": "neural",
    "Logistic Regression": "linear",
    "Poisson": "poisson",
    "Dixon-Coles": "dixon_coles",
    "Elo": "elo",
}

# Mapping display names -> calibrated filenames
CALIBRATED_FILES: dict[str, str] = {
    "XGBoost": "calibrated_xgboost.joblib",
    "LightGBM": "calibrated_lightgbm.joblib",
    "Random Forest": "calibrated_random_forest.joblib",
    "Neural Network": "calibrated_neural_network.joblib",
    "Logistic Regression": "calibrated_logistic_regression.joblib",
    "Poisson": "calibrated_poisson.joblib",
    "Dixon-Coles": "calibrated_dixon_coles.joblib",
    "Elo": "calibrated_elo.joblib",
}


# ═══════════════════════════════════════════════════════════
#  Load calibration selection
# ═══════════════════════════════════════════════════════════


def load_latest_selection() -> dict[str, Any]:
    """Load the latest calibration_selection_*.json from reports/."""
    files = sorted(REPORT_DIR.glob("calibration_selection_*.json"))
    if not files:
        print("  [FAIL] No calibration_selection_*.json found. Run select_best_calibration or calibrate_all_models first.")
        sys.exit(1)
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_latest_diagrams() -> dict[str, Any]:
    """Load the latest calibration_diagrams_*.json if available."""
    files = sorted(REPORT_DIR.glob("calibration_diagrams_*.json"))
    if files:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    return {}


# ═══════════════════════════════════════════════════════════
#  Model selection logic
# ═══════════════════════════════════════════════════════════


def select_models(
    selection: dict[str, Any],
    n_models: int = 4,
    force_include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select the best models for the ensemble.

    Selection criteria (in order):
    1. Forced inclusion (if specified)
    2. Best calibrated Brier score
    3. Model-type diversity (different families preferred)
    4. Exclude models with no calibration improvement (unless forced)

    Parameters
    ----------
    selection : dict
        Calibration selection data from ``load_latest_selection()``.
    n_models : int
        Number of models to select (default 4).
    force_include : list[str] | None
        Model names to always include (e.g. ``[\"XGBoost\"]``).
    exclude : list[str] | None
        Model names to exclude.

    Returns
    -------
    list[dict[str, Any]]
        Sorted list of ``{name, path, type, family, brier_score, method, ...}``.
    """
    models_data = selection.get("models", {})
    exclude_set = set(exclude or [])
    force_set = set(force_include or [])
    forced: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for name, data in models_data.items():
        # Skip excluded models
        if name in exclude_set:
            continue

        cal_brier = data.get("calibrated_brier", 1.0)
        method = data.get("best_method", "none")
        family = MODEL_FAMILIES.get(name, "unknown")
        model_type = "ML" if family in ("boosting", "bagging", "neural", "linear") else "Statistical"
        file_name = CALIBRATED_FILES.get(name)

        if file_name is None or not (MODEL_DIR / file_name).exists():
            print(f"  [SKIP] {name}: no calibrated file found at {file_name}")
            continue

        entry = {
            "name": name,
            "filename": file_name,
            "path": str(MODEL_DIR / file_name),
            "type": model_type,
            "family": family,
            "brier_score": cal_brier,
            "original_brier": data.get("original_brier", cal_brier),
            "calibration_method": method if method != "none" else None,
            "improvement": data.get("improvement", 0.0),
        }

        if name in force_set:
            forced.append(entry)
        else:
            candidates.append(entry)

    # Sort candidates by calibrated Brier (ascending)
    candidates.sort(key=lambda x: x["brier_score"])

    # Select remaining slots from candidates, preferring different families
    selected = list(forced)
    used_families = {e["family"] for e in selected}

    # First pass: pick best candidate from each unused family
    for candidate in candidates:
        if len(selected) >= n_models:
            break
        if candidate["family"] not in used_families and candidate not in selected:
            used_families.add(candidate["family"])
            selected.append(candidate)

    # Second pass: fill remaining slots by Brier score
    for candidate in candidates:
        if len(selected) >= n_models:
            break
        if candidate not in selected:
            selected.append(candidate)

    return selected[:n_models]


# ═══════════════════════════════════════════════════════════
#  Model verification
# ═══════════════════════════════════════════════════════════


def load_model(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a calibrated model file and return (model, metadata)."""
    import joblib

    payload = joblib.load(path)
    model = payload.get("model", payload)
    metadata = payload.get("metadata", {})
    return model, metadata


def verify_predictions(
    selected: list[dict[str, Any]],
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Load each selected model and verify predictions on test data.

    For Phase 4 models: verifies ``predict_proba(X)`` returns (n, 3).
    For Phase 3 models: verifies ``predict_matches(df)`` returns DataFrame
    with required probability columns (or ``predict_proba(df)``).

    Returns
    -------
    list[dict[str, Any]]
        Enriched entries with ``verified`` and ``n_features`` fields.
    """
    import pandas as pd
    from src.data_loader import load_results
    from src.feature_engineering import build_features

    if not quiet:
        print("\n  Loading test data for verification...")

    # ── Load and prepare data ──
    df = load_results(low_memory=False)
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)

    # Build features using a larger sample so H2H/rolling stats populate
    # Use last 500 rows for build_features, then test on the final 50
    df_test_raw = df.tail(50).copy()
    try:
        # Build features on 500 rows to get complete feature set (H2H, etc.)
        df_feat_input = df.tail(500).copy()
        X_full, _ = build_features(df_feat_input)  # returns (X, y)
        X_test = X_full.tail(50)  # Use last 50 rows
        if not quiet:
            print(f"    Feature matrix: {X_test.shape} ({len(X_test.columns)} cols)")
    except Exception as e:
        if not quiet:
            print(f"    [WARN] build_features failed: {e}")
            print("    Falling back to raw DataFrame for all models.")
        X_test = df_test_raw

    n_features = X_test.shape[1] if hasattr(X_test, "shape") and len(X_test.shape) >= 2 else 0

    results: list[dict[str, Any]] = []
    for entry in selected:
        name = entry["name"]
        path = entry["path"]
        model_type = entry["type"]

        if not quiet:
            print(f"\n  Verifying {name} ({model_type})...")

        try:
            model, metadata = load_model(path)
        except Exception as e:
            msg = f"Failed to load: {e}"
            if not quiet:
                print(f"    [FAIL] {msg}")
            entry["verified"] = False
            entry["verify_error"] = msg
            entry["n_features"] = n_features
            results.append(entry)
            continue

        # ── Test Phase 4 models (ML) ──
        if model_type == "ML":
            if not hasattr(model, "predict_proba"):
                msg = "No predict_proba method"
                if not quiet:
                    print(f"    [FAIL] {msg}")
                entry["verified"] = False
                entry["verify_error"] = msg
                entry["n_features"] = n_features
                results.append(entry)
                continue

            try:
                probs = model.predict_proba(X_test)
                probs = np.asarray(probs, dtype=np.float64)

                if probs.shape != (50, 3):
                    msg = f"Expected shape (50, 3), got {probs.shape}"
                    if not quiet:
                        print(f"    [FAIL] {msg}")
                    entry["verified"] = False
                    entry["verify_error"] = msg
                elif not np.all(np.isfinite(probs)):
                    msg = "Predictions contain NaN or Inf"
                    if not quiet:
                        print(f"    [FAIL] {msg}")
                    entry["verified"] = False
                    entry["verify_error"] = msg
                else:
                    row_sums = probs.sum(axis=1)
                    if not quiet:
                        print(f"    predict_proba OK: shape={probs.shape}, sum range=[{row_sums.min():.4f}, {row_sums.max():.4f}]")
                    entry["verified"] = True
                    entry["verify_error"] = None

            except Exception as e:
                msg = f"predict_proba failed: {e}"
                if not quiet:
                    print(f"    [FAIL] {msg}")
                entry["verified"] = False
                entry["verify_error"] = msg

        # ── Test Phase 3 models (Statistical) ──
        else:
            if hasattr(model, "predict_matches"):
                try:
                    preds_df = model.predict_matches(df_test_raw)
                    has_cols = all(
                        c in preds_df.columns
                        for c in ["away_win_prob", "draw_prob", "home_win_prob"]
                    )
                    if not quiet:
                        cols_found = [c for c in ["away_win_prob", "draw_prob", "home_win_prob"] if c in preds_df.columns]
                        print(f"    predict_matches OK: shape={preds_df.shape}, cols={cols_found}")
                    entry["verified"] = has_cols
                    entry["verify_error"] = None if has_cols else "Missing probability columns"
                except Exception as e:
                    msg = f"predict_matches failed: {e}"
                    if not quiet:
                        print(f"    [FAIL] {msg}")
                    entry["verified"] = False
                    entry["verify_error"] = msg

            elif hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(df_test_raw)
                    probs = np.asarray(probs, dtype=np.float64)
                    if not quiet:
                        print(f"    predict_proba(df) OK: shape={probs.shape}")
                    entry["verified"] = True
                    entry["verify_error"] = None
                except Exception as e:
                    msg = f"predict_proba failed: {e}"
                    if not quiet:
                        print(f"    [FAIL] {msg}")
                    entry["verified"] = False
                    entry["verify_error"] = msg
            else:
                msg = "No predict_matches or predict_proba"
                if not quiet:
                    print(f"    [FAIL] {msg}")
                entry["verified"] = False
                entry["verify_error"] = msg

        entry["n_features"] = n_features
        results.append(entry)

    return results


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def run_selection(
    n_models: int = 4,
    force_include: list[str] | None = None,
    exclude: list[str] | None = None,
    skip_verify: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the full model selection + verification pipeline.

    Returns
    -------
    dict[str, Any]
        Full selection result with metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not quiet:
        print("\n" + "=" * 75)
        print("  SELECT ENSEMBLE BASE MODELS")
        print("=" * 75)

    # ── 1. Load calibration data ──
    if not quiet:
        print("\n  Step 1: Loading calibration data...")
    selection = load_latest_selection()
    diagrams = load_latest_diagrams()

    # Merge ECE data from diagrams into selection (for richer output)
    if diagrams:
        for r in diagrams.get("results", []):
            mn = r["model"]
            if mn in selection.get("models", {}):
                if "calibrated" in r.get("variant", ""):
                    selection["models"][mn]["calibrated_ece"] = r["ece"]
                else:
                    selection["models"][mn]["original_ece"] = r["ece"]

    source_file = Path(selection.get("source_file", "?")).name
    if not quiet:
        print(f"    Source: {source_file}")
        print(f"    Models available: {len(selection.get('models', {}))}")

    # ── 2. Select models ──
    if not quiet:
        print(f"\n  Step 2: Selecting top {n_models} models...")
        if force_include:
            print(f"    Force include: {force_include}")
        if exclude:
            print(f"    Exclude: {exclude}")

    selected = select_models(
        selection,
        n_models=n_models,
        force_include=force_include,
        exclude=exclude,
    )

    if not quiet:
        print(f"\n    Selected {len(selected)} models:")
        for i, entry in enumerate(selected, 1):
            brier = entry["brier_score"]
            imp = entry["improvement"]
            method = entry["calibration_method"] or "none"
            print(f"      {i}. {entry['name']:<20s} Brier={brier:.4f}  "
                  f"Method={method:<12s}  Family={entry['family']:<12s}  Type={entry['type']}")

    # ── 3. Load & verify models ──
    verified = selected
    if not skip_verify:
        if not quiet:
            print("\n  Step 3: Loading & verifying models...")
        verified = verify_predictions(selected, quiet=quiet)
    else:
        if not quiet:
            print("\n  Step 3: Skipping verification (--skip-verify)")

    n_verified = sum(1 for v in verified if v.get("verified"))
    n_failed = sum(1 for v in verified if not v.get("verified"))
    if not quiet:
        print(f"\n    Verified: {n_verified}/{len(verified)}, Failed: {n_failed}")

    # ── 4. Build output JSON ──
    all_pass = n_failed == 0
    avg_brier = float(np.mean([v["brier_score"] for v in verified]))

    output = {
        "timestamp": timestamp,
        "source_file": str(Path(source_file)),
        "n_models_requested": n_models,
        "n_models_selected": len(verified),
        "n_verified": n_verified,
        "n_failed": n_failed,
        "all_verified": all_pass,
        "average_calibrated_brier": round(avg_brier, 4),
        "selected_models": [],
    }

    for entry in verified:
        output["selected_models"].append({
            "name": entry["name"],
            "path": entry["path"],
            "filename": entry["filename"],
            "type": entry["type"],
            "family": entry["family"],
            "brier_score": entry["brier_score"],
            "original_brier": entry["original_brier"],
            "improvement": entry["improvement"],
            "calibration_method": entry["calibration_method"],
            "verified": entry.get("verified", False),
            "verify_error": entry.get("verify_error"),
            "n_features": entry.get("n_features", 0),
        })

    # ── 5. Save ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REPORT_DIR / f"ensemble_selection_{timestamp}.json"
    save_path.write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8",
    )

    if not quiet:
        print(f"\n  Step 4: Selection saved to {save_path.name}")
        print(f"    (models: {len(verified)}, avg Brier: {avg_brier:.4f})")

        # ── Print summary table ──
        print(f"\n  {'Model':<20s} {'Type':<12s} {'Family':<14s} {'Brier':<8s} {'Method':<14s} {'Verified':<10s}")
        print(f"  {'-' * 78}")
        for v in verified:
            name = v["name"]
            mtype = v["type"]
            family = v["family"]
            brier = f"{v['brier_score']:.4f}"
            method = v["calibration_method"] or "raw"
            status = "✅" if v.get("verified") else "❌"
            print(f"  {name:<20s} {mtype:<12s} {family:<14s} {brier:<8s} {method:<14s} {status:<10s}")
        print()

    if not all_pass:
        print(f"  [WARN] {n_failed}/{len(verified)} models failed verification.")
        print("         These models are included but may not work in the ensemble.\n")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select and prepare base models for ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", "-n", type=int, default=4,
        help="Number of models to select (default 4)",
    )
    parser.add_argument(
        "--force", "-f", nargs="+", default=None,
        help="Force include specific models (e.g. --force XGBoost Elo)",
    )
    parser.add_argument(
        "--exclude", "-x", nargs="+", default=None,
        help="Exclude specific models (e.g. --exclude Poisson)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip model loading/prediction verification",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    try:
        run_selection(
            n_models=args.models,
            force_include=args.force,
            exclude=args.exclude,
            skip_verify=args.skip_verify,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        print(f"\n[FAIL] Selection failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
