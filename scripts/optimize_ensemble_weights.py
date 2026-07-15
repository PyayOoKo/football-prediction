#!/usr/bin/env python3
"""
Optimize Ensemble Weights — grid search over weight combinations to minimise
Brier Score on a validation set.  Saves optimised weights, retrains the
ensemble, and evaluates against base models.

Usage:
    python scripts/optimize_ensemble_weights.py
    python scripts/optimize_ensemble_weights.py --step 0.10
    python scripts/optimize_ensemble_weights.py --quick
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "reports"


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


def load_json(pattern: str, exclude: str | None = None) -> dict[str, Any]:
    files = sorted(REPORT_DIR.glob(pattern))
    if exclude:
        files = [f for f in files if exclude not in f.stem]
    if not files:
        print(f"  [FAIL] No files matching '{pattern}'")
        sys.exit(1)
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_model(path: str) -> tuple[Any, dict[str, Any]]:
    import joblib
    payload = joblib.load(path)
    model = payload.get("model", payload)
    metadata = payload.get("metadata", {})
    return model, metadata


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score."""
    n = len(y_true)
    y_onehot = np.zeros((n, 3))
    for i, v in enumerate(y_true):
        if 0 <= int(v) <= 2:
            y_onehot[i, int(v)] = 1
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


# ═══════════════════════════════════════════════════════════
#  Grid search
# ═══════════════════════════════════════════════════════════


def grid_search_brier(
    preds_list: list[np.ndarray],
    y_true: np.ndarray,
    n_models: int,
    step: float,
    model_names: list[str],
    max_weight: float = 1.0,
    quiet: bool = False,
) -> tuple[dict[str, float], float, int]:
    """Grid search over weight combinations minimising Brier score.

    Returns
    -------
    (best_weights_dict, best_brier, combinations_evaluated)
    """
    n_bins = int(round(max_weight / step))
    total_raw = (n_bins + 1) ** n_models

    # Coarse-to-fine guard
    MAX_COMBOS = 200_000
    if total_raw > MAX_COMBOS:
        n_bins = max(2, int(MAX_COMBOS ** (1.0 / max(n_models, 1))) - 1)
        if not quiet:
            print(f"    Capping grid to {n_bins+1}^{n_models} ≈ {(n_bins+1)**n_models} combos")

    if not quiet:
        print(f"    Grid: step={step}, bins={n_bins}, combinations≈{(n_bins+1)**n_models}")

    best_brier = float("inf")
    best_weights: list[float] = []
    seen: set[tuple[float, ...]] = set()
    total_combos = 0

    for raw in itertools.product(range(n_bins + 1), repeat=n_models):
        total = sum(raw)
        if total == 0:
            continue
        norm = tuple(r / total for r in raw)
        if norm in seen:
            continue
        seen.add(norm)
        total_combos += 1

        # Weighted average
        weighted = np.zeros_like(preds_list[0])
        for i, probs in enumerate(preds_list):
            weighted += norm[i] * probs
        row_sums = weighted.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        weighted = weighted / row_sums

        brier = brier_score(y_true, weighted)
        if brier < best_brier:
            best_brier = brier
            best_weights = list(norm)

    best_dict = {model_names[i]: round(best_weights[i], 4) for i in range(n_models)}

    # Adjust last weight so sum is exactly 1.0
    diff = round(1.0 - sum(best_dict.values()), 4)
    if diff != 0:
        last_key = list(best_dict.keys())[-1]
        best_dict[last_key] = round(best_dict[last_key] + diff, 4)

    return best_dict, best_brier, total_combos


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def run_optimization(
    n_val: int = 200,
    step: float = 0.05,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run weight optimisation + evaluation."""
    from src.data_loader import load_results
    from src.feature_engineering import build_features
    from src.ensemble import WeightedEnsemble

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not quiet:
        print("\n" + "=" * 75)
        print("  OPTIMISE ENSEMBLE WEIGHTS")
        print("=" * 75)

    # ── 1. Load config ──
    if not quiet:
        print("\n  Step 1: Loading configuration...")

    selection = load_json("ensemble_selection_*.json", exclude="_meta")
    initial_weights_data = load_json("ensemble_weights_*.json", exclude="_meta")
    initial_weights = {k: v for k, v in initial_weights_data.items() if isinstance(v, (int, float))}

    selected_models = selection.get("selected_models", [])
    if not selected_models:
        print("[FAIL] No selected_models in ensemble_selection")
        sys.exit(1)

    if not quiet:
        print(f"    Models: {len(selected_models)}")
        for m in selected_models:
            iw = initial_weights.get(m["name"], "?")
            print(f"      {m['name']:<20s} init_weight={iw}")

    # ── 2. Load models ──
    if not quiet:
        print("\n  Step 2: Loading models...")

    loaded: list[dict[str, Any]] = []
    for entry in selected_models:
        path = entry["path"]
        if not Path(path).exists():
            if not quiet:
                print(f"    [SKIP] {entry['name']}: file not found")
            continue
        try:
            model, _ = load_model(path)
            loaded.append({
                "name": entry["name"],
                "model": model,
                "type": entry["type"],
                "initial_weight": initial_weights.get(entry["name"], 0.25),
            })
            if not quiet:
                print(f"    [OK]   {entry['name']}")
        except Exception as e:
            if not quiet:
                print(f"    [FAIL] {entry['name']}: {e}")

    if len(loaded) < 2:
        print(f"[FAIL] Need ≥2 models, got {len(loaded)}")
        sys.exit(1)

    # ── 3. Load validation data ──
    if not quiet:
        print(f"\n  Step 3: Loading validation data (last ~{n_val} matches)...")

    df = load_results(low_memory=False)
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)

    # Filter out unresolved targets
    buffer = 500 + n_val * 2
    df_context = df.tail(buffer).copy()
    df_raw = df.tail(n_val * 2).copy()
    df_raw = df_raw[df_raw["target"] >= 0].head(n_val).reset_index(drop=True)
    y_val = df_raw["target"].values.astype(int)
    actual_n = len(df_raw)

    if not quiet:
        print(f"    Validation rows: {actual_n}/{n_val}")
        print(f"    Classes: Home={np.mean(y_val==2):.3f} Draw={np.mean(y_val==1):.3f} Away={np.mean(y_val==0):.3f}")

    # ── 4. Build feature matrix ──
    if not quiet:
        print("\n  Step 4: Building features...")

    try:
        X_full, _ = build_features(df_context)
        X_val = X_full.tail(actual_n).copy()
        if not quiet:
            print(f"    Feature matrix: {X_full.shape} ({len(X_full.columns)} cols)")
    except Exception as e:
        if not quiet:
            print(f"    [WARN] build_features failed: {e}")
            print("    Falling back to raw DataFrame.")
        X_val = df_raw

    # ── 5. Get individual predictions ──
    if not quiet:
        print("\n  Step 5: Getting individual predictions...")

    preds_list: list[np.ndarray] = []
    model_names: list[str] = []
    individual_briers: dict[str, float] = {}

    for entry in loaded:
        name = entry["name"]
        model = entry["model"]
        mtype = entry["type"]
        model_names.append(name)

        try:
            if mtype == "ML":
                probs = np.asarray(model.predict_proba(X_val), dtype=np.float64)
            else:
                if hasattr(model, "predict_matches"):
                    preds_df = model.predict_matches(df_raw)
                    probs = np.column_stack([
                        preds_df["away_win_prob"].values,
                        preds_df["draw_prob"].values,
                        preds_df["home_win_prob"].values,
                    ])
                else:
                    probs = np.asarray(model.predict_proba(df_raw), dtype=np.float64)

            preds_list.append(probs)
            brier = brier_score(y_val, probs)
            individual_briers[name] = round(brier, 4)
            if not quiet:
                print(f"    {name:<20s} Brier={brier:.4f}")
        except Exception as e:
            if not quiet:
                print(f"    [FAIL] {name}: {e}")
            continue

    if len(preds_list) < 2:
        print("[FAIL] Need ≥2 successful predictions")
        sys.exit(1)

    # ── 6. Grid search for optimal weights ──
    if not quiet:
        print(f"\n  Step 6: Grid search (step={step})...")

    best_weights, best_brier, n_combos = grid_search_brier(
        preds_list, y_val, len(preds_list), step, model_names,
    )

    if not quiet:
        print(f"    Best Brier: {best_brier:.4f}")
        for name, w in best_weights.items():
            print(f"      {name:<20s} weight={w:.4f}")
        print(f"    Combinations evaluated: {n_combos}")

    # ── 7. Evaluate optimised ensemble ──
    if not quiet:
        print("\n  Step 7: Evaluating optimised ensemble...")

    opt_ensemble = WeightedEnsemble(name="optimised")
    for entry in loaded:
        w = best_weights.get(entry["name"], 0.0)
        opt_ensemble.add_model(entry["model"], weight=w)

    try:
        opt_probs = opt_ensemble.predict_proba(X_val, df_raw=df_raw)
        opt_brier = brier_score(y_val, opt_probs)
        # Also compute log-loss and accuracy
        from sklearn.metrics import log_loss as sk_ll
        opt_logloss = float(sk_ll(y_val, opt_probs))
        opt_accuracy = float(np.mean(np.argmax(opt_probs, axis=1) == y_val))
    except Exception as e:
        if not quiet:
            print(f"    [FAIL] Ensemble prediction: {e}")
        # Fallback: manual weighted average
        weighted = np.zeros_like(preds_list[0])
        for i, probs in enumerate(preds_list):
            w = best_weights.get(model_names[i], 0.0)
            weighted += w * probs
        row_sums = weighted.sum(axis=1, keepdims=True)
        weighted = weighted / np.where(row_sums > 0, row_sums, 1.0)
        opt_probs = weighted
        opt_brier = brier_score(y_val, opt_probs)
        from sklearn.metrics import log_loss as sk_ll
        opt_logloss = float(sk_ll(y_val, opt_probs))
        opt_accuracy = float(np.mean(np.argmax(opt_probs, axis=1) == y_val))

    if not quiet:
        print(f"    Optimised Brier:  {opt_brier:.4f}")
        print(f"    LogLoss:          {opt_logloss:.4f}")
        print(f"    Accuracy:         {opt_accuracy:.4f}")

    # ── 8. Compare with initial weights ──
    init_ensemble = WeightedEnsemble(name="initial")
    for entry in loaded:
        w = entry["initial_weight"]
        init_ensemble.add_model(entry["model"], weight=w)

    try:
        init_probs = init_ensemble.predict_proba(X_val, df_raw=df_raw)
        init_brier = brier_score(y_val, init_probs)
    except Exception:
        init_brier = 0.0

    best_single_name = min(individual_briers, key=individual_briers.get)
    best_single_brier = individual_briers[best_single_name]

    if not quiet:
        print(f"\n  {'─' * 50}")
        print(f"  {'Metric':<25s} {'Value'}")
        print(f"  {'─' * 50}")
        print(f"  {'Best single model':<25s} {best_single_name} ({best_single_brier:.4f})")
        print(f"  {'Initial weights Brier':<25s} {init_brier:.4f}")
        print(f"  {'Optimised weights Brier':<25s} {opt_brier:.4f}")
        print(f"  {'Improvement vs best single':<25s} {best_single_brier - opt_brier:+.4f}")
        print(f"  {'Improvement vs initial':<25s} {init_brier - opt_brier:+.4f}")
        print(f"  {'─' * 50}")

    # ── 9. Save ──
    output_weights = {name: best_weights.get(name, 0.0) for name in model_names}
    output = {
        "timestamp": timestamp,
        "method": "grid_search_brier",
        "grid_step": step,
        "combinations_evaluated": n_combos,
        "n_validation": actual_n,
        "best_single_model": {"name": best_single_name, "brier": best_single_brier},
        "initial_weights": {e["name"]: e["initial_weight"] for e in loaded},
        "initial_brier": round(init_brier, 4),
        "optimised_weights": output_weights,
        "optimised_brier": round(opt_brier, 4),
        "optimised_log_loss": round(opt_logloss, 4),
        "optimised_accuracy": round(opt_accuracy, 4),
        "improvement_vs_best_single": round(best_single_brier - opt_brier, 4),
        "improvement_vs_initial": round(init_brier - opt_brier, 4),
        "individual_briers": individual_briers,
    }

    # Save weights as flat dict (spec format)
    weights_path = REPORT_DIR / f"ensemble_weights_optimised_{timestamp}.json"
    weights_path.write_text(json.dumps(output_weights, indent=2), encoding="utf-8")

    # Save full report
    report_path = REPORT_DIR / f"ensemble_optimisation_{timestamp}.json"
    report_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    if not quiet:
        print(f"\n  Optimised weights saved: {weights_path.name}")
        print(f"  Full report saved:       {report_path.name}")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimise ensemble weights via grid search")
    parser.add_argument("--val-size", type=int, default=200, help="Validation set size (default 200)")
    parser.add_argument("--step", type=float, default=0.05, help="Grid step size (default 0.05)")
    parser.add_argument("--quick", action="store_true", help="Quick run: step=0.10, val_size=100")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    if args.quick:
        args.step = 0.10
        args.val_size = 100

    try:
        run_optimization(n_val=args.val_size, step=args.step, quiet=args.quiet)
        return 0
    except Exception as e:
        print(f"\n[FAIL] Optimisation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
