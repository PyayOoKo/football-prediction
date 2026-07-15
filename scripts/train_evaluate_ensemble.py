#!/usr/bin/env python3
"""
Train & Evaluate Ensemble — build a WeightedEnsemble from selected calibrated
models, evaluate on a held-out validation set, compare against each base model,
and save the results to reports/ensemble_validation_{timestamp}.json.

Usage:
    python scripts/train_evaluate_ensemble.py
    python scripts/train_evaluate_ensemble.py --quiet
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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "reports"


# ═══════════════════════════════════════════════════════════
#  Load helpers
# ═══════════════════════════════════════════════════════════


def load_json(pattern: str, exclude_pattern: str | None = None) -> dict[str, Any]:
    """Load the latest JSON file matching *pattern*."""
    files = sorted(REPORT_DIR.glob(pattern))
    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in f.stem]
    if not files:
        print(f"  [FAIL] No files matching '{pattern}' found.")
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_model(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a calibrated model file and return (model, metadata)."""
    import joblib

    payload = joblib.load(path)
    model = payload.get("model", payload)
    metadata = payload.get("metadata", {})
    return model, metadata


# ═══════════════════════════════════════════════════════════
#  Metrics computation
# ═══════════════════════════════════════════════════════════


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    df_raw: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute all required metrics.

    Parameters
    ----------
    y_true : (n,) array of {0, 1, 2}
        True labels (Away, Draw, Home).
    probs : (n, 3) array
        Predicted probabilities [away, draw, home].
    df_raw : pd.DataFrame, optional
        Raw match data with home/away goals for BTTS and O/U metrics.

    Returns
    -------
    dict with keys: brier_score, log_loss, accuracy, btts_accuracy, over25_accuracy
    """
    from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

    n = len(y_true)
    if n == 0:
        return {"brier_score": 0.0, "log_loss": 0.0, "accuracy": 0.0,
                "btts_accuracy": 0.0, "over25_accuracy": 0.0}

    # ── Log loss ──
    logloss = float(sk_log_loss(y_true, probs))

    # ── Accuracy (1X2) ──
    pred_class = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred_class == y_true))

    # ── Multi-class Brier score ──
    y_onehot = np.zeros((n, 3))
    for i, v in enumerate(y_true):
        if 0 <= int(v) <= 2:
            y_onehot[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # ── BTTS accuracy (if raw data available) ──
    btts_acc = 0.0
    if df_raw is not None and "home_goals" in df_raw.columns and "away_goals" in df_raw.columns:
        # Actual BTTS
        actual_btts = ((df_raw["home_goals"].fillna(0) > 0) & (df_raw["away_goals"].fillna(0) > 0)).astype(int).values
        # Predicted BTTS: home > 0 AND away > 0 → product of home_nonzero and away_nonzero
        # BTTS_prob = 1 - P(home=0) - P(away=0) + P(both=0)
        # For Poisson-based predictions, this is computed from scoreline table.
        # For ML-based prob outputs, we use an approximation: P(goals=0) ≈ exp(-expected_goals)
        # Infer expected goals from win/draw probabilities:
        # Higher win prob → higher expected goals
        p_home = probs[:, 2]
        p_draw = probs[:, 1]
        p_away = probs[:, 0]
        # Approximate expected goals from probabilities (rough mapping)
        exp_home = np.clip(1.0 + p_home * 1.5 - p_away * 0.5, 0.1, 5.0)
        exp_away = np.clip(1.0 + p_away * 1.5 - p_home * 0.5, 0.1, 5.0)
        p_h0 = np.exp(-exp_home)
        p_a0 = np.exp(-exp_away)
        btts_probs = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)
        pred_btts = (btts_probs > 0.5).astype(int)
        btts_acc = float(np.mean(pred_btts == actual_btts))

    # ── Over 2.5 accuracy (if raw data available) ──
    over_acc = 0.0
    if df_raw is not None and "home_goals" in df_raw.columns and "away_goals" in df_raw.columns:
        actual_total = df_raw["home_goals"].fillna(0).values + df_raw["away_goals"].fillna(0).values
        actual_ou = (actual_total > 2.5).astype(int)
        # Approximate over 2.5 probability from total expected goals
        exp_total = exp_home + exp_away
        # Poisson probability of >2.5 goals
        from scipy.stats import poisson
        over_probs = 1.0 - poisson.cdf(2, exp_total)
        pred_ou = (over_probs > 0.5).astype(int)
        over_acc = float(np.mean(pred_ou == actual_ou))

    return {
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "accuracy": round(accuracy, 4),
        "btts_accuracy": round(btts_acc, 4),
        "over25_accuracy": round(over_acc, 4),
    }


# ═══════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════


def run_training(
    n_val: int = 200,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train and evaluate the ensemble, comparing against all base models.

    Parameters
    ----------
    n_val : int
        Number of most-recent rows to reserve for validation (default 200).
    quiet : bool
        Suppress console output.

    Returns
    -------
    dict with all results.
    """
    from src.data_loader import load_results
    from src.feature_engineering import build_features
    from src.ensemble import WeightedEnsemble

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not quiet:
        print("\n" + "=" * 75)
        print("  TRAIN & EVALUATE ENSEMBLE")
        print("=" * 75)

    # ── 1. Load configuration ──
    if not quiet:
        print("\n  Step 1: Loading configuration...")

    selection = load_json("ensemble_selection_*.json", exclude_pattern="_meta")
    weights_data = load_json("ensemble_weights_*.json", exclude_pattern="_meta")

    if not selection or not weights_data:
        print("  [FAIL] Missing ensemble_selection or ensemble_weights. Run select_ensemble_models.py then compute_ensemble_weights.py first.")
        sys.exit(1)

    selected_models = selection.get("selected_models", [])
    ensemble_weights = {k: v for k, v in weights_data.items() if isinstance(v, (int, float))}

    if not selected_models:
        print("  [FAIL] No selected_models found.")
        sys.exit(1)

    if not quiet:
        print(f"    Selection source: {Path(selection.get('source_file', '?')).name}")
        print(f"    Models: {len(selected_models)}")
        for m in selected_models:
            print(f"      {m['name']:<20s} weight={ensemble_weights.get(m['name'], 0):.4f}")

    # ── 2. Load models ──
    if not quiet:
        print("\n  Step 2: Loading models...")

    loaded_entries: list[dict[str, Any]] = []
    for entry in selected_models:
        name = entry["name"]
        path = entry["path"]
        weight = ensemble_weights.get(name, 0.25)

        if not Path(path).exists():
            if not quiet:
                print(f"    [SKIP] {name}: file not found at {path}")
            continue

        try:
            model, meta = load_model(path)
            loaded_entries.append({
                "name": name,
                "model": model,
                "weight": weight,
                "type": entry["type"],
                "brier_score": entry["brier_score"],
            })
            if not quiet:
                print(f"    [OK]   {name:<20s} weight={weight:.4f}")
        except Exception as e:
            if not quiet:
                print(f"    [FAIL] {name}: {e}")
            continue

    if len(loaded_entries) < 2:
        print(f"  [FAIL] Need at least 2 models, got {len(loaded_entries)}.")
        sys.exit(1)

    # ── 3. Load validation data ──
    if not quiet:
        print(f"\n  Step 3: Loading validation data (last {n_val} matches)...")

    df = load_results(low_memory=False)
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)

    # Validation set: last N rows (chronological), drop rows with unknown targets
    df_val = df.tail(n_val * 2).copy()  # grab extra to account for -1 filtering
    df_val = df_val[df_val["target"] >= 0].head(n_val).reset_index(drop=True)
    y_val = df_val["target"].values.astype(int)

    actual_n = len(df_val)
    if not quiet:
        print(f"    Validation rows requested: {n_val}, actual: {actual_n}")
        print(f"    Class distribution: Home={np.mean(y_val == 2):.3f}, "
              f"Draw={np.mean(y_val == 1):.3f}, Away={np.mean(y_val == 0):.3f}")

    # ── 4. Build feature matrix (use larger context for rolling stats) ──
    if not quiet:
        print("\n  Step 4: Building features...")

    try:
        # Use last 500 + n_val rows so rolling/H2H features populate for validation set
        context_rows = 500 + n_val
        df_context = df.tail(context_rows).copy()
        X_full, _ = build_features(df_context)
        # Keep as DataFrame with column names — models need matching feature names
        X_val = X_full.tail(actual_n).copy()
        if not quiet:
            print(f"    Feature matrix: {X_full.shape} ({len(X_full.columns)} cols)")
            print(f"    Validation features: {X_val.shape}")
    except Exception as e:
        if not quiet:
            print(f"    [WARN] build_features failed: {e}")
            print("    Falling back to raw DataFrame features.")
        X_val = df_val

    n_features = X_val.shape[1] if hasattr(X_val, "shape") and len(X_val.shape) >= 2 else 0

    # ── 5. Evaluate each individual model ──
    if not quiet:
        print("\n  Step 5: Evaluating individual models...")

    individual_results: dict[str, dict[str, float]] = {}
    for entry in loaded_entries:
        name = entry["name"]
        model = entry["model"]
        mtype = entry["type"]

        try:
            if mtype == "ML":
                probs = model.predict_proba(X_val)
            else:
                # Phase 3: use predict_matches on raw df, or predict_proba(df)
                if hasattr(model, "predict_matches"):
                    preds_df = model.predict_matches(df_val)
                    probs = np.column_stack([
                        preds_df["away_win_prob"].values,
                        preds_df["draw_prob"].values,
                        preds_df["home_win_prob"].values,
                    ])
                elif hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df_val)
                else:
                    if not quiet:
                        print(f"    [SKIP] {name}: no predict method")
                    continue

            probs = np.asarray(probs, dtype=np.float64)
            metrics = compute_metrics(y_val, probs, df_raw=df_val)
            individual_results[name] = metrics

            if not quiet:
                b = metrics["brier_score"]
                ll = metrics["log_loss"]
                a = metrics["accuracy"]
                print(f"    {name:<20s} Brier={b:.4f}  LogLoss={ll:.4f}  Acc={a:.4f}")

        except Exception as e:
            if not quiet:
                print(f"    [FAIL] {name}: {e}")

    # ── 6. Build and evaluate ensemble ──
    if not quiet:
        print("\n  Step 6: Building and evaluating ensemble...")

    ensemble = WeightedEnsemble(name="validation_ensemble")
    for entry in loaded_entries:
        ensemble.add_model(entry["model"], weight=entry["weight"])

    # Evaluate ensemble (uses predict_proba for ML, predict_matches for Statistical)
    try:
        ensemble_probs = ensemble.predict_proba(
            X_val,
            df_raw=df_val,
        )
        ensemble_metrics = compute_metrics(y_val, ensemble_probs, df_raw=df_val)
    except Exception as e:
        if not quiet:
            print(f"    [FAIL] Ensemble prediction: {e}")
        # Fallback: average individual model probabilities
        all_probs = []
        for entry in loaded_entries:
            name = entry["name"]
            if name in individual_results:
                model = entry["model"]
                mtype = entry["type"]
                if mtype == "ML":
                    p = model.predict_proba(X_val)
                else:
                    if hasattr(model, "predict_matches"):
                        preds_df = model.predict_matches(df_val)
                        p = np.column_stack([
                            preds_df["away_win_prob"].values,
                            preds_df["draw_prob"].values,
                            preds_df["home_win_prob"].values,
                        ])
                    else:
                        p = model.predict_proba(df_val)
                all_probs.append(np.asarray(p, dtype=np.float64) * entry["weight"])
        if all_probs:
            ensemble_probs = np.sum(all_probs, axis=0)
            row_sums = ensemble_probs.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            ensemble_probs = ensemble_probs / row_sums
            ensemble_metrics = compute_metrics(y_val, ensemble_probs, df_raw=df_val)
        else:
            ensemble_metrics = {"brier_score": 0, "log_loss": 0, "accuracy": 0,
                                "btts_accuracy": 0, "over25_accuracy": 0}

    if not quiet:
        b = ensemble_metrics["brier_score"]
        ll = ensemble_metrics["log_loss"]
        a = ensemble_metrics["accuracy"]
        print(f"    {'Ensemble (weighted)':<20s} Brier={b:.4f}  LogLoss={ll:.4f}  Acc={a:.4f}")

    # ── 7. Also evaluate equal-weight ensemble for comparison ──
    if not quiet:
        print("    Building equal-weight ensemble for comparison...")

    equal_ensemble = WeightedEnsemble(name="equal_ensemble")
    for entry in loaded_entries:
        equal_ensemble.add_model(entry["model"], weight=1.0)  # will be normalized

    try:
        equal_probs = equal_ensemble.predict_proba(
            X_val,
            df_raw=df_val,
        )
        equal_metrics = compute_metrics(y_val, equal_probs, df_raw=df_val)
    except Exception as e:
        if not quiet:
            print(f"    [FAIL] Equal-weight ensemble: {e}")
        equal_metrics = {"brier_score": 0, "log_loss": 0, "accuracy": 0,
                         "btts_accuracy": 0, "over25_accuracy": 0}

    if not quiet:
        b = equal_metrics["brier_score"]
        ll = equal_metrics["log_loss"]
        print(f"    {'Ensemble (equal)':<20s} Brier={b:.4f}  LogLoss={ll:.4f}  Acc={a:.4f}")

    # ── 8. Build output ──
    if not quiet:
        print(f"\n  Step 7: Building output...")

    # Find best single model
    best_single_name = min(individual_results, key=lambda n: individual_results[n]["brier_score"])
    best_single = individual_results[best_single_name]

    output = {
        "timestamp": timestamp,
        "n_validation": n_val,
        "n_features": n_features,
        "n_models": len(loaded_entries),
        "best_single_model": best_single_name,
        "ensemble_weights": {e["name"]: e["weight"] for e in loaded_entries},
        "ensemble_metrics": ensemble_metrics,
        "equal_weight_metrics": equal_metrics,
        "improvement_vs_best_single": {
            "brier_score": round(best_single["brier_score"] - ensemble_metrics["brier_score"], 4),
        },
        "individual_models": individual_results,
        "comparison_table": [],
    }

    # Build comparison table
    all_models = list(individual_results.keys()) + ["Ensemble (weighted)", "Ensemble (equal)"]
    all_metrics = dict(individual_results)
    all_metrics["Ensemble (weighted)"] = ensemble_metrics
    all_metrics["Ensemble (equal)"] = equal_metrics

    for name in all_models:
        m = all_metrics.get(name, {})
        output["comparison_table"].append({
            "model": name,
            "brier_score": m.get("brier_score", 0),
            "log_loss": m.get("log_loss", 0),
            "accuracy": m.get("accuracy", 0),
            "btts_accuracy": m.get("btts_accuracy", 0),
            "over25_accuracy": m.get("over25_accuracy", 0),
        })

    # ── 9. Save ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REPORT_DIR / f"ensemble_validation_{timestamp}.json"
    save_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    if not quiet:
        print(f"    Saved: {save_path.name}")

    # ── 10. Print summary table ──
    if not quiet:
        print(f"\n  {'─' * 85}")
        print(f"  {'Model':<22s} {'Brier':<8s} {'LogLoss':<10s} {'Accuracy':<10s} {'BTTS':<8s} {'O/U2.5':<8s}")
        print(f"  {'─' * 85}")
        for row in output["comparison_table"]:
            name = row["model"]
            b = row["brier_score"]
            ll = row["log_loss"]
            a = row["accuracy"]
            ba = row["btts_accuracy"]
            oa = row["over25_accuracy"]
            marker = "  ← best" if name == best_single_name else ""
            ens_marker = "  ← ensemble" if "Ensemble" in name else ""
            suffix = marker or ens_marker
            print(f"  {name:<22s} {b:<8.4f} {ll:<10.4f} {a:<10.4f} {ba:<8.4f} {oa:<8.4f}{suffix}")
        print(f"  {'─' * 85}")
        imp = output["improvement_vs_best_single"]["brier_score"]
        print(f"\n  Improvement vs best single ({best_single_name}): Δ={imp:+.4f} Brier")
        print()

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate ensemble model vs base models",
    )
    parser.add_argument(
        "--val-size", type=int, default=200,
        help="Number of most-recent matches for validation (default 200)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    try:
        run_training(n_val=args.val_size, quiet=args.quiet)
        return 0
    except Exception as e:
        print(f"\n[FAIL] Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
