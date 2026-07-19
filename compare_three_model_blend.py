"""
compare_three_model_blend.py — Compare 3-model blend vs current ensemble.

Evaluates both models on a held-out test set across all 4 betting markets
(1X2, Over2.5, BTTS, Over3.5) using Brier Score, Log Loss, Accuracy,
and (if odds available) ROI and CLV.

Usage:
    python compare_three_model_blend.py
    python compare_three_model_blend.py --test-split 0.15
"""

from __future__ import annotations

import argparse
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


# ── Metrics ───────────────────────────────────────────────

def brier_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    y_v, p_v = y_true[valid], probs[valid]
    y_oh = np.zeros_like(p_v)
    for i, v in enumerate(y_v):
        if 0 <= int(v) <= 2:
            y_oh[i, int(v)] = 1
    return round(float(np.mean(np.sum((p_v - y_oh) ** 2, axis=1))), 5)


def brier_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    return round(float(np.mean((probs[valid] - y_true[valid]) ** 2)), 5)


def log_loss_1x2(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import log_loss as sk_ll
        valid = ~np.isnan(y_true)
        y_v, p_v = y_true[valid], probs[valid]
        return round(float(sk_ll(y_v, p_v)), 5)
    except Exception:
        return None


def log_loss_binary(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import log_loss as sk_ll
        valid = ~np.isnan(y_true)
        p_v = np.clip(probs[valid], 1e-15, 1 - 1e-15)
        y_v = y_true[valid]
        return round(float(sk_ll(y_v, np.column_stack([1 - p_v, p_v]))), 5)
    except Exception:
        return None


def accuracy_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    preds = np.argmax(probs[valid], axis=1)
    return round(float(np.mean(preds == y_true[valid])), 5)


def accuracy_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    preds = (probs[valid] > 0.5).astype(float)
    return round(float(np.mean(preds == y_true[valid])), 5)


def compute_all_1x2_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "brier_score": brier_1x2(y_true, probs),
        "log_loss": log_loss_1x2(y_true, probs),
        "accuracy": accuracy_1x2(y_true, probs),
        "n_samples": int((~np.isnan(y_true)).sum()),
    }


def compute_all_binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    return {
        "brier_score": brier_binary(y_true, probs),
        "log_loss": log_loss_binary(y_true, probs),
        "accuracy": accuracy_binary(y_true, probs),
        "n_samples": int((~np.isnan(y_true)).sum()),
    }


def compute_roi(y_true: np.ndarray, probs: np.ndarray,
                odds_h: np.ndarray | None, odds_d: np.ndarray | None, odds_a: np.ndarray | None) -> float | None:
    """Compute ROI from 1X2 predictions if odds are available."""
    if odds_h is None or odds_d is None or odds_a is None:
        return None
    try:
        valid = ~np.isnan(y_true)
        y_v = y_true[valid]
        p_v = probs[valid]
        preds = np.argmax(p_v, axis=1)
        # Align odds arrays with valid mask
        oa_v, od_v, oh_v = odds_a[valid], odds_d[valid], odds_h[valid]
        odds_list = [oa_v, od_v, oh_v]  # [away, draw, home]
        bankroll = 100.0
        bets = 0
        for i in range(len(y_v)):
            pred = preds[i]
            prob = p_v[i, pred]
            implied = 1.0 / odds_list[pred][i]
            if prob > implied and prob > 0.40:
                stake = 1.0
                if pred == y_v[i]:
                    bankroll += stake * (odds_list[pred][i] - 1)
                else:
                    bankroll -= stake
                bets += 1
        if bets == 0:
            return 0.0
        roi = (bankroll - 100.0) / bets * 100
        return round(float(roi), 2)
    except Exception:
        return None


def compute_clv(probs: np.ndarray,
                odds_h: np.ndarray | None, odds_d: np.ndarray | None, odds_a: np.ndarray | None) -> float | None:
    """Compute Closing Line Value if odds are available.

    CLV = average of (model_prob - implied_prob) for predicted outcomes.
    """
    if odds_h is None or odds_d is None or odds_a is None:
        return None
    try:
        preds = np.argmax(probs, axis=1)
        odds_list = [odds_a, odds_d, odds_h]
        diffs = []
        for i in range(len(preds)):
            pred = preds[i]
            implied = 1.0 / odds_list[pred][i]
            diffs.append(probs[i, pred] - implied)
        return round(float(np.mean(diffs)), 5)
    except Exception:
        return None


# ── Model setup ──────────────────────────────────────────

def load_ensemble_model(ensemble_path: str | Path) -> Any | None:
    """Load and reconstruct the current ensemble model.

    The saved ensemble_model.joblib is a dict containing:
    - 'models': dict of sub-models
    - 'weights': dict of weights per model

    Uses WeightedEnsemble constructor with (model, weight) tuples to
    preserve original weighting, unlike add_model() which renormalises
    after each insertion.
    """
    import joblib

    path = Path(ensemble_path)
    if not path.exists():
        logger.warning("Ensemble model not found at %s", path)
        return None

    payload = joblib.load(path)
    if hasattr(payload, "predict_proba"):
        logger.info("Ensemble loaded as %s", type(payload).__name__)
        return payload  # Already a proper model object

    if not isinstance(payload, dict):
        logger.warning("Unexpected ensemble format: %s", type(payload))
        return None

    models = payload.get("models", {})
    weights = payload.get("weights", {})
    poisson_model = payload.get("poisson_model")

    logger.info("Ensemble sub-models: %s", list(models.keys()))
    logger.info("Ensemble weights: %s", weights)

    from src.ensemble import WeightedEnsemble

    # Build the (model, weight) tuple list for the constructor,
    # which preserves weights without renormalising after each add
    model_tuples = []
    for name, model in models.items():
        w = weights.get(name, 1.0)
        model_tuples.append((model, w))
    if poisson_model is not None and "poisson" not in models:
        pw = weights.get("poisson", 0.0)
        if pw > 0:
            model_tuples.append((poisson_model, pw))

    if not model_tuples:
        logger.error("No models found in ensemble payload")
        return None

    ensemble = WeightedEnsemble(model_tuples, name="Current Ensemble")
    logger.info("Ensemble loaded as WeightedEnsemble with %d members", len(model_tuples))
    return ensemble


def setup_three_model_blend(
    train_df: pd.DataFrame,
    cond_rates: Any,
    optimized_weights_path: str | Path | None = None,
) -> Any | None:
    """Set up the ThreeModelBlend with fitted models and optimised weights."""
    from src.poisson_model import PoissonModel
    from src.elo import EloSystem

    logger.info("Fitting Poisson on %d training matches...", len(train_df))
    poisson = PoissonModel(min_matches=0)
    poisson.fit(train_df)

    logger.info("Processing Elo on %d training matches...", len(train_df))
    elo = EloSystem()
    elo.process_matches(train_df)

    import joblib
    xgb = None
    for candidate in [
        PROJECT_ROOT / "models" / "xgboost_model.joblib",
        PROJECT_ROOT / "models" / "worldcup_lightgbm.joblib",
    ]:
        if candidate.exists():
            logger.info("Loading ML model from %s", candidate)
            xgb = joblib.load(candidate)
            break
    if xgb is None:
        logger.error("No ML model found for ThreeModelBlend")
        return None

    # Load optimised weights
    weights = None
    if optimized_weights_path:
        p = Path(optimized_weights_path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            weights = data.get("weights")
            logger.info("Loaded optimised weights from %s", p)

    from src.models.three_model_blend import ThreeModelBlend

    blend = ThreeModelBlend(
        poisson_model=poisson,
        elo_model=elo,
        xgb_model=xgb,
        weights=weights,
        conditional_rates=cond_rates,
        historical_df=train_df,
    )
    logger.info("ThreeModelBlend ready with %d markets", len(blend.available_markets))
    return blend


# ── Data loading ─────────────────────────────────────────

def load_data(test_split: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Load World Cup data and split into train/val/test chronologically.

    Returns
    -------
    tuple of (train_df, val_df, test_df, cond_rates)
    """
    data_path = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    val_split = int(n * (1 - 2 * test_split))
    test_split_idx = int(n * (1 - test_split))

    train_df = df.iloc[:val_split].copy()
    val_df = df.iloc[val_split:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    # Compute conditional rates from training data only (no leakage)
    from src.models.three_model_blend import ConditionalRates
    cond_rates = ConditionalRates.from_data(train_df)

    logger.info("Split: %d train + %d val + %d test = %d total",
                len(train_df), len(val_df), len(test_df), n)
    logger.info("  Train: %s to %s", train_df["date"].iloc[0], train_df["date"].iloc[-1])
    logger.info("  Val:   %s to %s", val_df["date"].iloc[0], val_df["date"].iloc[-1])
    logger.info("  Test:  %s to %s", test_df["date"].iloc[0], test_df["date"].iloc[-1])

    return train_df, val_df, test_df, cond_rates


# ── Ensemble predictions (compute once, reuse) ──────────

def compute_ensemble_predictions(ensemble: Any, test_df: pd.DataFrame,
                                 cond_rates: Any) -> dict[str, np.ndarray]:
    """Compute ALL ensemble predictions at once to avoid repeated feature engineering."""
    # 1X2 predictions (runs feature engineering once)
    try:
        from src.feature_engineering import build_features
        X_test, _ = build_features(test_df, is_training=False)
        probs_1x2 = ensemble.predict_proba(X_test, df_raw=test_df)
    except Exception as exc:
        logger.warning("Ensemble 1X2 failed: %s", exc)
        n = len(test_df)
        probs_1x2 = np.full((n, 3), 1.0 / 3.0)

    # Derive binary markets from 1X2 using training-only conditional rates
    btts = cond_rates.btts_from_1x2(probs_1x2)
    ou25 = cond_rates.ou_from_1x2(probs_1x2, 2.5)
    ou35 = cond_rates.ou_from_1x2(probs_1x2, 3.5)

    return {
        "1x2": probs_1x2,
        "btts": btts,
        "over_2_5": ou25,
        "over_3_5": ou35,
    }


def compute_blend_predictions(blend: Any, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute ALL ThreeModelBlend predictions via batch precomputation.

    Uses ``blend.precompute()`` to run Poisson, Elo, and XGBoost predictions
    ONCE in batch mode (XGBoost feature engineering runs once for all rows),
    then applies the blend's internal blending methods to get final predictions
    for each market.  This avoids the O(n * num_markets) overhead of row-by-row
    single-fixture prediction.
    """
    # Batch precompute: runs all models once (XGBoost feature eng. runs once)
    ppm = blend.precompute(test_df, cache_key="compare")

    # Get per-market weights from the blend
    w_1x2 = blend.weights.get("1X2", {})
    w_btts = blend.weights.get("BTTS", {})
    w_ou25 = blend.weights.get("Over2.5", {})
    w_ou35 = blend.weights.get("Over3.5", {})

    # Use internal blending methods (no additional model calls)
    probs_1x2 = blend._blend_1x2(ppm, w_1x2)
    probs_btts = blend._blend_binary(ppm, w_btts, "BTTS")
    probs_ou25 = blend._blend_binary(ppm, w_ou25, "Over2.5")
    probs_ou35 = blend._blend_binary(ppm, w_ou35, "Over3.5")

    return {
        "1x2": probs_1x2,
        "btts": probs_btts,
        "over_2_5": probs_ou25,
        "over_3_5": probs_ou35,
    }


# ── Main comparison ─────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare 3-model blend vs current ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Fraction of data for test set (default: 0.15)")
    parser.add_argument("--optimized-weights",
                        default=str(PROJECT_ROOT / "config" / "three_model_weights.json"),
                        help="Path to optimised weights JSON")
    parser.add_argument("--output", "-o", default=None,
                        help="Output report path (auto-generated if not provided)")
    args = parser.parse_args(argv)

    t_start = time.time()

    print()
    print("-" * 60)
    print("  3-MODEL BLEND vs CURRENT ENSEMBLE")
    print("-" * 60)

    # ── 1. Load data ──────────────────────────────────────
    print("\n-- Loading data ---------------------------------")
    train_df, val_df, test_df, cond_rates = load_data(args.test_split)

    # ── 2. Set up models ─────────────────────────────────
    print("\n-- Setting up models ---------------------------")

    print("  Loading ensemble model...")
    ensemble = load_ensemble_model(PROJECT_ROOT / "models" / "ensemble_model.joblib")
    if ensemble is None:
        logger.error("Ensemble model not available")
        return 1

    print("  Setting up ThreeModelBlend...")
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    blend = setup_three_model_blend(fit_df, cond_rates, args.optimized_weights)
    if blend is None:
        logger.error("ThreeModelBlend not available")
        return 1

    print(f"  Test set: {len(test_df)} matches")

    # ── 3. Check for odds columns ─────────────────────────
    odds_available = all(c in test_df.columns for c in ["BbAvH", "BbAvD", "BbAvA"])
    if not odds_available:
        odds_available = all(c in test_df.columns for c in ["avg_h", "avg_d", "avg_a"])
        if odds_available:
            odds_h = test_df["avg_h"].values.astype(float)
            odds_d = test_df["avg_d"].values.astype(float)
            odds_a = test_df["avg_a"].values.astype(float)
        else:
            odds_h = odds_d = odds_a = None
            print("  No odds columns found -- ROI/CLV will be skipped")
    else:
        odds_h = test_df["BbAvH"].values.astype(float)
        odds_d = test_df["BbAvD"].values.astype(float)
        odds_a = test_df["BbAvA"].values.astype(float)

    # ── 4. Compute predictions (once per model) ──────────
    print("\n-- Computing predictions ------------------------")
    print("  Computing ensemble predictions...")
    ens_preds = compute_ensemble_predictions(ensemble, test_df, cond_rates)
    print("  Computing ThreeModelBlend predictions...")
    blend_preds = compute_blend_predictions(blend, test_df)

    # ── 5. Actual outcomes ────────────────────────────────
    y_result = test_df["result"].map({"A": 0, "D": 1, "H": 2}).values
    hg = test_df["home_goals"].values.astype(float)
    ag = test_df["away_goals"].values.astype(float)
    y_btts = ((hg > 0) & (ag > 0)).astype(float)
    y_ou25 = ((hg + ag) > 2.5).astype(float)
    y_ou35 = ((hg + ag) > 3.5).astype(float)

    # ── 6. Compute metrics per market ────────────────────
    print("\n-- Results ---------------------------------------")

    markets = [
        ("1X2", y_result, ens_preds["1x2"], blend_preds["1x2"], "1x2"),
        ("Over2.5", y_ou25, ens_preds["over_2_5"], blend_preds["over_2_5"], "binary"),
        ("BTTS", y_btts, ens_preds["btts"], blend_preds["btts"], "binary"),
        ("Over3.5", y_ou35, ens_preds["over_3_5"], blend_preds["over_3_5"], "binary"),
    ]

    comparison = {}
    rows = []
    table_sep = "-" * 78

    print(f"  {table_sep}")
    header = f"  {'Market':<12} {'Metric':<14} {'Ensemble':>12} {'3-Blend':>12} {'Diff':>10} {'Winner':>10}"
    print(header)
    print(f"  {table_sep}")

    for name, y_true, ens_probs, blend_probs, mtype in markets:
        if mtype == "1x2":
            ens_metrics = compute_all_1x2_metrics(y_true, ens_probs)
            blend_metrics = compute_all_1x2_metrics(y_true, blend_probs)
        else:
            ens_metrics = compute_all_binary_metrics(y_true, ens_probs)
            blend_metrics = compute_all_binary_metrics(y_true, blend_probs)

        comparison[name] = {
            "ensemble": ens_metrics,
            "three_model_blend": blend_metrics,
        }

        # Add ROI/CLV for 1X2 if odds available
        if name == "1X2" and odds_available:
            roi_ens = compute_roi(y_true, ens_probs, odds_h, odds_d, odds_a)
            roi_blend = compute_roi(y_true, blend_probs, odds_h, odds_d, odds_a)
            clv_ens = compute_clv(ens_probs, odds_h, odds_d, odds_a)
            clv_blend = compute_clv(blend_probs, odds_h, odds_d, odds_a)
            comparison[name]["ensemble"]["roi"] = roi_ens
            comparison[name]["three_model_blend"]["roi"] = roi_blend
            comparison[name]["ensemble"]["clv"] = clv_ens
            comparison[name]["three_model_blend"]["clv"] = clv_blend

        for metric_name in ["brier_score", "log_loss", "accuracy", "n_samples"]:
            if metric_name == "n_samples":
                n_val = ens_metrics.get("n_samples", "?")
                print(f"  {name:<12} {'samples':<14} {n_val:>12} {n_val:>12} {'-':>10} {'-':>10}")
                continue

            e_val = ens_metrics.get(metric_name)
            b_val = blend_metrics.get(metric_name)
            if e_val is None or b_val is None:
                continue

            diff = b_val - e_val
            lower_better = metric_name in ("brier_score", "log_loss")
            winner = "Blend" if (diff < 0 and lower_better) or (diff > 0 and not lower_better) else "Ensemble"

            diff_str = f"{diff:+.5f}" if abs(diff) >= 0.00001 else " -    "
            print(f"  {name:<12} {metric_name:<14} {e_val:>12.5f} {b_val:>12.5f} {diff_str:>10} {winner:>10}")

            rows.append({
                "market": name,
                "metric": metric_name,
                "ensemble": e_val,
                "three_model_blend": b_val,
                "diff": round(diff, 5),
                "winner": winner,
            })

        # Print ROI/CLV if available
        if name == "1X2" and odds_available:
            for m_name in ["roi", "clv"]:
                ev = comparison[name]["ensemble"].get(m_name)
                bv = comparison[name]["three_model_blend"].get(m_name)
                if ev is not None and bv is not None:
                    diff_v = bv - ev if isinstance(bv, (int, float)) else 0
                    w = "Blend" if diff_v > 0 else "Ensemble" if diff_v < 0 else "Tie"
                    print(f"  {name:<12} {m_name:<14} {ev:>12} {bv:>12} {diff_v:>+10.2f} {w:>10}")
        else:
            if name == "1X2" and not odds_available:
                print(f"  {name:<12} roi               N/A          N/A          -          N/A")
                print(f"  {name:<12} clv               N/A          N/A          -          N/A")

    print(f"  {table_sep}")

    # ── 7. Summary ────────────────────────────────────────
    print()
    blend_wins = sum(1 for r in rows if r.get("winner") == "Blend" and r["metric"] != "n_samples")
    ens_wins = sum(1 for r in rows if r.get("winner") == "Ensemble" and r["metric"] != "n_samples")
    print(f"  Summary: 3-Model Blend wins {blend_wins}, Ensemble wins {ens_wins}")
    print(f"           (across {blend_wins + ens_wins} metric comparisons)")

    # ── 8. Save report ────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or PROJECT_ROOT / "reports" / f"three_model_vs_ensemble_{ts}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated": datetime.now().isoformat(),
        "test_config": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "test_period": f"{test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]}",
            "odds_available": odds_available,
        },
        "ensemble_model": {
            "path": str(PROJECT_ROOT / "models" / "ensemble_model.joblib"),
        },
        "three_model_blend": {
            "weights_source": args.optimized_weights,
        },
        "markets": comparison,
        "summary": {
            "blend_wins": blend_wins,
            "ensemble_wins": ens_wins,
            "total_comparisons": blend_wins + ens_wins,
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report saved: {output_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
