"""
compare_5v3_blend.py — A/B comparison: 5-model blend vs 3-model blend.

Evaluates both configurations on a held-out test set across all 4 betting markets
(1X2, Over2.5, BTTS, Over3.5) using Brier Score, Log Loss, and Accuracy.

The **5-model blend** uses Dixon-Coles + Elo + XGBoost + LightGBM + CatBoost
with the optimised 5-model weights from `config/three_model_weights.json`.

The **3-model blend** uses only Dixon-Coles + Elo + XGBoost with weights
renormalised from the 5-model config (excluding LGB/CAT contributions).

Usage:
    python compare_5v3_blend.py
    python compare_5v3_blend.py --data data/processed/results_clean.csv
    python compare_5v3_blend.py --test-split 0.15
    python compare_5v3_blend.py --output reports/5v3_comparison.json
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


# ── Weight helpers ────────────────────────────────────────


def _load_5model_weights(path: str | Path) -> dict[str, dict[str, float]]:
    """Load the 5-model optimised weights from config JSON."""
    p = Path(path)
    if not p.exists():
        logger.warning("Weights file not found: %s — using DEFAULT_WEIGHTS from ThreeModelBlend", p)
        from src.models.three_model_blend import DEFAULT_WEIGHTS
        return {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
    with open(p) as f:
        data = json.load(f)
    weights = data.get("weights", {})
    logger.info("Loaded 5-model weights from %s", p)
    for market, w in weights.items():
        logger.info("  %s: %s", market, w)
    return weights


def _make_3model_weights(w5: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Convert 5-model weights to 3-model weights by dropping lgb/cat and renormalising.

    Only keeps dc, elo, xgb — renormalises so they sum to 1.0 per market.
    """
    w3: dict[str, dict[str, float]] = {}
    for market, weights in w5.items():
        filtered = {k: v for k, v in weights.items() if k in ("dc", "elo", "xgb")}
        total = sum(filtered.values())
        if total > 0:
            w3[market] = {k: round(v / total, 4) for k, v in filtered.items()}
        else:
            # Fallback: equal split among available 3 models
            available = [k for k in ("dc", "elo", "xgb") if weights.get(k, 0) > 0 or True]
            w3[market] = {k: round(1.0 / len(available), 4) for k in available}
    return w3


# ── Model loading ─────────────────────────────────────────


def load_models() -> dict[str, Any]:
    """Load all 5 models from disk.

    Returns dict with keys: 'dc', 'elo', 'xgb', 'lgb', 'cat'
    Missing models are set to None with a warning.
    """
    import joblib

    models: dict[str, Any] = {}

    model_paths = {
        "dc": PROJECT_ROOT / "models" / "dixon_coles_model.joblib",
        "elo": PROJECT_ROOT / "models" / "elo_model.joblib",
        "xgb": PROJECT_ROOT / "models" / "xgboost_model.joblib",
        "lgb": PROJECT_ROOT / "models" / "lightgbm_model.joblib",
        "cat": PROJECT_ROOT / "models" / "catboost_model.joblib",
    }

    # Also try alternative paths
    alt_paths = {
        "xgb": [
            PROJECT_ROOT / "models" / "xgboost_model.joblib",
            PROJECT_ROOT / "models" / "worldcup_xgboost.joblib",
            PROJECT_ROOT / "models" / "xgboost_model",
        ],
        "lgb": [
            PROJECT_ROOT / "models" / "lightgbm_model.joblib",
            PROJECT_ROOT / "models" / "worldcup_lightgbm.joblib",
            PROJECT_ROOT / "models" / "lightgbm_model",
        ],
        "cat": [
            PROJECT_ROOT / "models" / "catboost_model.joblib",
        ],
    }

    for key, path in model_paths.items():
        if key in alt_paths:
            # Try alternatives
            loaded = None
            for p in alt_paths[key]:
                if p.exists():
                    try:
                        loaded = joblib.load(p)
                        logger.info("Loaded %s from %s (%s)", key, p, type(loaded).__name__)
                        break
                    except Exception as e:
                        logger.warning("Failed to load %s from %s: %s", key, p, e)
            models[key] = loaded
            if loaded is None:
                logger.warning("No %s model found — will be None in blend", key)
        else:
            if path.exists():
                try:
                    models[key] = joblib.load(path)
                    logger.info("Loaded %s from %s (%s)", key, path, type(models[key]).__name__)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", key, e)
                    models[key] = None
            else:
                logger.warning("Model not found: %s", path)
                models[key] = None

    return models


# ── Data loading ─────────────────────────────────────────


def load_data(data_path: str | Path, test_split: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and chronologically split data into train/val/test."""
    df = pd.read_csv(data_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    val_split = int(n * (1 - 2 * test_split))
    test_split_idx = int(n * (1 - test_split))

    train_df = df.iloc[:val_split].copy()
    val_df = df.iloc[val_split:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()

    logger.info("Split: %d train + %d val + %d test = %d total",
                len(train_df), len(val_df), len(test_df), n)
    if "date" in df.columns:
        logger.info("  Train: %s to %s", train_df["date"].iloc[0], train_df["date"].iloc[-1])
        logger.info("  Val:   %s to %s", val_df["date"].iloc[0], val_df["date"].iloc[-1])
        logger.info("  Test:  %s to %s", test_df["date"].iloc[0], test_df["date"].iloc[-1])

    return train_df, val_df, test_df


# ── Blend setup ──────────────────────────────────────────


def create_blend(
    dc_model: Any,
    elo_model: Any,
    xgb_model: Any,
    lgb_model: Any,
    cat_model: Any,
    historical_df: pd.DataFrame,
    weights: dict[str, dict[str, float]],
    use_5model: bool,
) -> Any:
    """Create a ThreeModelBlend with the given models and weights.

    If use_5model=True, all 5 models are passed. Otherwise only dc+elo+xgb.
    """
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    cond_rates = ConditionalRates.from_data(historical_df)

    blend = ThreeModelBlend(
        dc_model=dc_model,
        elo_model=elo_model,
        xgb_model=xgb_model if use_5model or True else None,
        lgb_model=lgb_model if use_5model else None,
        cat_model=cat_model if use_5model else None,
        weights=weights,
        conditional_rates=cond_rates,
        historical_df=historical_df,
    )
    return blend


# ── Evaluation ──────────────────────────────────────────


def evaluate_blend(
    blend: Any,
    test_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Evaluate a blend on test data across all markets.

    Returns dict of market -> {brier_score, log_loss, accuracy, n_samples}.
    """
    hg = test_df["home_goals"].values.astype(float)
    ag = test_df["away_goals"].values.astype(float)
    y_result = test_df["result"].map({"A": 0, "D": 1, "H": 2}).values
    y_btts = ((hg > 0) & (ag > 0)).astype(float)
    y_ou25 = ((hg + ag) > 2.5).astype(float)
    y_ou35 = ((hg + ag) > 3.5).astype(float)

    # Batch precompute
    ppm = blend.precompute(test_df, cache_key="5v3_eval")

    results: dict[str, dict[str, float]] = {}

    # 1X2
    w_1x2 = blend.weights.get("1X2", {})
    probs_1x2 = blend._blend_1x2(ppm, w_1x2)
    results["1X2"] = compute_all_1x2_metrics(y_result, probs_1x2)

    # Over2.5
    w_ou25 = blend.weights.get("Over2.5", {})
    probs_ou25 = blend._blend_binary(ppm, w_ou25, "Over2.5")
    results["Over2.5"] = compute_all_binary_metrics(y_ou25, probs_ou25)

    # Over3.5
    w_ou35 = blend.weights.get("Over3.5", {})
    probs_ou35 = blend._blend_binary(ppm, w_ou35, "Over3.5")
    results["Over3.5"] = compute_all_binary_metrics(y_ou35, probs_ou35)

    # BTTS
    w_btts = blend.weights.get("BTTS", {})
    probs_btts = blend._blend_binary(ppm, w_btts, "BTTS")
    results["BTTS"] = compute_all_binary_metrics(y_btts, probs_btts)

    return results


def evaluate_individual_models(
    blend: Any,
    test_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Evaluate each individual model separately on 1X2."""
    hg = test_df["home_goals"].values.astype(float)
    ag = test_df["away_goals"].values.astype(float)
    y_result = test_df["result"].map({"A": 0, "D": 1, "H": 2}).values

    ppm = blend.precompute(test_df, cache_key="5v3_individual")

    results: dict[str, dict[str, float]] = {}

    models = {
        "Dixon-Coles": ppm.dc_1x2,
        "Elo": ppm.elo_1x2,
        "XGBoost": ppm.xgb_1x2,
        "LightGBM": ppm.lgb_1x2,
        "CatBoost": ppm.cat_1x2,
    }

    for name, probs in models.items():
        if probs is not None and len(probs) > 0:
            results[name] = compute_all_1x2_metrics(y_result, probs)

    return results


# ── Main ─────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare 5-model blend vs 3-model blend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", "-d",
                        default=str(PROJECT_ROOT / "data" / "processed" / "results_clean.csv"),
                        help="Path to input CSV data (default: data/processed/results_clean.csv)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Fraction of data for test set (default: 0.15)")
    parser.add_argument("--weights",
                        default=str(PROJECT_ROOT / "config" / "three_model_weights.json"),
                        help="Path to 5-model optimised weights JSON")
    parser.add_argument("--output", "-o", default=None,
                        help="Output report path (auto-generated if not provided)")
    args = parser.parse_args(argv)

    t_start = time.time()

    print()
    print("=" * 72)
    print("  5-MODEL vs 3-MODEL BLEND COMPARISON")
    print("  Dixon-Coles + Elo + XGBoost + LightGBM + CatBoost")
    print("=" * 72)

    # ── 1. Load data ──────────────────────────────────────
    print("\n-- Loading data ------------------------------")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data not found: %s", data_path)
        return 1
    train_df, val_df, test_df = load_data(data_path, args.test_split)
    fit_df = pd.concat([train_df, val_df], ignore_index=True)

    # ── 2. Load models ──────────────────────────────────
    print("\n-- Loading models ----------------------------")
    models = load_models()
    if models["dc"] is None:
        logger.error("Dixon-Coles model is required but not found")
        return 1
    if models["elo"] is None:
        logger.error("Elo model is required but not found")
        return 1
    if models["xgb"] is None:
        logger.warning("XGBoost not found — 3-model blend will fall back to 2 models (dc+elo)")
        # That's still valid, just note it

    # ── 3. Load weights ────────────────────────────────
    print("\n-- Loading weights ---------------------------")
    w5 = _load_5model_weights(args.weights)
    w3 = _make_3model_weights(w5)

    print("\n  5-Model Weights:")
    for market, w in w5.items():
        print(f"    {market:<10}: {w}")
    print("\n  3-Model Weights (renormalised, no LGB/CAT):")
    for market, w in w3.items():
        print(f"    {market:<10}: {w}")

    # ── 4. Create blends ──────────────────────────────
    print("\n-- Creating blends ---------------------------")
    def _ck(x: Any) -> str:
        return "[OK]" if x is not None else "[..]"

    blend_5 = create_blend(
        dc_model=models["dc"],
        elo_model=models["elo"],
        xgb_model=models["xgb"],
        lgb_model=models["lgb"],
        cat_model=models["cat"],
        historical_df=fit_df,
        weights=w5,
        use_5model=True,
    )
    print(f"  5-Model Blend: dc={_ck(models['dc'])} elo={_ck(models['elo'])} "
          f"xgb={_ck(models['xgb'])} lgb={_ck(models['lgb'])} "
          f"cat={_ck(models['cat'])}")

    blend_3 = create_blend(
        dc_model=models["dc"],
        elo_model=models["elo"],
        xgb_model=models["xgb"],
        lgb_model=None,
        cat_model=None,
        historical_df=fit_df,
        weights=w3,
        use_5model=False,
    )
    print(f"  3-Model Blend: dc={_ck(models['dc'])} elo={_ck(models['elo'])} "
          f"xgb={_ck(models['xgb'])}")

    # ── 5. Evaluate blends ─────────────────────────────
    print(f"\n-- Evaluating on {len(test_df)} test matches -------")
    print("  (This runs feature engineering + all models in batch...)")

    print("  Evaluating 5-Model...")
    results_5 = evaluate_blend(blend_5, test_df)

    print("  Evaluating 3-Model...")
    results_3 = evaluate_blend(blend_3, test_df)

    # ── 6. Evaluate individual models ──────────────────
    print("  Evaluating individual models...")
    individual = evaluate_individual_models(blend_5, test_df)

    # ── 7. Print comparison table ──────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    markets_order = ["1X2", "Over2.5", "BTTS", "Over3.5"]
    metrics_display = [
        ("brier_score", "Brier (low)", True),
        ("log_loss", "LogLoss(low)", True),
        ("accuracy", "Accuracy(hi)", False),
    ]

    print(f"\n  {'Market':<12} {'Metric':<14} {'5-Model':>12} {'3-Model':>12} {'Diff':>10} {'Winner':>10}")
    print(f"  {'-'*70}")

    summary = {"markets": {}, "individual": individual}

    for market in markets_order:
        r5 = results_5.get(market, {})
        r3 = results_3.get(market, {})

        summary["markets"][market] = {
            "5_model": r5,
            "3_model": r3,
        }

        for metric_name, label, lower_better in metrics_display:
            v5 = r5.get(metric_name)
            v3 = r3.get(metric_name)
            if v5 is None or v3 is None:
                continue

            diff = v5 - v3
            if lower_better:
                winner = "5-Model" if diff < 0 else ("3-Model" if diff > 0 else "Tie")
            else:
                winner = "5-Model" if diff > 0 else ("3-Model" if diff < 0 else "Tie")

            diff_str = f"{diff:+.5f}" if abs(diff) >= 0.00001 else "  -    "
            print(f"  {market:<12} {label:<14} {v5:>12.5f} {v3:>12.5f} {diff_str:>10} {winner:>10}")

        # Samples row
        n5 = r5.get("n_samples", "?")
        print(f"  {market:<12} {'samples':<14} {str(n5):>12} {str(n5):>12} {'-':>10} {'-':>10}")

    print(f"  {'-'*70}")

    # ── 8. Summary stats ────────────────────────────────
    wins_5 = 0
    wins_3 = 0
    ties = 0
    total = 0

    for market in markets_order:
        r5 = results_5.get(market, {})
        r3 = results_3.get(market, {})
        for metric_name, _, lower_better in metrics_display:
            v5 = r5.get(metric_name)
            v3 = r3.get(metric_name)
            if v5 is None or v3 is None:
                continue
            total += 1
            if lower_better:
                if v5 < v3:
                    wins_5 += 1
                elif v3 < v5:
                    wins_3 += 1
                else:
                    ties += 1
            else:
                if v5 > v3:
                    wins_5 += 1
                elif v3 > v5:
                    wins_3 += 1
                else:
                    ties += 1

    print(f"\n  SUMMARY: 5-Model wins {wins_5}/{total}, 3-Model wins {wins_3}/{total}, Ties {ties}/{total}")
    pct = (wins_5 / total * 100) if total > 0 else 0
    print(f"           Win rate for 5-Model: {pct:.0f}%")
    summary["summary"] = {
        "5_model_wins": wins_5,
        "3_model_wins": wins_3,
        "ties": ties,
        "total_comparisons": total,
        "5_model_win_rate_pct": round(pct, 1),
    }

    # ── 9. Individual model leaderboard ────────────────
    print(f"\n{'='*72}")
    print("  INDIVIDUAL MODEL PERFORMANCE (1X2 Brier Score)")
    print(f"{'='*72}")
    print(f"\n  {'Model':<16} {'Brier':>10} {'LogLoss':>10} {'Accuracy':>10}")
    print(f"  {'-'*46}")

    ind_sorted = sorted(individual.items(), key=lambda x: x[1].get("brier_score", 1))
    for name, m in ind_sorted:
        b = m.get("brier_score", 0)
        ll = m.get("log_loss", 0) or 0
        acc = m.get("accuracy", 0)
        print(f"  {name:<16} {b:>10.5f} {ll:>10.5f} {acc:>10.4f}")
    summary["individual_leaderboard"] = {name: m for name, m in ind_sorted}

    # ── 10. Save report ─────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or PROJECT_ROOT / "reports" / f"5v3_blend_comparison_{ts}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated": datetime.now().isoformat(),
        "data_source": str(data_path),
        "test_config": {
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        },
        "available_models": {k: v is not None for k, v in models.items()},
        "weights_5_model": w5,
        "weights_3_model": w3,
        "results": summary,
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
