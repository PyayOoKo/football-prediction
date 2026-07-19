"""
calibrate_blend_1x2.py — Fit Platt scaling on the 3-model blend's 1X2 predictions.

Goal: Recalibrate the blend's 1X2 probabilities so they're accurate enough to
find profitable value bets in the match winner market.

Approach
--------
1. Load historical match data and the fitted 3-model blend
2. Get blend 1X2 predictions for ALL historical matches using batch predict_matches()
3. Split chronologically (60/20/20): train/val/test
4. Fit PlattScalingCalibrator + HybridTailCalibrator on the *blended* 1X2 probs
5. Compare raw vs calibrated metrics on held-out test set (Brier, log-loss, ECE)
6. Simulate value betting on held-out test set only (avoid data leakage)
7. Save the calibrator for production use

Usage
-----
    /c/Users/dell/AppData/Local/Python/bin/python calibrate_blend_1x2.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════
#  1. Load data
# ═══════════════════════════════════════════════════════════


def load_data() -> pd.DataFrame:
    """Load historical match data with results, sorted chronologically."""
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "results_clean.csv",
        PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv",
    ]
    for path in candidates:
        if path.exists():
            logger.info("Loading data from %s", path)
            df = pd.read_csv(path, low_memory=False)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if "result" in df.columns:
                df = df[df["result"].isin(["H", "D", "A"])].copy()
            logger.info("  %d matches loaded (with results)", len(df))
            return df
    raise FileNotFoundError("No data file found")


# ═══════════════════════════════════════════════════════════
#  2. Get blend predictions via batch predict_matches()
# ═══════════════════════════════════════════════════════════


def get_blend_predictions_batch(df: pd.DataFrame) -> np.ndarray:
    """Get the 3-model blend's 1X2 predictions via precompute() (batch).

    Uses ``blend.precompute(df)`` which calls ``build_features()`` ONCE
    for all home/away teams, avoiding the O(N²) cost of per-row prediction.

    Returns ``(n, 3)`` array in ``[away, draw, home]`` order.
    """
    import joblib
    from src.models.three_model_blend import ThreeModelBlend

    blend_path = PROJECT_ROOT / "models" / "three_model_blend.joblib"
    if not blend_path.exists():
        raise FileNotFoundError(
            f"Blend model not found at {blend_path} — "
            "run 'python run_pipeline.py' first"
        )

    logger.info("Loading 3-model blend from %s ...", blend_path)
    blend = ThreeModelBlend.load(str(blend_path), historical_df=df)
    logger.info("Blend loaded: %d markets", len(blend.available_markets))

    # Precompute: runs all 3 models ONCE in batch (feature engineering runs once)
    logger.info("Precomputing predictions for %d matches...", len(df))
    ppm = blend.precompute(df)

    # Blend 1X2 using the optimised weights
    w = blend.weights["1X2"]
    probs = (
        w["poisson"] * ppm.pois_1x2
        + w["elo"] * ppm.elo_1x2
        + w["xgb"] * ppm.xgb_1x2
    )
    # Renormalise each row to sum to 1.0
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums

    logger.info("  Batch precompute complete: %d rows", len(probs))
    return probs


# ═══════════════════════════════════════════════════════════
#  3. Calibration & Evaluation
# ═══════════════════════════════════════════════════════════


def calibrate_and_evaluate(
    probs: np.ndarray,
    y_true: np.ndarray,
    method: str = "platt",
) -> dict:
    """Fit calibrator on val set, evaluate on held-out test set.

    Splits data chronologically (60/20/20). The calibrator is fitted
    ONLY on the validation split (20%). The test split (20%) is never
    touched by the calibrator — it's a true held-out evaluation.
    """
    from sklearn.metrics import log_loss as sk_log_loss

    from src.calibration import (
        PlattScalingCalibrator,
        HybridTailCalibrator,
    )

    n = len(probs)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    X_train = probs[:train_end]
    y_train = y_true[:train_end]
    X_val = probs[train_end:val_end]
    y_val = y_true[train_end:val_end]
    X_test = probs[val_end:]
    y_test = y_true[val_end:]

    logger.info(
        "Split: train=%d, val=%d, test=%d (chronological)",
        len(X_train), len(X_val), len(X_test),
    )

    # Raw metrics on held-out test set
    y_onehot = np.eye(3)[y_test]
    raw_brier = float(np.mean(np.sum((X_test - y_onehot) ** 2, axis=1)))
    raw_ll = float(sk_log_loss(y_test, X_test))
    raw_acc = float(np.mean(np.argmax(X_test, axis=1) == y_test))

    # Fit calibrator on validation split ONLY
    if method == "hybrid":
        calibrator = HybridTailCalibrator(n_classes=3, tail_threshold=0.10)
    else:
        calibrator = PlattScalingCalibrator(n_classes=3, max_iter=2000)

    calibrator.fit(X_val, y_val)

    # Transform held-out test set only
    cal_probs = calibrator.transform(X_test)

    # Calibrated metrics
    cal_brier = float(np.mean(np.sum((cal_probs - y_onehot) ** 2, axis=1)))
    cal_ll = float(sk_log_loss(y_test, cal_probs))
    cal_acc = float(np.mean(np.argmax(cal_probs, axis=1) == y_test))

    # Per-region ECE on test set
    pred_conf = np.max(cal_probs, axis=1)
    pred_class = np.argmax(cal_probs, axis=1)
    correct = (pred_class == y_test).astype(float)

    def _ece(mask: np.ndarray) -> float:
        if mask.sum() < 2:
            return 0.0
        return float(np.abs(correct[mask] - pred_conf[mask]).mean())

    low_mask = pred_conf < 0.10
    mid_mask = (pred_conf >= 0.10) & (pred_conf <= 0.90)
    high_mask = pred_conf > 0.90

    results = {
        "method": method,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "raw": {
            "brier": round(raw_brier, 4),
            "log_loss": round(raw_ll, 4),
            "accuracy": round(raw_acc, 4),
        },
        "calibrated": {
            "brier": round(cal_brier, 4),
            "log_loss": round(cal_ll, 4),
            "accuracy": round(cal_acc, 4),
        },
        "improvement": {
            "brier": round(raw_brier - cal_brier, 4),
            "log_loss": round(raw_ll - cal_ll, 4),
            "accuracy": round(cal_acc - raw_acc, 4),
        },
        "ece": {
            "low_tail_samples": int(low_mask.sum()),
            "mid_samples": int(mid_mask.sum()),
            "high_tail_samples": int(high_mask.sum()),
            "low_tail_ece": round(_ece(low_mask), 4),
            "mid_ece": round(_ece(mid_mask), 4),
            "high_tail_ece": round(_ece(high_mask), 4),
        },
    }
    return results, calibrator, X_test, y_test


# ═══════════════════════════════════════════════════════════
#  4. Value Bet Simulation (on held-out test set only)
# ═══════════════════════════════════════════════════════════


def simulate_value_bets(
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    y_true: np.ndarray,
    min_ev: float = 0.02,
    label: str = "Raw",
) -> dict:
    """Simulate value betting on a held-out test set.

    Uses implied odds from raw probabilities + 5% overround to simulate
    realistic bookmaker odds. Only evaluates on the passed data (should
    be the held-out test set to avoid data leakage).

    Parameters
    ----------
    raw_probs : np.ndarray (n, 3)
        Raw blend probabilities (test set only).
    cal_probs : np.ndarray (n, 3)
        Calibrated probabilities (test set only).
    y_true : np.ndarray (n,)
        Actual outcomes (0=Away, 1=Draw, 2=Home).
    min_ev : float
        Minimum expected value threshold (default 2%).
    label : str
        Label for logging.

    Returns
    -------
    dict with raw/calibrated n_bets, win_rate, roi, total_profit.
    """
    n = len(raw_probs)
    outcomes_list = ["Away Win", "Draw", "Home Win"]

    # Simulate bookmaker odds: fair probs + 5% overround
    odds = np.zeros_like(raw_probs)
    for i in range(n):
        total = raw_probs[i].sum()
        fair = raw_probs[i] / total
        # 5% vig distributed proportionally
        implied = fair * 1.05
        odds[i] = np.clip(1.0 / implied, 1.01, 50.0)

    bets_raw, bets_cal = [], []

    for i in range(n):
        actual = y_true[i]
        for outcome_idx in range(3):
            decimal_odds = odds[i, outcome_idx]
            if not np.isfinite(decimal_odds) or decimal_odds <= 1.0:
                continue

            # Raw bet
            mp_raw = raw_probs[i, outcome_idx]
            ev_raw = mp_raw * decimal_odds - 1
            if ev_raw > min_ev:
                kelly = (mp_raw * decimal_odds - 1) / (decimal_odds - 1)
                kelly = max(0.0, min(kelly, 0.25))  # 25% fractional Kelly
                won = outcome_idx == actual
                bets_raw.append({
                    "outcome": outcomes_list[outcome_idx],
                    "odds": round(decimal_odds, 2),
                    "model_prob": round(mp_raw, 4),
                    "ev": round(ev_raw, 4),
                    "kelly": round(kelly, 4),
                    "won": won,
                    "profit": round(kelly * (decimal_odds - 1) if won else -kelly, 4),
                })

            # Calibrated bet
            mp_cal = cal_probs[i, outcome_idx]
            ev_cal = mp_cal * decimal_odds - 1
            if ev_cal > min_ev:
                kelly = (mp_cal * decimal_odds - 1) / (decimal_odds - 1)
                kelly = max(0.0, min(kelly, 0.25))
                won = outcome_idx == actual
                bets_cal.append({
                    "outcome": outcomes_list[outcome_idx],
                    "odds": round(decimal_odds, 2),
                    "model_prob": round(mp_cal, 4),
                    "ev": round(ev_cal, 4),
                    "kelly": round(kelly, 4),
                    "won": won,
                    "profit": round(kelly * (decimal_odds - 1) if won else -kelly, 4),
                })

    def _stats(bets: list) -> dict:
        if not bets:
            return {"n_bets": 0, "win_rate": 0.0, "roi": 0.0, "total_profit": 0.0}
        won = sum(1 for b in bets if b["won"])
        total_stake = sum(b["kelly"] for b in bets)
        total_profit = sum(b["profit"] for b in bets)
        return {
            "n_bets": len(bets),
            "win_rate": round(won / len(bets), 4),
            "roi": round((total_profit / total_stake) * 100, 2) if total_stake > 0 else 0.0,
            "total_profit": round(total_profit, 4),
            "mean_ev": round(np.mean([b["ev"] for b in bets]), 4),
            "mean_odds": round(np.mean([b["odds"] for b in bets]), 2),
        }

    raw_stats = _stats(bets_raw)
    cal_stats = _stats(bets_cal)

    logger.info(
        "  [%s] Raw: %d bets (ROI=%.1f%%)  |  Calibrated: %d bets (ROI=%.1f%%)",
        label,
        raw_stats["n_bets"], raw_stats["roi"],
        cal_stats["n_bets"], cal_stats["roi"],
    )

    return {"raw": raw_stats, "calibrated": cal_stats}


# ═══════════════════════════════════════════════════════════
#  5. Save Calibrator
# ═══════════════════════════════════════════════════════════


def save_calibrator(calibrator, method: str, metrics: dict):
    """Save the fitted calibrator alongside the blend model."""
    import joblib

    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    cal_path = output_dir / f"blend_calibrator_{method}.joblib"
    joblib.dump(calibrator, cal_path)
    logger.info("  Calibrator saved to %s", cal_path)

    report_path = output_dir / f"calibration_metrics_{method}.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("  Metrics saved to %s", report_path)

    return cal_path


# ═══════════════════════════════════════════════════════════
#  6. Main
# ═══════════════════════════════════════════════════════════


def main():
    logger.info("=" * 60)
    logger.info("  3-MODEL BLEND 1X2 CALIBRATION")
    logger.info("=" * 60)

    # 1. Load data
    df = load_data()
    y_true = df["result"].map({"A": 0, "D": 1, "H": 2}).values

    # 2. Get blend predictions in batch
    probs = get_blend_predictions_batch(df)

    raw_acc_all = np.mean(np.argmax(probs, axis=1) == y_true)
    logger.info("  Overall blend accuracy: %.2f%%", raw_acc_all * 100)

    # 3. Calibrate with both methods (train/val/test split)
    all_results = {}
    test_sets = {}  # store test sets for value bet sim

    for method in ["platt", "hybrid"]:
        logger.info("\n" + "-" * 50)
        logger.info("  Fitting %s calibration...", method.upper())
        logger.info("-" * 50)

        metrics, calibrator, X_test, y_test = calibrate_and_evaluate(
            probs, y_true, method=method
        )
        all_results[method] = metrics
        test_sets[method] = (X_test, y_test)

        r, c = metrics["raw"], metrics["calibrated"]
        imp = metrics["improvement"]
        ece = metrics["ece"]

        logger.info("  Raw Brier:        %.4f", r["brier"])
        logger.info("  Calibrated Brier: %.4f  (Δ%+.4f)", c["brier"], imp["brier"])
        logger.info("  Raw Log-loss:     %.4f", r["log_loss"])
        logger.info("  Calibrated LL:    %.4f  (Δ%+.4f)", c["log_loss"], imp["log_loss"])
        logger.info("  Raw Acc:          %.2f%%", r["accuracy"] * 100)
        logger.info("  Calibrated Acc:   %.2f%%", c["accuracy"] * 100)

        # Per-region ECE
        logger.info("  ECE low_tail:     %.4f (n=%d)", ece["low_tail_ece"], ece["low_tail_samples"])
        logger.info("  ECE mid:          %.4f (n=%d)", ece["mid_ece"], ece["mid_samples"])
        logger.info("  ECE high_tail:    %.4f (n=%d)", ece["high_tail_ece"], ece["high_tail_samples"])

        # 4. Value bet simulation on HELD-OUT TEST SET only
        cal_test = calibrator.transform(X_test)
        vb = simulate_value_bets(
            X_test, cal_test, y_test,
            min_ev=0.02, label=method.upper(),
        )
        all_results[method]["value_bets"] = vb

        # 5. Save calibrator
        save_calibrator(calibrator, method, metrics)

    # 6. Summary
    logger.info("\n" + "=" * 60)
    logger.info("  CALIBRATION SUMMARY (held-out test set)")
    logger.info("=" * 60)
    for method in ["platt", "hybrid"]:
        m = all_results[method]
        vb = m.get("value_bets", {}).get("calibrated", {})
        logger.info(
            "  %s: Brier %.4f→%.4f (Δ%+.4f)  |  LL %.4f→%.4f  |  "
            "Bets: %d (ROI=%.1f%%)",
            method.upper(),
            m["raw"]["brier"], m["calibrated"]["brier"], m["improvement"]["brier"],
            m["raw"]["log_loss"], m["calibrated"]["log_loss"],
            vb.get("n_bets", 0), vb.get("roi", 0.0),
        )

    # Determine best method
    best_method = min(all_results, key=lambda m: all_results[m]["calibrated"]["brier"])
    best_brier = all_results[best_method]["calibrated"]["brier"]
    brier_imp = all_results[best_method]["improvement"]["brier"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("  ✅ BEST METHOD: %s (Brier=%.4f, Δ=+%.4f)", best_method.upper(), best_brier, brier_imp)
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Calibrators saved to: models/blend_calibrator_*.joblib")
    logger.info("")
    logger.info("  To use in production:")
    logger.info("    1. load calibrator with joblib.load('models/blend_calibrator_platt.joblib')")
    logger.info("    2. Get raw blend 1X2: result = blend.predict(home, away)['1x2']")
    logger.info("    3. Convert to array [away, draw, home]")
    logger.info("    4. calibrator.transform(np.array([[away, draw, home]]))")
    logger.info("    5. Pass cal_probs to blend.predict_with_calibrated_probs()")


if __name__ == "__main__":
    main()
