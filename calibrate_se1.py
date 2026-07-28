"""
calibrate_se1.py — Fit probability calibration on the SE1 4-model blend.

Goal: Fix overconfident predictions (e.g., Helsingborgs at 48% when market
has them at 29%). Calibration pulls extreme probabilities toward realistic
frequencies.

Approach:
1. Load SE1 data and trained 4-model blend (DC + Elo + XGBoost + LightGBM)
2. Get blend 1X2 predictions using precompute() for performance
3. Split chronologically (60/20/20): train/val/test
4. Fit Platt + Isotonic + HybridTail calibrators on validation set blend probs
5. Compare raw vs calibrated metrics on held-out test set
6. Demonstrate on Falkenberg vs Helsingborgs match
7. Save the best calibrator for production use
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_se1")

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODEL_DIR = PROJECT_ROOT / "models" / "per_league" / "SE1"

EQUAL_WEIGHTS = {
    "1X2":    {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
    "Over2.5": {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
    "Over3.5": {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
    "BTTS":   {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
}

SEP = "=" * 65


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """SELECT * FROM matches
           WHERE league = 'SE1'
             AND home_goals IS NOT NULL AND away_goals IS NOT NULL
             AND result IN ('H', 'D', 'A')
           ORDER BY date ASC""",
        conn,
    )
    conn.close()
    logger.info("Loaded %d SE1 matches", len(df))
    return df


def get_blend_predictions(df: pd.DataFrame):
    """Get 4-model blend 1X2 predictions using precompute() for speed."""
    from src.dixon_coles import DixonColesModel
    from src.elo import EloSystem
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    logger.info("Loading models from %s", MODEL_DIR)
    dc = joblib.load(MODEL_DIR / "dixon_coles.joblib")
    elo = joblib.load(MODEL_DIR / "elo.joblib")
    xgb = joblib.load(MODEL_DIR / "xgboost.joblib")
    lgb = joblib.load(MODEL_DIR / "lightgbm.joblib")

    cond_rates = ConditionalRates.from_data(df)
    blend = ThreeModelBlend(
        dc_model=dc, elo_model=elo, xgb_model=xgb, lgb_model=lgb,
        weights=EQUAL_WEIGHTS, conditional_rates=cond_rates, historical_df=df,
    )

    # Use precompute() for bulk prediction (much faster than per-row)
    logger.info("Precomputing predictions for %d matches...", len(df))
    ppm = blend.precompute(df)
    w = blend.weights["1X2"]
    probs = (
        w.get("dc", 0) * ppm.dc_1x2
        + w.get("elo", 0) * ppm.elo_1x2
        + w.get("xgb", 0) * ppm.xgb_1x2
        + w.get("lgb", 0) * ppm.lgb_1x2
    )
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums
    logger.info("  Done — shape %s", probs.shape)
    return probs, blend


def calibrate_and_evaluate(probs, y_true, method="platt"):
    """Fit calibrator on val set, evaluate on held-out test set."""
    from sklearn.metrics import log_loss as sk_log_loss
    from src.calibration import (
        PlattScalingCalibrator,
        IsotonicRegressionCalibrator,
        HybridTailCalibrator,
    )

    n = len(probs)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    X_val = probs[train_end:val_end]
    y_val = y_true[train_end:val_end]
    X_test = probs[val_end:]
    y_test = y_true[val_end:]

    # Raw metrics on held-out test set
    y_onehot = np.eye(3)[y_test]
    raw_brier = float(np.mean(np.sum((X_test - y_onehot) ** 2, axis=1)))
    raw_ll = float(sk_log_loss(y_test, X_test))
    raw_acc = float(np.mean(np.argmax(X_test, axis=1) == y_test))

    # Fit calibrator
    if method == "hybrid":
        calibrator = HybridTailCalibrator(n_classes=3, tail_threshold=0.10)
    elif method == "isotonic":
        calibrator = IsotonicRegressionCalibrator(n_classes=3)
    else:
        calibrator = PlattScalingCalibrator(n_classes=3, max_iter=2000)

    calibrator.fit(X_val, y_val)
    cal_probs = calibrator.transform(X_test)

    cal_brier = float(np.mean(np.sum((cal_probs - y_onehot) ** 2, axis=1)))
    cal_ll = float(sk_log_loss(y_test, cal_probs))
    cal_acc = float(np.mean(np.argmax(cal_probs, axis=1) == y_test))

    # ECE
    pred_conf = np.max(cal_probs, axis=1)
    pred_class = np.argmax(cal_probs, axis=1)
    correct = (pred_class == y_test).astype(float)

    def _ece(mask):
        if mask.sum() < 2:
            return 0.0
        return float(np.abs(correct[mask] - pred_conf[mask]).mean())

    low = pred_conf < 0.10
    mid = (pred_conf >= 0.10) & (pred_conf <= 0.90)
    high = pred_conf > 0.90

    return {
        "method": method,
        "raw": {"brier": round(raw_brier, 4), "log_loss": round(raw_ll, 4), "accuracy": round(raw_acc, 4)},
        "calibrated": {"brier": round(cal_brier, 4), "log_loss": round(cal_ll, 4), "accuracy": round(cal_acc, 4)},
        "improvement": {"brier": round(raw_brier - cal_brier, 4), "log_loss": round(raw_ll - cal_ll, 4), "accuracy": round(cal_acc - raw_acc, 4)},
        "ece": {"low_tail_ece": round(_ece(low), 4), "mid_ece": round(_ece(mid), 4), "high_tail_ece": round(_ece(high), 4),
                "low_tail_n": int(low.sum()), "mid_n": int(mid.sum()), "high_tail_n": int(high.sum())},
    }, calibrator, X_test, y_test


def demonstrate_calibration(blend, calibrator, method_name):
    home, away = "Falkenberg", "Helsingborgs IF"
    print(f"\n  {method_name} -- Falkenberg vs Helsingborgs IF:")

    raw_pred = blend.predict(home, away)
    raw_1x2 = raw_pred["1x2"]
    raw_array = np.array([[raw_1x2["A"], raw_1x2["D"], raw_1x2["H"]]])
    cal_array = calibrator.transform(raw_array)[0]

    print(f"  {'Market':12s} {'Raw':>10s}  {'Calibrated':>10s}  {'1xBet':>10s}")
    print(f"  {'-'*12} {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Home (F)':12s} {raw_1x2['H']*100:>8.1f}%  {cal_array[2]*100:>8.1f}%  {'49.5%':>10s}")
    print(f"  {'Draw':12s} {raw_1x2['D']*100:>8.1f}%  {cal_array[1]*100:>8.1f}%  {'29.2%':>10s}")
    print(f"  {'Away (HIF)':12s} {raw_1x2['A']*100:>8.1f}%  {cal_array[0]*100:>8.1f}%  {'28.7%':>10s}")

    changes = {
        "Home": (raw_1x2['H'] - cal_array[2]) * 100,
        "Draw": (raw_1x2['D'] - cal_array[1]) * 100,
        "Away (HIF)": (raw_1x2['A'] - cal_array[0]) * 100,
    }
    biggest = max(changes, key=changes.get)
    print(f"  >> Biggest change: {biggest} ({changes[biggest]:+.1f}pp)")


def save_calibrator(calibrator, method, metrics):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cal_path = MODEL_DIR / f"blend_calibrator_{method}.joblib"
    joblib.dump(calibrator, cal_path)
    logger.info("  Calibrator saved to %s", cal_path)
    report_path = MODEL_DIR / f"calibration_metrics_{method}.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return cal_path


def print_metrics(method, metrics):
    r, c = metrics["raw"], metrics["calibrated"]
    imp = metrics["improvement"]
    ece = metrics["ece"]
    logger.info("  %s:", method.upper())
    logger.info("    Brier:    %.4f -> %.4f (d%+.4f)", r["brier"], c["brier"], imp["brier"])
    logger.info("    LogLoss:  %.4f -> %.4f (d%+.4f)", r["log_loss"], c["log_loss"], imp["log_loss"])
    logger.info("    Accuracy: %.2f%% -> %.2f%% (d%+.2f%%)", r["accuracy"]*100, c["accuracy"]*100, imp["accuracy"]*100)
    logger.info("    ECE: low=%.4f(n=%d) mid=%.4f(n=%d) high=%.4f(n=%d)",
                ece["low_tail_ece"], ece["low_tail_n"],
                ece["mid_ece"], ece["mid_n"],
                ece["high_tail_ece"], ece["high_tail_n"])


def main():
    print(f"\n{SEP}")
    print("  SE1 PROBABILITY CALIBRATION")
    print("  Fix overconfident predictions via Platt/Isotonic/HybridTail")
    print(SEP)

    # 1. Load data
    df = load_data()
    y_true = df["result"].map({"A": 0, "D": 1, "H": 2}).values

    # 2. Get blend predictions via precompute()
    probs, blend = get_blend_predictions(df)

    # 3. Calibrate with all 3 methods
    all_results = {}
    calibrators = {}

    for method in ["platt", "isotonic", "hybrid"]:
        print(f"\n  {'-'*50}")
        logger.info("Fitting %s calibration...", method.upper())
        print(f"  {'-'*50}")

        metrics, calibrator, X_test, y_test = calibrate_and_evaluate(probs, y_true, method=method)
        all_results[method] = metrics
        calibrators[method] = calibrator

        print_metrics(method, metrics)
        save_calibrator(calibrator, method, metrics)

    # 4. Demonstrate on Falkenberg vs Helsingborgs
    print(f"\n\n{SEP}")
    print("  DEMONSTRATION: Falkenberg vs Helsingborgs IF")
    print(SEP)
    for method in ["platt", "isotonic", "hybrid"]:
        demonstrate_calibration(blend, calibrators[method], method.upper())

    # 5. Summary
    print(f"\n\n{SEP}")
    print("  CALIBRATION SUMMARY (held-out test set)")
    print(SEP)
    best_method = min(all_results, key=lambda m: all_results[m]["calibrated"]["brier"])
    for method in ["platt", "isotonic", "hybrid"]:
        m = all_results[method]
        imp = m["improvement"]
        marker = " >>" if method == best_method else "   "
        print(f"  {marker} {method.upper():8s}: Brier {m['raw']['brier']:.4f} -> {m['calibrated']['brier']:.4f}"
              f" (d{imp['brier']:+.4f})  |  LL {m['raw']['log_loss']:.4f} -> {m['calibrated']['log_loss']:.4f}"
              f" (d{imp['log_loss']:+.4f})")

    print(f"\n  >> Best method: {best_method.upper()}")
    print(f"  >> Calibrators saved to: {MODEL_DIR}/blend_calibrator_*.joblib")
    print(f"\n  Production use:")
    print(f"    1. cal = joblib.load('{MODEL_DIR}/blend_calibrator_{best_method}.joblib')")
    print(f"    2. raw = blend.predict(home, away)['1x2']")
    print(f"    3. cal_probs = cal.transform(np.array([[raw['A'], raw['D'], raw['H']]]))")
    print(SEP)


if __name__ == "__main__":
    main()
