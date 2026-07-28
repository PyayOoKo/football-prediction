"""
calibrate_league.py — Fit probability calibration on any league's 4-model blend.

Fits Platt, Isotonic, and HybridTail calibrators on per-league blend predictions,
evaluates on held-out test data, and saves the best calibrator.

Usage
-----
    python calibrate_league.py --leagues E0 SP1 D1 I1 F1 SE1
    python calibrate_league.py --leagues SE1
    python calibrate_league.py --all          # Calibrate all leagues with trained models
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("calibrate_league")

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODELS_BASE = PROJECT_ROOT / "models" / "per_league"

EQUAL_WEIGHTS = {
    "1X2": {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
    "Over2.5": {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
    "Over3.5": {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
    "BTTS": {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
}

DC_ONLY_WEIGHTS = {
    "Over2.5": {"dc": 1.0},
    "Over3.5": {"dc": 1.0},
    "BTTS": {"dc": 1.0},
}

LEAGUE_NAMES = {
    "E0": "England Premier League",
    "SP1": "Spain La Liga",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
    "SE1": "Sweden Superettan",
}


def load_data(league: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """SELECT * FROM matches
           WHERE league = ?
             AND home_goals IS NOT NULL AND away_goals IS NOT NULL
             AND result IN ('H', 'D', 'A')
           ORDER BY date ASC""",
        conn, params=(league,),
    )
    conn.close()
    logger.info("Loaded %d %s matches", len(df), league)
    return df


def get_blend_predictions(df: pd.DataFrame, league: str, market: str = "1X2"):
    """Get blend predictions for a given market.

    Parameters
    ----------
    market : str
        ``"1X2"`` for match outcome, ``"over_under"`` for over/under 2.5.
    """
    from src.dixon_coles import DixonColesModel
    from src.elo import EloSystem
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    model_dir = MODELS_BASE / league
    dc = joblib.load(model_dir / "dixon_coles.joblib")
    elo = joblib.load(model_dir / "elo.joblib")
    xgb_path = model_dir / "xgboost.joblib"
    lgb_path = model_dir / "lightgbm.joblib"
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None
    lgb = joblib.load(lgb_path) if lgb_path.exists() else None

    if market == "over_under":
        # DC-only OU probabilities
        logger.info("Computing DC over/under predictions for %d matches...", len(df))
        probs = np.zeros((len(df), 2))
        for i, row in df.iterrows():
            try:
                pred = dc.predict(row["home_team"], row["away_team"])
                o25 = pred.over_2_5_prob
                probs[i] = [1.0 - o25, o25]  # [under, over]
            except Exception:
                probs[i] = [0.5, 0.5]
        logger.info("  Done")
        return probs, dc

    # 1X2: 4-model blend
    weights = {"1X2": {"dc": 0.25, "elo": 0.25}}
    if xgb is not None:
        weights["1X2"]["xgb"] = 0.25
    if lgb is not None:
        weights["1X2"]["lgb"] = 0.25
    n_models = len(weights["1X2"])
    for k in weights["1X2"]:
        weights["1X2"][k] = 1.0 / n_models

    cond_rates = ConditionalRates.from_data(df)
    blend = ThreeModelBlend(
        dc_model=dc, elo_model=elo, xgb_model=xgb, lgb_model=lgb,
        weights=weights, conditional_rates=cond_rates, historical_df=df,
    )

    logger.info("Precomputing 1X2 predictions for %d %s matches...", len(df), league)
    ppm = blend.precompute(df)
    w = blend.weights["1X2"]
    probs = (
        w.get("dc", 0) * ppm.dc_1x2
        + w.get("elo", 0) * ppm.elo_1x2
        + w.get("xgb", 0) * (ppm.xgb_1x2 if xgb is not None else 0)
        + w.get("lgb", 0) * (ppm.lgb_1x2 if lgb is not None else 0)
    )
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums
    logger.info("  Done")
    return probs, blend


def calibrate_and_evaluate(probs, y_true, method="platt"):
    """Fit calibrator on val set, evaluate on held-out test set."""
    from sklearn.metrics import log_loss as sk_log_loss
    from src.calibration import PlattScalingCalibrator, IsotonicRegressionCalibrator, HybridTailCalibrator

    n_classes = probs.shape[1]
    n = len(probs)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    X_val = probs[train_end:val_end]
    y_val = y_true[train_end:val_end]
    X_test = probs[val_end:]
    y_test = y_true[val_end:]

    y_onehot = np.eye(n_classes)[y_test]
    raw_brier = float(np.mean(np.sum((X_test - y_onehot) ** 2, axis=1)))
    raw_ll = float(sk_log_loss(y_test, X_test))
    raw_acc = float(np.mean(np.argmax(X_test, axis=1) == y_test))

    if method == "hybrid":
        calibrator = HybridTailCalibrator(n_classes=n_classes, tail_threshold=0.10)
    elif method == "isotonic":
        calibrator = IsotonicRegressionCalibrator(n_classes=n_classes)
    else:
        calibrator = PlattScalingCalibrator(n_classes=n_classes, max_iter=2000)

    calibrator.fit(X_val, y_val)
    cal_probs = calibrator.transform(X_test)

    cal_brier = float(np.mean(np.sum((cal_probs - y_onehot) ** 2, axis=1)))
    cal_ll = float(sk_log_loss(y_test, cal_probs))
    cal_acc = float(np.mean(np.argmax(cal_probs, axis=1) == y_test))

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
        "n_test": len(X_test),
        "raw": {"brier": round(raw_brier, 4), "log_loss": round(raw_ll, 4), "accuracy": round(raw_acc, 4)},
        "calibrated": {"brier": round(cal_brier, 4), "log_loss": round(cal_ll, 4), "accuracy": round(cal_acc, 4)},
        "improvement": {"brier": round(raw_brier - cal_brier, 4), "log_loss": round(raw_ll - cal_ll, 4), "accuracy": round(cal_acc - raw_acc, 4)},
        "ece": {"low_tail_ece": round(_ece(low), 4), "mid_ece": round(_ece(mid), 4), "high_tail_ece": round(_ece(high), 4),
                "low_tail_n": int(low.sum()), "mid_n": int(mid.sum()), "high_tail_n": int(high.sum())},
    }, calibrator


def save_calibrator(calibrator, method, league, metrics, market="1X2"):
    out_dir = MODELS_BASE / league
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{market}" if market != "1X2" else ""
    cal_path = out_dir / f"blend_calibrator_{method}{suffix}.joblib"
    joblib.dump(calibrator, cal_path)
    report_path = out_dir / f"calibration_metrics_{method}{suffix}.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return cal_path


def calibrate_league(league: str, market: str = "1X2"):
    """Run calibration for a single league. Returns summary dict.

    Parameters
    ----------
    market : str
        ``"1X2"`` for match outcome (3-class), ``"over_under"`` for O/U 2.5 (binary).
    """
    label = "OVER/UNDER" if market == "over_under" else "1X2"
    print(f"\n  {'=' * 60}")
    print(f"  {league} - {LEAGUE_NAMES.get(league, league)} [{label}]")
    print(f"  {'=' * 60}")

    # 1. Load data
    df = load_data(league)
    if len(df) < 200:
        logger.warning("  Only %d matches — skipping (need 200+)", len(df))
        return None

    if market == "over_under":
        total_goals = df["home_goals"] + df["away_goals"]
        y_true = (total_goals > 2.5).astype(int).values  # 0=under, 1=over
    else:
        y_true = df["result"].map({"A": 0, "D": 1, "H": 2}).values

    # 2. Get predictions
    probs, blend = get_blend_predictions(df, league, market=market)

    n_classes = probs.shape[1]

    raw_acc = float(np.mean(np.argmax(probs, axis=1) == y_true))
    y_onehot = np.eye(n_classes)[y_true]
    raw_brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    logger.info("  Raw: Acc=%.2f%%  Brier=%.4f", raw_acc * 100, raw_brier)

    # 3. Calibrate with all methods
    all_metrics = {}
    best_brier = float("inf")
    best_method = None
    best_calibrator = None

    for method in ["platt", "isotonic", "hybrid"]:
        print(f"\n  {'-' * 50}")
        logger.info("  Fitting %s...", method.upper())

        metrics, calibrator = calibrate_and_evaluate(probs, y_true, method=method)
        all_metrics[method] = metrics

        r, c = metrics["raw"], metrics["calibrated"]
        imp = metrics["improvement"]
        logger.info("    Brier: %.4f -> %.4f (d%+.4f)  LL: %.4f -> %.4f  Acc: %.1f%% -> %.1f%%",
                     r["brier"], c["brier"], imp["brier"],
                     r["log_loss"], c["log_loss"],
                     r["accuracy"] * 100, c["accuracy"] * 100)

        if c["brier"] < best_brier:
            best_brier = c["brier"]
            best_method = method
            best_calibrator = calibrator

        save_calibrator(calibrator, method, league, metrics, market=market)

    # 4. Save best calibrator as default
    if best_calibrator is not None:
        suffix = f"_{market}" if market != "1X2" else ""
        cal_path = MODELS_BASE / league / f"blend_calibrator{suffix}.joblib"
        joblib.dump(best_calibrator, cal_path)
        logger.info("  >> Best: %s (Brier=%.4f) saved as blend_calibrator%s.joblib",
                     best_method, best_brier, suffix)

    return {
        "league": league,
        "market": market,
        "name": LEAGUE_NAMES.get(league, league),
        "n_matches": len(df),
        "best_method": best_method,
        "best_brier": round(best_brier, 4),
        "raw_brier": round(raw_brier, 4),
        "raw_acc": round(raw_acc, 4),
        "metrics": all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Per-league probability calibration")
    parser.add_argument("--leagues", nargs="+", help="Leagues to calibrate")
    parser.add_argument("--all", action="store_true", help="Calibrate all available leagues")
    parser.add_argument("--market", choices=["1X2", "over_under"], default="1X2",
                        help="Market to calibrate (default: 1X2)")
    args = parser.parse_args()

    if args.all:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT league FROM matches ORDER BY league")
        all_leagues = [r[0] for r in cur.fetchall()]
        conn.close()
        league_codes = [l for l in all_leagues if (MODELS_BASE / l).exists()]
    elif args.leagues:
        league_codes = args.leagues
    else:
        league_codes = ["E0", "SP1", "D1", "I1", "F1", "SE1"]

    print(f"\n{'=' * 65}")
    label = "OVER/UNDER" if args.market == "over_under" else "1X2"
    print(f"  PER-LEAGUE PROBABILITY CALIBRATION [{label}]")
    print(f"  Leagues: {', '.join(league_codes)}")
    print(f"{'=' * 65}")

    results = []
    for league in league_codes:
        result = calibrate_league(league, market=args.market)
        if result is not None:
            results.append(result)

    # Cross-league comparison table
    print(f"\n\n{'=' * 75}")
    print("  CROSS-LEAGUE CALIBRATION COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'League':6s} {'Matches':>8s} {'RawBrier':>9s} {'BestBrier':>10s} {'Method':>10s} {'Imp%':>6s}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*10} {'-'*10} {'-'*6}")
    for r in sorted(results, key=lambda x: x["best_brier"]):
        imp_pct = (r["raw_brier"] - r["best_brier"]) / r["raw_brier"] * 100
        print(f"  {r['league']:6s} {r['n_matches']:>8} {r['raw_brier']:>9.4f} {r['best_brier']:>10.4f} {r['best_method']:>10s} {imp_pct:>+5.1f}%")

    print(f"\n  Summary: {len(results)}/{len(league_codes)} leagues calibrated successfully")
    print(f"{'=' * 75}\n")


if __name__ == "__main__":
    main()
