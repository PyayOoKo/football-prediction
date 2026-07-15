#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Dixon-Coles Model Validation                                             ║
║                                                                           ║
║  Validates the Dixon-Coles MLE model against the baseline Poisson model,  ║
║  with a focus on low-scoring matches (0-2 total goals) where the tau     ║
║  correction should provide the most improvement.                         ║
║                                                                           ║
║  Expected: 2-5% improvement on low-scoring matches vs Poisson baseline.   ║
║                                                                           ║
║  Outputs:                                                                 ║
║  - models/dixon_coles_model.joblib                                       ║
║  - reports/dixon_coles_validation_{timestamp}.json                      ║
║  - reports/dixon_coles_vs_poisson_{timestamp}.json                      ║
║                                                                           ║
║  Usage:                                                                   ║
║      python scripts/validate_dixon_coles.py                               ║
║      python scripts/validate_dixon_coles.py --data data/raw/results.csv   ║
║      python scripts/validate_dixon_coles.py --quiet                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from config import config
from src.dixon_coles import DixonColesModel
from src.poisson_model import PoissonModel

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
_DEFAULT_TRAIN_END = "2023-01-01"
_DEFAULT_TEST_END = "2026-12-31"
_REQUIRED_COLS = ["date", "home_team", "away_team", "result", "home_goals", "away_goals"]
_MODEL_DIR = PROJECT_ROOT / "models"
_REPORT_DIR = PROJECT_ROOT / "reports"

# Low-scoring threshold: matches with total goals <= 2
_LOW_SCORE_MAX_TOTAL = 2


# ═══════════════════════════════════════════════════════════
#  Data loading & splitting
# ═══════════════════════════════════════════════════════════


def load_data(path: str | Path, min_date: str = "2016-01-01") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_dates = df["date"].copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    if df["date"].isna().sum() > len(df) * 0.5:
        logger.warning("ISO format failed -- retrying with dayfirst=True")
        df["date"] = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")

    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(subset=["date"])

    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)].copy()

    df = df[df["result"].notna() & df["result"].isin(["H", "D", "A"])].copy()

    logger.info("Loaded %d matches from %s to %s", len(df),
                df["date"].min().strftime("%Y-%m-%d"),
                df["date"].max().strftime("%Y-%m-%d"))
    return df


def chronological_split(df: pd.DataFrame, train_end: str, test_end: str):
    train_cutoff = pd.Timestamp(train_end)
    test_cutoff = pd.Timestamp(test_end)
    train = df[df["date"] < train_cutoff].copy()
    test = df[(df["date"] >= train_cutoff) & (df["date"] <= test_cutoff)].copy()
    logger.info("Split: train=%d, test=%d", len(train), len(test))
    return train, test


def get_targets(df: pd.DataFrame) -> np.ndarray:
    mapping = {"A": 0, "D": 1, "H": 2}
    result = df["result"].map(mapping).fillna(-1).values.astype(int)
    n_bad = int((result == -1).sum())
    if n_bad > 0:
        logger.warning("get_targets: %d unmapped values", n_bad)
    return result


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, name: str = "Model"):
    from sklearn.metrics import log_loss as sk_ll
    pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred == y_true))
    ll = float(sk_ll(y_true, probs))
    y_oh = np.zeros((len(y_true), 3))
    for i, v in enumerate(y_true):
        if 0 <= v <= 2:
            y_oh[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))
    logger.info("%s -- accuracy=%.4f log_loss=%.4f brier=%.4f", name, accuracy, ll, brier)
    return {"accuracy": round(accuracy, 4), "log_loss": round(ll, 4),
            "brier_score": round(brier, 4), "n": int(len(y_true))}


def compute_btts_metrics(hg: np.ndarray, ag: np.ndarray, pred_probs: np.ndarray):
    actual = ((hg > 0) & (ag > 0)).astype(float)
    pred = (pred_probs > 0.5).astype(float)
    return {"btts_accuracy": round(float(np.mean(pred == actual)), 4),
            "btts_brier": round(float(np.mean((pred_probs - actual) ** 2)), 4)}


def compute_ou_metrics(hg: np.ndarray, ag: np.ndarray, pred_probs: np.ndarray, threshold: float = 2.5):
    actual = ((hg + ag) > threshold).astype(float)
    pred = (pred_probs > 0.5).astype(float)
    k = f"over_under_{threshold:.1f}".replace(".", "_")
    return {f"{k}_accuracy": round(float(np.mean(pred == actual)), 4),
            f"{k}_brier": round(float(np.mean((pred_probs - actual) ** 2)), 4)}


def evaluate_predictions(model, df_test: pd.DataFrame, name: str = "Model"):
    """Run full evaluation on a test set: 1X2, BTTS, O/U."""
    hg = df_test["home_goals"].values.astype(float)
    ag = df_test["away_goals"].values.astype(float)
    y_true = get_targets(df_test)

    # Both DC and Poisson models support predict_matches() with BTTS/O/U columns
    preds = model.predict_matches(df_test)
    probs = np.column_stack([preds["away_win_prob"].values,
                              preds["draw_prob"].values,
                              preds["home_win_prob"].values])
    btts = preds["btts_prob"].values
    ou = preds["over_2_5_prob"].values

    metrics = compute_metrics(y_true, probs, name)
    btts_m = compute_btts_metrics(hg, ag, btts)
    ou_m = compute_ou_metrics(hg, ag, ou)
    return {**metrics, **btts_m, **ou_m}


def low_scoring_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Return only matches with total goals <= 2 (low-scoring)."""
    total = df["home_goals"].values.astype(float) + df["away_goals"].values.astype(float)
    return df[total <= _LOW_SCORE_MAX_TOTAL].copy()


# ═══════════════════════════════════════════════════════════
#  Main validation
# ═══════════════════════════════════════════════════════════


def run_validation(data_path: str | Path, train_end: str = _DEFAULT_TRAIN_END,
                   test_end: str = _DEFAULT_TEST_END, quiet: bool = False) -> dict[str, Any]:
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "data_path": str(data_path),
        "train_end": train_end,
        "test_end": test_end,
        "models": {"dixon_coles": {}, "poisson": {}},
        "low_scoring": {"dixon_coles": {}, "poisson": {}},
        "comparison": {},
        "low_scoring_comparison": {},
    }

    print("\n" + "=" * 70)
    print("  DIXON-COLES MODEL VALIDATION")
    print("=" * 70)

    # [1/6] Load
    print(f"\n[1/6] Loading data from {data_path}...")
    df = load_data(data_path)
    report["total_matches"] = len(df)

    # [2/6] Split
    print(f"\n[2/6] Splitting (train < {train_end})...")
    df_train, df_test = chronological_split(df, train_end, test_end)
    report["train_size"] = len(df_train)
    report["test_size"] = len(df_test)
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError(f"Empty train ({len(df_train)}) or test ({len(df_test)})")

    # [3/6] Train Poisson baseline
    print(f"\n[3/6] Training Poisson model on {len(df_train)} matches...")
    poisson = PoissonModel(min_matches=0, max_goals=8)
    poisson.add_poisson_features(df_train)
    logger.info("Poisson fitted: mu_home=%.3f, mu_away=%.3f, %d teams",
                poisson.league_avg_home, poisson.league_avg_away, len(poisson.team_strengths))

    # [4/6] Train Dixon-Coles
    print(f"\n[4/6] Training Dixon-Coles MLE model on {len(df_train)} matches...")
    dc = DixonColesModel(
        decay_halflife_days=1460.0,
        use_importance=True,
        max_goals_table=8,
        regress_prior=True,
        prior_strength=0.01,
    )
    dc.fit(df_train, verbose=True)
    logger.info("DC fitted: gamma=%.3f, rho=%.3f, %d teams, converged=%s",
                dc.home_advantage, dc.rho, len(dc.team_attack), dc._optimise_success)

    # [5/6] Evaluate on full test set
    print(f"\n[5/6] Evaluating on {len(df_test)} test matches...")

    dc_metrics = evaluate_predictions(dc, df_test, "DC (full)")
    poisson_metrics = evaluate_predictions(poisson, df_test, "Poisson (full)")

    report["models"]["dixon_coles"] = dc_metrics
    report["models"]["poisson"] = poisson_metrics

    # Comparison on full set
    print("\n  -- Full set comparison --")
    comparison = {}
    for metric in ["accuracy", "log_loss", "brier_score"]:
        if metric in dc_metrics and metric in poisson_metrics:
            d, p = dc_metrics[metric], poisson_metrics[metric]
            diff = d - p
            better = "DC" if ((metric == "accuracy" and diff > 0) or (metric in ["log_loss", "brier_score"] and diff < 0)) else "Poisson" if diff != 0 else "tie"
            comparison[metric] = {"DC": d, "Poisson": p, "difference": round(diff, 4), "better": better}
            pct = f"{abs(diff) / max(abs(p), 0.001) * 100:.1f}%"
            print(f"  {metric:<20s} DC={d:<8.4f} Poisson={p:<8.4f} diff={diff:+.4f} ({pct}) better={better}")
    report["comparison"] = comparison

    # [5b] Low-scoring focus
    df_low = low_scoring_filter(df_test)
    n_low = len(df_low)
    print(f"\n  -- Low-scoring subset ({n_low} matches, total goals <= {_LOW_SCORE_MAX_TOTAL}) --")

    if n_low > 0:
        dc_low = evaluate_predictions(dc, df_low, "DC (low)")
        poisson_low = evaluate_predictions(poisson, df_low, "Poisson (low)")

        report["low_scoring"]["dixon_coles"] = dc_low
        report["low_scoring"]["poisson"] = poisson_low

        low_comparison = {}
        for metric in ["accuracy", "log_loss", "brier_score", "btts_accuracy", "btts_brier"]:
            if metric in dc_low and metric in poisson_low:
                d, p = dc_low[metric], poisson_low[metric]
                diff = d - p
                better = "DC" if ((metric in ["accuracy", "btts_accuracy"] and diff > 0) or (metric in ["log_loss", "brier_score", "btts_brier"] and diff < 0)) else "Poisson" if diff != 0 else "tie"
                low_comparison[metric] = {"DC": d, "Poisson": p, "difference": round(diff, 4), "better": better}
                pct = f"{abs(diff) / max(abs(p), 0.001) * 100:.1f}%"
                print(f"  {metric:<20s} DC={d:<8.4f} Poisson={p:<8.4f} diff={diff:+.4f} ({pct}) better={better}")
        report["low_scoring_comparison"] = low_comparison

        # Expected 2-5% improvement check
        for m in ["log_loss", "brier_score"]:
            if m in low_comparison:
                improvement = abs(low_comparison[m]["difference"]) / max(abs(low_comparison[m]["Poisson"]), 0.001) * 100
                report.setdefault("improvement_pct", {})[m] = round(improvement, 2)
    else:
        print("  No low-scoring matches in test set -- skipping")

    # [6/6] Save outputs
    print(f"\n[6/6] Saving outputs...")

    # Save DC model
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "dixon_coles_model.joblib"
    joblib.dump(dc, model_path)
    report["model_path"] = str(model_path)
    print(f"  Model: {model_path}")

    # Verify model loads
    loaded = joblib.load(model_path)
    assert loaded.fitted, "Loaded DC model must be fitted!"
    print(f"  [OK] Model loads correctly (rho={loaded.rho:.4f}, gamma={loaded.home_advantage:.4f})")

    # Save validation report
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    val_path = _REPORT_DIR / f"dixon_coles_validation_{timestamp}.json"
    with open(val_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {val_path}")

    # Save comparison report
    comp_path = _REPORT_DIR / f"dixon_coles_vs_poisson_{timestamp}.json"
    comp_report = {
        "timestamp": timestamp,
        "models": {"DC": dc_metrics, "Poisson": poisson_metrics},
        "low_scoring": {"DC": report["low_scoring"]["dixon_coles"] if n_low > 0 else None,
                        "Poisson": report["low_scoring"]["poisson"] if n_low > 0 else None},
        "comparison": comparison,
        "low_scoring_comparison": report.get("low_scoring_comparison", {}),
        "improvement_pct": report.get("improvement_pct", {}),
        "n_low_scoring": n_low,
    }
    with open(comp_path, "w") as f:
        json.dump(comp_report, f, indent=2, default=str)
    print(f"  Comparison: {comp_path}")

    # Summary
    duration = time.time() - start_time
    report["duration_seconds"] = round(duration, 2)

    print("\n" + "=" * 70)
    print("  RESULTS".center(68))
    print("=" * 70)
    print(f"  Duration:       {duration:.2f}s")
    print(f"  Train:          {len(df_train)} matches")
    print(f"  Test:           {len(df_test)} matches (low-scoring: {n_low})")
    print(f"  DC teams:       {len(dc.team_attack)}")
    print(f"  DC rho:         {dc.rho:.4f} (tau correction)")
    print(f"  DC gamma:       {dc.home_advantage:.4f} (home adv)")
    print(f"  DC converged:   {dc._optimise_success}")

    imp = report.get("improvement_pct", {})
    if imp:
        print(f"\n  DC improvement on low-scoring subset:")
        for m, pct in imp.items():
            arrow = "improvement" if pct > 0 else "degradation"
            print(f"    {m:<20s} {abs(pct):.1f}% {arrow} vs Poisson")

    print(f"\n  {'Metric':<22s} {'DC Full':<10s} {'Poisson Full':<12s} {'Better':<10s}")
    print(f"  {'-'*54}")
    for m in ["accuracy", "log_loss", "brier_score"]:
        if m in comparison:
            c = comparison[m]
            print(f"  {m:<22s} {c['DC']:<10.4f} {c['Poisson']:<12.4f} {c['better']:<10s}")

    if n_low > 0:
        print(f"\n  {'Low-scoring subset':<22s} {'DC Low':<10s} {'Poisson Low':<12s} {'Better':<10s}")
        print(f"  {'-'*54}")
        for m in ["accuracy", "log_loss", "brier_score", "btts_accuracy"]:
            if m in report.get("low_scoring_comparison", {}):
                c = report["low_scoring_comparison"][m]
                print(f"  {m:<22s} {c['DC']:<10.4f} {c['Poisson']:<12.4f} {c['better']:<10s}")

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE".center(68))
    print("=" * 70)

    return report


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Dixon-Coles vs Poisson")
    parser.add_argument("--data", default=str(config.paths.raw / config.data_collection.output_file))
    parser.add_argument("--train-end", default=_DEFAULT_TRAIN_END)
    parser.add_argument("--test-end", default=_DEFAULT_TEST_END)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    try:
        run_validation(data_path=args.data, train_end=args.train_end,
                       test_end=args.test_end, quiet=args.quiet)
        return 0
    except Exception as e:
        logger.error("Validation failed: %s", e, exc_info=True)
        print(f"\n[FAIL] Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
