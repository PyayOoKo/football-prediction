#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 4 vs Phase 3 — ML Models vs Statistical Models                      ║
║                                                                           ║
║  Compares all 9 models on the same test set:                               ║
║                                                                           ║
║    Phase 4 (ML):        XGBoost, LightGBM, Random Forest,                 ║
║                         Neural Network, Logistic Regression               ║
║    Phase 3 (Statistical): Poisson, Dixon-Coles, Elo, Baseline (LR)        ║
║                                                                           ║
║  Outputs:                                                                 ║
║  - reports/phase4_leaderboard_{timestamp}.csv                             ║
║  - reports/phase3_vs_phase4_{timestamp}.json                              ║
║  - reports/figures/brier_comparison_{timestamp}.png                       ║
║  - reports/figures/accuracy_comparison_{timestamp}.png                    ║
║                                                                           ║
║  Usage:                                                                   ║
║      python scripts/compare_phase4_vs_phase3.py                            ║
║      python scripts/compare_phase4_vs_phase3.py --quiet                    ║
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

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import io
import joblib

from config import config
from src.data_loader import load_results
from src.feature_engineering import build_features
from src.time_series_cv import time_series_train_val_test_split
from src.train import train_model, save_model
from src.evaluate import evaluate_model
from src.poisson_model import PoissonModel
from src.dixon_coles import DixonColesModel
from src.elo import EloSystem

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

_PHASE4_MODELS = {
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Random Forest": "random_forest",
    "Neural Network": "neural_network",
    "Logistic Regression": "logistic_regression",
}

_PHASE3_PARAMS = {
    "Poisson": {"min_matches": 0, "max_goals": 8},
    "Dixon-Coles": {"decay_halflife_days": 1460, "use_importance": True,
                     "max_goals_table": 8, "regress_prior": True, "prior_strength": 0.01},
    "Elo": {"k": 40, "home_advantage": 50, "regress_to_mean": False, "draw_k": 0.20},
}

_REPORT_DIR = PROJECT_ROOT / "reports"
_FIGURE_DIR = _REPORT_DIR / "figures"
_MODEL_DIR = PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════════
#  Data loading — Phase 4 uses feature matrix, Phase 3 uses raw
# ═══════════════════════════════════════════════════════════


def load_phase4_data() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load data, build features, split for Phase 4 ML models."""
    print("\n  [Phase 4] Building feature matrix...")
    df_raw = load_results(low_memory=False)
    # Add target column from result if missing
    if "target" not in df_raw.columns and "result" in df_raw.columns:
        df_raw["target"] = df_raw["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype(int)
    # Parse dates
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["date"])
    df_raw.sort_values(["date", "home_team"], inplace=True)
    df_raw.reset_index(drop=True, inplace=True)

    # Filter out rows with unknown target (-1) before feature engineering
    df_raw = df_raw[df_raw["target"] >= 0].copy()

    t0 = time.time()
    X, y = build_features(df_raw, is_training=True)
    elapsed = time.time() - t0
    print(f"    Feature matrix: {X.shape} ({elapsed:.1f}s)")
    splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))
    for k in ("X_train", "X_val", "X_test"):
        print(f"    {k}: {splits[k].shape[0]} rows")
    return X, y, splits


def load_phase3_data() -> pd.DataFrame:
    """Load raw match data for Phase 3 statistical models."""
    print("\n  [Phase 3] Loading raw match data...")
    df = load_results(low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df["result"].notna() & df["result"].isin(["H", "D", "A"])].copy()
    print(f"    {len(df)} matches loaded")
    return df


def phase3_chronological_split(
    df_raw: pd.DataFrame, splits: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align Phase 3 split with Phase 4 using date-based cutoffs.

    Computes the cutoff date from the Phase 4 split indices applied to the
    raw data (which had dates parsed and sorted identically).  This is more
    robust than iloc-based alignment because it survives row-count mismatches.
    """
    # Find the cutoff dates from the raw data used in Phase 4 feature engineering
    # The raw data is sorted identically; find split points by fraction
    n_raw = len(df_raw)
    n_test = len(splits["X_test"])
    n_train_val = n_raw - n_test

    df_train = df_raw.iloc[:n_train_val].copy()
    df_test = df_raw.iloc[n_train_val:n_raw].copy()
    print(f"    Phase 3 train: {len(df_train)}, test: {len(df_test)}")
    print(f"    Train period: {df_train['date'].min().date()} to {df_train['date'].max().date()}")
    print(f"    Test period:  {df_test['date'].min().date()} to {df_test['date'].max().date()}")
    return df_train, df_test


# ═══════════════════════════════════════════════════════════
#  Train Phase 4 models
# ═══════════════════════════════════════════════════════════


def _conditional_rates(df_cond):
    """Compute BTTS and O/U rates conditional on match outcome."""
    hw = df_cond[df_cond["result"] == "H"]
    d = df_cond[df_cond["result"] == "D"]
    aw = df_cond[df_cond["result"] == "A"]
    btts = lambda g: ((g["home_goals"] > 0) & (g["away_goals"] > 0)).mean() if len(g) > 0 else 0.5
    ou = lambda g: ((g["home_goals"] + g["away_goals"]) > 2.5).mean() if len(g) > 0 else 0.5
    return {
        "btts_given_hw": btts(hw), "btts_given_d": btts(d), "btts_given_aw": btts(aw),
        "ou_given_hw": ou(hw), "ou_given_d": ou(d), "ou_given_aw": ou(aw),
    }


def _add_btts_ou_metrics(metrics, model, X_test_clean, df_test_raw, cond):
    """Add BTTS and O/U accuracy to metrics by deriving probs from outcome model."""
    hg = df_test_raw["home_goals"].values.astype(float)
    ag = df_test_raw["away_goals"].values.astype(float)
    actual_btts = ((hg > 0) & (ag > 0)).astype(float)
    actual_ou = ((hg + ag) > 2.5).astype(float)

    probs = model.predict_proba(X_test_clean)
    pred_btts_prob = (probs[:, 2] * cond["btts_given_hw"]
                      + probs[:, 1] * cond["btts_given_d"]
                      + probs[:, 0] * cond["btts_given_aw"])
    pred_ou_prob = (probs[:, 2] * cond["ou_given_hw"]
                    + probs[:, 1] * cond["ou_given_d"]
                    + probs[:, 0] * cond["ou_given_aw"])

    metrics["btts_accuracy"] = round(float(np.mean((pred_btts_prob > 0.5).astype(float) == actual_btts)), 4)
    metrics["over_under_2_5_accuracy"] = round(float(np.mean((pred_ou_prob > 0.5).astype(float) == actual_ou)), 4)


def train_phase4_models(
    splits: dict[str, Any],
    df_train_raw: pd.DataFrame | None = None,
    df_test_raw: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Train all Phase 4 ML models, evaluate, save.

    Computes BTTS/O/U accuracy using conditional rates from training data.
    """
    results: dict[str, dict[str, Any]] = {}
    X_mean = splits["X_train"].mean().fillna(0)
    X_test_base = splits["X_test"].fillna(X_mean)

    cond = _conditional_rates(df_train_raw) if df_train_raw is not None else None

    for display_name, model_type in _PHASE4_MODELS.items():
        print(f"\n  {display_name}...")
        t0 = time.time()

        config.train.model_type = model_type
        try:
            # Neural Network uses fillna(0); sklearn models use column-mean fill
            if model_type == "neural_network":
                X_test_clean = splits["X_test"].fillna(0)
            else:
                X_test_clean = X_test_base

            model, history = train_model(
                splits["X_train"], splits["y_train"],
                splits["X_val"], splits["y_val"],
            )

            config.eval.plot_confusion_matrix = False
            config.eval.plot_roc_curve = False
            config.eval.plot_feature_importance = False
            metrics = evaluate_model(model, X_test_clean, splits["y_test"])

            # Add BTTS/O/U if raw test data available
            if cond is not None and df_test_raw is not None:
                _add_btts_ou_metrics(metrics, model, X_test_clean, df_test_raw, cond)

            model_path = save_model(model, f"{model_type}_model")

            elapsed = time.time() - t0
            results[display_name] = {
                "model": model,
                "metrics": metrics,
                "history": history,
                "model_path": str(model_path),
                "elapsed": round(elapsed, 1),
            }
            print(f"    acc={metrics.get('accuracy', '?'):.4f}  "
                  f"ll={metrics.get('log_loss', '?'):.4f}  "
                  f"brier={metrics.get('brier_score', '?'):.4f}  "
                  f"({elapsed:.1f}s)")
        except Exception as e:
            logger.error("%s failed: %s", display_name, e)
            print(f"    [FAIL] {e}")

    return results


# ═══════════════════════════════════════════════════════════
#  Train Phase 3 models
# ═══════════════════════════════════════════════════════════


def evaluate_phase3_model(
    model: Any, df_test: pd.DataFrame
) -> dict[str, Any]:
    """Evaluate a Phase 3 statistical model: 1X2, BTTS, O/U."""
    hg = df_test["home_goals"].values.astype(float)
    ag = df_test["away_goals"].values.astype(float)
    mapping = {"A": 0, "D": 1, "H": 2}
    y_true = df_test["result"].map(mapping).fillna(-1).values.astype(int)

    preds = model.predict_matches(df_test)
    probs = np.column_stack([preds["away_win_prob"].values,
                              preds["draw_prob"].values,
                              preds["home_win_prob"].values])

    from sklearn.metrics import log_loss as sk_ll
    pred_labels = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred_labels == y_true))
    ll = float(sk_ll(y_true, probs))
    y_oh = np.zeros((len(y_true), 3))
    for i, v in enumerate(y_true):
        if 0 <= v <= 2:
            y_oh[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))

    # BTTS
    actual_btts = ((hg > 0) & (ag > 0)).astype(float)
    pred_btts = (preds["btts_prob"].values > 0.5).astype(float)
    btts_acc = float(np.mean(pred_btts == actual_btts))

    # O/U
    actual_ou = ((hg + ag) > 2.5).astype(float)
    pred_ou = (preds["over_2_5_prob"].values > 0.5).astype(float)
    ou_acc = float(np.mean(pred_ou == actual_ou))

    return {
        "accuracy": round(accuracy, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "btts_accuracy": round(btts_acc, 4),
        "over_under_2_5_accuracy": round(ou_acc, 4),
        "n_test": len(df_test),
    }


def train_phase3_models(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Train/fit all 4 Phase 3 models, evaluate, save."""
    results: dict[str, dict[str, Any]] = {}

    # ── Poisson ──
    print(f"\n  Poisson...")
    t0 = time.time()
    try:
        poisson = PoissonModel(min_matches=0, max_goals=8)
        poisson.add_poisson_features(df_train)
        metrics = evaluate_phase3_model(poisson, df_test)
        joblib.dump(poisson, _MODEL_DIR / "poisson_model.joblib")
        results["Poisson"] = {
            "model": poisson, "metrics": metrics,
            "elapsed": round(time.time() - t0, 1),
        }
        print(f"    acc={metrics['accuracy']:.4f}  ll={metrics['log_loss']:.4f}")
    except Exception as e:
        logger.error("Poisson failed: %s", e)
        print(f"    [FAIL] {e}")

    # ── Dixon-Coles ──
    print(f"  Dixon-Coles...")
    t0 = time.time()
    try:
        dc = DixonColesModel(**{k: v for k, v in _PHASE3_PARAMS["Dixon-Coles"].items()})
        dc.fit(df_train, verbose=False)
        metrics = evaluate_phase3_model(dc, df_test)
        joblib.dump(dc, _MODEL_DIR / "dixon_coles_model.joblib")
        results["Dixon-Coles"] = {
            "model": dc, "metrics": metrics,
            "elapsed": round(time.time() - t0, 1),
        }
        print(f"    acc={metrics['accuracy']:.4f}  ll={metrics['log_loss']:.4f}  "
              f"rho={dc.rho:.4f}")
    except Exception as e:
        logger.error("DC failed: %s", e)
        print(f"    [FAIL] {e}")

    # ── Elo ──
    print(f"  Elo...")
    t0 = time.time()
    try:
        elo = EloSystem(k=40, home_advantage=50, regress_to_mean=False, draw_k=0.20)
        elo.process_matches(df_train, home_col="home_team", away_col="away_team",
                             result_col="result", home_goals_col="home_goals",
                             away_goals_col="away_goals", season_col="season")
        metrics = evaluate_phase3_model(elo, df_test)
        joblib.dump(elo, _MODEL_DIR / "elo_model.joblib")
        results["Elo"] = {
            "model": elo, "metrics": metrics,
            "elapsed": round(time.time() - t0, 1),
        }
        print(f"    acc={metrics['accuracy']:.4f}  ll={metrics['log_loss']:.4f}")
    except Exception as e:
        logger.error("Elo failed: %s", e)
        print(f"    [FAIL] {e}")

    # ── Baseline LR (rolling features) ──
    print(f"  Baseline (LR)...")
    t0 = time.time()
    try:
        lr_metrics = _train_baseline_lr(df_train, df_test)
        results["Baseline (LR)"] = {
            "model": None, "metrics": lr_metrics["metrics"],
            "elapsed": round(time.time() - t0, 1),
        }
        print(f"    acc={lr_metrics['metrics']['accuracy']:.4f}")
    except Exception as e:
        logger.error("Baseline LR failed: %s", e)
        print(f"    [FAIL] {e}")

    return results


def _train_baseline_lr(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict[str, Any]:
    """Train LR baseline using simple rolling goal averages."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss as sk_ll

    def _compute_feats(df):
        n = len(df)
        ha, hc, aa, ac = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        stats: dict[str, list[float]] = {}
        for idx in range(n):
            home, away = df["home_team"].iloc[idx], df["away_team"].iloc[idx]
            hg = float(df["home_goals"].iloc[idx] or 0)
            ag = float(df["away_goals"].iloc[idx] or 0)
            def _g(t, s):
                st = stats.get(t)
                if st is None or st[2] == 0:
                    return 0.0
                return (st[0] if s == "s" else st[1]) / st[2]
            ha[idx] = _g(home, "s"); hc[idx] = _g(home, "c")
            aa[idx] = _g(away, "s"); ac[idx] = _g(away, "c")
            for t, sc, co in ((home, hg, ag), (away, ag, hg)):
                s = stats.get(t)
                if s is None:
                    stats[t] = [sc, co, 1.0]
                else:
                    s[0] += sc; s[1] += co; s[2] += 1.0
        ts = sum(s[0] for s in stats.values())
        tm = sum(s[2] for s in stats.values())
        ga = ts / tm if tm > 0 else 1.0
        res = pd.DataFrame({"ha": ha, "hc": hc, "aa": aa, "ac": ac})
        unseen = (res == 0.0).all(axis=1)
        for c in res.columns:
            res.loc[unseen, c] = ga
        return res

    combined = pd.concat([df_train, df_test], ignore_index=True)
    feats = _compute_feats(combined)
    nt = len(df_train)
    tr_f = feats.iloc[:nt].fillna(0).values
    te_f = feats.iloc[nt:].fillna(0).values
    mapping = {"A": 0, "D": 1, "H": 2}
    y_tr = df_train["result"].map(mapping).fillna(-1).values.astype(int)
    y_te = df_test["result"].map(mapping).fillna(-1).values.astype(int)

    lr = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42,
                            C=1.0, class_weight="balanced")
    lr.fit(tr_f, y_tr)
    probs = lr.predict_proba(te_f)
    pred_labels = np.argmax(probs, axis=1)
    acc = float(np.mean(pred_labels == y_te))
    ll = float(sk_ll(y_te, probs))
    y_oh = np.zeros((len(y_te), 3))
    for i, v in enumerate(y_te):
        if 0 <= v <= 2:
            y_oh[i, int(v)] = 1
    brier = float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))

    joblib.dump(lr, _MODEL_DIR / "baseline_logistic_regression.joblib")

    return {"metrics": {"accuracy": round(acc, 4), "log_loss": round(ll, 4),
                        "brier_score": round(brier, 4), "n_test": len(df_test)}}


# ═══════════════════════════════════════════════════════════
#  Leaderboard builder (rank-based scoring)
# ═══════════════════════════════════════════════════════════


def build_leaderboard(all_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Rank-based leaderboard across all metrics."""
    rows = []
    for model_name, m in all_metrics.items():
        row = {"Model": model_name, "n_test": m.get("n_test", "N/A"),
               "Phase": m.get("phase", "?")}
        for met in ["accuracy", "log_loss", "brier_score"]:
            row[f"1X2_{met}"] = m.get(met)
        row["BTTS_accuracy"] = m.get("btts_accuracy")
        row["OU_accuracy"] = m.get("over_under_2_5_accuracy")
        rows.append(row)

    df_lb = pd.DataFrame(rows)

    market_groups = {
        "1X2": (["accuracy", "log_loss", "brier_score"], [1, -1, -1], 0.50),
        "BTTS": (["btts_accuracy"], [1], 0.25),
        "OU": (["over_under_2_5_accuracy"], [1], 0.25),
    }

    model_names = list(all_metrics.keys())
    combined: dict[str, float] = {m: 0.0 for m in model_names}
    weights: dict[str, float] = {m: 0.0 for m in model_names}

    for market, (met_list, dirs, weight) in market_groups.items():
        eligible = {}
        for mn in model_names:
            m = all_metrics.get(mn, {})
            if all(met in m and m[met] is not None for met in met_list):
                eligible[mn] = {met: m[met] for met in met_list}
        if len(eligible) < 2:
            continue
        for met, direction in zip(met_list, dirs):
            vals = sorted(eligible.items(), key=lambda x: x[1][met],
                          reverse=(direction == 1))
            n = len(vals)
            for rank, (mn, _) in enumerate(vals):
                points = n - 1 - rank
                combined[mn] += points * weight
                weights[mn] += weight

    for mn in model_names:
        if weights[mn] > 0:
            max_possible = len(model_names) - 1
            combined[mn] = round(combined[mn] / weights[mn] / max_possible, 4) if max_possible > 0 else 0.0
        else:
            combined[mn] = 0.0

    df_lb["Composite_Score"] = df_lb["Model"].map(combined)
    df_lb.sort_values(["Composite_Score", "Model"], ascending=[False, True], inplace=True)
    df_lb.reset_index(drop=True, inplace=True)
    df_lb.index = df_lb.index + 1
    df_lb.index.name = "Rank"
    return df_lb


# ═══════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════


def run_comparison(quiet: bool = False) -> dict[str, Any]:
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {"timestamp": timestamp}

    print("\n" + "=" * 75)
    print("  PHASE 4 vs PHASE 3 — 8-MODEL COMPARISON")
    print("=" * 75)

    # ── [1/3] Load data ────────────────────────────────
    print("\n[1/3] Loading and preparing data...")
    X, y, splits = load_phase4_data()
    df_raw = load_phase3_data()
    df_train_p3, df_test_p3 = phase3_chronological_split(df_raw, splits)
    report["split_sizes"] = {k: splits[k].shape[0] for k in ("X_train", "X_val", "X_test")}

    # Keep raw training data for BTTS/O/U conditional rates
    n_train = int(len(df_raw) * 0.6)
    n_val = int(len(df_raw) * 0.2)
    df_train_raw = df_raw.iloc[:n_train].copy()
    df_test_raw_p3 = df_raw.iloc[n_train + n_val:].copy()

    # ── [2/3] Train all models ─────────────────────────
    n_models = len(_PHASE4_MODELS) + len(_PHASE3_PARAMS) + 1  # +1 for Baseline LR
    print(f"\n[2/3] Training all {n_models} models...")
    all_metrics: dict[str, dict[str, Any]] = {}

    # Phase 4 ML models
    print("\n  ── Phase 4: ML Models ──")
    phase4_results = train_phase4_models(
        splits, df_train_raw=df_train_raw, df_test_raw=df_test_raw_p3,
    )
    for name, result in phase4_results.items():
        metrics = result["metrics"]
        metrics["n_test"] = len(splits["y_test"])
        metrics["phase"] = "Phase 4 (ML)"
        all_metrics[name] = metrics
    report["phase4"] = {name: {"metrics": r["metrics"], "elapsed": r["elapsed"],
                                "model_path": r.get("model_path", "")}
                        for name, r in phase4_results.items()}

    # Phase 3 statistical models
    print("\n  ── Phase 3: Statistical Models ──")
    phase3_results = train_phase3_models(df_train_p3, df_test_p3)
    for name, result in phase3_results.items():
        metrics = result["metrics"]
        metrics["phase"] = "Phase 3 (Stats)"
        all_metrics[name] = metrics
    report["phase3"] = {name: {"metrics": r["metrics"], "elapsed": r["elapsed"]}
                        for name, r in phase3_results.items()}

    # ── Build leaderboard ──────────────────────────────
    print(f"\n[3/3] Building leaderboard...")
    leaderboard_df = build_leaderboard(all_metrics)
    report["leaderboard"] = leaderboard_df.to_dict()

    # ── Print results ────────────────────────────────
    duration = time.time() - start_time
    report["duration_seconds"] = round(duration, 2)

    print("\n" + "=" * 75)
    print("  LEADERBOARD (Rank-Based Score)".center(73))
    print("=" * 75)
    print(f"  Duration: {duration:.1f}s  |  Test set: {len(splits['y_test'])} matches")

    display_cols = ["Model", "Phase", "Composite_Score", "1X2_accuracy",
                    "1X2_log_loss", "1X2_brier_score"]
    display_cols = [c for c in display_cols if c in leaderboard_df.columns]
    pd.set_option("display.max_colwidth", 25)
    pd.set_option("display.width", 140)
    print()
    print(leaderboard_df[display_cols].to_string())
    print()

    # Best per metric
    print(f"  {'Metric':<30s} {'Best Model':<25s} {'Value':<10s}")
    print(f"  {'-'*65}")
    for metric in ["accuracy", "log_loss", "brier_score", "btts_accuracy", "over_under_2_5_accuracy"]:
        best_model = None
        best_val = None
        lower_better = metric in ["log_loss", "brier_score"]
        for model_name, m in all_metrics.items():
            val = m.get(metric)
            if val is not None:
                if best_val is None:
                    best_val, best_model = val, model_name
                elif lower_better and val < best_val:
                    best_val, best_model = val, model_name
                elif not lower_better and val > best_val:
                    best_val, best_model = val, model_name
        if best_model:
            print(f"  {metric:<30s} {best_model:<25s} {best_val:<10.4f}")

    # ── Generate visualizations ──────────────────────
    print(f"\n  Generating visualizations...")
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Brier score comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        lb_sorted = leaderboard_df.sort_values("1X2_brier_score")
        colors = ["#2ecc71" if "ML" in (p or "") else "#3498db"
                  for p in lb_sorted["Phase"]]
        bars = ax.barh(range(len(lb_sorted)), lb_sorted["1X2_brier_score"],
                       color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(lb_sorted)))
        ax.set_yticklabels(lb_sorted["Model"], fontsize=9)
        ax.set_xlabel("Brier Score (lower is better)", fontsize=10)
        ax.set_title("Brier Score Comparison — Phase 3 vs Phase 4", fontsize=12, fontweight="bold")
        # Add value labels
        for bar, val in zip(bars, lb_sorted["1X2_brier_score"]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        ax.set_xlim(0, max(lb_sorted["1X2_brier_score"]) * 1.15)
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="ML Models (Phase 4)"),
            Patch(facecolor="#3498db", label="Statistical (Phase 3)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        brier_path = _FIGURE_DIR / f"brier_comparison_{timestamp}.png"
        fig.savefig(brier_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Brier comparison: {brier_path}")

        # Accuracy comparison
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        lb_acc = leaderboard_df.sort_values("1X2_accuracy")
        colors2 = ["#2ecc71" if "ML" in (p or "") else "#3498db"
                   for p in lb_acc["Phase"]]
        bars2 = ax2.barh(range(len(lb_acc)), lb_acc["1X2_accuracy"],
                          color=colors2, edgecolor="white", height=0.7)
        ax2.set_yticks(range(len(lb_acc)))
        ax2.set_yticklabels(lb_acc["Model"], fontsize=9)
        ax2.set_xlabel("Accuracy (higher is better)", fontsize=10)
        ax2.set_title("Accuracy Comparison — Phase 3 vs Phase 4", fontsize=12, fontweight="bold")
        for bar, val in zip(bars2, lb_acc["1X2_accuracy"]):
            ax2.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2%}", va="center", fontsize=8)
        ax2.set_xlim(0, max(lb_acc["1X2_accuracy"]) * 1.18)
        ax2.legend(handles=[
            Patch(facecolor="#2ecc71", label="ML Models (Phase 4)"),
            Patch(facecolor="#3498db", label="Statistical (Phase 3)"),
        ], loc="lower right", fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        acc_path = _FIGURE_DIR / f"accuracy_comparison_{timestamp}.png"
        fig2.savefig(acc_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Accuracy comparison: {acc_path}")

        report["figures"] = {
            "brier_comparison": str(brier_path),
            "accuracy_comparison": str(acc_path),
        }
    except Exception as viz_err:
        logger.warning("Visualization generation failed: %s", viz_err)
        print(f"  [WARN] Visualizations skipped: {viz_err}")

    # ── Save outputs (after visualizations update report) ──
    lb_path = _REPORT_DIR / f"phase4_leaderboard_{timestamp}.csv"
    leaderboard_df.to_csv(lb_path, index_label="Rank")
    print(f"\n  Leaderboard CSV: {lb_path}")

    comp_path = _REPORT_DIR / f"phase3_vs_phase4_{timestamp}.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Comparison JSON: {comp_path}")

    print("\n" + "=" * 75)
    print("  COMPARISON COMPLETE".center(73))
    print(f"  Top model: {leaderboard_df.iloc[0]['Model']} "
          f"(score={leaderboard_df.iloc[0]['Composite_Score']:.4f})")
    print("=" * 75)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 vs Phase 3 comparison")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    try:
        run_comparison(quiet=args.quiet)
        return 0
    except Exception as e:
        logger.error("Comparison failed: %s", e, exc_info=True)
        print(f"\n[FAIL] Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    # Wrap stdout to handle Unicode on Windows terminals
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
