"""
optimise_three_model_weights.py — Exhaustive grid-search weight optimisation.

For each market ('1X2', 'Over2.5', 'BTTS', 'Over3.5') we evaluate every
weight combination at step=0.1 on a held-out validation set and keep the
combination with the lowest Brier Score.

Usage:
    python optimise_three_model_weights.py
    python optimise_three_model_weights.py --output config/three_model_weights.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


# ── Search spaces (step = 0.1) ─────────────────────────────

SEARCH_SPACES: dict[str, dict[str, tuple[float, float, float]]] = {
    "1X2": {
        "poisson": (0.3, 0.7, 0.1),
        "elo": (0.2, 0.5, 0.1),
        "xgb": (0.1, 0.3, 0.1),
    },
    "Over2.5": {
        "poisson": (0.3, 0.6, 0.1),
        "elo": (0.0, 0.2, 0.1),
        "xgb": (0.4, 0.7, 0.1),
    },
    "BTTS": {
        "poisson": (0.3, 0.6, 0.1),
        "elo": (0.2, 0.4, 0.1),
        "xgb": (0.2, 0.4, 0.1),
    },
    "Over3.5": {
        "poisson": (0.2, 0.5, 0.1),
        "elo": (0.0, 0.2, 0.1),
        "xgb": (0.5, 0.8, 0.1),
    },
}

# ── Default fallback weights ──────────────────────────────

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "1X2": {"poisson": 0.50, "elo": 0.30, "xgb": 0.20},
    "Over2.5": {"poisson": 0.44, "elo": 0.00, "xgb": 0.56},
    "Over3.5": {"poisson": 0.33, "elo": 0.00, "xgb": 0.67},
    "BTTS": {"poisson": 0.40, "elo": 0.30, "xgb": 0.30},
}


# ── Helpers ───────────────────────────────────────────────

def _build_step_grid(
    space: dict[str, tuple[float, float, float]],
) -> list[dict[str, float]]:
    """Build all weight combinations where each weight is an exact multiple
    of 0.1, within its specified (lo, hi) range, and all sum to 1.0.

    For 3 models this is cheap: iterate the first two at step 0.1 and
    compute the third as 1.0 - w0 - w1, then verify it is within its
    range and an exact step-0.1 increment.
    """
    models = list(space.keys())
    step = 0.1
    combos: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    lo0, hi0, _ = space[models[0]]
    lo1, hi1, _ = space[models[1]]
    lo2, hi2, _ = space[models[2]]

    for w0_10 in range(int(round(lo0 / step)), int(round(hi0 / step)) + 1):
        w0 = round(w0_10 * step, 1)
        for w1_10 in range(int(round(lo1 / step)), int(round(hi1 / step)) + 1):
            w1 = round(w1_10 * step, 1)
            w2 = round(1.0 - w0 - w1, 1)
            # Check w2 is within its range and is an exact step increment
            if w2 < lo2 - 1e-9 or w2 > hi2 + 1e-9:
                continue
            # Verify w2 is a valid step increment
            w2_10 = int(round(w2 / step))
            if abs(w2 - w2_10 * step) > 1e-9:
                continue
            combo = {
                models[0]: round(w0, 4),
                models[1]: round(w1, 4),
                models[2]: round(w2, 4),
            }
            key = tuple(combo.values())
            if key in seen:
                continue
            seen.add(key)
            combos.append(combo)

    logger.info("  Generated %d valid weight combinations", len(combos))
    return combos


def _brier_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score for 1X2 (3 classes)."""
    valid = ~np.isnan(y_true)
    y_v, p_v = y_true[valid], probs[valid]
    y_oh = np.zeros_like(p_v)
    for i, v in enumerate(y_v):
        if 0 <= int(v) <= 2:
            y_oh[i, int(v)] = 1
    return float(np.mean(np.sum((p_v - y_oh) ** 2, axis=1)))


def _brier_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Binary Brier score."""
    valid = ~np.isnan(y_true)
    return float(np.mean((probs[valid] - y_true[valid]) ** 2))


# ── Main optimisation ─────────────────────────────────────

def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load World Cup data and perform a chronological 80/20 split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df) — both sorted chronologically.
    """
    data_path = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    df = df.sort_values("date").reset_index(drop=True)

    # Chronological 80/20 split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()

    logger.info("Data: %d train + %d val = %d total", len(train_df), len(val_df), len(df))
    logger.info("  Train period: %s to %s", train_df["date"].iloc[0], train_df["date"].iloc[-1])
    logger.info("  Val period:   %s to %s", val_df["date"].iloc[0], val_df["date"].iloc[-1])

    return train_df, val_df


def prepare_models(
    train_df: pd.DataFrame,
    xgb_model_path: str | None = None,
) -> tuple[Any, Any, Any]:
    """Fit Poisson and Elo on training data; load XGBoost from disk.

    Returns
    -------
    tuple[PoissonModel, EloSystem, XGBoost model]
    """
    from src.poisson_model import PoissonModel
    from src.elo import EloSystem
    import joblib

    # Poisson
    logger.info("Fitting Poisson model on %d training matches...", len(train_df))
    poisson = PoissonModel(min_matches=0)
    poisson.fit(train_df)
    logger.info("  Poisson fitted — league avg home=%.3f, away=%.3f",
                poisson.league_avg_home, poisson.league_avg_away)

    # Elo
    logger.info("Processing Elo ratings on %d training matches...", len(train_df))
    elo = EloSystem()
    elo.process_matches(train_df)
    logger.info("  Elo fitted — %d teams rated", len(elo._ratings))

    # XGBoost
    xgb = None
    if xgb_model_path:
        path = Path(xgb_model_path)
        if path.exists():
            logger.info("Loading XGBoost from %s", path)
            xgb = joblib.load(path)
            logger.info("  XGBoost loaded")
        else:
            logger.warning("XGBoost model not found at %s — will skip XGBoost blends", path)
    else:
        # Auto-detect: try common paths
        for candidate in [
            PROJECT_ROOT / "models" / "xgboost_model.joblib",
            PROJECT_ROOT / "models" / "worldcup_lightgbm.joblib",
            PROJECT_ROOT / "models" / "ensemble_model.joblib",
        ]:
            if candidate.exists():
                logger.info("Auto-detected XGBoost at %s", candidate)
                try:
                    payload = joblib.load(candidate)
                    # Handle EnsembleModel or dict payloads
                    if hasattr(payload, "predict_proba"):
                        xgb = payload
                    elif isinstance(payload, dict) and "models" in payload:
                        models_dict = payload["models"]
                        # Prefer xgboost inside ensemble
                        xgb = models_dict.get("xgboost", list(models_dict.values())[0])
                    else:
                        xgb = payload
                    logger.info("  Loaded model type: %s", type(xgb).__name__)
                    break
                except Exception as exc:
                    logger.warning("  Failed to load %s: %s", candidate, exc)

        if xgb is None:
            raise RuntimeError(
                "No XGBoost model found. Train one first:\n"
                "  python train_worldcup.py --model xgb"
            )

    return poisson, elo, xgb


def precompute_predictions(
    df: pd.DataFrame,
    poisson: Any,
    elo: Any,
    xgb: Any,
    feature_builder: Any,
) -> dict[str, np.ndarray]:
    """Pre-compute per-model predictions for all matches in df.

    This is the performance-critical step — computing predictions for
    each model ONCE and reusing them across all weight combinations.
    """
    n = len(df)
    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()

    pois_1x2_list: list[np.ndarray] = []
    elo_1x2_list: list[np.ndarray] = []
    pois_btts_list: list[float] = []
    pois_over25_list: list[float] = []
    pois_over35_list: list[float] = []

    for ht, at in zip(home_teams, away_teams):
        # Poisson
        try:
            r = poisson.predict(ht, at)
            pois_1x2_list.append(np.array([r["away_win_prob"], r["draw_prob"], r["home_win_prob"]]))
            pois_btts_list.append(r.get("btts_prob", 0.5))
            pois_over25_list.append(r.get("over_2_5_prob", 0.5))
            pois_over35_list.append(r.get("over_3_5_prob", 0.5))
        except Exception:
            pois_1x2_list.append(np.array([0.33, 0.34, 0.33]))
            pois_btts_list.append(0.5)
            pois_over25_list.append(0.5)
            pois_over35_list.append(0.5)

        # Elo
        try:
            df_single = pd.DataFrame([{"home_team": ht, "away_team": at}])
            elo_1x2_list.append(elo.predict_proba(df_single)[0])
        except Exception:
            elo_1x2_list.append(np.array([0.33, 0.34, 0.33]))

    # XGBoost — batch feature engineering
    xgb_1x2_list: list[np.ndarray] = []
    try:
        X = feature_builder.build(home_teams, away_teams)
        if X is not None and len(X) > 0:
            xgb_raw = xgb.predict_proba(X)
            xgb_1x2_list = [xgb_raw[i] for i in range(len(X))]
        else:
            xgb_1x2_list = [np.array([0.33, 0.34, 0.33])] * n
    except Exception as exc:
        logger.warning("XGBoost batch prediction failed: %s", exc)
        xgb_1x2_list = [np.array([0.33, 0.34, 0.33])] * n

    return {
        "pois_1x2": np.array(pois_1x2_list),
        "elo_1x2": np.array(elo_1x2_list),
        "xgb_1x2": np.array(xgb_1x2_list),
        "pois_btts": np.array(pois_btts_list),
        "pois_over_25": np.array(pois_over25_list),
        "pois_over_35": np.array(pois_over35_list),
    }


def blend_1x2(preds: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    """Weighted blend of 1X2 predictions."""
    wp, we, wx = w.get("poisson", 0), w.get("elo", 0), w.get("xgb", 0)
    total = wp + we + wx
    if total <= 0:
        return preds["pois_1x2"].copy()
    result = (wp / total) * preds["pois_1x2"] + (we / total) * preds["elo_1x2"] + (wx / total) * preds["xgb_1x2"]
    row_sums = result.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return result / row_sums


def blend_binary(preds: dict[str, np.ndarray], w: dict[str, float], market: str,
                 cond: Any) -> np.ndarray:
    """Weighted blend for binary markets (BTTS / Over/Under)."""
    wp, we, wx = w.get("poisson", 0), w.get("elo", 0), w.get("xgb", 0)
    total = wp + we + wx
    if total <= 0:
        return np.full(len(preds["pois_1x2"]), 0.5)

    if market == "BTTS":
        pois_val = preds["pois_btts"]
    elif market == "Over3.5":
        pois_val = preds["pois_over_35"]
    else:
        pois_val = preds["pois_over_25"]

    # Elo & XGBoost derive from 1X2 via conditional rates
    elo_val = cond.btts_from_1x2(preds["elo_1x2"]) if market == "BTTS" else cond.ou_from_1x2(
        preds["elo_1x2"], 3.5 if market == "Over3.5" else 2.5)
    xgb_val = cond.btts_from_1x2(preds["xgb_1x2"]) if market == "BTTS" else cond.ou_from_1x2(
        preds["xgb_1x2"], 3.5 if market == "Over3.5" else 2.5)

    return (wp * pois_val + we * elo_val + wx * xgb_val) / total


def optimise_market(
    market: str,
    combos: list[dict[str, float]],
    preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    cond: Any,
) -> dict[str, Any]:
    """Run grid search for a single market and return the best result."""
    best_score = float("inf")
    best_weights: dict[str, float] = {}
    all_scores: list[tuple[dict[str, float], float]] = []

    for i, w in enumerate(combos):
        if market == "1X2":
            blended = blend_1x2(preds, w)
            score = _brier_1x2(y_true, blended)
        else:
            blended = blend_binary(preds, w, market, cond)
            score = _brier_binary(y_true, blended)

        all_scores.append((dict(w), score))
        if score < best_score:
            best_score = score
            best_weights = dict(w)

    # Compute improvement over default
    default_w = DEFAULT_WEIGHTS.get(market, {})
    if market == "1X2":
        default_blend = blend_1x2(preds, default_w)
        default_score = _brier_1x2(y_true, default_blend)
    else:
        default_blend = blend_binary(preds, default_w, market, cond)
        default_score = _brier_binary(y_true, default_blend)

    improvement = ((default_score - best_score) / default_score * 100) if default_score > 0 else 0

    return {
        "market": market,
        "best_weights": dict(sorted(best_weights.items())),
        "best_brier": round(best_score, 4),
        "default_brier": round(default_score, 4),
        "improvement_pct": round(improvement, 2),
        "combos_evaluated": len(combos),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimise ThreeModelBlend weights via exhaustive grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(PROJECT_ROOT / "config" / "three_model_weights.json"),
        help="Output JSON path (default: config/three_model_weights.json)",
    )
    parser.add_argument(
        "--xgb-model",
        default=None,
        help="Path to XGBoost model file (auto-detected if not provided)",
    )
    args = parser.parse_args(argv)

    t_start = time.time()

    # ── 1. Load data ──────────────────────────────────────
    print()
    print("-" * 60)
    print("  THREE-MODEL BLEND - WEIGHT OPTIMISATION")
    print("-" * 60)

    train_df, val_df = load_and_prepare_data()

    # ── 2. Fit models ─────────────────────────────────────
    print("\n-- Models -------------------------------------")
    poisson, elo, xgb = prepare_models(train_df, args.xgb_model)

    # ── 3. Set up conditional rates from training data ────
    from src.models.three_model_blend import ConditionalRates
    cond = ConditionalRates.from_data(train_df)
    print("\n  Conditional rates from training data:")
    print(f"    BTTS given H/D/A: {cond.btts_given_home_win:.3f} / {cond.btts_given_draw:.3f} / {cond.btts_given_away_win:.3f}")
    print(f"    O/U 2.5 given H/D/A: {cond.ou_given_home_win:.3f} / {cond.ou_given_draw:.3f} / {cond.ou_given_away_win:.3f}")

    # ── 4. Feature builder for XGBoost ────────────────────
    from src.models.three_model_blend import _FeatureBuilder
    fb = _FeatureBuilder(train_df)

    # ── 5. Pre-compute predictions on validation set ──────
    print("\n-- Pre-computing predictions on validation set --")
    val_preds = precompute_predictions(val_df, poisson, elo, xgb, fb)

    # Prepare actual outcomes
    actual_result = val_df["result"].map({"A": 0, "D": 1, "H": 2}).values
    hg = val_df["home_goals"].values.astype(float)
    ag = val_df["away_goals"].values.astype(float)
    actual_btts = ((hg > 0) & (ag > 0)).astype(float)
    actual_ou25 = ((hg + ag) > 2.5).astype(float)
    actual_ou35 = ((hg + ag) > 3.5).astype(float)

    # ── 6. Grid search per market ────────────────────────
    print("\n-- Weight Optimisation --------------------------")
    results: list[dict[str, Any]] = []
    best_weights_all: dict[str, dict[str, float]] = {}

    for market in ["1X2", "Over2.5", "BTTS", "Over3.5"]:
        space = SEARCH_SPACES[market]
        combos = _build_step_grid(space)
        print(f"\n  [{market}] Evaluating {len(combos)} combinations...")

        if market == "1X2":
            y_true = actual_result
        elif market == "Over2.5":
            y_true = actual_ou25
        elif market == "Over3.5":
            y_true = actual_ou35
        elif market == "BTTS":
            y_true = actual_btts
        else:
            continue

        result = optimise_market(market, combos, val_preds, y_true, cond)
        results.append(result)
        best_weights_all[market] = result["best_weights"]

    # ── 7. Print results ─────────────────────────────────
    print("\n" + "-" * 60)
    print("  RESULTS")
    print("-" * 60)

    for r in results:
        w_str = ", ".join(f"{k}={v:.2f}" for k, v in r["best_weights"].items())
        direction = "IMPROVED" if r["improvement_pct"] > 0 else "WORSENED"
        print(f"\n  [{r['market']}] Best Brier: {r['best_brier']:.4f} "
              f"(default: {r['default_brier']:.4f}) — {direction} {abs(r['improvement_pct']):.1f}%")
        print(f"           Weights: {w_str}")
        print(f"           Combinations evaluated: {r['combos_evaluated']}")

    # ── 8. Save weights ──────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "weights": best_weights_all,
                "results": [
                    {
                        "market": r["market"],
                        "best_brier": r["best_brier"],
                        "default_brier": r["default_brier"],
                        "improvement_pct": r["improvement_pct"],
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\n  Saved to: {output_path}")

    # ── 9. Summary ───────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print()
    print("-- Recommended updated DEFAULT_WEIGHTS ---------")
    print("  Copy these into src/models/three_model_blend.py:\n")
    print("  DEFAULT_WEIGHTS = {")
    for mkt in ["1X2", "Over2.5", "BTTS", "Over3.5"]:
        w = best_weights_all.get(mkt, DEFAULT_WEIGHTS.get(mkt, {}))
        w_str = ", ".join(f'"{k}": {v}' for k, v in w.items())
        print(f'      "{mkt}": {{{w_str}}},')
    print("  }")

    return 0


if __name__ == "__main__":
    sys.exit(main())
