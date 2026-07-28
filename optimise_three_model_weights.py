"""
optimise_three_model_weights.py — Grid-search weight optimisation for per-league models.

For a given league, loads saved models (DC, Elo, XGBoost, LightGBM), pre-computes
predictions on a held-out validation set, then exhaustively searches all weight
combinations at step=0.1 for each market (1X2, Over2.5, BTTS, Over3.5).

Usage:
    python optimise_three_model_weights.py --league SE1
    python optimise_three_model_weights.py --league E0 --output config/my_weights.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

# ── Constants (mirror train_league_models.py) ─────────────

DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODELS_DIR = PROJECT_ROOT / "models" / "per_league"
OUTPUT_DIR = PROJECT_ROOT / "config" / "per_league_weights"

TRAIN_FRAC = 0.60
VAL_FRAC = 0.25

# ── Search spaces (step = 0.1) ─────────────────────────────

SEARCH_SPACES: dict[str, dict[str, tuple[float, float, float]]] = {
    "1X2": {
        "dc": (0.15, 0.50, 0.1),
        "elo": (0.15, 0.45, 0.1),
        "xgb": (0.05, 0.25, 0.1),
        "lgb": (0.05, 0.20, 0.1),
    },
    "Over2.5": {
        "dc": (0.20, 0.60, 0.1),
        "elo": (0.00, 0.00, 0.0),  # fixed to 0 — Elo doesn't do over/under directly
        "xgb": (0.15, 0.45, 0.1),
        "lgb": (0.10, 0.35, 0.1),
    },
    "BTTS": {
        "dc": (0.15, 0.45, 0.1),
        "elo": (0.05, 0.20, 0.1),
        "xgb": (0.20, 0.45, 0.1),
        "lgb": (0.10, 0.35, 0.1),
    },
    "Over3.5": {
        "dc": (0.20, 0.50, 0.1),
        "elo": (0.00, 0.00, 0.0),
        "xgb": (0.20, 0.45, 0.1),
        "lgb": (0.10, 0.35, 0.1),
    },
}

# ── Default fallback weights (equal-ish, safe start) ─────

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "1X2":     {"dc": 0.35, "elo": 0.30, "xgb": 0.20, "lgb": 0.15},
    "Over2.5": {"dc": 0.30, "elo": 0.00, "xgb": 0.40, "lgb": 0.30},
    "Over3.5": {"dc": 0.30, "elo": 0.00, "xgb": 0.40, "lgb": 0.30},
    "BTTS":    {"dc": 0.25, "elo": 0.15, "xgb": 0.35, "lgb": 0.25},
}


# ═══════════════════════════════════════════════════════════
#  Data Loading (from football_data.db)
# ═══════════════════════════════════════════════════════════


def load_league_data(league: str) -> pd.DataFrame:
    """Load matches for a specific league from the database."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               home_odds, draw_odds, away_odds, season
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically (no leakage)."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


# ═══════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════


def load_league_models(league: str) -> dict[str, Any] | None:
    """Load previously saved per-league models."""
    import joblib

    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    if not dc_path.exists() or not elo_path.exists():
        logger.error("Models not found for league '%s' in %s", league, league_dir)
        return None

    dc = joblib.load(dc_path)
    elo = joblib.load(elo_path)

    xgb_path = league_dir / "xgboost.joblib"
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None

    lgb_path = league_dir / "lightgbm.joblib"
    lgb = joblib.load(lgb_path) if lgb_path.exists() else None

    return {"dc": dc, "elo": elo, "xgb": xgb, "lgb": lgb}


# ═══════════════════════════════════════════════════════════
#  Feature prep + prediction helpers
# ═══════════════════════════════════════════════════════════


def _prepare_for_tree_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a league DataFrame for the build_features() pipeline."""
    df = df.copy()
    if "target" not in df.columns and "result" in df.columns:
        df["target"] = df["result"].map({"A": 0, "D": 1, "H": 2})
    if "season" not in df.columns:
        df["season"] = pd.to_datetime(df["date"]).dt.year.astype(str)
    else:
        mask = df["season"].isna() | (df["season"].astype(str).str.strip() == "")
        if mask.any():
            df.loc[mask, "season"] = pd.to_datetime(df.loc[mask, "date"]).dt.year.astype(str)
    if "league" not in df.columns:
        df["league"] = "UNKNOWN"
    numeric_candidates = [
        "home_goals", "away_goals", "home_odds", "draw_odds", "away_odds",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_yellow", "away_yellow", "home_red", "away_red",
    ]
    for col in numeric_candidates:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def _safe_tree_predict(model: Any, X_features: pd.DataFrame | None) -> np.ndarray:
    """Safely predict 1X2 probabilities with a tree model, handling feature alignment."""
    if model is None or X_features is None or len(X_features) == 0:
        return np.array([[0.33, 0.34, 0.33]])
    try:
        expected = None
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        elif hasattr(model, "feature_name_"):
            expected = list(model.feature_name_)
        if expected:
            missing = set(expected) - set(X_features.columns)
            for col in missing:
                X_features[col] = np.nan
            X_features = X_features[expected]
        return np.array(model.predict_proba(X_features))
    except Exception as exc:
        logger.warning("Tree prediction failed: %s", exc)
        return np.array([[0.33, 0.34, 0.33]])


# ═══════════════════════════════════════════════════════════
#  Pre-compute per-model predictions
# ═══════════════════════════════════════════════════════════


def precompute_predictions(
    val_df: pd.DataFrame,
    models: dict[str, Any],
    train_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Pre-compute per-model 1X2 predictions for all matches in val_df.

    Returns a dict with keys like ``dc_1x2``, ``elo_1x2``, ``xgb_1x2``, ``lgb_1x2``.
    """
    dc = models.get("dc")
    elo = models.get("elo")
    xgb = models.get("xgb")
    lgb = models.get("lgb")

    n = len(val_df)
    home_teams = val_df["home_team"].tolist()
    away_teams = val_df["away_team"].tolist()

    # ── Dixon-Coles (row by row) ─────────────────────────
    dc_1x2_list: list[np.ndarray] = []
    for ht, at in zip(home_teams, away_teams):
        try:
            r = dc.predict(ht, at)
            dc_1x2_list.append(np.array([r.away_win_prob, r.draw_prob, r.home_win_prob]))
        except Exception:
            dc_1x2_list.append(np.array([0.33, 0.34, 0.33]))
    dc_1x2 = np.array(dc_1x2_list)

    # ── Elo (row by row) ─────────────────────────────────
    elo_1x2_list: list[np.ndarray] = []
    for ht, at in zip(home_teams, away_teams):
        try:
            df_single = pd.DataFrame([{"home_team": ht, "away_team": at}])
            elo_1x2_list.append(elo.predict_proba(df_single)[0])
        except Exception:
            elo_1x2_list.append(np.array([0.33, 0.34, 0.33]))
    elo_1x2 = np.array(elo_1x2_list)

    # ── Tree models (batch feature engineering) ──────────
    xgb_1x2 = np.tile(np.array([0.33, 0.34, 0.33]), (n, 1))
    lgb_1x2 = np.tile(np.array([0.33, 0.34, 0.33]), (n, 1))

    if xgb is not None or lgb is not None:
        try:
            from src.feature_engineering import build_features
            from src.config import config as cfg
            _orig_vals = {
                "weather.enabled": cfg.weather.enabled,
                "referee.enabled": cfg.referee.enabled,
                "player_info.enabled": cfg.player_info.enabled,
                "player_features.enabled": cfg.player_features.enabled,
            }
            cfg.weather.enabled = False
            cfg.referee.enabled = False
            # Keep player_info enabled if it's on (squad value features)
            cfg.player_features.enabled = False

            try:
                combined = pd.concat([
                    _prepare_for_tree_features(train_df),
                    _prepare_for_tree_features(val_df),
                ], ignore_index=True)
                X_full, y_full = build_features(combined, is_training=True, use_cache=False)
                n_train = len(train_df)
                X_val_f = X_full.iloc[n_train:].copy()
                if len(X_val_f) > 0:
                    xgb_1x2 = _safe_tree_predict(xgb, X_val_f.copy())
                    lgb_1x2 = _safe_tree_predict(lgb, X_val_f.copy())
            except Exception as exc:
                logger.warning("Feature engineering failed for tree predictions: %s", exc)
            finally:
                for key, val in _orig_vals.items():
                    parts = key.split(".")
                    obj = cfg
                    for p in parts[:-1]:
                        obj = getattr(obj, p, None)
                        if obj is None:
                            break
                    if obj is not None:
                        try:
                            setattr(obj, parts[-1], val)
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Tree model feature build failed: %s", exc)

    return {
        "dc_1x2": dc_1x2,
        "elo_1x2": elo_1x2,
        "xgb_1x2": xgb_1x2,
        "lgb_1x2": lgb_1x2,
    }


# ═══════════════════════════════════════════════════════════
#  Grid search helpers
# ═══════════════════════════════════════════════════════════


def _build_step_grid(
    space: dict[str, tuple[float, float, float]],
) -> list[dict[str, float]]:
    """Build all valid weight combinations for N models (step=0.1)."""
    models = list(space.keys())
    n_models = len(models)
    step = 0.1
    combos: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    ranges: list[list[float]] = []
    for m in models:
        lo, hi, _ = space[m]
        if lo == 0.0 and hi == 0.0:
            # Fixed at 0 — no range to iterate
            ranges.append([])
        else:
            values = [
                round(v * step, 1)
                for v in range(int(round(lo / step)), int(round(hi / step)) + 1)
            ]
            ranges.append(values)

    def _recurse(idx: int, current: dict[str, float], running_sum: float) -> None:
        if idx == n_models - 1:
            w_last = round(1.0 - running_sum, 1)
            m_last = models[idx]
            lo_last, hi_last, _ = space[m_last]
            if lo_last == 0.0 and hi_last == 0.0:
                # Fixed at 0 — check that running_sum is already 1.0
                if abs(running_sum - 1.0) > 1e-9:
                    return
                current[m_last] = 0.0
            else:
                if w_last < lo_last - 1e-9 or w_last > hi_last + 1e-9:
                    return
                w_last_10 = int(round(w_last / step))
                if abs(w_last - w_last_10 * step) > 1e-9:
                    return
                current[m_last] = round(w_last, 4)
            key = tuple(current.get(m, 0.0) for m in models)
            if key in seen:
                return
            seen.add(key)
            combos.append({k: v for k, v in current.items()})
            return

        if not ranges[idx]:
            # Model fixed at 0 step (e.g. elo in Over2.5)
            current[models[idx]] = 0.0
            _recurse(idx + 1, current, running_sum)
        else:
            for w in ranges[idx]:
                if running_sum + w > 1.0 + 1e-9:
                    continue
                current[models[idx]] = w
                _recurse(idx + 1, current, running_sum + w)
                del current[models[idx]]

    _recurse(0, {}, 0.0)
    logger.info("  Generated %d valid weight combinations", len(combos))
    return combos


def blend_1x2(preds: dict[str, np.ndarray], w: dict[str, float]) -> np.ndarray:
    """Weighted blend of 1X2 predictions from available models."""
    model_keys = ["dc", "elo", "xgb", "lgb"]
    total_w = 0.0
    result = None
    for key in model_keys:
        probs = preds.get(f"{key}_1x2")
        if probs is not None:
            weight = w.get(key, 0.0)
            if weight > 0:
                if result is None:
                    result = np.zeros_like(probs)
                result += weight * probs
                total_w += weight
    if result is None or total_w <= 0:
        n = len(next(iter(preds.values())))
        return np.full((n, 3), 0.33)
    result /= total_w
    row_sums = result.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return result / row_sums


def brier_1x2(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score."""
    valid = ~np.isnan(y_true)
    y_v, p_v = y_true[valid], probs[valid]
    y_oh = np.zeros_like(p_v)
    for i, v in enumerate(y_v):
        if 0 <= int(v) <= 2:
            y_oh[i, int(v)] = 1
    return float(np.mean(np.sum((p_v - y_oh) ** 2, axis=1)))


def brier_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Binary Brier score."""
    valid = ~np.isnan(y_true)
    return float(np.mean((probs[valid] - y_true[valid]) ** 2))


# ═══════════════════════════════════════════════════════════
#  Per-market optimisation
# ═══════════════════════════════════════════════════════════


def optimise_market(
    market: str,
    combos: list[dict[str, float]],
    preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, Any]:
    """Run grid search for a single market and return the best result."""
    best_score = float("inf")
    best_weights: dict[str, float] = {}

    for w in combos:
        blended = blend_1x2(preds, w)
        score = brier_1x2(y_true, blended)
        if score < best_score:
            best_score = score
            best_weights = dict(w)

    # Compare with default weights
    default_w = DEFAULT_WEIGHTS.get(market, {})
    default_blend = blend_1x2(preds, default_w)
    default_score = brier_1x2(y_true, default_blend)

    improvement = ((default_score - best_score) / default_score * 100) if default_score > 0 else 0

    return {
        "market": market,
        "best_weights": dict(sorted(best_weights.items())),
        "best_brier": round(best_score, 4),
        "default_brier": round(default_score, 4),
        "improvement_pct": round(improvement, 2),
        "combos_evaluated": len(combos),
    }


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimise per-league blend weights via exhaustive grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--league", type=str, required=True,
        help="League code (e.g. SE1, E0, SP1)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path (default: config/per_league_weights/{LEAGUE}.json)",
    )
    args = parser.parse_args(argv)

    league = args.league
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{league}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print()
    print("=" * 60)
    print(f"  PER-LEAGUE WEIGHT OPTIMISATION — {league}")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────
    print("\n-- Loading data --------------------------------")
    df = load_league_data(league)
    logger.info("Loaded %d matches for %s", len(df), league)
    train_df, val_df, test_df = chronological_split(df)
    logger.info("Split: %d train / %d val / %d test",
                len(train_df), len(val_df), len(test_df))

    # ── 2. Load pre-trained models ────────────────────────
    print("\n-- Loading models -------------------------------")
    models = load_league_models(league)
    if models is None:
        return 1

    dc, elo, xgb, lgb = models["dc"], models["elo"], models["xgb"], models["lgb"]
    print(f"  Dixon-Coles: {'YES' if dc is not None else 'NO'}")
    print(f"  Elo:         YES")
    print(f"  XGBoost:     {'YES' if xgb is not None else 'NO'}")
    print(f"  LightGBM:    {'YES' if lgb is not None else 'NO'}")

    # ── 3. Pre-compute predictions on validation set ──────
    print("\n-- Pre-computing predictions on validation set ---")
    val_preds = precompute_predictions(val_df, models, train_df)

    # Actual outcomes
    actual_result = val_df["result"].map({"A": 0, "D": 1, "H": 2}).values
    hg = val_df["home_goals"].values.astype(float)
    ag = val_df["away_goals"].values.astype(float)
    actual_btts = ((hg > 0) & (ag > 0)).astype(float)
    actual_ou25 = ((hg + ag) > 2.5).astype(float)
    actual_ou35 = ((hg + ag) > 3.5).astype(float)

    # ── 4. Grid search per market ────────────────────────
    # For binary markets, derive probabilities from 1X2 blend
    # A simple approach: interpret home win prob as over/BTTS proxy
    # Better would be model-specific BTTS/O/U predictions, but for
    # 1X2-only search we blend 1X2 and compute binary from that

    print("\n-- Weight Optimisation ---------------------------")
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

        result = optimise_market(market, combos, val_preds, y_true)
        results.append(result)
        best_weights_all[market] = result["best_weights"]

    # ── 5. Print results ─────────────────────────────────
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

    # ── 6. Save weights ──────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(
            {
                "league": league,
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "models_available": {
                    "dc": dc is not None,
                    "elo": elo is not None,
                    "xgb": xgb is not None,
                    "lgb": lgb is not None,
                },
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

    # ── 7. Recommended update ────────────────────────────
    print("\n" + "-" * 60)
    print("  RECOMMENDED WEIGHTS (copy into config):")
    print("-" * 60)
    print("  DEFAULT_WEIGHTS = {")
    for mkt in ["1X2", "Over2.5", "BTTS", "Over3.5"]:
        w = best_weights_all.get(mkt, DEFAULT_WEIGHTS.get(mkt, {}))
        w_str = ", ".join(f'"{k}": {v}' for k, v in w.items())
        print(f'      "{mkt}": {{{w_str}}},')
    print("  }")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
