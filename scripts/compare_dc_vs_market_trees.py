"""
scripts/compare_dc_vs_market_trees.py — Compare DC-only vs DC+market-specific tree models.

Runs both variants on the same F1 backtest data (using real OU odds from CSV) and
compares:
  1. BTTS prediction accuracy (Brier, LogLoss, Accuracy)
  2. Over/Under 2.5 prediction accuracy (Brier, LogLoss, Accuracy)
  3. Over/Under 2.5 value betting performance (yield, profit, etc.)

Usage:
    python scripts/compare_dc_vs_market_trees.py
    python scripts/compare_dc_vs_market_trees.py --leagues E0 F1
    python scripts/compare_dc_vs_market_trees.py --leagues E0 --min-ev 0.10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("compare_dc_vs_market_trees")

from football_data.config import LEAGUE_NAMES

DB_PATH = Path("data/football_data.db")
MODELS_DIR = Path("models/per_league")
CSV_PATH = Path("data/raw/league_all.csv")
MARKET_MODELS_DIR = Path("models") / "per_league"

INITIAL_BANKROLL = 10_000.0
DEFAULT_MIN_EV = 0.05
DEFAULT_KELLY_FRAC = 0.25
BACKTEST_FRAC = 0.15  # last 15% for backtesting


# ── Helpers ────────────────────────────────────────────────


def implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1 else 0.0


def kelly_stake(prob: float, odds: float, fraction: float = 0.25) -> float:
    if odds <= 1 or prob <= 0:
        return 0.0
    full_kelly = (prob * odds - 1.0) / (odds - 1.0)
    return max(0.0, full_kelly * fraction)


def compute_metrics(
    probs: np.ndarray,
    actuals: np.ndarray,
) -> dict[str, float]:
    """Compute Brier, LogLoss, Accuracy for binary predictions."""
    brier = float(np.mean((probs - actuals) ** 2))
    eps = 1e-15
    clipped = np.clip(probs, eps, 1 - eps)
    logloss = float(
        -np.mean(actuals * np.log(clipped) + (1 - actuals) * np.log(1 - clipped))
    )
    preds = (probs > 0.5).astype(float)
    accuracy = float(np.mean(preds == actuals))
    return {
        "brier": round(brier, 4),
        "log_loss": round(logloss, 4),
        "accuracy": round(accuracy, 4),
        "n_matches": len(probs),
    }


def run_value_betting(
    over_probs: np.ndarray,
    df_odds: pd.DataFrame,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_frac: float = DEFAULT_KELLY_FRAC,
    initial_bankroll: float = INITIAL_BANKROLL,
) -> dict[str, Any]:
    """Run over/under value betting using given over probabilities."""
    n = len(df_odds)
    bankroll = initial_bankroll
    bets: list[dict[str, Any]] = []
    bankroll_history: list[dict[str, Any]] = []
    peak_bankroll = initial_bankroll

    for i in range(n):
        row = df_odds.iloc[i]
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        actual_over = (hg + ag) > 2.5

        model_over_prob = float(over_probs[i])
        model_under_prob = 1.0 - model_over_prob

        over_odds = float(row["over_odds"])
        under_odds = float(row["under_odds"])

        outcomes = [
            ("Over 2.5", over_odds, model_over_prob, actual_over),
            ("Under 2.5", under_odds, model_under_prob, not actual_over),
        ]

        for label, odds, model_prob, won in outcomes:
            if odds <= 1 or model_prob <= 0:
                continue
            implied = implied_prob(odds)
            ev = model_prob / implied - 1.0
            if ev < min_ev:
                continue
            stake_pct = kelly_stake(model_prob, odds, kelly_frac)
            if stake_pct <= 0:
                continue
            if bankroll <= 1.0:
                continue
            stake_amount = bankroll * stake_pct
            profit = stake_amount * (odds - 1.0) if won else -stake_amount

            bets.append({
                "date": str(row["date"])[:10],
                "home": row["home_team"],
                "away": row["away_team"],
                "market": label,
                "odds": round(odds, 2),
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied, 4),
                "ev": round(ev, 4),
                "stake_pct": round(stake_pct, 4),
                "stake": round(stake_amount, 2),
                "won": won,
                "profit": round(profit, 2),
                "actual_goals": f"{hg}-{ag}",
            })
            bankroll += profit

        bankroll_history.append({
            "date": str(row["date"])[:10],
            "bankroll": round(bankroll, 2),
        })

    if not bets:
        return {
            "n_bets": 0,
            "metrics": {"n_bets": 0, "yield_pct": 0, "total_profit": 0},
        }

    total_staked = sum(b["stake"] for b in bets)
    total_profit = sum(b["profit"] for b in bets)
    won_bets = [b for b in bets if b["won"]]
    lost_bets = [b for b in bets if not b["won"]]
    win_rate = len(won_bets) / len(bets) if bets else 0
    roi = total_profit / total_staked if total_staked > 0 else 0
    yield_pct = roi * 100

    peak = initial_bankroll
    max_drawdown = 0.0
    for bh in bankroll_history:
        b = bh["bankroll"]
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    gross_wins = sum(b["profit"] for b in won_bets) if won_bets else 0
    gross_losses = abs(sum(b["profit"] for b in lost_bets)) if lost_bets else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    avg_odds = float(np.mean([b["odds"] for b in bets])) if bets else 0
    avg_ev = float(np.mean([b["ev"] for b in bets])) if bets else 0

    over_bets = [b for b in bets if b["market"] == "Over 2.5"]
    under_bets = [b for b in bets if b["market"] == "Under 2.5"]

    metrics = {
        "n_bets": len(bets),
        "n_over_2_5": len(over_bets),
        "n_under_2_5": len(under_bets),
        "n_won": len(won_bets),
        "n_lost": len(lost_bets),
        "win_rate": round(win_rate, 4),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "yield_pct": round(yield_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "final_bankroll": round(bankroll, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "avg_odds": round(avg_odds, 2),
        "avg_ev": round(avg_ev, 4),
    }

    return {"bets": bets, "metrics": metrics, "n_bets": len(bets)}


# ═══════════════════════════════════════════════════════════
#  Data Loading (mirrors backtest_ou_btts.py)
# ═══════════════════════════════════════════════════════════


def load_csv_odds(league: str) -> pd.DataFrame:
    """Load league matches with OU odds from league_all.csv."""
    rows: list[dict[str, Any]] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("league", "").strip() == league:
                try:
                    hg = float(r["home_goals"])
                    ag = float(r["away_goals"])
                except (ValueError, KeyError):
                    continue

                over_odds_str = (r.get("bbav>2.5") or "").strip()
                under_odds_str = (r.get("bbav<2.5") or "").strip()
                if not over_odds_str:
                    over_odds_str = (r.get("avg>2.5") or "").strip()
                if not under_odds_str:
                    under_odds_str = (r.get("avg<2.5") or "").strip()
                if not over_odds_str or not under_odds_str:
                    continue

                try:
                    over_odds = float(over_odds_str)
                    under_odds = float(under_odds_str)
                except ValueError:
                    continue
                if over_odds <= 1 or under_odds <= 1:
                    continue

                rows.append({
                    "date": (r.get("date", "") or "").strip()[:10],
                    "home_team": (r.get("home_team", "") or "").strip(),
                    "away_team": (r.get("away_team", "") or "").strip(),
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "result": (r.get("result", "") or "").strip(),
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_db_data(league: str) -> pd.DataFrame:
    """Load matches from DB for model pre-compute alignment."""
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    return df


def load_league_models(league: str) -> dict[str, Any] | None:
    """Load per-league models (DC, Elo, XGB, LGB, Cat)."""
    import joblib
    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    xgb_path = league_dir / "xgboost.joblib"
    lgb_path = league_dir / "lightgbm.joblib"
    cat_path = league_dir / "catboost.joblib"
    if not dc_path.exists() or not elo_path.exists():
        return None
    models: dict[str, Any] = {
        "dc": joblib.load(dc_path),
        "elo": joblib.load(elo_path),
    }
    if xgb_path.exists():
        models["xgb"] = joblib.load(xgb_path)
    if lgb_path.exists():
        models["lgb"] = joblib.load(lgb_path)
    if cat_path.exists():
        models["cat"] = joblib.load(cat_path)
    return models


# ═══════════════════════════════════════════════════════════
#  Comparison Runner
# ═══════════════════════════════════════════════════════════


def run_comparison(league: str, min_ev: float, kelly_frac: float) -> dict[str, Any]:
    """Run DC-only vs DC+market-trees comparison for a league."""
    from src.dixon_coles import DixonColesModel
    from src.elo import EloSystem
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    league_name = LEAGUE_NAMES.get(league, league)
    logger.info("")
    logger.info("═" * 60)
    logger.info("  %s — %s", league, league_name)
    logger.info("═" * 60)

    # ── Load odds ──
    csv_df = load_csv_odds(league)
    if len(csv_df) < 50:
        logger.error("  Only %d OU odds rows — need 50+, skipping", len(csv_df))
        return {}

    # ── Load DB data (needed for model pre-compute alignment) ──
    db_df = load_db_data(league)
    if len(db_df) < 100:
        logger.error("  Only %d DB matches — need 100+, skipping", len(db_df))
        return {}

    logger.info("  CSV matches with OU odds: %d", len(csv_df))
    logger.info("  DB matches (full history): %d", len(db_df))

    # ── Merge CSV + DB on (date, teams) ──
    csv_df["_key"] = csv_df["home_team"] + "|" + csv_df["away_team"] + "|" + csv_df["date"].astype(str)
    db_df["_key"] = db_df["home_team"] + "|" + db_df["away_team"] + "|" + db_df["date"].astype(str)

    merged = csv_df.merge(
        db_df[["_key", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]],
        on="_key", suffixes=("", "_db"), how="inner",
    )
    if len(merged) < 50:
        logger.error("  Only %d merged rows — skipping", len(merged))
        return {}

    for col in ["date_db", "home_team_db", "away_team_db", "home_goals_db", "away_goals_db", "result_db"]:
        merged.drop(columns=[col], inplace=True, errors="ignore")
    merged = merged.sort_values("date").reset_index(drop=True)
    logger.info("  Merged DB+CSV matches: %d", len(merged))

    # ── Split train/test ──
    split_idx = int(len(merged) * (1 - BACKTEST_FRAC))
    train_df = merged.iloc[:split_idx].copy()
    test_df = merged.iloc[split_idx:].copy()
    logger.info("  Train: %d | Test: %d", len(train_df), len(test_df))

    # ── Full historical data for feature engineering ──
    merged["result_cat"] = merged.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else "A" if r["away_goals"] > r["home_goals"]
        else "D", axis=1
    )

    # ── Load models ──
    models = load_league_models(league)
    if models is None:
        logger.error("  No trained models for %s — skipping", league)
        return {}

    # ── Conditional rates ──
    cr = ConditionalRates.from_data(merged)

    # ── Ensure Elo is rated on all training data ──
    elo = models["elo"]
    # Make sure Elo has been processed through the training data
    # (if it was saved pre-processed, this is a no-op)

    # ═══════════════════════════════════════════════════════════
    #  VARIANT 1: DC-ONLY
    # ═══════════════════════════════════════════════════════════
    logger.info("")
    logger.info("  ── Variant 1: DC-only ──")

    # Create a blend with only DC model
    blend_dc = ThreeModelBlend(
        dc_model=models["dc"],
        elo_model=None,  # No Elo needed for pure DC comparison
        conditional_rates=cr,
        historical_df=merged,
        weights={
            "1X2": {"dc": 1.0},
            "Over2.5": {"dc": 1.0},
            "Over3.5": {"dc": 1.0},
            "BTTS": {"dc": 1.0},
        },
    )

    # Pre-compute DC-only predictions on test set
    ppm_dc = blend_dc.precompute(test_df, cache_key=f"dc_only_{league}_{len(test_df)}")

    dc_over_25 = ppm_dc.dc_over_25  # (n,) P(Over 2.5) from DC
    dc_btts = ppm_dc.dc_btts         # (n,) P(BTTS) from DC

    # Actual outcomes
    actual_over_25 = ((test_df["home_goals"].values + test_df["away_goals"].values) > 2.5).astype(float)
    actual_btts = ((test_df["home_goals"].values > 0) & (test_df["away_goals"].values > 0)).astype(float)

    # Metrics
    dc_over_metrics = compute_metrics(dc_over_25, actual_over_25)
    dc_btts_metrics = compute_metrics(dc_btts, actual_btts)

    # Value betting
    dc_betting = run_value_betting(dc_over_25, test_df, min_ev=min_ev, kelly_frac=kelly_frac)

    logger.info("  OU Brier:    %.4f | LogLoss: %.4f | Acc: %.2f%%",
                dc_over_metrics["brier"], dc_over_metrics["log_loss"],
                dc_over_metrics["accuracy"] * 100)
    logger.info("  BTTS Brier:  %.4f | LogLoss: %.4f | Acc: %.2f%%",
                dc_btts_metrics["brier"], dc_btts_metrics["log_loss"],
                dc_btts_metrics["accuracy"] * 100)
    if dc_betting["n_bets"] > 0:
        logger.info("  Value bets: %d | Yield: %+.2f%% | Profit: GBP %+.0f",
                    dc_betting["metrics"]["n_bets"],
                    dc_betting["metrics"]["yield_pct"],
                    dc_betting["metrics"]["total_profit"])
    else:
        logger.info(f"  Value bets: 0 (none met min_ev={min_ev:.0%})")

    # ═══════════════════════════════════════════════════════════
    #  VARIANT 2: DC + Market-Specific Trees
    # ═══════════════════════════════════════════════════════════
    logger.info("")
    logger.info("  ── Variant 2: DC + Market Trees ──")

    blend_full = ThreeModelBlend(
        dc_model=models["dc"],
        elo_model=models.get("elo"),
        xgb_model=models.get("xgb"),
        lgb_model=models.get("lgb"),
        cat_model=models.get("cat"),
        conditional_rates=cr,
        historical_df=merged,
        weights={
            "1X2": {"dc": 0.35, "elo": 0.25, "xgb": 0.15, "lgb": 0.15, "cat": 0.10},
            "Over2.5": {"dc": 0.10, "xgb": 0.40, "lgb": 0.30, "cat": 0.20},
            "Over3.5": {"dc": 0.30, "xgb": 0.30, "lgb": 0.20, "cat": 0.20},
            "BTTS": {"dc": 0.10, "xgb": 0.40, "lgb": 0.20, "cat": 0.30},
        },
    )

    # Load market-specific models
    market_dir = MARKET_MODELS_DIR / league
    if market_dir.exists():
        result = blend_full.load_market_models(models_dir=market_dir)
        logger.info("  Loaded market models: %d O/U, %d BTTS", result["ou"], result["btts"])
    else:
        logger.warning("  No market models found at %s", market_dir)

    # Use predict_matches() which properly uses market-specific models
    preds_df = blend_full.predict_matches(test_df)

    # Extract predictions
    blend_over_25 = preds_df["over_2_5_prob"].values
    blend_btts = preds_df["btts_prob"].values

    # Metrics
    blend_over_metrics = compute_metrics(blend_over_25, actual_over_25)
    blend_btts_metrics = compute_metrics(blend_btts, actual_btts)

    # Value betting
    blend_betting = run_value_betting(blend_over_25, test_df, min_ev=min_ev, kelly_frac=kelly_frac)

    logger.info("  OU Brier:    %.4f | LogLoss: %.4f | Acc: %.2f%%",
                blend_over_metrics["brier"], blend_over_metrics["log_loss"],
                blend_over_metrics["accuracy"] * 100)
    logger.info("  BTTS Brier:  %.4f | LogLoss: %.4f | Acc: %.2f%%",
                blend_btts_metrics["brier"], blend_btts_metrics["log_loss"],
                blend_btts_metrics["accuracy"] * 100)
    if blend_betting["n_bets"] > 0:
        logger.info("  Value bets: %d | Yield: %+.2f%% | Profit: GBP %+.0f",
                    blend_betting["metrics"]["n_bets"],
                    blend_betting["metrics"]["yield_pct"],
                    blend_betting["metrics"]["total_profit"])
    else:
        logger.info(f"  Value bets: 0 (none met min_ev={min_ev:.0%})")

    # ═══════════════════════════════════════════════════════════
    #  Print Comparison
    # ═══════════════════════════════════════════════════════════
    print()
    print(f"  {'─'*50}")
    print(f"  {'':>5} {'DC-only':>14} {'DC+Trees':>14} {'Δ':>10}")
    print(f"  {'─'*50}")

    # OU metrics
    print(f"  {'OU Brier':<8} {dc_over_metrics['brier']:>14.4f} {blend_over_metrics['brier']:>14.4f} "
          f"{blend_over_metrics['brier']-dc_over_metrics['brier']:>+10.4f}")
    print(f"  {'OU LogLoss':<8} {dc_over_metrics['log_loss']:>14.4f} {blend_over_metrics['log_loss']:>14.4f} "
          f"{blend_over_metrics['log_loss']-dc_over_metrics['log_loss']:>+10.4f}")
    print(f"  {'OU Accuracy':<8} {dc_over_metrics['accuracy']*100:>13.2f}% {blend_over_metrics['accuracy']*100:>13.2f}% "
          f"{(blend_over_metrics['accuracy']-dc_over_metrics['accuracy'])*100:>+9.2f}%")

    # BTTS metrics
    print(f"  {'BTTS Brier':<8} {dc_btts_metrics['brier']:>14.4f} {blend_btts_metrics['brier']:>14.4f} "
          f"{blend_btts_metrics['brier']-dc_btts_metrics['brier']:>+10.4f}")
    print(f"  {'BTTS LogLoss':<8} {dc_btts_metrics['log_loss']:>14.4f} {blend_btts_metrics['log_loss']:>14.4f} "
          f"{blend_btts_metrics['log_loss']-dc_btts_metrics['log_loss']:>+10.4f}")
    print(f"  {'BTTS Accuracy':<8} {dc_btts_metrics['accuracy']*100:>13.2f}% {blend_btts_metrics['accuracy']*100:>13.2f}% "
          f"{(blend_btts_metrics['accuracy']-dc_btts_metrics['accuracy'])*100:>+9.2f}%")

    # Betting metrics
    if dc_betting["n_bets"] > 0 or blend_betting["n_bets"] > 0:
        print(f"  {'─'*50}")
        dc_m = dc_betting["metrics"] if dc_betting["n_bets"] > 0 else {"n_bets": 0, "yield_pct": 0, "total_profit": 0, "win_rate": 0}
        bl_m = blend_betting["metrics"] if blend_betting["n_bets"] > 0 else {"n_bets": 0, "yield_pct": 0, "total_profit": 0, "win_rate": 0}
        print(f"  {'Bets':<8} {dc_m['n_bets']:>14} {bl_m['n_bets']:>14} "
              f"{bl_m['n_bets']-dc_m['n_bets']:>+10}")
        print(f"  {'Win Rate':<8} {dc_m.get('win_rate',0)*100:>13.1f}% {bl_m.get('win_rate',0)*100:>13.1f}% "
              f"{(bl_m.get('win_rate',0)-dc_m.get('win_rate',0))*100:>+9.1f}%")
        print(f"  {'Yield':<8} {dc_m.get('yield_pct',0):>+13.2f}% {bl_m.get('yield_pct',0):>+13.2f}% "
              f"{bl_m.get('yield_pct',0)-dc_m.get('yield_pct',0):>+10.2f}%")
        print(f"  {'Profit':<8} {dc_m.get('total_profit',0):>+13.0f} {bl_m.get('total_profit',0):>+13.0f} "
              f"{bl_m.get('total_profit',0)-dc_m.get('total_profit',0):>+10.0f}")
        print(f"  {'Final BR':<8} {dc_m.get('final_bankroll',INITIAL_BANKROLL):>13.0f} {bl_m.get('final_bankroll',INITIAL_BANKROLL):>13.0f} "
              f"{bl_m.get('final_bankroll',INITIAL_BANKROLL)-dc_m.get('final_bankroll',INITIAL_BANKROLL):>+10.0f}")
        print(f"  {'Max DD':<8} {dc_m.get('max_drawdown_pct',0):>13.1f}% {bl_m.get('max_drawdown_pct',0):>13.1f}% "
              f"{bl_m.get('max_drawdown_pct',0)-dc_m.get('max_drawdown_pct',0):>+10.1f}%")

    print(f"  {'─'*50}")

    return {
        "league": league,
        "n_test": len(test_df),
        "dc": {
            "over": dc_over_metrics,
            "btts": dc_btts_metrics,
            "betting": dc_betting.get("metrics", {}),
        },
        "blend": {
            "over": blend_over_metrics,
            "btts": blend_btts_metrics,
            "betting": blend_betting.get("metrics", {}),
        },
    }


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="DC-only vs DC+market-trees comparison")
    parser.add_argument("--leagues", nargs="+", default=["F1"],
                        help="Leagues to compare (default: F1)")
    parser.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV)
    parser.add_argument("--kelly", type=float, default=DEFAULT_KELLY_FRAC)
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  DC-ONLY vs DC + MARKET-SPECIFIC TREES")
    print(f"  Min EV: {args.min_ev:.0%} | Kelly: {args.kelly:.0%}")
    print("=" * 60)

    all_results: list[dict[str, Any]] = []

    for league in args.leagues:
        result = run_comparison(league, args.min_ev, args.kelly)
        if result:
            all_results.append(result)

    if len(all_results) >= 1:
        print()
        print("=" * 60)
        print("  CROSS-LEAGUE SUMMARY")
        print("=" * 60)
        for r in all_results:
            dc_over_improvement = r["dc"]["over"]["brier"] - r["blend"]["over"]["brier"]
            dc_btts_improvement = r["dc"]["btts"]["brier"] - r["blend"]["btts"]["brier"]
            print(f"  {r['league']} (n={r['n_test']}):")
            print(f"    OU Brier Δ:    {dc_over_improvement:+.4f} ({'BETTER' if dc_over_improvement > 0 else 'WORSE'})")
            print(f"    BTTS Brier Δ:  {dc_btts_improvement:+.4f} ({'BETTER' if dc_btts_improvement > 0 else 'WORSE'})")

    # Save report
    output_path = Path("reports") / "dc_vs_market_trees_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": pd.Timestamp.now().isoformat(),
            "config": {"min_ev": args.min_ev, "kelly_frac": args.kelly},
            "results": all_results,
        }, f, indent=2)
    logger.info("Comparison saved to %s", output_path)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
