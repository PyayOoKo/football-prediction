"""
backtest_ou_btts.py — Backtest Over/Under 2.5 and BTTS betting profitability on per-league models.

For each league:
1. Loads per-league models (DC + Elo + XGB + LGB)
2. Builds a ThreeModelBlend with market-specific weights
3. Pre-computes predictions for ALL historical matches (from DB)
4. For Over/Under 2.5: compares model's over_2_5_prob vs bookmaker odds
   from league_all.csv (bbav>2.5 / bbav<2.5) to find value bets
5. For BTTS: evaluates prediction accuracy (Brier, LogLoss) since no
   direct BTTS odds exist in the data
6. Generates profitability report per league

Usage:
    python backtest_ou_btts.py                           # Test all leagues with models
    python backtest_ou_btts.py --leagues E0 D1           # Specific leagues
    python backtest_ou_btts.py --leagues E0 --optimise   # Scan EV/Kelly params
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest_ou_btts")

from football_data.config import LEAGUE_NAMES

DB_PATH = Path("data/football_data.db")
MODELS_DIR = Path("models/per_league")
CSV_PATH = Path("data/raw/league_all.csv")
REPORTS_DIR = Path("reports")

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


# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════

def load_csv_odds(league: str) -> pd.DataFrame:
    """Load league matches with OU odds from league_all.csv.

    Uses `bbav>2.5` (best average Over 2.5 odds) and `bbav<2.5`
    (best average Under 2.5 odds) as the bookmaker price.

    Columns standardised to match DB naming:
        date, home_team, away_team, home_goals, away_goals, result,
        over_odds, under_odds
    """
    rows: list[dict[str, Any]] = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("league", "").strip() == league:
                # Validate we have goals and OU odds
                try:
                    hg = float(r["home_goals"])
                    ag = float(r["away_goals"])
                except (ValueError, KeyError):
                    continue

                over_odds_str = (r.get("bbav>2.5") or "").strip()
                under_odds_str = (r.get("bbav<2.5") or "").strip()

                # Try avg>2.5 / avg<2.5 as fallback
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

                date_str = r.get("date", "").strip()[:10]

                result_code = r.get("result", "").strip()
                row_out = {
                    "date": date_str,
                    "home_team": r.get("home_team", "").strip(),
                    "away_team": r.get("away_team", "").strip(),
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "result": result_code,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                }
                rows.append(row_out)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_db_data(league: str) -> pd.DataFrame:
    """Load matches with goals from DB (for model pre-compute alignment)."""
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


def load_ou_from_1x2(league: str) -> pd.DataFrame:
    """Derive Over/Under 2.5 odds from 1X2 odds using ConditionalRates.

    For leagues where direct OU odds are unavailable (e.g. SE1, NO2),
    this converts the bookmaker's 1X2 odds into implied OU probabilities
    using league-specific conditional rates:

        P(Over) = P(H) * P(Over|H) + P(D) * P(Over|D) + P(A) * P(Over|A)

    Where P(H), P(D), P(A) are the bookmaker's implied probabilities from
    1X2 odds (after removing the overround), and conditional rates are
    computed from historical match results.
    """
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               home_odds, draw_odds, away_odds
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
          AND home_odds IS NOT NULL AND draw_odds IS NOT NULL AND away_odds IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()

    if len(df) < 100:
        return pd.DataFrame()

    # Compute conditional rates from ALL historical data
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["is_over"] = (df["total_goals"] > 2.5).astype(float)

    cond_rates: dict[str, float] = {}
    for outcome, label in [("H", "home_win"), ("D", "draw"), ("A", "away_win")]:
        subset = df[df["result"] == outcome]
        if len(subset) > 0:
            cond_rates[label] = float(subset["is_over"].mean())
        else:
            cond_rates[label] = 0.5

    logger.info("  Conditional OU rates: H=%.3f D=%.3f A=%.3f",
                cond_rates["home_win"], cond_rates["draw"], cond_rates["away_win"])

    # For each match, convert 1X2 odds to implied OU probability
    def derive_ou(row: pd.Series) -> tuple[float, float]:
        h_odds, d_odds, a_odds = row["home_odds"], row["draw_odds"], row["away_odds"]
        if h_odds <= 1 or d_odds <= 1 or a_odds <= 1:
            return (2.0, 2.0)  # fallback

        # Convert odds to implied probabilities
        p_h = 1.0 / h_odds
        p_d = 1.0 / d_odds
        p_a = 1.0 / a_odds
        total = p_h + p_d + p_a
        if total <= 0:
            return (2.0, 2.0)

        # Remove overround (assume bookmaker margin is distributed proportionally)
        p_h_norm = p_h / total
        p_d_norm = p_d / total
        p_a_norm = p_a / total

        # Convert to implied OU probability
        implied_over = (
            p_h_norm * cond_rates["home_win"]
            + p_d_norm * cond_rates["draw"]
            + p_a_norm * cond_rates["away_win"]
        )
        implied_over = np.clip(implied_over, 0.05, 0.95)
        implied_under = 1.0 - implied_over

        # Convert back to decimal odds
        over_odds = round(1.0 / implied_over, 2)
        under_odds = round(1.0 / implied_under, 2)
        return (over_odds, under_odds)

    derived = df.apply(derive_ou, axis=1, result_type="expand")
    df["over_odds"] = derived[0]
    df["under_odds"] = derived[1]

    # Keep only the columns we need
    result_df = df[["date", "home_team", "away_team", "home_goals", "away_goals", "result", "over_odds", "under_odds"]].copy()
    result_df["date"] = pd.to_datetime(result_df["date"], errors="coerce")
    result_df = result_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    logger.info("  Derived OU odds from 1X2 for %d matches", len(result_df))
    return result_df


def load_league_models(league: str) -> dict[str, Any] | None:
    """Load per-league models.

    Prefers the new per-league ``dc_model.joblib`` (from blend quick-test)
    over the older ``dixon_coles.joblib`` so backtests reflect the latest
    Dixon-Coles fits.
    """
    import joblib
    league_dir = MODELS_DIR / league

    # Try new per-league DC model first, fall back to old one
    dc_new = league_dir / "dc_model.joblib"
    dc_old = league_dir / "dixon_coles.joblib"
    if dc_new.exists():
        dc_path = dc_new
        logger.info("  Using NEW per-league DC model: %s", dc_new.name)
    elif dc_old.exists():
        dc_path = dc_old
        logger.info("  Using OLD per-league DC model: %s", dc_old.name)
    else:
        return None

    elo_path = league_dir / "elo.joblib"
    xgb_path = league_dir / "xgboost.joblib"
    lgb_path = league_dir / "lightgbm.joblib"

    if not elo_path.exists():
        return None

    models: dict[str, Any] = {
        "dc": joblib.load(dc_path),
        "elo": joblib.load(elo_path),
    }
    if xgb_path.exists():
        models["xgb"] = joblib.load(xgb_path)
    if lgb_path.exists():
        models["lgb"] = joblib.load(lgb_path)
    return models


# ═══════════════════════════════════════════════════════════
#  OU Value Betting Backtest
# ═══════════════════════════════════════════════════════════

def run_ou_backtest(
    df: pd.DataFrame,
    models: dict[str, Any],
    df_full: pd.DataFrame,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_frac: float = DEFAULT_KELLY_FRAC,
    initial_bankroll: float = INITIAL_BANKROLL,
) -> dict[str, Any]:
    """Run Over/Under 2.5 value betting backtest using the full 4-model blend.

    For each match:
    1. Blend model predicts over_2_5_prob and under_2_5_prob
    2. Compare vs implied odds from bbav>2.5 / bbav<2.5
    3. Place Kelly-sized bet on Over if model sees value
    4. Place Kelly-sized bet on Under if model sees value
    5. Track bankroll, ROI, yield, max drawdown

    Returns full results dict.
    """
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    # Build the blend
    cr = ConditionalRates.from_data(df_full)
    blend = ThreeModelBlend(
        dc_model=models.get("dc"),
        elo_model=models.get("elo"),
        xgb_model=models.get("xgb"),
        lgb_model=models.get("lgb"),
        conditional_rates=cr,
        historical_df=df_full,
    )

    # Pre-compute predictions for ALL matches in df (aligns with CSV rows)
    logger.info("  Pre-computing blend predictions for %d matches...", len(df))
    ppm = blend.precompute(df, cache_key=f"backtest_{len(df)}")

    # Blend Over 2.5 probabilities using market weights
    w_ou = blend.weights.get("Over2.5", {})
    blend_over_probs = blend._blend_binary(ppm, w_ou, "Over2.5")  # P(Over 2.5)

    # Blend BTTS probabilities for BTTS evaluation
    w_btts = blend.weights.get("BTTS", {})
    blend_btts_probs = blend._blend_binary(ppm, w_btts, "BTTS")  # P(BTTS)

    # ── Run backtest ──
    bankroll = initial_bankroll
    bets: list[dict[str, Any]] = []
    bankroll_history: list[dict[str, Any]] = []
    peak_bankroll = initial_bankroll

    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        actual_over = (hg + ag) > 2.5
        actual_btts = (hg > 0) and (ag > 0)

        model_over_prob = float(blend_over_probs[i])
        model_under_prob = 1.0 - model_over_prob
        model_btts_prob = float(blend_btts_probs[i])

        over_odds = float(row["over_odds"])
        under_odds = float(row["under_odds"])

        outcomes = [
            ("Over 2.5", over_odds, model_over_prob, actual_over),
            ("Under 2.5", under_odds, model_under_prob, not actual_over),
        ]

        for label, odds, model_prob, actual_won in outcomes:
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
            won = actual_won
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

    # ── BTTS metrics (no odds available — just prediction accuracy) ──
    actual_btts_arr = (
        (df["home_goals"].values > 0) & (df["away_goals"].values > 0)
    ).astype(float)
    brier_btts = float(np.mean((blend_btts_probs - actual_btts_arr) ** 2))
    btts_preds = (blend_btts_probs > 0.5).astype(float)
    btts_accuracy = float(np.mean(btts_preds == actual_btts_arr))
    eps = 1e-15
    btts_clipped = np.clip(blend_btts_probs, eps, 1 - eps)
    btts_logloss = float(
        -np.mean(
            actual_btts_arr * np.log(btts_clipped)
            + (1 - actual_btts_arr) * np.log(1 - btts_clipped)
        )
    )

    btts_metrics = {
        "brier": round(brier_btts, 4),
        "log_loss": round(btts_logloss, 4),
        "accuracy": round(btts_accuracy, 4),
        "n_matches": n,
        "btts_rate": float(actual_btts_arr.mean()),
    }

    # ── Compute OU betting metrics ──
    if not bets:
        return {
            "bets": [],
            "metrics": {"n_bets": 0, "market": "Over2.5"},
            "btts_metrics": btts_metrics,
            "bankroll_history": bankroll_history,
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

    # Split by Over vs Under
    over_bets = [b for b in bets if b["market"] == "Over 2.5"]
    under_bets = [b for b in bets if b["market"] == "Under 2.5"]

    metrics = {
        "market": "Over2.5",
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
        "avg_stake_pct": round(float(np.mean([b["stake_pct"] for b in bets])), 4),
        "over_win_rate": round(len([b for b in over_bets if b["won"]]) / len(over_bets), 4) if over_bets else 0,
        "under_win_rate": round(len([b for b in under_bets if b["won"]]) / len(under_bets), 4) if under_bets else 0,
        "over_yield": round(
            sum(b["profit"] for b in over_bets)
            / sum(b["stake"] for b in over_bets) * 100
            if over_bets and sum(b["stake"] for b in over_bets) > 0 else 0, 2
        ),
        "under_yield": round(
            sum(b["profit"] for b in under_bets)
            / sum(b["stake"] for b in under_bets) * 100
            if under_bets and sum(b["stake"] for b in under_bets) > 0 else 0, 2
        ),
    }

    return {
        "bets": bets,
        "metrics": metrics,
        "btts_metrics": btts_metrics,
        "bankroll_history": bankroll_history,
    }


# ═══════════════════════════════════════════════════════════
#  Parameter Optimisation
# ═══════════════════════════════════════════════════════════

def scan_parameters(
    df: pd.DataFrame,
    models: dict[str, Any],
    df_full: pd.DataFrame,
) -> dict[str, Any]:
    """Scan min_ev and kelly_frac combos for optimal settings."""
    results = []
    for min_ev in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        for kelly_frac in [0.10, 0.25, 0.50, 1.0]:
            bt = run_ou_backtest(df, models, df_full, min_ev=min_ev, kelly_frac=kelly_frac)
            m = bt["metrics"]
            results.append({
                "min_ev": min_ev,
                "kelly_frac": kelly_frac,
                "n_bets": m["n_bets"],
                "yield_pct": m.get("yield_pct", 0),
                "profit": m.get("total_profit", 0),
                "max_dd": m.get("max_drawdown_pct", 0),
                "win_rate": m.get("win_rate", 0),
                "final_bankroll": m.get("final_bankroll", INITIAL_BANKROLL),
            })

    best_profit = max(results, key=lambda r: r["final_bankroll"]) if results else {}
    viable = [r for r in results if r["n_bets"] >= 20]
    best_yield = max(viable, key=lambda r: r["yield_pct"]) if viable else best_profit

    def safety_score(r):
        return r["yield_pct"] / max(r["max_dd"], 0.1) if r["n_bets"] >= 10 else 0
    safest = max(results, key=safety_score) if results else {}

    return {
        "all_results": results,
        "best_profit": best_profit,
        "best_yield": best_yield,
        "safest": safest,
    }


# ═══════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════

def print_report(league: str, bt: dict[str, Any], params: dict[str, Any] | None = None):
    """Print readable OU/BTTS backtest report."""
    metrics = bt["metrics"]
    btts_m = bt["btts_metrics"]
    league_name = LEAGUE_NAMES.get(league, league)
    bets = bt["bets"]

    print()
    print("=" * 70)
    print(f"  {league} - {league_name}")
    print("=" * 70)

    # ── BTTS Prediction Accuracy (no odds) ──
    print()
    print(f"  📊 BTTS PREDICTION ACCURACY (all {btts_m['n_matches']} matches)")
    print(f"  {'Brier':<25} {btts_m['brier']:<10}")
    print(f"  {'LogLoss':<25} {btts_m['log_loss']:<10}")
    print(f"  {'Accuracy':<25} {btts_m['accuracy']*100:.2f}%")
    print(f"  {'Actual BTTS rate':<25} {btts_m['btts_rate']*100:.2f}%")

    # ── OU Value Betting ──
    if metrics["n_bets"] == 0:
        print(f"\n  🎲 OVER/UNDER 2.5: No value bets found (min_ev={DEFAULT_MIN_EV})")
        return

    print(f"\n  🎲 OVER/UNDER 2.5 VALUE BETTING")
    print(f"  {'Bets placed':<25} {metrics['n_bets']}")
    print(f"  {'  Over 2.5 bets':<25} {metrics['n_over_2_5']}")
    print(f"  {'  Under 2.5 bets':<25} {metrics['n_under_2_5']}")
    print(f"  {'Won / Lost':<25} {metrics['n_won']} / {metrics['n_lost']}")
    print(f"  {'Win rate':<25} {metrics['win_rate']*100:.1f}%")
    print(f"  {'Total staked':<25} GBP {metrics['total_staked']:,.2f}")
    print(f"  {'Total profit':<25} GBP {metrics['total_profit']:+,.2f}")
    print(f"  {'Yield (ROI)':<25} {metrics['yield_pct']:+.2f}%")
    print(f"  {'Profit factor':<25} {metrics['profit_factor']:.2f}")
    print(f"  {'Final bankroll':<25} GBP {metrics['final_bankroll']:,.2f}")
    print(f"  {'Max drawdown':<25} {metrics['max_drawdown_pct']:.1f}%")
    print(f"  {'Avg odds':<25} {metrics['avg_odds']:.2f}")
    print(f"  {'Avg EV':<25} {metrics['avg_ev']:.2%}")

    # Split by Over vs Under
    print(f"\n  {'─'*40}")
    print(f"  {'Market':<20} {'Bets':>6} {'WR':>6} {'Yield':>8} {'Profit':>12}")
    print(f"  {'─'*40}")
    over_profit = sum(b["profit"] for b in bets if b["market"] == "Over 2.5")
    under_profit = sum(b["profit"] for b in bets if b["market"] == "Under 2.5")
    over_stake = sum(b["stake"] for b in bets if b["market"] == "Over 2.5")
    under_stake = sum(b["stake"] for b in bets if b["market"] == "Under 2.5")
    over_wr = len([b for b in bets if b["market"] == "Over 2.5" and b["won"]]) / max(len([b for b in bets if b["market"] == "Over 2.5"]), 1)
    under_wr = len([b for b in bets if b["market"] == "Under 2.5" and b["won"]]) / max(len([b for b in bets if b["market"] == "Under 2.5"]), 1)
    over_y = (over_profit / over_stake * 100) if over_stake > 0 else 0
    under_y = (under_profit / under_stake * 100) if under_stake > 0 else 0
    print(f"  {'Over 2.5':<20} {metrics['n_over_2_5']:>6} {over_wr*100:>5.0f}% {over_y:>+7.1f}% {over_profit:>+10.0f}")
    print(f"  {'Under 2.5':<20} {metrics['n_under_2_5']:>6} {under_wr*100:>5.0f}% {under_y:>+7.1f}% {under_profit:>+10.0f}")

    if params:
        print()
        print(f"  OPTIMISED SETTINGS:")
        bp = params.get("best_profit", {})
        if bp:
            print(f"    Best profit: min_ev={bp.get('min_ev','?')} kelly={bp.get('kelly_frac','?')} "
                  f"→ {bp.get('n_bets',0)} bets, {bp.get('yield_pct',0):+.2f}% yield, "
                  f"GBP {bp.get('profit',0):+,.0f} profit")

        by = params.get("best_yield", {})
        if by:
            print(f"    Best yield:   min_ev={by.get('min_ev','?')} kelly={by.get('kelly_frac','?')} "
                  f"→ {by.get('n_bets',0)} bets, {by.get('yield_pct',0):+.2f}% yield")

        safest = params.get("safest", {})
        if safest:
            print(f"    Safest:       min_ev={safest.get('min_ev','?')} kelly={safest.get('kelly_frac','?')} "
                  f"→ {safest.get('n_bets',0)} bets, max DD {safest.get('max_dd',0):.1f}%")

    # Top 10 bets
    if bets:
        print()
        print(f"  TOP VALUE BETS:")
        sorted_bets = sorted(bets, key=lambda b: b["ev"], reverse=True)[:5]
        for b in sorted_bets:
            result_str = "WON" if b["won"] else "LOST"
            print(f"    {b['date']} | {b['home']:20s} vs {b['away']:20s} | "
                  f"{b['market']:>10s} @ {b['odds']:.2f} | "
                  f"ev={b['ev']:.1%} | stake=GBP {b['stake']:.1f} | "
                  f"{result_str} ({b['actual_goals']}) | PnL=GBP {b['profit']:+.1f}")


def generate_cross_league_report(results: list[dict[str, Any]]):
    """Print comparison across all leagues."""
    print()
    print("=" * 80)
    print("  CROSS-LEAGUE OVER/UNDER 2.5 & BTTS COMPARISON")
    print("=" * 80)

    # OU table
    print()
    print(f"  {'League':<8} {'Type':<6} {'Bets':>6} {'WR':>5} {'Staked':>10} {'Profit':>10} {'Yield':>8} {'DD':>6} {'PF':>5}")
    print(f"  {'─'*8} {'─'*6} {'─'*6} {'─'*5} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*5}")
    for r in results:
        m = r["ou_metrics"]
        if m["n_bets"] > 0:
            wr = f"{m['win_rate']*100:.0f}%"
            profit_str = f"GBP {m['total_profit']:+,.0f}"
            yield_str = f"{m['yield_pct']:+.1f}%"
            dd_str = f"{m['max_drawdown_pct']:.0f}%"
            pf_str = f"{m['profit_factor']:.1f}" if m['profit_factor'] != float('inf') else "inf"
            print(f"  {r['league']:<8} {'OU':<6} {m['n_bets']:>6} {wr:>5s} {m['total_staked']:>10.0f} {profit_str:>10s} {yield_str:>8s} {dd_str:>6s} {pf_str:>5s}")
        else:
            print(f"  {r['league']:<8} {'OU':<6} {'—':>6} {'—':>5} {'—':>10} {'—':>10} {'—':>8} {'—':>6} {'—':>5}")

    # BTTS table
    print()
    print(f"  {'─'*50}")
    print(f"  {'League':<8} {'BTTS Brier':>12} {'BTTS LL':>10} {'BTTS Acc':>10} {'nMatches':>10}")
    print(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        b = r["btts_metrics"]
        print(f"  {r['league']:<8} {b['brier']:>12.4f} {b['log_loss']:>10.4f} {b['accuracy']*100:>9.1f}% {b['n_matches']:>10}")

    # Profitable summary
    ou_profitable = [r for r in results if r["ou_metrics"].get("total_profit", 0) > 0]
    print(f"\n  PROFITABLE OU leagues: {len(ou_profitable)}/{len(results)}")
    for r in ou_profitable:
        m = r["ou_metrics"]
        print(f"    {r['league']} - GBP {m['total_profit']:+,.0f} ({m['yield_pct']:+.2f}% yield)")

    best_btts = min(results, key=lambda r: r["btts_metrics"]["brier"])
    print(f"\n  Best BTTS model: {best_btts['league']} (Brier={best_btts['btts_metrics']['brier']})")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "ou_btts_backtest_report.json"
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_bankroll": INITIAL_BANKROLL,
        "default_min_ev": DEFAULT_MIN_EV,
        "default_kelly_frac": DEFAULT_KELLY_FRAC,
        "results": [
            {
                "league": r["league"],
                "over_under": r["ou_metrics"],
                "btts": r["btts_metrics"],
                "recommended_settings": r.get("params", {}).get("best_profit", {}) if r.get("params") else {},
            }
            for r in results
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("OU/BTTS backtest report saved to %s", report_path)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Over/Under 2.5 & BTTS backtest")
    parser.add_argument("--leagues", nargs="+", help="Leagues to test (default: all with models)")
    parser.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV)
    parser.add_argument("--kelly", type=float, default=DEFAULT_KELLY_FRAC)
    parser.add_argument("--optimise", action="store_true", help="Scan parameters")
    parser.add_argument("--derive", action="store_true",
                        help="Derive OU odds from 1X2 odds (for leagues without direct OU data)")
    args = parser.parse_args()

    # Determine leagues
    if args.leagues:
        league_codes = args.leagues
    else:
        # Auto-detect leagues with both trained models AND CSV data with OU odds
        league_codes = []
        seen = set()
        with open(CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                div = r.get("Div", "").strip()
                if div and div not in seen and r.get("bbav>2.5", "").strip():
                    model_path = MODELS_DIR / div / "dixon_coles.joblib"
                    if model_path.exists():
                        league_codes.append(div)
                        seen.add(div)
                    if len(league_codes) >= 10:
                        break
        # Also check DB for valid leagues
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT league FROM matches WHERE home_goals IS NOT NULL")
        db_leagues = {r[0] for r in cur.fetchall()}
        conn.close()
        league_codes = [l for l in league_codes if l in db_leagues]
        # Fallback: if CSV auto-detect found nothing, use leagues with both models + OU odds
        if not league_codes:
            # Scan CSV for leagues with OU odds
            logger.info("Auto-detecting leagues from CSV data...")
            seen_leagues = set()
            with open(CSV_PATH, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    league_val = r.get("league", "").strip()
                    if league_val and league_val not in seen_leagues:
                        has_ou = bool((r.get("bbav>2.5") or "").strip())
                        model_path = MODELS_DIR / league_val / "dixon_coles.joblib"
                        if has_ou and model_path.exists() and league_val in db_leagues:
                            league_codes.append(league_val)
                            seen_leagues.add(league_val)

    print()
    print("=" * 60)
    print("  OVER/UNDER 2.5 & BTTS BACKTEST")
    print(f"  Bankroll: GBP {INITIAL_BANKROLL:,.0f} | Min EV: {args.min_ev:.0%} | Kelly: {args.kelly:.0%}")
    print("=" * 60)

    all_results: list[dict[str, Any]] = []

    for league in league_codes:
        league_name = LEAGUE_NAMES.get(league, league)
        print()
        print("-" * 60)
        print(f"  {league} - {league_name}")

        models = load_league_models(league)
        if models is None:
            logger.warning("  No trained models for %s — skipping", league)
            continue

        # Load OU odds: try CSV first, fallback to derived from 1X2
        csv_df = load_csv_odds(league)
        if len(csv_df) < 50 and args.derive:
            logger.info("  CSV has no OU odds for %s — deriving from 1X2 odds...", league)
            csv_df = load_ou_from_1x2(league)
        if len(csv_df) < 50:
            logger.warning("  Only %d OU odds rows — need 50+, skipping", len(csv_df))
            continue

        # Load full DB data for model pre-compute (needs full history)
        db_df = load_db_data(league)
        if len(db_df) < 100:
            logger.warning("  Only %d DB matches — need 100+, skipping", len(db_df))
            continue

        logger.info("  CSV matches with OU odds: %d", len(csv_df))
        logger.info("  DB matches (full history): %d", len(db_df))

        # For pre-compute, we need aligned data. Best: merge CSV-DB on (date, teams)
        # to get goals + odds, then precompute on that same set.
        csv_df["_key"] = csv_df["home_team"] + "|" + csv_df["away_team"] + "|" + csv_df["date"].astype(str)
        db_df["_key"] = db_df["home_team"] + "|" + db_df["away_team"] + "|" + db_df["date"].astype(str)

        merged = csv_df.merge(
            db_df[["_key", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]],
            on="_key", suffixes=("", "_db"),
            how="inner",
        )
        if len(merged) < 50:
            logger.warning("  Only %d merged rows — skipping (data alignment issue)", len(merged))
            continue

        # Drop duplicate cols, sort by date
        for col in ["date_db", "home_team_db", "away_team_db", "home_goals_db", "away_goals_db", "result_db"]:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True)
        merged = merged.sort_values("date").reset_index(drop=True)
        logger.info("  Merged DB+CSV matches: %d", len(merged))

        # ── Split train/test ──
        split_idx = int(len(merged) * (1 - BACKTEST_FRAC))
        train_df_for_elo = merged.iloc[:split_idx]
        backtest_df = merged.iloc[split_idx:].copy()
        logger.info("  Backtest set: %d matches (last %d%%)", len(backtest_df), int(BACKTEST_FRAC * 100))

        # Train Elo on full train set for mature ratings
        from src.elo import EloSystem
        elo = EloSystem(k=32, home_advantage=100, initial_rating=1500)
        elo_core = models["elo"]
        # Use existing Elo but mature it on all data
        elo._ratings = elo_core._ratings.copy()

        # Use full merged data for conditional rates + historical features
        all_data_for_model = merged.copy()
        all_data_for_model["result"] = all_data_for_model.apply(
            lambda r: "H" if r["home_goals"] > r["away_goals"]
            else "A" if r["away_goals"] > r["home_goals"]
            else "D", axis=1
        )

        # ── Run backtest ──
        bt = run_ou_backtest(
            backtest_df, models, all_data_for_model,
            min_ev=args.min_ev, kelly_frac=args.kelly,
        )

        params_scan = None
        if args.optimise:
            logger.info("  Scanning parameters for optimal settings...")
            params_scan = scan_parameters(backtest_df, models, all_data_for_model)

        print_report(league, bt, params_scan)
        all_results.append({
            "league": league,
            "ou_metrics": bt["metrics"],
            "btts_metrics": bt["btts_metrics"],
            "params": params_scan,
            "n_bets": len(bt["bets"]),
            "bets": bt["bets"],
        })

    if len(all_results) >= 1:
        generate_cross_league_report(all_results)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
