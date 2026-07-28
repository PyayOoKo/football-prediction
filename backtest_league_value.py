"""
Per-League Value Betting Backtester — tests if our models can beat the bookies.

For each league:
1. Load trained model(s)
2. For each match in test period, compare model probs vs bookmaker odds
3. Place Kelly-sized bets when model sees value (model_prob > implied_prob)
4. Track bankroll, ROI, yield, max drawdown
5. Find optimal thresholds per league (min EV, Kelly fraction, etc.)
6. Generate profitability report

Usage
-----
    python backtest_league_value.py                               # DC+Elo blend (default) on all leagues
    python backtest_league_value.py --model blend                 # 5-model blend (DC+Elo+XGBoost+LGB)
    python backtest_league_value.py --leagues E0 SE1              # Specific leagues only
    python backtest_league_value.py --ou-only                     # Over/Under 2.5 only (skip 1X2)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
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
logger = logging.getLogger("backtest_league_value")

from football_data.config import LEAGUE_NAMES

DB_PATH = Path("data/football_data.db")
MODELS_DIR = Path("models/per_league")
REPORTS_DIR = Path("reports")

INITIAL_BANKROLL = 10_000.0  # GBP
DEFAULT_MIN_EV = 0.05        # 5% minimum expected value
DEFAULT_KELLY_FRAC = 0.25    # 25% fractional Kelly
BACKTEST_FRAC = 0.15         # Use last 15% of data for backtesting


# ═══════════════════════════════════════════════════════════
#  Data & Model Loading
# ═══════════════════════════════════════════════════════════

def load_data_with_odds(league: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               home_odds, draw_odds, away_odds, season,
               over25_odds, under25_odds
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
          AND home_odds IS NOT NULL AND draw_odds IS NOT NULL AND away_odds IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    # Fill NaN O/U odds with 0 so we can detect missing data
    df["over25_odds"] = df["over25_odds"].fillna(0.0)
    df["under25_odds"] = df["under25_odds"].fillna(0.0)
    return df


def load_league_models(league: str) -> dict[str, Any] | None:
    import joblib
    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    if not dc_path.exists() or not elo_path.exists():
        return None
    return {
        "dc": joblib.load(dc_path),
        "elo": joblib.load(elo_path),
    }


# ═══════════════════════════════════════════════════════════
#  Value Betting Logic
# ═══════════════════════════════════════════════════════════

def implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1.0 / odds if odds > 1 else 0.0


def kelly_stake(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Calculate fractional Kelly stake as fraction of bankroll."""
    if odds <= 1 or prob <= 0:
        return 0.0
    full_kelly = (prob * odds - 1.0) / (odds - 1.0)
    return max(0.0, full_kelly * fraction)


def load_calibrator(league: str, method: str = "hybrid"):
    """Load calibration model for a league. Returns None if unavailable."""
    import joblib
    path = MODELS_DIR / league / f"blend_calibrator_{method}.joblib"
    if path.exists():
        return joblib.load(path)
    path = MODELS_DIR / league / "blend_calibrator.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def run_backtest(
    df: pd.DataFrame,
    models: dict[str, Any],
    min_ev: float = DEFAULT_MIN_EV,
    kelly_frac: float = DEFAULT_KELLY_FRAC,
    initial_bankroll: float = INITIAL_BANKROLL,
    calibrator: Any = None,
    max_odds: float = 30.0,
    ou_only: bool = False,
    bet_on: str = "HDA",
    level_stake: float = 0.0,
    min_odds_1x2: float = 1.01,
    max_odds_1x2: float = 100.0,
) -> dict[str, Any]:
    """Run a value betting backtest for one league.

    Evaluates both 1X2 and Over/Under 2.5 markets when odds are available.
    Uses DC model predicted goals to compute Over/Under probabilities.

    Returns full backtest result dict with bets, metrics, bankroll history.
    """
    dc = models["dc"]
    elo = models["elo"]
    bankroll = initial_bankroll
    bets: list[dict[str, Any]] = []
    bankroll_history: list[dict[str, Any]] = []
    peak_bankroll = initial_bankroll

    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        result = row["result"]

        # Get odds
        odds_h = float(row["home_odds"])
        odds_d = float(row["draw_odds"])
        odds_a = float(row["away_odds"])
        odds_over25 = float(row["over25_odds"])
        odds_under25 = float(row["under25_odds"])

        # Get model predictions
        try:
            dc_pred = dc.predict(home, away)
            dc_probs = np.array([dc_pred.away_win_prob, dc_pred.draw_prob, dc_pred.home_win_prob])

            R_home = elo.get_rating(home)
            R_away = elo.get_rating(away)
            E_home = elo.expected_score(R_home, R_away)
            elo_away, elo_draw, elo_home = elo._expected_to_probs(E_home)
            elo_probs = np.array([elo_away, elo_draw, elo_home])

            # Blend: average
            blend_probs = (dc_probs + elo_probs) / 2.0

            # Apply calibration if available
            if calibrator is not None:
                blend_probs = calibrator.transform(blend_probs.reshape(1, -1))[0]

            # Update Elo ratings for future matches (but NOT from this prediction)
            elo.update_ratings(home, away, result, home_goals=row["home_goals"], away_goals=row["away_goals"])

        except Exception:
            bankroll_history.append({"date": row["date"], "bankroll": round(bankroll, 2)})
            continue

        # Actual outcomes
        actual_outcome_idx = {"A": 0, "D": 1, "H": 2}.get(result, -1)
        actual_total_goals = float(row["home_goals"] + row["away_goals"])
        actual_over25 = actual_total_goals > 2.5
        actual_under25 = actual_total_goals <= 2.5

        # ── Market 1: Over/Under 2.5 (from DC model goal distribution) ──
        ou_probs = [
            ("O2.5", dc_pred.over_2_5_prob, odds_over25, "Over2.5"),
            ("U2.5", dc_pred.under_2_5_prob, odds_under25, "Under2.5"),
        ]
        for outcome_label, model_prob, odds, market in ou_probs:
            if odds <= 1.0 or odds > max_odds or model_prob <= 0.0 or model_prob >= 1.0:
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
            if outcome_label == "O2.5":
                won = actual_over25
            else:
                won = actual_under25
            profit = stake_amount * (odds - 1.0) if won else -stake_amount
            bets.append({
                "date": row["date"],
                "home": home,
                "away": away,
                "outcome": outcome_label,
                "market": market,
                "odds": round(odds, 2),
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied, 4),
                "ev": round(ev, 4),
                "stake_pct": round(stake_pct, 4),
                "stake": round(stake_amount, 2),
                "won": won,
                "profit": round(profit, 2),
            })
            bankroll += profit

        # ── Market 2: 1X2 outcomes (skip if ou_only) ──
        if not ou_only:
            outcome_options = [
                ("H", 2, odds_h, blend_probs[2]),
                ("D", 1, odds_d, blend_probs[1]),
                ("A", 0, odds_a, blend_probs[0]),
            ]
            outcomes = []
            outcome_filter = bet_on.upper().strip()
            for label, idx, odds, prob in outcome_options:
                if label in outcome_filter:
                    if odds < min_odds_1x2 or odds > max_odds_1x2:
                        continue
                    outcomes.append((label, idx, odds, prob, "1X2"))
            for outcome_label, outcome_idx, odds, model_prob, market in outcomes:
                if odds <= 1 or model_prob <= 0:
                    continue
                implied = implied_prob(odds)
                ev = model_prob / implied - 1.0
                if ev < min_ev:
                    continue
                if level_stake > 0:
                    stake_amount = min(level_stake, bankroll)
                else:
                    stake_pct = kelly_stake(model_prob, odds, kelly_frac)
                    if stake_pct <= 0:
                        continue
                    stake_amount = bankroll * stake_pct
                if stake_amount <= 0 or bankroll <= 1.0:
                    continue
                won = outcome_idx == actual_outcome_idx
                profit = stake_amount * (odds - 1.0) if won else -stake_amount
                stake_pct = stake_amount / max(bankroll, 0.01)
                bets.append({
                    "date": row["date"],
                    "home": home,
                    "away": away,
                    "outcome": outcome_label,
                    "market": market,
                    "odds": round(odds, 2),
                    "model_prob": round(model_prob, 4),
                    "implied_prob": round(implied, 4),
                    "ev": round(ev, 4),
                    "stake_pct": round(stake_pct, 4),
                    "stake": round(stake_amount, 2),
                    "won": won,
                    "profit": round(profit, 2),
                })
                bankroll += profit

        bankroll_history.append({
            "date": row["date"],
            "bankroll": round(bankroll, 2),
            "bets_placed": len([b for b in bets if b["date"] == row["date"]]),
        })

    # ── Compute metrics ──
    if not bets:
        return {"bets": [], "metrics": {"n_bets": 0}, "bankroll_history": bankroll_history}

    total_staked = sum(b["stake"] for b in bets)
    total_profit = sum(b["profit"] for b in bets)
    won_bets = [b for b in bets if b["won"]]
    lost_bets = [b for b in bets if not b["won"]]
    win_rate = len(won_bets) / len(bets) if bets else 0
    roi = total_profit / total_staked if total_staked > 0 else 0
    yield_pct = roi * 100  # standard yield = profit/staked * 100

    # Max drawdown
    peak = initial_bankroll
    max_drawdown = 0.0
    for bh in bankroll_history:
        b = bh["bankroll"]
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    # Profit factor
    gross_wins = sum(b["profit"] for b in won_bets) if won_bets else 0
    gross_losses = abs(sum(b["profit"] for b in lost_bets)) if lost_bets else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Average odds and EV
    avg_odds = float(np.mean([b["odds"] for b in bets])) if bets else 0
    avg_ev = float(np.mean([b["ev"] for b in bets])) if bets else 0

    metrics = {
        "n_bets": len(bets),
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
    }

    return {"bets": bets, "metrics": metrics, "bankroll_history": bankroll_history}


# ═══════════════════════════════════════════════════════════
#  Parameter Optimisation
# ═══════════════════════════════════════════════════════════

def scan_parameters(df: pd.DataFrame, models: dict[str, Any], calibrator: Any = None) -> dict[str, Any]:
    """Scan different min_ev and kelly_frac combos to find optimal settings."""
    results = []
    for min_ev in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        for kelly_frac in [0.10, 0.25, 0.50, 1.0]:
            bt = run_backtest(df, models, min_ev=min_ev, kelly_frac=kelly_frac, calibrator=calibrator)
            m = bt["metrics"]
            results.append({
                "min_ev": min_ev,
                "kelly_frac": kelly_frac,
                "n_bets": m["n_bets"],
                "yield_pct": m.get("yield_pct", 0),
                "roi": m.get("roi", 0),
                "profit": m.get("total_profit", 0),
                "max_dd": m.get("max_drawdown_pct", 0),
                "win_rate": m.get("win_rate", 0),
                "final_bankroll": m.get("final_bankroll", INITIAL_BANKROLL),
            })

    # Find best based on final bankroll (total profit)
    best = max(results, key=lambda r: r["final_bankroll"]) if results else {}

    # Find settings with best yield (but minimum 20 bets for significance)
    viable = [r for r in results if r["n_bets"] >= 20]
    best_yield = max(viable, key=lambda r: r["yield_pct"]) if viable else best

    # Find safest (highest Sharpe-like: yield / max_dd)
    def safety_score(r):
        return r["yield_pct"] / max(r["max_dd"], 0.1) if r["n_bets"] >= 10 else 0
    safest = max(results, key=safety_score) if results else {}

    return {
        "all_results": results,
        "best_profit": best,
        "best_yield": best_yield,
        "safest": safest,
    }


# ═══════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════

def print_backtest_report(league: str, bt: dict[str, Any], params: dict[str, Any] | None = None):
    """Print a readable backtest report."""
    metrics = bt["metrics"]
    league_name = LEAGUE_NAMES.get(league, league)
    bets = bt["bets"]

    print()
    print("=" * 65)
    print(f"  {league} - {league_name}")
    print("=" * 65)

    if metrics["n_bets"] == 0:
        print("  NO VALUE BETS FOUND with current thresholds")
        return

    print(f"  Bets placed:       {metrics['n_bets']}")
    print(f"  Won / Lost:        {metrics['n_won']} / {metrics['n_lost']}")
    print(f"  Win rate:          {metrics['win_rate']*100:.1f}%")
    print(f"  Total staked:      GBP {metrics['total_staked']:,.2f}")
    print(f"  Total profit:      GBP {metrics['total_profit']:+,.2f}")
    print(f"  Yield (ROI):       {metrics['yield_pct']:+.2f}%")
    print(f"  Profit factor:     {metrics['profit_factor']:.2f}")
    print(f"  Final bankroll:    GBP {metrics['final_bankroll']:,.2f}")
    print(f"  Max drawdown:      {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Avg odds:          {metrics['avg_odds']:.2f}")
    print(f"  Avg EV:            {metrics['avg_ev']:.4f}")

    # Market breakdown
    if bets:
        markets: dict[str, list[dict]] = {}
        for b in bets:
            mkt = b.get("market", "1X2")
            if mkt not in markets:
                markets[mkt] = []
            markets[mkt].append(b)
        if len(markets) > 1:
            print(f"\n  MARKET BREAKDOWN:")
            for mkt_name in sorted(markets.keys()):
                mb = markets[mkt_name]
                m_won = sum(1 for b in mb if b["won"])
                m_staked = sum(b["stake"] for b in mb)
                m_profit = sum(b["profit"] for b in mb)
                m_yield = (m_profit / m_staked * 100) if m_staked > 0 else 0.0
                print(f"    {mkt_name:10s}: {len(mb):>4} bets, {m_won:>3} won, "
                      f"staked GBP {m_staked:>8.0f}, profit GBP {m_profit:>+8.0f}, yield {m_yield:+5.1f}%")

    if params:
        print()
        print(f"  BEST SETTINGS (profit):")
        bp = params.get("best_profit", {})
        if bp:
            print(f"    min_ev={bp.get('min_ev', '?')}, kelly={bp.get('kelly_frac', '?')}")
            print(f"    -> {bp.get('n_bets', 0)} bets, {bp.get('yield_pct', 0):+.2f}% yield, GBP {bp.get('profit', 0):+,.2f} profit")

        by = params.get("best_yield", {})
        if by:
            print(f"  BEST YIELD settings:")
            print(f"    min_ev={by.get('min_ev', '?')}, kelly={by.get('kelly_frac', '?')}")
            print(f"    -> {by.get('n_bets', 0)} bets, {by.get('yield_pct', 0):+.2f}% yield, GBP {by.get('profit', 0):+,.2f} profit")

        safest = params.get("safest", {})
        if safest:
            print(f"  SAFEST settings:")
            print(f"    min_ev={safest.get('min_ev', '?')}, kelly={safest.get('kelly_frac', '?')}")
            print(f"    -> {safest.get('n_bets', 0)} bets, {safest.get('yield_pct', 0):+.2f}% yield, max DD {safest.get('max_dd', 0):.1f}%")

    # Top 10 bets
    if bets:
        print()
        print(f"  TOP 5 VALUE BETS:")
        sorted_bets = sorted(bets, key=lambda b: b["ev"], reverse=True)[:5]
        for b in sorted_bets:
            result_str = "WON" if b["won"] else "LOST"
            print(f"    {b['date']} | {b['home']:20s} vs {b['away']:20s} | {b['outcome']} @ {b['odds']:.2f} | "
                  f"ev={b['ev']:.2%} | stake=GBP {b['stake']:.1f} | {result_str} (GBP {b['profit']:+.1f})")


def generate_cross_league_report(results: list[dict[str, Any]]):
    """Print a comparison table across all leagues."""
    print()
    print("=" * 75)
    print("  CROSS-LEAGUE VALUE BETTING COMPARISON")
    print("=" * 75)
    print(f"  {'League':6s} {'Bets':>6s} {'Won':>4s} {'WR':>5s} {'Staked':>10s} {'Profit':>10s} {'Yield':>7s} {'DD':>5s} {'PF':>5s}  Markets")
    print(f"  {'-'*6} {'-'*6} {'-'*4} {'-'*5} {'-'*10} {'-'*10} {'-'*7} {'-'*5} {'-'*5}  {'-'*16}")

    for r in results:
        m = r["metrics"]
        wr = f"{m['win_rate']*100:.0f}%"
        profit_str = f"GBP {m['total_profit']:+,.0f}"
        yield_str = f"{m['yield_pct']:+.1f}%"
        dd_str = f"{m['max_drawdown_pct']:.0f}%"
        pf_str = f"{m['profit_factor']:.1f}" if m['profit_factor'] != float('inf') else "inf"
        # Show which markets were included
        bets = r.get("bets", [])
        markets_used = sorted(set(b.get("market", "1X2") for b in bets)) if bets else ["?"]
        markets_str = ",".join(markets_used)
        print(f"  {r['league']:6s} {m['n_bets']:>6} {m['n_won']:>4} {wr:>5s} {m['total_staked']:>10.0f} {profit_str:>10s} {yield_str:>7s} {dd_str:>5s} {pf_str:>5s}  {markets_str:16s}")

    print()
    profitable = [r for r in results if r["metrics"].get("total_profit", 0) > 0]
    print(f"  PROFITABLE leagues: {len(profitable)}/{len(results)}")
    for r in profitable:
        m = r["metrics"]
        print(f"    {r['league']} - GBP {m['total_profit']:+,.0f} ({m['yield_pct']:+.2f}% yield)")

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "value_betting_comparison.json"
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "initial_bankroll": INITIAL_BANKROLL,
        "default_min_ev": DEFAULT_MIN_EV,
        "default_kelly_frac": DEFAULT_KELLY_FRAC,
        "results": [
            {
                "league": r["league"],
                "metrics": r["metrics"],
                "recommended_settings": r.get("params", {}).get("best_profit", {}) if r.get("params") else {},
            }
            for r in results
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Value betting report saved to %s", report_path)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def build_blend_probs(backtest_df, blend):
    """Pre-compute blend predictions for all matches in backtest_df."""
    from src.models.three_model_blend import ThreeModelBlend
    preds = blend.predict_matches(backtest_df)
    probs_map = {}
    for i, (_, row) in enumerate(backtest_df.iterrows()):
        key = (row["home_team"], row["away_team"], str(row["date"]))
        probs_map[key] = np.array([
            preds.iloc[i]["away_win_prob"],
            preds.iloc[i]["draw_prob"],
            preds.iloc[i]["home_win_prob"],
        ])
    return probs_map


def run_backtest_blend(
    df: pd.DataFrame,
    models: dict[str, Any],
    blend_probs_map: dict,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_frac: float = DEFAULT_KELLY_FRAC,
    initial_bankroll: float = INITIAL_BANKROLL,
    calibrator: Any = None,
    max_odds: float = 30.0,
    ou_only: bool = False,
    bet_on: str = "HDA",
    level_stake: float = 0.0,
    min_odds_1x2: float = 1.01,
    max_odds_1x2: float = 100.0,
) -> dict[str, Any]:
    """Run backtest using pre-computed blend probabilities.

    Evaluates both 1X2 and Over/Under 2.5 markets when odds are available.
    Uses pre-computed blend probs for 1X2, and DC model goal distribution for O/U.
    """
    dc = models["dc"]
    elo = models["elo"]
    bankroll = initial_bankroll
    bets: list[dict[str, Any]] = []
    bankroll_history: list[dict[str, Any]] = []
    peak_bankroll = initial_bankroll

    for idx, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        result = row["result"]

        odds_h = float(row["home_odds"])
        odds_d = float(row["draw_odds"])
        odds_a = float(row["away_odds"])
        odds_over25 = float(row["over25_odds"])
        odds_under25 = float(row["under25_odds"])

        # Actual outcomes
        actual_outcome_idx = {"A": 0, "D": 1, "H": 2}.get(result, -1)
        actual_total_goals = float(row["home_goals"] + row["away_goals"])
        actual_over25 = actual_total_goals > 2.5
        actual_under25 = actual_total_goals <= 2.5

        key = (home, away, str(row["date"]))
        blend_probs = blend_probs_map.get(key)
        if blend_probs is None:
            bankroll_history.append({"date": row["date"], "bankroll": round(bankroll, 2)})
            continue

        # Apply calibration if available
        if calibrator is not None and blend_probs is not None:
            blend_probs = calibrator.transform(np.array(blend_probs).reshape(1, -1))[0]

        elo.update_ratings(home, away, result, home_goals=row["home_goals"], away_goals=row["away_goals"])

        # Get DC model O/U probabilities
        dc_pred = dc.predict(home, away)

        # ── Market 1: O/U 2.5 ──
        ou_probs = [
            ("O2.5", dc_pred.over_2_5_prob, odds_over25, "Over2.5"),
            ("U2.5", dc_pred.under_2_5_prob, odds_under25, "Under2.5"),
        ]
        for outcome_label, model_prob, odds, market in ou_probs:
            if odds <= 1.0 or odds > max_odds or model_prob <= 0.0 or model_prob >= 1.0:
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
            won = actual_over25 if outcome_label == "O2.5" else actual_under25
            profit = stake_amount * (odds - 1.0) if won else -stake_amount
            bets.append({
                "date": row["date"],
                "home": home,
                "away": away,
                "outcome": outcome_label,
                "market": market,
                "odds": round(odds, 2),
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied, 4),
                "ev": round(ev, 4),
                "stake_pct": round(stake_pct, 4),
                "stake": round(stake_amount, 2),
                "won": won,
                "profit": round(profit, 2),
            })
            bankroll += profit

        # ── Market 2: 1X2 outcomes (skip if ou_only) ──
        if not ou_only:
            outcome_options = [
                ("H", 2, odds_h, blend_probs[2]),
                ("D", 1, odds_d, blend_probs[1]),
                ("A", 0, odds_a, blend_probs[0]),
            ]
            outcomes = []
            outcome_filter = bet_on.upper().strip()
            for label, idx, odds, prob in outcome_options:
                if label in outcome_filter:
                    if odds < min_odds_1x2 or odds > max_odds_1x2:
                        continue
                    outcomes.append((label, idx, odds, prob, "1X2"))
            for outcome_label, outcome_idx, odds, model_prob, market in outcomes:
                if odds <= 1 or model_prob <= 0:
                    continue
                implied = implied_prob(odds)
                ev = model_prob / implied - 1.0
                if ev < min_ev:
                    continue
                if level_stake > 0:
                    stake_amount = min(level_stake, bankroll)
                else:
                    stake_pct = kelly_stake(model_prob, odds, kelly_frac)
                    if stake_pct <= 0:
                        continue
                    stake_amount = bankroll * stake_pct
                if stake_amount <= 0 or bankroll <= 1.0:
                    continue
                won = outcome_idx == actual_outcome_idx
                profit = stake_amount * (odds - 1.0) if won else -stake_amount
                stake_pct = stake_amount / max(bankroll, 0.01)
                bets.append({
                    "date": row["date"],
                    "home": home,
                    "away": away,
                    "outcome": outcome_label,
                    "market": market,
                    "odds": round(odds, 2),
                    "model_prob": round(model_prob, 4),
                    "implied_prob": round(implied, 4),
                    "ev": round(ev, 4),
                    "stake_pct": round(stake_pct, 4),
                    "stake": round(stake_amount, 2),
                    "won": won,
                    "profit": round(profit, 2),
                })
                bankroll += profit

        bankroll_history.append({
            "date": row["date"],
            "bankroll": round(bankroll, 2),
            "bets_placed": len([b for b in bets if b["date"] == row["date"]]),
        })

    if not bets:
        return {"bets": [], "metrics": {"n_bets": 0}, "bankroll_history": bankroll_history}

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

    metrics = {
        "n_bets": len(bets),
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
    }

    return {"bets": bets, "metrics": metrics, "bankroll_history": bankroll_history}


def scan_parameters_blend(df, models, blend_probs_map, calibrator=None):
    results = []
    for min_ev in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        for kelly_frac in [0.10, 0.25, 0.50, 1.0]:
            bt = run_backtest_blend(df, models, blend_probs_map, min_ev=min_ev, kelly_frac=kelly_frac, calibrator=calibrator)
            m = bt["metrics"]
            results.append({
                "min_ev": min_ev,
                "kelly_frac": kelly_frac,
                "n_bets": m["n_bets"],
                "yield_pct": m.get("yield_pct", 0),
                "roi": m.get("roi", 0),
                "profit": m.get("total_profit", 0),
                "max_dd": m.get("max_drawdown_pct", 0),
                "win_rate": m.get("win_rate", 0),
                "final_bankroll": m.get("final_bankroll", INITIAL_BANKROLL),
            })
    best = max(results, key=lambda r: r["final_bankroll"]) if results else {}
    viable = [r for r in results if r["n_bets"] >= 20]
    best_yield = max(viable, key=lambda r: r["yield_pct"]) if viable else best
    def safety_score(r):
        return r["yield_pct"] / max(r["max_dd"], 0.1) if r["n_bets"] >= 10 else 0
    safest = max(results, key=safety_score) if results else {}
    return {"all_results": results, "best_profit": best, "best_yield": best_yield, "safest": safest}


def main():
    parser = argparse.ArgumentParser(description="Per-league value betting backtester")
    parser.add_argument("--leagues", nargs="+", help="Leagues to backtest (default: all)")
    parser.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV, help=f"Min EV threshold (default: {DEFAULT_MIN_EV})")
    parser.add_argument("--kelly", type=float, default=DEFAULT_KELLY_FRAC, help=f"Kelly fraction (default: {DEFAULT_KELLY_FRAC})")
    parser.add_argument("--optimise", action="store_true", help="Scan parameters to find optimal per-league settings")
    parser.add_argument("--model", choices=["simple", "blend"], default="simple",
                        help="Model to use: simple (DC+Elo) or blend (DC+Elo+XGBoost+LGB) (default: simple)")
    parser.add_argument("--calibrate", choices=["none", "hybrid", "platt", "isotonic"], default="none",
                        help="Probability calibration method (default: none)")
    parser.add_argument("--calibrate-blend", action="store_true",
                        help="Shorthand for --calibrate hybrid")
    parser.add_argument("--max-odds", type=float, default=30.0,
                        help="Max decimal odds for Over/Under bets (default: 30.0)")
    parser.add_argument("--ou-only", action="store_true",
                        help="Only bet on Over/Under 2.5, skip 1X2")
    parser.add_argument("--bet-on", type=str, default="HDA",
                        help="1X2 outcomes to bet on: H=Home, D=Draw, A=Away, HD, HA, DA, HDA (default: HDA)")
    parser.add_argument("--level-stake", type=float, default=0.0,
                        help="Flat stake per bet in GBP (0 = use Kelly staking, default: 0)")
    parser.add_argument("--min-odds-1x2", type=float, default=1.01,
                        help="Minimum decimal odds for 1X2 bets (default: 1.01)")
    parser.add_argument("--max-odds-1x2", type=float, default=100.0,
                        help="Maximum decimal odds for 1X2 bets (default: 100.0)")
    args = parser.parse_args()
    if args.calibrate_blend:
        args.calibrate = "hybrid"

    if args.leagues:
        league_codes = args.leagues
    else:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("""
            SELECT league FROM matches
            WHERE home_odds IS NOT NULL AND draw_odds IS NOT NULL AND away_odds IS NOT NULL
            GROUP BY league HAVING COUNT(*) >= 200
            ORDER BY COUNT(*) DESC
        """)
        league_codes = [r[0] for r in cur.fetchall()]
        conn.close()

    print()
    print("=" * 60)
    model_label = "DC+Elo+XGBoost+LGB" if args.model == "blend" else "DC+Elo"
    cal_label = f" | Calibrate: {args.calibrate}" if args.calibrate != "none" else ""
    print(f"  PER-LEAGUE VALUE BETTING BACKTEST  [{model_label}{cal_label}]")
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
            logger.warning("  No trained models found - run train_league_models.py --leagues %s first", league)
            continue

        df = load_data_with_odds(league)
        if len(df) < 100:
            logger.warning("  Only %d matches with odds - need 100+, skipping", len(df))
            continue

        logger.info("  Loaded %d matches with odds", len(df))

        # Use last 15% for backtesting
        split_idx = int(len(df) * (1 - BACKTEST_FRAC))
        backtest_df = df.iloc[split_idx:].copy()
        logger.info("  Backtesting on %d matches", len(backtest_df))

        # Train Elo on full dataset before backtesting (so ratings are mature)
        from src.elo import EloSystem
        elo = EloSystem(k=32, home_advantage=100, initial_rating=1500)
        train_df = df.iloc[:split_idx].copy()
        elo.process_matches(train_df)
        models["elo"]._ratings = elo._ratings  # Use mature ratings

        # Build 3-model blend if requested
        blend_probs_map = None
        if args.model == "blend":
            import joblib
            xgb_path = MODELS_DIR / league / "xgboost.joblib"
            lgb_path = MODELS_DIR / league / "lightgbm.joblib"
            xgb_model = joblib.load(xgb_path) if xgb_path.exists() else None
            lgb_model = joblib.load(lgb_path) if lgb_path.exists() else None
            has_ml = xgb_model is not None or lgb_model is not None
            if has_ml:
                from src.models.three_model_blend import ThreeModelBlend
                blend = ThreeModelBlend(
                    dc_model=models["dc"],
                    elo_model=models["elo"],
                    xgb_model=xgb_model,
                    lgb_model=lgb_model,
                    historical_df=df,
                )
                logger.info("  Computing blend predictions for %d matches...", len(backtest_df))
                blend_probs_map = build_blend_probs(backtest_df, blend)
                logger.info("  Got predictions for %d/%d matches", len(blend_probs_map), len(backtest_df))
                logger.info("  Blend weights: %s", blend.weights.get("1X2", {}))
            else:
                logger.info("  No XGBoost/LightGBM models found for %s, falling back to DC+Elo", league)

        # Load calibrator
        calibrator = None
        if args.calibrate != "none":
            calibrator = load_calibrator(league, args.calibrate)
            if calibrator is not None:
                logger.info("  Loaded calibrator: %s", args.calibrate)
            else:
                logger.info("  No calibrator found for %s, running uncalibrated", league)

        # Parse bet_on filter
        bet_on = args.bet_on.upper().strip()
        if not all(c in "HDA" for c in bet_on):
            logger.warning("  Invalid --bet-on value '%s', defaulting to HDA", args.bet_on)
            bet_on = "HDA"

        # Run backtest
        if blend_probs_map:
            bt = run_backtest_blend(backtest_df, models, blend_probs_map,
                                    min_ev=args.min_ev, kelly_frac=args.kelly,
                                    calibrator=calibrator, max_odds=args.max_odds,
                                    ou_only=args.ou_only,
                                    bet_on=bet_on, level_stake=args.level_stake,
                                    min_odds_1x2=args.min_odds_1x2,
                                    max_odds_1x2=args.max_odds_1x2)
        else:
            bt = run_backtest(backtest_df, models,
                              min_ev=args.min_ev, kelly_frac=args.kelly,
                              calibrator=calibrator, max_odds=args.max_odds,
                              ou_only=args.ou_only,
                              bet_on=bet_on, level_stake=args.level_stake,
                              min_odds_1x2=args.min_odds_1x2,
                              max_odds_1x2=args.max_odds_1x2)

        # Parameter optimisation
        params_scan = None
        if args.optimise:
            logger.info("  Scanning parameters for optimal settings...")
            if blend_probs_map:
                params_scan = scan_parameters_blend(backtest_df, models, blend_probs_map, calibrator=calibrator)
            else:
                params_scan = scan_parameters(backtest_df, models, calibrator=calibrator)

        print_backtest_report(league, bt, params_scan)
        all_results.append({
            "league": league,
            "metrics": bt["metrics"],
            "params": params_scan,
            "bets": bt["bets"],
            "bankroll_history": bt["bankroll_history"],
        })

    if len(all_results) >= 2:
        generate_cross_league_report(all_results)

    # Save individual results
    for r in all_results:
        league_dir = MODELS_DIR / r["league"]
        league_dir.mkdir(parents=True, exist_ok=True)
        result_path = league_dir / "value_backtest.json"
        with open(result_path, "w") as f:
            json.dump({
                "league": r["league"],
                "metrics": r["metrics"],
                "n_bets": len(r["bets"]),
                "bankroll_events": len(r["bankroll_history"]),
            }, f, indent=2)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
