"""
train_gap_model.py — Train and evaluate GAP (Goal-Adjusted Performance) ratings.

Based on Wheatcroft (2020) "A Profitable Model for Predicting the Over/Under
Market in Football".

Trains a GAP model on league match data using high-frequency stats (shots on
target, total shots, corners) to predict Over/Under 2.5 probabilities, then
backtests against bookmaker odds.

Usage:
    python scripts/train_gap_model.py F1                    # Ligue 1 (default SOT)
    python scripts/train_gap_model.py F1 --stat home_shots  # Use total shots
    python scripts/train_gap_model.py F1 --stat home_corners  # Use corners
    python scripts/train_gap_model.py F1 --k 48            # Higher learning rate
    python scripts/train_gap_model.py --leagues E0 F1 D1   # Multiple leagues
    python scripts/train_gap_model.py --leagues all        # All top 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Fix Windows console encoding for Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on the Python path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_gap")

PROJECT_ROOT = _project_root
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODELS_DIR = PROJECT_ROOT / "models" / "per_league"

INITIAL_BANKROLL = 10_000.0
DEFAULT_MIN_EV = 0.05
DEFAULT_KELLY_FRAC = 0.25
BACKTEST_FRAC = 0.15


# ── Data loading ────────────────────────────────────────

def load_league_data(league: str) -> pd.DataFrame:
    """Load match data with all stat columns from the DB."""
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               home_shots, away_shots,
               home_shots_target, away_shots_target,
               home_corners, away_corners,
               home_odds, draw_odds, away_odds
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
          AND home_shots IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d matches for %s", len(df), league)
    return df


# ── Backtest ────────────────────────────────────────────

def compute_ou_odds_from_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Over/Under 2.5 odds from 1X2 odds using conditional rates.

    P(Over) = P(H) * P(Over|H) + P(D) * P(Over|D) + P(A) * P(Over|A)
    """
    df = df.copy()

    # Compute conditional rates from the FULL dataframe
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["is_over"] = (df["total_goals"] > 2.5).astype(float)

    cond_rates: dict[str, float] = {}
    for outcome, label in [("H", "home_win"), ("D", "draw"), ("A", "away_win")]:
        subset = df[df["result"] == outcome]
        cond_rates[label] = float(subset["is_over"].mean()) if len(subset) > 0 else 0.5

    def _derive(row):
        h_odds = float(row.get("home_odds", 2.0))
        d_odds = float(row.get("draw_odds", 3.5))
        a_odds = float(row.get("away_odds", 2.5))
        if h_odds <= 1 or d_odds <= 1 or a_odds <= 1:
            return 2.0, 2.0

        p_h, p_d, p_a = 1.0 / h_odds, 1.0 / d_odds, 1.0 / a_odds
        total = p_h + p_d + p_a
        if total <= 0:
            return 2.0, 2.0

        # Remove overround
        p_h_n, p_d_n, p_a_n = p_h / total, p_d / total, p_a / total

        implied_over = (
            p_h_n * cond_rates["home_win"]
            + p_d_n * cond_rates["draw"]
            + p_a_n * cond_rates["away_win"]
        )
        implied_over = np.clip(implied_over, 0.05, 0.95)

        over_odds = round(1.0 / implied_over, 2)
        under_odds = round(1.0 / (1.0 - implied_over), 2)
        return over_odds, under_odds

    derived = df.apply(_derive, axis=1, result_type="expand")
    df["over_odds"] = derived[0]
    df["under_odds"] = derived[1]
    return df


def run_gap_backtest(
    df_hist: pd.DataFrame,
    df_test: pd.DataFrame,
    gap_model: Any,
    min_ev: float = DEFAULT_MIN_EV,
    kelly_frac: float = DEFAULT_KELLY_FRAC,
) -> dict[str, Any]:
    """Run Over/Under backtest using GAP model predictions and derived OU odds."""
    # Derive OU odds from 1X2 odds for the test set
    all_data = pd.concat([df_hist, df_test], ignore_index=True)
    all_data = compute_ou_odds_from_1x2(all_data)
    df_test_with_odds = all_data.iloc[-len(df_test):].copy().reset_index(drop=True)

    bankroll = INITIAL_BANKROLL
    bets: list[dict[str, Any]] = []
    peak = INITIAL_BANKROLL
    max_dd = 0.0

    # Get model predictions for all test matches
    probs = gap_model.predict_proba(df_test_with_odds)

    for i in range(len(df_test_with_odds)):
        row = df_test_with_odds.iloc[i]
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        actual_over = (hg + ag) > 2.5

        model_over = float(probs[i])
        model_under = 1.0 - model_over

        over_odds = float(row["over_odds"])
        under_odds = float(row["under_odds"])

        if over_odds <= 1 or under_odds <= 1:
            continue

        outcomes = [
            ("Over 2.5", over_odds, model_over, actual_over),
            ("Under 2.5", under_odds, model_under, not actual_over),
        ]

        for label, odds, model_prob, actual_won in outcomes:
            if odds <= 1 or model_prob <= 0:
                continue
            implied = 1.0 / odds
            ev = model_prob / implied - 1.0
            if ev < min_ev:
                continue
            full_kelly = (model_prob * odds - 1.0) / (odds - 1.0)
            if full_kelly <= 0:
                continue
            stake_pct = full_kelly * kelly_frac
            stake_amount = bankroll * stake_pct
            if stake_amount <= 1.0:
                continue

            won = actual_won
            profit = stake_amount * (odds - 1.0) if won else -stake_amount
            bankroll += profit

            bets.append({
                "date": str(row.get("date", ""))[:10],
                "home": row["home_team"],
                "away": row["away_team"],
                "market": label,
                "odds": round(odds, 2),
                "model_prob": round(model_prob, 4),
                "implied_prob": round(implied, 4),
                "ev": round(ev, 4),
                "stake": round(stake_amount, 2),
                "won": won,
                "profit": round(profit, 2),
                "actual_goals": f"{hg}-{ag}",
            })

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Compute metrics
    if not bets:
        return {"metrics": {"n_bets": 0, "market": "Over2.5", "model": "GAP"},
                "bets": []}

    total_staked = sum(b["stake"] for b in bets)
    total_profit = sum(b["profit"] for b in bets)
    won_bets = [b for b in bets if b["won"]]
    win_rate = len(won_bets) / len(bets)
    roi = total_profit / total_staked if total_staked > 0 else 0
    yield_pct = roi * 100

    over_bets = [b for b in bets if b["market"] == "Over 2.5"]
    under_bets = [b for b in bets if b["market"] == "Under 2.5"]
    over_profit = sum(b["profit"] for b in over_bets)
    under_profit = sum(b["profit"] for b in under_bets)
    over_stake = sum(b["stake"] for b in over_bets)
    under_stake = sum(b["stake"] for b in under_bets)
    over_wr = len([b for b in over_bets if b["won"]]) / max(len(over_bets), 1)
    under_wr = len([b for b in under_bets if b["won"]]) / max(len(under_bets), 1)
    over_yield = (over_profit / over_stake * 100) if over_stake > 0 else 0
    under_yield = (under_profit / under_stake * 100) if under_stake > 0 else 0

    metrics = {
        "model": "GAP",
        "n_bets": len(bets),
        "n_over_2_5": len(over_bets),
        "n_under_2_5": len(under_bets),
        "n_won": len(won_bets),
        "n_lost": len(bets) - len(won_bets),
        "win_rate": round(win_rate, 4),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "yield_pct": round(yield_pct, 2),
        "final_bankroll": round(bankroll, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "over_win_rate": round(over_wr, 4),
        "under_win_rate": round(under_wr, 4),
        "over_yield": round(over_yield, 2),
        "under_yield": round(under_yield, 2),
    }

    return {"metrics": metrics, "bets": bets}


# ── Reporting ───────────────────────────────────────────

def print_report(league: str, gap_model: Any, bt: dict[str, Any],
                 extra_title: str = ""):
    """Print GAP model evaluation and backtest results."""
    metrics = bt["metrics"]
    bets = bt["bets"]
    title = f"  {league} - GAP Ratings ({gap_model.stat_column})"
    if extra_title:
        title += f" {extra_title}"

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    # Model quality
    print()
    print("  📊 MODEL QUALITY")
    k_val = gap_model.k
    print(f"  {'Learning rate (K)':<25} {k_val}")
    print(f"  {'Stat column':<25} {gap_model.stat_column}")
    print(f"  {'Training matches':<25} {gap_model._n_matches}")

    # Over/Under backtest
    if metrics["n_bets"] == 0:
        print(f"\n  🎲 No value bets found (min_ev={DEFAULT_MIN_EV})")
        return

    print(f"\n  🎲 OVER/UNDER 2.5 VALUE BETTING (Kelly {DEFAULT_KELLY_FRAC:.0%})")
    print(f"  {'Bets placed':<25} {metrics['n_bets']}")
    print(f"  {'  Over 2.5 bets':<25} {metrics['n_over_2_5']}")
    print(f"  {'  Under 2.5 bets':<25} {metrics['n_under_2_5']}")
    print(f"  {'Won / Lost':<25} {metrics['n_won']} / {metrics['n_lost']}")
    print(f"  {'Win rate':<25} {metrics['win_rate']*100:.1f}%")
    print(f"  {'Total staked':<25} GBP {metrics['total_staked']:,.2f}")
    print(f"  {'Total profit':<25} GBP {metrics['total_profit']:+,.2f}")
    print(f"  {'Yield (ROI)':<25} {metrics['yield_pct']:+.2f}%")
    print(f"  {'Final bankroll':<25} GBP {metrics['final_bankroll']:,.2f}")
    print(f"  {'Max drawdown':<25} {metrics['max_drawdown_pct']:.1f}%")

    # Split
    print(f"\n  {'-'*40}")
    print(f"  {'Market':<20} {'Bets':>6} {'WR':>6} {'Yield':>8} {'Profit':>10}")
    print(f"  {'-'*40}")
    over_profit = sum(b["profit"] for b in bets if b["market"] == "Over 2.5")
    under_profit = sum(b["profit"] for b in bets if b["market"] == "Under 2.5")
    print(f"  {'Over 2.5':<20} {metrics['n_over_2_5']:>6} "
          f"{metrics['over_win_rate']*100:>5.0f}% "
          f"{metrics['over_yield']:>+7.1f}% {over_profit:>+10.0f}")
    print(f"  {'Under 2.5':<20} {metrics['n_under_2_5']:>6} "
          f"{metrics['under_win_rate']*100:>5.0f}% "
          f"{metrics['under_yield']:>+7.1f}% {under_profit:>+10.0f}")

    # Top bets
    if bets:
        print(f"\n  TOP VALUE BETS:")
        sorted_bets = sorted(bets, key=lambda b: b["ev"], reverse=True)[:5]
        for b in sorted_bets:
            result = "WON" if b["won"] else "LOST"
            print(f"    {b['date']} | {b['home']:20s} vs {b['away']:20s} | "
                  f"{b['market']:>10s} @ {b['odds']:.2f} | "
                  f"ev={b['ev']:.1%} | stake=GBP {b['stake']:.1f} | "
                  f"{result} ({b['actual_goals']}) | PnL=GBP {b['profit']:+.1f}")


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate GAP ratings for Over/Under 2.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("leagues", nargs="+",
                        help="League codes (e.g. F1 E0 D1) or 'all'")
    parser.add_argument("--stat", default="home_shots_target",
                        choices=["home_shots_target", "home_shots", "home_corners"],
                        help="Stat column to rate teams on (default: home_shots_target)")
    parser.add_argument("--k", type=float, default=32.0,
                        help="GAP learning rate (default: 32)")
    parser.add_argument("--decay", type=float, default=365.0,
                        help="Time decay halflife in days (default: 365)")
    parser.add_argument("--home-adv", type=float, default=50.0,
                        help="Home advantage bonus (default: 50)")
    parser.add_argument("--min-ev", type=float, default=DEFAULT_MIN_EV,
                        help="Minimum EV threshold (default: 0.05)")
    parser.add_argument("--kelly", type=float, default=DEFAULT_KELLY_FRAC,
                        help="Kelly fraction (default: 0.25)")
    args = parser.parse_args()

    if "all" in args.leagues:
        leagues = ["E0", "F1", "D1", "I1", "SP1"]
    else:
        leagues = [l.upper() for l in args.leagues]

    from src.gap_model import GAPModel

    all_results = []

    for league in leagues:
        print()
        print("-" * 60)
        print(f"  {league}")
        print("-" * 60)

        df = load_league_data(league)
        if len(df) < 200:
            logger.warning("Not enough data for %s (%d rows)", league, len(df))
            continue

        # Split chronologically
        split_idx = int(len(df) * (1 - BACKTEST_FRAC))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Check stat column availability
        home_stat = args.stat
        away_stat = home_stat.replace("home_", "away_")
        if home_stat not in train_df.columns:
            logger.warning("Stat column '%s' not in data for %s, skipping", home_stat, league)
            continue

        # Check for nulls in stat columns
        train_stat_null = train_df[home_stat].isna().sum()
        if train_stat_null > len(train_df) * 0.5:
            logger.warning("Too many nulls in stat column for %s (%d/%d), skipping",
                          league, train_stat_null, len(train_df))
            continue

        # Train GAP model
        logger.info("Training GAP model on %d matches...", len(train_df))
        model = GAPModel(
            k=args.k,
            stat_column=args.stat,
            home_adv=args.home_adv,
            decay_halflife_days=args.decay,
        )
        model.fit(train_df)

        # Evaluate on test set
        logger.info("Evaluating on %d test matches...", len(test_df))
        eval_results = model.evaluate(test_df)
        print(f"\n  GAP EVALUATION:")
        print(f"  {'Brier':<25} {eval_results['brier']:<10}")
        print(f"  {'Accuracy':<25} {eval_results['accuracy']*100:.2f}%")
        print(f"  {'LogLoss':<25} {eval_results['log_loss']:<10}")
        print(f"  {'Actual Over rate':<25} {eval_results['over_rate']*100:.2f}%")

        # Backtest
        bt = run_gap_backtest(train_df, test_df, model,
                              min_ev=args.min_ev, kelly_frac=args.kelly)

        title = f"(K={args.k}, decay={args.decay}d, home_adv={args.home_adv})"
        print_report(league, model, bt, extra_title=title)

        all_results.append({
            "league": league,
            "model": "GAP",
            "stat_column": args.stat,
            "k": args.k,
            "eval": eval_results,
            "backtest": bt["metrics"],
            "n_bets": len(bt["bets"]),
            "bets": bt["bets"],
        })

    # Cross-league summary
    if all_results:
        print()
        print("=" * 80)
        print("  CROSS-LEAGUE GAP MODEL COMPARISON")
        print("=" * 80)
        print()
        print(f"  {'League':<8} {'Stat':<18} {'K':>4} {'Brier':>8} {'Acc':>6} "
              f"{'Bets':>6} {'Over%':>8} {'Under%':>8}")
        print(f"  {'─'*8} {'─'*18} {'─'*4} {'─'*8} {'─'*6} "
              f"{'─'*6} {'─'*8} {'─'*8}")
        for r in all_results:
            e = r["eval"]
            bt = r["backtest"]
            over_y = f"{bt['over_yield']:+.1f}%" if bt['n_over_2_5'] > 0 else "—"
            under_y = f"{bt['under_yield']:+.1f}%" if bt['n_under_2_5'] > 0 else "—"
            print(f"  {r['league']:<8} {r['stat_column']:<18} {r['k']:>4.0f} "
                  f"{e['brier']:>8.4f} {e['accuracy']*100:>5.1f}% "
                  f"{bt['n_bets']:>6} {over_y:>8s} {under_y:>8s}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
