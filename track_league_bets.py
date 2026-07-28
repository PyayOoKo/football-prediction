"""
track_league_bets.py — Track profitability of live league value bets.

Loads saved bet reports from today_league_value_bets.py, looks up
match results in the DB, and computes profit/loss.

Usage
-----
    python track_league_bets.py                          # Track all leagues
    python track_league_bets.py --leagues SE1             # Specific league
    python track_league_bets.py --leagues SE1 --since 7   # Last 7 days
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("track_league_bets")

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
REPORTS_DIR = PROJECT_ROOT / "reports" / "value_bets"

OUTCOME_MAP = {"H": 2, "D": 1, "A": 0}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Track league value bet profitability")
    p.add_argument("--leagues", nargs="+", help="League codes (default: all)")
    p.add_argument("--since", type=int, default=30, help="Days back to check (default: 30)")
    p.add_argument("--detail", action="store_true", help="Show individual bet detail")
    return p.parse_args(argv)


def get_db_result(home: str, away: str, date_str: str) -> str | None:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(
        "SELECT result FROM matches WHERE home_team=? AND away_team=? AND date=?",
        (home, away, date_str),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def main(argv=None):
    args = parse_args(argv)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.since)).strftime("%Y-%m-%d")

    # Find all report files
    pattern = str(REPORTS_DIR / "value_bets_*.json")
    report_files = sorted(glob.glob(pattern))

    if not report_files:
        logger.warning("No bet reports found in %s", REPORTS_DIR)
        return 1

    # Group reports by league and aggregate bets
    league_bets: dict[str, list[dict[str, Any]]] = {}
    for fpath in report_files:
        with open(fpath) as f:
            data = json.load(f)
        league = data.get("league", "unknown")
        if args.leagues and league not in args.leagues:
            continue
        gen_time = data.get("generated_at", "")[:10]
        if gen_time < cutoff:
            continue
        league_bets.setdefault(league, []).extend(data.get("bets", []))

    if not league_bets:
        logger.info("No bets found in the specified time range")
        return 0

    total_bankroll = 0.0
    total_staked = 0.0
    total_profit = 0.0
    total_won = 0
    total_lost = 0
    total_pending = 0

    for league in sorted(league_bets.keys()):
        bets = league_bets[league]
        won = lost = pending = 0
        staked = profit = 0.0
        settled_bets = []

        for b in bets:
            date_str = b.get("date", "")
            home = b["home"]
            away = b["away"]
            outcome = b["outcome"]
            odds = b["odds"]
            stake = b["stake"]

            if not date_str:
                pending += 1
                continue

            result = get_db_result(home, away, date_str)

            if result is None:
                pending += 1
                continue

            outcome_idx = OUTCOME_MAP.get(outcome, -1)
            actual_idx = OUTCOME_MAP.get(result, -1)
            is_win = outcome_idx == actual_idx

            bet_profit = stake * (odds - 1) if is_win else -stake
            staked += stake
            profit += bet_profit
            staked = round(staked, 2)
            profit = round(profit, 2)

            if is_win:
                won += 1
            else:
                lost += 1

            settled_bets.append({**b, "actual_result": result, "won": is_win, "profit": bet_profit})

        n_settled = won + lost
        yield_pct = (profit / staked * 100) if staked > 0 else 0.0
        win_rate = (won / n_settled * 100) if n_settled > 0 else 0.0

        total_bankroll += 0
        total_staked += staked
        total_profit += profit
        total_won += won
        total_lost += lost
        total_pending += pending

        print()
        print("=" * 70)
        print(f"  {league}")
        print("=" * 70)

        if n_settled == 0:
            print(f"  No settled bets ({pending} pending)")
            continue

        print(f"  Settled:     {n_settled}")
        print(f"  Won / Lost:  {won} / {lost}")
        print(f"  Win rate:    {win_rate:.1f}%")
        print(f"  Total staked: GBP {staked:,.2f}")
        print(f"  Total profit: GBP {profit:+,.2f}")
        print(f"  Yield (ROI): {yield_pct:+.2f}%")
        print(f"  Pending:     {pending}")

        if args.detail and settled_bets:
            print()
            print(f"  {'Date':<12} {'Home':<18} {'Away':<18} {'Pick':<5} {'Odds':<6} {'Result':<7} {'Profit':<10}")
            print(f"  {'-'*76}")
            settled_bets.sort(key=lambda x: x.get("date", ""))
            for b in settled_bets:
                d = b.get("date", "")[-5:]
                r = b.get("actual_result", "?")
                won_str = "WON" if b.get("won") else "LOST"
                pf = f"GBP {b['profit']:+,.1f}"
                print(f"  {d:<12} {b['home'][:16]:<18} {b['away'][:16]:<18} {b['outcome']:<5} {b['odds']:<6.2f} {r:<7} {pf:<10}")

    # Overall summary
    print()
    print("=" * 70)
    print("  OVERALL PROFITABILITY")
    print("=" * 70)
    total_settled = total_won + total_lost
    overall_yield = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    overall_win_rate = (total_won / total_settled * 100) if total_settled > 0 else 0.0

    print(f"  Leagues:     {', '.join(league_bets.keys())}")
    print(f"  Settled:     {total_settled}")
    print(f"  Won / Lost:  {total_won} / {total_lost}")
    print(f"  Win rate:    {overall_win_rate:.1f}%")
    print(f"  Total staked: GBP {total_staked:,.2f}")
    print(f"  Total profit: GBP {total_profit:+,.2f}")
    print(f"  Yield (ROI): {overall_yield:+.2f}%")
    print(f"  Pending:     {total_pending}")

    if total_settled > 0:
        print()
        if total_profit > 0:
            print(f"  \U0001f7e2 PROFITABLE: +GBP {total_profit:+,.2f} ({overall_yield:+.2f}% yield)")
        elif total_profit < 0:
            print(f"  \U0001f534 NOT PROFITABLE: GBP {total_profit:+,.2f} ({overall_yield:+.2f}% yield)")
        else:
            print(f"  \U0001f7e1 BREAK EVEN: GBP {total_profit:+,.2f}")

    return 0 if total_profit >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
