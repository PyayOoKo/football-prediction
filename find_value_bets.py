"""
find_value_bets.py — Find value bets using live odds (if available) or hardcoded fallback.

Tries to fetch live odds from The Odds API for each match in the predictions CSV.
If the API is unavailable or no API key is set, falls back to hardcoded industry-average odds.

Usage:
    python find_value_bets.py                          # Auto: live odds with fallback
    python find_value_bets.py --force-hardcoded        # Skip API, use hardcoded odds
    python find_value_bets.py --live-only              # Only use API, fail if unavailable
    python find_value_bets.py --sport-key soccer_epl   # Different sport key
    python find_value_bets.py --bookmaker bet365       # Specific bookmaker
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from src.odds_api import OddsAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("find_value_bets")

PREDICTIONS_CSV = Path("reports/predictions_worldcup/worldcup_predictions.csv")
OUTCOME_LABELS = ["Away Win", "Draw", "Home Win"]

# ── Hardcoded fallback odds (industry averages for 2026 R16) ──
# Used when the Odds API is unavailable or --force-hardcoded is set.
HARDCODED_ODDS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("Brazil", "Norway"):        (4.20, 3.70, 1.83),   # Brazil heavy favorite
    ("Mexico", "England"):       (3.10, 3.15, 2.42),   # England favorite
    ("Portugal", "Spain"):       (2.80, 3.10, 2.50),   # Portugal slight fave (home)
    ("USA", "Belgium"):          (3.75, 3.40, 1.95),   # Belgium favorite
    ("Argentina", "Egypt"):      (7.50, 4.40, 1.40),   # Argentina huge favorite
    ("Switzerland", "Colombia"): (3.25, 3.00, 2.35),   # Switzerland slight fave (home)
    ("France", "Morocco"):       (3.25, 3.10, 2.30),   # France slight favorite
    ("France", "Morocco"):       (3.25, 3.10, 2.30),   # France slight favorite (QF Jul 9)
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find value bets from model predictions vs live/hardcoded odds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force-hardcoded", action="store_true",
        help="Skip API, use hardcoded odds only",
    )
    parser.add_argument(
        "--live-only", action="store_true",
        help="Only use API odds — fail if unavailable",
    )
    parser.add_argument(
        "--sport-key", type=str, default="soccer_fifa_world_cup",
        help="Sport key for The Odds API (default: soccer_fifa_world_cup)",
    )
    parser.add_argument(
        "--bookmaker", type=str, default=None,
        help="Specific bookmaker (e.g. bet365). Default: best across all",
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Bankroll for Kelly stake calculation (default: 1000)",
    )
    parser.add_argument(
        "--kelly", type=float, default=0.25,
        help="Kelly fraction (default: 0.25 = 25 pct Kelly)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 95)
    print("  WORLD CUP 2026 — VALUE BET ANALYSIS")
    print("=" * 95)

    # ── Load predictions ──
    if not PREDICTIONS_CSV.exists():
        print(f"\n  [X] Predictions not found at {PREDICTIONS_CSV}")
        print("    Run: python train_worldcup.py")
        return 1

    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"\n  Loaded {len(df)} match predictions from {PREDICTIONS_CSV}")

    # ── Fetch odds ──
    team_pairs = list(zip(df["home_team"], df["away_team"]))

    if args.force_hardcoded:
        print("  Odds source: HARDCODED (--force-hardcoded)")
        live_odds: dict[tuple[str, str], dict[str, float]] = {}
    else:
        client = OddsAPIClient()
        if not client.api_key and args.live_only:
            print("\n  [X] THE_ODDS_API_KEY not set and --live-only specified.")
            print("  Set the env var or remove --live-only to use fallback.")
            print("  Get a free key at https://the-odds-api.com/")
            return 1

        if client.api_key:
            print(f"  Odds source: LIVE (region={config.odds_api.regions}, "
                  f"sport={args.sport_key}, bookmaker={args.bookmaker or 'best'})")
            live_odds = client.get_value_bet_odds(
                team_pairs,
                sport_key=args.sport_key,
                bookmaker=args.bookmaker,
            )
            if live_odds:
                print(f"  [OK] Fetched live odds for {len(live_odds)}/{len(team_pairs)} matches")
            elif args.live_only:
                print("\n  [X] Live odds returned no matches and --live-only is set.")
                return 1
            else:
                print("  [!] Live odds unavailable — falling back to hardcoded odds")
                live_odds = {}
        else:
            print("  Odds source: HARDCODED (no API key — set THE_ODDS_API_KEY for live odds)")
            live_odds = {}

    # ── Compute value bets ──
    all_bets = []
    odds_source_info = "HARDCODED" if not live_odds else "LIVE"

    for _, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        model_probs = [row["away_win_prob"], row["draw_prob"], row["home_win_prob"]]
        match_key = (h, a)

        # Get odds: prefer live, fall back to hardcoded
        if match_key in live_odds:
            odds_data = live_odds[match_key]
            away_odds = odds_data["away_odds"]
            draw_odds = odds_data["draw_odds"]
            home_odds = odds_data["home_odds"]
            bookmaker = odds_data.get("bookmaker", "API")
            source_label = f"LIVE ({bookmaker})"
        elif match_key in HARDCODED_ODDS:
            away_odds, draw_odds, home_odds = HARDCODED_ODDS[match_key]
            bookmaker = "industry avg"
            source_label = "HARDCODED"
        else:
            print(f"  [!] No odds for {h} vs {a} — skipping")
            continue

        odds_array = [away_odds, draw_odds, home_odds]

        # ── Compute value metrics ──
        implied_probs = 1.0 / np.array(odds_array)
        margin = implied_probs.sum() - 1.0
        fair_probs = implied_probs / (1.0 + margin)

        for idx, outcome in enumerate(OUTCOME_LABELS):
            mod_prob = model_probs[idx]
            dec_odds = odds_array[idx]
            fair_prob = fair_probs[idx]

            ev = (mod_prob * dec_odds) - 1.0
            edge = mod_prob - fair_prob

            # Kelly stake
            if dec_odds > 1.0 and ev > 0:
                full_kelly = ev / (dec_odds - 1.0)
            else:
                full_kelly = 0.0
            kelly_pct = max(full_kelly * args.kelly, 0.0)
            kelly_stake = args.bankroll * kelly_pct

            is_value = bool(ev > 0 and edge > 0 and kelly_pct > 0)

            all_bets.append({
                "match": f"{h} vs {a}",
                "home": h,
                "away": a,
                "outcome": outcome,
                "odds": dec_odds,
                "model_prob": round(mod_prob, 4),
                "fair_prob": round(fair_prob, 4),
                "edge_pp": round(edge * 100, 2),
                "ev_pct": round(ev * 100, 2),
                "kelly_pct": round(kelly_pct * 100, 2),
                "kelly_stake": round(kelly_stake, 2),
                "is_value": is_value,
                "source": source_label,
                "margin_pct": round(margin * 100, 2),
            })

    if not all_bets:
        print("\n  No odds data available for any matches.")
        return 1

    # ── Sort ──
    bets_df = pd.DataFrame(all_bets)
    bets_df.sort_values(["is_value", "ev_pct"], ascending=[False, False], inplace=True)
    bets_df.reset_index(drop=True, inplace=True)

    value_bets = bets_df[bets_df["is_value"]]
    no_value = bets_df[~bets_df["is_value"]]

    # ── Display header ──
    print(f"\n  {'=' * 95}")
    print(f"  RESULTS")
    print(f"  Odds source: {odds_source_info}")
    print(f"  {'=' * 95}")

    # ── Value bets ──
    if len(value_bets) > 0:
        print(f"\n  VALUE BETS — {len(value_bets)} opportunities found")
        print(f"  {'-' * 95}")
        print(f"  {'Match':<28} {'Outcome':<18} {'Odds':<8} {'Model%':<8} "
              f"{'Fair%':<8} {'Edge':<8} {'EV':<8} {'Stake':<10}")
        print(f"  {'-' * 95}")
        for _, b in value_bets.iterrows():
            ms = f"{b['home'][:12]} vs {b['away'][:12]}"
            print(f"  {ms:<28} {b['outcome']:<18} {b['odds']:<8.2f} "
                  f"{b['model_prob']:<7.1%} {b['fair_prob']:<7.1%} "
                  f"{b['edge_pp']:+7.1f}pp {b['ev_pct']:+7.1f}% "
                  f"Pound{b['kelly_stake']:<6.2f}")

        # Detailed explanations
        print(f"\n  DETAILED EXPLANATION:\n")
        for _, b in value_bets.iterrows():
            print(f"  {b['match']}")
            print(f"     Bet: {b['outcome']} @ {b['odds']:.2f}")
            print(f"     Source: {b['source']} | Bookmaker margin: {b['margin_pct']:.1f}%")
            print(f"     Model probability:      {b['model_prob']:.1%}")
            print(f"     Bookmaker fair prob:    {b['fair_prob']:.1%}")
            print(f"     Edge:                   {b['edge_pp']:+.1f}pp")
            print(f"     Expected Value:         {b['ev_pct']:+.1f}%")
            print(f"     Kelly stake ({args.kelly*100:.0f}pct):    Pound{b['kelly_stake']:.2f} ({b['kelly_pct']:.1f}% of Pound{args.bankroll:.0f})")
            print()

        # Best value bet
        best = value_bets.iloc[0]
        print(f"  {'*' * 95}")
        print(f"  BEST VALUE BET: {best['match']} — {best['outcome']} @ {best['odds']:.2f}")
        print(f"    Model sees {best['model_prob']:.1%} vs market's {best['fair_prob']:.1%}")
        print(f"    Edge: {best['edge_pp']:+.1f}pp  |  EV: {best['ev_pct']:+.1f}%  |  Stake: Pound{best['kelly_stake']:.2f}")
        print(f"  {'*' * 95}")
    else:
        print("\n  No value bets found with current odds.")
        print("  The market may be efficiently pricing these matchups.")

    # ── Full summary table ──
    print(f"\n  {'=' * 95}")
    print(f"  ALL MATCHES — OUTCOME BREAKDOWN")
    print(f"  {'=' * 95}")
    print(f"  {'Match':<28} {'Home':<22} {'Draw':<8} {'Away':<22}")
    print(f"  {'-' * 82}")
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hw, dr, aw = row["home_win_prob"], row["draw_prob"], row["away_win_prob"]
        print(f"  {h:<12} vs {a:<12}  "
              f"Home {hw:<5.1%}  Draw {dr:<5.1%}  Away {aw:<5.1%}")

    # ── API setup instructions ──
    if not live_odds:
        print(f"\n  {'=' * 95}")
        print(f"  To use LIVE odds:")
        print(f"    1. Get a free API key at https://the-odds-api.com/")
        print(f"    2. Set:  export THE_ODDS_API_KEY='your_key_here'")
        print(f"    3. Run:  python find_value_bets.py")
        print(f"  {'=' * 95}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
