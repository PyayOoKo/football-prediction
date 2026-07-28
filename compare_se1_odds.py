"""Compare SE1 model predictions vs live bookmaker odds.

Loads saved predictions (from predict_se1.py) and fetches live odds
from The Odds API. Shows recommended level-stake bets based on the
only historically profitable strategy: home wins with odds 2-3.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

API_KEY = "e89f00453039792b523194c09cb186d7"
API_BASE = "https://api.the-odds-api.com/v4"

# Level stake per recommended bet (e.g. 1 unit = £10)
UNIT_STAKE = 10.0

DB_TO_API = {
    "Falkenberg": "Falkenbergs FF",
    "Norrby": "Norrby IF",
    "Östersund": "Östersunds FK",
    "Örebro": "Örebro SK",
    "Öster": "Östers IF",
    "Värnamo": "IFK Värnamo",
    "Norrköping": "IFK Norrkoping",
    "Nordic United": "Nordic United FC",
    "Sandvikens If": "Sandvikens IF",
    "Ljungskile Sk": "Ljungskile SK",
    "Helsingborgs IF": "Helsingborgs IF",
    "GIF Sundsvall": "GIF Sundsvall",
    "IK Brage": "IK Brage",
    "IK Oddevold": "IK Oddevold",
    "Varbergs BoIS": "Varbergs BoIS",
    "Landskrona BoIS": "Landskrona BoIS",
    "Utsikten": "Utsiktens BK",
}


def fetch_live_odds():
    url = f"{API_BASE}/sports/soccer_sweden_superettan/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "uk,eu,se",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    odds_map = {}
    for event in data:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    odds_map[(home, away)] = {
                        "home_odds": outcomes.get(home, 0),
                        "draw_odds": outcomes.get("Draw", 0),
                        "away_odds": outcomes.get(away, 0),
                        "bookmaker": bk["title"],
                    }
    return odds_map


def resolve_team_name(name):
    return DB_TO_API.get(name, name)


def find_match_odds(home, away, live_odds):
    api_home = resolve_team_name(home)
    api_away = resolve_team_name(away)
    odds = live_odds.get((api_home, api_away))
    if odds is None:
        odds = live_odds.get((home, away))
    if odds is None:
        odds = live_odds.get((away, home))
    if odds is None:
        for (oh, oa), v in live_odds.items():
            if oh.lower() == home.lower() and oa.lower() == away.lower():
                odds = v
                break
            if oa.lower() == home.lower() and oh.lower() == away.lower():
                odds = v
                break
    return odds


def get_implied_probs(odds_dict):
    h = 1.0 / odds_dict["home_odds"] if odds_dict["home_odds"] > 0 else 0
    d = 1.0 / odds_dict["draw_odds"] if odds_dict["draw_odds"] > 0 else 0
    a = 1.0 / odds_dict["away_odds"] if odds_dict["away_odds"] > 0 else 0
    margin = (h + d + a) - 1.0
    return {
        "home": h / (1 + margin),
        "draw": d / (1 + margin),
        "away": a / (1 + margin),
        "margin": margin,
    }


def log_bet(home, away, odds, stake, model_prob):
    """Append a bet to the bet log for performance tracking."""
    log_path = Path("reports/bet_log_se1.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "home": home,
        "away": away,
        "bet": "H",
        "odds": round(odds, 2),
        "stake": round(stake, 2),
        "model_prob": round(model_prob, 4),
        "result": "",
        "profit": "",
    }
    if log_path.exists():
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(log_path, index=False)
    print(f"    -> Logged to {log_path}")


def main():
    preds_path = Path("reports/predictions/SE1_today.csv")
    if not preds_path.exists():
        print("No saved predictions found. Run predict_se1.py first.")
        return
    preds = pd.read_csv(preds_path)

    print("Fetching live odds from The Odds API...")
    live_odds = fetch_live_odds()
    print(f"Got odds for {len(live_odds)} matches from API\n")

    if not live_odds:
        print("No live odds available. SE1 may not be in season or API limit reached.")
        return

    # ═══════════════════════════════════════════════════════════
    #  RECOMMENDED BETS  (profitable strategy: level stakes)
    #  Only home wins with odds 2-3 where model prob > implied
    # ═══════════════════════════════════════════════════════════
    recommended = []
    comparison_rows = []

    for _, row in preds.iterrows():
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()

        h_prob = row["home_win_prob"]
        d_prob = row["draw_prob"]
        a_prob = row["away_win_prob"]

        odds = find_match_odds(home, away, live_odds)
        if odds is None:
            continue

        implied = get_implied_probs(odds)
        diff_h = (h_prob - implied["home"]) * 100

        comparison_rows.append((home, away, h_prob, d_prob, a_prob, odds, implied, diff_h))

        # Only consider home win bets with odds 2-3
        home_odds = odds["home_odds"]
        if home_odds < 2.0 or home_odds > 3.0:
            continue

        model_edge = h_prob - implied["home"]
        if model_edge <= 0:
            continue

        ev_pct = (h_prob * home_odds - 1.0) * 100

        recommended.append({
            "home": home,
            "away": away,
            "odds": home_odds,
            "model_prob": h_prob,
            "fair_prob": implied["home"],
            "edge_pp": round(model_edge * 100, 1),
            "ev_pct": round(ev_pct, 1),
            "stake": UNIT_STAKE,
        })

    # Print recommended bets
    if recommended:
        print("=" * 80)
        print("  >>> RECOMMENDED BETS  (level stakes: £%.0f/unit)" % UNIT_STAKE)
        print("  >>> Strategy: Home win, odds 2.0-3.0, model prob > fair prob")
        print("=" * 80)
        total_stake = 0
        for r in recommended:
            total_stake += r["stake"]
            print(f"\n  {r['home']:<28s} vs {r['away']:<28s}")
            print(f"    Odds: {r['odds']:.2f}  |  Model: {r['model_prob']:.1%}  |  Fair: {r['fair_prob']:.1%}")
            print(f"    Edge: {r['edge_pp']:+.1f}pp  |  EV: {r['ev_pct']:+.1f}%")
            print(f"    >>> STAKE: £{r['stake']:.0f}  (level stake)")
            log_bet(r["home"], r["away"], r["odds"], r["stake"], r["model_prob"])

        print(f"\n  Total units: {len(recommended)}  |  Total stake: £{total_stake:.0f}")
        print(f"  Max loss: £{total_stake:.0f}  (if all lose)")
    else:
        print("NO RECOMMENDED BETS — no home wins with odds 2-3 where model > fair")
        print("Either no matches qualify or the model sees no edge this round.")

    # ═══════════════════════════════════════════════════════════
    #  FULL COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════
    print("\n\n" + "=" * 155)
    print(f"{'Match':^60} | {'Model Probs':^25} | {'Fair Odds':^20} | {'Bookmaker Odds':^25} | {'Diff (pp)':^18}")
    print("=" * 155)

    for home, away, h_prob, d_prob, a_prob, odds, implied, diff_h in comparison_rows:
        label = f"{home[:18]} vs {away[:18]}"
        model_str = f"H{h_prob:.0%} D{d_prob:.0%} A{a_prob:.0%}"
        fair_str = f"{1/h_prob:.2f}  {1/d_prob:.2f}  {1/a_prob:.2f}"
        book_str = f"{odds['home_odds']:.2f}  {odds['draw_odds']:.2f}  {odds['away_odds']:.2f}"
        diff_d = (d_prob - implied["draw"]) * 100
        diff_a = (a_prob - implied["away"]) * 100
        diff_str = f"H{diff_h:+6.1f}  D{diff_d:+6.1f}  A{diff_a:+6.1f}"

        # Highlight recommended bets
        marker = " <<<" if any(r["home"] == home and r["away"] == away for r in recommended) else ""
        print(f"{label:^60} | {model_str:^25} | {fair_str:^20} | {book_str:^25} | {diff_str:^18}{marker}")

    print(f"\nCompared {len(comparison_rows)} matches with live odds")
    print(f"Positive diff = model > fair prob = potential value bet")
    print(f"'<<<' marks recommended level-stake bets (home odds 2-3, model > fair)")


if __name__ == "__main__":
    main()
