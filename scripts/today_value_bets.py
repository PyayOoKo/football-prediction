"""Today's Value Bets -- July 6, 2026 (World Cup R16)"""

import numpy as np
import pandas as pd
import sys

OUTCOME_LABELS = ["Away Win", "Draw", "Home Win"]
BANKROLL = 1000.0
KELLY_FRACTION = 0.25

HARDCODED_ODDS = {
    ("Portugal", "Spain"): (2.80, 3.10, 2.50),
    ("USA", "Belgium"): (3.75, 3.40, 1.95),
}

df = pd.read_csv("reports/predictions_worldcup/worldcup_predictions.csv")
today_matches = df[df["date"] == "2026-07-06"]

print("=" * 95)
print("  WORLD CUP 2026 -- TODAY'S VALUE BETS  (July 6 -- Round of 16)")
print("=" * 95)

for _, row in today_matches.iterrows():
    h, a = row["home_team"], row["away_team"]
    model_probs = [row["away_win_prob"], row["draw_prob"], row["home_win_prob"]]
    match_key = (h, a)

    if match_key not in HARDCODED_ODDS:
        print(f"\n  No odds data for {h} vs {a}")
        continue

    away_odds, draw_odds, home_odds = HARDCODED_ODDS[match_key]
    odds_array = [away_odds, draw_odds, home_odds]

    implied_probs = 1.0 / np.array(odds_array)
    margin = implied_probs.sum() - 1.0
    fair_probs = implied_probs / (1.0 + margin)

    print(f"\n  {'-' * 55}")
    print(f"  {h} vs {a}")
    print(f"  {'-' * 55}")

    value_bets = []

    for idx, outcome in enumerate(OUTCOME_LABELS):
        mod_prob = model_probs[idx]
        dec_odds = odds_array[idx]
        fair_prob = fair_probs[idx]
        ev = (mod_prob * dec_odds) - 1.0
        edge = mod_prob - fair_prob

        if dec_odds > 1.0 and ev > 0:
            full_kelly = ev / (dec_odds - 1.0)
        else:
            full_kelly = 0.0
        kelly_pct = max(full_kelly * KELLY_FRACTION, 0.0)
        kelly_stake = BANKROLL * kelly_pct
        is_value = bool(ev > 0 and edge > 0 and kelly_pct > 0)

        tag = "[VALUE]" if is_value else " [----]"
        print(f"    {outcome:<12} @ {dec_odds:<5.2f}  "
              f"Model: {mod_prob:.1%}  Fair: {fair_prob:.1%}  "
              f"Edge: {edge:+.1%}  EV: {ev:+.1%}  {tag}")
        if is_value:
            value_bets.append((outcome, dec_odds, kelly_stake, kelly_pct, ev, edge))

    print(f"    {'-' * 55}")
    print(f"    Bookmaker margin: {margin:.1%}")

    if value_bets:
        print(f"\n  [BETS] VALUE BETS IN THIS MATCH:")
        for outcome, odds, stake, kelly_pct, ev, edge in value_bets:
            print(f"     -> {outcome} @ {odds:.2f} -- Bet GBP{stake:.2f} ({kelly_pct:.1%} of bankroll)")
    else:
        print(f"\n  [NO] No value bets in this match - market is efficient.")

print(f"\n{'=' * 95}")
print(f"  SUMMARY")
print(f"{'=' * 95}")

all_bets = []
for _, row in today_matches.iterrows():
    h, a = row["home_team"], row["away_team"]
    probs = [row["away_win_prob"], row["draw_prob"], row["home_win_prob"]]
    match_key = (h, a)
    if match_key not in HARDCODED_ODDS:
        continue
    odds_arr = list(HARDCODED_ODDS[match_key])
    imp = 1.0 / np.array(odds_arr)
    margin = imp.sum() - 1.0
    fair_arr = imp / (1.0 + margin)
    for idx, outcome in enumerate(OUTCOME_LABELS):
        ev = (probs[idx] * odds_arr[idx]) - 1.0
        edge = probs[idx] - fair_arr[idx]
        if ev > 0 and edge > 0:
            all_bets.append((h, a, outcome, odds_arr[idx], ev, edge, probs[idx], fair_arr[idx]))

all_bets.sort(key=lambda x: x[4], reverse=True)

if all_bets:
    best = all_bets[0]
    print(f"\n  [BEST] BEST VALUE BET TODAY:")
    print(f"     {best[0]} vs {best[1]} - {best[2]} @ {best[3]:.2f}")
    print(f"     Model: {best[6]:.1%}  |  Market fair: {best[7]:.1%}  |  Edge: {best[5]:+.1%}  |  EV: {best[4]:+.1%}")

    print(f"\n  All value bets ranked by EV:")
    print(f"  {'Match':<28} {'Outcome':<14} {'Odds':<8} {'Model%':<10} {'Edge':<10} {'EV':<10}")
    print(f"  {'-' * 75}")
    for h, a, outcome, odds, ev, edge, mp, fp in all_bets:
        ms = f"{h[:14]:14} vs {a[:14]:14}"
        print(f"  {ms:<28} {outcome:<14} {odds:<8.2f} {mp:<9.1%} {edge:<+9.1%} {ev:<+9.1%}")
else:
    print("\n  No value bets found for today's matches.")

print(f"\n{'=' * 95}")
print(f"  Odds source: Hardcoded (industry avg) | Kelly: {KELLY_FRACTION:.0%}")
print(f"  Bankroll: GBP{BANKROLL:.0f}")
print(f"  To get live odds: set THE_ODDS_API_KEY and run without --force-hardcoded")
print(f"{'=' * 95}")
sys.stdout.flush()
