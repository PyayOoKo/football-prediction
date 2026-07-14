"""
what_if_canada_morocco.py — What-If Analysis: Canada vs Morocco R16 Outcome Scenarios.

Forces each possible result of Canada vs Morocco (R16-1) and simulates the
full knockout bracket to show how the tournament tree changes.

Usage:
    python what_if_canada_morocco.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = PROJECT_ROOT / "reports" / "predictions_worldcup" / "worldcup_predictions.csv"

# ── Bracket structure ──
R16_MATCHUPS = [
    ("R16-1", "Canada", "Morocco"),
    ("R16-2", "Paraguay", "France"),
    ("R16-3", "Brazil", "Norway"),
    ("R16-4", "Mexico", "England"),
    ("R16-5", "Portugal", "Spain"),
    ("R16-6", "USA", "Belgium"),
    ("R16-7", "Argentina", "Egypt"),
    ("R16-8", "Switzerland", "Colombia"),
]

QF_WINNER_PAIRS = [
    ("QF-1", "R16-1", "R16-2"),
    ("QF-2", "R16-3", "R16-4"),
    ("QF-3", "R16-5", "R16-6"),
    ("QF-4", "R16-7", "R16-8"),
]

SF_PAIRS = [
    ("SF-1", "QF-1", "QF-2"),
    ("SF-2", "QF-3", "QF-4"),
]

FINAL_MATCHUP = ("Final", "SF-1", "SF-2")


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        print(f"  [X] Predictions not found at {PREDICTIONS_CSV}")
        sys.exit(1)
    return pd.read_csv(PREDICTIONS_CSV).sort_values("date").reset_index(drop=True)


def build_match_probs(predictions: pd.DataFrame) -> dict:
    match_probs = {}
    for _, row in predictions.iterrows():
        h, a = row["home_team"], row["away_team"]
        hw, dr, aw = row["home_win_prob"], row["draw_prob"], row["away_win_prob"]
        match_probs[(h, a)] = (hw, dr, aw)
        match_probs[(a, h)] = (aw, dr, hw)
    return match_probs


def simulate_bracket(
    predictions: pd.DataFrame,
    match_probs: dict,
    forced_r16_results: dict[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate bracket with optional forced R16 outcomes."""
    use_mc = rng is not None
    results = {"rounds": {}}

    # Round of 16
    r16_results = {}
    for r16_id, h, a in R16_MATCHUPS:
        hw, dr, aw = match_probs.get((h, a), (1/3, 1/3, 1/3))
        probs = np.array([aw, dr, hw])
        probs = probs / probs.sum()

        if forced_r16_results and r16_id in forced_r16_results:
            winner = forced_r16_results[r16_id]
        elif use_mc:
            winner_idx = rng.choice(3, p=probs)
            if winner_idx == 2:
                winner = h
            elif winner_idx == 1:
                winner = h if hw >= aw else a
            else:
                winner = a
        else:
            winner_idx = np.argmax(probs)
            if winner_idx == 2:
                winner = h
            elif winner_idx == 0:
                winner = a
            else:
                winner = h if hw >= aw else a

        r16_results[r16_id] = {
            "home": h, "away": a, "winner": winner,
            "home_win": hw, "draw": dr, "away_win": aw,
            "confidence": max(hw, dr, aw),
        }
    results["rounds"]["Round of 16"] = r16_results

    # Quarter-Finals
    qf_results = {}
    for qf_id, r16_a_key, r16_b_key in QF_WINNER_PAIRS:
        r16_a, r16_b = r16_results[r16_a_key], r16_results[r16_b_key]
        h, a = r16_a["winner"], r16_b["winner"]

        if (h, a) in match_probs:
            hw, dr, aw = match_probs[(h, a)]
        else:
            edge = (r16_a["confidence"] - r16_b["confidence"]) * 0.15
            hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
            hw, aw = np.clip(hw, 0.25, 0.55), np.clip(aw, 0.20, 0.50)
            total = hw + dr + aw
            hw, dr, aw = hw / total, dr / total, aw / total

        probs = np.array([aw, dr, hw])
        if use_mc:
            winner_idx = rng.choice(3, p=probs)
        else:
            winner_idx = np.argmax(probs)

        if winner_idx == 2:
            winner = h
        elif winner_idx == 0:
            winner = a
        else:
            winner = h if hw >= aw else a

        qf_results[qf_id] = {
            "home": h, "away": a, "winner": winner,
            "home_win": hw, "draw": dr, "away_win": aw,
            "confidence": max(hw, dr, aw),
        }
    results["rounds"]["Quarter-Finals"] = qf_results

    # Semi-Finals
    sf_results = {}
    for sf_id, qf_a_key, qf_b_key in SF_PAIRS:
        h, a = qf_results[qf_a_key]["winner"], qf_results[qf_b_key]["winner"]

        if (h, a) in match_probs:
            hw, dr, aw = match_probs[(h, a)]
        else:
            prev_h, prev_a = qf_results[qf_a_key]["confidence"], qf_results[qf_b_key]["confidence"]
            edge = (prev_h - prev_a) * 0.15
            hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
            hw, aw = np.clip(hw, 0.25, 0.55), np.clip(aw, 0.20, 0.50)
            total = hw + dr + aw
            hw, dr, aw = hw / total, dr / total, aw / total

        probs = np.array([aw, dr, hw])
        if use_mc:
            winner_idx = rng.choice(3, p=probs)
        else:
            winner_idx = np.argmax(probs)

        if winner_idx == 2:
            winner = h
        elif winner_idx == 0:
            winner = a
        else:
            winner = h if hw >= aw else a

        sf_results[sf_id] = {
            "home": h, "away": a, "winner": winner,
            "home_win": hw, "draw": dr, "away_win": aw,
            "confidence": max(hw, dr, aw),
        }
    results["rounds"]["Semi-Finals"] = sf_results

    # Final
    _, sf_a, sf_b = FINAL_MATCHUP
    h, a = sf_results[sf_a]["winner"], sf_results[sf_b]["winner"]

    if (h, a) in match_probs:
        hw, dr, aw = match_probs[(h, a)]
    else:
        prev_h, prev_a = sf_results[sf_a]["confidence"], sf_results[sf_b]["confidence"]
        edge = (prev_h - prev_a) * 0.15
        hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
        hw, aw = np.clip(hw, 0.25, 0.55), np.clip(aw, 0.20, 0.50)
        total = hw + dr + aw
        hw, dr, aw = hw / total, dr / total, aw / total

    probs = np.array([aw, dr, hw])
    probs = probs / probs.sum()

    if use_mc:
        winner_idx = rng.choice(3, p=probs)
    else:
        winner_idx = np.argmax(probs)

    if winner_idx == 2:
        winner = h
    elif winner_idx == 0:
        winner = a
    else:
        winner = h if hw >= aw else a

    results["rounds"]["Final"] = {
        "Final": {
            "home": h, "away": a, "winner": winner,
            "home_win": hw, "draw": dr, "away_win": aw,
            "confidence": max(hw, dr, aw),
        }
    }
    results["champion"] = winner
    return results


def print_bracket(results: dict, label: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    for round_name in ["Round of 16", "Quarter-Finals", "Semi-Finals"]:
        print(f"\n  ── {round_name} ──")
        for match_id, data in results["rounds"][round_name].items():
            icon = "▲" if data["winner"] == data["home"] else "▼"
            print(f"  {data['home']:<18} vs {data['away']:<18} → {icon} {data['winner']:<18}")

    final = results["rounds"]["Final"]["Final"]
    print(f"\n  ── FINAL ──")
    icon = "▲" if final["winner"] == final["home"] else "▼"
    print(f"  {final['home']:<18} vs {final['away']:<18} → 🏆 {icon} {final['winner']:<18}")
    print(f"\n{'=' * 72}")
    print(f"  🏆 CHAMPION: {results['champion']}")
    print(f"{'=' * 72}")


def monte_carlo_forced(
    predictions: pd.DataFrame,
    match_probs: dict,
    forced: dict[str, str] | None,
    n_sims: int = 10000,
) -> dict:
    rng = np.random.default_rng(42)
    r16_teams = set()
    for _, r in predictions.iterrows():
        r16_teams.add(r["home_team"])
        r16_teams.add(r["away_team"])

    champion_counts = {team: 0 for team in r16_teams}
    for _ in range(n_sims):
        result = simulate_bracket(predictions, match_probs, forced_r16_results=forced, rng=rng)
        champ = result["champion"]
        if champ in champion_counts:
            champion_counts[champ] += 1

    return {team: count / n_sims * 100 for team, count in
            sorted(champion_counts.items(), key=lambda x: -x[1])}


def print_mc_table(scenarios: dict[str, dict]) -> None:
    all_teams = sorted(set().union(*(set(s.keys()) for s in scenarios.values())))

    print(f"\n{'=' * 72}")
    print(f"  CHAMPION PROBABILITIES ACROSS SCENARIOS")
    print(f"  Each scenario: 10,000 Monte Carlo simulations")
    print(f"{'=' * 72}")

    headers = ["Team"] + list(scenarios.keys())
    print(f"\n  {'Team':<18}", end="")
    for h in headers[1:]:
        print(f"  {h:>16}", end="")
    print(f"  {'Max Δ':>7}")
    print(f"  {'─' * 18}" + "─" * 25 * len(headers[1:]) + "─" * 12)

    baseline = scenarios.get("Default", {})
    for team in all_teams:
        base_pct = baseline.get(team, 0)
        vals = [scenario.get(team, 0) for scenario in scenarios.values()]
        max_delta = max(abs(v - base_pct) for v in vals) if baseline else 0
        print(f"  {team:<18}", end="")
        for v in vals:
            bar = "█" * max(1, int(v / 3))
            print(f"  {v:>5.1f}% {bar:<10}", end="")
        print(f"  {max_delta:>+5.1f}%")
    print()


def main() -> int:
    t0 = time.time()
    predictions = load_predictions()
    match_probs = build_match_probs(predictions)
    N_SIMS = 10_000

    # Current Canada vs Morocco prediction
    can_row = predictions[predictions["home_team"] == "Canada"].iloc[0]
    print(f"\n{'#' * 72}")
    print(f"  #  WHAT-IF: CANADA vs MOROCCO (R16-1)")
    print(f"  #  Current prediction:")
    print(f"  #    Canada wins: {can_row['home_win_prob']:.1%}")
    print(f"  #    Draw:        {can_row['draw_prob']:.1%}")
    print(f"  #    Morocco wins: {can_row['away_win_prob']:.1%}")
    print(f"  #    Confidence: {can_row['home_win_prob']:.0%} / {can_row['away_win_prob']:.0%}")
    print(f"  #  (Morocco Elo penalty: -50)")
    print(f"{'#' * 72}")

    # Morocco's current bracket path (from default simulation)
    print(f"\n  Morocco's current projected path:")
    print(f"  R16:  Canada  → Morocco")
    print(f"  QF:   France  → Morocco (? SF-1)")
    print(f"  SF:   Brazil  → Morocco (? Final)")
    print(f"  Final: Portugal → Morocco (Champion)")

    # Show the current projected bracket path for Canada if they win
    print(f"\n  Canada's projected path if they win R16:")
    print(f"  R16:  Morocco → Canada")
    print(f"  QF:   France  → Canada (? SF-1)")
    print(f"  SF:   Brazil  → Canada (? Final)")
    print(f"  Final: Portugal → Canada (??)")

    # ── Scenarios ──
    scenarios = {
        "Default": None,
        "Canada wins": {"R16-1": "Canada"},
        "Morocco wins": {"R16-1": "Morocco"},
    }

    # ── Deterministic brackets ──
    det_results = {}
    for label, forced in scenarios.items():
        result = simulate_bracket(predictions, match_probs, forced_r16_results=forced)
        det_results[label] = result
        print_bracket(result, f"SCENARIO: {label}")

    # ── Monte Carlo ──
    print(f"\n  Running {N_SIMS:,} Monte Carlo simulations per scenario ...\n")
    mc_results = {}
    for label, forced in scenarios.items():
        mc_results[label] = monte_carlo_forced(predictions, match_probs, forced, n_sims=N_SIMS)

    print_mc_table(mc_results)

    # ── Key comparisons ──
    def_mc = mc_results["Default"]
    can_mc = mc_results["Canada wins"]
    mor_mc = mc_results["Morocco wins"]

    print(f"{'─' * 72}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'─' * 72}")

    can_def = det_results["Default"]["champion"]
    can_can = det_results["Canada wins"]["champion"]
    can_mor = det_results["Morocco wins"]["champion"]

    print(f"\n  Default champion:         {can_def}")
    print(f"  If Canada wins R16:      {can_can}")
    print(f"  If Morocco wins R16:     {can_mor}")
    print()

    # Probability swings
    print(f"  {'Team':<18} {'Default':>8} {'Canada wins':>14} {'Morocco wins':>15}")
    print(f"  {'─' * 18} {'─' * 8} {'─' * 14} {'─' * 15}")
    for team in list(def_mc.keys())[:10]:
        d = def_mc.get(team, 0)
        c = can_mc.get(team, 0)
        m = mor_mc.get(team, 0)
        dc = c - d
        dm = m - d
        c_str = f"{c:5.1f}% ({dc:+>+.1f})" if abs(dc) >= 0.3 else f"{c:5.1f}%"
        m_str = f"{m:5.1f}% ({dm:+>+.1f})" if abs(dm) >= 0.3 else f"{m:5.1f}%"
        print(f"  {team:<18} {d:>5.1f}%   {c_str:>14} {m_str:>15}")
    print()

    # Most affected teams
    print(f"  Biggest winner if Canada advances:")
    can_gainers = [(team, can_mc.get(team, 0) - def_mc.get(team, 0))
                   for team in def_mc if abs(can_mc.get(team, 0) - def_mc.get(team, 0)) > 0.5]
    can_gainers.sort(key=lambda x: -x[1])
    for team, delta in can_gainers[:5]:
        print(f"    {team}: {delta:+.1f}pp")

    print(f"\n  Biggest winner if Morocco advances:")
    mor_gainers = [(team, mor_mc.get(team, 0) - def_mc.get(team, 0))
                   for team in def_mc if abs(mor_mc.get(team, 0) - def_mc.get(team, 0)) > 0.5]
    mor_gainers.sort(key=lambda x: -x[1])
    for team, delta in mor_gainers[:5]:
        print(f"    {team}: {delta:+.1f}pp")

    collapser_mor = [(team, def_mc.get(team, 0) - mor_mc.get(team, 0))
                     for team in def_mc if def_mc.get(team, 0) - mor_mc.get(team, 0) > 0.5]
    collapser_mor.sort(key=lambda x: -x[1])
    print(f"\n  Biggest loser if Morocco advances (eliminated from Canada's path):")
    for team, delta in collapser_mor[:3]:
        print(f"    {team}: -{delta:.1f}pp")

    print(f"\n  Done in {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
