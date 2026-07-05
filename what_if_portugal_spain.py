"""
what_if_portugal_spain.py — What-If Analysis: Portugal vs Spain R16 Outcome Scenarios.

Forces each possible result of Portugal vs Spain (R16-5) and simulates the
full knockout bracket to show how the tournament tree changes.

Usage:
    python what_if_portugal_spain.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = PROJECT_ROOT / "reports" / "predictions_worldcup" / "worldcup_predictions.csv"

# ── Bracket structure (same as bracket_simulator.py) ──
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
        print("    Run:  python train_worldcup.py")
        sys.exit(1)
    df = pd.read_csv(PREDICTIONS_CSV)
    return df.sort_values("date").reset_index(drop=True)


def build_match_probs(predictions: pd.DataFrame) -> dict:
    """Build lookup dictionary for all known head-to-head probabilities."""
    match_probs = {}
    for _, row in predictions.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        hw = row["home_win_prob"]
        dr = row["draw_prob"]
        aw = row["away_win_prob"]
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

    # ── Round of 16 ──
    r16_results = {}
    for r16_id, h, a in R16_MATCHUPS:
        hw, dr, aw = match_probs.get((h, a), (1 / 3, 1 / 3, 1 / 3))
        probs = np.array([aw, dr, hw])
        probs = probs / probs.sum()

        # Check if this match has a forced outcome
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
            "home": h,
            "away": a,
            "winner": winner,
            "home_win": hw,
            "draw": dr,
            "away_win": aw,
            "confidence": max(hw, dr, aw),
        }

    results["rounds"]["Round of 16"] = r16_results

    # ── Quarter-Finals onward ──
    for round_name, pairings, next_fn in [
        ("Quarter-Finals", QF_WINNER_PAIRS, lambda qf, r16_a, r16_b: (
            r16_a["winner"], r16_b["winner"],
            qf.get((r16_a["winner"], r16_b["winner"]), None),
            qf.get((r16_b["winner"], r16_a["winner"]), None),
        )),
    ]:
        round_results = {}
        for match_id, r16_a_key, r16_b_key in pairings:
            r16_a = r16_results[r16_a_key]
            r16_b = r16_results[r16_b_key]
            h = r16_a["winner"]
            a = r16_b["winner"]

            if (h, a) in match_probs:
                hw, dr, aw = match_probs[(h, a)]
            else:
                edge = (r16_a["confidence"] - r16_b["confidence"]) * 0.15
                hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
                hw = np.clip(hw, 0.25, 0.55)
                aw = np.clip(aw, 0.20, 0.50)
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

            round_results[match_id] = {
                "home": h,
                "away": a,
                "winner": winner,
                "home_win": hw,
                "draw": dr,
                "away_win": aw,
                "confidence": max(hw, dr, aw),
            }

        results["rounds"][round_name] = round_results

    # ── Semi-Finals ──
    qf_results = results["rounds"]["Quarter-Finals"]
    sf_results = {}
    for sf_id, qf_a_key, qf_b_key in SF_PAIRS:
        h = qf_results[qf_a_key]["winner"]
        a = qf_results[qf_b_key]["winner"]

        if (h, a) in match_probs:
            hw, dr, aw = match_probs[(h, a)]
        else:
            prev_h = qf_results[qf_a_key]["confidence"]
            prev_a = qf_results[qf_b_key]["confidence"]
            edge = (prev_h - prev_a) * 0.15
            hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
            hw = np.clip(hw, 0.25, 0.55)
            aw = np.clip(aw, 0.20, 0.50)
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
            "home": h,
            "away": a,
            "winner": winner,
            "home_win": hw,
            "draw": dr,
            "away_win": aw,
            "confidence": max(hw, dr, aw),
        }

    results["rounds"]["Semi-Finals"] = sf_results

    # ── Final ──
    final_id, sf_a, sf_b = FINAL_MATCHUP
    h = sf_results[sf_a]["winner"]
    a = sf_results[sf_b]["winner"]

    if (h, a) in match_probs:
        hw, dr, aw = match_probs[(h, a)]
    else:
        prev_h = sf_results[sf_a]["confidence"]
        prev_a = sf_results[sf_b]["confidence"]
        edge = (prev_h - prev_a) * 0.15
        hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
        hw = np.clip(hw, 0.25, 0.55)
        aw = np.clip(aw, 0.20, 0.50)
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
        final_id: {
            "home": h,
            "away": a,
            "winner": winner,
            "home_win": hw,
            "draw": dr,
            "away_win": aw,
            "confidence": max(hw, dr, aw),
        }
    }

    results["champion"] = winner
    return results


def print_bracket(results: dict, label: str) -> None:
    """Pretty-print a bracket scenario."""
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    for round_name in ["Round of 16", "Quarter-Finals", "Semi-Finals"]:
        print(f"\n  ── {round_name} ──")
        for match_id, data in results["rounds"][round_name].items():
            icon = "🏠" if data["winner"] == data["home"] else "✈️"
            print(f"  {data['home']:<18} vs {data['away']:<18} → {icon} {data['winner']:<18}")

    final = results["rounds"]["Final"]["Final"]
    print(f"\n  ── FINAL ──")
    icon = "🏠" if final["winner"] == final["home"] else "✈️"
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
    """Run MC with a forced R16 result."""
    rng = np.random.default_rng(42)

    # Gather all R16 teams
    r16_teams = set()
    for _, r in predictions.iterrows():
        r16_teams.add(r["home_team"])
        r16_teams.add(r["away_team"])

    champion_counts: dict[str, int] = {team: 0 for team in r16_teams}

    for _ in range(n_sims):
        result = simulate_bracket(predictions, match_probs, forced_r16_results=forced, rng=rng)
        champ = result["champion"]
        if champ in champion_counts:
            champion_counts[champ] += 1

    return {team: count / n_sims * 100 for team, count in
            sorted(champion_counts.items(), key=lambda x: -x[1])}


def print_mc_table(scenarios: dict[str, dict]) -> None:
    """Print comparison table of champion probabilities across scenarios."""
    all_teams = sorted(set().union(*(set(s.keys()) for s in scenarios.values())))

    print(f"\n{'=' * 72}")
    print(f"  CHAMPION PROBABILITIES ACROSS SCENARIOS")
    print(f"  Each scenario: 10,000 Monte Carlo simulations with forced R16 outcome")
    print(f"{'=' * 72}")

    # Header
    headers = ["Team"] + list(scenarios.keys())
    print(f"\n  {'Team':<18}", end="")
    for h in headers[1:]:
        print(f"  {h:>15}", end="")
    print(f"  {'Max Δ':>8}")
    print(f"  {'─' * 18}" + "─" * 24 * len(headers[1:]) + "─" * 12)

    baseline = scenarios.get("Default", {})
    for team in all_teams:
        base_pct = baseline.get(team, 0)
        vals = [scenario.get(team, 0) for scenario in scenarios.values()]
        max_delta = max(abs(v - base_pct) for v in vals) if baseline else 0
        print(f"  {team:<18}", end="")
        for v in vals:
            bar = "█" * max(1, int(v / 3))
            print(f"  {v:>5.1f}% {bar:<8}", end="")
        print(f"  {max_delta:>+5.1f}%")

    print()


def main() -> int:
    t0 = time.time()
    predictions = load_predictions()
    match_probs = build_match_probs(predictions)

    N_SIMS = 10_000

    # ── Define scenarios ──
    scenarios = {
        "Default (Portugal)": None,  # normal simulation (no forced outcome)
        "Portugal wins": {"R16-5": "Portugal"},
        "Spain wins": {"R16-5": "Spain"},
    }

    # ── Run deterministic brackets ──
    print(f"\n{'#' * 72}")
    print(f"  #  WHAT-IF: PORTUGAL vs SPAIN — BRACKET SCENARIOS")
    print(f"  #  Current R16 prediction: Portugal 43.3% | Draw 19.8% | Spain 36.9%")
    print(f"  #  (Swap-averaged, confidence 43.3%)")
    print(f"{'#' * 72}")

    deterministic_results = {}
    for label, forced in scenarios.items():
        result = simulate_bracket(predictions, match_probs, forced_r16_results=forced)
        deterministic_results[label] = result
        print_bracket(result, label)

    # ── Run Monte Carlo for each scenario ──
    print(f"\n  Running {N_SIMS:,} Monte Carlo simulations per scenario ...\n")
    mc_results = {}
    for label, forced in scenarios.items():
        mc_results[label] = monte_carlo_forced(predictions, match_probs, forced, n_sims=N_SIMS)

    print_mc_table(mc_results)

    # ── Key comparisons ──
    def_mc = mc_results["Default (Portugal)"]
    por_mc = mc_results["Portugal wins"]
    spa_mc = mc_results["Spain wins"]

    print(f"{'─' * 72}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'─' * 72}")

    por_champ = deterministic_results["Portugal wins"]["champion"]
    spa_champ = deterministic_results["Spain wins"]["champion"]
    def_champ = deterministic_results["Default (Portugal)"]["champion"]

    print(f"\n  Default champion:       {def_champ}")
    print(f"  If Portugal wins R16:   {por_champ}")
    print(f"  If Spain wins R16:      {spa_champ}")
    print()

    # Biggest swing
    por_swing = por_mc.get("Spain", 0) - def_mc.get("Spain", 0)
    spa_swing = spa_mc.get("Portugal", 0) - def_mc.get("Portugal", 0)
    print(f"  Spain's probability drops by {abs(por_swing):.1f}pp if Portugal wins R16")
    print(f"  Portugal's probability drops by {abs(spa_swing):.1f}pp if Spain wins R16")
    print()

    # Top 5 comparison
    print(f"  {'Team':<18} {'Default':>10} {'Portugal wins':>15} {'Spain wins':>13}")
    print(f"  {'─' * 18} {'─' * 10} {'─' * 15} {'─' * 13}")
    for team in list(def_mc.keys())[:8]:
        d = def_mc.get(team, 0)
        p = por_mc.get(team, 0)
        s = spa_mc.get(team, 0)
        delta_p = p - d
        delta_s = s - d
        p_str = f"{p:5.1f}% ({delta_p:+>+.1f})" if abs(delta_p) > 0.3 else f"{p:5.1f}%"
        s_str = f"{s:5.1f}% ({delta_s:+>+.1f})" if abs(delta_s) > 0.3 else f"{s:5.1f}%"
        print(f"  {team:<18} {d:>5.1f}%     {p_str:>15} {s_str:>13}")

    print(f"\n  Done in {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
