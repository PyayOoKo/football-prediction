"""
bracket_simulator.py — Full World Cup 2026 Knockout Bracket Simulation.

Uses the trained XGBoost model and swap-averaged probabilities to simulate
the entire knockout tournament from Round of 16 through to the Final.

For each match, the winner is determined by the model's probabilities.
Uncertain matches are flagged with confidence levels.

Usage:
    python bracket_simulator.py
    python bracket_simulator.py --show-detail   # Show per-round simulations
    python bracket_simulator.py --monte-carlo N  # Run N simulations for probability
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = PROJECT_ROOT / "reports" / "predictions_worldcup" / "worldcup_predictions.csv"
DATA_CSV = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"


def load_predictions() -> pd.DataFrame:
    """Load the latest R16 predictions."""
    if not PREDICTIONS_CSV.exists():
        print(f"  [X] Predictions not found at {PREDICTIONS_CSV}")
        print("    Run:  python train_worldcup.py")
        sys.exit(1)
    df = pd.read_csv(PREDICTIONS_CSV)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── Bracket structure ──────────────────────────────────
# The 2026 World Cup knockout bracket follows a fixed path.
# R16 matches are paired into QF groups.

# ── Bracket structure ──────────────────────────────────
# Built dynamically from the 2026 match data to handle already-completed rounds.

def _load_actual_results() -> dict:
    """Load completed 2026 knockout results from the data file."""
    if not DATA_CSV.exists():
        return {}
    df = pd.read_csv(DATA_CSV, low_memory=False, parse_dates=['date'])
    wc26 = df[(df['season'] == 2026) & df['result'].notna()].copy()
    # Map R16 matchups: the data already has "Round of 16" in round col
    results = {}
    for _, r in wc26.iterrows():
        rnd = str(r.get('round', '')).strip()
        if 'Round of 16' in rnd:
            h, a, res = r['home_team'], r['away_team'], r['result']
            winner = h if res == 'H' else a if res == 'A' else None
            results[(h, a)] = winner
    return results

ACTUAL_R16 = _load_actual_results()

# Build R16 matchups: skip already-decided ones; those with unknown results stay
# Group by the fixed bracket path
_R16_ORDER = [
    ("Canada", "Morocco"), ("Paraguay", "France"),
    ("Brazil", "Norway"),  ("Mexico", "England"),
    ("Portugal", "Spain"), ("USA", "Belgium"),
    ("Argentina", "Egypt"), ("Switzerland", "Colombia"),
]

R16_MATCHUPS = []
r16_idx = 1
R16_WINNERS = {}
for h, a in _R16_ORDER:
    key = (h, a)
    rid = f"R16-{r16_idx}"
    if key in ACTUAL_R16:
        R16_WINNERS[rid] = ACTUAL_R16[key]
        # Still register the matchup so the bracket shows the result
        R16_MATCHUPS.append((rid, h, a))
    else:
        R16_MATCHUPS.append((rid, h, a))
    r16_idx += 1

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


def simulate_bracket(
    predictions: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate the full knockout bracket.

    Parameters
    ----------
    predictions : pd.DataFrame
        R16 match predictions with home_team, away_team, home_win_prob, etc.
    rng : np.random.Generator, optional
        Random number generator for Monte Carlo. If None, uses deterministic
        picks (highest probability wins).

    Returns
    -------
    dict
        Full bracket results with all rounds.
    """
    use_mc = rng is not None

    # Build lookup: (team1, team2) -> probabilities
    _EPS = 1e-12
    match_probs = {}
    for _, row in predictions.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        hw = row["home_win_prob"]
        dr = row["draw_prob"]
        aw = row["away_win_prob"]
        match_probs[(h, a)] = (hw, dr, aw)
        match_probs[(a, h)] = (aw, dr, hw)  # swapped

    results = {"rounds": {}}

    # ── Round of 16 ──
    r16_results = {}
    for r16_id, h, a in R16_MATCHUPS:
        # Use actual result if already played
        if r16_id in R16_WINNERS:
            winner = R16_WINNERS[r16_id]
            hw, dr, aw = (1.0, 0.0, 0.0) if winner == h else (0.0, 0.0, 1.0)
            r16_results[r16_id] = {
                "home": h, "away": a, "winner": winner,
                "home_win": hw, "draw": dr, "away_win": aw,
                "confidence": 1.0,
            }
            continue

        hw, dr, aw = match_probs.get((h, a), (1/3, 1/3, 1/3))
        probs = np.array([aw, dr, hw])
        probs = probs / probs.sum()

        if use_mc:
            winner_idx = rng.choice(3, p=probs)
        else:
            winner_idx = np.argmax(probs)

        if winner_idx == 2:
            winner = h
            loser = a
        elif winner_idx == 1:
            winner = h if hw >= aw else a
        else:
            winner = a
            loser = h

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

    # ── Quarter-Finals ──
    qf_results = {}
    for qf_id, r16_a_key, r16_b_key in QF_WINNER_PAIRS:
        r16_a = r16_results[r16_a_key]
        r16_b = r16_results[r16_b_key]

        # QF match: winner of R16-A vs winner of R16-B
        h = r16_a["winner"]
        a = r16_b["winner"]

        # Look up probabilities if model has seen this exact matchup
        if (h, a) in match_probs:
            hw, dr, aw = match_probs[(h, a)]
        else:
            # Estimate from round-based neutral: slightly favor the team
            # that won with higher previous-round confidence
            prev_win_conf = (r16_a["confidence"], r16_b["confidence"])
            edge = (prev_win_conf[0] - prev_win_conf[1]) * 0.15  # home edge = +
            hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
            hw = np.clip(hw, 0.25, 0.55)
            aw = np.clip(aw, 0.20, 0.50)
            total = hw + dr + aw
            hw, dr, aw = hw / total, dr / total, aw / total

        probs = np.array([aw, dr, hw])
        probs = probs / probs.sum()  # float-precision guard
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
            "home": h,
            "away": a,
            "winner": winner,
            "home_win": hw,
            "draw": dr,
            "away_win": aw,
            "confidence": max(hw, dr, aw),
        }

    results["rounds"]["Quarter-Finals"] = qf_results

    # ── Semi-Finals ──
    sf_results = {}
    for sf_id, qf_a_key, qf_b_key in SF_PAIRS:
        h = qf_results[qf_a_key]["winner"]
        a = qf_results[qf_b_key]["winner"]

        if (h, a) in match_probs:
            hw, dr, aw = match_probs[(h, a)]
        else:
            # Use QF confidence as a strength signal
            prev_h = qf_results[qf_a_key]["confidence"]
            prev_a = qf_results[qf_b_key]["confidence"]
            edge = (prev_h - prev_a) * 0.15
            hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
            hw = np.clip(hw, 0.25, 0.55)
            aw = np.clip(aw, 0.20, 0.50)
            total = hw + dr + aw
            hw, dr, aw = hw / total, dr / total, aw / total

        probs = np.array([aw, dr, hw])
        probs = probs / probs.sum()  # float-precision guard
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
        # Use semi-final confidence as strength signal
        prev_h = sf_results[sf_a]["confidence"]
        prev_a = sf_results[sf_b]["confidence"]
        edge = (prev_h - prev_a) * 0.15
        hw, dr, aw = 0.4 + edge, 0.25, 0.35 - edge
        hw = np.clip(hw, 0.25, 0.55)
        aw = np.clip(aw, 0.20, 0.50)
        total = hw + dr + aw
        hw, dr, aw = hw / total, dr / total, aw / total

    probs = np.array([aw, dr, hw])
    # Guard: ensure exact sum-to-1 for numpy's rng.choice
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


def print_bracket(results: dict, show_detail: bool = False) -> None:
    """Pretty-print the full knockout bracket."""
    print()
    print("=" * 72)
    print("  WORLD CUP 2026 — FULL KNOCKOUT BRACKET")
    print("=" * 72)

    # R16
    print(f"\n  {'-' * 28} ROUND OF 16 {'-' * 28}")
    r16 = results["rounds"]["Round of 16"]
    for r16_id, data in r16.items():
        winner_mark = "[H]" if data["winner"] == data["home"] else "[A]"
        print(f"  {data['home']:<20} vs {data['away']:<20} -> {winner_mark} {data['winner']:<20} ({data['confidence']:.0%})")

    # QF
    print(f"\n  {'-' * 26} QUARTER-FINALS {'-' * 26}")
    qf = results["rounds"]["Quarter-Finals"]
    for qf_id, data in qf.items():
        winner_mark = "[H]" if data["winner"] == data["home"] else "[A]"
        print(f"  {data['home']:<20} vs {data['away']:<20} -> {winner_mark} {data['winner']:<20} ({data['confidence']:.0%})")

    # SF
    print(f"\n  {'-' * 28} SEMI-FINALS {'-' * 28}")
    sf = results["rounds"]["Semi-Finals"]
    for sf_id, data in sf.items():
        winner_mark = "[H]" if data["winner"] == data["home"] else "[A]"
        print(f"  {data['home']:<20} vs {data['away']:<20} -> {winner_mark} {data['winner']:<20} ({data['confidence']:.0%})")

    # Final
    print(f"\n  {'-' * 30} FINAL {'-' * 30}")
    final = results["rounds"]["Final"]["Final"]
    print(f"  {final['home']:<20} vs {final['away']:<20} -> CHAMPION: {final['winner']:<20} ({final['confidence']:.0%})")

    print(f"\n{'=' * 72}")
    print(f"  CHAMPION: {results['champion']}")
    print(f"{'=' * 72}")
    print()


def monte_carlo_simulation(predictions: pd.DataFrame, n_sims: int = 10000) -> dict:
    """Run Monte Carlo simulation for each team's championship probability."""
    rng = np.random.default_rng(42)
    champion_counts: dict[str, int] = {}

    # Track which R16 teams exist
    r16_teams = set()
    for _, r in predictions.iterrows():
        r16_teams.add(r["home_team"])
        r16_teams.add(r["away_team"])

    for team in r16_teams:
        champion_counts[team] = 0

    for _ in range(n_sims):
        result = simulate_bracket(predictions, rng=rng)
        champ = result["champion"]
        if champ in champion_counts:
            champion_counts[champ] += 1

    # Convert to percentages
    probs = {team: count / n_sims * 100 for team, count in
             sorted(champion_counts.items(), key=lambda x: -x[1])}

    return probs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Knockout Bracket Simulation",
    )
    parser.add_argument("--show-detail", action="store_true", help="Show per-round details")
    parser.add_argument("--monte-carlo", type=int, default=0,
                        help="Run N Monte Carlo simulations for champion probabilities")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 72)
    print("  WORLD CUP 2026 — KNOCKOUT BRACKET SIMULATOR")
    print("=" * 72)

    predictions = load_predictions()

    # Deterministic bracket
    results = simulate_bracket(predictions)
    print_bracket(results, show_detail=args.show_detail)

    # Save bracket predictions
    output_dir = PROJECT_ROOT / "reports" / "predictions_worldcup"
    output_dir.mkdir(parents=True, exist_ok=True)

    bracket_rows = []
    for round_name, matches in results["rounds"].items():
        for match_id, data in matches.items():
            bracket_rows.append({
                "round": round_name,
                "match": match_id,
                "team1": data["home"],
                "team2": data["away"],
                "winner": data["winner"],
                "team1_win_prob": round(data["home_win"], 4),
                "draw_prob": round(data["draw"], 4),
                "team2_win_prob": round(data["away_win"], 4),
                "confidence": round(data["confidence"], 4),
            })

    bracket_df = pd.DataFrame(bracket_rows)
    bracket_path = output_dir / "bracket_prediction.csv"
    bracket_df.to_csv(bracket_path, index=False)
    print(f"  Bracket saved to {bracket_path}")

    # Monte Carlo simulation
    if args.monte_carlo > 0:
        print(f"\n  Running {args.monte_carlo:,} Monte Carlo simulations ...")
        mc_probs = monte_carlo_simulation(predictions, n_sims=args.monte_carlo)

        mc_path = output_dir / "monte_carlo_probs.csv"
        mc_df = pd.DataFrame([
            {"team": team, "champion_prob": round(prob, 2)}
            for team, prob in mc_probs.items() if prob > 0
        ])
        mc_df.to_csv(mc_path, index=False)

        print(f"\n  {'-' * 40}")
        print(f"  CHAMPION PROBABILITIES ({args.monte_carlo:,} simulations)")
        print(f"  {'-' * 40}")
        for _, row in mc_df.iterrows():
            bar = "#" * max(1, int(row["champion_prob"] / 2))
            print(f"  {row['team']:<20} {row['champion_prob']:>5.1f}% {bar}")
        print(f"  Saved to {mc_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
