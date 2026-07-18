#!/usr/bin/env python3
"""
bracket_simulator.py — World Cup 2026 Knockout Bracket Simulator.

Simulates the full knockout bracket (Round of 16 → Quarter-Finals →
Semi-Finals → Final) using model-generated match probabilities,
with optional Monte Carlo analysis for championship probabilities.

Usage
-----
    # Deterministic simulation (most likely winner)
    python bracket_simulator.py

    # Detailed per-round output
    python bracket_simulator.py --show-detail

    # Monte Carlo: run 10,000 simulations for championship probabilities
    python bracket_simulator.py --monte-carlo 10000

    # Use a specific model instead of CSV predictions
    python bracket_simulator.py --model models/worldcup_lightgbm.joblib

    # Use a specific predictions CSV
    python bracket_simulator.py --predictions reports/predictions_worldcup/worldcup_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ── Ensure project root is on sys.path ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import config


# ============================================================
#  CONSTANTS
# ============================================================

# World Cup 2026 Round of 16 matchups (from bracket structure)
# Format: (home_team, away_team) — home/away are nominal for neutral venues
R16_MATCHUPS: list[tuple[str, str]] = [
    ("Switzerland", "Portugal"),       # Match 49: 1A vs 2C
    ("Netherlands", "Iraq"),           # Match 50: 1B vs 2D
    ("Italy", "South Korea"),          # Match 51: 1C vs 2F  — adjusted for 2026 qualified teams
    ("Brazil", "Ivory Coast"),         # Match 52: 1D vs 2E
    ("England", "Norway"),             # Match 53: 1E vs 2B
    ("Morocco", "Mexico"),             # Match 54: 1F vs 2A — adjusted for 2026 qualified teams
    ("Germany", "Canada"),             # Match 55: 1G vs 2H
    ("Argentina", "USA"),              # Match 56: 1H vs 2G
]

# Knockout bracket structure as (match_id, home_team_key, away_team_key, round_name)
# Team keys reference previous match winners: f"W{prev_match_id}" or direct team name
BRACKET_ROUNDS: list[dict[str, Any]] = [
    # Round of 16 — direct matchups
    {"id": 49, "home": "Switzerland", "away": "Portugal",    "round": "Round of 16"},
    {"id": 50, "home": "Netherlands", "away": "Iraq",        "round": "Round of 16"},
    {"id": 51, "home": "Italy",       "away": "South Korea", "round": "Round of 16"},
    {"id": 52, "home": "Brazil",      "away": "Ivory Coast", "round": "Round of 16"},
    {"id": 53, "home": "England",     "away": "Norway",      "round": "Round of 16"},
    {"id": 54, "home": "Morocco",     "away": "Mexico",      "round": "Round of 16"},
    {"id": 55, "home": "Germany",     "away": "Canada",      "round": "Round of 16"},
    {"id": 56, "home": "Argentina",   "away": "USA",         "round": "Round of 16"},

    # Quarter-Finals
    {"id": 57, "home": "W49", "away": "W50", "round": "Quarter-Final"},
    {"id": 58, "home": "W51", "away": "W52", "round": "Quarter-Final"},
    {"id": 59, "home": "W53", "away": "W54", "round": "Quarter-Final"},
    {"id": 60, "home": "W55", "away": "W56", "round": "Quarter-Final"},

    # Semi-Finals
    {"id": 61, "home": "W57", "away": "W58", "round": "Semi-Final"},
    {"id": 62, "home": "W59", "away": "W60", "round": "Semi-Final"},

    # Third Place Play-off
    {"id": 63, "home": "L61", "away": "L62", "round": "Third Place Play-off"},

    # Final
    {"id": 64, "home": "W61", "away": "W62", "round": "Final"},
]

# Expected dates for knockout rounds (from 2026 schedule)
ROUND_DATES: dict[str, str] = {
    "Round of 16": "2026-07-11",
    "Quarter-Final": "2026-07-15",
    "Semi-Final": "2026-07-18",
    "Third Place Play-off": "2026-07-19",
    "Final": "2026-07-20",
}


# ============================================================
#  PATHS
# ============================================================

def _default_predictions_path() -> Path:
    """Default path to World Cup predictions CSV."""
    return _PROJECT_ROOT / config.worldcup.predictions_dir / config.worldcup.predictions_file


def _fallback_predictions_paths() -> list[Path]:
    """Try alternative prediction files in order."""
    pred_dir = _PROJECT_ROOT / config.worldcup.predictions_dir
    candidates = [
        pred_dir / "worldcup_predictions.csv",
        pred_dir / "worldcup_predictions_ensemble.csv",
        pred_dir / "latest_predictions.csv",
        pred_dir / "predictions_blend_20260717_174743.csv",
    ]
    return [p for p in candidates if p.exists()]


# ============================================================
#  PREDICTION LOADER
# ============================================================

def load_predictions_from_csv(
    csv_path: str | Path | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load match predictions from a CSV file.

    Returns a dict mapping ``(home_team, away_team)`` to
    ``{"home_win": float, "draw": float, "away_win": float, "confidence": float}``.
    """
    path = Path(csv_path) if csv_path else _default_predictions_path()

    if not path.exists():
        fallbacks = _fallback_predictions_paths()
        if fallbacks:
            path = fallbacks[0]
            print(f"  [ℹ] Using fallback predictions file: {path.name}")
        else:
            print(f"  [✗] Predictions file not found: {path}")
            print(f"  [ℹ] Run ``train_worldcup.py`` or ``today_value_bets_live.py`` first.")
            return {}

    predictions: dict[tuple[str, str], dict[str, float]] = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = (row.get("home_team") or row.get("home", "")).strip()
            away = (row.get("away_team") or row.get("away", "")).strip()
            if not home or not away:
                continue

            try:
                hw = float(row.get("home_win_prob", row.get("prob_home_win", 0.33)))
                dr = float(row.get("draw_prob", row.get("prob_draw", 0.33)))
                aw = float(row.get("away_win_prob", row.get("prob_away_win", 0.33)))
                conf = float(row.get("confidence", max(hw, dr, aw)))
            except (ValueError, TypeError):
                continue

            # Normalise to sum 1.0
            total = hw + dr + aw
            if total > 0:
                hw /= total
                dr /= total
                aw /= total

            predictions[(home, away)] = {
                "home_win": hw,
                "draw": dr,
                "away_win": aw,
                "confidence": conf,
            }

            # Also store the reverse matchup (for swap-averaging later)
            predictions[(away, home)] = {
                "home_win": aw,
                "draw": dr,
                "away_win": hw,
                "confidence": conf,
            }

    print(f"  [✓] Loaded {len(predictions) // 2} match predictions from {path.name}")
    return predictions


def load_predictions_from_model(
    model_path: str | Path | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load a trained model and generate predictions for all bracket matchups.

    Uses the project's ``PredictionEngine`` for model loading and prediction.
    """
    try:
        from src.prediction_engine import PredictionEngine
    except ImportError:
        print("  [✗] Could not import PredictionEngine. Falling back to CSV.")
        return {}

    engine = PredictionEngine(
        model_path=str(model_path) if model_path else None,
    )

    if not engine.model_loaded:
        print("  [✗] No model could be loaded. Falling back to CSV.")
        return {}

    print(f"  [✓] Loaded model: {engine.model_name} ({engine.model_type})")

    predictions: dict[tuple[str, str], dict[str, float]] = {}

    # Generate predictions for every R16 matchup
    for home, away in R16_MATCHUPS:
        try:
            probs = engine.predict_proba(home, away)
            predictions[(home, away)] = {
                "home_win": probs.get("home_win", 0.33),
                "draw": probs.get("draw", 0.34),
                "away_win": probs.get("away_win", 0.33),
                "confidence": max(probs.get("home_win", 0), probs.get("draw", 0), probs.get("away_win", 0)),
            }
            # Also store reverse for swap-averaging
            predictions[(away, home)] = {
                "home_win": probs.get("away_win", 0.33),
                "draw": probs.get("draw", 0.34),
                "away_win": probs.get("home_win", 0.33),
                "confidence": predictions[(home, away)]["confidence"],
            }
        except Exception as exc:
            print(f"  [⚠] Prediction failed for {home} vs {away}: {exc}")

    return predictions


def get_predictions(
    csv_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Get predictions from the best available source.

    Priority: 1) Explicit model path  2) Explicit CSV path
              3) Auto-detected model  4) Auto-detected CSV
    """
    # Try explicit model first
    if model_path:
        preds = load_predictions_from_model(model_path)
        if preds:
            return preds

    # Try explicit CSV
    if csv_path:
        preds = load_predictions_from_csv(csv_path)
        if preds:
            return preds

    # Try auto-detecting a saved model
    model_dir = _PROJECT_ROOT / "models"
    model_candidates = [
        model_dir / "worldcup_lightgbm.joblib",
        model_dir / "worldcup_xgboost.joblib",
        model_dir / "ensemble_model.joblib",
        model_dir / "worldcup_ensemble.joblib",
        model_dir / "xgboost_model.joblib",
    ]
    for mc in model_candidates:
        if mc.exists():
            print(f"  [ℹ] Auto-detected model: {mc.name}")
            preds = load_predictions_from_model(mc)
            if preds:
                return preds

    # Fall back to CSV
    preds = load_predictions_from_csv()
    if preds:
        return preds

    print("  [✗] No predictions or model found!")
    print("  [ℹ] Run ``train_worldcup.py`` or place predictions in:")
    print(f"      {_default_predictions_path()}")
    return {}


# ============================================================
#  SWAP-AVERAGED PROBABILITIES
# ============================================================

def get_swap_averaged_probs(
    home: str,
    away: str,
    predictions: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """Get swap-averaged probabilities for a matchup.

    Averages ``(home vs away)`` with ``(away vs home)`` to neutralise
    any home/away bias in the model — important for neutral-venue
    knockout matches.
    """
    forward = predictions.get((home, away))
    reverse = predictions.get((away, home))  # Already stored as reverse mapping

    if forward and reverse:
        # Swap-averaged: average the probabilities
        avg_home = (forward["home_win"] + reverse["away_win"]) / 2.0
        avg_draw = (forward["draw"] + reverse["draw"]) / 2.0
        avg_away = (forward["away_win"] + reverse["home_win"]) / 2.0
        avg_conf = (forward["confidence"] + reverse["confidence"]) / 2.0
        total = avg_home + avg_draw + avg_away
        if total > 0:
            return {
                "home_win": avg_home / total,
                "draw": avg_draw / total,
                "away_win": avg_away / total,
                "confidence": avg_conf,
            }

    # Fall back to forward only
    if forward:
        return forward

    # Neutral fallback
    return {"home_win": 0.34, "draw": 0.32, "away_win": 0.34, "confidence": 0.34}


# ============================================================
#  MATCH RESOLVER
# ============================================================

def resolve_team(
    team_key: str,
    match_winners: dict[int, str],
    match_losers: dict[int, str],
) -> str | None:
    """Resolve a team key to an actual team name.

    ``W49`` → winner of match 49, ``L61`` → loser of match 61.
    Returns ``None`` if the referenced match hasn't been played yet.
    """
    if team_key.startswith("W"):
        match_id = int(team_key[1:])
        return match_winners.get(match_id)
    if team_key.startswith("L"):
        match_id = int(team_key[1:])
        return match_losers.get(match_id)
    return team_key  # Direct team name


# ============================================================
#  SIMULATE A SINGLE MATCH
# ============================================================

def simulate_match(
    home: str,
    away: str,
    probs: dict[str, float],
    deterministic: bool = True,
    rng: random.Random | None = None,
) -> tuple[str, str, str]:
    """Simulate a single match and return ``(winner, loser, outcome)``.

    Parameters
    ----------
    home : str
        Home team name.
    away : str
        Away team name.
    probs : dict
        ``{"home_win": float, "draw": float, "away_win": float, ...}``.
    deterministic : bool
        If True, pick the most likely outcome. If False, sample
        from the probability distribution (Monte Carlo mode).
    rng : random.Random, optional
        Seeded RNG for reproducibility in Monte Carlo mode.

    Returns
    -------
    tuple[str, str, str]
        ``(winner, loser, outcome)`` where outcome is ``"H"``, ``"D"``, or ``"A"``.
        For draws, ``winner`` is the team that advances (penalty shootout winner
        — simulated proportionally).
    """
    hw = probs["home_win"]
    dr = probs["draw"]
    aw = probs["away_win"]
    total = hw + dr + aw
    if total <= 0:
        hw, dr, aw = 0.34, 0.32, 0.34
    else:
        hw /= total
        dr /= total
        aw /= total

    if deterministic:
        if hw >= dr and hw >= aw:
            outcome = "H"
        elif dr >= hw and dr >= aw:
            # Draw → simulate extra time / penalties
            # Higher-win-probability team is more likely to win on pens
            pen_hw = hw / (hw + aw) if (hw + aw) > 0 else 0.5
            if (rng or random).random() < pen_hw:
                outcome = "H"
            else:
                outcome = "A"
        else:
            outcome = "A"
    else:
        # Sample from the probability distribution
        rand = (rng or random).random()
        if rand < hw:
            outcome = "H"
        elif rand < hw + dr:
            # Draw → penalty shootout (proportional to strength)
            pen_hw = hw / (hw + aw) if (hw + aw) > 0 else 0.5
            if (rng or random).random() < pen_hw:
                outcome = "H"
            else:
                outcome = "A"
        else:
            outcome = "A"

    if outcome == "H":
        return home, away, "H"
    elif outcome == "A":
        return away, home, "A"
    else:
        return home, away, "D"


# ============================================================
#  SIMULATE THE FULL BRACKET
# ============================================================

def simulate_bracket(
    predictions: dict[tuple[str, str], dict[str, float]],
    deterministic: bool = True,
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[
    dict[int, str],         # match_winners: match_id -> team_name
    dict[int, str],         # match_losers: match_id -> team_name
    dict[int, dict],        # match_details: match_id -> details
]:
    """Simulate the full World Cup 2026 knockout bracket.

    Parameters
    ----------
    predictions : dict
        Mapping of ``(home, away)`` to probability dicts.
    deterministic : bool
        If True, most-likely outcomes. If False, sampled (MC mode).
    verbose : bool
        Print progress per match.
    seed : int, optional
        Seed for the RNG (Monte Carlo mode).

    Returns
    -------
    tuple
        ``(match_winners, match_losers, match_details)``
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    match_winners: dict[int, str] = {}
    match_losers: dict[int, str] = {}
    match_details: dict[int, dict] = {}

    for match in BRACKET_ROUNDS:
        mid = match["id"]
        round_name = match["round"]

        # Resolve home/away team keys
        home_raw = match["home"]
        away_raw = match["away"]

        home = resolve_team(home_raw, match_winners, match_losers)
        away = resolve_team(away_raw, match_winners, match_losers)

        if home is None or away is None:
            if verbose:
                print(f"  [⚠] Match {mid} ({round_name}) — cannot resolve teams: {home_raw} vs {away_raw}")
            continue

        # Get swap-averaged probabilities
        probs = get_swap_averaged_probs(home, away, predictions)

        # Simulate
        winner, loser, outcome = simulate_match(home, away, probs, deterministic, rng)

        match_winners[mid] = winner
        match_losers[mid] = loser
        match_details[mid] = {
            "home": home,
            "away": away,
            "winner": winner,
            "loser": loser,
            "outcome": outcome,
            "round": round_name,
            "probs": {
                "home_win": round(probs["home_win"], 4),
                "draw": round(probs["draw"], 4),
                "away_win": round(probs["away_win"], 4),
            },
            "confidence": round(probs["confidence"], 4),
        }

        if verbose:
            date = ROUND_DATES.get(round_name, "TBD")
            hw_pct = f"{probs['home_win']:.1%}"
            dr_pct = f"{probs['draw']:.1%}"
            aw_pct = f"{probs['away_win']:.1%}"
            conf = f"{probs['confidence']:.1%}"
            score = ("⬆" if outcome == "H" else "▶" if outcome == "D" else "⬇")
            print(
                f"  {date}  | {home:.<20s} {hw_pct:>7s}  "
                f"{dr_pct:>7s}  {aw_pct:>7s}  {conf:>7s}  "
                f"→  {winner:<20s}  {score}"
            )

    return match_winners, match_losers, match_details


# ============================================================
#  MONTE CARLO SIMULATION
# ============================================================

def run_monte_carlo(
    predictions: dict[tuple[str, str], dict[str, float]],
    n_simulations: int = 10000,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run multiple bracket simulations to compute championship probabilities.

    Parameters
    ----------
    predictions : dict
        Match prediction probabilities.
    n_simulations : int
        Number of simulations to run (default 10000).
    verbose : bool
        Print progress every 1000 simulations.

    Returns
    -------
    dict
        ``{"champion_probs": {team: probability}, "finalist_probs": {team: probability},
        "semifinal_probs": {team: probability}, "avg_winner_probs": {team: float},
        "n_simulations": int}``
    """
    champion_counter: Counter[str] = Counter()
    finalist_counter: Counter[str] = Counter()
    semifinal_counter: Counter[str] = Counter()
    quarterfinal_counter: Counter[str] = Counter()

    t_start = time.perf_counter()

    for i in range(n_simulations):
        if verbose and (i + 1) % max(1, n_simulations // 10) == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_simulations - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:>{len(str(n_simulations))}}/{n_simulations}] "
                f"{rate:.0f} sim/s | ETA: {remaining:.0f}s"
            )

        winners, losers, details = simulate_bracket(
            predictions, deterministic=False, verbose=False, seed=i,
        )

        # The final winner (match 64) is the champion
        champion = winners.get(64)
        if champion:
            champion_counter[champion] += 1

        # Finalist (match 64 loser)
        finalist = losers.get(64)
        if finalist:
            finalist_counter[finalist] += 1

        # Semi-finalists (match 61 and 62 participants)
        for mid in [61, 62]:
            winner = winners.get(mid)
            loser = losers.get(mid)
            if winner:
                semifinal_counter[winner] += 1
            if loser:
                semifinal_counter[loser] += 1

        # Quarter-finalists (match 57-60 participants)
        for mid in range(57, 61):
            winner = winners.get(mid)
            loser = losers.get(mid)
            if winner:
                quarterfinal_counter[winner] += 1
            if loser:
                quarterfinal_counter[loser] += 1

    elapsed = time.perf_counter() - t_start

    # Compute probabilities
    all_champion_teams = sorted(champion_counter.keys())
    champion_probs = {
        team: champion_counter[team] / n_simulations
        for team in all_champion_teams
    }
    finalist_probs = {
        team: finalist_counter[team] / n_simulations
        for team in sorted(finalist_counter.keys())
    }
    semifinal_probs = {
        team: semifinal_counter[team] / n_simulations
        for team in sorted(semifinal_counter.keys())
    }
    quarterfinal_probs = {
        team: quarterfinal_counter[team] / n_simulations
        for team in sorted(quarterfinal_counter.keys())
    }

    return {
        "champion_probs": champion_probs,
        "finalist_probs": finalist_probs,
        "semifinal_probs": semifinal_probs,
        "quarterfinal_probs": quarterfinal_probs,
        "n_simulations": n_simulations,
        "elapsed_seconds": elapsed,
    }


# ============================================================
#  OUTPUT FORMATTING
# ============================================================

def print_header(title: str, char: str = "═", width: int = 74) -> None:
    """Print a decorative section header."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def print_bracket_results(
    match_winners: dict[int, str],
    match_losers: dict[int, str],
    match_details: dict[int, dict],
) -> None:
    """Print a formatted bracket summary grouped by round."""
    rounds = ["Round of 16", "Quarter-Final", "Semi-Final", "Third Place Play-off", "Final"]

    for round_name in rounds:
        round_matches = {k: v for k, v in match_details.items() if v["round"] == round_name}
        if not round_matches:
            continue

        date = ROUND_DATES.get(round_name, "")
        print_header(f"{round_name}  ({date})", char="─")

        if round_name == "Final":
            print(f"\n  🏆  {match_winners.get(64, 'TBD'):^30s}  🏆")
            print(f"  vs  {match_losers.get(64, 'TBD'):^30s}")
            print()
            print(f"  Third Place:  {match_winners.get(63, 'TBD'):>20s}")
            print(f"  Fourth Place: {match_losers.get(63, 'TBD'):>20s}")
            continue

        for mid in sorted(round_matches.keys()):
            info = round_matches[mid]
            hw = f"{info['probs']['home_win']:.1%}"
            dr = f"{info['probs']['draw']:.1%}"
            aw = f"{info['probs']['away_win']:.1%}"
            conf = f"{info['confidence']:.1%}"
            score = ("⬆" if info["outcome"] == "H" else "▶" if info["outcome"] == "D" else "⬇")

            print(
                f"  Match {mid:<2d} | {info['home']:<20s} {hw:>7s}  "
                f"{dr:>7s}  {aw:>7s}  {conf:>7s}  "
                f"→  {info['winner']:<20s}  {score}"
            )
            if info["outcome"] == "D":
                print(f"           {'↳ Advanced on penalties':>58s}")


def print_deterministic_summary(
    match_winners: dict[int, str],
    match_losers: dict[int, str],
    match_details: dict[int, dict],
) -> None:
    """Print the full deterministic bracket results."""
    print_header("WORLD CUP 2026 KNOCKOUT BRACKET — DETERMINISTIC SIMULATION")
    print(f"\n  Method: Most likely outcome per match (swap-averaged probabilities)")
    print_bracket_results(match_winners, match_losers, match_details)


def print_monte_carlo_results(mc_results: dict[str, Any]) -> None:
    """Print Monte Carlo championship probabilities."""
    results = mc_results

    print_header(
        f"MONTE CARLO SIMULATION ({results['n_simulations']:,} simulations, "
        f"{results['elapsed_seconds']:.1f}s)", char="═",
    )
    print()

    champion_probs = results["champion_probs"]
    finalist_probs = results["finalist_probs"]
    semifinal_probs = results["semifinal_probs"]

    # ── Championship probabilities (bar chart) ──
    print("  ┌──────────────────────────────────────────────────────────────────────────┐")
    print("  │  CHAMPIONSHIP PROBABILITIES                                              │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")

    sorted_teams = sorted(champion_probs.items(), key=lambda x: -x[1])
    max_bar = 40
    for team, prob in sorted_teams:
        if prob < 0.005:
            continue
        bar_len = max(1, int(prob * max_bar * 2))
        bar = "█" * min(bar_len, max_bar)
        pct = f"{prob:.1%}"
        print(f"  │  {team:<22s} {bar:<{max_bar}s}  {pct:>6s}  │")

    # ── Legend ──
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print("  │  TOP SEMI-FINALISTS & FINALISTS                                          │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print(f"  │  {'Team':<22s} {'Champion':>10s} {'Finalist':>10s} {'Semi-Final':>11s}  │")
    print(f"  │  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 11}  │")

    all_teams = set(list(champion_probs.keys()) + list(semifinal_probs.keys()))
    for team in sorted(all_teams, key=lambda t: -champion_probs.get(t, 0)):
        champ = champion_probs.get(team, 0)
        final = finalist_probs.get(team, 0) + champ  # Finalist = runner-up + champion
        semi = semifinal_probs.get(team, 0)
        if semi < 0.01 and champ < 0.01:
            continue
        print(f"  │  {team:<22s} {champ:>9.1%}  {final:>9.1%}  {semi:>10.1%}  │")

    print("  └──────────────────────────────────────────────────────────────────────────┘")
    print()

    # ── Full probability table ──
    print("  Full probabilities:")
    print(f"  {'Team':<22s} {'Champion':>10s} {'Finalist':>10s} {'Semi-Final':>11s} {'Quarter-Final':>14s}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 10} {'─' * 11} {'─' * 14}")

    for team in sorted(all_teams, key=lambda t: -champion_probs.get(t, 0)):
        champ = champion_probs.get(team, 0)
        final = finalist_probs.get(team, 0) + champ
        semi = semifinal_probs.get(team, 0)
        qf = results.get("quarterfinal_probs", {}).get(team, 0)
        if champ < 0.001 and semi < 0.001:
            continue
        print(f"  {team:<22s} {champ:>9.1%}  {final:>9.1%}  {semi:>10.1%}  {qf:>13.1%}  ")

    # ── Favourite matchups ──
    print()
    print("  Most likely Final:")
    sorted_finalists = sorted(finalist_probs.items(), key=lambda x: -x[1])
    top_finalists = [t for t, _ in sorted_finalists[:4]]
    if len(top_finalists) >= 2:
        print(f"    {top_finalists[0]:>22s} vs {top_finalists[1]:<22s}")
    if len(top_finalists) >= 4:
        print(f"    {top_finalists[2]:>22s} vs {top_finalists[3]:<22s}")


def save_bracket_results(
    match_winners: dict[int, str],
    match_losers: dict[int, str],
    match_details: dict[int, dict],
    output_dir: str | Path = "",
) -> str:
    """Save bracket results to a CSV file.

    Returns the path to the saved file.
    """
    if not output_dir:
        output_dir = _PROJECT_ROOT / config.worldcup.predictions_dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "bracket_prediction.csv"

    rows: list[dict[str, Any]] = []
    for mid in sorted(match_details.keys()):
        info = match_details[mid]
        rows.append({
            "match_id": mid,
            "round": info["round"],
            "home_team": info["home"],
            "away_team": info["away"],
            "winner": info["winner"],
            "loser": info["loser"],
            "outcome": info["outcome"],
            "home_win_prob": info["probs"]["home_win"],
            "draw_prob": info["probs"]["draw"],
            "away_win_prob": info["probs"]["away_win"],
            "confidence": info["confidence"],
            "date": ROUND_DATES.get(info["round"], ""),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Bracket results saved to: {out_path}")
    return str(out_path)


def save_monte_carlo_results(
    mc_results: dict[str, Any],
    output_dir: str | Path = "",
) -> str:
    """Save Monte Carlo simulation results to a CSV file.

    Returns the path to the saved file.
    """
    if not output_dir:
        output_dir = _PROJECT_ROOT / config.worldcup.predictions_dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "monte_carlo_probs.csv"

    champion_probs = mc_results["champion_probs"]
    finalist_probs = mc_results["finalist_probs"]
    semifinal_probs = mc_results["semifinal_probs"]
    quarterfinal_probs = mc_results.get("quarterfinal_probs", {})

    all_teams = sorted(
        set(list(champion_probs.keys()) + list(semifinal_probs.keys())),
        key=lambda t: -champion_probs.get(t, 0),
    )

    rows: list[dict[str, Any]] = []
    for team in all_teams:
        rows.append({
            "team": team,
            "champion_prob": round(champion_probs.get(team, 0), 6),
            "finalist_prob": round(finalist_probs.get(team, 0) + champion_probs.get(team, 0), 6),
            "semifinal_prob": round(semifinal_probs.get(team, 0), 6),
            "quarterfinal_prob": round(quarterfinal_probs.get(team, 0), 6),
            "simulations": mc_results["n_simulations"],
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Monte Carlo results saved to: {out_path}")
    return str(out_path)


# ============================================================
#  MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Knockout Bracket Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python bracket_simulator.py\n"
            "  python bracket_simulator.py --monte-carlo 10000\n"
            "  python bracket_simulator.py --show-detail\n"
            "  python bracket_simulator.py --model models/worldcup_lightgbm.joblib\n"
        ),
    )
    parser.add_argument(
        "--predictions", "-p", type=str, default=None,
        help="Path to predictions CSV (default: auto-detect)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path to trained model file (overrides CSV)",
    )
    parser.add_argument(
        "--monte-carlo", "-mc", type=int, default=None,
        help="Run N Monte Carlo simulations for championship probabilities",
    )
    parser.add_argument(
        "--show-detail", "-d", action="store_true",
        help="Show detailed per-match probabilities during simulation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save results to CSV (default: True)",
    )

    # Handle no-arg mode elegantly
    if len(sys.argv) == 1:
        print()
        print("  ╔══════════════════════════════════════════════════════════════════════════╗")
        print("  ║           WORLD CUP 2026 — KNOCKOUT BRACKET SIMULATOR                   ║")
        print("  ╚══════════════════════════════════════════════════════════════════════════╝")
        print()

    args = parser.parse_args()

    # ── 1. Load predictions ──
    print()
    print_header("STEP 1/3: LOADING PREDICTIONS", char="─")
    predictions = get_predictions(
        csv_path=args.predictions,
        model_path=args.model,
    )
    if not predictions:
        print("\n  [✗] No predictions available. Exiting.")
        return 1
    print(f"  [✓] {len(predictions) // 2} unique matchups loaded")

    # ── 2. Run simulation ──
    if args.monte_carlo is not None:
        # Monte Carlo mode
        print_header(f"STEP 2/3: MONTE CARLO SIMULATION ({args.monte_carlo:,} simulations)", char="─")
        print(f"  Using seed base: {args.seed}")

        mc_results = run_monte_carlo(
            predictions,
            n_simulations=args.monte_carlo,
            verbose=True,
        )

        # ── 3. Results ──
        print_header("STEP 3/3: RESULTS", char="─")
        print_monte_carlo_results(mc_results)

        # Save
        if args.save:
            save_monte_carlo_results(mc_results)

    else:
        # Deterministic mode
        print_header("STEP 2/3: BRACKET SIMULATION", char="─")
        print("  Using swap-averaged probabilities (deterministic mode)\n")

        winners, losers, details = simulate_bracket(
            predictions,
            deterministic=True,
            verbose=args.show_detail,
            seed=args.seed,
        )

        # ── 3. Results ──
        print_header("STEP 3/3: RESULTS", char="─")
        print_deterministic_summary(winners, losers, details)

        # Save
        if args.save:
            save_bracket_results(winners, losers, details)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
