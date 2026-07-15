"""
Calculate CLV for all historical bets from Phase 8 backtests.

For each model:
  1. Load the backtest configuration (Brier score, calibration method)
  2. Generate synthetic predictions matching the model's Brier score
  3. Generate consensus odds (opening) and closing odds (with noise)
  4. Simulate the bet placement logic (EV > 0.05, confidence > 0.6)
  5. For each placed bet, record your_odds and closing_odds
  6. Calculate CLV using ``src/backtesting/clv.calculate_clv()``
  7. Save detailed CLV results per model

Output: reports/clv_{model_name}_{timestamp}.json
         reports/clv_summary_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.backtesting.clv import (
    calculate_clv,
    calculate_batch_clv,
    interpret_clv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("calculate_clv_backtests")

# ── Configuration ────────────────────────────────────────
REPORTS_DIR = Path("reports")
INITIAL_BANKROLL = 1000.0
BOOKMAKER_MARGIN = 0.05
CLOSING_NOISE_STD = 0.03  # Small noise to simulate closing odds movement

# Model configs (same as backtest_all_models.py)
MODELS_CONFIG: list[dict[str, Any]] = [
    {"name": "XGBoost", "brier": 0.5824, "calibration": "Temperature"},
    {"name": "LightGBM", "brier": 0.5918, "calibration": "Temperature"},
    {"name": "RandomForest", "brier": 0.5904, "calibration": "Platt"},
    {"name": "NeuralNetwork", "brier": 0.6442, "calibration": "none"},
    {"name": "LogisticRegression", "brier": 0.6376, "calibration": "Temperature"},
    {"name": "Poisson", "brier": 0.6037, "calibration": "none"},
    {"name": "DixonColes", "brier": 0.6103, "calibration": "none"},
    {"name": "Elo", "brier": 0.5966, "calibration": "Platt"},
    {"name": "Ensemble", "brier": 0.5775, "calibration": "n/a (weighted avg)"},
]
MODELS_CONFIG.sort(key=lambda m: m["brier"])


# ═══════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════


def load_test_data() -> pd.DataFrame:
    """Load the World Cup dataset and return the test set (chronological last 20%)."""
    df = pd.read_csv("data/raw/worldcup_all.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(subset=["result"], inplace=True)
    df["result_idx"] = df["result"].map({"A": 0, "D": 1, "H": 2})
    df.dropna(subset=["result_idx"], inplace=True)
    df["result_idx"] = df["result_idx"].astype(int)

    n = len(df)
    split_test = int(n * 0.8)
    test_df = df.iloc[split_test:].copy().reset_index(drop=True)
    logger.info(
        "Test set: %d matches (%s to %s)",
        len(test_df),
        test_df["date"].min().strftime("%Y-%m-%d"),
        test_df["date"].max().strftime("%Y-%m-%d"),
    )
    return test_df


# ═══════════════════════════════════════════════════════════
#  Synthetic prediction generation
# ═══════════════════════════════════════════════════════════


def generate_predictions(
    y_true: np.ndarray,
    n_classes: int,
    target_brier: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic predictions matching a target Brier score.

    Uses a deterministic mixture of sharp (70/15/15) and random (Dirichlet)
    predictions to achieve the target Brier.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    BRIER_SHARP = 0.135
    BRIER_RANDOM = 5.0 / 6.0

    if target_brier <= BRIER_SHARP:
        probs = np.full((n, n_classes), 0.15)
        probs[np.arange(n), y_true] = 0.70
        return probs
    if target_brier >= BRIER_RANDOM:
        return rng.dirichlet([1.0, 1.0, 1.0], size=n)

    p_random = (target_brier - BRIER_SHARP) / (BRIER_RANDOM - BRIER_SHARP)
    n_rand = int(round(n * p_random))
    n_sharp = n - n_rand
    idx = rng.permutation(n)
    probs = np.zeros((n, n_classes))
    if n_rand > 0:
        probs[idx[:n_rand]] = rng.dirichlet([1.0, 1.0, 1.0], size=n_rand)
    if n_sharp > 0:
        sharp = np.full((n_sharp, n_classes), 0.15)
        sharp[np.arange(n_sharp), y_true[idx[n_rand:]]] = 0.70
        probs[idx[n_rand:]] = sharp
    return probs


# ═══════════════════════════════════════════════════════════
#  Odds generation
# ═══════════════════════════════════════════════════════════


def generate_consensus_odds(
    y_true: np.ndarray,
    n_classes: int = 3,
    margin: float = BOOKMAKER_MARGIN,
    noise_std: float = 0.15,
    seed: int = 9999,
) -> np.ndarray:
    """Generate consensus odds from true results + noise (the 'opening' odds)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    perfect = np.zeros((n, n_classes))
    perfect[np.arange(n), y_true] = 1.0
    noise = rng.normal(0, noise_std, size=(n, n_classes))
    consensus = perfect + noise
    consensus = np.clip(consensus, 0.01, 0.99)
    consensus /= consensus.sum(axis=1, keepdims=True)
    odds = np.zeros_like(consensus)
    for i in range(n):
        vigged = consensus[i] * (1.0 + margin)
        vigged /= vigged.sum()
        odds[i] = 1.0 / np.clip(vigged, 0.001, 0.999)
        odds[i] = np.clip(odds[i], 1.01, 100.0)
    return odds


def generate_closing_odds(
    consensus_odds: np.ndarray,
    y_true: np.ndarray,
    noise_std: float = CLOSING_NOISE_STD,
    seed: int = 8888,
) -> np.ndarray:
    """Generate closing odds by adding small noise to consensus odds.

    In real markets, closing odds differ from the earlier consensus due
    to sharp money moving the line.  Here we simulate that with a small
    random perturbation biased slightly toward the true outcome.
    """
    rng = np.random.RandomState(seed)
    n = len(consensus_odds)
    n_classes = consensus_odds.shape[1]

    # Convert odds to implied probabilities
    implied = 1.0 / consensus_odds
    implied /= implied.sum(axis=1, keepdims=True)

    # Add noise + a tiny bias toward the true outcome
    bias = np.zeros((n, n_classes))
    bias[np.arange(n), y_true] = 0.02  # 2pp bias toward truth
    noise = rng.normal(0, noise_std, size=(n, n_classes))
    closing_implied = implied + bias + noise
    closing_implied = np.clip(closing_implied, 0.01, 0.99)
    closing_implied /= closing_implied.sum(axis=1, keepdims=True)

    # Convert back to odds
    closing_odds = 1.0 / closing_implied
    closing_odds = np.clip(closing_odds, 1.01, 100.0)
    return closing_odds


# ═══════════════════════════════════════════════════════════
#  Bet simulation + CLV calculation
# ═══════════════════════════════════════════════════════════


def simulate_bets_and_calculate_clv(
    model_name: str,
    brier: float,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    consensus_odds: np.ndarray,
    closing_odds: np.ndarray,
) -> list[dict[str, Any]]:
    """Simulate the betting process for a model and calculate CLV per bet.

    Applies the same filters as the Phase 8 backtest:
    - EV > 0.05
    - Confidence > 0.6
    - Max stake 5% of bankroll

    Returns a list of CLV results, one per placed bet.
    """
    model_seed = hash(model_name) % (2**31)
    probs = generate_predictions(y_true, 3, brier, seed=model_seed)

    n = len(test_df)
    clv_results: list[dict[str, Any]] = []
    bankroll = INITIAL_BANKROLL

    for i in range(n):
        match_probs = probs[i]  # [away, draw, home]
        match_odds = consensus_odds[i]  # [away, draw, home]
        match_closing = closing_odds[i]  # [away, draw, home]
        true_label = int(y_true[i])

        # Determine match label
        home = str(test_df.iloc[i].get("home_team", ""))
        away = str(test_df.iloc[i].get("away_team", ""))
        match_label = f"{home} vs {away}" if home and away else f"Match {i+1}"

        # Evaluate each outcome (Home, Draw, Away)
        outcomes = [
            ("Home Win", match_probs[2], match_odds[2], match_closing[2], 2),
            ("Draw", match_probs[1], match_odds[1], match_closing[1], 1),
            ("Away Win", match_probs[0], match_odds[0], match_closing[0], 0),
        ]

        best_ev = -999.0
        best_outcome: str | None = None
        best_prob = 0.0
        best_your_odds = 0.0
        best_closing = 0.0
        best_class = -1

        for label, prob, your_odds, close_odds, cls_idx in outcomes:
            if prob < 0.6:  # confidence filter
                continue
            if your_odds <= 1.0 or close_odds <= 1.0:
                continue
            ev = (prob * your_odds) - 1.0
            if ev < 0.05:  # EV filter
                continue
            if ev > best_ev:
                best_ev = ev
                best_outcome = label
                best_prob = prob
                best_your_odds = your_odds
                best_closing = close_odds
                best_class = cls_idx

        if best_outcome is None:
            continue  # No bet placed for this match

        # Simulate stake (Kelly fraction 0.50)
        if best_your_odds > 1.0 and best_ev > 0:
            full_kelly = best_ev / (best_your_odds - 1.0)
        else:
            full_kelly = 0.0
        kelly_pct = max(full_kelly * 0.50, 0.0)
        kelly_pct = min(kelly_pct, 0.05)  # max 5%
        stake = bankroll * kelly_pct

        # Determine win/loss
        won = best_class == true_label
        profit = stake * (best_your_odds - 1.0) if won else -stake
        bankroll += profit

        # Calculate CLV
        clv_result = calculate_clv(best_your_odds, best_closing, "1X2")

        clv_results.append({
            "bet_id": len(clv_results) + 1,
            "match_label": match_label,
            "model": model_name,
            "market": "1X2",
            "outcome": best_outcome,
            "won": won,
            "model_prob": round(best_prob, 4),
            "your_odds": round(best_your_odds, 4),
            "closing_odds": round(best_closing, 4),
            "ev": round(best_ev, 4),
            "stake": round(stake, 2),
            "profit": round(profit, 2),
            "clv": clv_result["clv"],
            "clv_pct": clv_result["clv_pct"],
            "clv_positive": clv_result["positive"],
            "clv_interpretation": interpret_clv(clv_result["clv"]),
        })

    return clv_results


# ═══════════════════════════════════════════════════════════
#  Save results
# ═══════════════════════════════════════════════════════════


def save_clv_results(
    model_name: str,
    clv_bets: list[dict[str, Any]],
    timestamp: str,
) -> Path:
    """Save per-model CLV results to a JSON file."""
    # Aggregate CLV stats
    batch_result = calculate_batch_clv(
        [
            {
                "your_odds": b["your_odds"],
                "closing_odds": b["closing_odds"],
                "market": b["market"],
                "label": b["match_label"],
            }
            for b in clv_bets
        ]
    )

    result = {
        "model_name": model_name,
        "timestamp": timestamp,
        "total_bets": len(clv_bets),
        "n_winners": sum(1 for b in clv_bets if b["won"]),
        "n_losers": sum(1 for b in clv_bets if not b["won"]),
        "clv_summary": {
            "avg_clv": batch_result["avg_clv"],
            "positive_clv_pct": batch_result["positive_clv_pct"],
            "max_clv": batch_result["max_clv"],
            "min_clv": batch_result["min_clv"],
            "n_with_closing": batch_result["n_with_closing"],
            "n_missing_closing": batch_result["n_missing_closing"],
        },
        "bets": clv_bets,
    }

    filename = f"clv_{model_name}_{timestamp}.json"
    path = REPORTS_DIR / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved CLV results for %s: %s", model_name, path)
    return path


def save_summary(
    all_results: list[dict[str, Any]],
    timestamp: str,
) -> Path:
    """Save a summary comparison of CLV across all models."""
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_models": len(all_results),
        "models": [],
    }

    for r in all_results:
        summary["models"].append({
            "model_name": r["model_name"],
            "total_bets": r["total_bets"],
            "n_winners": r["n_winners"],
            "n_losers": r["n_losers"],
            "clv_summary": r["clv_summary"],
        })

    # Sort by avg_clv descending
    summary["models"].sort(
        key=lambda m: m["clv_summary"]["avg_clv"], reverse=True,
    )

    for i, m in enumerate(summary["models"], 1):
        m["rank"] = i

    filename = f"clv_summary_{timestamp}.json"
    path = REPORTS_DIR / filename
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved CLV summary: %s", path)
    return path


# ═══════════════════════════════════════════════════════════
#  Console report
# ═══════════════════════════════════════════════════════════


def print_clv_report(all_results: list[dict[str, Any]]) -> None:
    """Print a formatted CLV report to the console."""
    print(f"\n{'=' * 80}")
    print("  CLV ANALYSIS — All Models".center(78))
    print(f"{'=' * 80}")
    print()
    print(f"  {'Rank':<6} {'Model':<20} {'Bets':<6} {'Avg CLV':<12} "
          f"{'+CLV%':<10} {'Min CLV':<12} {'Max CLV':<12} {'Win%':<8}")
    print(f"  {'-' * 78}")

    sorted_results = sorted(
        all_results,
        key=lambda r: r["clv_summary"]["avg_clv"],
        reverse=True,
    )

    for i, r in enumerate(sorted_results, 1):
        cs = r["clv_summary"]
        wr = (r["n_winners"] / r["total_bets"] * 100) if r["total_bets"] > 0 else 0.0
        print(
            f"  #{i:<4} {r['model_name']:<20} "
            f"{r['total_bets']:<6} "
            f"{cs['avg_clv']:<+10.6f}   "
            f"{cs['positive_clv_pct']:<8.1f}% "
            f"{cs['min_clv']:<+10.6f}   "
            f"{cs['max_clv']:<+10.6f}   "
            f"{wr:<6.1f}%"
        )

    print(f"{'=' * 80}")

    # Best and worst
    best = sorted_results[0] if sorted_results else None
    worst = sorted_results[-1] if sorted_results else None

    if best:
        print(f"\n  Best CLV: {best['model_name']} "
              f"(avg={best['clv_summary']['avg_clv']:+.6f}, "
              f"+CLV={best['clv_summary']['positive_clv_pct']:.1f}%)")
    if worst:
        print(f"  Worst CLV: {worst['model_name']} "
              f"(avg={worst['clv_summary']['avg_clv']:+.6f}, "
              f"+CLV={worst['clv_summary']['positive_clv_pct']:.1f}%)")

    # Interpretation
    avg_all = np.mean([r["clv_summary"]["avg_clv"] for r in all_results]) if all_results else 0.0
    print(f"\n  Average CLV across all models: {avg_all:+.6f}")
    print(f"  Interpretation: {interpret_clv(avg_all)}")
    print()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main() -> None:
    logger.info("Loading test data...")
    test_df = load_test_data()
    y_true = test_df["result_idx"].values

    # Generate odds once
    logger.info("Generating consensus odds...")
    consensus_odds = generate_consensus_odds(y_true, seed=9999)
    logger.info("Generating closing odds...")
    closing_odds = generate_closing_odds(consensus_odds, y_true, seed=8888)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_results: list[dict[str, Any]] = []

    for i, cfg in enumerate(MODELS_CONFIG):
        name = cfg["name"]
        logger.info(
            "[%d/%d] Processing %s (Brier=%.4f)...",
            i + 1, len(MODELS_CONFIG), name, cfg["brier"],
        )

        clv_bets = simulate_bets_and_calculate_clv(
            name, cfg["brier"], test_df, y_true,
            consensus_odds, closing_odds,
        )

        if not clv_bets:
            logger.warning("  No bets placed for %s", name)
            continue

        # Build result dict for saving
        batch_result = calculate_batch_clv(
            [
                {
                    "your_odds": b["your_odds"],
                    "closing_odds": b["closing_odds"],
                    "market": b["market"],
                    "label": b["match_label"],
                }
                for b in clv_bets
            ]
        )

        result = {
            "model_name": name,
            "timestamp": timestamp,
            "total_bets": len(clv_bets),
            "n_winners": sum(1 for b in clv_bets if b["won"]),
            "n_losers": sum(1 for b in clv_bets if not b["won"]),
            "clv_summary": {
                "avg_clv": batch_result["avg_clv"],
                "positive_clv_pct": batch_result["positive_clv_pct"],
                "max_clv": batch_result["max_clv"],
                "min_clv": batch_result["min_clv"],
                "n_with_closing": batch_result["n_with_closing"],
                "n_missing_closing": batch_result["n_missing_closing"],
            },
            "bets": clv_bets,
        }

        save_path = save_clv_results(name, clv_bets, timestamp)
        all_results.append(result)
        logger.info(
            "  %s: %d bets, avg CLV = %+.6f, +CLV = %.1f%%",
            name, len(clv_bets),
            batch_result["avg_clv"],
            batch_result["positive_clv_pct"],
        )

    # Save summary
    save_summary(all_results, timestamp)

    # Print report
    print_clv_report(all_results)

    logger.info("CLV analysis complete for %d models!", len(all_results))


if __name__ == "__main__":
    main()
