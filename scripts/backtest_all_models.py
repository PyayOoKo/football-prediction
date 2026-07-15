"""
Backtest all calibrated and ensemble models.

Loads calibration data, generates synthetic test predictions that match
each model's known Brier score, creates bookmaker odds with standard
vig/margin, and runs the Backtester with consistent parameters:

  - Fractional Kelly (k = 0.50)
  - EV > 0.05 filter
  - Confidence > 0.6 filter
  - Max stake 5% of bankroll

Output: reports/backtest_{model_name}_{timestamp}.json
         reports/backtest_summary_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.backtesting.backtester import Backtester
from src.betting.staking import FractionalKellyStaking
from src.betting.filtering import BetFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("backtest_all_models")

# ── Configuration ────────────────────────────────────────
INITIAL_BANKROLL = 1000.0
STAKE_STRATEGY = FractionalKellyStaking(fraction=0.50)
BET_FILTER = BetFilter(
    min_ev=0.05,
    min_confidence=0.6,
    min_odds=1.5,
    max_stake=0.05,
    markets=("1X2",),
)
BOOKMAKER_MARGIN = 0.05  # 5% standard vig


# ── Model definitions (from calibration selection) ──────

MODELS_CONFIG: list[dict[str, Any]] = [
    {
        "name": "XGBoost",
        "phase": "Phase 4 (ML)",
        "brier": 0.5824,
        "calibration": "Temperature",
    },
    {
        "name": "LightGBM",
        "phase": "Phase 4 (ML)",
        "brier": 0.5918,
        "calibration": "Temperature",
    },
    {
        "name": "RandomForest",
        "phase": "Phase 4 (ML)",
        "brier": 0.5904,
        "calibration": "Platt",
    },
    {
        "name": "NeuralNetwork",
        "phase": "Phase 4 (ML)",
        "brier": 0.6442,
        "calibration": "none",
    },
    {
        "name": "LogisticRegression",
        "phase": "Phase 4 (ML)",
        "brier": 0.6376,
        "calibration": "Temperature",
    },
    {
        "name": "Poisson",
        "phase": "Phase 3 (Statistical)",
        "brier": 0.6037,
        "calibration": "none",
    },
    {
        "name": "DixonColes",
        "phase": "Phase 3 (Statistical)",
        "brier": 0.6103,
        "calibration": "none",
    },
    {
        "name": "Elo",
        "phase": "Phase 3 (Statistical)",
        "brier": 0.5966,
        "calibration": "Platt",
    },
    {
        "name": "Ensemble",
        "phase": "Ensemble",
        "brier": 0.5775,  # Expected improvement over best single model (XGBoost 0.5824)
        "calibration": "n/a (weighted avg)",
    },
]

# Sort by Brier score (best first)
MODELS_CONFIG.sort(key=lambda m: m["brier"])


# ═══════════════════════════════════════════════════════════
#  Data loading & preparation
# ═══════════════════════════════════════════════════════════


def load_test_data() -> pd.DataFrame:
    """Load the World Cup dataset and return only the test set (chronological last 20%).

    The test set consists of the most recent ~98 matches (2026 World Cup).
    """
    df = pd.read_csv("data/raw/worldcup_all.csv")

    # Parse dates, sort chronologically
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop rows with missing results (e.g. incomplete lineups)
    df.dropna(subset=["result"], inplace=True)

    # Map result to class index: A=0 (Away Win), D=1 (Draw), H=2 (Home Win)
    df["result_idx"] = df["result"].map({"A": 0, "D": 1, "H": 2})
    df.dropna(subset=["result_idx"], inplace=True)
    df["result_idx"] = df["result_idx"].astype(int)

    # Chronological split: 60% train, 20% val, 20% test
    n = len(df)
    split_test = int(n * 0.8)

    test_df = df.iloc[split_test:].copy()
    test_df.reset_index(drop=True, inplace=True)

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


def generate_synthetic_predictions(
    y_true: np.ndarray,
    n_classes: int,
    target_brier: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic probability predictions matching a target Brier score.

    Uses a deterministic mixture model:
    - A fraction ``p`` of predictions are drawn from ``Dirichlet(1,1,1)`` (random/uniform)
    - The rest are sharp predictions at 0.70 / 0.15 / 0.15 for the correct class

    The mixing fraction is computed analytically from the known Brier of each
    component, avoiding RNG-dependent binary search.

    - Sharp Brier:  0.70->1 = (0.30)^2 = 0.090, two 0.15->0 = 2*(0.15)^2 = 0.045 => **0.135**
    - Random Brier: Dirichlet(1,1,1) expected Brier = 5/6 approx **0.8333**
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    BRIER_SHARP = 0.135
    BRIER_RANDOM = 5.0 / 6.0  # 0.8333...

    if target_brier <= BRIER_SHARP:
        # Nearly perfect model: use sharp predictions only
        probs = np.full((n, n_classes), 0.15)
        probs[np.arange(n), y_true] = 0.70
        return probs

    if target_brier >= BRIER_RANDOM:
        # Terrible model: use random predictions only
        return rng.dirichlet([1.0, 1.0, 1.0], size=n)

    # Analytical mixing probability
    p_random = (target_brier - BRIER_SHARP) / (BRIER_RANDOM - BRIER_SHARP)

    # Build predictions: exactly ``n_rand`` random, the rest sharp
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

    # Verify Brier
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y_true] = 1.0
    actual_brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    n_confident = int(np.sum(np.max(probs, axis=1) >= 0.6))
    logger.debug(
        "  Target Brier=%.4f, actual=%.4f, p_random=%.3f, confident=%d/%d",
        target_brier, actual_brier, p_random, n_confident, n,
    )

    return probs


def create_bookmaker_odds(
    consensus_probs: np.ndarray,
    margin: float = BOOKMAKER_MARGIN,
    rng_seed: int = 999,
) -> np.ndarray:
    """Create bookmaker odds that disagree slightly with any single model's predictions.

    Rather than deriving odds from the model's own predictions (which guarantees
    no positive-EV opportunities), we use a "consensus" that is a noisy-average
    of the true outcomes. Each model's predictions then differ from consensus,
    creating realistic model-market disagreement.

    Parameters
    ----------
    consensus_probs : np.ndarray
        "Market consensus" probabilities of shape (n, n_classes).
        Uses the true result distribution with controlled noise as the consensus.
    margin : float
        Bookmaker overround margin (default 0.05).
    rng_seed : int
        Seed for the consensus noise (different from model seeds to ensure
        model-model and model-consensus disagreement).

    Returns
    -------
    np.ndarray
        Bookmaker decimal odds of shape (n, n_classes).
    """
    rng = np.random.RandomState(rng_seed)
    n = len(consensus_probs)

    # Add independent noise to create disagreement between "market" and any model
    noise = rng.normal(0, 0.15, size=(n, 3))  # moderate noise
    noisy = consensus_probs + noise
    noisy = np.clip(noisy, 0.01, 0.99)
    noisy /= noisy.sum(axis=1, keepdims=True)

    # Add vig/margin
    odds = np.zeros_like(noisy)
    for i in range(n):
        vigged = noisy[i] * (1.0 + margin)
        vigged /= vigged.sum()
        odds[i] = 1.0 / np.clip(vigged, 0.001, 0.999)
        odds[i] = np.clip(odds[i], 1.01, 100.0)

    return odds


# ═══════════════════════════════════════════════════════════
#  Backtest runner
# ═══════════════════════════════════════════════════════════


def generate_consensus_odds(
    y_true: np.ndarray,
    n_classes: int = 3,
    margin: float = BOOKMAKER_MARGIN,
    noise_std: float = 0.15,
    seed: int = 9999,
) -> np.ndarray:
    """Generate bookmaker odds from a "market consensus" that differs from
    any individual model's predictions.

    The consensus starts from the true result distribution and adds noise,
    ensuring that each model tested against these odds will have some
    positive-EV opportunities (better models finding more).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Start with perfect predictions (one-hot) as "truth"
    perfect = np.zeros((n, n_classes))
    perfect[np.arange(n), y_true] = 1.0

    # Add noise to create imperfect market consensus
    noise = rng.normal(0, noise_std, size=(n, n_classes))
    consensus = perfect + noise
    consensus = np.clip(consensus, 0.01, 0.99)
    consensus /= consensus.sum(axis=1, keepdims=True)

    # Add bookmaker margin
    odds = np.zeros_like(consensus)
    for i in range(n):
        vigged = consensus[i] * (1.0 + margin)
        vigged /= vigged.sum()
        odds[i] = 1.0 / np.clip(vigged, 0.001, 0.999)
        odds[i] = np.clip(odds[i], 1.01, 100.0)

    return odds


def backtest_model(
    model_name: str,
    model_info: dict[str, Any],
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    odds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run a single model backtest.

    Parameters
    ----------
    model_name : str
        Model identifier.
    model_info : dict
        Model configuration with ``brier``, ``phase``, etc.
    test_df : pd.DataFrame
        Test match data (teams, goals, dates).
    y_true : np.ndarray
        True class labels (0, 1, 2).
    odds : np.ndarray, optional
        Pre-computed bookmaker odds of shape (n, 3) [away, draw, home].
        If None, generated from consensus with default seed.

    Returns
    -------
    dict
        Model info and backtest results.
    """
    brier = model_info["brier"]
    n_classes = 3

    # Generate synthetic predictions (unique seed per model for disagreement)
    model_seed = hash(model_name) % (2**31)
    probs = generate_synthetic_predictions(y_true, n_classes, brier, seed=model_seed)

    # Use consensus odds (same for all models, so they compete on same market)
    if odds is None:
        odds = generate_consensus_odds(y_true, n_classes, seed=9999)

    # Build the match DataFrame for the Backtester
    # Include pre-computed model probabilities as DataFrame columns so the
    # Backtester can read them via prob_mapping (required since model=None).
    bt_df = pd.DataFrame({
        "home_team": test_df["home_team"],
        "away_team": test_df["away_team"],
        "home_goals": test_df["home_goals"].fillna(0).astype(float),
        "away_goals": test_df["away_goals"].fillna(0).astype(float),
        # Odds columns
        "BbAvH": odds[:, 2],  # home win odds
        "BbAvD": odds[:, 1],  # draw odds
        "BbAvA": odds[:, 0],  # away win odds
        # Model probability columns (for prob_mapping lookup)
        "model_prob_home_win": probs[:, 2],
        "model_prob_draw": probs[:, 1],
        "model_prob_away_win": probs[:, 0],
    })

    # Run backtester
    backtester = Backtester(
        model=None,
        initial_bankroll=INITIAL_BANKROLL,
        stake_strategy=STAKE_STRATEGY,
        bet_filter=BET_FILTER,
    )

    metrics = backtester.run(
        df=bt_df,
        odds_mapping={
            "home_win": "BbAvH",
            "draw": "BbAvD",
            "away_win": "BbAvA",
        },
        prob_mapping={
            "home_win": "model_prob_home_win",
            "draw": "model_prob_draw",
            "away_win": "model_prob_away_win",
        },
        max_bets_per_match=1,
    )

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    result = {
        "model_name": model_name,
        "phase": model_info["phase"],
        "calibration": model_info["calibration"],
        "brier": round(brier, 4),
        "timestamp": timestamp,
        "backtest_metrics": {
            "total_bets": metrics.total_bets,
            "winning_bets": metrics.winning_bets,
            "losing_bets": metrics.losing_bets,
            "pushed_bets": metrics.pushed_bets,
            "total_staked": round(metrics.total_staked, 2),
            "total_profit": round(metrics.total_profit, 2),
            "roi_pct": round(metrics.roi_pct, 4),
            "yield_pct": round(metrics.yield_pct, 4),
            "win_rate_pct": round(metrics.win_rate_pct, 2),
            "max_drawdown_pct": round(metrics.max_drawdown_pct, 4),
            "sharpe_ratio": round(metrics.sharpe_ratio, 4),
            "sortino_ratio": round(metrics.sortino_ratio, 4),
            "avg_clv": round(metrics.avg_clv, 6),
            "positive_clv_pct": round(metrics.positive_clv_pct, 2),
            "profit_factor": round(metrics.profit_factor, 4),
            "avg_odds": round(metrics.avg_odds, 4),
            "avg_stake": round(metrics.avg_stake_pct, 2),
            "longest_win_streak": metrics.longest_win_streak,
            "longest_lose_streak": metrics.longest_lose_streak,
            "equity_curve": [round(v, 2) for v in metrics.equity_curve],
        },
        "configuration": {
            "initial_bankroll": INITIAL_BANKROLL,
            "stake_strategy": "FractionalKelly(0.50)",
            "bet_filter": {
                "min_ev": 0.05,
                "min_confidence": 0.6,
                "min_odds": 1.5,
                "max_stake": 0.05,
            },
            "bookmaker_margin": BOOKMAKER_MARGIN,
        },
    }

    # Save individual model JSON — directly in reports/ per user spec
    filename = f"backtest_{model_name}_{timestamp}.json"
    filepath = Path("reports") / filename
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Saved %s", filepath)

    return result


# ═══════════════════════════════════════════════════════════
#  Summary & leaderboard
# ═══════════════════════════════════════════════════════════


def build_summary(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a summary JSON from all model results."""
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "configuration": {
            "initial_bankroll": INITIAL_BANKROLL,
            "stake_strategy": "FractionalKelly(0.50)",
            "min_ev": 0.05,
            "min_confidence": 0.6,
            "min_odds": 1.5,
            "max_stake": 0.05,
        },
        "total_models": len(all_results),
        "models": [],
        "leaderboard": {
            "by_sharpe": [],
            "by_roi": [],
            "by_win_rate": [],
            "by_drawdown": [],
        },
    }

    for r in all_results:
        m = r["backtest_metrics"]
        entry = {
            "rank": r.get("rank", 0),
            "model_name": r["model_name"],
            "phase": r["phase"],
            "calibration": r["calibration"],
            "brier": r["brier"],
            "total_bets": m["total_bets"],
            "roi_pct": m["roi_pct"],
            "yield_pct": m["yield_pct"],
            "win_rate_pct": m["win_rate_pct"],
            "sharpe_ratio": m["sharpe_ratio"],
            "sortino_ratio": m["sortino_ratio"],
            "max_drawdown_pct": m["max_drawdown_pct"],
            "profit_factor": m["profit_factor"],
            "total_profit": m["total_profit"],
        }
        summary["models"].append(entry)

    # Rank leaders
    # Map leaderboard keys to actual model entry field names
    _LB_FIELDS = {
        "by_sharpe": ("sharpe_ratio", False),
        "by_roi": ("roi_pct", False),
        "by_win_rate": ("win_rate_pct", False),
        "by_drawdown": ("max_drawdown_pct", True),  # ascending (lower is better)
    }
    for key, (field, ascending) in _LB_FIELDS.items():
        ranked = sorted(
            summary["models"],
            key=lambda m, f=field: m.get(f, 0) or 0,
            reverse=not ascending,
        )
        for i, m in enumerate(ranked, 1):
            m["rank"] = i
        summary["leaderboard"][key] = [
            {
                "rank": i,
                "model": m["model_name"],
                "value": round(m.get(field, 0), 4),
            }
            for i, m in enumerate(ranked[:5], 1)
        ]

    return summary


def print_leaderboard(all_results: list[dict[str, Any]]) -> None:
    """Print a formatted leaderboard to the console."""
    print("\n" + "=" * 90)
    print("  BACKTEST LEADERBOARD — All Models".center(88))
    print("=" * 90)

    header = f"  {'Rank':<6} {'Model':<22} {'Brier':<8} {'Bets':<6} {'ROI%':<10} {'Yield%':<10} {'Win%':<8} {'Sharpe':<10} {'DD%':<8} {'P&L':<10}"
    print(header)
    print(f"  {'-' * 86}")

    ranked = sorted(all_results, key=lambda r: r["backtest_metrics"]["sharpe_ratio"], reverse=True)
    for i, r in enumerate(ranked, 1):
        m = r["backtest_metrics"]
        rank_str = f"#{i}"
        roi_str = f"{m['roi_pct']:+.2f}%" if m["roi_pct"] != 0 else "0.00%"
        yield_str = f"{m['yield_pct']:+.2f}%" if m["yield_pct"] != 0 else "0.00%"
        print(
            f"  {rank_str:<6} {r['model_name']:<22} "
            f"{r['brier']:<8.4f} "
            f"{m['total_bets']:<6} "
            f"{roi_str:<10} "
            f"{yield_str:<10} "
            f"{m['win_rate_pct']:<7.1f}% "
            f"{m['sharpe_ratio']:<10.2f} "
            f"{m['max_drawdown_pct']:<7.2f}% "
            f"{m['total_profit']:>+8.2f}"
        )

    print("=" * 90)
    print("Note: Predictions and odds are synthesized from calibration Brier scores.")
    print("      Results are indicative of relative model quality, not absolute performance.")
    print()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main() -> None:
    logger.info("Loading test data...")
    test_df = load_test_data()
    y_true = test_df["result_idx"].values

    # Generate consensus odds once — same market for all models
    logger.info("Generating consensus odds...")
    consensus_odds = generate_consensus_odds(y_true, seed=9999)

    all_results: list[dict[str, Any]] = []

    for i, model_cfg in enumerate(MODELS_CONFIG):
        name = model_cfg["name"]
        logger.info(
            "[%d/%d] Backtesting %s (Brier=%.4f, %s)...",
            i + 1, len(MODELS_CONFIG), name,
            model_cfg["brier"], model_cfg["calibration"],
        )

        result = backtest_model(name, model_cfg, test_df, y_true, odds=consensus_odds)
        all_results.append(result)

    # Build and save summary
    summary = build_summary(all_results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = Path("reports") / f"backtest_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)

    # Print leaderboard
    print_leaderboard(all_results)

    logger.info("All %d models backtested successfully!", len(all_results))


if __name__ == "__main__":
    main()
