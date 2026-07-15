"""
Optimize bankroll management strategy via grid search.

Searches over a parameter space of stake strategies and risk limits to find
the combination that maximises the Sharpe ratio while keeping the maximum
drawdown below 20% (capital preservation constraint).

Search Space
------------
Stake strategies:
  - Kelly (full)
  - Fractional Kelly: k ∈ {0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.75, 1.0}
  - Fixed Ratio: ratio ∈ {1%, 2%, 3%, 4%, 5%, 7%, 10%}

Risk limits:
  - Max daily loss: {none, 1%, 2%, 3%, 5%, 7%, 10%}
  - Max drawdown:  {none, 10%, 15%, 20%, 25%, 30%, 40%, 50%}

Objective
---------
  Maximise    Sharpe ratio
  Subject to  max_drawdown < 20%
              (a positive Sharpe is preferred over negative)

Output
------
  reports/bankroll_optimization_{timestamp}.json  — full grid + best strategy
  Console — ranked leaderboard with best strategy details
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.backtesting.backtester import Backtester
from src.betting.staking import (
    StakingStrategy,
    FixedRatioStaking,
    KellyStaking,
    FractionalKellyStaking,
)
from src.betting.filtering import BetFilter
from src.betting.risk_management import RiskManager
from src.betting.models import Bankroll, BetSlip, Outcome

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("optimize_bankroll")


# ── Configuration ────────────────────────────────────────

INITIAL_BANKROLL = 1000.0
BASE_BET_FILTER = BetFilter(
    min_ev=0.05,
    min_confidence=0.6,
    min_odds=1.5,
    max_stake=0.10,
    markets=("1X2",),
)
BOOKMAKER_MARGIN = 0.05
REPORTS_DIR = Path("reports")

# Drawdown constraint: strategies exceeding this are penalised
MAX_DRAWDOWN_CONSTRAINT = 20.0  # %

# Number of top candidates to refine (final pass with fixed seed)
N_REFINE_CANDIDATES = 10


# ═══════════════════════════════════════════════════════════
#  1. SEARCH SPACE DEFINITIONS
# ═══════════════════════════════════════════════════════════

# Fractional Kelly fractions to sweep
KELLY_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.33, 0.40, 0.50, 0.75, 1.0]

# Fixed ratios to sweep (as decimals, e.g. 0.01 = 1%)
FIXED_RATIOS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

# Risk limit sweep values
MAX_LOSS_PCTS = [None, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
MAX_DRAWDOWN_PCTS = [None, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]


def build_strategy_grid() -> list[dict[str, Any]]:
    """Build the full grid of stake strategy configurations.

    Returns a list of dicts with 'label', 'strategy', and 'params' keys.
    """
    grid: list[dict[str, Any]] = []

    # 1. Full Kelly
    grid.append({
        "label": "Full Kelly",
        "strategy_type": "kelly",
        "factory": lambda: KellyStaking(),
        "params": {"type": "FullKelly"},
    })

    # 2. Fractional Kelly sweep
    for k in KELLY_FRACTIONS:
        pct_str = f"{k * 100:.0f}%" if k < 1.0 else f"{k * 100:.0f}%"
        grid.append({
            "label": f"Kelly {pct_str}",
            "strategy_type": "fractional_kelly",
            "factory": lambda k=k: FractionalKellyStaking(fraction=k),
            "params": {"type": "FractionalKelly", "kelly_fraction": k},
        })

    # 3. Fixed Ratio sweep
    for r in FIXED_RATIOS:
        pct_str = f"{r * 100:.0f}%"
        grid.append({
            "label": f"FixedRatio {pct_str}",
            "strategy_type": "fixed_ratio",
            "factory": lambda r=r: FixedRatioStaking(ratio=r),
            "params": {"type": "FixedRatio", "ratio": r},
        })

    return grid


def build_risk_grid() -> list[dict[str, Any]]:
    """Build the grid of risk limit configurations.

    Returns a list of (label, risk_config_dict) tuples.
    """
    grid: list[dict[str, Any]] = []

    for max_loss, max_dd in itertools.product(MAX_LOSS_PCTS, MAX_DRAWDOWN_PCTS):
        enabled = max_loss is not None or max_dd is not None
        risk_config = {
            "risk_manager": {
                "enabled": enabled,
                "daily_loss": {
                    "enabled": max_loss is not None,
                    "max_loss_pct": max_loss or 0.0,
                },
                "drawdown": {
                    "enabled": max_dd is not None,
                    "max_drawdown_pct": max_dd or 0.0,
                    "cooldown_on_breach": True,
                    "cooldown_bets": 3,
                },
                "consecutive_losses": {"enabled": False},
                "frequency": {"enabled": False},
                "stake": {"enabled": False},
                "diversification": {"enabled": False},
                "exposure": {"enabled": False},
            }
        }

        # Build label
        parts = []
        if max_loss is not None:
            parts.append(f"Loss<{max_loss:.0f}%")
        if max_dd is not None:
            parts.append(f"DD<{max_dd:.0f}%")
        if not parts:
            parts.append("NoRisk")

        grid.append({
            "label": ", ".join(parts),
            "risk_config": risk_config,
            "params": {
                "max_loss_pct": max_loss,
                "max_drawdown_pct": max_dd,
            },
        })

    return grid


# ═══════════════════════════════════════════════════════════
#  MODEL CONFIG
# ═══════════════════════════════════════════════════════════

MODEL_CONFIG = {
    "name": "Ensemble",
    "brier": 0.5775,
    "calibration": "n/a (weighted avg)",
}


# ═══════════════════════════════════════════════════════════
#  DATA LOADING & PREDICTION GENERATION
# ═══════════════════════════════════════════════════════════


def load_test_data() -> pd.DataFrame:
    """Load World Cup dataset and return the test set (chronological last 20%)."""
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


def generate_predictions(
    y_true: np.ndarray,
    n_classes: int,
    target_brier: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic predictions matching a target Brier score."""
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


def generate_consensus_odds(
    y_true: np.ndarray,
    n_classes: int = 3,
    margin: float = BOOKMAKER_MARGIN,
    noise_std: float = 0.15,
    seed: int = 9999,
) -> np.ndarray:
    """Generate bookmaker odds from consensus with noise."""
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


# ═══════════════════════════════════════════════════════════
#  BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════


def build_test_df(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    odds: np.ndarray,
) -> pd.DataFrame:
    """Build the DataFrame for the Backtester."""
    return pd.DataFrame({
        "home_team": test_df["home_team"],
        "away_team": test_df["away_team"],
        "home_goals": test_df["home_goals"].fillna(0).astype(float),
        "away_goals": test_df["away_goals"].fillna(0).astype(float),
        "BbAvH": odds[:, 2],
        "BbAvD": odds[:, 1],
        "BbAvA": odds[:, 0],
        "model_prob_home_win": probs[:, 2],
        "model_prob_draw": probs[:, 1],
        "model_prob_away_win": probs[:, 0],
    })


# ═══════════════════════════════════════════════════════════
#  CORE OPTIMIZATION RUNNER
# ═══════════════════════════════════════════════════════════


def evaluate_candidate(
    bt_df: pd.DataFrame,
    stake_strategy: StakingStrategy,
    risk_config: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Evaluate a single (stake_strategy, risk_config) candidate.

    Runs the manual backtest simulation with RiskManager integration,
    then returns all performance metrics.

    Parameters
    ----------
    bt_df : pd.DataFrame
        Test match data with model probs and odds columns.
    stake_strategy : StakingStrategy
        The stake sizing strategy to evaluate.
    risk_config : dict
        RiskManager configuration override.
    label : str
        Human-readable label for this candidate.

    Returns
    -------
    dict
        All performance metrics + candidate metadata.
    """
    rm = RiskManager(auto_load=False)
    rm.load_config(config_override=risk_config)

    bankroll = float(INITIAL_BANKROLL)
    peak = bankroll

    # Track detailed metrics
    n_bets = 0
    n_wins = 0
    n_rejected_by_risk = 0
    total_staked = 0.0
    total_profit = 0.0
    drawdowns: list[float] = []
    returns: list[float] = []
    equity_curve = [0.0]
    running_pnl = 0.0

    # Single Bankroll object reused across all matches
    bankroll_obj = Bankroll(initial_balance=INITIAL_BANKROLL)

    for match_idx, row in bt_df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        home_goals = float(row.get("home_goals", 0) or 0)
        away_goals = float(row.get("away_goals", 0) or 0)

        # Evaluate outcomes
        outcomes = [
            ("home_win", float(row.get("model_prob_home_win", 0.0)), float(row.get("BbAvH", 0.0))),
            ("draw", float(row.get("model_prob_draw", 0.0)), float(row.get("BbAvD", 0.0))),
            ("away_win", float(row.get("model_prob_away_win", 0.0)), float(row.get("BbAvA", 0.0))),
        ]

        best_ev = -999.0
        best_outcome: str | None = None
        best_prob = 0.0
        best_odds = 0.0

        for outcome, prob, odds_val in outcomes:
            if prob < BASE_BET_FILTER.min_confidence:
                continue
            if odds_val <= 1.0:
                continue
            ev = (prob * odds_val) - 1.0
            if ev < BASE_BET_FILTER.min_ev:
                continue
            if ev > best_ev:
                best_ev = ev
                best_outcome = outcome
                best_prob = prob
                best_odds = odds_val

        if best_outcome is None or bankroll <= 0:
            continue

        # Calculate stake
        stake_amount = stake_strategy.calculate_stake(
            best_prob, best_odds, bankroll, ev=best_ev,
        )
        if stake_amount <= 0:
            continue
        stake_pct = stake_amount / max(bankroll, 0.01)

        # --- Risk Manager check ---
        # Build minimal BetSlip for risk checks
        slip = BetSlip(
            match_id=str(match_idx),
            home_team=home,
            away_team=away,
            outcome=Outcome.HOME if best_outcome == "home_win" else (
                Outcome.DRAW if best_outcome == "draw" else Outcome.AWAY
            ),
            decimal_odds=Decimal(str(best_odds)),
            model_prob=Decimal(str(best_prob)),
            fair_prob=Decimal(str(1.0 / best_odds)),
            odds_source="synthetic",
            stake_amount=stake_amount,
            stake_pct=stake_pct,
            ev=best_ev,
        )

        bankroll_obj.current_balance = bankroll
        bankroll_obj.peak_balance = peak

        allowed, reason = rm.check_bet(slip, bankroll_obj)
        if not allowed:
            n_rejected_by_risk += 1
            continue

        # Settle bet
        won = False
        if best_outcome == "home_win":
            won = home_goals > away_goals
        elif best_outcome == "draw":
            won = home_goals == away_goals
        elif best_outcome == "away_win":
            won = away_goals > home_goals

        profit = stake_amount * (best_odds - 1.0) if won else -stake_amount
        bankroll += profit
        bankroll_obj.record_result(profit, won)
        rm.record_result(slip, profit=profit, won=won)
        # Update peak after the bet is placed

        if bankroll > peak:
            peak = bankroll

        # Track metrics
        n_bets += 1
        if won:
            n_wins += 1
        total_staked += stake_amount
        total_profit += profit
        running_pnl += profit
        equity_curve.append(round(running_pnl, 2))

        # Track drawdown
        dd = (peak - bankroll) / peak * 100 if peak > 0 else 0.0
        drawdowns.append(dd)

        # Track returns for Sharpe
        if stake_amount > 0:
            returns.append(profit / stake_amount)

    # ── Compute metrics ──
    roi_pct = (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    yield_pct = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (n_wins / n_bets * 100) if n_bets > 0 else 0.0
    max_dd = max(drawdowns) if drawdowns else 0.0

    # Compute profit factor from equity curve changes
    gross_profit = 0.0
    gross_loss = 0.0
    for i in range(1, len(equity_curve)):
        diff = equity_curve[i] - equity_curve[i - 1]
        if diff > 0:
            gross_profit += diff
        elif diff < 0:
            gross_loss += abs(diff)

    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0
        else float("inf") if gross_profit > 0
        else 0.0
    )

    # Sharpe ratio (annualised, assuming ~500 bets/year)
    returns_arr = np.array(returns)
    if len(returns_arr) > 1 and np.std(returns_arr) > 0:
        sharpe = float(
            (np.mean(returns_arr) / np.std(returns_arr, ddof=1)) * np.sqrt(500)
        )
        sortino = float(
            (np.mean(returns_arr) / np.std(returns_arr[returns_arr < 0], ddof=1)) * np.sqrt(500)
            if len(returns_arr[returns_arr < 0]) > 1
            else 0.0
        )
    else:
        sharpe = 0.0
        sortino = 0.0

    return {
        "label": label,
        "total_bets": n_bets,
        "winning_bets": n_wins,
        "losing_bets": n_bets - n_wins,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi_pct, 4),
        "yield_pct": round(yield_pct, 4),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "profit_factor": round(profit_factor, 4),
        "final_bankroll": round(bankroll, 2),
        "n_rejected_by_risk": n_rejected_by_risk,
        "avg_stake_pct": round(
            (total_staked / n_bets / max(INITIAL_BANKROLL, 0.01) * 100)
            if n_bets > 0 else 0.0, 4,
        ),
    }


def score_candidate(
    metrics: dict[str, Any],
    max_dd_constraint: float = MAX_DRAWDOWN_CONSTRAINT,
) -> float:
    """Compute the optimisation score for a candidate.

    The score is the Sharpe ratio, but candidates that violate the
    drawdown constraint are heavily penalised (score = -999).

    If no bet was placed, score = -infinity.
    """
    if metrics.get("total_bets", 0) == 0:
        return float("-inf")

    dd = metrics.get("max_drawdown_pct", 0.0)
    sharpe = metrics.get("sharpe_ratio", -999.0)

    if dd >= max_dd_constraint:
        # Heavily penalised — violated drawdown constraint
        return -999.0 + (sharpe / 1000.0)  # tiny tiebreaker

    if sharpe <= 0:
        return sharpe * 0.01  # squeeze negative Sharpe close to 0

    return sharpe


# ═══════════════════════════════════════════════════════════
#  CONSOLE OUTPUT
# ═══════════════════════════════════════════════════════════


def print_leaderboard(
    results: list[dict[str, Any]],
    title: str = "OPTIMISATION LEADERBOARD",
    top_n: int = 20,
) -> None:
    """Print ranked leaderboard of optimisation results."""
    ranked = sorted(
        results,
        key=lambda r: r.get("_score", -999),
        reverse=True,
    )[:top_n]

    print(f"\n{'=' * 110}")
    print(f"  {title}".center(108))
    print(f"{'=' * 110}")
    print()
    header = (
        f"  {'Rank':<6} {'Strategy + Risk':<42} {'Bets':<6} "
        f"{'ROI%':<9} {'Sharpe':<9} {'DD%':<9} {'PF':<8} "
        f"{'Final':<10} {'Score':<8}"
    )
    print(header)
    print(f"  {'-' * 108}")

    for i, r in enumerate(ranked, 1):
        label = r.get("label", r.get("_label", "?"))
        bets = r.get("total_bets", 0)
        roi = r.get("roi_pct", 0)
        sharpe = r.get("sharpe_ratio", 0)
        dd = r.get("max_drawdown_pct", 0)
        pf = r.get("profit_factor", 0)
        final = r.get("final_bankroll", 0)
        score = r.get("_score", 0)
        rank_str = f"#{i}"
        dd_flag = " ⚠" if dd >= MAX_DRAWDOWN_CONSTRAINT else "  "

        print(
            f"  {rank_str:<6} {label:<42}{dd_flag} {bets:<6} "
            f"{roi:>+7.2f}% {sharpe:<8.4f} {dd:<8.2f}% "
            f"{pf:<7.4f} {final:>+8.2f} {score:<8.2f}"
        )

    print(f"{'=' * 110}")
    print()

    if ranked:
        best = ranked[0]
        print(f"  🏆 Optimal Strategy: {best['label']}")
        print(f"     Sharpe={best['sharpe_ratio']:.4f}, "
              f"ROI={best['roi_pct']:+.2f}%, "
              f"Drawdown={best['max_drawdown_pct']:.2f}%, "
              f"Profit Factor={best['profit_factor']:.2f}")
        print(f"     Final Bankroll: {best['final_bankroll']:.2f} "
              f"(from {INITIAL_BANKROLL:.0f}) "
              f"over {best['total_bets']} bets")
    print()


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════


def main() -> None:
    # ── 1. Load data ──────────────────────────────────
    logger.info("Loading test data...")
    test_df = load_test_data()
    y_true = test_df["result_idx"].values

    model_seed = 42
    probs = generate_predictions(
        y_true, 3, MODEL_CONFIG["brier"], seed=model_seed,
    )
    logger.info("Generating consensus odds...")
    consensus_odds = generate_consensus_odds(y_true, seed=9999)
    bt_df = build_test_df(test_df, probs, consensus_odds)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── 2. Build search grids ─────────────────────────
    stake_grid = build_strategy_grid()
    risk_grid = build_risk_grid()

    total_combos = len(stake_grid) * len(risk_grid)
    logger.info(
        "Search space: %d stake strategies × %d risk configs = %d candidates",
        len(stake_grid), len(risk_grid), total_combos,
    )

    # ── 3. Grid search ────────────────────────────────
    all_results: list[dict[str, Any]] = []
    n_completed = 0

    for stake_cfg in stake_grid:
        for risk_cfg in risk_grid:
            label = f"{stake_cfg['label']} + {risk_cfg['label']}"
            n_completed += 1

            if n_completed % 25 == 0 or n_completed == 1:
                logger.info(
                    "[%d/%d] Evaluating %s...",
                    n_completed, total_combos, label,
                )

            try:
                strategy = stake_cfg["factory"]()
                metrics = evaluate_candidate(
                    bt_df, strategy, risk_cfg["risk_config"], label,
                )

                # Compute score
                score = score_candidate(metrics)
                metrics["_score"] = score
                metrics["_stake_params"] = stake_cfg["params"]
                metrics["_risk_params"] = risk_cfg["params"]
                metrics["_passes_dd_constraint"] = (
                    metrics.get("max_drawdown_pct", 999) < MAX_DRAWDOWN_CONSTRAINT
                )

                all_results.append(metrics)

            except Exception as exc:
                logger.error(
                    "Failed for %s: %s\n%s",
                    label, exc, traceback.format_exc(),
                )

    # ── 4. Find best strategy ─────────────────────────
    valid_results = [r for r in all_results if r.get("_score", -999) > -999]
    passing_dd = [
        r for r in valid_results
        if r.get("_passes_dd_constraint", False)
    ]

    if not passing_dd:
        logger.warning(
            "No strategies satisfy the drawdown constraint < %.0f%%! "
            "Showing best available regardless of constraint.",
            MAX_DRAWDOWN_CONSTRAINT,
        )
        passing_dd = valid_results

    passing_dd.sort(key=lambda r: r.get("_score", -999), reverse=True)
    best = passing_dd[0] if passing_dd else None

    # ── 5. Save results ───────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "configuration": {
            "model": MODEL_CONFIG["name"],
            "brier": MODEL_CONFIG["brier"],
            "initial_bankroll": INITIAL_BANKROLL,
            "bet_filter": {
                "min_ev": BASE_BET_FILTER.min_ev,
                "min_confidence": BASE_BET_FILTER.min_confidence,
                "min_odds": BASE_BET_FILTER.min_odds,
                "max_stake": BASE_BET_FILTER.max_stake,
            },
            "max_drawdown_constraint": MAX_DRAWDOWN_CONSTRAINT,
        },
        "search_space": {
            "stake_strategies": {
                "full_kelly": 1,
                "fractional_kelly_fractions": KELLY_FRACTIONS,
                "fixed_ratio_ratios": FIXED_RATIOS,
            },
            "risk_limits": {
                "max_loss_pcts": MAX_LOSS_PCTS,
                "max_drawdown_pcts": MAX_DRAWDOWN_PCTS,
            },
            "total_candidates": total_combos,
            "completed": len(all_results),
            "passed_dd_constraint": len(passing_dd),
        },
        "optimal_strategy": best,
        "leaderboard": [
            {
                "rank": i + 1,
                "label": r["label"],
                "score": r["_score"],
                "sharpe_ratio": r["sharpe_ratio"],
                "roi_pct": r["roi_pct"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "final_bankroll": r["final_bankroll"],
                "total_bets": r["total_bets"],
                "stake_params": r["_stake_params"],
                "risk_params": r["_risk_params"],
                "passes_dd_constraint": r["_passes_dd_constraint"],
            }
            for i, r in enumerate(
                sorted(
                    all_results,
                    key=lambda x: x.get("_score", -999),
                    reverse=True,
                )[:50]
            )
        ],
        "all_results": sorted(
            all_results,
            key=lambda x: x.get("_score", -999),
            reverse=True,
        ),
    }

    filename = f"bankroll_optimization_{timestamp}.json"
    filepath = REPORTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Full optimisation results saved to %s", filepath)

    # ── 6. Print leaderboard ──────────────────────────
    print_leaderboard(all_results)

    # ── 7. Summary ────────────────────────────────────
    if best:
        print(f"{'=' * 60}")
        print("  OPTIMAL BANKROLL MANAGEMENT STRATEGY")
        print(f"{'=' * 60}")
        print(f"  Strategy:    {best['label']}")
        print(f"  Sharpe:      {best['sharpe_ratio']:.4f}")
        print(f"  ROI:         {best['roi_pct']:+.2f}%")
        print(f"  Drawdown:    {best['max_drawdown_pct']:.2f}%  "
              f"{'✅ < 20%' if best['max_drawdown_pct'] < 20 else '❌ >= 20%'}")
        print(f"  Profit Fac:  {best['profit_factor']:.4f}")
        print(f"  Final Bank:  {best['final_bankroll']:.2f} "
              f"(from {INITIAL_BANKROLL:.0f})")
        print(f"  Total Bets:  {best['total_bets']}")
        print(f"  Win Rate:    {best['win_rate_pct']:.1f}%")
        print(f"  Yield:       {best['yield_pct']:+.2f}%")
        print(f"  Bets Rej'd:  {best.get('n_rejected_by_risk', 0)}")
        print()

        # Configuration summary
        stake_p = best.get("_stake_params", {})
        risk_p = best.get("_risk_params", {})
        print("  Configuration:")
        print(f"    Stake:     {stake_p}")
        print(f"    Risk:      {risk_p}")
        print(f"{'=' * 60}")

    n_passing = sum(
        1 for r in all_results if r.get("_passes_dd_constraint", False)
    )
    logger.info(
        "Optimisation complete! %d/%d candidates pass the < %.0f%% drawdown "
        "constraint.",
        n_passing, len(all_results), MAX_DRAWDOWN_CONSTRAINT,
    )


if __name__ == "__main__":
    main()
