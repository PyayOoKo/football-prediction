"""
Backtest all stake sizing & risk management strategies.

Runs the same model (XGBoost/Best) against the same test data with each
configured stake strategy and risk management scenario, then compares
results to identify the optimal bankroll management approach.

Two comparison modes:
  1. STAKE STRATEGIES — compare all staking methods (Flat, Kelly, etc.)
  2. RISK SCENARIOS   — compare risk config permutations (daily loss, drawdown, etc.)

Output:
  reports/bankroll_management_{timestamp}.json  — full comparison data
  reports/console — ranked leaderboard per mode
"""

from __future__ import annotations

import json
import logging
import math
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.backtesting.backtester import Backtester
from src.betting.staking import (
    StakingFactory,
    StakingStrategy,
    FlatStaking,
    PercentageStaking,
    FixedRatioStaking,
    VariableRatioStaking,
    VolatilityStaking,
    PortfolioStaking,
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
logger = logging.getLogger("backtest_staking_strategies")

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


# ═══════════════════════════════════════════════════════════
#  1. STAKE STRATEGY DEFINITIONS
# ═══════════════════════════════════════════════════════════

STAKE_STRATEGIES: list[dict[str, Any]] = [
    # (label, strategy_factory, description)
    {
        "label": "Flat £25",
        "factory": lambda: FlatStaking(stake_per_bet=25.0),
        "description": "Fixed £25 per bet regardless of edge",
        "category": "fixed",
    },
    {
        "label": "Flat £50",
        "factory": lambda: FlatStaking(stake_per_bet=50.0),
        "description": "Fixed £50 per bet regardless of edge",
        "category": "fixed",
    },
    {
        "label": "Percentage 1%",
        "factory": lambda: PercentageStaking(stake_pct=0.01),
        "description": "1% of bankroll per bet",
        "category": "percentage",
    },
    {
        "label": "Percentage 2%",
        "factory": lambda: PercentageStaking(stake_pct=0.02),
        "description": "2% of bankroll per bet",
        "category": "percentage",
    },
    {
        "label": "Percentage 5%",
        "factory": lambda: PercentageStaking(stake_pct=0.05),
        "description": "5% of bankroll per bet",
        "category": "percentage",
    },
    {
        "label": "FixedRatio 1%",
        "factory": lambda: FixedRatioStaking(ratio=0.01),
        "description": "1% of bankroll (rejects neg EV)",
        "category": "percentage",
    },
    {
        "label": "FixedRatio 2%",
        "factory": lambda: FixedRatioStaking(ratio=0.02),
        "description": "2% of bankroll (rejects neg EV)",
        "category": "percentage",
    },
    {
        "label": "FixedRatio 3%",
        "factory": lambda: FixedRatioStaking(ratio=0.03),
        "description": "3% of bankroll (rejects neg EV)",
        "category": "percentage",
    },
    {
        "label": "VariableRatio 2%",
        "factory": lambda: VariableRatioStaking(base_ratio=0.02, max_ratio=0.10),
        "description": "2% base scaled by EV magnitude",
        "category": "dynamic",
    },
    {
        "label": "VariableRatio 3%",
        "factory": lambda: VariableRatioStaking(base_ratio=0.03, max_ratio=0.15),
        "description": "3% base scaled by EV magnitude",
        "category": "dynamic",
    },
    {
        "label": "Volatility 2%",
        "factory": lambda: VolatilityStaking(base_ratio=0.02, window=10, sensitivity=2.0),
        "description": "2% base, CV-adjusted for volatility",
        "category": "dynamic",
    },
    {
        "label": "Volatility 3%",
        "factory": lambda: VolatilityStaking(base_ratio=0.03, window=15, sensitivity=1.5),
        "description": "3% base, gentler volatility adjustment",
        "category": "dynamic",
    },
    {
        "label": "Portfolio Equal/5",
        "factory": lambda: PortfolioStaking(total_concurrent_bets=5),
        "description": "Equal split across 5 concurrent bets",
        "category": "portfolio",
    },
    {
        "label": "Portfolio Equal/3",
        "factory": lambda: PortfolioStaking(total_concurrent_bets=3),
        "description": "Equal split across 3 concurrent bets",
        "category": "portfolio",
    },
    {
        "label": "Portfolio EV-w/5",
        "factory": lambda: PortfolioStaking(
            total_concurrent_bets=5, allocation_method="ev_weighted",
        ),
        "description": "EV-weighted split across 5",
        "category": "portfolio",
    },
    {
        "label": "Full Kelly",
        "factory": lambda: KellyStaking(),
        "description": "Full Kelly Criterion (aggressive)",
        "category": "kelly",
    },
    {
        "label": "Kelly 50%",
        "factory": lambda: FractionalKellyStaking(fraction=0.50),
        "description": "50% Kelly — moderate",
        "category": "kelly",
    },
    {
        "label": "Kelly 25%",
        "factory": lambda: FractionalKellyStaking(fraction=0.25),
        "description": "25% Kelly — conservative (default)",
        "category": "kelly",
    },
    {
        "label": "Kelly 10%",
        "factory": lambda: FractionalKellyStaking(fraction=0.10),
        "description": "10% Kelly — very conservative",
        "category": "kelly",
    },
]


# ═══════════════════════════════════════════════════════════
#  2. RISK MANAGEMENT SCENARIOS
# ═══════════════════════════════════════════════════════════

RISK_SCENARIOS: list[dict[str, Any]] = [
    {
        "label": "No Risk Mgmt (Baseline)",
        "description": "No risk limits — pure staking",
        "config": {"risk_manager": {"enabled": False}},
    },
    {
        "label": "Daily Loss 10%",
        "description": "Stop at 10% daily loss",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": True, "max_loss_pct": 10.0},
            "drawdown": {"enabled": False},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": False},
            "stake": {"enabled": False},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Daily Loss 20%",
        "description": "Stop at 20% daily loss",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": True, "max_loss_pct": 20.0},
            "drawdown": {"enabled": False},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": False},
            "stake": {"enabled": False},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Drawdown 15%",
        "description": "Stop if 15% down from peak",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": False},
            "drawdown": {"enabled": True, "max_drawdown_pct": 15.0},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": False},
            "stake": {"enabled": False},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Drawdown 25%",
        "description": "Stop if 25% down from peak",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": False},
            "drawdown": {"enabled": True, "max_drawdown_pct": 25.0},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": False},
            "stake": {"enabled": False},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Freq: 5/day, 20/week",
        "description": "Max 5 bets/day, 20/week",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": False},
            "drawdown": {"enabled": False},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": True, "max_per_day": 5, "max_per_week": 20, "max_per_hour": 3},
            "stake": {"enabled": False},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Freq: 3/hr, 10/day",
        "description": "Max 3/hr, 10/day",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": False},
            "drawdown": {"enabled": False},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": True, "max_per_hour": 3, "max_per_day": 10, "max_per_week": 40},
            "stake": {"enabled": False},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Stake: max 5% single",
        "description": "Max 5% of bankroll per bet",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": False},
            "drawdown": {"enabled": False},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": False},
            "stake": {"enabled": True, "max_single_pct": 5.0, "min_odds": 1.5, "max_odds": 20.0},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Stake: max 20% single",
        "description": "Max 20% of bankroll per bet",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": False},
            "drawdown": {"enabled": False},
            "consecutive_losses": {"enabled": False},
            "frequency": {"enabled": False},
            "stake": {"enabled": True, "max_single_pct": 20.0, "min_odds": 1.0, "max_odds": 50.0},
            "diversification": {"enabled": False},
            "exposure": {"enabled": False},
        }},
    },
    {
        "label": "Conserv. (All Limits)",
        "description": "Tight daily loss + drawdown + freq",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": True, "max_loss_pct": 10.0},
            "drawdown": {"enabled": True, "max_drawdown_pct": 20.0},
            "consecutive_losses": {"enabled": True, "max_consecutive": 4},
            "frequency": {"enabled": True, "max_per_day": 8, "max_per_week": 30, "max_per_hour": 2},
            "stake": {"enabled": True, "max_single_pct": 10.0, "min_odds": 1.5, "max_odds": 15.0},
            "diversification": {"enabled": False},
            "exposure": {"enabled": True, "max_open_bets": 10},
        }},
    },
    {
        "label": "Moderate (All Limits)",
        "description": "Balanced risk limits",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": True, "max_loss_pct": 15.0},
            "drawdown": {"enabled": True, "max_drawdown_pct": 25.0},
            "consecutive_losses": {"enabled": True, "max_consecutive": 6},
            "frequency": {"enabled": True, "max_per_day": 10, "max_per_week": 40, "max_per_hour": 3},
            "stake": {"enabled": True, "max_single_pct": 25.0, "min_odds": 1.5, "max_odds": 20.0},
            "diversification": {"enabled": False},
            "exposure": {"enabled": True, "max_open_bets": 15},
        }},
    },
    {
        "label": "Aggressive (All Limits)",
        "description": "Loose risk limits",
        "config": {"risk_manager": {
            "enabled": True,
            "daily_loss": {"enabled": True, "max_loss_pct": 25.0},
            "drawdown": {"enabled": True, "max_drawdown_pct": 40.0},
            "consecutive_losses": {"enabled": True, "max_consecutive": 10},
            "frequency": {"enabled": True, "max_per_day": 20, "max_per_week": 80, "max_per_hour": 5},
            "stake": {"enabled": True, "max_single_pct": 50.0, "min_odds": 1.1, "max_odds": 50.0},
            "diversification": {"enabled": False},
            "exposure": {"enabled": True, "max_open_bets": 30},
        }},
    },
]


# ═══════════════════════════════════════════════════════════
#  MODEL CONFIG (use Ensemble — best Brier)
# ═══════════════════════════════════════════════════════════

MODEL_CONFIG = {
    "name": "Ensemble",
    "brier": 0.5775,
    "calibration": "n/a (weighted avg)",
}


# ═══════════════════════════════════════════════════════════
#  Data loading & prediction generation
# ═══════════════════════════════════════════════════════════


def load_test_data() -> pd.DataFrame:
    """Load the World Cup dataset and return the test set (last 20%)."""
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
#  Backtest runner
# ═══════════════════════════════════════════════════════════


def build_test_df(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    odds: np.ndarray,
) -> pd.DataFrame:
    """Build the DataFrame for the Backtester with model probs and odds."""
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


def _setup_volatility_staking(
    bt_df: pd.DataFrame,
    stake_strategy: VolatilityStaking,
) -> None:
    """Pre-seed volatility staking with CLV-like values from historical data."""
    # Simulate initial volatility by recording CLV estimates from the first few matches
    for idx, row in bt_df.head(stake_strategy.window).iterrows():
        home_prob = float(row.get("model_prob_home_win", 0.0))
        draw_prob = float(row.get("model_prob_draw", 0.0))
        away_prob = float(row.get("model_prob_away_win", 0.0))

        # Generate a synthetic CLV signal based on prediction confidence
        max_prob = max(home_prob, draw_prob, away_prob)
        clv_signal = (max_prob - 0.333) * 0.1  # Higher confidence = positive CLV signal
        stake_strategy.record_result(clv_signal)


def _setup_portfolio_staking(
    bt_df: pd.DataFrame,
    stake_strategy: PortfolioStaking,
) -> None:
    """Pre-seed portfolio staking with concurrent bet info for weighted allocation."""
    if stake_strategy.allocation_method == "equal":
        return  # equal method doesn't need weights

    # Build concurrent bet list from first N matches
    concurrent_bets = []
    for _, row in bt_df.head(stake_strategy.total_concurrent_bets).iterrows():
        outcomes = [
            ("home_win", float(row.get("model_prob_home_win", 0.0)), float(row.get("BbAvH", 0.0))),
            ("draw", float(row.get("model_prob_draw", 0.0)), float(row.get("BbAvD", 0.0))),
            ("away_win", float(row.get("model_prob_away_win", 0.0)), float(row.get("BbAvA", 0.0))),
        ]
        for outcome, prob, odds_val in outcomes:
            if prob > 0.0 and odds_val > 1.0:
                ev = (prob * odds_val) - 1.0
                kelly = max(ev / (odds_val - 1.0), 0.0) if odds_val > 1.0 else 0.0
                concurrent_bets.append({
                    "ev": ev,
                    "prob": prob,
                    "model_prob": prob,
                    "kelly_fraction": kelly,
                    "kelly": kelly,
                })
                break  # one bet per match

    if concurrent_bets:
        stake_strategy.set_concurrent_bets(concurrent_bets)


def run_single_backtest(
    bt_df: pd.DataFrame,
    stake_strategy: StakingStrategy,
    bet_filter: BetFilter = BASE_BET_FILTER,
) -> dict[str, Any]:
    """Run the backtester with a given stake strategy and filter.

    Returns a dict of flattened metrics.

    Note
    ----
    - ``VolatilityStaking`` is pre-seeded with historical CLV signals.
    - ``PortfolioStaking`` is pre-seeded with concurrent bet info for
      weighted allocation methods.
    """
    # Pre-setup for strategies that need external input
    if isinstance(stake_strategy, VolatilityStaking):
        _setup_volatility_staking(bt_df, stake_strategy)
    if isinstance(stake_strategy, PortfolioStaking):
        _setup_portfolio_staking(bt_df, stake_strategy)

    backtester = Backtester(
        model=None,
        initial_bankroll=INITIAL_BANKROLL,
        stake_strategy=stake_strategy,
        bet_filter=bet_filter,
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

    return {
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
        "avg_ev": round(metrics.avg_ev, 6),
        "avg_stake_pct": round(metrics.avg_stake_pct, 4),
        "longest_win_streak": metrics.longest_win_streak,
        "longest_lose_streak": metrics.longest_lose_streak,
        "final_bankroll": round(metrics.final_bankroll, 2),
        "initial_bankroll": metrics.initial_bankroll,
    }


def run_backtest_with_risk(
    bt_df: pd.DataFrame,
    stake_strategy: StakingStrategy,
    risk_config: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """Run the backtester with an external RiskManager for risk checks.

    Because the Backtester doesn't natively support RiskManager, we
    run the simulation manually, checking each bet against the risk
    manager before placing it.
    """
    # Create RiskManager with the given config override
    rm = RiskManager(auto_load=False)
    rm.load_config(config_override=risk_config)

    bankroll_obj = Bankroll(initial_balance=INITIAL_BANKROLL)
    bankroll = float(INITIAL_BANKROLL)
    peak = bankroll
    bets_placed: list[dict[str, Any]] = []
    bet_id_counter = 0
    n_rejected_by_risk = 0

    for match_idx, row in bt_df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        match_label = f"{home} vs {away}" if home and away else f"Match {match_idx}"
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

        # Check with RiskManager
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

        allowed, reason = rm.check_bet(slip, bankroll_obj)
        if not allowed:
            n_rejected_by_risk += 1
            logger.debug("RiskManager rejected bet: %s", reason)
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

        if bankroll > peak:
            peak = bankroll

        bet_id_counter += 1
        bets_placed.append({
            "match": match_label,
            "outcome": best_outcome,
            "odds": round(best_odds, 4),
            "prob": round(best_prob, 4),
            "ev": round(best_ev, 4),
            "stake": round(stake_amount, 2),
            "profit": round(profit, 2),
            "won": won,
        })

    # Compute metrics
    n_bets = len(bets_placed)
    n_wins = sum(1 for b in bets_placed if b["won"])
    total_staked = sum(b["stake"] for b in bets_placed)
    total_profit = sum(b["profit"] for b in bets_placed)
    roi_pct = (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    yield_pct = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (n_wins / n_bets * 100) if n_bets > 0 else 0.0

    return {
        "risk_label": label,
        "total_bets": n_bets,
        "winning_bets": n_wins,
        "losing_bets": n_bets - n_wins,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(roi_pct, 4),
        "yield_pct": round(yield_pct, 4),
        "win_rate_pct": round(win_rate, 2),
        "final_bankroll": round(bankroll, 2),
        "n_rejected_by_risk": n_rejected_by_risk,
    }


# ═══════════════════════════════════════════════════════════
#  Console output
# ═══════════════════════════════════════════════════════════


def print_strategy_leaderboard(results: list[dict[str, Any]]) -> None:
    """Print ranked leaderboard for stake strategy comparison."""
    print(f"\n{'=' * 100}")
    print("  STAKE STRATEGY COMPARISON — Ranked by Sharpe Ratio".center(98))
    print(f"{'=' * 100}")
    print()
    header = (
        f"  {'Rank':<6} {'Strategy':<22} {'Cat':<12} {'Bets':<6} "
        f"{'ROI%':<9} {'Yield%':<9} {'Win%':<7} {'Sharpe':<9} "
        f"{'DD%':<8} {'PF':<8} {'CLV':<9} {'Final':<10}"
    )
    print(header)
    print(f"  {'-' * 98}")

    ranked = sorted(results, key=lambda r: r.get("sharpe_ratio", -999), reverse=True)
    for i, r in enumerate(ranked, 1):
        label = r["label"]
        cat = r.get("category", "")
        bets = r.get("total_bets", 0)
        roi = r.get("roi_pct", 0)
        yld = r.get("yield_pct", 0)
        wr = r.get("win_rate_pct", 0)
        sharpe = r.get("sharpe_ratio", 0)
        dd = r.get("max_drawdown_pct", 0)
        pf = r.get("profit_factor", 0)
        clv = r.get("avg_clv", 0)
        final = r.get("final_bankroll", 0)
        rank_str = f"#{i}"

        print(
            f"  {rank_str:<6} {label:<22} {cat:<12} {bets:<6} "
            f"{roi:>+7.2f}% {yld:>+7.2f}% "
            f"{wr:<6.1f}% {sharpe:<8.4f} {dd:<7.2f}% "
            f"{pf:<7.4f} {clv:<9.6f} {final:>+8.2f}"
        )

    print(f"{'=' * 100}")
    if ranked:
        best = ranked[0]
        print(f"\n  Best Strategy: {best['label']} (Sharpe={best['sharpe_ratio']:.4f}, "
              f"ROI={best['roi_pct']:+.2f}%, "
              f"Final Bankroll={best['final_bankroll']:.2f})")
        print(f"  Description: {best.get('description', '')}")
    print()


def print_risk_leaderboard(results: list[dict[str, Any]]) -> None:
    """Print ranked leaderboard for risk scenario comparison."""
    print(f"\n{'=' * 100}")
    print("  RISK MANAGEMENT COMPARISON — Ranked by Sharpe Ratio".center(98))
    print(f"{'=' * 100}")
    print()
    header = (
        f"  {'Rank':<6} {'Scenario':<30} {'Bets':<6} "
        f"{'ROI%':<9} {'Yield%':<9} {'Win%':<7} "
        f"{'Final':<10} {'Desc':<30}"
    )
    print(header)
    print(f"  {'-' * 98}")

    ranked = sorted(results, key=lambda r: r.get("roi_pct", -999), reverse=True)
    for i, r in enumerate(ranked, 1):
        label = r.get("risk_label", r.get("label", "?"))
        bets = r.get("total_bets", 0)
        roi = r.get("roi_pct", 0)
        yld = r.get("yield_pct", 0)
        wr = r.get("win_rate_pct", 0)
        final = r.get("final_bankroll", 0)
        desc = r.get("description", "")
        rank_str = f"#{i}"

        print(
            f"  {rank_str:<6} {label:<30} {bets:<6} "
            f"{roi:>+7.2f}% {yld:>+7.2f}% "
            f"{wr:<6.1f}% "
            f"{final:>+8.2f} {desc:<30}"
        )

    print(f"{'=' * 100}")
    print()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main() -> None:
    logger.info("Loading test data...")
    test_df = load_test_data()
    y_true = test_df["result_idx"].values

    # Generate predictions for the model
    model_seed = 42
    probs = generate_predictions(
        y_true, 3, MODEL_CONFIG["brier"], seed=model_seed,
    )

    # Generate consensus odds
    logger.info("Generating consensus odds...")
    consensus_odds = generate_consensus_odds(y_true, seed=9999)

    # Build test DataFrame
    bt_df = build_test_df(test_df, y_true, probs, consensus_odds)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ══════════════════════════════════════════════════
    #  MODE 1: Compare all stake strategies
    # ══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("MODE 1: Comparing %d stake strategies...", len(STAKE_STRATEGIES))
    logger.info("=" * 60)

    stake_results: list[dict[str, Any]] = []
    for i, cfg in enumerate(STAKE_STRATEGIES):
        label = cfg["label"]
        logger.info(
            "[%d/%d] Backtesting %s...",
            i + 1, len(STAKE_STRATEGIES), label,
        )

        try:
            strategy = cfg["factory"]()
            metrics = run_single_backtest(bt_df, strategy)
            metrics["label"] = label
            metrics["description"] = cfg["description"]
            metrics["category"] = cfg["category"]
            stake_results.append(metrics)

            logger.info(
                "  %s: %d bets, ROI=%+.2f%%, Sharpe=%.4f, Final=%.2f",
                label,
                metrics["total_bets"],
                metrics["roi_pct"],
                metrics["sharpe_ratio"],
                metrics["final_bankroll"],
            )
        except Exception as exc:
            logger.error("  Failed for %s: %s\n%s", label, exc, traceback.format_exc())

    # ══════════════════════════════════════════════════
    #  MODE 2: Compare risk management scenarios
    # ══════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("MODE 2: Comparing %d risk scenarios...", len(RISK_SCENARIOS))
    logger.info("=" * 60)

    # Use Kelly 25% as the baseline stake strategy for risk comparisons
    baseline_strategy = FractionalKellyStaking(fraction=0.25)

    risk_results: list[dict[str, Any]] = []
    for i, scenario in enumerate(RISK_SCENARIOS):
        label = scenario["label"]
        logger.info(
            "[%d/%d] Risk scenario: %s...",
            i + 1, len(RISK_SCENARIOS), label,
        )

        try:
            metrics = run_backtest_with_risk(
                bt_df, baseline_strategy, scenario["config"], label,
            )
            metrics["description"] = scenario["description"]
            risk_results.append(metrics)

            logger.info(
                "  %s: %d bets, ROI=%+.2f%%, Final=%.2f",
                label,
                metrics["total_bets"],
                metrics["roi_pct"],
                metrics["final_bankroll"],
            )
        except Exception as exc:
            logger.error("  Failed for %s: %s\n%s", label, exc, traceback.format_exc())

    # ══════════════════════════════════════════════════
    #  Save results
    # ══════════════════════════════════════════════════
    output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        },
        "stake_strategies": {
            "count": len(stake_results),
            "results": stake_results,
            "leaderboard": [
                {
                    "rank": i + 1,
                    "label": r["label"],
                    "sharpe_ratio": r.get("sharpe_ratio", 0),
                    "roi_pct": r.get("roi_pct", 0),
                    "final_bankroll": r.get("final_bankroll", 0),
                    "max_drawdown_pct": r.get("max_drawdown_pct", 0),
                }
                for i, r in enumerate(
                    sorted(
                        stake_results,
                        key=lambda x: x.get("sharpe_ratio", -999),
                        reverse=True,
                    )
                )
            ],
        },
        "risk_scenarios": {
            "count": len(risk_results),
            "results": risk_results,
            "leaderboard": [
                {
                    "rank": i + 1,
                    "label": r.get("risk_label", r.get("label", "?")),
                    "roi_pct": r.get("roi_pct", 0),
                    "final_bankroll": r.get("final_bankroll", 0),
                }
                for i, r in enumerate(
                    sorted(
                        risk_results,
                        key=lambda x: x.get("roi_pct", -999),
                        reverse=True,
                    )
                )
            ],
        },
    }

    filename = f"bankroll_management_{timestamp}.json"
    filepath = REPORTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Full results saved to %s", filepath)

    # ══════════════════════════════════════════════════
    #  Print leaderboards
    # ══════════════════════════════════════════════════
    print_strategy_leaderboard(stake_results)
    print_risk_leaderboard(risk_results)

    # ══════════════════════════════════════════════════
    #  Best overall recommendations
    # ══════════════════════════════════════════════════
    if stake_results:
        best_stake = max(stake_results, key=lambda r: r.get("sharpe_ratio", -999))
        safest_stake = min(
            [r for r in stake_results if r.get("total_bets", 0) > 0],
            key=lambda r: r.get("max_drawdown_pct", 999),
        )
        best_roi = max(stake_results, key=lambda r: r.get("roi_pct", -999))
        print(f"\n{'=' * 60}")
        print("  RECOMMENDATIONS")
        print(f"{'=' * 60}")
        print(f"  Best Stake Strategy (by Sharpe):  {best_stake['label']}")
        print(f"    Sharpe={best_stake['sharpe_ratio']:.4f}, "
              f"ROI={best_stake['roi_pct']:+.2f}%, "
              f"DD={best_stake['max_drawdown_pct']:.1f}%")
        print(f"  Safest Strategy (lowest drawdown): {safest_stake['label']}")
        print(f"    DD={safest_stake['max_drawdown_pct']:.1f}%, "
              f"ROI={safest_stake['roi_pct']:+.2f}%")
        print(f"  Best ROI: {best_roi['label']} "
              f"({best_roi['roi_pct']:+.2f}%)")

    if risk_results:
        best_risk = max(risk_results, key=lambda r: r.get("roi_pct", -999))
        print(f"  Best Risk Scenario (by ROI): {best_risk.get('risk_label', '?')}")
        print(f"    ROI={best_risk['roi_pct']:+.2f}%, "
              f"Final={best_risk['final_bankroll']:.2f}")

    print()
    logger.info("Done! All strategies and scenarios backtested.")


if __name__ == "__main__":
    main()
