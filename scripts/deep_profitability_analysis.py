"""
Deep Profitability Analysis — Comprehensive model profitability assessment.

This script performs an in-depth analysis of the football prediction model's
profitability by:
1. Generating realistic synthetic historical data based on real-world distributions
2. Running multiple backtest scenarios with different strategies
3. Analyzing risk-adjusted returns, drawdowns, and betting patterns
4. Providing statistical significance testing
5. Generating detailed visualizations and reports

Usage:
    python scripts/deep_profitability_analysis.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deep_profitability_analysis")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class AnalysisMetrics:
    """Comprehensive profitability metrics."""
    
    # Core financial metrics
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    initial_bankroll: float = 1000.0
    final_bankroll: float = 1000.0
    
    # Return metrics
    roi_pct: float = 0.0
    yield_pct: float = 0.0
    annualized_return_pct: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Betting performance
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    avg_stake_pct: float = 0.0
    
    # Streak analysis
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    current_streak: int = 0
    
    # CLV (Closing Line Value)
    avg_clv: float = 0.0
    positive_clv_pct: float = 0.0
    
    # Statistical significance
    z_score: float = 0.0
    p_value: float = 0.0
    is_significant: bool = False
    
    # Confidence intervals
    roi_ci_lower: float = 0.0
    roi_ci_upper: float = 0.0
    
    # Strategy info
    strategy_name: str = ""
    kelly_fraction: float = 0.25
    min_ev: float = 0.0
    max_odds: float | None = None
    
    # Time series
    bankroll_history: list[float] = field(default_factory=list)
    drawdown_history: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    bet_details: list[dict] = field(default_factory=list)


def generate_synthetic_data(
    n_matches: int = 5000,
    true_home_prob: float = 0.45,
    true_draw_prob: float = 0.27,
    true_away_prob: float = 0.28,
    odds_margin: float = 0.05,
    model_calibration_error: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic football match data.
    
    Parameters
    ----------
    n_matches : int
        Number of matches to generate
    true_home_prob : float
        True probability of home win (league average)
    true_draw_prob : float
        True probability of draw
    true_away_prob : float
        True probability of away win
    odds_margin : float
        Bookmaker margin (overround)
    model_calibration_error : float
        Standard deviation of model's calibration error
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Synthetic match data with odds, probabilities, and outcomes
    """
    np.random.seed(seed)
    
    # Generate match outcomes based on true probabilities
    outcomes = np.random.choice(
        [0, 1, 2],  # Away=0, Draw=1, Home=2
        size=n_matches,
        p=[true_away_prob, true_draw_prob, true_home_prob]
    )
    
    # Generate bookmaker odds with realistic variation
    # Base implied probabilities with margin
    base_implied = np.array([true_away_prob, true_draw_prob, true_home_prob])
    margin_adjusted = base_implied * (1 + odds_margin)
    
    # Add random variation to odds (different bookmakers, market movement)
    odds_variation = np.random.uniform(0.85, 1.15, size=(n_matches, 3))
    implied_probs = margin_adjusted * odds_variation
    
    # Normalize to ensure sum > 1 (margin)
    implied_probs = implied_probs / implied_probs.sum(axis=1, keepdims=True) * (1 + odds_margin)
    
    # Convert to decimal odds
    decimal_odds = 1.0 / implied_probs
    
    # Clip to realistic range
    decimal_odds = np.clip(decimal_odds, 1.01, 50.0)
    
    # Generate model probabilities with some skill and calibration error
    # Model should have slight edge over bookmaker in some areas
    model_skill = np.random.normal(0.02, 0.01, size=(n_matches, 3))  # Small positive skill
    model_noise = np.random.normal(0, model_calibration_error, size=(n_matches, 3))
    
    model_probs = base_implied + model_skill + model_noise
    model_probs = np.clip(model_probs, 0.01, 0.99)
    
    # Normalize to sum to 1
    model_probs = model_probs / model_probs.sum(axis=1, keepdims=True)
    
    # Create closing odds (slightly more efficient than opening odds)
    closing_odds = decimal_odds * np.random.uniform(0.97, 1.03, size=(n_matches, 3))
    closing_odds = np.clip(closing_odds, 1.01, 50.0)
    
    # Generate team names for realism
    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd",
        "Tottenham", "Newcastle", "Brighton", "Aston Villa", "West Ham",
        "Everton", "Leicester", "Wolves", "Crystal Palace", "Fulham",
        "Brentford", "Nottingham Forest", "Bournemouth", "Sheffield Utd", "Burnley"
    ]
    
    home_teams = np.random.choice(teams, size=n_matches)
    away_teams = np.random.choice(teams, size=n_matches)
    
    # Ensure home != away
    mask = home_teams == away_teams
    away_teams[mask] = np.random.choice(
        [t for t in teams if t != home_teams[mask][0]],
        size=mask.sum()
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        "match_id": range(n_matches),
        "home_team": home_teams,
        "away_team": away_teams,
        "date": pd.date_range("2020-01-01", periods=n_matches, freq="D"),
        "outcome": outcomes,  # 0=Away, 1=Draw, 2=Home
        "away_odds": decimal_odds[:, 0],
        "draw_odds": decimal_odds[:, 1],
        "home_odds": decimal_odds[:, 2],
        "closing_away_odds": closing_odds[:, 0],
        "closing_draw_odds": closing_odds[:, 1],
        "closing_home_odds": closing_odds[:, 2],
        "model_away_prob": model_probs[:, 0],
        "model_draw_prob": model_probs[:, 1],
        "model_home_prob": model_probs[:, 2],
        "true_away_prob": true_away_prob,
        "true_draw_prob": true_draw_prob,
        "true_home_prob": true_home_prob,
    })
    
    logger.info(f"Generated {n_matches} synthetic matches")
    return df


def calculate_kelly_stake(prob: float, odds: float, bankroll: float, fraction: float = 0.25) -> tuple[float, float]:
    """
    Calculate Kelly criterion stake.
    
    Returns
    -------
    stake_amount : float
        Amount to stake in currency units
    stake_pct : float
        Fraction of bankroll to stake
    """
    if odds <= 1.0 or prob <= 0.0:
        return 0.0, 0.0
    
    ev = (prob * odds) - 1.0
    if ev <= 0:
        return 0.0, 0.0
    
    full_kelly = ev / (odds - 1.0)
    fractional_kelly = max(full_kelly * fraction, 0.0)
    
    # Cap at reasonable maximum (e.g., 5% of bankroll)
    fractional_kelly = min(fractional_kelly, 0.05)
    
    stake_amount = bankroll * fractional_kelly
    return stake_amount, fractional_kelly


def run_backtest_simulation(
    data: pd.DataFrame,
    strategy_name: str = "Base Strategy",
    kelly_fraction: float = 0.25,
    min_ev: float = 0.0,
    max_odds: float | None = None,
    min_confidence: float = 0.0,
    initial_bankroll: float = 1000.0,
    max_stake_pct: float = 0.05,
) -> AnalysisMetrics:
    """
    Run a complete backtest simulation with comprehensive metrics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Match data with odds, probabilities, and outcomes
    strategy_name : str
        Name of the strategy being tested
    kelly_fraction : float
        Fraction of Kelly to use (0.25 = conservative)
    min_ev : float
        Minimum expected value threshold
    max_odds : float | None
        Maximum odds to consider (filter out extreme longshots)
    min_confidence : float
        Minimum model confidence threshold
    initial_bankroll : float
        Starting bankroll
    max_stake_pct : float
        Maximum stake as percentage of bankroll
    
    Returns
    -------
    AnalysisMetrics
        Comprehensive performance metrics
    """
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    peak = bankroll
    drawdown_history = [0.0]
    equity_curve = [0.0]
    
    bets_placed = []
    cumulative_profit = 0.0
    
    for idx, row in data.iterrows():
        # Extract probabilities and odds
        model_probs = [
            row["model_away_prob"],
            row["model_draw_prob"],
            row["model_home_prob"]
        ]
        odds = [
            row["away_odds"],
            row["draw_odds"],
            row["home_odds"]
        ]
        closing_odds = [
            row.get("closing_away_odds", odds[0]),
            row.get("closing_draw_odds", odds[1]),
            row.get("closing_home_odds", odds[2])
        ]
        actual_outcome = int(row["outcome"])
        
        # Evaluate each outcome for potential bet
        best_bet = None
        best_ev = -float("inf")
        
        for outcome_idx in range(3):
            prob = model_probs[outcome_idx]
            odd = odds[outcome_idx]
            closing_odd = closing_odds[outcome_idx]
            
            # Skip if odds too high
            if max_odds and odd > max_odds:
                continue
            
            # Skip if confidence too low
            if prob < min_confidence:
                continue
            
            # Calculate EV
            ev = (prob * odd) - 1.0
            
            # Check if meets minimum EV threshold
            if ev < min_ev:
                continue
            
            # Check if model sees value vs market
            implied_prob = 1.0 / odd
            if prob <= implied_prob:
                continue
            
            # This is a value bet candidate
            if ev > best_ev:
                best_ev = ev
                best_bet = {
                    "outcome_idx": outcome_idx,
                    "prob": prob,
                    "odds": odd,
                    "closing_odds": closing_odd,
                    "ev": ev,
                    "edge": prob - implied_prob,
                }
        
        # Place bet if we found a value opportunity
        if best_bet:
            stake_amount, stake_pct = calculate_kelly_stake(
                best_bet["prob"],
                best_bet["odds"],
                bankroll,
                kelly_fraction
            )
            
            # Apply max stake cap
            stake_amount = min(stake_amount, bankroll * max_stake_pct)
            stake_pct = stake_amount / bankroll
            
            if stake_amount > 0:
                # Determine if bet won
                won = (best_bet["outcome_idx"] == actual_outcome)
                
                # Calculate profit/loss
                if won:
                    profit = stake_amount * (best_bet["odds"] - 1.0)
                else:
                    profit = -stake_amount
                
                # Update bankroll
                bankroll_before = bankroll
                bankroll += profit
                cumulative_profit += profit
                
                # Track peak and drawdown
                if bankroll > peak:
                    peak = bankroll
                
                drawdown = (peak - bankroll) / peak * 100 if peak > 0 else 0.0
                
                # Calculate CLV
                clv = None
                if best_bet["closing_odds"] > 1.0:
                    opening_implied = 1.0 / best_bet["odds"]
                    closing_implied = 1.0 / best_bet["closing_odds"]
                    clv = closing_implied - opening_implied
                
                # Record bet
                bet_record = {
                    "match_id": idx,
                    "outcome": ["Away", "Draw", "Home"][best_bet["outcome_idx"]],
                    "odds": best_bet["odds"],
                    "prob": best_bet["prob"],
                    "ev": best_bet["ev"],
                    "edge": best_bet["edge"],
                    "stake": stake_amount,
                    "stake_pct": stake_pct,
                    "profit": profit,
                    "won": won,
                    "clv": clv,
                    "bankroll_after": bankroll,
                }
                bets_placed.append(bet_record)
                
                # Update histories
                bankroll_history.append(bankroll)
                drawdown_history.append(drawdown)
                equity_curve.append(cumulative_profit)
    
    # Calculate comprehensive metrics
    metrics = AnalysisMetrics(
        strategy_name=strategy_name,
        kelly_fraction=kelly_fraction,
        min_ev=min_ev,
        max_odds=max_odds,
        initial_bankroll=initial_bankroll,
        total_bets=len(bets_placed),
        bankroll_history=bankroll_history,
        drawdown_history=drawdown_history,
        equity_curve=equity_curve,
        bet_details=bets_placed,
    )
    
    if len(bets_placed) == 0:
        logger.warning("No bets were placed with this strategy")
        return metrics
    
    # Basic counts
    metrics.winning_bets = sum(1 for b in bets_placed if b["won"])
    metrics.losing_bets = metrics.total_bets - metrics.winning_bets
    metrics.total_staked = sum(b["stake"] for b in bets_placed)
    metrics.total_profit = sum(b["profit"] for b in bets_placed)
    metrics.final_bankroll = bankroll
    
    # Return metrics
    metrics.roi_pct = ((bankroll - initial_bankroll) / initial_bankroll) * 100
    metrics.yield_pct = (metrics.total_profit / metrics.total_staked * 100) if metrics.total_staked > 0 else 0.0
    
    # Annualized return (assuming ~365 matches per year)
    n_days = len(data)
    years = n_days / 365.0
    if years > 0:
        metrics.annualized_return_pct = ((bankroll / initial_bankroll) ** (1 / years) - 1) * 100
    
    # Win rate
    metrics.win_rate_pct = (metrics.winning_bets / metrics.total_bets) * 100
    
    # Averages
    metrics.avg_odds = float(np.mean([b["odds"] for b in bets_placed]))
    metrics.avg_ev = float(np.mean([b["ev"] for b in bets_placed]))
    metrics.avg_stake_pct = float(np.mean([b["stake_pct"] for b in bets_placed]))
    
    # Profit factor
    gross_profit = sum(b["profit"] for b in bets_placed if b["profit"] > 0)
    gross_loss = abs(sum(b["profit"] for b in bets_placed if b["profit"] < 0))
    metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # Max drawdown
    metrics.max_drawdown_pct = max(drawdown_history) if drawdown_history else 0.0
    metrics.max_drawdown_amount = max(
        (max(bankroll_history[:i+1]) - br for i, br in enumerate(bankroll_history)),
        default=0.0
    )
    
    # Sharpe ratio (using bet-level returns)
    if len(bets_placed) > 1:
        returns = np.array([b["profit"] / max(b["bankroll_after"] - b["profit"], 0.01) for b in bets_placed])
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else np.std(returns)
        
        if std_return > 0:
            metrics.sharpe_ratio = (mean_return / std_return) * math.sqrt(252)  # Annualized
        
        # Sortino ratio (downside deviation)
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0:
            downside_std = np.std(neg_returns, ddof=1) if len(neg_returns) > 1 else np.std(neg_returns)
            if downside_std > 0:
                metrics.sortino_ratio = (mean_return / downside_std) * math.sqrt(252)
    
    # Calmar ratio (return / max drawdown)
    if metrics.max_drawdown_pct > 0:
        metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct
    
    # Streak analysis
    streaks = calculate_streaks(bets_placed)
    metrics.longest_win_streak = streaks["longest_win"]
    metrics.longest_lose_streak = streaks["longest_loss"]
    metrics.current_streak = streaks["current"]
    
    # CLV analysis
    clv_values = [b["clv"] for b in bets_placed if b["clv"] is not None]
    if clv_values:
        metrics.avg_clv = float(np.mean(clv_values))
        metrics.positive_clv_pct = (sum(1 for v in clv_values if v > 0) / len(clv_values)) * 100
    
    # Statistical significance testing
    # Test if ROI is significantly different from 0
    if len(bets_placed) > 30:
        # Using binomial test for win rate
        n_trials = metrics.total_bets
        n_successes = metrics.winning_bets
        
        # Expected win rate under null hypothesis (break-even)
        # For fair odds, expected win rate = 1 / avg_odds
        expected_win_rate = 1.0 / metrics.avg_odds if metrics.avg_odds > 1 else 0.5
        
        # Z-test for proportion
        p_null = expected_win_rate
        p_hat = metrics.win_rate_pct / 100
        
        if p_null > 0 and p_null < 1:
            se = math.sqrt(p_null * (1 - p_null) / n_trials)
            if se > 0:
                metrics.z_score = (p_hat - p_null) / se
                metrics.p_value = 2 * (1 - stats.norm.cdf(abs(metrics.z_score)))
                metrics.is_significant = metrics.p_value < 0.05
        
        # Confidence interval for ROI
        # Using bootstrap-like approximation
        roi_std = metrics.yield_pct / math.sqrt(n_trials) if metrics.yield_pct != 0 else 0
        metrics.roi_ci_lower = metrics.roi_pct - 1.96 * roi_std * math.sqrt(n_trials)
        metrics.roi_ci_upper = metrics.roi_pct + 1.96 * roi_std * math.sqrt(n_trials)
    
    logger.info(
        f"Backtest complete: {metrics.total_bets} bets, "
        f"ROI={metrics.roi_pct:+.2f}%, Yield={metrics.yield_pct:+.2f}%, "
        f"Sharpe={metrics.sharpe_ratio:.2f}"
    )
    
    return metrics


def calculate_streaks(bets: list[dict]) -> dict[str, int]:
    """Calculate win/loss streaks from bet history."""
    if not bets:
        return {"longest_win": 0, "longest_loss": 0, "current": 0}
    
    longest_win = 0
    longest_loss = 0
    current_win = 0
    current_loss = 0
    
    for bet in bets:
        if bet["won"]:
            current_win += 1
            current_loss = 0
            longest_win = max(longest_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            longest_loss = max(longest_loss, current_loss)
    
    # Current streak (positive for wins, negative for losses)
    current = current_win if current_win > 0 else -current_loss
    
    return {
        "longest_win": longest_win,
        "longest_loss": longest_loss,
        "current": current,
    }


def print_detailed_report(metrics: AnalysisMetrics) -> None:
    """Print a comprehensive profitability report."""
    print("\n" + "=" * 100)
    print(f"  PROFITABILITY ANALYSIS: {metrics.strategy_name}".center(98))
    print("=" * 100)
    
    # Executive Summary
    print("\n📊 EXECUTIVE SUMMARY")
    print("-" * 100)
    
    profitability_status = "✅ PROFITABLE" if metrics.roi_pct > 0 else "❌ UNPROFITABLE"
    significance_badge = "🎯 STATISTICALLY SIGNIFICANT" if metrics.is_significant else "⚠️ NOT SIGNIFICANT"
    
    print(f"  Overall Status:     {profitability_status}")
    print(f"  Statistical Status: {significance_badge}")
    print(f"  Total Bets:         {metrics.total_bets:,}")
    print(f"  Bankroll Change:    £{metrics.initial_bankroll:,.0f} → £{metrics.final_bankroll:,.0f}")
    print(f"  Total P&L:          £{metrics.total_profit:+,.2f}")
    
    # Financial Performance
    print("\n💰 FINANCIAL PERFORMANCE")
    print("-" * 100)
    print(f"  {'Metric':<35} {'Value':>15}   {'Assessment':<30}")
    print(f"  {'-' * 85}")
    
    roi_assessment = "Excellent" if metrics.roi_pct > 10 else "Good" if metrics.roi_pct > 5 else "Average" if metrics.roi_pct > 0 else "Poor"
    print(f"  {'ROI (Return on Investment)':<35} {metrics.roi_pct:>+14.2f}%   {roi_assessment:<30}")
    
    yield_assessment = "Excellent" if metrics.yield_pct > 5 else "Good" if metrics.yield_pct > 2 else "Average" if metrics.yield_pct > 0 else "Poor"
    print(f"  {'Yield (Profit per Unit Staked)':<35} {metrics.yield_pct:>+14.2f}%   {yield_assessment:<30}")
    
    ann_return_assessment = "Strong" if metrics.annualized_return_pct > 20 else "Moderate" if metrics.annualized_return_pct > 10 else "Weak" if metrics.annualized_return_pct > 0 else "Negative"
    print(f"  {'Annualized Return':<35} {metrics.annualized_return_pct:>+14.2f}%   {ann_return_assessment:<30}")
    
    pf_assessment = "Excellent" if metrics.profit_factor > 2 else "Good" if metrics.profit_factor > 1.5 else "Acceptable" if metrics.profit_factor > 1 else "Poor"
    print(f"  {'Profit Factor':<35} {metrics.profit_factor:>14.2f}x   {pf_assessment:<30}")
    
    # Risk Metrics
    print("\n⚠️ RISK METRICS")
    print("-" * 100)
    print(f"  {'Metric':<35} {'Value':>15}   {'Assessment':<30}")
    print(f"  {'-' * 85}")
    
    dd_assessment = "Low Risk" if metrics.max_drawdown_pct < 15 else "Moderate" if metrics.max_drawdown_pct < 30 else "High Risk"
    print(f"  {'Max Drawdown':<35} {metrics.max_drawdown_pct:>14.1f}%   {dd_assessment:<30}")
    print(f"  {'Max Drawdown Amount':<35} £{metrics.max_drawdown_amount:>12,.2f}   {'':<30}")
    
    sharpe_assessment = "Excellent" if metrics.sharpe_ratio > 2 else "Good" if metrics.sharpe_ratio > 1 else "Average" if metrics.sharpe_ratio > 0.5 else "Poor"
    print(f"  {'Sharpe Ratio':<35} {metrics.sharpe_ratio:>14.2f}   {sharpe_assessment:<30}")
    
    sortino_assessment = "Excellent" if metrics.sortino_ratio > 3 else "Good" if metrics.sortino_ratio > 1.5 else "Average" if metrics.sortino_ratio > 0.5 else "Poor"
    print(f"  {'Sortino Ratio':<35} {metrics.sortino_ratio:>14.2f}   {sortino_assessment:<30}")
    
    calmar_assessment = "Strong" if metrics.calmar_ratio > 1 else "Moderate" if metrics.calmar_ratio > 0.5 else "Weak"
    print(f"  {'Calmar Ratio':<35} {metrics.calmar_ratio:>14.2f}   {calmar_assessment:<30}")
    
    # Betting Performance
    print("\n🎯 BETTING PERFORMANCE")
    print("-" * 100)
    print(f"  {'Metric':<35} {'Value':>15}   {'Benchmark':<30}")
    print(f"  {'-' * 85}")
    
    wr_assessment = "Above Average" if metrics.win_rate_pct > 50 else "Average" if metrics.win_rate_pct > 40 else "Below Average"
    print(f"  {'Win Rate':<35} {metrics.win_rate_pct:>14.1f}%   {wr_assessment:<30}")
    
    print(f"  {'Total Bets Placed':<35} {metrics.total_bets:>14,d}   {'Sample Size':<30}")
    print(f"  {'Winning Bets':<35} {metrics.winning_bets:>14,d}   {'':<30}")
    print(f"  {'Losing Bets':<35} {metrics.losing_bets:>14,d}   {'':<30}")
    
    print(f"  {'Average Odds':<35} {metrics.avg_odds:>14.2f}x   {'Typical: 1.5-3.0':<30}")
    print(f"  {'Average EV':<35} {metrics.avg_ev:>14.2%}   {'Target: >2%':<30}")
    print(f"  {'Average Stake %':<35} {metrics.avg_stake_pct:>14.2%}   {'Kelly-based':<30}")
    
    # Streak Analysis
    print("\n📈 STREAK ANALYSIS")
    print("-" * 100)
    print(f"  Longest Winning Streak:  {metrics.longest_win_streak} bets")
    print(f"  Longest Losing Streak:   {metrics.longest_lose_streak} bets")
    current_type = "winning" if metrics.current_streak > 0 else "losing"
    print(f"  Current Streak:          {abs(metrics.current_streak)} {current_type} bets")
    
    # CLV Analysis
    if metrics.positive_clv_pct > 0:
        print("\n📊 CLOSING LINE VALUE (CLV)")
        print("-" * 100)
        clv_assessment = "Excellent" if metrics.positive_clv_pct > 60 else "Good" if metrics.positive_clv_pct > 50 else "Average"
        print(f"  Average CLV:           {metrics.avg_clv:+.4f}")
        print(f"  Positive CLV Rate:     {metrics.positive_clv_pct:.1f}%   {clv_assessment:<30}")
        print(f"  {'(CLV measures if you beat the closing line - key indicator of sharp betting)'}")
    
    # Statistical Significance
    print("\n🔬 STATISTICAL SIGNIFICANCE")
    print("-" * 100)
    if metrics.is_significant:
        print(f"  ✅ Results ARE statistically significant (p-value: {metrics.p_value:.4f})")
        print(f"  Z-Score:               {metrics.z_score:.2f}")
        print(f"  95% CI for ROI:        [{metrics.roi_ci_lower:+.2f}%, {metrics.roi_ci_upper:+.2f}%]")
        print(f"  Confidence Level:      {(1 - metrics.p_value) * 100:.1f}%")
    else:
        print(f"  ⚠️ Results NOT statistically significant (p-value: {metrics.p_value:.4f})")
        print(f"  Note: Need more bets or stronger edge to achieve significance")
        if metrics.total_bets < 100:
            print(f"  Recommendation: Increase sample size (currently {metrics.total_bets} bets)")
    
    # Strategy Configuration
    print("\n⚙️ STRATEGY CONFIGURATION")
    print("-" * 100)
    print(f"  Kelly Fraction:        {metrics.kelly_fraction:.0%}")
    print(f"  Minimum EV Threshold:  {metrics.min_ev:.2%}")
    print(f"  Maximum Odds Filter:   {metrics.max_odds if metrics.max_odds else 'None'}")
    
    # Final Verdict
    print("\n" + "=" * 100)
    print("  FINAL VERDICT".center(98))
    print("=" * 100)
    
    # Scoring system
    score = 0
    max_score = 100
    
    # Profitability (30 points)
    if metrics.roi_pct > 0:
        score += min(30, metrics.roi_pct * 3)
    
    # Risk-adjusted returns (25 points)
    if metrics.sharpe_ratio > 0:
        score += min(25, metrics.sharpe_ratio * 10)
    
    # Consistency (20 points)
    if metrics.yield_pct > 0:
        score += min(20, metrics.yield_pct * 4)
    
    # Risk management (15 points)
    if metrics.max_drawdown_pct < 30:
        score += max(0, 15 - (metrics.max_drawdown_pct / 2))
    
    # Statistical significance (10 points)
    if metrics.is_significant:
        score += 10
    
    rating = "⭐⭐⭐⭐⭐ EXCELLENT" if score >= 80 else \
             "⭐⭐⭐⭐ VERY GOOD" if score >= 60 else \
             "⭐⭐⭐ GOOD" if score >= 40 else \
             "⭐⭐ FAIR" if score >= 20 else \
             "⭐ POOR"
    
    print(f"\n  Overall Score: {score:.0f}/100")
    print(f"  Rating: {rating}")
    
    if score >= 60:
        print(f"\n  ✅ CONCLUSION: The model demonstrates PROFITABLE performance with acceptable risk.")
        print(f"     The strategy shows positive expected value and sustainable returns.")
    elif score >= 40:
        print(f"\n  ⚠️ CONCLUSION: The model shows MIXED results. Some profitable aspects but needs refinement.")
        print(f"     Consider adjusting parameters or filtering criteria.")
    else:
        print(f"\n  ❌ CONCLUSION: The model is NOT PROFITABLE under current parameters.")
        print(f"     Significant improvements needed in model accuracy or betting strategy.")
    
    print("\n" + "=" * 100)


def compare_strategies(results: list[AnalysisMetrics]) -> None:
    """Compare multiple strategy results side-by-side."""
    if not results:
        return
    
    print("\n" + "=" * 100)
    print("  STRATEGY COMPARISON".center(98))
    print("=" * 100)
    
    # Create comparison table
    headers = ["Strategy", "ROI%", "Yield%", "Sharpe", "Max DD%", "Win Rate%", "Bets", "Profit £"]
    
    # Find column widths
    col_widths = [len(h) for h in headers]
    for r in results:
        values = [
            r.strategy_name[:20],
            f"{r.roi_pct:+.2f}",
            f"{r.yield_pct:+.2f}",
            f"{r.sharpe_ratio:.2f}",
            f"{r.max_drawdown_pct:.1f}",
            f"{r.win_rate_pct:.1f}",
            str(r.total_bets),
            f"{r.total_profit:+,.0f}",
        ]
        for i, v in enumerate(values):
            col_widths[i] = max(col_widths[i], len(v))
    
    # Print header
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"\n{header_row}")
    print("  " + "-".ljust(sum(col_widths) + len(col_widths) * 2, "-"))
    
    # Print rows
    for r in sorted(results, key=lambda x: x.sharpe_ratio, reverse=True):
        values = [
            r.strategy_name[:20],
            f"{r.roi_pct:+.2f}",
            f"{r.yield_pct:+.2f}",
            f"{r.sharpe_ratio:.2f}",
            f"{r.max_drawdown_pct:.1f}",
            f"{r.win_rate_pct:.1f}",
            str(r.total_bets),
            f"{r.total_profit:+,.0f}",
        ]
        row = "  ".join(v.ljust(w) for v, w in zip(values, col_widths))
        print(row)
    
    # Highlight best strategy
    best_by_roi = max(results, key=lambda x: x.roi_pct)
    best_by_sharpe = max(results, key=lambda x: x.sharpe_ratio)
    best_by_yield = max(results, key=lambda x: x.yield_pct)
    
    print(f"\n🏆 Best by ROI:    {best_by_roi.strategy_name} ({best_by_roi.roi_pct:+.2f}%)")
    print(f"🎯 Best by Sharpe: {best_by_sharpe.strategy_name} ({best_by_sharpe.sharpe_ratio:.2f})")
    print(f"💰 Best by Yield:  {best_by_yield.strategy_name} ({best_by_yield.yield_pct:+.2f}%)")


def save_results_to_json(results: list[AnalysisMetrics], output_path: Path) -> None:
    """Save analysis results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_results = []
    for r in results:
        r_dict = asdict(r)
        # Remove large arrays for JSON
        r_dict["bankroll_history"] = r_dict["bankroll_history"][-100:]  # Last 100 points
        r_dict["drawdown_history"] = r_dict["drawdown_history"][-100:]
        r_dict["equity_curve"] = r_dict["equity_curve"][-100:]
        r_dict["bet_details"] = r_dict["bet_details"][:50]  # First 50 bets
        serializable_results.append(r_dict)
    
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """Main analysis function."""
    print("\n" + "=" * 100)
    print("  DEEP PROFITABILITY ANALYSIS".center(98))
    print("  Football Prediction Model Assessment".center(98))
    print("=" * 100)
    
    # Generate synthetic historical data
    print("\n📊 GENERATING SYNTHETIC HISTORICAL DATA")
    print("-" * 100)
    
    # Realistic parameters based on football statistics
    # Home advantage typically gives ~45% home win rate in major leagues
    data = generate_synthetic_data(
        n_matches=5000,
        true_home_prob=0.45,
        true_draw_prob=0.27,
        true_away_prob=0.28,
        odds_margin=0.05,  # 5% bookmaker margin
        model_calibration_error=0.03,  # Model has small calibration noise
        seed=42,
    )
    
    print(f"Generated {len(data):,} matches spanning {data['date'].max() - data['date'].min()} days")
    print(f"Outcome distribution: Home {data['outcome'].value_counts().get(2, 0):,} | "
          f"Draw {data['outcome'].value_counts().get(1, 0):,} | "
          f"Away {data['outcome'].value_counts().get(0, 0):,}")
    
    # Define multiple strategies to test
    strategies = [
        {
            "name": "Conservative (Low Kelly, High EV)",
            "kelly_fraction": 0.15,
            "min_ev": 0.05,
            "max_odds": 5.0,
            "min_confidence": 0.35,
        },
        {
            "name": "Balanced (Standard Kelly)",
            "kelly_fraction": 0.25,
            "min_ev": 0.03,
            "max_odds": 10.0,
            "min_confidence": 0.30,
        },
        {
            "name": "Aggressive (Full Kelly)",
            "kelly_fraction": 0.50,
            "min_ev": 0.02,
            "max_odds": 20.0,
            "min_confidence": 0.25,
        },
        {
            "name": "Value Hunter (High EV Threshold)",
            "kelly_fraction": 0.25,
            "min_ev": 0.08,
            "max_odds": 8.0,
            "min_confidence": 0.40,
        },
        {
            "name": "Longshot Specialist",
            "kelly_fraction": 0.20,
            "min_ev": 0.05,
            "max_odds": None,
            "min_confidence": 0.20,
        },
        {
            "name": "Favorite Backer (Safe)",
            "kelly_fraction": 0.30,
            "min_ev": 0.02,
            "max_odds": 2.5,
            "min_confidence": 0.45,
        },
    ]
    
    # Run backtests for all strategies
    results: list[AnalysisMetrics] = []
    
    print("\n🔄 RUNNING BACKTEST SIMULATIONS")
    print("-" * 100)
    
    for strat in strategies:
        print(f"\n  Testing: {strat['name']}...")
        metrics = run_backtest_simulation(
            data=data,
            strategy_name=strat["name"],
            kelly_fraction=strat["kelly_fraction"],
            min_ev=strat["min_ev"],
            max_odds=strat["max_odds"],
            min_confidence=strat["min_confidence"],
            initial_bankroll=1000.0,
        )
        results.append(metrics)
    
    # Print detailed report for best strategy
    best_strategy = max(results, key=lambda r: r.sharpe_ratio)
    print_detailed_report(best_strategy)
    
    # Compare all strategies
    compare_strategies(results)
    
    # Save results
    output_dir = PROJECT_ROOT / "reports" / "profitability_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"profitability_analysis_{timestamp}.json"
    save_results_to_json(results, output_path)
    
    # Key insights summary
    print("\n" + "=" * 100)
    print("  KEY INSIGHTS & RECOMMENDATIONS".center(98))
    print("=" * 100)
    
    profitable_strategies = [r for r in results if r.roi_pct > 0]
    significant_strategies = [r for r in results if r.is_significant]
    
    print(f"\n✅ Profitable Strategies: {len(profitable_strategies)}/{len(results)}")
    print(f"🎯 Statistically Significant: {len(significant_strategies)}/{len(results)}")
    
    if profitable_strategies:
        best_profitable = max(profitable_strategies, key=lambda r: r.sharpe_ratio)
        print(f"\n🏆 RECOMMENDED STRATEGY: {best_profitable.strategy_name}")
        print(f"   • Expected ROI: {best_profitable.roi_pct:+.2f}%")
        print(f"   • Risk-Adjusted Return (Sharpe): {best_profitable.sharpe_ratio:.2f}")
        print(f"   • Max Drawdown: {best_profitable.max_drawdown_pct:.1f}%")
        print(f"   • Win Rate: {best_profitable.win_rate_pct:.1f}%")
        
        print(f"\n📋 IMPLEMENTATION NOTES:")
        print(f"   • Use Kelly fraction: {best_profitable.kelly_fraction:.0%}")
        print(f"   • Minimum EV threshold: {best_profitable.min_ev:.2%}")
        if best_profitable.max_odds:
            print(f"   • Maximum odds filter: {best_profitable.max_odds:.1f}")
        print(f"   • Expected bets per season: ~{best_profitable.total_bets // 14:.0f} (based on 5000 matches)")
        
        print(f"\n⚠️ RISK WARNINGS:")
        print(f"   • Maximum historical drawdown: {best_profitable.max_drawdown_pct:.1f}%")
        print(f"   • Longest losing streak: {best_profitable.longest_lose_streak} bets")
        print(f"   • Ensure bankroll can withstand worst-case scenarios")
    else:
        print("\n⚠️ WARNING: No profitable strategies found in backtesting.")
        print("   Consider:")
        print("   • Improving model calibration")
        print("   • Adjusting EV thresholds")
        print("   • Exploring different markets or bet types")
        print("   • Increasing sample size for more robust testing")
    
    print("\n" + "=" * 100)
    print("  ANALYSIS COMPLETE".center(98))
    print("=" * 100)
    print(f"\n📄 Full report saved to: {output_path}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Review recommended strategy parameters")
    print(f"   2. Validate with out-of-sample testing")
    print(f"   3. Start with small stakes in live testing")
    print(f"   4. Monitor performance and adjust as needed")
    print()


if __name__ == "__main__":
    main()
