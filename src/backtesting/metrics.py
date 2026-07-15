"""
Backtesting Metrics — standalone, reusable metric computation for betting
simulations.

Provides pure functions and a ``MetricsCalculator`` class for computing
all standard backtest metrics from a list of bet results.  Designed to
be imported and used by any backtester (``BacktestEngine``, ``Backtester``,
or custom implementations).

Metric Definitions (user-specified)
------------------------------------
- **Total profit/loss:** absolute P&L and % return on initial bankroll
- **ROI (Return on Investment):** ``total_profit / total_staked``
- **Yield:** ``total_profit / number_of_bets`` (average profit per bet)
- **Win rate:** ``winning_bets / total_bets``
- **Maximum drawdown:** max % loss from peak bankroll
- **Sharpe ratio:** ``(mean_return - risk_free_rate) / std_dev_returns``
- **CLV (Closing Line Value):** ``(your_odds - closing_odds) / closing_odds``
- **Profit factor:** ``total_returns / total_staked``

Usage
-----
::

    from src.backtesting.metrics import MetricsCalculator, BetResult

    bets = [
        BetResult(stake=50, profit=55, odds=2.10, won=True),
        BetResult(stake=25, profit=-25, odds=1.80, won=False),
    ]

    calc = MetricsCalculator(initial_bankroll=1000.0)
    result = calc.compute(bets)

    print(result.roi)           # 0.20  (20% return on staked)
    print(result.sharpe_ratio)  # 1.23
    print(result.max_drawdown)  # 2.50  (2.5% peak-to-trough)

    path = calc.save_report("reports/backtest", "my_model")
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Input data structure
# ═══════════════════════════════════════════════════════════


@dataclass
class BetResult:
    """A single bet result for metrics computation.

    This is the universal input format for the metrics calculator.
    Convert from any bet record format (``BetRecord``,
    ``BacktestBetRecord``, or raw dicts) by mapping fields.

    Parameters
    ----------
    stake : float
        Amount staked on the bet.
    profit : float
        Net profit/loss (positive = profit, negative = loss).
    odds : float
        Decimal odds at which the bet was placed.
    won : bool
        True if the bet was a winner.
    pushed : bool
        True if the bet was void/refunded (stake returned, no profit/loss).
    bankroll_before : float
        Bankroll balance just before this bet.
    closing_odds : float | None
        Closing (kick-off) decimal odds for CLV calculation.
    market : str
        Market type (``"1X2"``, ``"BTTS"``, ``"Over/Under"``).
    outcome_label : str
        Human-readable outcome label (e.g. ``"Home Win"``).
    match_label : str
        Human-readable match identifier.
    """
    stake: float
    profit: float
    odds: float
    won: bool
    pushed: bool = False
    bankroll_before: float = 0.0
    closing_odds: float | None = None
    market: str = "1X2"
    outcome_label: str = ""
    match_label: str = ""


# ═══════════════════════════════════════════════════════════
#  Result data structure — all computed metrics
# ═══════════════════════════════════════════════════════════


@dataclass
class MetricsResult:
    """All computed backtest metrics from a set of bet results.

    Contains the 8 user-requested metrics plus additional computed
    values for completeness and auditing.
    """

    # ── Basic counts ────────────────────────────────────
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    pushed_bets: int = 0

    # ── Financial metrics (user-specified formulas) ─────
    total_profit: float = 0.0                      # Absolute P&L
    total_profit_pct: float = 0.0                  # (final-initial)/initial * 100
    roi: float = 0.0                               # total_profit / total_staked
    yield_per_bet: float = 0.0                     # total_profit / number_of_bets
    win_rate: float = 0.0                          # winning_bets / total_bets
    profit_factor: float = 1.0                     # total_returns / total_staked

    # ── Risk metrics ───────────────────────────────────
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # ── CLV ────────────────────────────────────────────
    avg_clv: float = 0.0
    positive_clv_pct: float = 0.0

    # ── Additional goodies ──────────────────────────────
    total_staked: float = 0.0
    total_returns: float = 0.0                     # Used in profit_factor calc
    initial_bankroll: float = 1000.0
    final_bankroll: float = 1000.0
    avg_odds: float = 0.0
    avg_stake: float = 0.0
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    bankroll_history: list[float] = field(default_factory=list)

    # ── Market breakdown ───────────────────────────────
    bets_per_market: dict[str, int] = field(default_factory=dict)
    profit_per_market: dict[str, float] = field(default_factory=dict)

    # ── Standard alternatives (for reference) ──────────
    roi_on_bankroll_pct: float = 0.0               # (final-initial)/initial * 100 (standard)
    yield_on_staked_pct: float = 0.0               # total_profit / total_staked * 100 (standard yield)
    profit_factor_standard: float = 1.0             # gross_profit / gross_loss


# ═══════════════════════════════════════════════════════════
#  Pure computation functions (stateless, reusable)
# ═══════════════════════════════════════════════════════════


def compute_total_profit(bets: list[BetResult]) -> float:
    """Sum of all bet profits (pushed bets contribute 0)."""
    return sum(b.profit for b in bets if not b.pushed)


def compute_total_profit_pct(
    total_profit: float, initial_bankroll: float,
) -> float:
    """Profit as a percentage of initial bankroll."""
    if initial_bankroll <= 0:
        return 0.0
    return (total_profit / initial_bankroll) * 100


def compute_roi(total_profit: float, total_staked: float) -> float:
    """Return on Investment: profit / staked.

    Per user specification: ``total_profit / total_staked``.
    A value of 0.05 means 5% return on staked capital.
    """
    if total_staked <= 0:
        return 0.0
    return total_profit / total_staked


def compute_yield_per_bet(total_profit: float, num_bets: int) -> float:
    """Average profit per bet.

    Per user specification: ``total_profit / number_of_bets``.
    """
    if num_bets <= 0:
        return 0.0
    return total_profit / num_bets


def compute_win_rate(wins: int, total: int) -> float:
    """Win rate as a fraction (0 to 1).

    Per user specification: ``winning_bets / total_bets``.
    """
    if total <= 0:
        return 0.0
    return wins / total


def compute_profit_factor(total_returns: float, total_staked: float) -> float:
    """Profit factor: total returns / total staked.

    Per user specification.  ``total_returns`` is the sum of all
    returns from winning bets (stake + profit).  A profit factor
    of 2.0 means you get back GBP2 for every GBP1 staked.
    """
    if total_staked <= 0:
        return 1.0
    return total_returns / total_staked


def compute_profit_factor_standard(bets: list[BetResult]) -> float:
    """Standard profit factor: gross profit / gross loss.

    Used as an alternative metric.  A value of 2.0 means you make
    GBP2 in profits for every GBP1 in losses.
    """
    gross_profit = sum(b.profit for b in bets if not b.pushed and b.profit > 0)
    gross_loss = abs(sum(b.profit for b in bets if not b.pushed and b.profit < 0))
    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return float("inf")
    return 0.0


def compute_max_drawdown(bankroll_history: list[float]) -> tuple[float, float]:
    """Maximum drawdown from peak bankroll.

    Parameters
    ----------
    bankroll_history : list[float]
        Chronological bankroll values.

    Returns
    -------
    tuple[float, float]
        ``(max_drawdown_pct, max_drawdown_amount)``.
    """
    if not bankroll_history:
        return 0.0, 0.0

    peak = bankroll_history[0]
    max_dd_pct = 0.0
    max_dd_amount = 0.0

    for value in bankroll_history:
        if value > peak:
            peak = value
        dd_amount = peak - value
        dd_pct = (peak - value) / peak * 100 if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_amount = dd_amount

    return max_dd_pct, max_dd_amount


def compute_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    annualise: bool = True,
    bets_per_year: int = 500,
) -> float:
    """Sharpe ratio: risk-adjusted return.

    Formula: ``(mean_return - risk_free_rate) / std_dev_returns``

    Parameters
    ----------
    returns : list[float]
        Bet-level returns (profit / bankroll_before for each bet).
    risk_free_rate : float
        Risk-free rate as a fraction (default 0.0).
    annualise : bool
        If True, annualise using ``sqrt(bets_per_year)``.
    bets_per_year : int
        Assumed number of bets per year for annualisation (default 500).

    Returns
    -------
    float
        Sharpe ratio.  > 1.0 is good, > 2.0 is excellent.
    """
    if len(returns) < 2:
        return 0.0

    arr = np.array(returns)
    mean_return = float(np.mean(arr))
    std_return = float(np.std(arr, ddof=1))

    if std_return <= 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return

    if annualise:
        sharpe *= math.sqrt(bets_per_year)

    return sharpe


def compute_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    annualise: bool = True,
    bets_per_year: int = 500,
) -> float:
    """Sortino ratio: risk-adjusted return using only downside deviation.

    Formula: ``(mean_return - risk_free_rate) / downside_std``

    Parameters
    ----------
    returns : list[float]
        Bet-level returns (profit / bankroll_before for each bet).
    risk_free_rate : float
        Risk-free rate as a fraction (default 0.0).
    annualise : bool
        If True, annualise using ``sqrt(bets_per_year)``.
    bets_per_year : int
        Assumed number of bets per year for annualisation (default 500).

    Returns
    -------
    float
        Sortino ratio.  Higher values indicate better downside risk-adjusted
        returns.
    """
    if len(returns) < 2:
        return 0.0

    arr = np.array(returns)
    mean_return = float(np.mean(arr))
    neg_returns = arr[arr < 0]

    if len(neg_returns) == 0:
        # No negative returns — effectively infinite Sortino.
        # Cap at 999.0 for clean display and JSON serialization.
        return 999.0 if mean_return > 0 else 0.0

    downside_std = float(
        np.std(neg_returns, ddof=1) if len(neg_returns) > 1
        else np.std(neg_returns)
    )

    if downside_std <= 0:
        return 0.0

    sortino = (mean_return - risk_free_rate) / downside_std

    if annualise:
        sortino *= math.sqrt(bets_per_year)

    return sortino


def compute_clv(your_odds: float, closing_odds: float) -> float:
    """Closing Line Value for a single bet.

    Per user specification: ``(your_odds - closing_odds) / closing_odds``

    Positive CLV means you got better odds than the market closed at
    (you beat the closing line).

    Parameters
    ----------
    your_odds : float
        Decimal odds at which you placed the bet.
    closing_odds : float
        Decimal odds at market close (kick-off).

    Returns
    -------
    float
        CLV as a fraction.  0.05 = 5% better odds than closing.
    """
    if closing_odds <= 1.0 or your_odds <= 1.0:
        return 0.0
    return (your_odds - closing_odds) / closing_odds


def compute_avg_clv(bets: list[BetResult]) -> tuple[float, float]:
    """Average CLV and percentage of bets with positive CLV.

    Parameters
    ----------
    bets : list[BetResult]
        Bet results with optional ``closing_odds``.

    Returns
    -------
    tuple[float, float]
        ``(avg_clv, positive_clv_pct)``.
        ``avg_clv`` is 0.0 if no bets have closing odds.
    """
    clv_values = [
        compute_clv(b.odds, b.closing_odds)
        for b in bets
        if not b.pushed and b.closing_odds is not None and b.closing_odds > 1.0
    ]

    if not clv_values:
        return 0.0, 0.0

    avg_clv = float(np.mean(clv_values))
    positive_pct = sum(1 for v in clv_values if v > 0) / len(clv_values) * 100

    return avg_clv, positive_pct


def compute_streaks(bets: list[BetResult]) -> tuple[int, int, int, int]:
    """Compute win/loss streaks.

    Parameters
    ----------
    bets : list[BetResult]
        Chronologically ordered bet results (pushed bets excluded).

    Returns
    -------
    tuple[int, int, int, int]
        ``(longest_win, longest_lose, current_win, current_lose)``.
    """
    if not bets:
        return 0, 0, 0, 0

    longest_win = 0
    longest_lose = 0
    current_win = 0
    current_lose = 0
    max_win = 0
    max_lose = 0

    for b in bets:
        if b.won:
            current_win += 1
            current_lose = 0
            max_win = max(max_win, current_win)
        else:
            current_lose += 1
            current_win = 0
            max_lose = max(max_lose, current_lose)

    return max_win, max_lose, current_win, current_lose


def compute_bet_level_returns(bets: list[BetResult]) -> list[float]:
    """Compute bet-level returns (profit / bankroll_before).

    Used for Sharpe and Sortino ratio calculations.
    Pushed bets are excluded.
    """
    return [
        b.profit / max(b.bankroll_before, 0.01)
        for b in bets
        if not b.pushed and b.bankroll_before > 0
    ]


def compute_bankroll_history(
    bets: list[BetResult],
    initial_bankroll: float,
) -> list[float]:
    """Build a chronological bankroll history.

    Pushed bets do not affect the bankroll.
    """
    history = [initial_bankroll]
    bankroll = initial_bankroll

    for b in bets:
        if not b.pushed:
            bankroll += b.profit
            history.append(bankroll)

    if len(history) == 1:
        history.append(bankroll)

    return history


# ═══════════════════════════════════════════════════════════
#  Brand-specific metrics (from existing codebase)
# ═══════════════════════════════════════════════════════════


def compute_roi_on_bankroll(
    final_bankroll: float, initial_bankroll: float,
) -> float:
    """ROI on initial bankroll (standard definition)."""
    if initial_bankroll <= 0:
        return 0.0
    return ((final_bankroll - initial_bankroll) / initial_bankroll) * 100


def compute_yield_on_staked(total_profit: float, total_staked: float) -> float:
    """Yield as profit per unit staked (standard definition)."""
    if total_staked <= 0:
        return 0.0
    return (total_profit / total_staked) * 100


# ═══════════════════════════════════════════════════════════
#  MetricsCalculator — orchestrator
# ═══════════════════════════════════════════════════════════


class MetricsCalculator:
    """Compute and report backtest metrics.

    Parameters
    ----------
    initial_bankroll : float
        Starting bankroll in currency units (default 1000).
    risk_free_rate : float
        Risk-free rate for Sharpe/Sortino calculation (default 0.0).
    annualise_sharpe : bool
        If True, annualise Sharpe/Sortino (default True).
    bets_per_year : int
        Assumed bets per year for annualisation (default 500).
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        risk_free_rate: float = 0.0,
        annualise_sharpe: bool = True,
        bets_per_year: int = 500,
    ) -> None:
        self.initial_bankroll = initial_bankroll
        self.risk_free_rate = risk_free_rate
        self.annualise_sharpe = annualise_sharpe
        self.bets_per_year = bets_per_year
        self._last_result: MetricsResult | None = None

    # ── Main computation ────────────────────────────────

    def compute(
        self,
        bets: list[BetResult],
        initial_bankroll: float | None = None,
    ) -> MetricsResult:
        """Compute all metrics from a list of bet results.

        Parameters
        ----------
        bets : list[BetResult]
            Chronologically ordered bet results.
        initial_bankroll : float, optional
            Override the initial bankroll for this computation.

        Returns
        -------
        MetricsResult
            All computed metrics.
        """
        bankroll = initial_bankroll if initial_bankroll is not None else self.initial_bankroll
        active = [b for b in bets if not b.pushed]
        n_total = len(active)
        n_pushed = len(bets) - n_total
        n_wins = sum(1 for b in active if b.won)
        n_losses = n_total - n_wins

        # Financial aggregates
        total_staked = sum(b.stake for b in active)
        total_profit = compute_total_profit(bets)
        final_bankroll = bankroll + total_profit

        # Total returns = sum of (stake + profit) for winning bets
        total_returns = sum(
            b.stake + b.profit for b in active if b.won
        )

        # Bankroll history
        bankroll_history = compute_bankroll_history(bets, bankroll)

        # Drawdown
        max_dd_pct, max_dd_amount = compute_max_drawdown(bankroll_history)

        # Bet-level returns for Sharpe / Sortino
        returns = compute_bet_level_returns(bets)

        # CLV
        avg_clv, positive_clv_pct = compute_avg_clv(bets)

        # Streaks
        lw, ll, cw, cl = compute_streaks(active)

        # ── Build result ────────────────────────────────
        m = MetricsResult(
            # Counts
            total_bets=n_total,
            winning_bets=n_wins,
            losing_bets=n_losses,
            pushed_bets=n_pushed,

            # Financial (user-specified formulas)
            total_profit=round(total_profit, 2),
            total_profit_pct=round(compute_total_profit_pct(total_profit, bankroll), 4),
            roi=round(compute_roi(total_profit, total_staked), 6),
            yield_per_bet=round(compute_yield_per_bet(total_profit, n_total), 4),
            win_rate=round(compute_win_rate(n_wins, n_total), 6),
            profit_factor=round(compute_profit_factor(total_returns, total_staked), 4),

            # Risk
            max_drawdown_pct=round(max_dd_pct, 4),
            max_drawdown_amount=round(max_dd_amount, 2),
            sharpe_ratio=round(compute_sharpe_ratio(
                returns, self.risk_free_rate, self.annualise_sharpe, self.bets_per_year,
            ), 4),
            sortino_ratio=round(compute_sortino_ratio(
                returns, self.risk_free_rate, self.annualise_sharpe, self.bets_per_year,
            ), 4),

            # CLV
            avg_clv=round(avg_clv, 6),
            positive_clv_pct=round(positive_clv_pct, 2),

            # Additional
            total_staked=round(total_staked, 2),
            total_returns=round(total_returns, 2),
            initial_bankroll=bankroll,
            final_bankroll=round(final_bankroll, 2),
            avg_odds=round(float(np.mean([b.odds for b in active])), 4) if active else 0.0,
            avg_stake=round(total_staked / n_total, 2) if n_total > 0 else 0.0,
            longest_win_streak=lw,
            longest_lose_streak=ll,
            bankroll_history=[round(v, 2) for v in bankroll_history],

            # Market breakdown
            bets_per_market=_breakdown_markets(active),
            profit_per_market=_breakdown_profits(active),

            # Standard alternatives
            roi_on_bankroll_pct=round(compute_roi_on_bankroll(final_bankroll, bankroll), 4),
            yield_on_staked_pct=round(compute_yield_on_staked(total_profit, total_staked), 4),
            profit_factor_standard=round(compute_profit_factor_standard(bets), 4),
        )

        self._last_result = m
        return m

    # ── Report saving ───────────────────────────────────

    def save_report(
        self,
        bets: list[BetResult],
        output_dir: str | Path = "reports/backtest",
        model_name: str = "metrics",
        initial_bankroll: float | None = None,
    ) -> str:
        """Compute metrics and save results to a JSON report file.

        Parameters
        ----------
        bets : list[BetResult]
            Chronologically ordered bet results.
        output_dir : str | Path
            Directory to save the report (default ``reports/backtest/``).
        model_name : str
            Model identifier for the filename (default ``"metrics"``).
        initial_bankroll : float, optional
            Override initial bankroll.

        Returns
        -------
        str
            Path to the saved JSON report file.
        """
        metrics = self.compute(bets, initial_bankroll=initial_bankroll)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{model_name}_{timestamp}.json"
        filepath = out_dir / filename

        # Build export dict
        data = {
            "model_name": model_name,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": asdict(metrics),
            "bets": [
                {
                    "match": b.match_label,
                    "outcome": b.outcome_label,
                    "market": b.market,
                    "odds": b.odds,
                    "stake": b.stake,
                    "profit": b.profit,
                    "won": b.won,
                    "pushed": b.pushed,
                    "closing_odds": b.closing_odds,
                }
                for b in bets
            ],
            "configuration": {
                "initial_bankroll": initial_bankroll or self.initial_bankroll,
                "risk_free_rate": self.risk_free_rate,
                "annualise_sharpe": self.annualise_sharpe,
                "bets_per_year": self.bets_per_year,
            },
        }

        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, float):
                if math.isinf(obj):
                    return None
                if math.isnan(obj):
                    return None
                return round(obj, 6)
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        data = _sanitize(data)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Metrics report saved to %s", filepath)
        return str(filepath)

    # ── Format / display ────────────────────────────────

    def format_report(self, metrics: MetricsResult | None = None) -> str:
        """Format metrics as a human-readable string.

        Parameters
        ----------
        metrics : MetricsResult, optional
            Metrics to format. Uses last computed result if omitted.

        Returns
        -------
        str
            Formatted report string.
        """
        m = metrics or self._last_result or MetricsResult()

        lines = []
        lines.append("=" * 72)
        lines.append("  BACKTEST METRICS REPORT")
        lines.append("=" * 72)

        if m.total_bets == 0:
            lines.append("  No bets recorded.")
            lines.append("=" * 72)
            return "\n".join(lines)

        # ── Summary ──
        lines.append(f"  Bets: {m.total_bets}  |  "
                      f"Wins: {m.winning_bets}  |  "
                      f"Losses: {m.losing_bets}  |  "
                      f"Pushed: {m.pushed_bets}")
        lines.append(f"  Win Rate: {m.win_rate:.1%}")
        lines.append(f"  Total Staked: {m.total_staked:.2f}")

        # ── P&L ──
        pnl_str = f"+{m.total_profit:.2f}" if m.total_profit >= 0 else f"{m.total_profit:.2f}"
        lines.append(f"")
        lines.append(f"  P&L")
        lines.append(f"  {'-' * 40}")
        lines.append(f"    Total Profit/Loss:      {pnl_str}")
        lines.append(f"    Return on Bankroll:     {m.total_profit_pct:+.2f}%")
        lines.append(f"    Final Bankroll:         {m.final_bankroll:.2f}")

        # ── Performance ratios (user-specified) ──
        lines.append(f"")
        lines.append(f"  Performance Ratios (user-specified)")
        lines.append(f"  {'-' * 40}")
        lines.append(f"    ROI (profit / staked):     {m.roi:.4f}  ({m.roi*100:+.2f}%)")
        lines.append(f"    Yield (profit / bet):      {m.yield_per_bet:.4f}")
        lines.append(f"    Profit Factor (returns / staked): {m.profit_factor:.4f}")

        # ── Risk ──
        lines.append(f"")
        lines.append(f"  Risk Metrics")
        lines.append(f"  {'-' * 40}")
        lines.append(f"    Max Drawdown:           {m.max_drawdown_pct:.2f}%  "
                      f"({m.max_drawdown_amount:.2f})")
        if m.sharpe_ratio != 0.0:
            lines.append(f"    Sharpe Ratio:           {m.sharpe_ratio:.4f}")
        if m.sortino_ratio != 0.0:
            lines.append(f"    Sortino Ratio:          {m.sortino_ratio:.4f}")

        # ── CLV ──
        if m.avg_clv != 0.0:
            lines.append(f"")
            lines.append(f"  Closing Line Value")
            lines.append(f"  {'-' * 40}")
            lines.append(f"    Avg CLV:                {m.avg_clv:+.6f}")
            lines.append(f"    Positive CLV:           {m.positive_clv_pct:.1f}%")

        # ── Additional ──
        lines.append(f"")
        lines.append(f"  Additional")
        lines.append(f"  {'-' * 40}")
        lines.append(f"    Avg Odds:               {m.avg_odds:.4f}")
        lines.append(f"    Avg Stake:              {m.avg_stake:.2f}")
        lines.append(f"    Longest Win Streak:     {m.longest_win_streak}")
        lines.append(f"    Longest Lose Streak:    {m.longest_lose_streak}")

        # ── Standard alternatives ──
        lines.append(f"")
        lines.append(f"  Standard Alternatives (for reference)")
        lines.append(f"  {'-' * 40}")
        lines.append(f"    ROI on Bankroll:        {m.roi_on_bankroll_pct:+.2f}%")
        lines.append(f"    Yield on Staked:        {m.yield_on_staked_pct:+.2f}%")
        lines.append(f"    Profit Factor (std):    {m.profit_factor_standard:.4f}")

        # ── Market breakdown ──
        if m.bets_per_market:
            lines.append(f"")
            lines.append(f"  Market Breakdown")
            lines.append(f"  {'-' * 40}")
            for market in sorted(m.bets_per_market.keys()):
                n = m.bets_per_market[market]
                pnl = m.profit_per_market.get(market, 0.0)
                pnl_sign = "+" if pnl >= 0 else ""
                lines.append(f"    {market:<12s}  {n:>4d} bets  "
                              f"{pnl_sign}{pnl:>+.2f} P&L")

        lines.append("=" * 72)
        return "\n".join(lines)

    def print_report(self, metrics: MetricsResult | None = None) -> None:
        """Print the formatted report to the console."""
        print(self.format_report(metrics))

    # ── Converter helpers ────────────────────────────────

    @staticmethod
    def from_dicts(bet_dicts: list[dict[str, Any]]) -> list[BetResult]:
        """Convert a list of dicts to ``BetResult`` instances.

        Accepts dicts with keys: ``stake``, ``profit``, ``odds``,
        ``won``, ``pushed``, ``bankroll_before``, ``closing_odds``,
        ``market``, ``outcome_label``, ``match_label``.
        """
        return [
            BetResult(
                stake=float(b.get("stake", 0)),
                profit=float(b.get("profit", 0)),
                odds=float(b.get("odds") or b.get("decimal_odds") or _raise_missing("odds")),
                won=bool(b.get("won", False)),
                pushed=bool(b.get("pushed", False)),
                bankroll_before=float(b.get("bankroll_before", 0)),
                closing_odds=(
                    float(b["closing_odds"])
                    if b.get("closing_odds") is not None and b["closing_odds"] > 1.0
                    else None
                ),
                market=str(b.get("market", b.get("market_type", "1X2"))),
                outcome_label=str(b.get("outcome", b.get("outcome_label", ""))),
                match_label=str(b.get("match", b.get("match_label", ""))),
            )
            for b in bet_dicts
        ]

    @property
    def last_result(self) -> MetricsResult | None:
        return self._last_result


# ═══════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════


def _raise_missing(field: str) -> None:
    """Raise a ValueError for a missing required field."""
    raise ValueError(
        f"Missing required field '{field}' in bet dict. "
        f"Provide '{field}' or 'decimal_odds'."
    )


def _breakdown_markets(bets: list[BetResult]) -> dict[str, int]:
    """Count bets per market type."""
    result: dict[str, int] = {}
    for b in bets:
        result[b.market] = result.get(b.market, 0) + 1
    return result


def _breakdown_profits(bets: list[BetResult]) -> dict[str, float]:
    """Sum profit per market type."""
    result: dict[str, float] = {}
    for b in bets:
        result[b.market] = result.get(b.market, 0.0) + b.profit
    return result


# ═══════════════════════════════════════════════════════════
#  Convenience functions (one-liner API)
# ═══════════════════════════════════════════════════════════


def quick_metrics(
    bets: list[BetResult],
    initial_bankroll: float = 1000.0,
) -> MetricsResult:
    """One-liner: compute all metrics.

    Parameters
    ----------
    bets : list[BetResult]
        Chronologically ordered bet results.
    initial_bankroll : float
        Starting bankroll (default 1000).

    Returns
    -------
    MetricsResult
        All computed metrics.
    """
    return MetricsCalculator(initial_bankroll=initial_bankroll).compute(bets)


def metrics_from_dicts(
    bet_dicts: list[dict[str, Any]],
    initial_bankroll: float = 1000.0,
) -> MetricsResult:
    """One-liner: compute metrics from dicts.

    Parameters
    ----------
    bet_dicts : list[dict]
        List of bet dicts (see ``from_dicts`` for required keys).
    initial_bankroll : float
        Starting bankroll (default 1000).

    Returns
    -------
    MetricsResult
        All computed metrics.
    """
    bets = MetricsCalculator.from_dicts(bet_dicts)
    return quick_metrics(bets, initial_bankroll)
