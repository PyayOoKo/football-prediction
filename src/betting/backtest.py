"""
Backtesting — simulate betting strategies on historical match data.

The ``Backtester`` walks through historical bet opportunities chronologically,
applies filters, computes stakes, settles bets against actual outcomes, and
tracks comprehensive performance metrics.

Metrics tracked
---------------
- **Total P&L** — absolute profit/loss in currency units
- **ROI** — total return on initial bankroll (percentage)
- **Yield** — profit per unit staked (percentage)
- **Win Rate** — percentage of bets that won
- **Max Drawdown** — largest peak-to-trough decline (percentage)
- **Sharpe Ratio** — risk-adjusted return (uses bet-level returns)
- **CLV** — closing line value (market movement vs bet placement)
- **Number of bets** — total, winning, losing
- **Profit factor** — gross profit / gross loss
- **Longest win/loss streaks**

Usage
-----
::

    from src.betting.backtest import Backtester
    from src.betting.staking import KellyStaking
    from src.betting.filtering import BetFilter

    historical_bets = [
        {
            "match": "Arsenal vs Chelsea",
            "outcome": "Home Win",
            "market": "1X2",
            "model_prob": 0.52,
            "decimal_odds": 2.10,
            "closing_odds": 2.05,
            "actual_result": True,      # bet won
        },
        ...
    ]

    backtester = Backtester(
        initial_bankroll=1000.0,
        stake_strategy=KellyStaking(),
        bet_filter=BetFilter(min_ev=0.0, min_confidence=0.3),
    )
    results = backtester.run(historical_bets)
    backtester.print_report()
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.betting.ev import calculate_ev
from src.betting.filtering import BetFilter
from src.betting.staking import StakingFactory, StakingStrategy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════


@dataclass
class BacktestBetRecord:
    """A single placed bet during the backtest simulation."""

    match_label: str
    outcome: str
    market: str
    decimal_odds: float
    model_prob: float
    ev: float
    edge_vs_market: float
    stake_amount: float
    stake_pct: float
    profit: float
    won: bool
    clv: float | None = None
    bankroll_before: float = 0.0
    bankroll_after: float = 0.0
    timestamp: int = 0


@dataclass
class BacktestMetrics:
    """Aggregate performance metrics from a backtest run."""

    # Core stats
    initial_bankroll: float = 1000.0
    final_bankroll: float = 1000.0
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0

    # Performance ratios
    roi_pct: float = 0.0
    yield_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # CLV
    avg_clv: float = 0.0
    positive_clv_pct: float = 0.0

    # Bet quality
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    avg_stake_pct: float = 0.0

    # Streaks
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    current_win_streak: int = 0
    current_lose_streak: int = 0

    # History
    bankroll_history: list[float] = field(default_factory=list)
    drawdown_history: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    # Timestamps
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    # Configuration snapshot
    stake_strategy: str = ""
    bet_filter_config: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  Main Backtester class
# ═══════════════════════════════════════════════════════════


class Backtester:
    """Simulate a betting strategy on historical data.

    Parameters
    ----------
    initial_bankroll : float
        Starting bankroll in currency units (default 1000).
    stake_strategy : StakingStrategy, optional
        A ``StakingStrategy`` instance. Defaults to
        ``FractionalKellyStaking(fraction=0.25)``.
    bet_filter : BetFilter, optional
        A ``BetFilter`` instance. Defaults to a filter with
        ``min_ev=0.0, min_confidence=0.3, min_odds=1.5``.
    max_bets_per_match : int
        Maximum number of bets allowed per match (default 1).
    track_equity_curve : bool
        If True, record bankroll after every match for detailed
        equity curve plotting. Default True.
    """

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        stake_strategy: StakingStrategy | None = None,
        bet_filter: BetFilter | None = None,
        max_bets_per_match: int = 1,
        track_equity_curve: bool = True,
    ) -> None:
        if initial_bankroll <= 0:
            raise ValueError(f"initial_bankroll must be > 0, got {initial_bankroll}")

        self.initial_bankroll = initial_bankroll
        self.stake_strategy = stake_strategy or StakingFactory.create(
            "fractional_kelly",
            fraction=0.25,
        )
        self.bet_filter = bet_filter or BetFilter(
            min_ev=0.0,
            min_confidence=0.3,
            min_odds=1.5,
        )
        self.max_bets_per_match = max_bets_per_match
        self.track_equity_curve = track_equity_curve

        # Internal state
        self._bankroll: float = initial_bankroll
        self._peak: float = initial_bankroll
        self._bets: list[BacktestBetRecord] = []
        self._metrics: BacktestMetrics | None = None
        self._start_time: datetime = datetime.now(timezone.utc)

    # ── Run simulation ──────────────────────────────────

    def run(
        self,
        historical_bets: list[dict[str, Any]],
    ) -> BacktestMetrics:
        """Run the backtest simulation on historical bet opportunities.

        Parameters
        ----------
        historical_bets : list[dict]
            Each dict must have:
            - ``model_prob`` (float): model's estimated probability
            - ``decimal_odds`` (float): bookmaker's decimal odds
            - ``actual_result`` (bool): True if the bet won

            Optional keys:
            - ``match`` (str): match label for display
            - ``outcome`` (str): outcome label (e.g. "Home Win")
            - ``market`` (str): market type (e.g. "1X2")
            - ``closing_odds`` (float): closing odds for CLV calc
            - ``ev`` (float): pre-computed EV (auto-calculated if missing)

        Returns
        -------
        BacktestMetrics
            Aggregate performance metrics.
        """
        self._start_time = datetime.now(timezone.utc)
        self._bankroll = self.initial_bankroll
        self._peak = self.initial_bankroll
        self._bets = []

        if not historical_bets:
            logger.warning("No historical bets provided — backtest is empty")
            self._metrics = self._compute_metrics()
            return self._metrics

        # Run filter first to get (passed, rejected)
        passed_bets, rejected_bets = self.bet_filter.filter_bets(historical_bets)

        if not passed_bets:
            logger.info(
                "All %d bets rejected by filter — no bets placed",
                len(historical_bets),
            )
            self._metrics = self._compute_metrics()
            return self._metrics

        logger.info(
            "Running backtest on %d/%d bets passing filter",
            len(passed_bets),
            len(historical_bets),
        )

        # Track bets per match for max_bets_per_match
        match_counts: dict[str, int] = {}

        for bet in passed_bets:
            match_label = bet.get("match", bet.get("match_label", "Unknown"))
            outcome_label = bet.get("outcome", bet.get("outcome_label", ""))
            market = bet.get("market", bet.get("market_type", "1X2"))

            # Enforce max bets per match
            match_counts[match_label] = match_counts.get(match_label, 0) + 1
            if match_counts[match_label] > self.max_bets_per_match:
                continue

            model_prob = bet.get("model_prob", 0.0)
            decimal_odds = bet.get("decimal_odds", 0.0)
            actual_result = bet.get("actual_result", False)

            if decimal_odds <= 1.0 or model_prob <= 0.0:
                continue

            # ── Compute EV if not provided ──
            ev = bet.get("ev")
            if ev is None:
                ev_result = calculate_ev(model_prob, decimal_odds)
                ev = ev_result["ev"]
                edge = ev_result["edge_vs_market"]
            else:
                edge = model_prob - (1.0 / decimal_odds)

            # ── Compute stake ──
            stake_amount = self.stake_strategy.calculate_stake(
                model_prob,
                decimal_odds,
                self._bankroll,
            )

            if stake_amount <= 0:
                continue

            stake_pct = stake_amount / max(self._bankroll, 0.01)

            # ── Settle bet ──
            if actual_result:
                profit = stake_amount * (decimal_odds - 1.0)
            else:
                profit = -stake_amount

            bankroll_before = self._bankroll
            self._bankroll += profit

            if self._bankroll > self._peak:
                self._peak = self._bankroll

            # ── CLV (if closing odds available) ──
            clv: float | None = None
            closing_odds = bet.get("closing_odds")
            if closing_odds is not None and closing_odds > 1.0:
                opening_implied = 1.0 / decimal_odds
                closing_implied = 1.0 / closing_odds
                clv = closing_implied - opening_implied

            # ── Record ──
            record = BacktestBetRecord(
                match_label=match_label,
                outcome=outcome_label,
                market=market,
                decimal_odds=round(decimal_odds, 4),
                model_prob=round(model_prob, 4),
                ev=round(ev, 4),
                edge_vs_market=round(edge, 4),
                stake_amount=round(stake_amount, 2),
                stake_pct=round(stake_pct, 6),
                profit=round(profit, 2),
                won=actual_result,
                clv=round(clv, 4) if clv is not None else None,
                bankroll_before=round(bankroll_before, 2),
                bankroll_after=round(self._bankroll, 2),
                timestamp=int(datetime.now(timezone.utc).timestamp()),
            )
            self._bets.append(record)

        self._metrics = self._compute_metrics()

        logger.info(
            "Backtest complete — %d bets placed, P&L=%.2f, ROI=%.2f%%",
            self._metrics.total_bets,
            self._metrics.total_profit,
            self._metrics.roi_pct,
        )

        return self._metrics

    # ── Metrics computation ─────────────────────────────

    def _compute_metrics(self) -> BacktestMetrics:
        """Compute all performance metrics from the bet history."""
        m = BacktestMetrics(
            initial_bankroll=self.initial_bankroll,
            final_bankroll=self._bankroll,
            total_bets=len(self._bets),
            stake_strategy=self.stake_strategy.__class__.__name__,
            bet_filter_config={
                "min_ev": self.bet_filter.min_ev,
                "min_confidence": self.bet_filter.min_confidence,
                "min_odds": self.bet_filter.min_odds,
                "max_stake": self.bet_filter.max_stake,
                "markets": list(self.bet_filter.markets),
            },
            started_at=self._start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_seconds=(
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds(),
        )

        if m.total_bets == 0:
            m.bankroll_history = [self.initial_bankroll, self._bankroll]
            m.drawdown_history = [0.0, 0.0]
            return m

        # Basic counts
        m.winning_bets = sum(1 for b in self._bets if b.won)
        m.losing_bets = m.total_bets - m.winning_bets
        m.total_staked = sum(b.stake_amount for b in self._bets)
        m.total_profit = sum(b.profit for b in self._bets)

        # ROI
        m.roi_pct = (m.final_bankroll - m.initial_bankroll) / m.initial_bankroll * 100

        # Yield
        m.yield_pct = (
            (m.total_profit / m.total_staked * 100) if m.total_staked > 0 else 0.0
        )

        # Win rate
        m.win_rate_pct = (m.winning_bets / m.total_bets) * 100

        # Averages
        m.avg_odds = float(np.mean([b.decimal_odds for b in self._bets]))
        m.avg_ev = float(np.mean([b.ev for b in self._bets]))
        m.avg_stake_pct = float(np.mean([b.stake_pct for b in self._bets]))

        # Profit factor
        gross_profit = sum(b.profit for b in self._bets if b.profit > 0)
        gross_loss = abs(sum(b.profit for b in self._bets if b.profit < 0))
        m.profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
            if gross_profit > 0
            else 0.0
        )

        # CLV
        clv_values = [b.clv for b in self._bets if b.clv is not None]
        if clv_values:
            m.avg_clv = float(np.mean(clv_values))
            m.positive_clv_pct = (
                sum(1 for v in clv_values if v > 0) / len(clv_values) * 100
            )

        # Bankroll history & drawdown
        m.bankroll_history = self._build_bankroll_history()
        m.drawdown_history = self._build_drawdown_history(m.bankroll_history)

        peak = m.bankroll_history[0]
        max_dd = 0.0
        max_dd_amount = 0.0
        for value in m.bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak > 0 else 0.0
            dd_amount = peak - value
            if dd > max_dd:
                max_dd = dd
                max_dd_amount = dd_amount
        m.max_drawdown_pct = max_dd
        m.max_drawdown_amount = max_dd_amount

        # Equity curve (cumulative P&L)
        m.equity_curve = self._build_equity_curve()

        # Sharpe ratio (using bet-level returns)
        returns = np.array(
            [b.profit / max(b.bankroll_before, 0.01) for b in self._bets]
        )
        if len(returns) > 1 and np.std(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)  # sample std
            # Annualised Sharpe (assuming ~500 bets/year)
            m.sharpe_ratio = float(
                (mean_return / std_return) * math.sqrt(500) if std_return > 0 else 0.0
            )

            # Sortino ratio (downside deviation only)
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0:
                downside_std = (
                    np.std(neg_returns, ddof=1)
                    if len(neg_returns) > 1
                    else np.std(neg_returns)
                )
                m.sortino_ratio = float(
                    (mean_return / downside_std) * math.sqrt(500)
                    if downside_std > 0
                    else 0.0
                )

        # Streaks
        m.longest_win_streak, m.longest_lose_streak = self._compute_streaks()
        m.current_win_streak = self._current_streak(won=True)
        m.current_lose_streak = self._current_streak(won=False)

        return m

    # ── Reporting ───────────────────────────────────────

    def print_report(self) -> None:
        """Print a formatted backtest report to the console."""
        m = self._metrics or BacktestMetrics()

        logger.info("")
        logger.info("=" * 80)
        logger.info("  BACKTEST REPORT".center(78))
        logger.info("=" * 80)

        if m.total_bets == 0:
            logger.info("  No bets were placed during the backtest period.")
            logger.info("  Final bankroll: £%.2f", m.final_bankroll)
            logger.info("=" * 80)
            return

        logger.info("  %-30s %15s   %-33s", "Metric", "Value", "Notes")
        logger.info("  %s", "-" * 78)

        # P&L section
        logger.info("  %-30s %10d   %20s", "TOTAL BETS", m.total_bets, "bets placed")
        logger.info(
            "  %-30s %8d / %-8d  %20s", "Winning", m.winning_bets, m.total_bets, ""
        )
        logger.info("  %-30s %13.1f%%   %20s", "Win Rate", m.win_rate_pct, "")

        profit_color = "+" if m.total_profit >= 0 else ""
        logger.info(
            "  %-30s %s£%+11.2f   %20s", "Total P&L", profit_color, m.total_profit, ""
        )
        logger.info("  %-30s £%11.2f   %20s", "Total Staked", m.total_staked, "")

        roi_color = "+" if m.roi_pct >= 0 else ""
        logger.info("  %-30s %s%+11.2f%%   %20s", "ROI", roi_color, m.roi_pct, "")
        yield_color = "+" if m.yield_pct >= 0 else ""
        logger.info(
            "  %-30s %s%+11.2f%%   %20s",
            "Yield",
            yield_color,
            m.yield_pct,
            "profit per unit staked",
        )

        bankroll_change = m.final_bankroll - m.initial_bankroll
        change_sign = "+" if bankroll_change >= 0 else ""
        logger.info(
            "  %-30s £%11.2f   (%s£%.2f from £%.0f)",
            "Final Bankroll",
            m.final_bankroll,
            change_sign,
            bankroll_change,
            m.initial_bankroll,
        )

        # Risk section
        logger.info(
            "  %-30s %12.2f%%   (£%.2f peak-to-trough)",
            "Max Drawdown",
            m.max_drawdown_pct,
            m.max_drawdown_amount,
        )
        logger.info(
            "  %-30s %14.2f   %20s",
            "Sharpe Ratio",
            m.sharpe_ratio,
            "risk-adjusted return",
        )
        logger.info(
            "  %-30s %14.2f   %20s",
            "Sortino Ratio",
            m.sortino_ratio,
            "downside risk-adjusted",
        )

        # Quality section
        logger.info(
            "  %-30s %14.2f   %20s",
            "Profit Factor",
            m.profit_factor,
            "gross profit / gross loss",
        )
        logger.info("  %-30s %14.4f   %20s", "Avg Odds", m.avg_odds, "")
        logger.info("  %-30s %+14.2f%%   %20s", "Avg EV", m.avg_ev * 100, "")

        if m.avg_clv != 0.0:
            logger.info(
                "  %-30s %+14.4f   %.0f%% positive bets",
                "Avg CLV",
                m.avg_clv,
                m.positive_clv_pct,
            )

        # Streaks
        logger.info(
            "  %-30s %8d bets   %20s", "Longest Win Streak", m.longest_win_streak, ""
        )
        logger.info(
            "  %-30s %8d bets   %20s", "Longest Lose Streak", m.longest_lose_streak, ""
        )

        # Performance assessment
        logger.info("  ASSESSMENT")
        logger.info("  %s", self._assess_performance(m))
        logger.info("=" * 80)

    def _assess_performance(self, m: BacktestMetrics) -> str:
        """Generate a human-readable performance assessment."""
        lines = []
        if m.sharpe_ratio >= 2.0:
            lines.append("  ✅ Excellent risk-adjusted returns (Sharpe ≥ 2.0)")
        elif m.sharpe_ratio >= 1.0:
            lines.append("  ✅ Good risk-adjusted returns (Sharpe ≥ 1.0)")
        elif m.sharpe_ratio >= 0.5:
            lines.append("  ⚠ Moderate risk-adjusted returns (Sharpe 0.5–1.0)")
        else:
            lines.append("  🔴 Poor risk-adjusted returns (Sharpe < 0.5)")

        if m.max_drawdown_pct < 10:
            lines.append(
                f"  ✅ Low drawdown ({m.max_drawdown_pct:.1f}%) — good risk mgmt"
            )
        elif m.max_drawdown_pct < 25:
            lines.append(f"  ⚠ Moderate drawdown ({m.max_drawdown_pct:.1f}%)")
        else:
            lines.append(
                f"  🔴 High drawdown ({m.max_drawdown_pct:.1f}%) — risk of ruin"
            )

        if m.profit_factor >= 2.0:
            lines.append("  ✅ Profit factor ≥ 2.0 — strong risk/reward")
        elif m.profit_factor >= 1.5:
            lines.append("  ✅ Profit factor ≥ 1.5 — solid")
        elif m.profit_factor >= 1.0:
            lines.append("  ⚠ Profit factor ≥ 1.0 — marginal")
        else:
            lines.append("  🔴 Profit factor < 1.0 — losses exceed gains")

        if m.roi_pct > 0:
            lines.append(f"  ✅ Profitable: {m.roi_pct:+.1f}% ROI")
        elif m.roi_pct == 0:
            lines.append("  ⚪ Break-even")
        else:
            lines.append(f"  🔴 Loss-making: {m.roi_pct:+.1f}% ROI")

        if m.total_bets < 100:
            lines.append(
                f"  ⚠ Small sample ({m.total_bets} bets) — results may not be significant"
            )
        elif m.total_bets < 500:
            lines.append(f"  📊 Moderate sample ({m.total_bets} bets)")
        else:
            lines.append(f"  📊 Large sample ({m.total_bets} bets) — more reliable")

        return "\n".join(lines)

    # ── Results export ──────────────────────────────────

    def save_results(
        self,
        output_dir: str | Path = "reports/backtest",
        model_name: str = "unknown",
    ) -> str:
        """Save backtest results to a JSON file.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save results to (default ``reports/backtest/``).
        model_name : str
            Model identifier to include in the filename.

        Returns
        -------
        str
            Path to the saved JSON file.
        """
        m = self._metrics or self._compute_metrics()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{model_name}_{timestamp}.json"
        filepath = out_dir / filename

        # Convert to serializable dict
        data = {
            "model_name": model_name,
            "metrics": asdict(m),
            "bets": [asdict(b) for b in self._bets],
            "configuration": {
                "initial_bankroll": self.initial_bankroll,
                "stake_strategy": self.stake_strategy.__class__.__name__,
                "bet_filter": {
                    "min_ev": self.bet_filter.min_ev,
                    "min_confidence": self.bet_filter.min_confidence,
                    "min_odds": self.bet_filter.min_odds,
                    "max_stake": self.bet_filter.max_stake,
                    "markets": list(self.bet_filter.markets),
                },
                "max_bets_per_match": self.max_bets_per_match,
            },
        }

        # Handle special float values
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

        logger.info("Backtest results saved to %s", filepath)
        return str(filepath)

    # ── Properties ──────────────────────────────────────

    @property
    def metrics(self) -> BacktestMetrics:
        """Return the computed metrics, or empty metrics if not run yet."""
        if self._metrics is None:
            return BacktestMetrics()
        return self._metrics

    @property
    def bets(self) -> list[BacktestBetRecord]:
        """Return the list of placed bets."""
        return list(self._bets)

    # ── Internal helpers ────────────────────────────────

    def _build_bankroll_history(self) -> list[float]:
        """Build a chronological bankroll history."""
        history = [self.initial_bankroll]
        for bet in self._bets:
            history.append(bet.bankroll_after)
        if len(history) == 1:
            history.append(self._bankroll)
        return history

    def _build_drawdown_history(
        self,
        history: list[float],
    ) -> list[float]:
        """Compute drawdown % for each point in bankroll history."""
        if not history:
            return []
        peak = history[0]
        drawdowns: list[float] = []
        for value in history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak > 0 else 0.0
            drawdowns.append(dd)
        return drawdowns

    def _build_equity_curve(self) -> list[float]:
        """Build cumulative P&L equity curve."""
        curve = [0.0]
        cumulative = 0.0
        for bet in self._bets:
            cumulative += bet.profit
            curve.append(round(cumulative, 2))
        return curve

    def _compute_streaks(self) -> tuple[int, int]:
        """Return (longest_win_streak, longest_lose_streak)."""
        if not self._bets:
            return 0, 0
        win_streak = 0
        lose_streak = 0
        max_win = 0
        max_lose = 0
        for b in self._bets:
            if b.won:
                win_streak += 1
                lose_streak = 0
                max_win = max(max_win, win_streak)
            else:
                lose_streak += 1
                win_streak = 0
                max_lose = max(max_lose, lose_streak)
        return max_win, max_lose

    def _current_streak(self, won: bool) -> int:
        """Return the current ongoing streak."""
        count = 0
        for b in reversed(self._bets):
            if b.won == won:
                count += 1
            else:
                break
        return count
