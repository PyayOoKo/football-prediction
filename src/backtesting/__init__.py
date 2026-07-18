"""
Backtesting Engine — simulate value betting on historical match data.

Walks through test-set matches chronologically, computes value bets using
model probabilities vs actual bookmaker odds, places Kelly-sized stakes,
and tracks bankroll performance.

Metrics
-------
**ROI (Return on Investment)**
    ``(final_bankroll - initial_bankroll) / initial_bankroll × 100``
    Total percentage return — the headline number every bettor looks at.

**Yield**
    ``total_profit / total_staked × 100``
    Profit per unit staked.  A yield of 5% means for every £1 wagered
    you expect £0.05 profit.  This normalises for the number of bets.

**Profit**
    ``final_bankroll - initial_bankroll``
    Absolute profit/loss in currency units.  Simple but doesn't account
    for stake volume.

**Win Rate**
    ``winning_bets / total_bets × 100``
    Percentage of bets that won.  A low win rate with high odds can still
    be profitable (e.g. 30% win rate on 4.0 odds → positive EV).

**Maximum Drawdown**
    ``max(peak - trough) / peak × 100``
    Largest peak-to-trough decline in bankroll.  Measures the worst
    losing streak.  Critical for bankroll management — if drawdown
    exceeds bankroll, the strategy is ruined.

Usage
-----
::

    from src.backtesting import BacktestEngine

    engine = BacktestEngine(model, initial_bankroll=1000.0)
    results = engine.run(X_test, y_test, odds_df=odds_df)
    metrics = engine.calculate_metrics()
    engine.print_report()
    engine.plot_results(output_dir="reports/backtest")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if matplotlib.get_backend() in ("", None):
    matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ── Style configuration (consistent with EDA module) ────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class BetRecord:
    """A single placed bet during the backtest simulation."""

    match_index: int
    match_label: str
    outcome_bet: str          # "Away Win" | "Draw" | "Home Win"
    outcome_actual: str       # Actual result label
    decimal_odds: float
    model_prob: float
    fair_prob: float
    ev: float
    stake_pct: float          # Fraction of bankroll staked
    stake_amount: float       # Actual currency amount staked
    profit: float             # Profit/loss from this bet (negative = loss)
    won: bool                 # True if bet was a winner
    bankroll_before: float
    bankroll_after: float


@dataclass
class BacktestMetrics:
    """Aggregate performance metrics from a backtest run."""

    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    final_bankroll: float = 0.0
    roi_pct: float = 0.0
    yield_pct: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    profit_factor: float = 0.0  # gross_profit / gross_loss
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    kelly_fraction: float = 0.25
    initial_bankroll: float = 1000.0
    bankroll_history: list[float] = field(default_factory=list)
    drawdown_history: list[float] = field(default_factory=list)
    periods: int = 0


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


class BacktestEngine:
    """Simulate a value-betting strategy over historical test data.

    Parameters
    ----------
    model : Any
        Trained classifier with ``predict_proba`` returning
        ``[away_prob, draw_prob, home_prob]`` (class order 0→away, 1→draw, 2→home).
    initial_bankroll : float
        Starting bankroll in currency units (default 1000).
    kelly_fraction : float
        Fraction of full Kelly to use (default 0.25 = conservative).
    min_ev : float
        Minimum EV threshold (default 0.0).
    bet_on : list[str] | None
        Which outcomes to consider betting on.  Default ``None`` = all.
        Example: ``["Home Win", "Draw"]``.
    """

    def __init__(
        self,
        model: Any,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_ev: float = 0.0,
        bet_on: list[str] | None = None,
    ) -> None:
        self.model = model
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_ev = min_ev
        self.bet_on = bet_on or ["Away Win", "Draw", "Home Win"]

        self._bets: list[BetRecord] = []
        self._bankroll: float = initial_bankroll
        self._peak: float = initial_bankroll
        self._metrics: BacktestMetrics | None = None

        # Outcome mapping: class index → short label → full label
        self._idx_to_label = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        self._idx_to_short = {0: "A", 1: "D", 2: "H"}
        self._label_to_idx = {v: k for k, v in self._idx_to_label.items()}

        logger.info(
            "BacktestEngine initialised — bankroll=%.0f, kelly=%.0f%%, min_ev=%.2f",
            initial_bankroll, kelly_fraction * 100, min_ev,
        )

    # ── Run simulation ──────────────────────────────────

    def run(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series | np.ndarray,
        odds_df: pd.DataFrame | None = None,
        odds_cols: tuple[str, str, str] = ("BbAvA", "BbAvD", "BbAvH"),
        team_cols: tuple[str, str] = ("home_team", "away_team"),
    ) -> BacktestMetrics:
        """Run the backtest simulation on test data.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix (from chronological train/val/test split).
        y_test : pd.Series | np.ndarray
            True outcomes: 0 = Away Win, 1 = Draw, 2 = Home Win.
        odds_df : pd.DataFrame, optional
            DataFrame with odds columns, aligned by index to ``X_test``.
            Must contain *odds_cols* if provided.
        odds_cols : tuple[str, str, str]
            Column names for ``(away_odds, draw_odds, home_odds)``.
            Default: ``("BbAvA", "BbAvD", "BbAvH")``.
        team_cols : tuple[str, str]
            Column names for ``(home_team, away_team)``.
            Default: ``("home_team", "away_team")``.

        Returns
        -------
        BacktestMetrics
            Aggregate performance metrics.
        """
        # Get model probabilities — shape (n_test, 3) [away, draw, home]
        y_proba = self.model.predict_proba(X_test)
        n_matches = len(X_test)

        logger.info("Running backtest on %d test matches", n_matches)

        # Reset state
        self._bets = []
        self._bankroll = self.initial_bankroll
        self._peak = self.initial_bankroll

        # Check odds availability
        has_odds = odds_df is not None
        if has_odds:
            missing_odds = [c for c in odds_cols if c not in odds_df.columns]
            if missing_odds:
                logger.warning("Odds columns not found in odds_df: %s", missing_odds)
                has_odds = False

        if not has_odds:
            logger.warning("No odds data available — backtest will use simulated flat odds")
            # Create synthetic odds from model probabilities (no value possible)
            # This allows the engine to run even without real odds
            synthetic_odds = np.array([
                [1.0 / p if p > 0.01 else 100.0 for p in row]
                for row in y_proba
            ])

        for i in range(n_matches):
            match_probs = y_proba[i]  # [away, draw, home]
            actual_idx = y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i]
            actual_label = self._idx_to_label.get(int(actual_idx), str(actual_idx))

            # Build match label
            home_team = ""
            away_team = ""
            if has_odds:
                if team_cols[0] in odds_df.columns:
                    home_team = str(odds_df.iloc[i].get(team_cols[0], ""))
                if team_cols[1] in odds_df.columns:
                    away_team = str(odds_df.iloc[i].get(team_cols[1], ""))
            match_label = f"{home_team} vs {away_team}" if home_team and away_team else f"Match {i + 1}"

            # Get odds for this match
            if has_odds:
                raw_odds = [
                    float(odds_df.iloc[i][odds_cols[0]]),  # away
                    float(odds_df.iloc[i][odds_cols[1]]),  # draw
                    float(odds_df.iloc[i][odds_cols[2]]),  # home
                ]
            else:
                raw_odds = synthetic_odds[i].tolist()

            # Handle NaN odds — skip bet evaluation for this match
            if any(np.isnan(raw_odds)):
                continue

            # ── Compute implied probabilities & margin ──
            implied = 1.0 / np.array(raw_odds)
            margin = implied.sum() - 1.0
            fair = implied / (1.0 + margin)

            # ── Evaluate each outcome for value ──────────
            for j, outcome_label in enumerate(["Away Win", "Draw", "Home Win"]):
                if outcome_label not in self.bet_on:
                    continue

                dec_odds = raw_odds[j]
                mod_prob = match_probs[j]
                fair_prob = fair[j]
                ev = (mod_prob * dec_odds) - 1.0

                # Kelly stake
                if dec_odds > 1.0 and ev > 0:
                    full_kelly = ev / (dec_odds - 1.0)
                else:
                    full_kelly = 0.0
                kelly_pct = max(full_kelly * self.kelly_fraction, 0.0)

                # Positive EV check
                is_value = bool(
                    ev >= self.min_ev
                    and mod_prob > fair_prob
                    and kelly_pct > 0.0
                )

                if is_value:
                    stake_amount = self._bankroll * kelly_pct
                    won = (j == int(actual_idx))

                    if won:
                        profit = stake_amount * (dec_odds - 1.0)
                    else:
                        profit = -stake_amount

                    bankroll_before = self._bankroll
                    self._bankroll += profit

                    # Track peak for drawdown
                    if self._bankroll > self._peak:
                        self._peak = self._bankroll

                    record = BetRecord(
                        match_index=i,
                        match_label=match_label,
                        outcome_bet=outcome_label,
                        outcome_actual=actual_label,
                        decimal_odds=round(dec_odds, 4),
                        model_prob=round(mod_prob, 4),
                        fair_prob=round(fair_prob, 4),
                        ev=round(ev, 4),
                        stake_pct=round(kelly_pct, 6),
                        stake_amount=round(stake_amount, 2),
                        profit=round(profit, 2),
                        won=won,
                        bankroll_before=round(bankroll_before, 2),
                        bankroll_after=round(self._bankroll, 2),
                    )
                    self._bets.append(record)

            # No bet placed for this match — bankroll stays unchanged

        logger.info(
            "Backtest complete — %d bets placed from %d matches",
            len(self._bets), n_matches,
        )
        self._metrics = self.calculate_metrics()
        return self._metrics

    # ── Metrics ─────────────────────────────────────────

    def calculate_metrics(self) -> BacktestMetrics:
        """Compute aggregate performance metrics from the bet history.

        Returns
        -------
        BacktestMetrics
        """
        m = BacktestMetrics(
            initial_bankroll=self.initial_bankroll,
            kelly_fraction=self.kelly_fraction,
            total_bets=len(self._bets),
        )

        if m.total_bets == 0:
            m.final_bankroll = self._bankroll
            m.bankroll_history = [self.initial_bankroll, self._bankroll]
            m.drawdown_history = [0.0, 0.0]
            return m

        # Basic counts
        m.winning_bets = sum(1 for b in self._bets if b.won)
        m.losing_bets = m.total_bets - m.winning_bets
        m.total_staked = sum(b.stake_amount for b in self._bets)
        m.total_profit = sum(b.profit for b in self._bets)
        m.final_bankroll = self._bankroll

        # ROI
        m.roi_pct = ((m.final_bankroll - m.initial_bankroll) / m.initial_bankroll) * 100

        # Yield
        m.yield_pct = (m.total_profit / m.total_staked * 100) if m.total_staked > 0 else 0.0

        # Win rate
        m.win_rate_pct = (m.winning_bets / m.total_bets) * 100

        # Average odds & EV
        m.avg_odds = float(np.mean([b.decimal_odds for b in self._bets]))
        m.avg_ev = float(np.mean([b.ev for b in self._bets]))

        # Profit factor: gross profit / gross loss
        gross_profit = sum(b.profit for b in self._bets if b.profit > 0)
        gross_loss = abs(sum(b.profit for b in self._bets if b.profit < 0))
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        m.bankroll_history = self._build_bankroll_history()
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

        # Drawdown history
        m.drawdown_history = self._build_drawdown_history()

        # Streaks
        m.longest_win_streak = self._longest_streak(won=True)
        m.longest_lose_streak = self._longest_streak(won=False)

        self._metrics = m
        return m

    # ── Reporting ───────────────────────────────────────

    def print_report(self) -> None:
        """Pretty-print the backtest results to the console."""

        m = self._metrics or self.calculate_metrics()

        print("\n" + "=" * 90)
        print("  BACKTEST RESULTS".center(88))
        print("=" * 90)

        if m.total_bets == 0:
            print("\n  No bets were placed during the backtest period.")
            print("     Possible causes:")
            print("     * No odds data available for the test period")
            print("     * Model found no positive-EV opportunities")
            print("     * The min_ev threshold may be too high")
            print(f"\n  Final bankroll: GBP{m.final_bankroll:.2f}")
            print("=" * 90)
            return

        # ── Summary metrics ─────────────────────────────
        print(f"\n  {'METRIC':<30} {'VALUE':>15}   {'NOTES':<35}")
        print(f"  {'-' * 82}")
        print(f"  {'Total bets':<30} {m.total_bets:>10,d}   {'Bets placed during test period':<35}")
        print(f"  {'Winning bets':<30} {m.winning_bets:>8,d} / {m.total_bets:<8,d}    {'':<35}")
        win_rate_str = f"{m.win_rate_pct:.1f}%"
        print(f"  {'Win rate':<30} {win_rate_str:>15}   {'% of bets that won':<35}")
        print(f"  {'Total staked':<30} GBP{m.total_staked:>12.2f}   {'Sum of all stakes':<35}")
        print(f"  {'Total profit / loss':<30} GBP{m.total_profit:>+11.2f}   {'+ = profit, - = loss':<35}")

        # ROI
        roi_marker = "+" if m.roi_pct > 0 else "-" if m.roi_pct < 0 else "="
        print(f"  {'ROI (Return on Investment)':<30} {roi_marker} {m.roi_pct:>+9.2f}%   {'Total return on initial bankroll':<35}")

        # Yield
        yield_marker = "+" if m.yield_pct > 0 else "-" if m.yield_pct < 0 else "="
        print(f"  {'Yield (profit / staked)':<30} {yield_marker} {m.yield_pct:>+9.2f}%   {'Return per unit staked':<35}")

        # Final bankroll
        bankroll_change = m.final_bankroll - m.initial_bankroll
        change_str = f"+GBP{bankroll_change:.2f}" if bankroll_change >= 0 else f"-GBP{abs(bankroll_change):.2f}"
        print(f"  {'Final bankroll':<30} GBP{m.final_bankroll:>11.2f}   ({change_str} from GBP{m.initial_bankroll:.0f})")

        # Drawdown
        print(f"  {'Max drawdown':<30} {m.max_drawdown_pct:>12.2f}%   (GBP{m.max_drawdown_amount:.2f} peak-to-trough)")

        # Other metrics
        print(f"  {'Average odds':<30} {m.avg_odds:>14.4f}   {'Weighted by stake':<35}")
        print(f"  {'Average EV':<30} {m.avg_ev:>+14.2%}   {'Expected value per bet':<35}")
        print(f"  {'Profit factor':<30} {m.profit_factor:>14.2f}   {'Gross profit / gross loss':<35}")
        print(f"  {'Longest win streak':<30} {m.longest_win_streak:>8d} bets   {'':<35}")
        print(f"  {'Longest losing streak':<30} {m.longest_lose_streak:>8d} bets   {'':<35}")

        # ── Performance assessment ──────────────────────
        print(f"\n  {'PERFORMANCE ASSESSMENT':-^82}")
        assessment = self._assess_performance(m)
        print(f"  {assessment}")
        print("=" * 90)

    def _assess_performance(self, m: BacktestMetrics) -> str:
        """Return a human-readable performance assessment."""
        parts = []
        if m.roi_pct > 0:
            parts.append(f"(+) Profitable strategy with {m.roi_pct:+.1f}% ROI")
        elif m.roi_pct == 0:
            parts.append("(=) Break-even -- no profit or loss")
        else:
            parts.append(f"(-) Loss-making strategy with {m.roi_pct:+.1f}% ROI")

        if m.max_drawdown_pct < 10:
            parts.append(f"(/) Low drawdown ({m.max_drawdown_pct:.1f}%) -- good risk management")
        elif m.max_drawdown_pct < 25:
            parts.append(f"(!) Moderate drawdown ({m.max_drawdown_pct:.1f}%) -- acceptable")
        else:
            parts.append(f"(!) High drawdown ({m.max_drawdown_pct:.1f}%) -- high risk of ruin")

        if m.profit_factor >= 2.0:
            parts.append("(/) Profit factor >= 2.0 -- strong risk/reward")
        elif m.profit_factor >= 1.0:
            parts.append("(~) Profit factor >= 1.0 -- marginal profitability")
        else:
            parts.append("(!) Profit factor < 1.0 -- losses exceed gains")

        if m.longest_lose_streak > 15:
            parts.append(f"(!) Long losing streak ({m.longest_lose_streak} bets) -- test psychological resilience")
        elif m.longest_lose_streak > 8:
            parts.append(f"(~) Losing streak of {m.longest_lose_streak} bets -- within normal variance")

        return "  * " + "\n  * ".join(parts)

    # ── Charts ──────────────────────────────────────────

    def plot_results(
        self,
        output_dir: str | Path = "reports/backtest",
        show: bool = False,
    ) -> dict[str, str]:
        """Generate 4 backtest visualisation charts.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save charts to (default ``reports/backtest/``).
        show : bool
            If True, display each figure via ``plt.show()``.

        Returns
        -------
        dict[str, str]
            Mapping of chart name to file path.
        """
        m = self._metrics or self.calculate_metrics()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict[str, str] = {}

        try:
            path = self._plot_bankroll_curve(m, out)
            if path:
                paths["bankroll_curve"] = path
        except Exception as exc:
            logger.warning("Bankroll curve chart failed: %s", exc)

        try:
            path = self._plot_drawdown(m, out)
            if path:
                paths["drawdown"] = path
        except Exception as exc:
            logger.warning("Drawdown chart failed: %s", exc)

        try:
            path = self._plot_cumulative_profit(m, out)
            if path:
                paths["cumulative_profit"] = path
        except Exception as exc:
            logger.warning("Cumulative profit chart failed: %s", exc)

        try:
            path = self._plot_bet_outcomes(m, out)
            if path:
                paths["bet_outcomes"] = path
        except Exception as exc:
            logger.warning("Bet outcomes chart failed: %s", exc)

        if show:
            plt.show()

        logger.info("Backtest charts saved to %s", out)
        return paths

    def _plot_bankroll_curve(
        self, m: BacktestMetrics, out_dir: Path,
    ) -> str | None:
        """Chart 1: Bankroll growth over time."""
        fig, ax = plt.subplots(figsize=(12, 5))

        history = m.bankroll_history
        x = range(len(history))

        # Main line
        color = "#2ecc71" if m.roi_pct >= 0 else "#e74c3c"
        ax.plot(x, history, color=color, linewidth=1.5, alpha=0.9)
        ax.fill_between(x, self.initial_bankroll, history,
                        where=np.array(history) >= self.initial_bankroll,
                        color="#2ecc71", alpha=0.1, label="Above starting bankroll")
        ax.fill_between(x, self.initial_bankroll, history,
                        where=np.array(history) < self.initial_bankroll,
                        color="#e74c3c", alpha=0.1, label="Below starting bankroll")

        # Starting bankroll line
        ax.axhline(self.initial_bankroll, color="#555555", linestyle="--",
                   linewidth=0.8, alpha=0.6, label=f"Starting bankroll (GBP{self.initial_bankroll:.0f})")

        ax.set_xlabel("Bet Number (chronological)")
        ax.set_ylabel("Bankroll (GBP)")
        ax.set_title("Backtest -- Bankroll Growth Curve", fontweight="bold", fontsize=14)
        ax.legend(fontsize=9, loc="upper left")

        # Annotations
        end_val = history[-1] if history else self.initial_bankroll
        ax.text(0.98, 0.05,
                f"Start: GBP{self.initial_bankroll:.2f}\n"
                f"Final: GBP{end_val:.2f}\n"
                f"ROI: {m.roi_pct:+.2f}%\n"
                f"Bets: {m.total_bets}",
                transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

        plt.tight_layout()
        path = str(out_dir / "01_bankroll_curve.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _plot_drawdown(
        self, m: BacktestMetrics, out_dir: Path,
    ) -> str | None:
        """Chart 2: Drawdown (peak-to-trough decline) over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})

        history = m.bankroll_history
        drawdowns = m.drawdown_history
        x = range(len(history))

        # Upper: Bankroll with peaks marked
        peaks = self._compute_peaks(history)
        ax1.plot(x, history, color="#2c3e50", linewidth=1.2, alpha=0.8, label="Bankroll")
        ax1.scatter([p[0] for p in peaks], [p[1] for p in peaks],
                    color="#2ecc71", s=30, zorder=5, marker="^", label="Peak")
        ax1.fill_between(x, [self.initial_bankroll] * len(history), history,
                         color="#3498db", alpha=0.08)
        ax1.set_ylabel("Bankroll (GBP)")
        ax1.set_title("Drawdown Analysis -- Peak-to-Trough Decline", fontweight="bold", fontsize=14)
        ax1.legend(fontsize=9, loc="upper left")

        # Lower: Drawdown %
        ax2.fill_between(x, 0, drawdowns, color="#e74c3c", alpha=0.5, step="mid")
        ax2.plot(x, drawdowns, color="#c0392b", linewidth=1.0, alpha=0.8)
        ax2.set_xlabel("Bet Number (chronological)")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(bottom=0)

        # Annotate max drawdown
        max_dd_idx = int(np.argmax(drawdowns)) if drawdowns else 0
        ax2.annotate(
            f"Max DD: {m.max_drawdown_pct:.1f}%",
            xy=(max_dd_idx, m.max_drawdown_pct),
            xytext=(max_dd_idx, m.max_drawdown_pct * 1.3),
            ha="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", alpha=0.8),
        )

        plt.tight_layout()
        path = str(out_dir / "02_drawdown.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _plot_cumulative_profit(
        self, m: BacktestMetrics, out_dir: Path,
    ) -> str | None:
        """Chart 3: Cumulative profit/loss as a step chart."""
        fig, ax = plt.subplots(figsize=(12, 5))

        if not self._bets:
            ax.text(0.5, 0.5, "No bets placed", ha="center", va="center", fontsize=14)
            plt.tight_layout()
            path = str(out_dir / "03_cumulative_profit.png")
            fig.savefig(path)
            plt.close(fig)
            return path

        cumulative = np.cumsum([b.profit for b in self._bets])
        x = range(len(cumulative))

        # Color segments by profit/loss
        for i in range(1, len(cumulative)):
            color = "#2ecc71" if cumulative[i] >= cumulative[i - 1] else "#e74c3c"
            ax.plot([i - 1, i], [cumulative[i - 1], cumulative[i]],
                    color=color, linewidth=1.5, alpha=0.8)

        ax.fill_between(x, 0, cumulative,
                        where=cumulative >= 0, color="#2ecc71", alpha=0.08,
                        label="Profit")
        ax.fill_between(x, 0, cumulative,
                        where=cumulative < 0, color="#e74c3c", alpha=0.08,
                        label="Loss")
        ax.axhline(0, color="#555555", linestyle="-", linewidth=0.6, alpha=0.5)

        # Markers for individual bets
        ax.scatter(x, cumulative, c=["#2ecc71" if b.won else "#e74c3c" for b in self._bets],
                   s=8, alpha=0.6, zorder=5)

        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Cumulative Profit/Loss (GBP)")
        ax.set_title("Backtest -- Cumulative Profit / Loss", fontweight="bold", fontsize=14)
        ax.legend(fontsize=9, loc="upper left")

        # Stats box
        final_profit = cumulative[-1] if len(cumulative) > 0 else 0
        ax.text(0.98, 0.05,
                f"Total profit: GBP{final_profit:+.2f}\n"
                f"Yield: {m.yield_pct:+.2f}%\n"
                f"Win rate: {m.win_rate_pct:.1f}%\n"
                f"Profit factor: {m.profit_factor:.2f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

        plt.tight_layout()
        path = str(out_dir / "03_cumulative_profit.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    def _plot_bet_outcomes(
        self, m: BacktestMetrics, out_dir: Path,
    ) -> str | None:
        """Chart 4: Bet outcomes -- scatter of win/loss with stake size."""
        fig, ax = plt.subplots(figsize=(12, 5))

        if not self._bets:
            ax.text(0.5, 0.5, "No bets placed", ha="center", va="center", fontsize=14)
            plt.tight_layout()
            path = str(out_dir / "04_bet_outcomes.png")
            fig.savefig(path)
            plt.close(fig)
            return path

        bets_df = pd.DataFrame([b.__dict__ for b in self._bets])

        # Separate wins and losses
        wins = bets_df[bets_df["won"]]
        losses = bets_df[~bets_df["won"]]

        # Calculate running bet index
        wins["bet_num"] = wins.index
        losses["bet_num"] = losses.index

        # Scatter plot
        ax.scatter(
            wins["bet_num"], wins["profit"],
            s=np.maximum(wins["stake_amount"].values * 2, 10),
            c="#2ecc71", alpha=0.6, edgecolors="white", linewidth=0.5,
            label=f"Wins ({len(wins)})",
        )
        ax.scatter(
            losses["bet_num"], losses["profit"],
            s=np.maximum(losses["stake_amount"].values * 2, 10),
            c="#e74c3c", alpha=0.6, edgecolors="white", linewidth=0.5,
            label=f"Losses ({len(losses)})",
        )
        ax.axhline(0, color="#555555", linestyle="-", linewidth=0.6, alpha=0.5)

        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Profit/Loss per Bet (GBP)")
        ax.set_title("Backtest -- Individual Bet Outcomes", fontweight="bold", fontsize=14)
        ax.legend(fontsize=9, loc="upper left")

        # Size legend note
        ax.text(0.98, 0.05,
                "Marker size prop stake amount\n"
                f"Bets: {m.total_bets}  |  "
                f"Avg stake: GBP{m.total_staked / m.total_bets:.1f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

        # Zero-crossing buffer
        y_margin = max(abs(bets_df["profit"]).max() * 0.15, 5)
        ax.set_ylim(-y_margin, y_margin)

        plt.tight_layout()
        path = str(out_dir / "04_bet_outcomes.png")
        fig.savefig(path)
        plt.close(fig)
        return path

    # ── Internal helpers ────────────────────────────────

    def _build_bankroll_history(self) -> list[float]:
        """Build a chronological bankroll history including non-betting periods."""
        history = [self.initial_bankroll]
        if not self._bets:
            return history
        # Collect unique match indices
        indices = sorted(set(b.match_index for b in self._bets))
        for i in indices:
            bets_at_match = [b for b in self._bets if b.match_index == i]
            if bets_at_match:
                # Multiple bets at same match (unlikely but handle)
                last = bets_at_match[-1].bankroll_after
            else:
                last = history[-1] if history else self.initial_bankroll
            history.append(last)
        return history

    def _build_drawdown_history(self) -> list[float]:
        """Compute drawdown % for each point in bankroll history."""
        history = self._build_bankroll_history()
        if not history:
            return []
        peak = history[0]
        drawdowns = []
        for value in history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100 if peak > 0 else 0.0
            drawdowns.append(dd)
        return drawdowns

    def _compute_peaks(self, history: list[float]) -> list[tuple[int, float]]:
        """Return (index, value) for all local peaks in the bankroll."""
        peaks = [(0, history[0])]
        for i in range(1, len(history) - 1):
            if history[i] > history[i - 1] and history[i] >= history[i + 1]:
                peaks.append((i, history[i]))
        if history:
            peaks.append((len(history) - 1, history[-1]))
        return peaks

    def _longest_streak(self, won: bool) -> int:
        """Return the longest consecutive streak of winning or losing bets."""
        if not self._bets:
            return 0
        longest = 0
        current = 0
        for b in self._bets:
            if b.won == won:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest


# ═══════════════════════════════════════════════════════════
#  Convenience wrapper
# ═══════════════════════════════════════════════════════════


def run_backtest(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    odds_df: pd.DataFrame | None = None,
    odds_cols: tuple[str, str, str] = ("BbAvA", "BbAvD", "BbAvH"),
    team_cols: tuple[str, str] = ("home_team", "away_team"),
    initial_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_ev: float = 0.0,
    output_dir: str | Path = "reports/backtest",
    print_report: bool = True,
    show_charts: bool = False,
) -> dict[str, Any]:
    """Run a complete backtest and return all results.

    Parameters
    ----------
    model : Any
        Trained classifier with ``predict_proba``.
    X_test, y_test : DataFrame / Series
        Test features and true labels.
    odds_df : DataFrame, optional
        DataFrame with bookmaker odds columns.
    odds_cols : tuple[str, str, str]
        Column names for ``(away_odds, draw_odds, home_odds)``.
    team_cols : tuple[str, str]
        Column names for ``(home_team, away_team)``.
    initial_bankroll : float
        Starting bankroll.
    kelly_fraction : float
        Fraction of Kelly to use.
    min_ev : float
        Minimum EV threshold.
    output_dir : str | Path
        Directory for saving charts.
    print_report : bool
        If True, pretty-print the results to console.
    show_charts : bool
        If True, display charts via ``plt.show()``.

    Returns
    -------
    dict[str, Any]
        ``{metrics, chart_paths, bets, engine}``
    """
    engine = BacktestEngine(
        model=model,
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        min_ev=min_ev,
    )

    metrics = engine.run(X_test, y_test, odds_df=odds_df,
                         odds_cols=odds_cols, team_cols=team_cols)

    chart_paths = engine.plot_results(output_dir=output_dir, show=show_charts)

    if print_report:
        engine.print_report()

    return {
        "metrics": metrics,
        "chart_paths": chart_paths,
        "bets": engine._bets,
        "engine": engine,
    }


# ═══════════════════════════════════════════════════════════
#  Calculation guide (for display)
# ═══════════════════════════════════════════════════════════


def get_backtest_guide() -> str:
    """Return a plain-text explanation of all backtest metrics."""
    return """
BACKTESTING -- METRIC GUIDE

1.  ROI (RETURN ON INVESTMENT)
    Formula:    ROI = (final_bankroll - initial_bankroll) / initial_bankroll x 100

    What it means:
    The total percentage return on your starting bankroll.
    ROI = +15% means you turned GBP1,000 into GBP1,150.
    This is the headline metric -- positive is good, negative is bad.

2.  YIELD
    Formula:    Yield = total_profit / total_staked x 100

    What it means:
    Return per unit staked.  If you bet GBP5,000 total and made GBP250 profit,
    your yield is 5%.  This normalises for bet volume -- a high ROI with
    tiny stakes may not be meaningful.

3.  PROFIT
    Formula:    Profit = final_bankroll - initial_bankroll

    What it means:
    Absolute profit/loss.  Simple, but doesn't tell you if you got lucky
    (one big win) or were consistently profitable (many small wins).

4.  WIN RATE
    Formula:    Win Rate = winning_bets / total_bets x 100

    What it means:
    What percentage of your bets won.  A low win rate (e.g. 30%) can
    still be profitable if your winners have high odds (e.g. 4.0+).
    Conversely, a 90% win rate on 1.1 odds is likely losing money.

5.  MAXIMUM DRAWDOWN
    Formula:    MDD = max(peak - trough) / peak x 100

    What it means:
    The largest peak-to-trough decline in your bankroll.  If your
    bankroll grew to GBP1,200 then fell to GBP900, that's a 25% drawdown.
    This is the most important risk metric -- a 50% drawdown requires
    a 100% gain just to break even.

6.  PROFIT FACTOR
    Formula:    Profit Factor = gross_profit / gross_loss

    What it means:
    The ratio of total winning profits to total losing losses.
    A profit factor of 2.0 means you make GBP2 for every GBP1 you lose.
    Below 1.0 means you lose more than you make (unprofitable).

7.  LONGEST LOSING STREAK
    What it means:
    The most consecutive losing bets.  Even a profitable strategy can
    have 10+ losses in a row due to variance.  Understanding your
    worst streak helps you set realistic bankroll requirements.
"""


# ── Re-export the new Backtester from the sub-module ────
from src.backtesting.backtester import Backtester  # noqa: E402, F401
