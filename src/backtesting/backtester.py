"""
Backtester — model-driven betting simulation with multi-market support.

Walks through historical match data chronologically, calls the model for
1X2 probabilities, evaluates value opportunities across multiple markets
(1X2, BTTS, Over/Under), applies configurable filters and stake sizing,
and tracks comprehensive bankroll metrics including pushed/void bets.

Features
--------
- **Model-driven 1X2** — uses ``model.predict_proba(features)`` for
  Home / Draw / Away probabilities
- **Additional markets** — BTTS, Over/Under via pre-computed probability
  columns in the match DataFrame
- **Pushed / void bets** — configurable void detection (abandoned matches,
  Draw-No-Bet scenarios, integer-line pushes)
- **Pluggable staking** — any ``StakingStrategy`` (Flat, Percentage,
  Kelly, FractionalKelly)
- **Pluggable filtering** — any ``BetFilter`` (EV, confidence, odds,
  market, stake thresholds)
- **Comprehensive metrics** — P&L, ROI, Yield, Win Rate, Max Drawdown,
  Sharpe, Sortino, Profit Factor, CLV, streaks

Usage
-----
::

    from src.backtesting.backtester import Backtester
    from src.betting.staking import FractionalKellyStaking
    from src.betting.filtering import BetFilter

    # Sample usage with a trained model
    import pandas as pd

    matches = pd.DataFrame({
        "home_team": ["Arsenal", "Liverpool"],
        "away_team": ["Chelsea", "Man City"],
        "home_goals": [2, 1],
        "away_goals": [1, 1],
        "BbAvH": [2.10, 2.50],
        "BbAvD": [3.40, 3.30],
        "BbAvA": [3.80, 3.00],
        "btts_model_prob": [0.65, 0.55],
        "BbAvBTTS_Yes": [1.80, 1.90],
        "over25_model_prob": [0.70, 0.50],
        "BbAvO25_Yes": [1.65, 2.00],
        "void": [False, False],
    })

    backtester = Backtester(
        model=my_model,  # must have predict_proba(X) -> [away, draw, home]
        initial_bankroll=1000.0,
        stake_strategy=FractionalKellyStaking(fraction=0.25),
        bet_filter=BetFilter(min_ev=0.0, min_confidence=0.3),
    )

    metrics = backtester.run(
        df=matches,
        feature_cols=["home_elo", "away_elo", ...],  # model features
        odds_mapping={
            "home_win": "BbAvH",
            "draw": "BbAvD",
            "away_win": "BbAvA",
            "btts_yes": "BbAvBTTS_Yes",
            "over25": "BbAvO25_Yes",
        },
        prob_mapping={
            "btts_yes": "btts_model_prob",
            "over25": "over25_model_prob",
        },
        void_col="void",
    )
    backtester.print_report()
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtesting import BacktestMetrics
from src.betting.ev import calculate_ev
from src.betting.staking import StakingStrategy, StakingFactory
from src.betting.filtering import BetFilter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Default market & settlement definitions
# ═══════════════════════════════════════════════════════════


def _settle_1x2(
    home_goals: float,
    away_goals: float,
    outcome: str,
) -> bool:
    """Settle a 1X2 market bet.

    Parameters
    ----------
    home_goals, away_goals : float
        Actual goals scored.
    outcome : str
        ``"home_win"``, ``"draw"``, or ``"away_win"``.

    Returns
    -------
    bool
        True if the bet wins.
    """
    if outcome == "home_win":
        return home_goals > away_goals
    if outcome == "draw":
        return home_goals == away_goals
    if outcome == "away_win":
        return away_goals > home_goals
    return False


def _settle_btts(home_goals: float, away_goals: float, outcome: str) -> bool:
    """Settle a Both Teams To Score market bet.

    Parameters
    ----------
    home_goals, away_goals : float
        Actual goals scored.
    outcome : str
        ``"yes"`` (both scored) or ``"no"`` (at least one didn't score).

    Returns
    -------
    bool
        True if the bet wins.
    """
    btts = home_goals > 0 and away_goals > 0
    if outcome == "yes":
        return btts
    if outcome == "no":
        return not btts
    return False


def _settle_over_under(
    home_goals: float,
    away_goals: float,
    outcome: str,
    line: float = 2.5,
) -> tuple[bool, bool]:
    """Settle an Over/Under market bet.

    Parameters
    ----------
    home_goals, away_goals : float
        Actual goals scored.
    outcome : str
        ``"over"`` or ``"under"``.
    line : float
        The total-goal line (default 2.5).

    Returns
    -------
    tuple[bool, bool]
        ``(won, pushed)``. A push occurs when the total exactly equals
        the line for integer-goal lines (e.g., line=2.0, total=2).
    """
    total = home_goals + away_goals

    # Check for push (exact on the line for integer/0.5 lines)
    # For lines like 2.5, exact hits are impossible since total is integer
    # For lines like 2.0, a total of 2 is a push
    if line % 1 == 0 and total == line:
        return False, True  # pushed (refund)
    if outcome == "over":
        return total > line, False
    if outcome == "under":
        return total < line, False
    return False, False


# ── Registry of settlement functions for each market ────

MARKET_SETTLEMENT: dict[str, Callable[..., tuple[bool, bool]]] = {
    "home_win": lambda hg, ag: (_settle_1x2(hg, ag, "home_win"), False),
    "draw": lambda hg, ag: (_settle_1x2(hg, ag, "draw"), False),
    "away_win": lambda hg, ag: (_settle_1x2(hg, ag, "away_win"), False),
    "btts_yes": lambda hg, ag: (_settle_btts(hg, ag, "yes"), False),
    "btts_no": lambda hg, ag: (_settle_btts(hg, ag, "no"), False),
    "over25": lambda hg, ag: _settle_over_under(hg, ag, "over", 2.5),
    "under25": lambda hg, ag: _settle_over_under(hg, ag, "under", 2.5),
}


# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════


@dataclass
class BacktestBetRecord:
    """A single placed bet during the backtest simulation."""

    match_label: str
    outcome: str                         # "home_win", "draw", "away_win", "btts_yes", "over25", etc.
    market: str                          # "1X2", "BTTS", "Over/Under"
    decimal_odds: float
    model_prob: float
    ev: float
    edge_vs_market: float
    stake_amount: float
    stake_pct: float
    profit: float
    won: bool
    pushed: bool = False                 # True if bet was void/refunded
    void_reason: str | None = None       # e.g. "match_abandoned", "exact_line_push"
    clv: float | None = None
    bankroll_before: float = 0.0
    bankroll_after: float = 0.0
    timestamp: int = 0


@dataclass
class ExtendedBacktestMetrics:
    """Aggregate performance metrics from a multi-market backtest run.

    Extends the base ``BacktestMetrics`` with additional metrics for
    pushed bets, Sharpe/Sortino ratios, and market-level breakdown.
    """

    # Core stats
    initial_bankroll: float = 1000.0
    final_bankroll: float = 1000.0
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    pushed_bets: int = 0
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

    # Market breakdown
    bets_per_market: dict[str, int] = field(default_factory=dict)
    profit_per_market: dict[str, float] = field(default_factory=dict)

    # Timestamps
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    # Configuration snapshot
    stake_strategy: str = ""
    bet_filter_config: dict[str, Any] = field(default_factory=dict)
    allowed_markets: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
#  Main Backtester class
# ═══════════════════════════════════════════════════════════


class Backtester:
    """Model-driven backtesting with multi-market support.

    Parameters
    ----------
    model : Any
        Trained model with ``predict_proba(X)`` returning
        ``[away_prob, draw_prob, home_prob]``. Used for 1X2 markets.
        Can also be ``None`` if only pure-data markets (BTTS, Over/Under)
        are being backtested (probabilities from DataFrame columns).
    initial_bankroll : float
        Starting bankroll in currency units (default 1000).
    stake_strategy : StakingStrategy, optional
        A ``StakingStrategy`` instance for computing stake amounts.
        Defaults to ``FractionalKellyStaking(fraction=0.25)``.
    bet_filter : BetFilter, optional
        A ``BetFilter`` instance for gating bet opportunities.
        Defaults to ``BetFilter(min_ev=0.0, min_confidence=0.3, min_odds=1.5)``.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        initial_bankroll: float = 1000.0,
        stake_strategy: StakingStrategy | None = None,
        bet_filter: BetFilter | None = None,
    ) -> None:
        if initial_bankroll <= 0:
            raise ValueError(f"initial_bankroll must be > 0, got {initial_bankroll}")

        self.model = model
        self.initial_bankroll = initial_bankroll
        self.stake_strategy = stake_strategy or StakingFactory.create(
            "fractional_kelly", fraction=0.25,
        )
        self.bet_filter = bet_filter or BetFilter(
            min_ev=0.0, min_confidence=0.3, min_odds=1.5,
        )

        # Internal state
        self._bankroll: float = initial_bankroll
        self._peak: float = initial_bankroll
        self._bets: list[BacktestBetRecord] = []
        self._metrics: ExtendedBacktestMetrics | None = None
        self._start_time: datetime = datetime.now(timezone.utc)

    # ── Run simulation ──────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str] | None = None,
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        odds_mapping: dict[str, str] | None = None,
        prob_mapping: dict[str, str] | None = None,
        void_col: str | None = None,
        max_bets_per_match: int = 1,
        markets_1x2: list[str] | None = None,
    ) -> ExtendedBacktestMetrics:
        """Run the backtest simulation on historical match data.

        Parameters
        ----------
        df : pd.DataFrame
            Historical match data with columns for features, goals, and odds.
        feature_cols : list[str], optional
            Column names to use as model features for 1X2 prediction.
            Required if a model is provided and 1X2 markets are used.
        home_goals_col, away_goals_col : str
            Column names for actual goals scored (default ``home_goals``,
            ``away_goals``).
        home_team_col, away_team_col : str
            Column names for team names (default ``home_team``,
            ``away_team``).
        odds_mapping : dict[str, str], optional
            Maps market outcome names to DataFrame column names for odds.
            Example::

                {
                    "home_win": "BbAvH",
                    "draw": "BbAvD",
                    "away_win": "BbAvA",
                    "btts_yes": "BbAvBTTS_Yes",
                    "over25": "BbAvO25_Yes",
                }

        prob_mapping : dict[str, str], optional
            Maps market outcome names to DataFrame column names for model
            probabilities (for non-1X2 markets like BTTS, Over/Under).
            Example::

                {
                    "btts_yes": "btts_model_prob",
                    "over25": "over25_model_prob",
                }

        void_col : str, optional
            Column name indicating a match should be void (e.g. abandoned).
            If the value is truthy, all bets for that match are pushed/refunded.
        max_bets_per_match : int
            Maximum bets allowed per match (default 1).
        markets_1x2 : list[str], optional
            Which 1X2 outcomes to consider. Defaults to
            ``["home_win", "draw", "away_win"]``.

        Returns
        -------
        ExtendedBacktestMetrics
            Aggregate performance metrics.
        """
        self._start_time = datetime.now(timezone.utc)
        self._bankroll = self.initial_bankroll
        self._peak = self.initial_bankroll
        self._bets = []

        if df.empty:
            logger.warning("Empty DataFrame provided — no bets placed")
            self._metrics = self._compute_metrics()
            return self._metrics

        # ── Default mappings ────────────────────────────
        odds_mapping = odds_mapping or {}
        prob_mapping = prob_mapping or {}
        markets_1x2 = markets_1x2 or ["home_win", "draw", "away_win"]

        # Identify which markets to iterate
        all_market_outcomes = list(odds_mapping.keys())

        # Pre-compute 1X2 model predictions if model available
        model_1x2_probs: np.ndarray | None = None
        if self.model is not None and feature_cols is not None:
            # Ensure all feature columns exist
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                logger.warning(
                    "Feature columns missing from DataFrame: %s — "
                    "falling back to prob_mapping for 1X2",
                    missing,
                )
            else:
                X = df[feature_cols].copy()
                # Fill NaN with 0
                for c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
                model_1x2_probs = self.model.predict_proba(X)

        logger.info(
            "Running backtest on %d matches with %d market outcomes",
            len(df), len(all_market_outcomes),
        )

        match_counts: dict[str, int] = {}

        for match_idx, row in df.iterrows():
            home_team = str(row.get(home_team_col, ""))
            away_team = str(row.get(away_team_col, ""))
            match_label = f"{home_team} vs {away_team}" if home_team and away_team else f"Match {match_idx}"
            home_goals = float(row.get(home_goals_col, 0) or 0)
            away_goals = float(row.get(away_goals_col, 0) or 0)

            # Check if the entire match is void
            is_void = False
            void_reason: str | None = None
            if void_col is not None:
                void_val = row.get(void_col, False)
                if void_val and void_val not in (0, "0", "false", "False", "FALSE", ""):
                    is_void = True
                    void_reason = "match_void"

            # Get 1X2 model probabilities for this match
            _1x2_probs: dict[str, float] = {}
            if model_1x2_probs is not None:
                match_probs = model_1x2_probs[match_idx]  # [away, draw, home]
                _1x2_probs = {
                    "home_win": float(match_probs[2]),
                    "draw": float(match_probs[1]),
                    "away_win": float(match_probs[0]),
                }

            # ── Evaluate each market outcome ────────────
            bets_this_match = 0

            for outcome_name in all_market_outcomes:
                if bets_this_match >= max_bets_per_match:
                    break

                # --- Get model probability ---
                # 1X2 outcomes use model.predict_proba; others use prob_mapping cols
                if outcome_name in markets_1x2 and outcome_name in _1x2_probs:
                    model_prob = _1x2_probs[outcome_name]
                elif outcome_name in prob_mapping:
                    prob_col = prob_mapping[outcome_name]
                    model_prob = float(row.get(prob_col, 0.0) or 0.0)
                else:
                    # No probability source for this outcome
                    continue

                # --- Get odds ---
                odds_col = odds_mapping.get(outcome_name)
                if odds_col is None or odds_col not in df.columns:
                    continue
                decimal_odds = float(row.get(odds_col, 0.0) or 0.0)

                if decimal_odds <= 1.0 or math.isnan(decimal_odds):
                    continue
                if model_prob <= 0.0 or model_prob >= 1.0:
                    continue

                # --- Determine market type ---
                if outcome_name in markets_1x2:
                    market_type = "1X2"
                elif outcome_name.startswith("btts"):
                    market_type = "BTTS"
                elif outcome_name.startswith("over") or outcome_name.startswith("under"):
                    market_type = "Over/Under"
                else:
                    market_type = "Other"

                # --- Compute EV ---
                ev_result = calculate_ev(model_prob, decimal_odds)
                ev = ev_result["ev"]
                edge = ev_result["edge_vs_market"]

                # --- Apply BetFilter ---
                candidate_bet = {
                    "match": match_label,
                    "outcome": outcome_name,
                    "market": market_type,
                    "model_prob": model_prob,
                    "decimal_odds": decimal_odds,
                    "ev": ev,
                    "bankroll_pct": None,  # computed below
                }

                passed_bets, _ = self.bet_filter.filter_bets([candidate_bet])
                if not passed_bets:
                    continue

                # --- Enforce max bets per match ---
                match_counts[match_label] = match_counts.get(match_label, 0) + 1
                if match_counts[match_label] > max_bets_per_match:
                    continue

                # --- Compute stake ---
                stake_amount = self.stake_strategy.calculate_stake(
                    model_prob, decimal_odds, self._bankroll,
                )
                if stake_amount <= 0:
                    continue
                stake_pct = stake_amount / max(self._bankroll, 0.01)

                # --- Settle bet ---
                if is_void:
                    pushed = True
                    won = False
                    profit = 0.0  # refund (stake is returned)
                else:
                    won, pushed = self._settle_outcome(
                        outcome_name, home_goals, away_goals,
                    )
                    if pushed:
                        profit = 0.0  # refund
                    elif won:
                        profit = stake_amount * (decimal_odds - 1.0)
                    else:
                        profit = -stake_amount

                bankroll_before = self._bankroll

                if pushed:
                    # Bankroll unchanged (bet refunded)
                    pass
                else:
                    self._bankroll += profit

                if self._bankroll > self._peak:
                    self._peak = self._bankroll

                # --- CLV (if closing odds available) ---
                clv: float | None = None
                closing_col = odds_mapping.get(f"{outcome_name}_closing")
                if closing_col and closing_col in df.columns:
                    closing_odds = float(row.get(closing_col, 0.0) or 0.0)
                    if closing_odds > 1.0:
                        open_implied = 1.0 / decimal_odds
                        close_implied = 1.0 / closing_odds
                        clv = close_implied - open_implied

                # --- Record bet ---
                record = BacktestBetRecord(
                    match_label=match_label,
                    outcome=outcome_name,
                    market=market_type,
                    decimal_odds=round(decimal_odds, 4),
                    model_prob=round(model_prob, 4),
                    ev=round(ev, 4),
                    edge_vs_market=round(edge, 4),
                    stake_amount=round(stake_amount, 2),
                    stake_pct=round(stake_pct, 6),
                    profit=round(profit, 2),
                    won=won,
                    pushed=pushed,
                    void_reason=void_reason if pushed else None,
                    clv=round(clv, 4) if clv is not None else None,
                    bankroll_before=round(bankroll_before, 2),
                    bankroll_after=round(self._bankroll, 2),
                    timestamp=int(datetime.now(timezone.utc).timestamp()),
                )
                self._bets.append(record)
                bets_this_match += 1

        self._metrics = self._compute_metrics(allowed_markets=list(set(
            b.market for b in self._bets
        )))

        logger.info(
            "Backtest complete — %d bets placed (W:%d L:%d P:%d), "
            "P&L=%.2f, ROI=%.2f%%",
            self._metrics.total_bets,
            self._metrics.winning_bets,
            self._metrics.losing_bets,
            self._metrics.pushed_bets,
            self._metrics.total_profit,
            self._metrics.roi_pct,
        )

        return self._metrics

    # ── Settlement ───────────────────────────────────────

    def _settle_outcome(
        self,
        outcome: str,
        home_goals: float,
        away_goals: float,
    ) -> tuple[bool, bool]:
        """Settle a bet for a given market outcome.

        Parameters
        ----------
        outcome : str
            Market outcome name (e.g. ``"home_win"``, ``"btts_yes"``).
        home_goals, away_goals : float
            Actual goals scored.

        Returns
        -------
        tuple[bool, bool]
            ``(won, pushed)``. ``pushed=True`` means the bet is refunded.
        """
        settle_fn = MARKET_SETTLEMENT.get(outcome)
        if settle_fn is None:
            logger.warning("Unknown outcome '%s' — settling as loss", outcome)
            return False, False
        return settle_fn(home_goals, away_goals)

    # ── Metrics computation ─────────────────────────────

    def _compute_metrics(
        self,
        allowed_markets: list[str] | None = None,
    ) -> ExtendedBacktestMetrics:
        """Compute all performance metrics from the bet history."""
        active_bets = [b for b in self._bets if not b.pushed]
        n_total = len(self._bets)
        n_active = len(active_bets)
        n_pushed = n_total - n_active

        m = ExtendedBacktestMetrics(
            initial_bankroll=self.initial_bankroll,
            final_bankroll=self._bankroll,
            total_bets=n_active,
            winning_bets=sum(1 for b in active_bets if b.won),
            losing_bets=sum(1 for b in active_bets if b.won is False),
            pushed_bets=n_pushed,
            stake_strategy=self.stake_strategy.__class__.__name__,
            bet_filter_config={
                "min_ev": self.bet_filter.min_ev,
                "min_confidence": self.bet_filter.min_confidence,
                "min_odds": self.bet_filter.min_odds,
                "max_stake": self.bet_filter.max_stake,
                "markets": list(self.bet_filter.markets),
            },
            allowed_markets=allowed_markets or [],
            started_at=self._start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            finished_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_seconds=(datetime.now(timezone.utc) - self._start_time).total_seconds(),
        )

        if m.total_bets == 0:
            m.bankroll_history = [self.initial_bankroll, self._bankroll]
            m.drawdown_history = [0.0, 0.0]
            return m

        # Basic financials
        m.total_staked = sum(b.stake_amount for b in active_bets)
        m.total_profit = sum(b.profit for b in active_bets)

        # ROI
        m.roi_pct = (
            (m.final_bankroll - m.initial_bankroll)
            / m.initial_bankroll * 100
        )

        # Yield
        m.yield_pct = (
            (m.total_profit / m.total_staked * 100)
            if m.total_staked > 0 else 0.0
        )

        # Win rate (active bets only)
        m.win_rate_pct = (m.winning_bets / m.total_bets) * 100

        # Averages
        m.avg_odds = float(np.mean([b.decimal_odds for b in active_bets]))
        m.avg_ev = float(np.mean([b.ev for b in active_bets]))
        m.avg_stake_pct = float(np.mean([b.stake_pct for b in active_bets]))

        # Profit factor
        gross_profit = sum(b.profit for b in active_bets if b.profit > 0)
        gross_loss = abs(sum(b.profit for b in active_bets if b.profit < 0))
        m.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0
            else float("inf") if gross_profit > 0
            else 0.0
        )

        # CLV
        clv_values = [b.clv for b in active_bets if b.clv is not None]
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

        # Equity curve (cumulative P&L from active bets)
        m.equity_curve = self._build_equity_curve(active_bets)

        # Sharpe ratio (using bet-level returns from active bets)
        returns = np.array([
            b.profit / max(b.bankroll_before, 0.01) for b in active_bets
        ])
        if len(returns) > 1 and np.std(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            # Annualised Sharpe assuming ~500 bets / year
            m.sharpe_ratio = float(
                (mean_return / std_return) * math.sqrt(500)
                if std_return > 0 else 0.0
            )

            # Sortino ratio (downside deviation only)
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0:
                downside_std = (
                    np.std(neg_returns, ddof=1) if len(neg_returns) > 1
                    else np.std(neg_returns)
                )
                m.sortino_ratio = float(
                    (mean_return / downside_std) * math.sqrt(500)
                    if downside_std > 0 else 0.0
                )

        # Streaks
        m.longest_win_streak, m.longest_lose_streak = self._compute_streaks(active_bets)
        m.current_win_streak = self._current_streak(active_bets, won=True)
        m.current_lose_streak = self._current_streak(active_bets, won=False)

        # Per-market breakdown
        m.bets_per_market = {}
        m.profit_per_market = {}
        for b in active_bets:
            m.bets_per_market[b.market] = m.bets_per_market.get(b.market, 0) + 1
            m.profit_per_market[b.market] = m.profit_per_market.get(b.market, 0.0) + b.profit

        return m

    # ── Reporting ───────────────────────────────────────

    def print_report(self) -> None:
        """Print a formatted backtest report to the console."""
        m = self._metrics or ExtendedBacktestMetrics()

        print("\n" + "=" * 80)
        print("  BACKTEST REPORT (Multi-Market)".center(78))
        print("=" * 80)

        if m.total_bets == 0:
            print("\n  No bets were placed during the backtest period.")
            print(f"  Final bankroll: GBP{m.final_bankroll:.2f}")
            print("=" * 80)
            return

        print(f"  {'Metric':<30} {'Value':>15} {'Notes':<33}")
        print(f"  {'-' * 78}")

        # ── P&L section ──
        print(f"  {'TOTAL BETS':<30} {m.total_bets:>10d}   {'active bets placed':>20}")
        print(f"  {'Winning':<30} {m.winning_bets:>8d} / {m.total_bets:<8d}  {'':>20}")
        print(f"  {'Win Rate':<30} {m.win_rate_pct:>13.1f}%   {'':>20}")
        print(f"  {'Pushed / Void':<30} {m.pushed_bets:>10d}   {'refunded bets':>20}")

        profit_marker = "+" if m.total_profit >= 0 else ""
        print(f"  {'Total P&L':<30} {profit_marker}GBP{m.total_profit:>+9.2f}   {'':>20}")
        print(f"  {'Total Staked':<30} GBP{m.total_staked:>10.2f}   {'':>20}")

        roi_marker = "+" if m.roi_pct >= 0 else ""
        print(f"  {'ROI':<30} {roi_marker}{m.roi_pct:>+11.2f}%   {'':>20}")
        yield_marker = "+" if m.yield_pct >= 0 else ""
        print(f"  {'Yield':<30} {yield_marker}{m.yield_pct:>+11.2f}%   {'profit per unit staked':>20}")

        bankroll_change = m.final_bankroll - m.initial_bankroll
        change_sign = "+" if bankroll_change >= 0 else ""
        print(f"  {'Final Bankroll':<30} GBP{m.final_bankroll:>10.2f}   "
              f"({change_sign}GBP{bankroll_change:.2f} from GBP{m.initial_bankroll:.0f})")

        # ── Risk section ──
        print(f"  {'Max Drawdown':<30} {m.max_drawdown_pct:>12.2f}%   "
              f"(GBP{m.max_drawdown_amount:.2f} peak-to-trough)")
        print(f"  {'Sharpe Ratio':<30} {m.sharpe_ratio:>14.2f}   {'risk-adjusted return':>20}")
        print(f"  {'Sortino Ratio':<30} {m.sortino_ratio:>14.2f}   {'downside risk-adjusted':>20}")

        # ── Quality section ──
        print(f"  {'Profit Factor':<30} {m.profit_factor:>14.2f}   {'gross profit / gross loss':>20}")
        print(f"  {'Avg Odds':<30} {m.avg_odds:>14.4f}   {'':>20}")
        print(f"  {'Avg EV':<30} {m.avg_ev:>+14.2%}   {'':>20}")

        if m.avg_clv != 0.0:
            print(f"  {'Avg CLV':<30} {m.avg_clv:>+14.4f}   "
                  f"{m.positive_clv_pct:.0f}% positive bets")

        # ── Streaks ──
        print(f"  {'Longest Win Streak':<30} {m.longest_win_streak:>8d} bets   {'':>20}")
        print(f"  {'Longest Lose Streak':<30} {m.longest_lose_streak:>8d} bets   {'':>20}")

        # ── Per-market breakdown ──
        if m.bets_per_market:
            print(f"\n  {'MARKET BREAKDOWN':-^78}")
            for market in sorted(m.bets_per_market.keys()):
                n = m.bets_per_market.get(market, 0)
                pnl = m.profit_per_market.get(market, 0.0)
                pnl_marker = "+" if pnl >= 0 else ""
                print(f"  {market:<20s}  {n:>4d} bets  "
                      f"{pnl_marker}GBP{pnl:>+9.2f} P&L")

        # ── Performance assessment ──
        print(f"\n  {'ASSESSMENT':-^78}")
        print(f"  {self._assess_performance(m)}")
        print("=" * 80)

    def _assess_performance(self, m: ExtendedBacktestMetrics) -> str:
        """Generate a human-readable performance assessment."""
        lines = []
        if m.sharpe_ratio >= 2.0:
            lines.append("  (+) Excellent risk-adjusted returns (Sharpe >= 2.0)")
        elif m.sharpe_ratio >= 1.0:
            lines.append("  (+) Good risk-adjusted returns (Sharpe >= 1.0)")
        elif m.sharpe_ratio >= 0.5:
            lines.append("  (!) Moderate risk-adjusted returns (Sharpe 0.5-1.0)")
        else:
            lines.append("  (!) Poor risk-adjusted returns (Sharpe < 0.5)")

        if m.max_drawdown_pct < 10:
            lines.append(f"  (+) Low drawdown ({m.max_drawdown_pct:.1f}%) -- good risk mgmt")
        elif m.max_drawdown_pct < 25:
            lines.append(f"  (!) Moderate drawdown ({m.max_drawdown_pct:.1f}%)")
        else:
            lines.append(f"  (!) High drawdown ({m.max_drawdown_pct:.1f}%) -- risk of ruin")

        if m.profit_factor >= 2.0:
            lines.append("  (+) Profit factor >= 2.0 -- strong risk/reward")
        elif m.profit_factor >= 1.5:
            lines.append("  (+) Profit factor >= 1.5 -- solid")
        elif m.profit_factor >= 1.0:
            lines.append("  (!) Profit factor >= 1.0 -- marginal")
        else:
            lines.append("  (!) Profit factor < 1.0 -- losses exceed gains")

        if m.roi_pct > 0:
            lines.append(f"  (+) Profitable: {m.roi_pct:+.1f}% ROI")
        elif m.roi_pct == 0:
            lines.append("  (=) Break-even")
        else:
            lines.append(f"  (!) Loss-making: {m.roi_pct:+.1f}% ROI")

        if m.total_bets < 100:
            lines.append(f"  (!) Small sample ({m.total_bets} bets) -- results may not be significant")
        elif m.total_bets < 500:
            lines.append(f"  (~) Moderate sample ({m.total_bets} bets)")
        else:
            lines.append(f"  (~) Large sample ({m.total_bets} bets) -- more reliable")

        # Multi-market note
        if len(m.bets_per_market) > 1:
            markets_str = ", ".join(sorted(m.bets_per_market.keys()))
            lines.append(f"  (~) Multi-market: {markets_str}")

        return "\n".join(lines)

    # ── Results export ──────────────────────────────────

    def save_results(
        self,
        output_dir: str | Path = "reports/backtest",
        model_name: str = "backtester",
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

        logger.info("Backtest results saved to %s", filepath)
        return str(filepath)

    # ── Compatibility wrapper ──────────────────────────

    def to_backtest_metrics(self) -> BacktestMetrics:
        """Convert the extended metrics to the base ``BacktestMetrics`` format.

        Useful for compatibility with existing reporting/plotting code
        that expects the ``BacktestMetrics`` dataclass from ``__init__.py``.
        """
        m = self._metrics or ExtendedBacktestMetrics()
        return BacktestMetrics(
            initial_bankroll=m.initial_bankroll,
            final_bankroll=m.final_bankroll,
            total_bets=m.total_bets,
            winning_bets=m.winning_bets,
            losing_bets=m.losing_bets,
            total_staked=m.total_staked,
            total_profit=m.total_profit,
            roi_pct=m.roi_pct,
            yield_pct=m.yield_pct,
            win_rate_pct=m.win_rate_pct,
            max_drawdown_pct=m.max_drawdown_pct,
            max_drawdown_amount=m.max_drawdown_amount,
            avg_odds=m.avg_odds,
            avg_ev=m.avg_ev,
            profit_factor=m.profit_factor,
            longest_win_streak=m.longest_win_streak,
            longest_lose_streak=m.longest_lose_streak,
            bankroll_history=m.bankroll_history,
            drawdown_history=m.drawdown_history,
        )

    # ── Properties ──────────────────────────────────────

    @property
    def metrics(self) -> ExtendedBacktestMetrics:
        """Return the computed metrics, or empty metrics if not run yet."""
        if self._metrics is None:
            return ExtendedBacktestMetrics()
        return self._metrics

    @property
    def bets(self) -> list[BacktestBetRecord]:
        """Return the list of all placed bets (including pushed bets)."""
        return list(self._bets)

    # ── Internal helpers ────────────────────────────────

    def _build_bankroll_history(self) -> list[float]:
        """Build a chronological bankroll history."""
        history = [self.initial_bankroll]
        for bet in self._bets:
            if not bet.pushed:
                history.append(bet.bankroll_after)
        if len(history) == 1:
            history.append(self._bankroll)
        return history

    def _build_drawdown_history(
        self, history: list[float],
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

    def _build_equity_curve(
        self, active_bets: list[BacktestBetRecord],
    ) -> list[float]:
        """Build cumulative P&L equity curve from active (non-pushed) bets."""
        curve = [0.0]
        cumulative = 0.0
        for bet in active_bets:
            cumulative += bet.profit
            curve.append(round(cumulative, 2))
        return curve

    def _compute_streaks(
        self, active_bets: list[BacktestBetRecord],
    ) -> tuple[int, int]:
        """Return (longest_win_streak, longest_lose_streak)."""
        if not active_bets:
            return 0, 0
        win_streak = 0
        lose_streak = 0
        max_win = 0
        max_lose = 0
        for b in active_bets:
            if b.won:
                win_streak += 1
                lose_streak = 0
                max_win = max(max_win, win_streak)
            else:
                lose_streak += 1
                win_streak = 0
                max_lose = max(max_lose, lose_streak)
        return max_win, max_lose

    def _current_streak(
        self, active_bets: list[BacktestBetRecord], won: bool,
    ) -> int:
        """Return the current ongoing streak."""
        count = 0
        for b in reversed(active_bets):
            if b.won == won:
                count += 1
            else:
                break
        return count
