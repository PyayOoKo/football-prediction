"""
Betting Engine — orchestrates the full betting pipeline.

Composes all modules together:
  1. Probability Source → get model predictions
  2. Odds Source → get bookmaker odds
  3. Market Filter → reject unsuitable markets
  4. Calculator (EV, Kelly, CLV) → compute metrics
  5. Bet Filter → reject unsuitable bets
  6. Portfolio Optimizer → allocate across bets
  7. Risk Manager → final risk check
  8. Bankroll Manager → place bets, track P&L
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.betting.calculator import CalculatorFactory
from src.betting.models import (
    Bankroll,
    BetFilterConfig,
    BetOutcome,
    BetSlip,
    BetStatus,
    BettingSessionReport,
    MarketFilterConfig,
    MatchOdds,
    ModelPrediction,
    Outcome,
    PortfolioAllocation,
    PortfolioResult,
)
from src.betting.registry import BettingRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Default implementations (bundled with the framework)
# ═══════════════════════════════════════════════════════════


class DefaultBankrollManager:
    """Simple in-memory bankroll manager.

    Tracks balance, history, P&L.  Does NOT persist to DB by default.
    """

    def __init__(self, initial_balance: float = 1000.0, currency: str = "GBP") -> None:
        self._bankroll = Bankroll(
            initial_balance=initial_balance,
            currency=currency,
        )

    @property
    def bankroll(self) -> Bankroll:
        return self._bankroll

    def reset(self, initial_balance: float | None = None) -> None:
        if initial_balance is not None:
            self._bankroll = Bankroll(initial_balance=initial_balance, currency=self._bankroll.currency)
        else:
            self._bankroll.reset()

    def can_cover_stake(self, amount: float) -> bool:
        return (self._bankroll.current_balance or 0) >= amount

    def place_bet(self, amount: float) -> bool:
        if not self.can_cover_stake(amount):
            logger.warning("Insufficient bankroll for stake %.2f", amount)
            return False
        self._bankroll.record_stake(amount)
        return True

    def settle_bet(self, profit: float, won: bool) -> None:
        self._bankroll.record_result(profit, won)


class DefaultRiskManager:
    """Simple risk manager with configurable limits.

    Parameters
    ----------
    max_drawdown_pct : float, optional
        Maximum allowed drawdown % before stopping (default 50.0).
    max_daily_loss : float, optional
        Maximum daily loss in currency (default None = no limit).
    max_stake_pct : float
        Maximum single stake as % of bankroll (default 0.25 = 25%).
    max_consecutive_losses : int, optional
        Stop after N consecutive losses (default None = no limit).
    """

    def __init__(
        self,
        max_drawdown_pct: float | None = 50.0,
        max_daily_loss: float | None = None,
        max_stake_pct: float = 0.25,
        max_consecutive_losses: int | None = None,
    ) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss = max_daily_loss
        self.max_stake_pct = max_stake_pct
        self.max_consecutive_losses = max_consecutive_losses
        self._daily_loss: float = 0.0
        self._daily_date: str | None = None
        self._consecutive_losses: int = 0

    def check_bet(
        self, slip: BetSlip, bankroll: Bankroll, **kwargs: Any,
    ) -> tuple[bool, str]:
        # Drawdown check
        if self.max_drawdown_pct is not None:
            dd = bankroll.max_drawdown_pct
            if dd >= self.max_drawdown_pct:
                return False, f"Max drawdown ({dd:.1f}%) exceeds limit ({self.max_drawdown_pct:.1f}%)"

        # Stake % check
        if self.max_stake_pct is not None and slip.stake_pct is not None:
            if slip.stake_pct > self.max_stake_pct:
                return False, f"Stake {slip.stake_pct:.1%} exceeds max {self.max_stake_pct:.1%}"

        # Consecutive losses check
        if self.max_consecutive_losses is not None:
            if self._consecutive_losses >= self.max_consecutive_losses:
                return False, f"Too many consecutive losses ({self._consecutive_losses})"

        # Daily loss check
        if self.max_daily_loss is not None:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._daily_date != today:
                self._daily_loss = 0.0
                self._daily_date = today
            if self._daily_loss >= self.max_daily_loss:
                return False, f"Daily loss limit reached ({self._daily_loss:.2f})"

        return True, ""

    def check_batch(
        self, slips: list[BetSlip], bankroll: Bankroll, **kwargs: Any,
    ) -> dict[str, tuple[bool, str]]:
        return {s.bet_id: self.check_bet(s, bankroll) for s in slips}

    def record_loss(self, amount: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_date != today:
            self._daily_loss = 0.0
            self._daily_date = today
        self._daily_loss += abs(amount)
        self._consecutive_losses += 1

    def record_win(self) -> None:
        self._consecutive_losses = 0


class DefaultBetFilter:
    """Standard bet filter — applies BetFilterConfig rules."""

    def filter(
        self, slip: BetSlip, config: BetFilterConfig, **kwargs: Any,
    ) -> tuple[bool, str]:
        # Min EV
        if config.min_ev > 0 and (slip.ev is None or slip.ev < config.min_ev):
            return False, f"EV {slip.ev:.4f} below min {config.min_ev}"

        # Min edge
        if config.min_edge > 0 and (slip.edge is None or slip.edge < config.min_edge):
            return False, f"Edge {slip.edge:.4f} below min {config.min_edge}"

        # Min/max odds
        if slip.decimal_odds < config.min_odds:
            return False, f"Odds {slip.decimal_odds} below min {config.min_odds}"
        if slip.decimal_odds > config.max_odds:
            return False, f"Odds {slip.decimal_odds} above max {config.max_odds}"

        # Max stake %
        if slip.stake_pct is not None and slip.stake_pct > config.max_stake_pct_of_bankroll:
            return False, f"Stake {slip.stake_pct:.1%} exceeds max {config.max_stake_pct_of_bankroll:.1%}"

        # Allowed outcomes
        if config.allowed_outcomes is not None and slip.outcome not in config.allowed_outcomes:
            return False, f"Outcome {slip.outcome.value} not in allowed set"

        return True, ""


class DefaultMarketFilter:
    """Standard market filter — applies MarketFilterConfig rules."""

    def filter(
        self,
        match_id: str,
        prediction: ModelPrediction,
        odds: MatchOdds,
        config: MarketFilterConfig,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        # Bookmaker margin
        if config.max_bookmaker_margin < 1.0:
            margin = odds.margin()
            if margin > Decimal(str(config.max_bookmaker_margin)):
                return False, f"Bookmaker margin {float(margin):.2%} exceeds max {config.max_bookmaker_margin:.2%}"

        # Min market confidence
        if config.min_market_confidence is not None:
            fair = odds.fair_probs()
            max_fair = max(float(fair[o]) for o in Outcome)
            if max_fair < config.min_market_confidence:
                return False, f"Market confidence {max_fair:.1%} below min {config.min_market_confidence:.1%}"

        return True, ""


class DefaultPortfolioOptimizer:
    """Simple portfolio optimizer — naive equal-weight allocation.

    Distributes bankroll equally across all positive-EV bets,
    capped so total allocation ≤ 100% of bankroll.
    """

    def optimize(
        self,
        slips: list[BetSlip],
        bankroll: Bankroll,
        **kwargs: Any,
    ) -> PortfolioResult:
        if not slips:
            return PortfolioResult(
                allocations=[],
                total_bankroll_fraction=0.0,
                expected_return=0.0,
                portfolio_variance=0.0,
                sharpe_ratio=0.0,
                method="naive",
            )

        n_bets = len(slips)
        weight = 1.0 / n_bets if n_bets > 0 else 0.0
        total_expected_return = 0.0
        for slip in slips:
            expected_return = (slip.ev or 0) * weight
            total_expected_return += expected_return
            allocations.append(PortfolioAllocation(
                bet_slip=slip,
                weight=weight,
                expected_return=expected_return,
            ))

        return PortfolioResult(
            allocations=allocations,
            total_bankroll_fraction=total_weight,
            expected_return=total_expected_return,
            portfolio_variance=0.0,
            sharpe_ratio=0.0,
            method="naive_equal_weight",
            diversified=n_bets > 1,
        )


# ═══════════════════════════════════════════════════════════
#  Betting Engine — orchestrates the full pipeline
# ═══════════════════════════════════════════════════════════


class BettingEngine:
    """Primary entry point for the betting pipeline.

    Orchestrates all modules in the correct order:
      probabilities → odds → filters → calculators → staking → risk → bankroll

    Parameters
    ----------
    registry : BettingRegistry, optional
        Registry with registered modules. Creates a default one if omitted.
    bankroll_manager : Any, optional
        BankrollManager implementation. Defaults to ``DefaultBankrollManager``.
    risk_manager : Any, optional
        RiskManager implementation. Defaults to ``DefaultRiskManager``.
    bet_filter : Any, optional
        BetFilter implementation. Defaults to ``DefaultBetFilter``.
    market_filter : Any, optional
        MarketFilter implementation. Defaults to ``DefaultMarketFilter``.
    portfolio_optimizer : Any, optional
        PortfolioOptimizer implementation. Defaults to ``DefaultPortfolioOptimizer``.
    bet_filter_config : BetFilterConfig, optional
        Default bet filter configuration.
    market_filter_config : MarketFilterConfig, optional
        Default market filter configuration.
    """

    def __init__(
        self,
        registry: BettingRegistry | None = None,
        bankroll_manager: Any | None = None,
        risk_manager: Any | None = None,
        bet_filter: Any | None = None,
        market_filter: Any | None = None,
        portfolio_optimizer: Any | None = None,
        bet_filter_config: BetFilterConfig | None = None,
        market_filter_config: MarketFilterConfig | None = None,
    ) -> None:
        self.registry = registry or BettingRegistry(auto_discover=True)
        self.bankroll = bankroll_manager or DefaultBankrollManager()
        self.risk = risk_manager or DefaultRiskManager()
        self.bet_filter = bet_filter or DefaultBetFilter()
        self.market_filter = market_filter or DefaultMarketFilter()
        self.portfolio_optimizer = portfolio_optimizer or DefaultPortfolioOptimizer()

        self.bet_filter_config = bet_filter_config or BetFilterConfig()
        self.market_filter_config = market_filter_config or MarketFilterConfig()

        # Internal state
        self._slips: list[BetSlip] = []
        self._outcomes: list[BetOutcome] = []
        self._report: BettingSessionReport | None = None

        # Calculator instances (use built-in by default)
        self.ev_calculator = CalculatorFactory.create("ev")
        self.kelly_calculator = CalculatorFactory.create("kelly")
        self.fractional_kelly = CalculatorFactory.create("fractional_kelly")
        self.flat_stake_calculator = CalculatorFactory.create("flat_stake")
        self.clv_calculator = CalculatorFactory.create("clv")

    # ── Core pipeline ─────────────────────────────────

    def run_pipeline(
        self,
        matches: list[dict[str, Any]],
        staking_method: str = "fractional_kelly",
        staking_params: dict[str, Any] | None = None,
        return_report: bool = True,
    ) -> BettingSessionReport:
        """Run the full betting pipeline on a set of matches.

        Parameters
        ----------
        matches : list[dict]
            List of match dicts. Each must have:
            ``match_id``, ``home_team``, ``away_team``.
            Optional: ``opening_odds`` (MatchOdds dict) for CLV.
        staking_method : str
            One of: ``kelly``, ``fractional_kelly``, ``flat_stake``.
        staking_params : dict, optional
            Parameters for the staking calculator (e.g.
            ``{"fraction": 0.25}`` for fractional Kelly).
        return_report : bool
            If True, return a ``BettingSessionReport``.

        Returns
        -------
        BettingSessionReport
            Summary of the betting session.
        """
        start_time = datetime.now(timezone.utc)
        params = staking_params or {}

        # Resolve staking calculator
        if staking_method == "kelly":
            stake_fn = self._stake_kelly
        elif staking_method == "fractional_kelly":
            stake_fn = self._stake_fractional_kelly
        elif staking_method == "flat_stake":
            stake_fn = self._stake_flat
        else:
            raise ValueError(f"Unknown staking method: {staking_method}")

        self._slips = []
        self._outcomes = []

        for match in matches:
            match_id = match.get("match_id", "")
            home_team = match.get("home_team", "")
            away_team = match.get("away_team", "")

            # Step 1: Get model probabilities
            prob_source = self._resolve_probability_source(match)
            if prob_source is None:
                logger.debug("No probability source for %s, skipping", match_id)
                continue
            prediction = prob_source.get_probabilities(match_id)
            if prediction is None:
                continue

            # Step 2: Get bookmaker odds
            odds_source = self._resolve_odds_source(match)
            if odds_source is None:
                continue
            match_odds = odds_source.get_odds(match_id)
            if match_odds is None:
                continue

            # Step 3: Market filter
            passed, reason = self.market_filter.filter(
                match_id, prediction, match_odds, self.market_filter_config,
            )
            if not passed:
                logger.debug("Market filter rejected %s: %s", match_id, reason)
                continue

            # Step 4: Evaluate each outcome
            for outcome in Outcome:
                slip = self._evaluate_outcome(
                    match_id, home_team, away_team,
                    outcome, prediction, match_odds,
                    match.get("opening_odds"),
                )
                if slip is None:
                    continue

                # Step 5: Calculate stake
                stake_fn(slip, params)

                # Step 6: Bet filter
                passed, reason = self.bet_filter.filter(slip, self.bet_filter_config)
                if passed:
                    # Step 7: Risk check
                    risk_ok, risk_reason = self.risk.check_bet(
                        slip, self.bankroll.bankroll,
                    )
                    if risk_ok:
                        slip.recommended = True
                        slip.rank = len(self._slips) + 1
                        self._slips.append(slip)

        # Step 8: Portfolio optimisation
        if self._slips and len(self._slips) > 1:
            portfolio_result = self.portfolio_optimizer.optimize(
                self._slips, self.bankroll.bankroll,
            )
            # Apply portfolio weights
            for alloc in portfolio_result.allocations:
                alloc.bet_slip.stake_amount = alloc.stake_amount
                alloc.bet_slip.stake_pct = alloc.weight
        elif self._slips:
            # Single bet — use full stake
            pass  # stake already computed

        # Step 9: Place bets
        for slip in self._slips:
            if slip.recommended and slip.stake_amount is not None:
                placed = self.bankroll.place_bet(slip.stake_amount)
                if placed:
                    logger.info(
                        "Placed bet: %s %s @ %.2f (stake=%.2f)",
                        slip.match_label, slip.outcome.value,
                        float(slip.decimal_odds), slip.stake_amount,
                    )

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        if return_report:
            return self._build_report(start_time, elapsed)
        return BettingSessionReport(
            bankroll=self.bankroll.bankroll,
            start_time=start_time,
            duration_seconds=elapsed,
        )

    # ── Per-outcome evaluation ─────────────────────────

    def _evaluate_outcome(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        outcome: Outcome,
        prediction: ModelPrediction,
        match_odds: MatchOdds,
        opening_odds: MatchOdds | None,
    ) -> BetSlip | None:
        decimal_odds = float(match_odds.odds_for(outcome))
        model_prob = float(prediction.prob_for(outcome))
        fair_prob = float(match_odds.fair_prob(outcome))

        # EV
        ev = self.ev_calculator.calculate_ev(model_prob, decimal_odds)
        edge = self.ev_calculator.calculate_edge(model_prob, fair_prob)

        # CLV
        clv = None
        if opening_odds is not None:
            clv = self.clv_calculator.calculate_clv(outcome, opening_odds, match_odds)

        slip = BetSlip(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            outcome=outcome,
            decimal_odds=Decimal(str(decimal_odds)),
            model_prob=Decimal(str(model_prob)),
            fair_prob=Decimal(str(fair_prob)),
            odds_source=match_odds.source,
            ev=ev,
            edge=edge,
            clv=clv,
        )
        return slip

    # ── Staking methods ───────────────────────────────

    def _stake_kelly(self, slip: BetSlip, params: dict[str, Any]) -> None:
        kelly_pct = self.kelly_calculator.calculate(
            float(slip.model_prob), float(slip.decimal_odds),
        )
        slip.kelly_fraction = kelly_pct
        slip.stake_pct = kelly_pct
        slip.stake_amount = (self.bankroll.bankroll.current_balance or 0) * kelly_pct

    def _stake_fractional_kelly(self, slip: BetSlip, params: dict[str, Any]) -> None:
        fraction = params.get("fraction", 0.25)
        kelly_pct = self.fractional_kelly.calculate(
            float(slip.model_prob), float(slip.decimal_odds),
            fraction=fraction,
        )
        slip.kelly_fraction = kelly_pct
        slip.fractional_kelly = kelly_pct
        slip.stake_pct = kelly_pct
        slip.stake_amount = (self.bankroll.bankroll.current_balance or 0) * kelly_pct

    def _stake_flat(self, slip: BetSlip, params: dict[str, Any]) -> None:
        stake_per_bet = params.get("stake_per_bet")
        stake_pct = params.get("stake_pct", 0.02)
        amount = self.flat_stake_calculator.calculate(
            self.bankroll.bankroll.current_balance or 0,
            stake_per_bet=stake_per_bet,
            stake_pct=stake_pct,
        )
        slip.stake_amount = amount
        slip.stake_pct = amount / (self.bankroll.bankroll.current_balance or 1)

    # ── Source resolution ─────────────────────────────

    def _resolve_probability_source(self, match: dict) -> Any | None:
        """Resolve the probability source for a match.

        Checks the registry first, then falls back to inline
        probabilities if provided in the match dict.
        """
        source_name = match.get("probability_source")
        if source_name:
            source = self.registry.get_probability_source(source_name)
            if source:
                return source

        # Inline probabilities
        if "model_probs" in match:
            arr = match["model_probs"]
            return _InlineProbabilitySource(
                match.get("match_id", ""),
                ModelPrediction.from_array(arr),
            )
        return None

    def _resolve_odds_source(self, match: dict) -> Any | None:
        """Resolve the odds source for a match."""
        source_name = match.get("odds_source")
        if source_name:
            source = self.registry.get_odds_source(source_name)
            if source:
                return source

        # Inline odds
        if "odds" in match:
            odds_dict = match["odds"]
            mo = MatchOdds(
                home_odds=Decimal(str(odds_dict.get("home_odds", 2.0))),
                draw_odds=Decimal(str(odds_dict.get("draw_odds", 3.5))),
                away_odds=Decimal(str(odds_dict.get("away_odds", 4.0))),
                source=odds_dict.get("source", "inline"),
            )
            return _InlineOddsSource(mo)
        return None

    # ── Reporting ─────────────────────────────────────

    def _build_report(
        self, start_time: datetime, elapsed: float,
    ) -> BettingSessionReport:
        bk = self.bankroll.bankroll
        report = BettingSessionReport(
            bankroll=bk,
            total_bets=len(self._slips),
            positive_ev_bets=sum(1 for s in self._slips if s.positive_ev),
            bets_placed=sum(1 for s in self._slips if s.recommended),
            total_staked=bk.total_staked,
            total_profit=bk.total_profit,
            roi_pct=bk.roi_pct,
            yield_pct=bk.yield_pct,
            win_rate_pct=bk.win_rate_pct,
            avg_odds=float(
                sum(float(s.decimal_odds) for s in self._slips) / max(len(self._slips), 1)
            ),
            avg_ev=float(
                sum(s.ev or 0 for s in self._slips) / max(len(self._slips), 1)
            ),
            avg_edge=float(
                sum(s.edge or 0 for s in self._slips) / max(len(self._slips), 1)
            ),
            max_drawdown_pct=bk.max_drawdown_pct,
            start_time=start_time,
            end_time=datetime.now(timezone.utc),
            duration_seconds=elapsed,
        )
        self._report = report
        return report

    def print_summary(self) -> None:
        """Print a summary of the last pipeline run to the console."""
        r = self._report
        if r is None:
            print("No pipeline run yet. Call run_pipeline() first.")
            return

        print("\n" + "=" * 70)
        print("  BETTING ENGINE — SESSION SUMMARY")
        print("=" * 70)
        print(f"  Bets evaluated:   {r.total_bets}")
        print(f"  Positive EV:      {r.positive_ev_bets}")
        print(f"  Bets placed:      {r.bets_placed}")
        print(f"  Total staked:     £{r.total_staked:.2f}")
        print(f"  Total profit:     £{r.total_profit:+.2f}")
        print(f"  ROI:              {r.roi_pct:+.2f}%")
        print(f"  Yield:            {r.yield_pct:+.2f}%")
        print(f"  Win rate:         {r.win_rate_pct:.1f}%")
        print(f"  Avg odds:         {r.avg_odds:.3f}")
        print(f"  Avg EV:           {r.avg_ev:+.4f}")
        print(f"  Avg edge:         {r.avg_edge:+.4f}")
        print(f"  Max drawdown:     {r.max_drawdown_pct:.2f}%")
        print(f"  Duration:         {r.duration_seconds:.1f}s")
        print("=" * 70)

    @property
    def pending_slips(self) -> list[BetSlip]:
        """Return the current pending bet slips."""
        return self._slips


# ═══════════════════════════════════════════════════════════
#  Inline helpers (simple sources for one-off use)
# ═══════════════════════════════════════════════════════════


class _InlineProbabilitySource:
    """Wraps a single ModelPrediction as a ProbabilitySource."""

    def __init__(self, match_id: str, prediction: ModelPrediction) -> None:
        self._match_id = match_id
        self._prediction = prediction

    def get_probabilities(self, match_id: str, **kwargs: Any) -> ModelPrediction | None:
        if match_id == self._match_id:
            return self._prediction
        return None

    def get_batch_probabilities(
        self, match_ids: list[str], **kwargs: Any,
    ) -> dict[str, ModelPrediction]:
        result = {}
        if self._match_id in match_ids:
            result[self._match_id] = self._prediction
        return result


class _InlineOddsSource:
    """Wraps a single MatchOdds as an OddsSource."""

    def __init__(self, odds: MatchOdds) -> None:
        self._odds = odds
        self._source_name = odds.source

    @property
    def source_name(self) -> str:
        return self._source_name

    def get_odds(self, match_id: str, **kwargs: Any) -> MatchOdds | None:
        return self._odds

    def get_batch_odds(
        self, match_ids: list[str], **kwargs: Any,
    ) -> dict[str, MatchOdds]:
        result = {}
        for mid in match_ids:
            result[mid] = self._odds
        return result
