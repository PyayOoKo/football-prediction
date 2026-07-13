"""
Abstract Base Classes — interfaces for every module in the betting engine.

Each module type defines a clear Protocol or ABC so that concrete
implementations can be swapped, tested, and discovered automatically
by the plugin system.

Module taxonomy
---------------
**Input modules** (feed data into the pipeline):
    - ``ProbabilitySource``      — provides model probabilities
    - ``OddsSource``             — provides bookmaker odds

**Calculators** (transform data / compute metrics):
    - ``ExpectedValueCalculator``    — EV = (model_prob × odds) − 1
    - ``KellyCalculator``            — Full Kelly stake %
    - ``FractionalKellyCalculator``  — f × Full Kelly
    - ``FlatStakeCalculator``        — Fixed stake per bet
    - ``ClosingLineValueCalculator`` — CLV = fair_close − fair_open

**Management modules** (stateful, store data):
    - ``BankrollManager``        — tracks balance, P&L, history
    - ``RiskManager``            — enforces risk limits, drawdown stops

**Filter modules** (gate bets before placement):
    - ``BetFilter``              — filters individual bet proposals
    - ``MarketFilter``           — filters matches / markets

**Optimisation modules** (multi-bet allocation):
    - ``PortfolioOptimizer``     — allocates across concurrent bets
"""

from __future__ import annotations

import abc
from typing import Any, Protocol

from src.betting.models import (
    Bankroll,
    BetFilterConfig,
    BetOutcome,
    BetSlip,
    BetStatus,
    MarketFilterConfig,
    MatchOdds,
    ModelPrediction,
    Outcome,
    PortfolioAllocation,
    PortfolioResult,
)


# ═══════════════════════════════════════════════════════════
#  Input modules
# ═══════════════════════════════════════════════════════════


class ProbabilitySource(Protocol):
    """Provides model-predicted probabilities for matches.

    Implementations could read from:
    - An in-memory classifier (predict_proba)
    - A predictions database table
    - A CSV file
    - An external API
    """

    def get_probabilities(self, match_id: str, **kwargs: Any) -> ModelPrediction | None:
        """Return model probabilities for a single match, or None if unknown."""
        ...

    def get_batch_probabilities(
        self, match_ids: list[str], **kwargs: Any,
    ) -> dict[str, ModelPrediction]:
        """Return model probabilities for multiple matches at once."""
        ...


class OddsSource(Protocol):
    """Provides bookmaker odds for matches.

    Implementations could read from:
    - The Odds API (live)
    - A database (historical)
    - CSV files
    - Multiple bookmaker aggregators
    """

    def get_odds(self, match_id: str, **kwargs: Any) -> MatchOdds | None:
        """Return odds for a single match, or None if unavailable."""
        ...

    def get_batch_odds(
        self, match_ids: list[str], **kwargs: Any,
    ) -> dict[str, MatchOdds]:
        """Return odds for multiple matches at once."""
        ...

    @property
    def source_name(self) -> str:
        """Human-readable name of this odds source."""
        ...


# ═══════════════════════════════════════════════════════════
#  Calculator modules (stateless)
# ═══════════════════════════════════════════════════════════


class ExpectedValueCalculator(Protocol):
    """Computes expected value for a bet proposal.

    Formula: EV = (model_prob × decimal_odds) − 1
    """

    def calculate_ev(
        self, model_prob: float, decimal_odds: float, **kwargs: Any,
    ) -> float:
        """Return the expected value (e.g. 0.05 = 5% expected return)."""
        ...

    def calculate_edge(
        self, model_prob: float, fair_prob: float, **kwargs: Any,
    ) -> float:
        """Return the probability edge: model_prob − fair_prob."""
        ...


class KellyCalculator(Protocol):
    """Computes the full Kelly stake fraction.

    Formula: f* = (p × b − 1) / (b − 1)
        where p = model probability, b = decimal odds
    """

    def calculate(
        self, model_prob: float, decimal_odds: float, **kwargs: Any,
    ) -> float:
        """Return the full Kelly fraction of bankroll (0-1).

        Returns 0.0 if the bet has no positive edge.
        """
        ...


class FractionalKellyCalculator(Protocol):
    """Computes a fractional Kelly stake.

    Formula: f_fractional = f_full × fraction
    """

    def calculate(
        self,
        model_prob: float,
        decimal_odds: float,
        fraction: float = 0.25,
        **kwargs: Any,
    ) -> float:
        """Return the fractional Kelly stake as a fraction of bankroll.

        Parameters
        ----------
        model_prob : float
            Model probability (0-1).
        decimal_odds : float
            Decimal odds.
        fraction : float
            Fraction of full Kelly to use (default 0.25 = 25% Kelly).

        Returns
        -------
        float
            Fraction of bankroll to stake (0-1).
        """
        ...


class FlatStakeCalculator(Protocol):
    """Computes a fixed stake amount regardless of edge size."""

    def calculate(
        self,
        bankroll: float,
        stake_per_bet: float | None = None,
        stake_pct: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Return the flat stake amount.

        Parameters
        ----------
        bankroll : float
            Current bankroll balance.
        stake_per_bet : float, optional
            Fixed currency amount per bet. Overrides percentage.
        stake_pct : float, optional
            Fixed percentage of bankroll per bet (0-1).

        Returns
        -------
        float
            Stake amount in currency units.
        """
        ...


class ClosingLineValueCalculator(Protocol):
    """Computes Closing Line Value for a match.

    Formula: CLV = fair_prob_closing − fair_prob_opening
    """

    def calculate_clv(
        self,
        outcome: Outcome,
        opening_odds: MatchOdds,
        closing_odds: MatchOdds,
        **kwargs: Any,
    ) -> float:
        """Return CLV as a probability difference (e.g. 0.05 = +5pp)."""
        ...

    def calculate_all_clv(
        self, opening_odds: MatchOdds, closing_odds: MatchOdds, **kwargs: Any,
    ) -> dict[Outcome, float]:
        """Return CLV for all three outcomes."""
        ...


# ═══════════════════════════════════════════════════════════
#  Management modules (stateful)
# ═══════════════════════════════════════════════════════════


class BankrollManager(Protocol):
    """Manages betting capital over time.

    Stateful — tracks balance, history, P&L, and drawdown.
    The underlying state is a ``Bankroll`` dataclass.
    """

    @property
    def bankroll(self) -> Bankroll:
        """Current bankroll state (read-only view)."""
        ...

    def reset(self, initial_balance: float | None = None) -> None:
        """Reset bankroll to initial state."""
        ...

    def can_cover_stake(self, amount: float) -> bool:
        """Check if the bankroll can cover a given stake amount."""
        ...

    def place_bet(self, amount: float) -> bool:
        """Deduct stake from bankroll. Returns True if successful."""
        ...

    def settle_bet(self, profit: float, won: bool) -> None:
        """Record the result of a settled bet."""
        ...


class RiskManager(Protocol):
    """Enforces risk limits on betting activity.

    Checks include:
    - Maximum drawdown limit
    - Maximum daily loss
    - Maximum stake percentage
    - Consecutive loss limit
    - Maximum exposure
    """

    def check_bet(
        self, slip: BetSlip, bankroll: Bankroll, **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if a bet is allowed under current risk rules.

        Returns
        -------
        tuple[bool, str]
            (allowed, reason) — reason is empty if allowed.
        """
        ...

    def check_batch(
        self, slips: list[BetSlip], bankroll: Bankroll, **kwargs: Any,
    ) -> dict[str, tuple[bool, str]]:
        """Check multiple bets. Returns ``{bet_id: (allowed, reason)}``."""
        ...


# ═══════════════════════════════════════════════════════════
#  Filter modules
# ═══════════════════════════════════════════════════════════


class BetFilter(Protocol):
    """Filters individual bet proposals before placement.

    Filters are applied AFTER calculators have populated
    the BetSlip with EV, Kelly, etc.
    """

    def filter(
        self, slip: BetSlip, config: BetFilterConfig, **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if a bet proposal passes this filter.

        Returns
        -------
        tuple[bool, str]
            (passes, reason) — reason is empty if it passes.
        """
        ...


class MarketFilter(Protocol):
    """Filters matches / markets before analysis.

    Market filters are applied BEFORE calculators run,
    to avoid wasting computation on unsuitable markets.
    """

    def filter(
        self,
        match_id: str,
        prediction: ModelPrediction,
        odds: MatchOdds,
        config: MarketFilterConfig,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if a match/market passes this filter.

        Returns
        -------
        tuple[bool, str]
            (passes, reason) — reason is empty if it passes.
        """
        ...


# ═══════════════════════════════════════════════════════════
#  Optimisation modules
# ═══════════════════════════════════════════════════════════


class PortfolioOptimizer(Protocol):
    """Allocates bankroll across multiple concurrent bets.

    Different strategies:
    - Naive: equal weight across all positive EV bets
    - Kelly: Kelly-optimised allocation per bet, capped at total
    - Variance-aware: Markowitz-style mean-variance optimisation
    - Monte Carlo: simulated allocation
    """

    def optimize(
        self,
        slips: list[BetSlip],
        bankroll: Bankroll,
        **kwargs: Any,
    ) -> PortfolioResult:
        """Allocate bankroll across a set of concurrent bets.

        Parameters
        ----------
        slips : list[BetSlip]
            All proposed bets for the current round.
        bankroll : Bankroll
            Current bankroll state.

        Returns
        -------
        PortfolioResult
            Allocations for each bet plus portfolio metrics.
        """
        ...


# ═══════════════════════════════════════════════════════════
#  Abstract base class (ABC) with auto-logging
# ═══════════════════════════════════════════════════════════


class BettingModule(abc.ABC):
    """Optional ABC for modules that want auto-logging and metadata.

    Unlike the Protocol-based interfaces above (which focus on the
    minimum required API), this ABC provides convenience methods
    that concrete modules can inherit from.

    Usage
    -----
    ::

        class MyOddsSource(BettingModule, OddsSource):
            ...
    """

    module_type: str = "base"
    module_name: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.metadata: dict[str, Any] = {
            "module_type": self.module_type,
            "module_name": self.module_name or self.__class__.__name__,
            **(kwargs.pop("metadata", {})),
        }
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"<{self.module_type}: {self.module_name or self.__class__.__name__}>"


class BettingABC(BettingModule):
    """Full ABC for betting modules that need both Protocol and ABC features.

    Subclasses must implement all Protocol methods plus can override
    ``__init__`` for custom initialisation.
    """
    pass
