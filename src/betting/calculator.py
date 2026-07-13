"""
Calculator Modules — stateless computation functions and factory for
the 5 calculator types: EV, Kelly, Fractional Kelly, Flat Stake, CLV.

Each calculator is a namespace of pure functions (no state) that
implements the corresponding Protocol from ``src.betting.base``.
"""

from __future__ import annotations

import logging
from typing import Any

from src.betting.models import MatchOdds, Outcome

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Expected Value
# ═══════════════════════════════════════════════════════════


class EVCalculator:
    """Expected Value calculator — stateless, pure functions.

    Formula: EV = (model_prob × decimal_odds) − 1
             edge = model_prob − fair_prob
    """

    @staticmethod
    def calculate_ev(
        model_prob: float, decimal_odds: float, **kwargs: Any,
    ) -> float:
        """Compute expected value.

        Parameters
        ----------
        model_prob : float
            Model-predicted probability (0-1).
        decimal_odds : float
            Bookmaker decimal odds.

        Returns
        -------
        float
            Expected value per unit staked (e.g. 0.05 = 5%).
        """
        if decimal_odds <= 1.0 or model_prob <= 0.0 or model_prob >= 1.0:
            return 0.0
        return (model_prob * decimal_odds) - 1.0

    @staticmethod
    def calculate_edge(
        model_prob: float, fair_prob: float, **kwargs: Any,
    ) -> float:
        """Compute probability edge over the market.

        Parameters
        ----------
        model_prob : float
            Model-predicted probability (0-1).
        fair_prob : float
            Fair (no-margin) probability from bookmaker odds (0-1).

        Returns
        -------
        float
            Probability difference (e.g. 0.05 = 5 percentage points edge).
        """
        if fair_prob <= 0.0 or model_prob <= 0.0:
            return 0.0
        return model_prob - fair_prob


# ═══════════════════════════════════════════════════════════
#  Kelly Criterion
# ═══════════════════════════════════════════════════════════


class KellyCalculator:
    """Full Kelly Criterion stake calculator.

    Formula: f* = (p × b − 1) / (b − 1)
        where p = model probability, b = decimal odds (European)

    The result is the fraction of bankroll to wager for maximising
    long-term logarithmic growth.  Returns 0.0 for negative-EV bets.
    """

    @staticmethod
    def calculate(
        model_prob: float, decimal_odds: float, **kwargs: Any,
    ) -> float:
        """Compute the full Kelly fraction.

        Parameters
        ----------
        model_prob : float
            Model-predicted probability (0-1).
        decimal_odds : float
            Bookmaker decimal odds.

        Returns
        -------
        float
            Fraction of bankroll to stake (0 to 1).  Returns 0.0 if
            the bet has no positive edge.
        """
        if decimal_odds <= 1.0 or model_prob <= 0.0 or model_prob >= 1.0:
            return 0.0

        ev = (model_prob * decimal_odds) - 1.0
        if ev <= 0.0:
            return 0.0

        # Full Kelly: f = (p × (b - 1) - (1 - p)) / (b - 1)
        # Simplified: f = (p × b - 1) / (b - 1)
        kelly = (model_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
        return max(min(kelly, 1.0), 0.0)


# ═══════════════════════════════════════════════════════════
#  Fractional Kelly
# ═══════════════════════════════════════════════════════════


class FractionalKellyCalculator:
    """Fractional Kelly — conservative variant of full Kelly.

    Formula: f_frac = f_full × fraction

    Common fractions:
    - 0.25 (25% Kelly) — conservative, recommended for most bettors
    - 0.50 (50% Kelly) — moderately aggressive
    - 0.10 (10% Kelly) — very conservative, low variance
    """

    @staticmethod
    def calculate(
        model_prob: float,
        decimal_odds: float,
        fraction: float = 0.25,
        **kwargs: Any,
    ) -> float:
        """Compute fractional Kelly stake fraction.

        Parameters
        ----------
        model_prob : float
            Model-predicted probability (0-1).
        decimal_odds : float
            Bookmaker decimal odds.
        fraction : float
            Fraction of full Kelly to use (default 0.25).

        Returns
        -------
        float
            Fraction of bankroll to stake (0 to 1).
        """
        full_kelly = KellyCalculator.calculate(model_prob, decimal_odds)
        return max(full_kelly * fraction, 0.0)


# ═══════════════════════════════════════════════════════════
#  Flat Stake
# ═══════════════════════════════════════════════════════════


class FlatStakeCalculator:
    """Fixed stake size regardless of edge.

    Supports two modes:
    - ``stake_per_bet`` — fixed currency amount (e.g. £25 per bet)
    - ``stake_pct`` — fixed percentage of bankroll (e.g. 2%)
    """

    @staticmethod
    def calculate(
        bankroll: float,
        stake_per_bet: float | None = None,
        stake_pct: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute the flat stake amount.

        Parameters
        ----------
        bankroll : float
            Current bankroll balance.
        stake_per_bet : float, optional
            Fixed currency amount per bet. Takes priority.
        stake_pct : float, optional
            Fixed percentage of bankroll (0-1).

        Returns
        -------
        float
            Stake amount in currency units.
        """
        if stake_per_bet is not None and stake_per_bet > 0:
            return min(stake_per_bet, bankroll)

        if stake_pct is not None and 0 < stake_pct <= 1:
            return bankroll * stake_pct

        # Default: 2% of bankroll
        return bankroll * 0.02


# ═══════════════════════════════════════════════════════════
#  Closing Line Value
# ═══════════════════════════════════════════════════════════


class CLVCalculator:
    """Closing Line Value calculator.

    CLV measures how the market moved from opening to closing.
    Positive CLV for an outcome means its fair probability
    INCREASED (odds shortened — market moved toward it).

    Formula: CLV_outcome = fair_prob_closing − fair_prob_opening
    """

    @staticmethod
    def calculate_clv(
        outcome: Outcome,
        opening_odds: MatchOdds,
        closing_odds: MatchOdds,
        **kwargs: Any,
    ) -> float:
        """Compute CLV for a single outcome.

        Parameters
        ----------
        outcome : Outcome
            The match outcome.
        opening_odds : MatchOdds
            Odds at market open.
        closing_odds : MatchOdds
            Odds at market close (kick-off).

        Returns
        -------
        float
            CLV as a probability difference.
            Positive = market moved toward this outcome.
        """
        fair_open = opening_odds.fair_prob(outcome)
        fair_close = closing_odds.fair_prob(outcome)
        return float(fair_close - fair_open)

    @staticmethod
    def calculate_all_clv(
        opening_odds: MatchOdds,
        closing_odds: MatchOdds,
        **kwargs: Any,
    ) -> dict[Outcome, float]:
        """Compute CLV for all three outcomes."""
        return {
            o: CLVCalculator.calculate_clv(o, opening_odds, closing_odds)
            for o in Outcome
        }


# ═══════════════════════════════════════════════════════════
#  Calculator registry / factory
# ═══════════════════════════════════════════════════════════


class CalculatorFactory:
    """Factory for creating calculator instances.

    Maps calculator type names to their implementations.
    """

    _REGISTRY: dict[str, type] = {
        "ev": EVCalculator,
        "kelly": KellyCalculator,
        "fractional_kelly": FractionalKellyCalculator,
        "flat_stake": FlatStakeCalculator,
        "clv": CLVCalculator,
    }

    @classmethod
    def create(cls, calculator_type: str, **kwargs: Any) -> Any:
        """Create a calculator instance by type name.

        Parameters
        ----------
        calculator_type : str
            One of: ``ev``, ``kelly``, ``fractional_kelly``, ``flat_stake``, ``clv``.
        **kwargs
            Passed to the calculator constructor.

        Returns
        -------
        object
            Calculator instance.

        Raises
        ------
        ValueError
            If the calculator type is unknown.
        """
        calc_cls = cls._REGISTRY.get(calculator_type)
        if calc_cls is None:
            raise ValueError(
                f"Unknown calculator type: {calculator_type}. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return calc_cls(**kwargs)

    @classmethod
    def register(cls, name: str, calculator_cls: type) -> None:
        """Register a custom calculator type.

        Parameters
        ----------
        name : str
            Unique type identifier.
        calculator_cls : type
            Calculator class. Should implement the corresponding Protocol.
        """
        if name in cls._REGISTRY:
            logger.warning("Overwriting calculator '%s'", name)
        cls._REGISTRY[name] = calculator_cls
        logger.info("Registered calculator: %s (%s)", name, calculator_cls.__name__)

    @classmethod
    def list_types(cls) -> list[str]:
        """List all available calculator type names."""
        return list(cls._REGISTRY.keys())
