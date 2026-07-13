"""
ModelFactory — builds BettingEngine instances from configuration.

Supports config-driven engine creation with:
- Default staking methods and parameters
- Custom module overrides
- Automatic plugin discovery
"""

from __future__ import annotations

import logging
from typing import Any

from src.betting.engine import (
    BetFilterConfig,
    BettingEngine,
    DefaultBankrollManager,
    DefaultBetFilter,
    DefaultMarketFilter,
    DefaultPortfolioOptimizer,
    DefaultRiskManager,
    MarketFilterConfig,
)
from src.betting.registry import BettingRegistry

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating configured BettingEngine instances.

    Usage
    -----
    ::

        from src.betting.factory import EngineFactory

        engine = EngineFactory.create(
            staking_method="fractional_kelly",
            staking_params={"fraction": 0.25},
            initial_bankroll=2000.0,
        )
    """

    @staticmethod
    def create(
        staking_method: str = "fractional_kelly",
        staking_params: dict[str, Any] | None = None,
        initial_bankroll: float = 1000.0,
        registry: BettingRegistry | None = None,
        bankroll_manager: Any | None = None,
        risk_manager: Any | None = None,
        bet_filter: Any | None = None,
        market_filter: Any | None = None,
        portfolio_optimizer: Any | None = None,
        bet_filter_config: BetFilterConfig | None = None,
        market_filter_config: MarketFilterConfig | None = None,
        **kwargs: Any,
    ) -> BettingEngine:
        """Create and return a configured BettingEngine.

        Parameters
        ----------
        staking_method : str
            One of: ``kelly``, ``fractional_kelly``, ``flat_stake``.
        staking_params : dict, optional
            Parameters for the staking calculator.
        initial_bankroll : float
            Starting bankroll balance.
        registry : BettingRegistry, optional
        bankroll_manager : Any, optional
        risk_manager : Any, optional
        bet_filter : Any, optional
        market_filter : Any, optional
        portfolio_optimizer : Any, optional
        bet_filter_config : BetFilterConfig, optional
        market_filter_config : MarketFilterConfig, optional
        **kwargs
            Additional engine parameters.

        Returns
        -------
        BettingEngine
        """
        # Build defaults if not provided
        if bankroll_manager is None:
            bankroll_manager = DefaultBankrollManager(
                initial_balance=initial_bankroll,
                currency=kwargs.get("currency", "GBP"),
            )
        if risk_manager is None:
            risk_manager = DefaultRiskManager(
                max_drawdown_pct=kwargs.get("max_drawdown_pct", 50.0),
                max_stake_pct=kwargs.get("max_stake_pct", 0.25),
            )
        if bet_filter is None:
            bet_filter = DefaultBetFilter()
        if market_filter is None:
            market_filter = DefaultMarketFilter()
        if portfolio_optimizer is None:
            portfolio_optimizer = DefaultPortfolioOptimizer()

        # Build configs
        if bet_filter_config is None:
            bet_filter_config = BetFilterConfig(
                min_ev=kwargs.get("min_ev", 0.0),
                max_odds=DecimalOrFloat(kwargs.get("max_odds", 100.0)),
                min_odds=DecimalOrFloat(kwargs.get("min_odds", 1.0)),
            )
        if market_filter_config is None:
            market_filter_config = MarketFilterConfig(
                max_bookmaker_margin=kwargs.get("max_bookmaker_margin", 0.10),
            )

        engine = BettingEngine(
            registry=registry,
            bankroll_manager=bankroll_manager,
            risk_manager=risk_manager,
            bet_filter=bet_filter,
            market_filter=market_filter,
            portfolio_optimizer=portfolio_optimizer,
            bet_filter_config=bet_filter_config,
            market_filter_config=market_filter_config,
        )

        logger.info(
            "Created BettingEngine — bankroll=%.0f, method=%s, params=%s",
            initial_bankroll, staking_method, staking_params or {},
        )
        return engine


def DecimalOrFloat(value: Any) -> Any:
    """Convert a value to the appropriate type for BetFilterConfig fields."""
    from decimal import Decimal
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return value
