"""
Module Registry — central registry for betting modules.

Supports registering sources, calculators, filters, staking
strategies, and portfolio optimisers by name, with querying
and lifecycle management.

Designed to mirror the ``ModelRegistry`` pattern from ``src.models``
so users familiar with one registry can use the other.
"""

from __future__ import annotations

import logging
from typing import Any

from src.betting.models import BetSlip, PortfolioResult

logger = logging.getLogger(__name__)


class BettingRegistry:
    """Central registry for all betting modules.

    Maintains separate registries for each module type so that
    the engine can look up the right module by role.

    Parameters
    ----------
    auto_discover : bool
        If True, runs plugin discovery on init (default True).
    """

    def __init__(self, auto_discover: bool = True) -> None:
        # Internal registries: {name: instance}
        self._probability_sources: dict[str, Any] = {}
        self._odds_sources: dict[str, Any] = {}
        self._calculators: dict[str, Any] = {}
        self._staking_strategies: dict[str, Any] = {}
        self._bankroll_managers: dict[str, Any] = {}
        self._risk_managers: dict[str, Any] = {}
        self._bet_filters: dict[str, Any] = {}
        self._market_filters: dict[str, Any] = {}
        self._portfolio_optimizers: dict[str, Any] = {}

        if auto_discover:
            self.discover()

    # ── Properties ─────────────────────────────────────

    @property
    def registered_modules(self) -> dict[str, list[str]]:
        """Summary of all registered modules by type."""
        return {
            "probability_sources": list(self._probability_sources.keys()),
            "odds_sources": list(self._odds_sources.keys()),
            "calculators": list(self._calculators.keys()),
            "staking_strategies": list(self._staking_strategies.keys()),
            "bankroll_managers": list(self._bankroll_managers.keys()),
            "risk_managers": list(self._risk_managers.keys()),
            "bet_filters": list(self._bet_filters.keys()),
            "market_filters": list(self._market_filters.keys()),
            "portfolio_optimizers": list(self._portfolio_optimizers.keys()),
        }

    # ── Registration ──────────────────────────────────

    def register_probability_source(self, name: str, source: Any) -> None:
        self._probability_sources[name] = source
        logger.debug("Registered probability source: %s", name)

    def register_odds_source(self, name: str, source: Any) -> None:
        self._odds_sources[name] = source
        logger.debug("Registered odds source: %s", name)

    def register_calculator(self, name: str, calculator: Any) -> None:
        self._calculators[name] = calculator
        logger.debug("Registered calculator: %s", name)

    def register_staking_strategy(self, name: str, strategy: Any) -> None:
        self._staking_strategies[name] = strategy
        logger.debug("Registered staking strategy: %s", name)

    def register_bankroll_manager(self, name: str, manager: Any) -> None:
        self._bankroll_managers[name] = manager
        logger.debug("Registered bankroll manager: %s", name)

    def register_risk_manager(self, name: str, manager: Any) -> None:
        self._risk_managers[name] = manager
        logger.debug("Registered risk manager: %s", name)

    def register_bet_filter(self, name: str, filter_obj: Any) -> None:
        self._bet_filters[name] = filter_obj
        logger.debug("Registered bet filter: %s", name)

    def register_market_filter(self, name: str, filter_obj: Any) -> None:
        self._market_filters[name] = filter_obj
        logger.debug("Registered market filter: %s", name)

    def register_portfolio_optimizer(self, name: str, optimizer: Any) -> None:
        self._portfolio_optimizers[name] = optimizer
        logger.debug("Registered portfolio optimizer: %s", name)

    # ── Lookup ────────────────────────────────────────

    def get_probability_source(self, name: str) -> Any | None:
        return self._probability_sources.get(name)

    def get_odds_source(self, name: str) -> Any | None:
        return self._odds_sources.get(name)

    def get_calculator(self, name: str) -> Any | None:
        return self._calculators.get(name)

    def get_staking_strategy(self, name: str) -> Any | None:
        return self._staking_strategies.get(name)

    def get_bankroll_manager(self, name: str) -> Any | None:
        return self._bankroll_managers.get(name)

    def get_risk_manager(self, name: str) -> Any | None:
        return self._risk_managers.get(name)

    def get_bet_filter(self, name: str) -> Any | None:
        return self._bet_filters.get(name)

    def get_market_filter(self, name: str) -> Any | None:
        return self._market_filters.get(name)

    def get_portfolio_optimizer(self, name: str) -> Any | None:
        return self._portfolio_optimizers.get(name)

    # ── Discovery ─────────────────────────────────────

    def discover(self) -> int:
        """Auto-discover modules from installed plugins.

        Delegates to ``PluginRegistry.discover()``.

        Returns
        -------
        int
            Number of modules discovered.
        """
        try:
            from src.betting.plugins import PluginRegistry
            pr = PluginRegistry()
            return pr.discover_into(self)
        except ImportError:
            return 0

    # ── Serialisation ─────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Export registry as a dictionary for reporting."""
        return self.registered_modules

    def __repr__(self) -> str:
        counts = []
        for role, names in self.registered_modules.items():
            counts.append(f"{role}: {len(names)}")
        return f"<BettingRegistry: {', '.join(counts)}>"
