"""
Modular Betting Engine — composable framework for football betting.

Provides a Plugin-based architecture with 12 module types:

**Input modules**
    - ProbabilitySource — model probability inputs
    - OddsSource — bookmaker odds inputs

**Calculators** (stateless)
    - ExpectedValue — EV = (model_prob × odds) − 1
    - KellyCriterion — full Kelly stake %
    - FractionalKelly — conservative Kelly variant
    - FlatStake — fixed stake per bet
    - ClosingLineValue — CLV = fair_close − fair_open

**Management modules** (stateful)
    - BankrollManager — balance, P&L, history tracking
    - RiskManager — drawdown limits, daily loss, exposure

**Filters**
    - BetFilter — per-bet acceptance rules
    - MarketFilter — per-match market suitability rules

**Optimisation**
    - PortfolioOptimizer — multi-bet bankroll allocation

Quick Start
-----------
::

    from src.betting.engine import BettingEngine

    engine = BettingEngine()

    matches = [
        {
            "match_id": "m1",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "model_probs": [0.52, 0.28, 0.20],  # [away, draw, home]
            "odds": {"home_odds": 2.10, "draw_odds": 3.40, "away_odds": 3.80},
        }
    ]
    report = engine.run_pipeline(matches, staking_method="fractional_kelly")
    engine.print_summary()
"""

from __future__ import annotations

from src.betting.base import (
    BankrollManager,
    BetFilter,
    BettingModule,
    ClosingLineValueCalculator,
    ExpectedValueCalculator,
    FlatStakeCalculator,
    FractionalKellyCalculator,
    KellyCalculator,
    MarketFilter,
    OddsSource,
    PortfolioOptimizer,
    ProbabilitySource,
    RiskManager,
)
from src.betting.calculator import (
    CLVCalculator,
    CalculatorFactory,
    EVCalculator,
    FlatStakeCalculator as FlatStakeCalc,
    FractionalKellyCalculator as FracKellyCalc,
    KellyCalculator as KellyCalc,
)
from src.betting.cli import BettingCLI, main
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
from src.betting.factory import EngineFactory
from src.betting.models import (
    Bankroll,
    BetFilterConfig as BetFilterConfigModel,
    BetOutcome,
    BetSlip,
    BetStatus,
    BettingSessionReport,
    MarketFilterConfig as MarketFilterConfigModel,
    MatchOdds,
    ModelPrediction,
    Outcome,
    PortfolioAllocation,
    PortfolioResult,
    StakingMethod,
)
from src.betting.plugins import PluginRegistry
from src.betting.registry import BettingRegistry

__all__ = [
    # Engine
    "BettingEngine",
    "EngineFactory",
    "BettingCLI",
    "main",
    # Protocols / ABCs
    "ProbabilitySource",
    "OddsSource",
    "ExpectedValueCalculator",
    "KellyCalculator",
    "FractionalKellyCalculator",
    "FlatStakeCalculator",
    "ClosingLineValueCalculator",
    "BankrollManager",
    "RiskManager",
    "BetFilter",
    "MarketFilter",
    "PortfolioOptimizer",
    "BettingModule",
    # Default implementations
    "DefaultBankrollManager",
    "DefaultRiskManager",
    "DefaultBetFilter",
    "DefaultMarketFilter",
    "DefaultPortfolioOptimizer",
    # Calculators
    "EVCalculator",
    "KellyCalc",
    "FracKellyCalc",
    "FlatStakeCalc",
    "CLVCalculator",
    "CalculatorFactory",
    # Registry / Plugins
    "BettingRegistry",
    "PluginRegistry",
    # Configs
    "BetFilterConfig",
    "BetFilterConfigModel",
    "MarketFilterConfig",
    "MarketFilterConfigModel",
    # Models
    "MatchOdds",
    "ModelPrediction",
    "BetSlip",
    "BetOutcome",
    "BetStatus",
    "Bankroll",
    "Outcome",
    "StakingMethod",
    "PortfolioAllocation",
    "PortfolioResult",
    "BettingSessionReport",
]

# ── Version ──────────────────────────────────────────

__version__ = "0.1.0"
