"""
Feature Implementations — concrete FeatureTransformer subclasses.

Each module in this package implements a ``FeatureTransformer`` for a
specific domain of football analytics.  All transformers are auto-
discovered by ``FeaturePluginRegistry`` and follow the lifecycle
defined in ``src.feature_framework.base``.

Available transformers
---------------------
- :class:`TeamFormTransformer` — rolling form statistics
- :class:`EloTransformer` — Elo rating system
- :class:`ScheduleTransformer` — fixture schedule / congestion features
"""

from __future__ import annotations

from src.feature_framework.features.betting_market import (
    BettingMarketTransformer,
    create_betting_market_transformer,
)
from src.feature_framework.features.team_form import TeamFormTransformer
from src.feature_framework.features.elo_rating import (
    EloTransformer,
    EloEngine,
    EloMatchRecord,
    EloSnapshot,
    create_elo_transformer,
)
from src.feature_framework.features.schedule import (
    ScheduleTransformer,
    create_schedule_transformer,
)
from src.feature_framework.features.h2h import (
    H2HTransformer,
    create_h2h_transformer,
)

__all__ = [
    "BettingMarketTransformer",
    "create_betting_market_transformer",
    "TeamFormTransformer",
    "EloTransformer",
    "EloEngine",
    "EloMatchRecord",
    "EloSnapshot",
    "create_elo_transformer",
    "ScheduleTransformer",
    "create_schedule_transformer",
    "H2HTransformer",
    "create_h2h_transformer",
]
