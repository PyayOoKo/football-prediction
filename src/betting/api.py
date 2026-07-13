"""
REST API Integration — bridge between the betting engine and a web API.

Provides request/response models and handlers that can be plugged into
a FastAPI, Flask, or Streamlit application.

This module does NOT implement a full web server.  It provides the
data contracts and handler functions that a web framework would call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.betting.engine import BettingEngine
from src.betting.factory import EngineFactory
from src.betting.models import (
    Bankroll,
    BetFilterConfig,
    BetSlip,
    BettingSessionReport,
    MarketFilterConfig,
    Outcome,
    PortfolioResult,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  API request / response models
# ═══════════════════════════════════════════════════════════


@dataclass
class EvaluateBetsRequest:
    """Request body for ``POST /api/v1/bets/evaluate``."""

    matches: list[dict[str, Any]] = field(default_factory=list)
    staking_method: str = "fractional_kelly"
    staking_params: dict[str, Any] = field(default_factory=lambda: {"fraction": 0.25})
    initial_bankroll: float = 1000.0
    min_ev: float = 0.0
    max_bookmaker_margin: float = 0.10


@dataclass
class EvaluateBetsResponse:
    """Response body for ``POST /api/v1/bets/evaluate``."""

    status: str = "ok"
    bets: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class BankrollStatusResponse:
    """Response body for ``GET /api/v1/bankroll``."""

    initial_balance: float = 0.0
    current_balance: float = 0.0
    total_staked: float = 0.0
    total_profit: float = 0.0
    roi_pct: float = 0.0
    yield_pct: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_bets: int = 0
    currency: str = "GBP"


# ═══════════════════════════════════════════════════════════
#  API handlers (framework-agnostic)
# ═══════════════════════════════════════════════════════════


class BettingAPI:
    """Framework-agnostic handler for betting API endpoints.

    Designed to be called from FastAPI, Flask, or Streamlit route handlers.
    Manages a singleton engine instance across requests.

    Usage (FastAPI)
    -----
    ::

        from fastapi import FastAPI
        from src.betting.api import BettingAPI

        app = FastAPI()
        api = BettingAPI()

        @app.post("/api/v1/bets/evaluate")
        async def evaluate_bets(req: dict):
            return api.evaluate_bets(**req)
    """

    def __init__(self, engine: BettingEngine | None = None) -> None:
        self._engine = engine

    @property
    def engine(self) -> BettingEngine:
        """Get or create the default engine."""
        if self._engine is None:
            self._engine = EngineFactory.create()
            self._engine_created = True
        return self._engine

    # ── Endpoint handlers ─────────────────────────────

    def evaluate_bets(self, request: EvaluateBetsRequest | dict[str, Any]) -> dict[str, Any]:
        """Handle a bet evaluation request.

        Maps directly to ``POST /api/v1/bets/evaluate``.
        """
        if isinstance(request, dict):
            request = EvaluateBetsRequest(**request)

        errors: list[str] = []

        # Create a fresh engine for this request
        engine = EngineFactory.create(
            staking_method=request.staking_method,
            staking_params=request.staking_params,
            initial_bankroll=request.initial_bankroll,
            bet_filter_config=BetFilterConfig(min_ev=request.min_ev),
            market_filter_config=MarketFilterConfig(
                max_bookmaker_margin=request.max_bookmaker_margin,
            ),
        )

        if not request.matches:
            return EvaluateBetsResponse(
                status="error",
                errors=["No matches provided"],
            ).__dict__

        try:
            report = engine.run_pipeline(
                matches=request.matches,
                staking_method=request.staking_method,
                staking_params=request.staking_params,
            )

            bets_data = [_slip_to_dict(s) for s in engine.pending_slips]
            report_data = _report_to_dict(report)

            return EvaluateBetsResponse(
                bets=bets_data,
                report=report_data,
            ).__dict__

        except Exception as exc:
            logger.error("Bet evaluation failed: %s", exc, exc_info=True)
            return EvaluateBetsResponse(
                status="error",
                errors=[str(exc)],
            ).__dict__

    def get_bankroll_status(self) -> dict[str, Any]:
        """Handle a bankroll status request.

        Maps directly to ``GET /api/v1/bankroll``.
        """
        bk = self.engine.bankroll.bankroll
        return BankrollStatusResponse(
            initial_balance=bk.initial_balance,
            current_balance=bk.current_balance or 0,
            total_staked=bk.total_staked,
            total_profit=bk.total_profit,
            roi_pct=bk.roi_pct,
            yield_pct=bk.yield_pct,
            win_rate_pct=bk.win_rate_pct,
            max_drawdown_pct=bk.max_drawdown_pct,
            total_bets=bk.total_bets,
            currency=bk.currency,
        ).__dict__

    def get_engine_config(self) -> dict[str, Any]:
        """Handle a config status request.

        Maps directly to ``GET /api/v1/config``.
        """
        return {
            "registry": self.engine.registry.registered_modules,
            "bankroll_initial": self.engine.bankroll.bankroll.initial_balance,
            "bankroll_current": self.engine.bankroll.bankroll.current_balance,
            "modules": {
                "bankroll": type(self.engine.bankroll).__name__,
                "risk": type(self.engine.risk).__name__,
                "bet_filter": type(self.engine.bet_filter).__name__,
                "market_filter": type(self.engine.market_filter).__name__,
                "portfolio": type(self.engine.portfolio_optimizer).__name__,
            },
        }


# ═══════════════════════════════════════════════════════════
#  Serialisation helpers
# ═══════════════════════════════════════════════════════════


def _slip_to_dict(slip: BetSlip) -> dict[str, Any]:
    """Convert a BetSlip to a JSON-serialisable dict."""
    return {
        "bet_id": slip.bet_id,
        "match": slip.match_label,
        "home_team": slip.home_team,
        "away_team": slip.away_team,
        "outcome": slip.outcome.value,
        "decimal_odds": float(slip.decimal_odds),
        "model_prob": float(slip.model_prob),
        "fair_prob": float(slip.fair_prob),
        "odds_source": slip.odds_source,
        "ev": slip.ev,
        "kelly_fraction": slip.kelly_fraction,
        "stake_amount": slip.stake_amount,
        "stake_pct": slip.stake_pct,
        "clv": slip.clv,
        "edge": slip.edge,
        "recommended": slip.recommended,
        "rank": slip.rank,
        "positive_ev": slip.positive_ev,
    }


def _report_to_dict(report: BettingSessionReport) -> dict[str, Any]:
    """Convert a BettingSessionReport to a dict."""
    return {
        "total_bets": report.total_bets,
        "positive_ev_bets": report.positive_ev_bets,
        "bets_placed": report.bets_placed,
        "total_staked": report.total_staked,
        "total_profit": report.total_profit,
        "roi_pct": report.roi_pct,
        "yield_pct": report.yield_pct,
        "win_rate_pct": report.win_rate_pct,
        "avg_odds": report.avg_odds,
        "avg_ev": report.avg_ev,
        "avg_edge": report.avg_edge,
        "max_drawdown_pct": report.max_drawdown_pct,
        "profit_factor": report.profit_factor,
        "duration_seconds": report.duration_seconds,
        "start_time": report.start_time.isoformat() if report.start_time else None,
        "end_time": report.end_time.isoformat() if report.end_time else None,
    }
