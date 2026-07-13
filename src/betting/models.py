"""
Core Data Models — shared dataclasses and enums for the betting engine.

Every module in the betting framework uses one or more of these
models to pass data between stages of the betting pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any


# ═══════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════


class Outcome(Enum):
    """Match outcome labels."""
    HOME = "H"
    DRAW = "D"
    AWAY = "A"

    @classmethod
    def from_index(cls, idx: int) -> Outcome:
        return [cls.AWAY, cls.DRAW, cls.HOME][idx]

    @classmethod
    def from_label(cls, label: str) -> Outcome:
        mapping = {e.value: e for e in cls}
        return mapping[label.upper()[0]]

    def to_index(self) -> int:
        return [Outcome.AWAY, Outcome.DRAW, Outcome.HOME].index(self)

    def to_label(self) -> str:
        return {"H": "Home Win", "D": "Draw", "A": "Away Win"}[self.value]


class BetStatus(Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    VOID = "void"       # Match abandoned, bet refunded
    CANCELLED = "cancelled"


class StakingMethod(Enum):
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    FLAT = "flat"
    CUSTOM = "custom"
    PORTFOLIO = "portfolio"


# ═══════════════════════════════════════════════════════════
#  Core dataclasses
# ═══════════════════════════════════════════════════════════


@dataclass
class MatchOdds:
    """Odds for a single match from a specific source/bookmaker.

    Parameters
    ----------
    home_odds : Decimal
        Decimal odds for home win.
    draw_odds : Decimal
        Decimal odds for draw.
    away_odds : Decimal
        Decimal odds for away win.
    source : str
        Bookmaker name or "consensus" / "model".
    timestamp : datetime, optional
        When the odds were recorded. Defaults to now.
    metadata : dict, optional
        Additional info (league, region, etc.).
    """
    home_odds: Decimal
    draw_odds: Decimal
    away_odds: Decimal
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, val in [("home_odds", self.home_odds), ("draw_odds", self.draw_odds), ("away_odds", self.away_odds)]:
            if val <= Decimal("1.0"):
                raise ValueError(f"{name} must be > 1.0, got {val}")

    def odds_for(self, outcome: Outcome) -> Decimal:
        return {"H": self.home_odds, "D": self.draw_odds, "A": self.away_odds}[outcome.value]

    def implied_prob(self, outcome: Outcome) -> Decimal:
        return Decimal("1.0") / self.odds_for(outcome)

    def implied_probs(self) -> dict[Outcome, Decimal]:
        return {o: self.implied_prob(o) for o in Outcome}

    def margin(self) -> Decimal:
        return sum(self.implied_probs().values()) - Decimal("1.0")

    def fair_prob(self, outcome: Outcome) -> Decimal:
        margin = self.margin()
        if margin <= 0:
            return self.implied_prob(outcome)
        return self.implied_prob(outcome) / (Decimal("1.0") + margin)

    def fair_probs(self) -> dict[Outcome, Decimal]:
        return {o: self.fair_prob(o) for o in Outcome}


@dataclass
class ModelPrediction:
    """Model-predicted probabilities for a single match.

    Parameters
    ----------
    home_prob : Decimal
        Model probability for home win (0-1).
    draw_prob : Decimal
        Model probability for draw (0-1).
    away_prob : Decimal
        Model probability for away win (0-1).
    model_name : str
        Identifier for the model.
    model_version : str, optional
        Model version string.
    confidence : float, optional
        Model confidence score (0-100).
    metadata : dict, optional
        Additional prediction metadata.
    """
    home_prob: Decimal
    draw_prob: Decimal
    away_prob: Decimal
    model_name: str = "unknown"
    model_version: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.home_prob + self.draw_prob + self.away_prob
        if not (Decimal("0.99") <= total <= Decimal("1.01")):
            raise ValueError(
                f"Probabilities must sum to ~1.0, got {float(total):.4f}"
            )

    def prob_for(self, outcome: Outcome) -> Decimal:
        return {"H": self.home_prob, "D": self.draw_prob, "A": self.away_prob}[outcome.value]

    def as_array(self) -> list[float]:
        """Return [away_prob, draw_prob, home_prob] (model output order)."""
        return [float(self.away_prob), float(self.draw_prob), float(self.home_prob)]

    @classmethod
    def from_array(cls, arr: list[float], **kwargs: Any) -> ModelPrediction:
        """Create from [away_prob, draw_prob, home_prob] array."""
        return cls(
            home_prob=Decimal(str(arr[2])),
            draw_prob=Decimal(str(arr[1])),
            away_prob=Decimal(str(arr[0])),
            **kwargs,
        )


@dataclass
class BetSlip:
    """A single bet proposal with all computed metrics.

    Parameters
    ----------
    match_id : str
        Unique match identifier.
    home_team : str
        Home team name.
    away_team : str
        Away team name.
    outcome : Outcome
        The outcome being bet on.
    decimal_odds : Decimal
        Odds at which the bet would be placed.
    model_prob : Decimal
        Model probability for this outcome.
    fair_prob : Decimal
        Fair (no-margin) probability from the odds source.
    odds_source : str
        Bookmaker or odds source.
    """
    match_id: str
    home_team: str
    away_team: str
    outcome: Outcome
    decimal_odds: Decimal
    model_prob: Decimal
    fair_prob: Decimal
    odds_source: str

    # Computed fields (populated by calculators)
    ev: float | None = None
    kelly_fraction: float | None = None
    fractional_kelly: float | None = None
    stake_amount: float | None = None
    stake_pct: float | None = None
    clv: float | None = None
    edge: float | None = None
    recommended: bool = False
    rank: int = 0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bet_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def match_label(self) -> str:
        return f"{self.home_team} vs {self.away_team}"

    @property
    def outcome_label(self) -> str:
        return self.outcome.to_label()

    @property
    def positive_ev(self) -> bool:
        return self.ev is not None and self.ev > 0 and self.edge is not None and self.edge > 0


@dataclass
class BetOutcome:
    """The result of a placed bet."""

    bet_slip: BetSlip
    status: BetStatus
    actual_outcome: Outcome | None = None
    profit: float | None = None
    roi_pct: float | None = None
    bankroll_before: float | None = None
    bankroll_after: float | None = None
    settled_at: datetime | None = None
    notes: str = ""


@dataclass
class Bankroll:
    """Tracks betting capital over time."""

    initial_balance: float
    currency: str = "GBP"
    current_balance: float | None = None
    peak_balance: float | None = None
    total_staked: float = 0.0
    total_profit: float = 0.0
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    history: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bankroll_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        if self.current_balance is None:
            self.current_balance = self.initial_balance
        if self.peak_balance is None:
            self.peak_balance = self.initial_balance
        if not self.history:
            self.history = [self.initial_balance]

    @property
    def roi_pct(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return ((self.current_balance or 0) - self.initial_balance) / self.initial_balance * 100

    @property
    def yield_pct(self) -> float:
        if self.total_staked == 0:
            return 0.0
        return (self.total_profit / self.total_staked) * 100

    @property
    def win_rate_pct(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return (self.winning_bets / self.total_bets) * 100

    @property
    def max_drawdown_pct(self) -> float:
        if not self.history:
            return 0.0
        peak = self.history[0]
        max_dd = 0.0
        for val in self.history:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def record_stake(self, amount: float) -> None:
        self.total_staked += amount
        if self.current_balance is not None:
            self.current_balance -= amount
            self.history.append(self.current_balance)

    def record_result(self, profit: float, won: bool) -> None:
        self.total_bets += 1
        self.total_profit += profit
        if won:
            self.winning_bets += 1
        else:
            self.losing_bets += 1
        if self.current_balance is not None:
            self.current_balance += profit
            if self.current_balance > (self.peak_balance or 0):
                self.peak_balance = self.current_balance
            self.history.append(self.current_balance)

    def reset(self) -> None:
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.total_staked = 0.0
        self.total_profit = 0.0
        self.total_bets = 0
        self.winning_bets = 0
        self.losing_bets = 0
        self.history = [self.initial_balance]


@dataclass
class BetFilterConfig:
    """Configuration for filtering bets before placement."""

    min_ev: float = 0.0
    min_edge: float = 0.0
    max_kelly_pct: float = 1.0
    min_kelly_pct: float = 0.0
    min_odds: Decimal = Decimal("1.0")
    max_odds: Decimal = Decimal("100.0")
    max_stake_amount: float | None = None
    min_confidence: float | None = None
    max_bets_per_match: int = 1
    allowed_outcomes: list[Outcome] | None = None
    max_stake_pct_of_bankroll: float = 1.0


@dataclass
class MarketFilterConfig:
    """Configuration for filtering which markets/matches to analyse."""

    min_market_confidence: float | None = None
    max_bookmaker_margin: float = 0.10  # 10%
    allowed_leagues: list[str] | None = None
    excluded_leagues: list[str] | None = None
    min_league_matches: int = 0
    require_h2h_odds: bool = True
    require_closing_odds: bool = False
    only_premium_bookmakers: bool = False
    premium_bookmakers: list[str] = field(default_factory=lambda: ["pinnacle", "bet365", "betfair"])


@dataclass
class PortfolioAllocation:
    """Allocation for a single bet within a portfolio."""

    bet_slip: BetSlip
    weight: float  # Fraction of total bankroll allocated (0-1)
    expected_return: float
    variance: float = 0.0
    sharpe_ratio: float = 0.0

    @property
    def stake_amount(self) -> float:
        return self.weight  # Weight IS the fraction of bankroll


@dataclass
class PortfolioResult:
    """Result of portfolio optimisation across multiple bets."""

    allocations: list[PortfolioAllocation]
    total_bankroll_fraction: float
    expected_return: float
    portfolio_variance: float
    sharpe_ratio: float
    method: str = "naive"
    diversified: bool = False
    optimization_metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  Aggregated metrics (for reporting)
# ═══════════════════════════════════════════════════════════


@dataclass
class BettingSessionReport:
    """Summary of a betting session or backtest period."""

    bankroll: Bankroll
    total_bets: int = 0
    positive_ev_bets: int = 0
    bets_placed: int = 0
    bets_filtered: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    roi_pct: float = 0.0
    yield_pct: float = 0.0
    win_rate_pct: float = 0.0
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    avg_edge: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    longest_win_streak: int = 0
    longest_lose_streak: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
