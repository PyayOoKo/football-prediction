"""
Stake Sizing Strategies — unified interface for bet stake calculation.

Each strategy implements a ``calculate_stake()`` method with the same
signature, allowing them to be used interchangeably in betting pipelines.

Strategies
----------
- **FlatStaking** — fixed currency amount per bet (ignores odds/prob)
- **PercentageStaking** — fixed % of current bankroll per bet
- **FixedRatioStaking** — stake = fixed_ratio × bankroll
- **VariableRatioStaking** — stake = (EV + 1) × fixed_ratio × bankroll
- **VolatilityStaking** — adjusts stake based on recent result volatility
- **PortfolioStaking** — divides bankroll across concurrent bets
- **KellyStaking** — full Kelly Criterion (optimal growth)
- **FractionalKellyStaking** — fractional Kelly for reduced variance

Usage
-----
::

    from src.betting.staking import (
        FlatStaking, FixedRatioStaking, VariableRatioStaking,
        VolatilityStaking, PortfolioStaking, KellyStaking,
    )

    # Fixed ratio: 2% of bankroll
    strategy = FixedRatioStaking(ratio=0.02)
    stake = strategy.calculate_stake(
        model_prob=0.60, decimal_odds=2.00, bankroll=1000, ev=0.20,
    )  # → 20.0

    # Variable ratio: scales with EV magnitude
    strategy = VariableRatioStaking(base_ratio=0.02)
    stake = strategy.calculate_stake(
        model_prob=0.60, decimal_odds=2.00, bankroll=1000, ev=0.20,
    )  # → 24.0  (20 × 1.20)

    # Volatility-based: adjusts down in high-volatility periods
    strategy = VolatilityStaking(base_ratio=0.02, window=10)
    stake = strategy.calculate_stake(
        model_prob=0.60, decimal_odds=2.00, bankroll=1000, ev=0.20,
    )  # Depends on recent volatility

    # Portfolio: divide bankroll across concurrent bets
    strategy = PortfolioStaking(total_concurrent_bets=3)
    stake = strategy.calculate_stake(
        model_prob=0.60, decimal_odds=2.00, bankroll=1000, ev=0.20,
    )  # → 333.33 (bankroll / 3)
"""

from __future__ import annotations

import abc
import collections
import logging
from typing import Any

import numpy as np

from src.betting.kelly import calculate_kelly, calculate_fractional_kelly

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  Abstract base
# ══════════════════════════════════════════════════════════════════


class StakingStrategy(abc.ABC):
    """Abstract base for all stake sizing strategies.

    Subclasses must implement ``calculate_stake()``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.metadata: dict[str, Any] = {
            "strategy": self.__class__.__name__,
            **kwargs,
        }

    @abc.abstractmethod
    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Compute the stake amount for a single bet.

        Parameters
        ----------
        model_prob : float
            Model-predicted probability of the outcome (0 to 1).
        decimal_odds : float
            Bookmaker decimal odds.
        bankroll : float
            Current bankroll balance.
        ev : float
            Expected value of the bet (as a decimal, e.g. 0.05 = 5% edge).
            Default 0.0.

        Returns
        -------
        float
            Stake amount in currency units.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __str__(self) -> str:
        return self.__class__.__name__


# ══════════════════════════════════════════════════════════════════
#  Flat Staking
# ══════════════════════════════════════════════════════════════════


class FlatStaking(StakingStrategy):
    """Stake a fixed currency amount on every bet, regardless of edge.

    Parameters
    ----------
    stake_per_bet : float
        Fixed amount to stake on each bet (e.g. 25.0 for £25).
        Must be > 0.

    Examples
    --------
    >>> strategy = FlatStaking(stake_per_bet=25.0)
    >>> strategy.calculate_stake(0.60, 2.00, 1000)
    25.0
    >>> strategy.calculate_stake(0.30, 3.00, 500)
    25.0
    """

    def __init__(self, stake_per_bet: float = 25.0) -> None:
        if stake_per_bet <= 0:
            raise ValueError(f"stake_per_bet must be > 0, got {stake_per_bet}")
        super().__init__(stake_per_bet=stake_per_bet)
        self.stake_per_bet = stake_per_bet

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the fixed stake amount, capped at the current bankroll."""
        _ = model_prob, decimal_odds, ev  # unused for flat staking
        return min(self.stake_per_bet, max(bankroll, 0.0))


# ══════════════════════════════════════════════════════════════════
#  Percentage Staking
# ══════════════════════════════════════════════════════════════════


class PercentageStaking(StakingStrategy):
    """Stake a fixed percentage of the current bankroll on every bet.

    The percentage is applied regardless of edge — all qualifying bets
    receive the same fraction of bankroll.

    Parameters
    ----------
    stake_pct : float
        Fraction of bankroll to stake per bet (0 to 1).
        E.g. 0.02 = 2% of bankroll per bet.

    Examples
    --------
    >>> strategy = PercentageStaking(stake_pct=0.02)
    >>> strategy.calculate_stake(0.60, 2.00, 1000)
    20.0
    >>> strategy.calculate_stake(0.45, 2.50, 500)
    10.0
    """

    def __init__(self, stake_pct: float = 0.02) -> None:
        if not 0 < stake_pct <= 1:
            raise ValueError(f"stake_pct must be in (0, 1], got {stake_pct}")
        super().__init__(stake_pct=stake_pct)
        self.stake_pct = stake_pct

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the percentage-based stake, capped at bankroll."""
        _ = model_prob, decimal_odds, ev  # unused for percentage staking
        return round(max(bankroll, 0.0) * self.stake_pct, 2)


# ══════════════════════════════════════════════════════════════════
#  Fixed Ratio Staking
# ══════════════════════════════════════════════════════════════════


class FixedRatioStaking(StakingStrategy):
    """Stake a fixed ratio of the current bankroll on every bet.

    Formula: ``stake = ratio × bankroll``

    Unlike ``PercentageStaking``, the ratio is expressed as a simple
    multiplier (e.g. 0.02 = 2%) rather than a percentage float.  This
    is mathematically identical but uses naming more commonly found in
    professional betting frameworks.

    Parameters
    ----------
    ratio : float
        Fraction of bankroll to stake per bet (0 to 1).
        E.g. 0.02 = 2% of bankroll per bet.

    Examples
    --------
    >>> strategy = FixedRatioStaking(ratio=0.02)
    >>> strategy.calculate_stake(0.60, 2.00, 1000)
    20.0
    >>> strategy.calculate_stake(0.45, 2.50, 500)
    10.0
    """

    def __init__(self, ratio: float = 0.02) -> None:
        if not 0 < ratio <= 1:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        super().__init__(ratio=ratio)
        self.ratio = ratio

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the fixed-ratio stake amount, capped at bankroll.

        Stake = ratio × bankroll, regardless of edge magnitude (but
        negative-EV bets return 0.0).
        """
        _ = model_prob, decimal_odds  # unused for fixed ratio staking
        if ev <= 0.0 and bankroll > 0:
            return 0.0
        return round(max(bankroll, 0.0) * self.ratio, 2)


# ══════════════════════════════════════════════════════════════════
#  Variable Ratio Staking
# ══════════════════════════════════════════════════════════════════


class VariableRatioStaking(StakingStrategy):
    """Stake scaled by EV magnitude — larger edges get larger stakes.

    Formula: ``stake = base_ratio × bankroll × (1 + ev)``

    The base ratio is multiplied by ``(1 + ev)`` so that a 5% edge
    gets 5% more stake than the baseline, while a 20% edge gets 20%
    more.  This is a simple yet effective way to size bets proportionally
    to conviction while remaining conservative on marginal edges.

    Parameters
    ----------
    base_ratio : float
        Baseline fraction of bankroll (0 to 1).
        E.g. 0.02 = 2% of bankroll as the baseline.
    max_ratio : float
        Maximum allowed fraction of bankroll, regardless of EV
        (default 0.10 = 10%).  Prevents over-exposure on extreme edges.
    min_ev : float
        Minimum EV threshold — bets below this return 0.0 (default 0.0).

    Examples
    --------
    >>> strategy = VariableRatioStaking(base_ratio=0.02)
    >>> strategy.calculate_stake(0.60, 2.00, 1000, ev=0.20)
    24.0
    >>> strategy.calculate_stake(0.55, 1.80, 1000, ev=0.0)
    0.0
    """

    def __init__(
        self,
        base_ratio: float = 0.02,
        max_ratio: float = 0.10,
        min_ev: float = 0.0,
    ) -> None:
        if not 0 < base_ratio <= 1:
            raise ValueError(f"base_ratio must be in (0, 1], got {base_ratio}")
        if not 0 < max_ratio <= 1:
            raise ValueError(f"max_ratio must be in (0, 1], got {max_ratio}")
        if max_ratio < base_ratio:
            raise ValueError(
                f"max_ratio ({max_ratio}) must be >= base_ratio ({base_ratio})"
            )
        super().__init__(
            base_ratio=base_ratio, max_ratio=max_ratio, min_ev=min_ev,
        )
        self.base_ratio = base_ratio
        self.max_ratio = max_ratio
        self.min_ev = min_ev

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the EV-scaled stake amount.

        Stake = min(base_ratio × bankroll × (1 + ev), max_ratio × bankroll)

        Returns 0.0 if ``ev < min_ev``.
        """
        _ = model_prob, decimal_odds  # ev captures the edge already
        if ev < self.min_ev or bankroll <= 0:
            return 0.0

        scaled_ratio = self.base_ratio * (1.0 + ev)
        capped_ratio = min(scaled_ratio, self.max_ratio)
        return round(max(bankroll, 0.0) * capped_ratio, 2)


# ══════════════════════════════════════════════════════════════════
#  Volatility-Based Staking
# ══════════════════════════════════════════════════════════════════


class VolatilityStaking(StakingStrategy):
    """Adjust stake size based on recent betting result volatility.

    In high-volatility periods (large swings in recent results), the
    stake is reduced to preserve bankroll.  In low-volatility periods,
    the stake returns to the base ratio.

    Volatility is measured as the **coefficient of variation** (CV)
    of recent CLV or profit values stored in the strategy's rolling
    window.  CV = std / mean; higher CV → higher volatility.

    Formula::

        cv = std(recent_results) / max(|mean(recent_results)|, epsilon)
        volatility_factor = 1.0 / (1.0 + cv * sensitivity)
        stake = base_ratio × volatility_factor × bankroll

    Parameters
    ----------
    base_ratio : float
        Baseline fraction of bankroll (0 to 1).  Default 0.02 (2%).
    window : int
        Number of recent results to track for volatility (default 10).
        Must be >= 3 for meaningful statistics.
    sensitivity : float
        How aggressively to reduce stake with increasing volatility.
        Higher values = more reduction (default 2.0).
    min_ratio : float
        Minimum allowed fraction of bankroll even at extreme volatility
        (default 0.002 = 0.2%).  Prevents stake from going to zero.
    min_ev : float
        Minimum EV threshold — bets below this return 0.0 (default 0.0).

    Examples
    --------
    >>> strategy = VolatilityStaking(base_ratio=0.02, window=10)
    >>> strategy.record_result(0.0244)   # record CLV after each bet
    >>> strategy.record_result(-0.0150)
    >>> stake = strategy.calculate_stake(0.60, 2.00, 1000, ev=0.20)
    """

    def __init__(
        self,
        base_ratio: float = 0.02,
        window: int = 10,
        sensitivity: float = 2.0,
        min_ratio: float = 0.002,
        min_ev: float = 0.0,
    ) -> None:
        if not 0 < base_ratio <= 1:
            raise ValueError(f"base_ratio must be in (0, 1], got {base_ratio}")
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")
        if not 0 <= min_ratio <= base_ratio:
            raise ValueError(
                f"min_ratio ({min_ratio}) must be in [0, base_ratio ({base_ratio})]"
            )
        super().__init__(
            base_ratio=base_ratio, window=window,
            sensitivity=sensitivity, min_ratio=min_ratio, min_ev=min_ev,
        )
        self.base_ratio = base_ratio
        self.window = window
        self.sensitivity = sensitivity
        self.min_ratio = min_ratio
        self.min_ev = min_ev

        # Rolling window of recent results (CLV or profit values)
        self._recent_results: collections.deque[float] = collections.deque(
            maxlen=window,
        )

    # ── Public API ──────────────────────────────────────────

    def record_result(self, clv_or_profit: float) -> None:
        """Record a recent betting result for volatility calculation.

        Call this after each bet settles with the CLV or profit value.
        Higher absolute values increase measured volatility.

        Parameters
        ----------
        clv_or_profit : float
            CLV or profit amount from the settled bet.
        """
        self._recent_results.append(float(clv_or_profit))
        logger.debug(
            "VolatilityStaking: recorded result %.6f — window=%d/%d",
            clv_or_profit, len(self._recent_results), self.window,
        )

    def reset_history(self) -> None:
        """Clear the recent results history (resets volatility to baseline)."""
        self._recent_results.clear()
        logger.debug("VolatilityStaking: history reset")

    @property
    def current_volatility(self) -> float:
        """Compute the current coefficient of variation.

        Returns
        -------
        float
            CV = std / max(|mean|, 1e-8).  0.0 if insufficient data.
        """
        if len(self._recent_results) < 3:
            return 0.0

        arr = np.array(self._recent_results, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))

        if std_val == 0.0 or mean_val == 0.0:
            return 0.0

        return std_val / max(abs(mean_val), 1e-8)

    @property
    def volatility_factor(self) -> float:
        """Get the current volatility adjustment factor (0 to 1).

        1.0 = no adjustment (low volatility).
        Closer to 0.0 = heavy reduction (high volatility).
        """
        cv = self.current_volatility
        if cv <= 0.0:
            return 1.0
        return 1.0 / (1.0 + cv * self.sensitivity)

    # ── Core calculation ────────────────────────────────────

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the volatility-adjusted stake amount.

        Higher recent volatility = smaller stake.
        Returns 0.0 if ``ev < min_ev``.
        """
        _ = model_prob, decimal_odds
        if ev < self.min_ev or bankroll <= 0:
            return 0.0

        vf = self.volatility_factor
        effective_ratio = max(
            self.base_ratio * vf,
            self.min_ratio,
        )
        return round(max(bankroll, 0.0) * effective_ratio, 2)


# ══════════════════════════════════════════════════════════════════
#  Portfolio Staking
# ══════════════════════════════════════════════════════════════════


class PortfolioStaking(StakingStrategy):
    """Divide bankroll across multiple concurrent bets.

    Instead of allocating the full stake for each bet independently,
    this strategy divides the available bankroll across ``n`` concurrent
    bets, optionally weighted by EV, probability, or custom weights.

    This prevents over-exposure when multiple qualifying bets occur on
    the same day — a common failure mode of per-bet staking strategies.

    Allocation methods
    ------------------
    - ``"equal"`` — equal division across all concurrent bets (default)
    - ``"ev_weighted"`` — proportional to each bet's EV magnitude
    - ``"prob_weighted"`` — proportional to each bet's model probability
    - ``"kelly_weighted"`` — proportional to each bet's full Kelly fraction

    Parameters
    ----------
    total_concurrent_bets : int
        Maximum number of bets expected concurrently (default 5).
        The bankroll is divided by this number for equal allocation.
    allocation_method : str
        How to allocate across bets (``"equal"``, ``"ev_weighted"``,
        ``"prob_weighted"``, ``"kelly_weighted"``).  Default ``"equal"``.
    min_ev : float
        Minimum EV threshold — bets below this return 0.0 (default 0.0).

    Examples
    --------
    >>> strategy = PortfolioStaking(total_concurrent_bets=3)
    >>> strategy.calculate_stake(0.60, 2.00, 1000, ev=0.20)
    333.33

    >>> strategy = PortfolioStaking(
    ...     total_concurrent_bets=3, allocation_method=\"ev_weighted\",
    ... )
    """

    def __init__(
        self,
        total_concurrent_bets: int = 5,
        allocation_method: str = "equal",
        min_ev: float = 0.0,
    ) -> None:
        if total_concurrent_bets < 1:
            raise ValueError(
                f"total_concurrent_bets must be >= 1, got {total_concurrent_bets}"
            )
        valid_methods = {"equal", "ev_weighted", "prob_weighted", "kelly_weighted"}
        if allocation_method not in valid_methods:
            raise ValueError(
                f"allocation_method must be one of {valid_methods}, "
                f"got {allocation_method!r}"
            )
        super().__init__(
            total_concurrent_bets=total_concurrent_bets,
            allocation_method=allocation_method,
            min_ev=min_ev,
        )
        self.total_concurrent_bets = total_concurrent_bets
        self.allocation_method = allocation_method
        self.min_ev = min_ev

        # Track concurrent bet weights for weighted allocation methods.
        # In practice, the pipeline calls set_concurrent_bets() before
        # calculating stakes for a batch of simultaneous opportunities.
        self._concurrent_weights: list[dict[str, float]] = []

    # ── Public API ──────────────────────────────────────────

    def set_concurrent_bets(
        self,
        bets: list[dict[str, float]],
    ) -> None:
        """Set the concurrent bet pool for weighted allocation.

        Each bet dict requires keys matching the allocation method:
        - ``"ev"`` for ``ev_weighted``
        - ``"prob"`` or ``"model_prob"`` for ``prob_weighted``
        - ``"kelly_fraction"`` or ``"kelly"`` for ``kelly_weighted``

        For ``"equal"`` method, this call is optional — only the count
        of concurrent bets matters (``total_concurrent_bets``).

        Parameters
        ----------
        bets : list[dict[str, float]]
            List of concurrent bet metadata dicts.
        """
        self._concurrent_weights = list(bets)
        logger.debug(
            "PortfolioStaking: set %d concurrent bets (method=%s)",
            len(bets), self.allocation_method,
        )

    def clear_concurrent_bets(self) -> None:
        """Clear the concurrent bet pool."""
        self._concurrent_weights.clear()

    def _compute_stake_for_bet(
        self,
        bankroll: float,
        index: int,
        total_n: int,
    ) -> float:
        """Compute the stake for a single bet within the portfolio."""
        if total_n <= 0:
            return 0.0

        if self.allocation_method == "equal":
            return round(bankroll / total_n, 2)

        # Weighted methods require concurrent bet metadata
        if not self._concurrent_weights:
            logger.warning(
                "PortfolioStaking: no concurrent bets set for '%s' "
                "allocation — falling back to equal division",
                self.allocation_method,
            )
            return round(bankroll / total_n, 2)

        # Build weight array
        n_actual = len(self._concurrent_weights)
        weights = np.zeros(n_actual, dtype=np.float64)

        for i, bet in enumerate(self._concurrent_weights):
            if i >= total_n:
                break
            if self.allocation_method == "ev_weighted":
                w = abs(bet.get("ev", 0.0))
            elif self.allocation_method == "prob_weighted":
                w = bet.get("prob", bet.get("model_prob", 0.0))
            elif self.allocation_method == "kelly_weighted":
                w = abs(bet.get("kelly_fraction", bet.get("kelly", 0.0)))
            else:
                w = 1.0
            weights[i] = max(w, 0.0)

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return round(bankroll / max(total_n, 1), 2)

        if index < 0 or index >= n_actual:
            return 0.0

        return round(bankroll * (weights[index] / weight_sum), 2)

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the portfolio-divided stake for this bet.

        If ``concurrent_bets`` have been set (via ``set_concurrent_bets()``),
        the bankroll is divided among them using the configured allocation
        method.  Otherwise, it divides by ``total_concurrent_bets`` equally.

        Returns 0.0 if ``ev < min_ev``.
        """
        _ = decimal_odds
        if ev < self.min_ev or bankroll <= 0:
            return 0.0

        # Determine total number of concurrent bets
        if self._concurrent_weights:
            total_n = max(len(self._concurrent_weights), 1)
        else:
            total_n = max(self.total_concurrent_bets, 1)

        # Find this bet's index in the concurrent pool
        if self._concurrent_weights and self.allocation_method != "equal":
            # We need to identify which concurrent bet this is.
            # Use model_prob + ev as a fingerprint to find the match.
            idx = self._find_bet_index(model_prob, ev)
        else:
            idx = 0  # equal allocation — index doesn't matter

        return self._compute_stake_for_bet(bankroll, idx, total_n)

    def _find_bet_index(
        self,
        model_prob: float,
        ev: float,
    ) -> int:
        """Find this bet's index in the concurrent pool by matching params."""
        if not self._concurrent_weights:
            return 0

        # Try exact match first
        for i, bet in enumerate(self._concurrent_weights):
            b_prob = bet.get("prob", bet.get("model_prob", None))
            b_ev = bet.get("ev", None)
            if b_prob is not None and b_ev is not None:
                if abs(b_prob - model_prob) < 1e-6 and abs(b_ev - ev) < 1e-6:
                    return i

        # Fallback: closest match by combined score
        best_idx = 0
        best_score = float("inf")
        for i, bet in enumerate(self._concurrent_weights):
            b_prob = bet.get("prob", bet.get("model_prob", model_prob))
            b_ev = bet.get("ev", ev)
            score = abs(b_prob - model_prob) + abs(b_ev - ev)
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx


# ══════════════════════════════════════════════════════════════════
#  Kelly Staking
# ══════════════════════════════════════════════════════════════════


class KellyStaking(StakingStrategy):
    """Stake the full Kelly Criterion fraction of bankroll.

    Uses the formula:  f* = (p × odds − 1) / (odds − 1)

    This maximises long-term logarithmic growth but is aggressive —
    recommended stakes can exceed 50% of bankroll for large edges.
    Consider ``FractionalKellyStaking`` for a more conservative approach.

    Negative-EV bets return a stake of **0.0** (no bet).

    Examples
    --------
    >>> strategy = KellyStaking()
    >>> strategy.calculate_stake(0.60, 2.00, 1000)
    200.0
    >>> strategy.calculate_stake(0.30, 3.00, 1000)
    0.0
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the full Kelly stake amount, or 0.0 for negative-EV bets."""
        _ = ev  # Kelly recomputes EV internally
        result = calculate_kelly(
            model_prob, decimal_odds, bankroll=bankroll, round_to=2,
        )
        return result.get("stake_amount", 0.0) or 0.0


# ══════════════════════════════════════════════════════════════════
#  Fractional Kelly Staking
# ══════════════════════════════════════════════════════════════════


class FractionalKellyStaking(StakingStrategy):
    """Stake a fraction of the full Kelly amount for reduced variance.

    Parameters
    ----------
    fraction : float
        Fraction of full Kelly to use (default 0.25 = 25% Kelly).
        Common values: 0.10, 0.25, 0.50, 0.75, 1.0.

    Examples
    --------
    >>> strategy = FractionalKellyStaking(fraction=0.25)
    >>> strategy.calculate_stake(0.60, 2.00, 1000)
    50.0
    >>> strategy.calculate_stake(0.30, 3.00, 1000)
    0.0
    """

    def __init__(self, fraction: float = 0.25) -> None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        super().__init__(fraction=fraction)
        self.fraction = fraction

    def calculate_stake(
        self,
        model_prob: float,
        decimal_odds: float,
        bankroll: float,
        ev: float = 0.0,
    ) -> float:
        """Return the fractional Kelly stake amount, or 0.0 for negative-EV."""
        _ = ev  # Fractional Kelly recomputes EV internally
        result = calculate_fractional_kelly(
            model_prob, decimal_odds,
            fraction=self.fraction,
            bankroll=bankroll,
            round_to=2,
        )
        return result.get("stake_amount", 0.0) or 0.0


# ══════════════════════════════════════════════════════════════════
#  Strategy registry / factory
# ══════════════════════════════════════════════════════════════════


class StakingFactory:
    """Factory for creating stake sizing strategies by name.

    Usage
    -----
    ::

        from src.betting.staking import StakingFactory

        # Create with default parameters
        strategy = StakingFactory.create("kelly")
        stake = strategy.calculate_stake(0.60, 2.00, 1000)

        # Create with custom parameters
        strategy = StakingFactory.create(
            "fractional_kelly", fraction=0.25,
        )
        strategy = StakingFactory.create(
            "flat", stake_per_bet=50.0,
        )
        strategy = StakingFactory.create(
            "fixed_ratio", ratio=0.02,
        )
        strategy = StakingFactory.create(
            "variable_ratio", base_ratio=0.02,
        )
        strategy = StakingFactory.create(
            "volatility", base_ratio=0.02, window=15,
        )
        strategy = StakingFactory.create(
            "portfolio", total_concurrent_bets=4,
        )
    """

    _REGISTRY: dict[str, type[StakingStrategy]] = {
        "flat": FlatStaking,
        "percentage": PercentageStaking,
        "fixed_ratio": FixedRatioStaking,
        "variable_ratio": VariableRatioStaking,
        "volatility": VolatilityStaking,
        "portfolio": PortfolioStaking,
        "kelly": KellyStaking,
        "fractional_kelly": FractionalKellyStaking,
    }

    @classmethod
    def create(
        cls,
        strategy_name: str,
        **kwargs: Any,
    ) -> StakingStrategy:
        """Create a staking strategy by name.

        Parameters
        ----------
        strategy_name : str
            One of: ``flat``, ``percentage``, ``fixed_ratio``,
            ``variable_ratio``, ``volatility``, ``portfolio``,
            ``kelly``, ``fractional_kelly``.
        **kwargs
            Passed to the strategy constructor (e.g. ``stake_per_bet``,
            ``stake_pct``, ``ratio``, ``base_ratio``, ``fraction``,
            ``total_concurrent_bets``).

        Returns
        -------
        StakingStrategy
            The instantiated strategy.

        Raises
        ------
        ValueError
            If the strategy name is unknown.
        """
        strategy_cls = cls._REGISTRY.get(strategy_name)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown staking strategy: {strategy_name!r}. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        logger.info("Creating staking strategy: %s(%s)", strategy_name, kwargs)
        return strategy_cls(**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all available strategy names."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def register(cls, name: str, strategy_cls: type[StakingStrategy]) -> None:
        """Register a custom staking strategy.

        Parameters
        ----------
        name : str
            Unique type identifier.
        strategy_cls : type[StakingStrategy]
            Strategy class that implements ``calculate_stake()``.
        """
        if name in cls._REGISTRY:
            logger.warning("Overwriting staking strategy '%s'", name)
        cls._REGISTRY[name] = strategy_cls
        logger.info(
            "Registered staking strategy: %s (%s)",
            name, strategy_cls.__name__,
        )
