"""
Bet Filtering — gate bet opportunities against configurable criteria.

The ``BetFilter`` class accepts a list of bet opportunity dicts and returns
only those that pass all configured filters, with detailed rejection reasons
for transparency.

Usage
-----
::

    from src.betting.filtering import BetFilter

    bets = [
        {
            "match": "Arsenal vs Chelsea",
            "outcome": "Home Win",
            "market": "1X2",
            "model_prob": 0.52,
            "decimal_odds": 2.10,
            "ev": 0.092,
            "bankroll_pct": 0.03,
        },
        ...
    ]

    filter = BetFilter(min_ev=0.05, min_confidence=0.45)
    passed, rejected = filter.filter_bets(bets)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default filter thresholds
_DEFAULT_MIN_EV = 0.0
_DEFAULT_MIN_CONFIDENCE = 0.6
_DEFAULT_MIN_ODDS = 1.5
_DEFAULT_MAX_STAKE = 0.05
_DEFAULT_MARKETS = ("1X2", "BTTS", "Over2.5")


class BetFilter:
    """Configurable filter for bet opportunities.

    Parameters
    ----------
    min_ev : float
        Minimum expected value threshold. Bets with EV below this
        are rejected. Default 0.0 (reject only negative-EV bets).
    min_confidence : float
        Minimum model probability. Bets with probability below this
        are rejected. Default 0.3.
    min_odds : float
        Minimum decimal odds. Bets with odds below this are
        rejected. Default 1.5.
    max_stake : float
        Maximum stake as a fraction of bankroll (0 to 1).
        Bets requiring more than this % of bankroll are rejected.
        Default 0.05 (5%).
    markets : tuple[str, ...]
        Set of allowed market types. Bets with a market not in this
        set are rejected. Default ``(\"1X2\", \"BTTS\", \"Over2.5\")``.
    reject_on_missing : bool
        If True, reject bets that are missing required fields.
        If False, skip filters where data is missing. Default True.

    Examples
    --------
    >>> f = BetFilter(min_ev=0.05, min_confidence=0.45)
    >>> bets = [
    ...     {"outcome": "H", "market": "1X2", "model_prob": 0.52,
    ...      "decimal_odds": 2.10, "ev": 0.092, "bankroll_pct": 0.03},
    ...     {"outcome": "H", "market": "1X2", "model_prob": 0.38,
    ...      "decimal_odds": 3.00, "ev": 0.140, "bankroll_pct": 0.06},
    ... ]
    >>> passed, rejected = f.filter_bets(bets)
    >>> len(passed)
    1
    """

    def __init__(
        self,
        min_ev: float = _DEFAULT_MIN_EV,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        min_odds: float = _DEFAULT_MIN_ODDS,
        max_stake: float = _DEFAULT_MAX_STAKE,
        markets: tuple[str, ...] = _DEFAULT_MARKETS,
        reject_on_missing: bool = True,
    ) -> None:
        if min_odds < 1.0:
            raise ValueError(f"min_odds must be >= 1.0, got {min_odds}")
        if not 0 <= min_ev <= 1:
            raise ValueError(f"min_ev must be in [0, 1], got {min_ev}")
        if not 0 < min_confidence <= 1:
            raise ValueError(f"min_confidence must be in (0, 1], got {min_confidence}")
        if not 0 < max_stake <= 1:
            raise ValueError(f"max_stake must be in (0, 1], got {max_stake}")
        if not markets:
            raise ValueError("markets must not be empty")

        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.min_odds = min_odds
        self.max_stake = max_stake
        self.markets = set(markets)
        self.reject_on_missing = reject_on_missing

        self._rejection_counts: dict[str, int] = {}

    def filter_bets(
        self,
        bets: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Filter a list of bet opportunities.

        Parameters
        ----------
        bets : list[dict]
            Each dict should contain keys like ``model_prob``,
            ``decimal_odds``, ``ev``, ``bankroll_pct``, ``market``.
            Keys not required for a particular filter can be omitted
            (they will pass that filter, or be rejected if
            ``reject_on_missing=True``).

        Returns
        -------
        tuple[list[dict], list[dict]]
            ``(passed_bets, rejected_bets)`` — passed bets are in the
            original order. Rejected bets have an extra ``_rejection``
            key explaining why.
        """
        self._rejection_counts = {}

        passed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for bet in bets:
            reason = self._check_bet(bet)
            if reason:
                rejected.append({**bet, "_rejection": reason})
                self._rejection_counts[reason] = (
                    self._rejection_counts.get(reason, 0) + 1
                )
            else:
                passed.append(bet)

        # Log summary
        n_passed = len(passed)
        n_total = len(bets)
        n_rejected = n_total - n_passed
        if n_rejected > 0:
            logger.info(
                "BetFilter: %d/%d passed, %d rejected (%s)",
                n_passed, n_total, n_rejected,
                ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(self._rejection_counts.items())
                ),
            )

        return passed, rejected

    # ── Internal check dispatch ────────────────────────

    def _check_bet(self, bet: dict[str, Any]) -> str | None:
        """Run all filters on a single bet. Returns rejection reason or None."""
        # Chain of checks — return first failure
        reason = self._check_min_ev(bet)
        if reason:
            return reason

        reason = self._check_min_confidence(bet)
        if reason:
            return reason

        reason = self._check_min_odds(bet)
        if reason:
            return reason

        reason = self._check_max_stake(bet)
        if reason:
            return reason

        reason = self._check_market(bet)
        if reason:
            return reason

        return None  # passed all filters

    # ── Individual filter checks ──────────────────────

    def _check_min_ev(self, bet: dict[str, Any]) -> str | None:
        """Reject if EV is below min_ev."""
        ev = bet.get("ev")
        if ev is None:
            # Compute EV from prob and odds if available
            prob = bet.get("model_prob")
            odds = bet.get("decimal_odds")
            if prob is not None and odds is not None and odds > 1.0:
                ev = (prob * odds) - 1.0
        if ev is not None:
            if ev < self.min_ev:
                return f"EV too low: {ev:.4f} < {self.min_ev}"
            return None
        # No EV data available
        if self.reject_on_missing:
            return "EV data missing"
        return None

    def _check_min_confidence(self, bet: dict[str, Any]) -> str | None:
        """Reject if model probability is below min_confidence."""
        prob = bet.get("model_prob")
        if prob is not None:
            if prob < self.min_confidence:
                return (
                    f"Confidence too low: {prob:.3f} < {self.min_confidence}"
                )
            return None
        if self.reject_on_missing:
            return "Model probability missing"
        return None

    def _check_min_odds(self, bet: dict[str, Any]) -> str | None:
        """Reject if decimal odds are below min_odds."""
        odds = bet.get("decimal_odds")
        if odds is not None:
            if odds < self.min_odds:
                return f"Odds too low: {odds:.2f} < {self.min_odds}"
            return None
        if self.reject_on_missing:
            return "Decimal odds missing"
        return None

    def _check_max_stake(self, bet: dict[str, Any]) -> str | None:
        """Reject if stake % of bankroll exceeds max_stake."""
        stake_pct = bet.get("bankroll_pct", bet.get("stake_pct"))
        if stake_pct is not None:
            if stake_pct > self.max_stake:
                return (
                    f"Stake too high: {stake_pct:.1%} > {self.max_stake:.1%}"
                )
            return None
        # No stake data — can't check this filter
        return None

    def _check_market(self, bet: dict[str, Any]) -> str | None:
        """Reject if market type is not in allowed set."""
        market = bet.get("market", bet.get("market_type"))
        if market is not None:
            if market not in self.markets:
                return f"Market not allowed: {market!r}"
            return None
        if self.reject_on_missing:
            return "Market type missing"
        return None

    # ── Reporting ─────────────────────────────────────

    @property
    def rejection_summary(self) -> dict[str, int]:
        """Return a count of how many bets hit each rejection reason."""
        return dict(self._rejection_counts)

    def __repr__(self) -> str:
        return (
            f"<BetFilter min_ev={self.min_ev} "
            f"min_conf={self.min_confidence} "
            f"min_odds={self.min_odds} "
            f"max_stake={self.max_stake:.0%} "
            f"markets={sorted(self.markets)}>"
        )


# ═══════════════════════════════════════════════════════════
#  Convenience helpers
# ═══════════════════════════════════════════════════════════


def filter_passed_only(
    bets: list[dict[str, Any]],
    **filter_kwargs: Any,
) -> list[dict[str, Any]]:
    """Shortcut: return only passed bets (discard rejections).

    Parameters
    ----------
    bets : list[dict]
        Bet opportunities to filter.
    **filter_kwargs
        Passed to ``BetFilter`` constructor.

    Returns
    -------
    list[dict]
        Only the bets that passed all filters.
    """
    bf = BetFilter(**filter_kwargs)
    passed, _ = bf.filter_bets(bets)
    return passed


def count_passing(
    bets: list[dict[str, Any]],
    **filter_kwargs: Any,
) -> int:
    """Shortcut: count how many bets pass all filters.

    Parameters
    ----------
    bets : list[dict]
        Bet opportunities to evaluate.
    **filter_kwargs
        Passed to ``BetFilter`` constructor.

    Returns
    -------
    int
        Number of bets that pass all filters.
    """
    bf = BetFilter(**filter_kwargs)
    passed, _ = bf.filter_bets(bets)
    return len(passed)
