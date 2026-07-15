"""
Kelly Criterion — optimal stake sizing for positive-EV betting opportunities.

The Kelly Criterion answers: *"Given my edge and the odds, what fraction of my
bankroll should I stake to maximise long-term logarithmic growth?"*

Formula
-------
    f* = (b × p − q) / b

    Where:
    - p — your estimated probability of the outcome (0 to 1)
    - q — probability of the outcome NOT happening (1 − p)
    - b — decimal odds minus 1 (the net odds received on the bet)

    Simplified:
    f* = (p × decimal_odds − 1) / (decimal_odds − 1)

    f* = 0    →  no edge, don't bet
    f* > 0    →  optimal fraction of bankroll to wager

Fractional Kelly
----------------
    Kelly is aggressive — it can recommend stakes > 50% of bankroll.
    Fractional Kelly uses a fraction of full Kelly for lower variance:

        f_k = f* × k

    Common k values:
    - 0.10 (10% Kelly) — very conservative, minimal variance
    - 0.25 (25% Kelly) — conservative, recommended for most bettors
    - 0.50 (50% Kelly) — moderately aggressive
    - 0.75 (75% Kelly) — aggressive
    - 1.00 (Full Kelly) — maximum growth, maximum variance

Usage
-----
::

    from src.betting.kelly import calculate_kelly, calculate_fractional_kelly

    # Full Kelly
    result = calculate_kelly(model_prob=0.60, decimal_odds=2.00, bankroll=1000)
    print(result["kelly_fraction"])   # 0.2  (20% of bankroll)
    print(result["stake_amount"])     # 200.0

    # Fractional Kelly (25%)
    result = calculate_fractional_kelly(0.60, 2.00, fraction=0.25, bankroll=1000)
    print(result["kelly_fraction"])   # 0.05  (5% of bankroll)
    print(result["stake_amount"])     # 50.0
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
_MIN_ODDS = 1.0          # Minimum valid decimal odds
_MAX_PROB = 1.0          # Maximum valid probability
_MIN_PROB = 0.0          # Minimum valid probability
_DEFAULT_FRACTION = 0.25  # Default fractional Kelly multiplier


# ═══════════════════════════════════════════════════════════
#  Core Kelly function
# ═══════════════════════════════════════════════════════════


def calculate_kelly(
    model_prob: float,
    decimal_odds: float,
    bankroll: float | None = None,
    *,
    round_to: int | None = 6,
) -> dict[str, Any]:
    """Calculate the full Kelly Criterion optimal stake fraction.

    Parameters
    ----------
    model_prob : float
        Your estimated probability of the outcome (0.0 to 1.0).
        Values outside [0, 1] are clipped.
    decimal_odds : float
        Bookmaker's decimal odds. Must be > 1.0.
    bankroll : float, optional
        Current bankroll balance. If provided, the absolute stake
        amount is also returned.
    round_to : int | None
        Number of decimal places for rounding (default 6).
        Set to ``None`` to disable rounding.

    Returns
    -------
    dict[str, Any]
        ``{
            "kelly_fraction",       # fraction of bankroll (0 to 1)
            "kelly_pct",            # fraction as percentage
            "stake_amount",         # absolute stake (if bankroll given)
            "is_positive",          # True if kelly_fraction > 0
            "model_prob",
            "decimal_odds",
            "ev",                   # expected value at these odds
            "net_odds"              # b = decimal_odds - 1
        }``

    Examples
    --------
    >>> r = calculate_kelly(0.60, 2.00)
    >>> r["kelly_fraction"]
    0.2
    >>> r["is_positive"]
    True

    >>> r = calculate_kelly(0.30, 3.00)
    >>> r["kelly_fraction"]
    0.0
    >>> r["is_positive"]
    False
    """
    # ── Validate & clip inputs ──
    prob = float(np.clip(model_prob, _MIN_PROB, _MAX_PROB))
    odds = float(decimal_odds)

    if odds <= _MIN_ODDS:
        logger.warning(
            "decimal_odds must be > %.1f, got %.4f — returning zero Kelly",
            _MIN_ODDS, odds,
        )
        return _zero_kelly(prob, odds, bankroll, error="odds must be > 1.0")

    # ── Compute EV first (negative EV → no bet) ──
    ev = (prob * odds) - 1.0

    if ev <= 0.0:
        return _zero_kelly(prob, odds, bankroll, ev=ev, error="no positive edge")

    # ── Full Kelly ──
    # f* = (p × odds − 1) / (odds − 1)
    net_odds = odds - 1.0  # b = decimal_odds - 1
    kelly_fraction = ev / net_odds  # = (p*odds - 1) / (odds - 1)
    kelly_fraction = max(min(kelly_fraction, 1.0), 0.0)  # clamp to [0, 1]

    # ── Build result ──
    return _build_kelly_result(
        kelly_fraction=kelly_fraction,
        prob=prob,
        odds=odds,
        ev=ev,
        net_odds=net_odds,
        bankroll=bankroll,
        round_to=round_to,
    )


# ═══════════════════════════════════════════════════════════
#  Fractional Kelly
# ═══════════════════════════════════════════════════════════


def calculate_fractional_kelly(
    model_prob: float,
    decimal_odds: float,
    fraction: float = _DEFAULT_FRACTION,
    bankroll: float | None = None,
    *,
    round_to: int | None = 6,
) -> dict[str, Any]:
    """Calculate a fractional Kelly stake.

    Fractional Kelly reduces variance by staking only a fraction of
    the full Kelly recommendation.

    Parameters
    ----------
    model_prob : float
        Estimated probability of the outcome (0 to 1).
    decimal_odds : float
        Bookmaker decimal odds.
    fraction : float
        Fraction of full Kelly to use (default 0.25 = 25% Kelly).
        Common values: 0.10, 0.25, 0.50, 0.75, 1.0.
    bankroll : float, optional
        Current bankroll balance. If provided, the absolute stake
        amount is also returned.
    round_to : int | None
        Number of decimal places for rounding (default 6).

    Returns
    -------
    dict[str, Any]
        Same structure as ``calculate_kelly()`` but with additional
        ``fraction`` and ``full_kelly_fraction`` keys.

    Examples
    --------
    >>> r = calculate_fractional_kelly(0.60, 2.00, fraction=0.25)
    >>> r["kelly_fraction"]
    0.05
    >>> r["full_kelly_fraction"]
    0.2
    """
    # Get full Kelly first
    full = calculate_kelly(
        model_prob, decimal_odds,
        bankroll=None,  # don't compute stake yet
        round_to=None,  # keep full precision for now
    )

    if not full["is_positive"]:
        return _zero_kelly(
            prob=full["model_prob"],
            odds=full["decimal_odds"],
            bankroll=bankroll,
            ev=full["ev"],
            error="no positive edge",
        )

    kelly_fraction = max(full["kelly_fraction"] * fraction, 0.0)
    kelly_fraction = min(kelly_fraction, 1.0)

    result = _build_kelly_result(
        kelly_fraction=kelly_fraction,
        prob=full["model_prob"],
        odds=full["decimal_odds"],
        ev=full["ev"],
        net_odds=full["net_odds"],
        bankroll=bankroll,
        round_to=round_to,
    )

    # Add fractional Kelly metadata
    result["fraction"] = round(fraction, 4) if round_to is not None else fraction
    result["full_kelly_fraction"] = round(full["kelly_fraction"], round_to) if round_to else full["kelly_fraction"]
    result["full_kelly_pct"] = round(full["kelly_fraction"] * 100, max(round_to - 2, 2) if round_to else 4)

    return result


# ═══════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════


def _build_kelly_result(
    kelly_fraction: float,
    prob: float,
    odds: float,
    ev: float,
    net_odds: float,
    bankroll: float | None,
    round_to: int | None,
) -> dict[str, Any]:
    """Build the standard Kelly result dict with rounding."""
    if round_to is not None:
        kelly_fraction = round(kelly_fraction, round_to)
        net_odds = round(net_odds, round_to)
        ev = round(ev, round_to)

    kelly_pct = round(kelly_fraction * 100, max(round_to - 2, 2) if round_to else 4)

    # Stake amount
    stake_amount = None
    if bankroll is not None and bankroll > 0:
        stake_amount = round(bankroll * kelly_fraction, 2)

    return {
        "kelly_fraction": kelly_fraction,
        "kelly_pct": kelly_pct,
        "stake_amount": stake_amount,
        "is_positive": kelly_fraction > 0,
        "model_prob": round(prob, round_to) if round_to else prob,
        "decimal_odds": round(odds, round_to) if round_to else odds,
        "ev": ev,
        "net_odds": net_odds,
    }


def _zero_kelly(
    prob: float,
    odds: float,
    bankroll: float | None = None,
    ev: float = 0.0,
    error: str = "",
) -> dict[str, Any]:
    """Return a zero-Kelly result for invalid inputs."""
    result = {
        "kelly_fraction": 0.0,
        "kelly_pct": 0.0,
        "stake_amount": 0.0 if bankroll is not None else None,
        "is_positive": False,
        "model_prob": prob,
        "decimal_odds": odds,
        "ev": round(ev, 6) if ev != 0.0 else 0.0,
        "net_odds": max(odds - 1.0, 0.0),
        "error": error,
    }
    return result


# ═══════════════════════════════════════════════════════════
#  Convenience helpers
# ═══════════════════════════════════════════════════════════


def kelly_fraction_only(
    model_prob: float,
    decimal_odds: float,
) -> float:
    """Compute only the full Kelly fraction (no dict overhead).

    A lightweight version of ``calculate_kelly()`` for when you just
    need the raw fraction value.

    Parameters
    ----------
    model_prob : float
        Estimated probability of the outcome (0 to 1).
    decimal_odds : float
        Bookmaker decimal odds.

    Returns
    -------
    float
        Kelly fraction (0 to 1). 0.0 if no edge.

    Examples
    --------
    >>> kelly_fraction_only(0.60, 2.00)
    0.2
    >>> kelly_fraction_only(0.30, 3.00)
    0.0
    """
    prob = float(np.clip(model_prob, _MIN_PROB, _MAX_PROB))
    odds = float(decimal_odds)

    if odds <= _MIN_ODDS or prob <= 0.0:
        return 0.0

    ev = (prob * odds) - 1.0
    if ev <= 0.0:
        return 0.0

    net_odds = odds - 1.0
    return max(min(ev / net_odds, 1.0), 0.0)


def kelly_stake_amount(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = _DEFAULT_FRACTION,
) -> float:
    """Compute the absolute stake amount in one call.

    A convenience function that combines probability, odds, and
    bankroll into a single currency amount.

    Parameters
    ----------
    model_prob : float
        Estimated probability of the outcome (0 to 1).
    decimal_odds : float
        Bookmaker decimal odds.
    bankroll : float
        Current bankroll balance.
    fraction : float
        Fractional Kelly multiplier (default 0.25 for 25% Kelly).

    Returns
    -------
    float
        Absolute stake amount in currency units.

    Examples
    --------
    >>> kelly_stake_amount(0.60, 2.00, 1000, fraction=0.25)
    50.0
    >>> kelly_stake_amount(0.30, 3.00, 1000)
    0.0
    """
    if bankroll <= 0:
        return 0.0

    result = calculate_fractional_kelly(
        model_prob, decimal_odds,
        fraction=fraction,
        bankroll=bankroll,
        round_to=2,
    )
    return result.get("stake_amount", 0.0) or 0.0


def recommended_fraction(
    model_prob: float,
    decimal_odds: float,
    risk_profile: str = "conservative",
) -> dict[str, Any]:
    """Get Kelly recommendations for common risk profiles.

    Parameters
    ----------
    model_prob : float
        Estimated probability of the outcome (0 to 1).
    decimal_odds : float
        Bookmaker decimal odds.
    risk_profile : str
        One of ``"conservative"`` (10%), ``"moderate"`` (25%),
        ``"balanced"`` (50%), ``"aggressive"`` (75%),
        ``"max"`` (100%), or ``"all"`` to return all profiles.

    Returns
    -------
    dict[str, Any]
        If a single profile is requested, a single Kelly result dict.
        If ``risk_profile="all"``, a dict mapping profile name to
        result dict.

    Raises
    ------
    ValueError
        If ``risk_profile`` is not a recognized profile name.

    Examples
    --------
    >>> r = recommended_fraction(0.60, 2.00, "conservative")
    >>> r["kelly_pct"]
    2.0

    >>> all_r = recommended_fraction(0.60, 2.00, "all")
    >>> all_r["moderate"]["kelly_pct"]
    5.0
    """
    profiles = {
        "conservative": 0.10,
        "moderate": 0.25,
        "balanced": 0.50,
        "aggressive": 0.75,
        "max": 1.00,
    }

    if risk_profile == "all":
        return {
            name: calculate_fractional_kelly(model_prob, decimal_odds, fraction=f)
            for name, f in profiles.items()
        }

    fraction = profiles.get(risk_profile)
    if fraction is None:
        raise ValueError(
            f"Unknown risk_profile: {risk_profile!r}. "
            f"Choose from: {list(profiles.keys())} or 'all'."
        )

    return calculate_fractional_kelly(model_prob, decimal_odds, fraction=fraction)
