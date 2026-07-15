"""
Expected Value (EV) Calculation — identify positive-expected-value betting opportunities.

The EV formula answers: *"If I place this bet many times, how much would I
expect to win (or lose) per unit staked on average?"*

Formula
-------
    EV = (model_probability × decimal_odds) − 1

    Where:
    - model_probability — your estimated probability of the outcome (0 to 1)
    - decimal_odds     — the bookmaker's decimal odds (≥ 1.0)

    EV > 0  →  profitable in the long run (value bet)
    EV = 0  →  break-even
    EV < 0  →  unprofitable in the long run

    EV can also be expressed as a percentage:
    EV% = EV × 100

Markets supported
------------------
- **1X2** (Home Win / Draw / Away Win)
- **BTTS** (Both Teams To Score / No)
- **Over/Under** (e.g. Over 2.5 Goals, Under 2.5 Goals)
- **Double Chance** (1X, 12, X2)
- **Any binary market** with a single probability and odds

Usage
-----
::

    from src.betting.ev import calculate_ev

    # Single bet
    result = calculate_ev(model_prob=0.45, decimal_odds=2.50)
    print(result["ev"])        # 0.125
    print(result["ev_pct"])    # 12.5
    print(result["is_value"])  # True

    # Batch processing
    results = calculate_ev_batch(
        probs=[0.45, 0.30, 0.25],
        odds=[2.50, 3.40, 4.00],
        labels=["Home", "Draw", "Away"],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
_MIN_ODDS = 1.0          # Minimum valid decimal odds
_MAX_PROB = 1.0          # Maximum valid probability
_MIN_PROB = 0.0          # Minimum valid probability


# ═══════════════════════════════════════════════════════════
#  Core EV function
# ═══════════════════════════════════════════════════════════


def calculate_ev(
    model_prob: float,
    decimal_odds: float,
    *,
    round_to: int | None = 6,
) -> dict[str, Any]:
    """Calculate Expected Value for a single betting outcome.

    Parameters
    ----------
    model_prob : float
        Your estimated probability of the outcome (0.0 to 1.0).
        Values outside [0, 1] are clipped.
    decimal_odds : float
        Bookmaker's decimal odds. Must be > 1.0.
    round_to : int | None
        Number of decimal places for rounding (default 6).
        Set to ``None`` to disable rounding.

    Returns
    -------
    dict[str, Any]
        ``{"ev", "ev_pct", "is_value", "model_prob", "decimal_odds",
        "fair_odds", "edge_vs_market"}``

    Examples
    --------
    >>> r = calculate_ev(0.45, 2.50)
    >>> r["ev"]
    0.125
    >>> r["ev_pct"]
    12.5
    >>> r["is_value"]
    True

    >>> r = calculate_ev(0.30, 3.00)
    >>> r["ev"]
    -0.1
    >>> r["is_value"]
    False
    """
    # ── Validate & clip inputs ──
    prob = float(np.clip(model_prob, _MIN_PROB, _MAX_PROB))
    odds = float(decimal_odds)

    if odds <= _MIN_ODDS:
        logger.warning(
            "decimal_odds must be > %.1f, got %.4f — returning zero EV",
            _MIN_ODDS, odds,
        )
        return _zero_ev(prob, odds, error="odds must be > 1.0")

    # ── Compute EV ──
    ev = (prob * odds) - 1.0

    # ── Edge analysis ──
    # Fair odds = 1 / prob  (the odds that would make this a break-even bet)
    fair_odds = 1.0 / max(prob, 1e-10)

    # Edge vs market = model_prob - (1 / odds)  (how much our prob beats the market)
    implied_market_prob = 1.0 / odds
    edge_vs_market = prob - implied_market_prob

    # ── Round ──
    if round_to is not None:
        ev = round(ev, round_to)
        fair_odds = round(fair_odds, round_to)
        edge_vs_market = round(edge_vs_market, round_to)

    ev_pct = round(ev * 100, max(round_to - 2, 2) if round_to else 4)

    return {
        "ev": ev,
        "ev_pct": ev_pct,
        "is_value": ev > 0,
        "model_prob": round(prob, round_to) if round_to else prob,
        "decimal_odds": round(odds, round_to) if round_to else odds,
        "implied_market_prob": round(implied_market_prob, round_to) if round_to else implied_market_prob,
        "fair_odds": fair_odds,
        "edge_vs_market": edge_vs_market,
    }


def _zero_ev(prob: float, odds: float, error: str = "") -> dict[str, Any]:
    """Return a zero-EV result for invalid inputs."""
    return {
        "ev": 0.0,
        "ev_pct": 0.0,
        "is_value": False,
        "model_prob": prob,
        "decimal_odds": odds,
        "implied_market_prob": 1.0 / max(odds, 1.0),
        "fair_odds": 0.0,
        "edge_vs_market": 0.0,
        "error": error,
    }


# ═══════════════════════════════════════════════════════════
#  Market-specific helpers
# ═══════════════════════════════════════════════════════════


def calculate_ev_1x2(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Calculate EV for all three 1X2 outcomes.

    Parameters
    ----------
    home_prob, draw_prob, away_prob : float
        Model probabilities for each outcome (must sum to ~1.0).
    home_odds, draw_odds, away_odds : float
        Bookmaker decimal odds for each outcome.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{"home": {...}, "draw": {...}, "away": {...}}``
        each containing the full ``calculate_ev()`` result.

    Examples
    --------
    >>> evs = calculate_ev_1x2(0.50, 0.30, 0.20, 2.10, 3.40, 4.00)
    >>> evs["home"]["ev_pct"]
    5.0
    """
    return {
        "home": calculate_ev(home_prob, home_odds, **kwargs),
        "draw": calculate_ev(draw_prob, draw_odds, **kwargs),
        "away": calculate_ev(away_prob, away_odds, **kwargs),
    }


def calculate_ev_binary(
    model_prob: float,
    decimal_odds: float,
    label: str = "outcome",
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate EV for a binary market (BTTS, Over/Under, etc.).

    Parameters
    ----------
    model_prob : float
        Estimated probability of the binary event occurring.
    decimal_odds : float
        Bookmaker decimal odds for the "Yes" side.
    label : str
        Human-readable label for the outcome (used in logging).

    Returns
    -------
    dict[str, Any]
        Full ``calculate_ev()`` result with ``label`` key added.
    """
    result = calculate_ev(model_prob, decimal_odds, **kwargs)
    result["label"] = label
    return result


def calculate_ev_batch(
    probs: list[float] | np.ndarray,
    odds: list[float] | np.ndarray,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Calculate EV for multiple outcomes in batch.

    Parameters
    ----------
    probs : list[float] or np.ndarray
        Model probabilities for each outcome.
    odds : list[float] or np.ndarray
        Bookmaker decimal odds for each outcome.
    labels : list[str], optional
        Optional outcome labels (e.g. ``["Home", "Draw", "Away"]``).

    Returns
    -------
    list[dict[str, Any]]
        List of ``calculate_ev()`` results, one per outcome.

    Examples
    --------
    >>> results = calculate_ev_batch(
    ...     probs=[0.50, 0.30, 0.20],
    ...     odds=[2.10, 3.40, 4.00],
    ...     labels=["H", "D", "A"],
    ... )
    >>> [r["is_value"] for r in results]
    [True, False, False]
    """
    probs_arr = np.asarray(probs, dtype=np.float64).ravel()
    odds_arr = np.asarray(odds, dtype=np.float64).ravel()

    n = min(len(probs_arr), len(odds_arr))
    if labels is None:
        labels = [f"outcome_{i}" for i in range(n)]

    results = []
    for i in range(n):
        result = calculate_ev(float(probs_arr[i]), float(odds_arr[i]), **kwargs)
        result["label"] = labels[i] if i < len(labels) else f"outcome_{i}"
        result["index"] = i
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════
#  Portfolio helpers
# ═══════════════════════════════════════════════════════════


def filter_value_bets(
    results: list[dict[str, Any]] | dict[str, dict[str, Any]],
    min_ev: float = 0.0,
) -> list[dict[str, Any]]:
    """Filter EV results to only positive-EV opportunities.

    Parameters
    ----------
    results : list[dict] or dict of dicts
        EV results from any ``calculate_ev_*`` function.
    min_ev : float
        Minimum EV threshold (default 0.0).

    Returns
    -------
    list[dict[str, Any]]
        Sorted by descending EV.
    """
    # Flatten dict-of-dicts (e.g. from calculate_ev_1x2)
    if isinstance(results, dict):
        flat: list[dict[str, Any]] = []
        for key, value in results.items():
            if isinstance(value, dict) and "ev" in value:
                value["outcome_key"] = key
                flat.append(value)
        results = flat

    filtered = [r for r in results if r.get("ev", -1) > min_ev]
    filtered.sort(key=lambda x: x.get("ev", -1), reverse=True)
    return filtered


def best_value_bet(
    results: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the single best value bet (highest EV) from a set of results.

    Parameters
    ----------
    results : list[dict] or dict of dicts
        EV results.

    Returns
    -------
    dict | None
        The result with the highest EV, or ``None`` if no positive-EV bets.
    """
    filtered = filter_value_bets(results, min_ev=0.0)
    return filtered[0] if filtered else None
