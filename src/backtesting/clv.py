"""
Closing Line Value (CLV) — measure of betting skill vs the market.

CLV measures how your bet price compares to the final (closing) market
price.  Consistently positive CLV is widely considered the single best
indicator of genuine betting skill (as opposed to luck).

Formula
-------
    CLV = (your_odds - closing_odds) / closing_odds

- **Positive CLV** → you got better odds than the market closed at
  (odds shortened after you bet — market moved toward your side).
- **CLV near zero** → your odds matched the market consensus.
- **Negative CLV** → you got worse odds than the market closed at
  (odds drifted after you bet — market moved against your side).

Markets supported
-----------------
- ``"1X2"`` — Home / Draw / Away win
- ``"BTTS"`` — Both Teams To Score (Yes / No)
- ``"Over/Under"`` — Over / Under a goal line (default 2.5)

Edge cases handled
------------------
- Missing (None / NaN) closing odds → CLV = 0.0
- Your odds <= 1.0 or closing odds <= 1.0 → CLV = 0.0
- Single outcome, all outcomes, or market-level CLV

Usage
-----
::

    from src.backtesting.clv import calculate_clv, calculate_market_clv

    # Single outcome
    clv = calculate_clv(your_odds=2.10, closing_odds=2.05)
    # → {"clv": 0.0244, "clv_pct": 2.44}

    # Entire 1X2 market
    result = calculate_market_clv(
        your_odds=[2.10, 3.40, 3.80],
        closing_odds=[2.05, 3.50, 3.60],
        market="1X2",
    )
    # → {"market": "1X2", "outcomes": [...], "avg_clv": 0.0187, ...}
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Core CLV calculation
# ═══════════════════════════════════════════════════════════


def calculate_clv(
    your_odds: float,
    closing_odds: float | None,
    market: str = "1X2",
) -> dict[str, Any]:
    """Calculate Closing Line Value for a single bet outcome.

    Parameters
    ----------
    your_odds : float
        Decimal odds at which you placed the bet.  Must be > 1.0.
    closing_odds : float | None
        Decimal odds at market close (kick-off).  ``None`` means
        closing odds are not available — CLV defaults to 0.0.
    market : str
        Market identifier (``"1X2"``, ``"BTTS"``, or ``"Over/Under"``).
        Used only for metadata in the result dict.

    Returns
    -------
    dict
        ``{clv, clv_pct, your_odds, closing_odds, market}``

        - ``clv`` — CLV as a float (e.g. ``0.0244`` = 2.44% better odds)
        - ``clv_pct`` — CLV as a percentage string (e.g. ``2.44%``)
        - ``positive`` — ``True`` if CLV > 0 (you beat the closing line)

    Examples
    --------
    >>> calculate_clv(2.10, 2.05, "1X2")
    {"clv": 0.0244, "clv_pct": "2.44%", "positive": True, ...}

    >>> calculate_clv(2.10, 2.20, "1X2")
    {"clv": -0.0455, "clv_pct": "-4.55%", "positive": False, ...}

    >>> calculate_clv(2.10, None, "1X2")
    {"clv": 0.0, "clv_pct": "0.00%", "positive": False, ...}
    """
    # Edge case: closing odds not available
    if closing_odds is None:
        logger.debug("CLV = 0.0 for %.4f odds — no closing odds available", your_odds)
        return _clv_result(0.0, your_odds, closing_odds, market)

    # Edge case: invalid odds
    if your_odds <= 1.0 or closing_odds <= 1.0:
        logger.debug(
            "CLV = 0.0 — invalid odds (your=%.4f, closing=%.4f)",
            your_odds, closing_odds,
        )
        return _clv_result(0.0, your_odds, closing_odds, market)

    # Edge case: NaN / infinite
    if not np.isfinite(your_odds) or not np.isfinite(closing_odds):
        logger.debug("CLV = 0.0 — non-finite odds")
        return _clv_result(0.0, your_odds, closing_odds, market)

    # Core formula: (your_odds - closing_odds) / closing_odds
    clv = (your_odds - closing_odds) / closing_odds

    return _clv_result(clv, your_odds, closing_odds, market)


def _clv_result(
    clv: float,
    your_odds: float,
    closing_odds: float | None,
    market: str,
) -> dict[str, Any]:
    """Build the standardised CLV result dict."""
    return {
        "clv": round(clv, 6),
        "clv_pct": f"{clv * 100:.2f}%",
        "positive": clv > 0,
        "your_odds": round(your_odds, 4) if np.isfinite(your_odds) else your_odds,
        "closing_odds": (
            round(closing_odds, 4)
            if closing_odds is not None and np.isfinite(closing_odds)
            else closing_odds
        ),
        "market": market,
    }


# ═══════════════════════════════════════════════════════════
#  Multi-outcome / market-level CLV
# ═══════════════════════════════════════════════════════════


_OUTCOME_LABELS: dict[str, list[str]] = {
    "1X2": ["Home Win", "Draw", "Away Win"],
    "BTTS": ["Yes", "No"],
    "Over/Under": ["Over 2.5", "Under 2.5"],
}


def calculate_market_clv(
    your_odds: list[float],
    closing_odds: list[float | None],
    market: str = "1X2",
) -> dict[str, Any]:
    """Calculate CLV for all outcomes in a market.

    Parameters
    ----------
    your_odds : list[float]
        Decimal odds at which bets were placed, one per outcome.
    closing_odds : list[float | None]
        Closing decimal odds, one per outcome.  ``None`` = unavailable.
    market : str
        Market type (``"1X2"``, ``"BTTS"``, ``"Over/Under"``).
        Determines the outcome labels in the result.

    Returns
    -------
    dict
        ``{market, outcomes: [{label, clv, clv_pct, positive, ...}],
        avg_clv, max_clv, min_clv, n_positive, n_negative}``

    Examples
    --------
    >>> result = calculate_market_clv(
    ...     [2.10, 3.40, 3.80],
    ...     [2.05, 3.50, 3.60],
    ...     "1X2",
    ... )
    >>> result["avg_clv"]
    0.0187
    """
    if not your_odds:
        return {
            "market": market,
            "outcomes": [],
            "avg_clv": 0.0,
            "max_clv": 0.0,
            "min_clv": 0.0,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
        }

    labels = _OUTCOME_LABELS.get(market, [f"Outcome {i+1}" for i in range(len(your_odds))])

    # Pad labels if fewer than outcomes
    while len(labels) < len(your_odds):
        labels.append(f"Outcome {len(labels) + 1}")

    outcomes: list[dict[str, Any]] = []
    clv_values: list[float] = []

    for i in range(len(your_odds)):
        odds = your_odds[i]
        close = closing_odds[i] if i < len(closing_odds) else None
        label = labels[i] if i < len(labels) else f"Outcome {i + 1}"

        result = calculate_clv(odds, close, market)
        result["label"] = label
        outcomes.append(result)
        clv_values.append(result["clv"])

    # Aggregate
    clv_arr = np.array(clv_values)
    n_positive = int(np.sum(clv_arr > 0))
    n_negative = int(np.sum(clv_arr < 0))
    n_zero = int(np.sum(clv_arr == 0))

    return {
        "market": market,
        "outcomes": outcomes,
        "avg_clv": round(float(np.mean(clv_arr)), 6),
        "max_clv": round(float(np.max(clv_arr)), 6),
        "min_clv": round(float(np.min(clv_arr)), 6),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
    }


# ═══════════════════════════════════════════════════════════
#  Batch CLV (multiple bets / multiple markets)
# ═══════════════════════════════════════════════════════════


def calculate_batch_clv(
    bets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate CLV for a batch of individual bets.

    Each bet dict should contain:
        - ``your_odds`` (float) — decimal odds at placement
        - ``closing_odds`` (float | None) — closing decimal odds
        - ``market`` (str, optional) — ``"1X2"``, ``"BTTS"``, ``"Over/Under"``
        - ``label`` (str, optional) — human-readable label

    Parameters
    ----------
    bets : list[dict]
        List of bet dicts, each with ``your_odds`` and ``closing_odds``.

    Returns
    -------
    dict
        ``{total_bets, n_with_closing, n_missing_closing,
        avg_clv, positive_clv_pct, bet_results: [{...}]}``

    Examples
    --------
    >>> bets = [
    ...     {"your_odds": 2.10, "closing_odds": 2.05, "market": "1X2", "label": "Arsenal vs Chelsea"},
    ...     {"your_odds": 1.80, "closing_odds": None, "market": "1X2", "label": "Liverpool vs City"},
    ... ]
    >>> result = calculate_batch_clv(bets)
    """
    if not bets:
        return {
            "total_bets": 0,
            "n_with_closing": 0,
            "n_missing_closing": 0,
            "avg_clv": 0.0,
            "positive_clv_pct": 0.0,
            "non_zero_clv_pct": 0.0,
            "max_clv": 0.0,
            "min_clv": 0.0,
            "bet_results": [],
        }

    bet_results: list[dict[str, Any]] = []
    clv_values: list[float] = []
    n_with_closing = 0
    n_missing_closing = 0

    for bet in bets:
        your_odds = bet.get("your_odds", 0.0)
        closing_odds = bet.get("closing_odds")
        market = bet.get("market", "1X2")
        label = bet.get("label", "")

        result = calculate_clv(your_odds, closing_odds, market)
        result["label"] = label
        bet_results.append(result)
        clv_values.append(result["clv"])

        if closing_odds is not None and closing_odds > 1.0:
            n_with_closing += 1
        else:
            n_missing_closing += 1

    # Aggregate
    clv_arr = np.array(clv_values)
    n_positive = int(np.sum(clv_arr > 0))
    total = len(bets)
    n_nonzero = int(np.sum(clv_arr != 0))

    return {
        "total_bets": total,
        "n_with_closing": n_with_closing,
        "n_missing_closing": n_missing_closing,
        "avg_clv": round(float(np.mean(clv_arr)), 6) if total > 0 else 0.0,
        "positive_clv_pct": round(
            (n_positive / total) * 100, 2,
        ) if total > 0 else 0.0,
        "non_zero_clv_pct": round(
            (n_nonzero / total) * 100, 2,
        ) if total > 0 else 0.0,
        "max_clv": round(float(np.max(clv_arr)), 6) if total > 0 else 0.0,
        "min_clv": round(float(np.min(clv_arr)), 6) if total > 0 else 0.0,
        "bet_results": bet_results,
    }


# ═══════════════════════════════════════════════════════════
#  CLV from model predictions + odds
# ═══════════════════════════════════════════════════════════


def calculate_clv_from_probs(
    model_prob: float,
    your_odds: float,
    closing_fair_prob: float,
) -> dict[str, Any]:
    """Calculate CLV using fair (no-margin) probabilities.

    This is an alternative CLV definition that measures the difference
    between your model's probability and the market's closing fair
    probability.

    Formula: CLV = (closing_fair_prob - model_prob) / model_prob

    A negative value means your model assigned LOWER probability than
    the market's closing fair estimate — i.e., you identified value
    the market eventually priced in.

    Parameters
    ----------
    model_prob : float
        Your model's predicted probability (0 to 1).
    your_odds : float
        Decimal odds at which you placed the bet.
    closing_fair_prob : float
        Fair (no-margin) probability implied by closing odds.

    Returns
    -------
    dict
        ``{clv, clv_pct, model_prob, your_odds, closing_fair_prob}``
    """
    if model_prob <= 0.0 or closing_fair_prob <= 0.0:
        return {
            "clv": 0.0,
            "clv_pct": "0.00%",
            "model_prob": model_prob,
            "your_odds": your_odds,
            "closing_fair_prob": closing_fair_prob,
        }

    clv = (closing_fair_prob - model_prob) / model_prob

    return {
        "clv": round(clv, 6),
        "clv_pct": f"{clv * 100:.2f}%",
        "model_prob": round(model_prob, 4),
        "your_odds": round(your_odds, 4),
        "closing_fair_prob": round(closing_fair_prob, 4),
    }


# ═══════════════════════════════════════════════════════════
#  CLV interpretation
# ═══════════════════════════════════════════════════════════


def interpret_clv(clv: float) -> str:
    """Return a human-readable interpretation of a CLV value.

    Parameters
    ----------
    clv : float
        CLV value (e.g. 0.0244 for +2.44%).

    Returns
    -------
    str
        Short interpretation string.
    """
    if clv > 0.10:
        return "Excellent — you significantly beat the closing line"
    if clv > 0.05:
        return "Good — you clearly beat the market"
    if clv > 0.02:
        return "Moderate — slight edge over the closing line"
    if clv > 0.0:
        return "Marginal — barely positive CLV"
    if clv == 0.0:
        return "Neutral — matched the market"
    if clv > -0.02:
        return "Marginal — slightly worse than closing line"
    if clv > -0.05:
        return "Moderate — below market consensus"
    if clv > -0.10:
        return "Poor — significantly below closing line"
    return "Very poor — far below the market"


# ═══════════════════════════════════════════════════════════
#  Explanation guide
# ═══════════════════════════════════════════════════════════


def get_clv_guide() -> str:
    """Return a plain-text explanation of CLV calculation."""
    return """
CLOSING LINE VALUE (CLV) — CALCULATION GUIDE

1. DEFINITION
   CLV measures how your bet price compares to the final market price
   (the "closing line").  Positive CLV is the single best indicator of
   genuine betting skill.

2. FORMULA
   CLV = (your_odds - closing_odds) / closing_odds

   Example: You bet at 2.10, the market closed at 2.05
   CLV = (2.10 - 2.05) / 2.05 = 0.05 / 2.05 = +0.0244 = +2.44%

   This means you got 2.44% better odds than the closing market.

3. INTERPRETATION
   + Positive CLV → the market moved toward your bet after you placed it
     (odds shortened).  This suggests sharp money agreed with you.

   + CLV near zero → your odds matched the market consensus.
     You got a fair price but no edge over the sharp money.

   + Negative CLV → the market moved against your bet (odds drifted).
     The sharp money disagreed with you.

4. WHY CLV MATTERS
   Research across hundreds of thousands of bets shows:
   - Bettors with consistently positive CLV are PROFITABLE over time
   - Bettors with negative CLV are UNPROFITABLE over time
   - CLV predicts FUTURE profitability even when current results are random

   Professional bettors target:
   - Average CLV of +2% to +5% (strong skill edge)
   - 55-65% of bets with positive CLV
   - Minimum CLV of -5% on any single bet (stop-loss)

5. LIMITATIONS
   - CLV requires accurate closing odds, which are not always available
   - Some bookmakers restrict odds or offer limited markets
   - CLV is a long-term metric — individual bet CLV is noisy
   - Opening odds are needed to compute CLV as probability difference
     (the fair-probability method compares opening vs closing)

6. DIFFERENT DEFINITIONS
   Two common CLV formulas exist:

   a) Odds-based (used here):
      CLV = (your_odds - closing_odds) / closing_odds
      Measures the % advantage in PRICE.

   b) Probability-based (alternative):
      CLV = fair_prob_closing - fair_prob_opening
      Measures the change in the market's PROBABILITY estimate.
      This requires opening odds in addition to closing odds.
"""
