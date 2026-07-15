"""
Closing Line Value (CLV) — measure of betting skill vs the closing market price.

CLV measures how your bet price compares to the final (closing) market
price.  Consistently positive CLV is widely regarded as the single best
indicator of genuine betting skill.

Formula
-------
    CLV = (your_odds - closing_odds) / closing_odds

    - Positive CLV → you got better odds than the market closed at
    - CLV ≈ 0      → your odds matched the market consensus
    - Negative CLV → you got worse odds than the market closed at

Markets supported
-----------------
- **1X2** (Home Win / Draw / Away Win)
- **BTTS** (Both Teams To Score / No)
- **Over/Under** (e.g. Over 2.5 Goals, Under 2.5 Goals)
- **Any market** with decimal odds

Usage
-----
::

    from src.betting.clv import calculate_clv, calculate_clv_batch

    # Single bet
    result = calculate_clv(your_odds=2.10, closing_odds=2.05)
    print(result["clv"])       # 0.0244
    print(result["clv_pct"])   # 2.44
    print(result["positive"])  # True

    # Batch processing
    results = calculate_clv_batch(
        your_odds_list=[2.10, 3.40, 3.80],
        closing_odds_list=[2.05, 3.50, 3.60],
        labels=["Home", "Draw", "Away"],
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
_MIN_ODDS = 1.0  # Minimum valid decimal odds


# ═══════════════════════════════════════════════════════════
#  Core CLV function
# ═══════════════════════════════════════════════════════════


def calculate_clv(
    your_odds: float,
    closing_odds: float | None,
    *,
    round_to: int | None = 6,
) -> dict[str, Any]:
    """Calculate Closing Line Value for a single bet outcome.

    Parameters
    ----------
    your_odds : float
        Decimal odds at which you placed the bet. Must be > 1.0.
        Values <= 1.0 produce a zero-CLV result.
    closing_odds : float | None
        Decimal odds at market close (kick-off). ``None`` or <= 1.0
        means closing odds are not available — CLV defaults to 0.0.
    round_to : int | None
        Number of decimal places for rounding (default 6).
        Set to ``None`` to disable rounding.

    Returns
    -------
    dict[str, Any]
        ``{
            "clv",              # CLV as a float (e.g. 0.0244)
            "clv_pct",          # CLV as a percentage (e.g. 2.44)
            "positive",         # True if CLV > 0
            "your_odds",
            "closing_odds",
        }``

    Examples
    --------
    >>> r = calculate_clv(2.10, 2.05)
    >>> r["clv"]
    0.0244
    >>> r["positive"]
    True

    >>> r = calculate_clv(2.10, 2.20)
    >>> r["clv"]
    -0.0455
    >>> r["positive"]
    False

    >>> r = calculate_clv(2.10, None)
    >>> r["clv"]
    0.0

    >>> r = calculate_clv(2.10, 0.0)  # closing_odds = 0 → edge case
    >>> r["clv"]
    0.0
    """
    # ── Validate inputs ──
    your_odds_f = float(your_odds)
    closing_f = closing_odds

    # Handle None
    if closing_f is None:
        logger.debug("CLV = 0.0 — no closing odds available (your=%.4f)", your_odds_f)
        return _zero_clv(your_odds_f, None, error="no closing odds")

    closing_f = float(closing_f)

    # Edge case: closing_odds = 0 → division by zero
    if closing_f == 0.0:
        logger.debug("CLV = 0.0 — closing odds is zero (your=%.4f)", your_odds_f)
        return _zero_clv(your_odds_f, closing_f, error="closing odds is zero")

    # Edge case: invalid odds (<= 1.0)
    if your_odds_f <= _MIN_ODDS:
        logger.debug("CLV = 0.0 — your_odds must be > 1.0, got %.4f", your_odds_f)
        return _zero_clv(your_odds_f, closing_f, error="your_odds must be > 1.0")

    if closing_f <= _MIN_ODDS:
        logger.debug("CLV = 0.0 — closing_odds must be > 1.0, got %.4f", closing_f)
        return _zero_clv(your_odds_f, closing_f, error="closing_odds must be > 1.0")

    # Edge case: non-finite values
    if not np.isfinite(your_odds_f) or not np.isfinite(closing_f):
        logger.debug("CLV = 0.0 — non-finite odds (your=%.4f, closing=%.4f)", your_odds_f, closing_f)
        return _zero_clv(your_odds_f, closing_f, error="non-finite odds")

    # ── Compute CLV ──
    clv = (your_odds_f - closing_f) / closing_f

    # ── Round ──
    if round_to is not None:
        clv = round(clv, round_to)

    clv_pct = round(clv * 100, max(round_to - 2, 2) if round_to else 4)

    return {
        "clv": clv,
        "clv_pct": clv_pct,
        "positive": clv > 0,
        "your_odds": round(your_odds_f, round_to) if round_to else your_odds_f,
        "closing_odds": round(closing_f, round_to) if round_to else closing_f,
    }


def _zero_clv(
    your_odds: float,
    closing_odds: float | None,
    error: str = "",
) -> dict[str, Any]:
    """Return a zero-CLV result for invalid inputs."""
    return {
        "clv": 0.0,
        "clv_pct": 0.0,
        "positive": False,
        "your_odds": your_odds,
        "closing_odds": closing_odds,
        "error": error,
    }


# ═══════════════════════════════════════════════════════════
#  Market-specific helpers
# ═══════════════════════════════════════════════════════════


def calculate_clv_1x2(
    your_odds: list[float],
    closing_odds: list[float | None],
    labels: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate CLV for all three 1X2 outcomes.

    Parameters
    ----------
    your_odds : list[float]
        Decimal odds for Home, Draw, Away (in that order).
    closing_odds : list[float | None]
        Closing decimal odds for Home, Draw, Away (in that order).
    labels : list[str], optional
        Custom outcome labels. Defaults to ``["Home", "Draw", "Away"]``.

    Returns
    -------
    dict[str, Any]
        Market-level result with ``outcomes``, ``avg_clv``, etc.
    """
    if labels is None:
        labels = ["Home", "Draw", "Away"]
    return calculate_clv_market(your_odds, closing_odds, labels=labels, **kwargs)


def calculate_clv_binary(
    your_odds: float,
    closing_odds: float | None,
    label: str = "outcome",
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate CLV for a binary market (BTTS, Over/Under, etc.).

    Parameters
    ----------
    your_odds : float
        Decimal odds at which you placed the bet.
    closing_odds : float | None
        Closing decimal odds.
    label : str
        Human-readable label for the outcome (used in logging).

    Returns
    -------
    dict[str, Any]
        Full ``calculate_clv()`` result with ``label`` key added.
    """
    result = calculate_clv(your_odds, closing_odds, **kwargs)
    result["label"] = label
    return result


# ═══════════════════════════════════════════════════════════
#  Multi-outcome / market-level CLV
# ═══════════════════════════════════════════════════════════


def calculate_clv_market(
    your_odds: list[float],
    closing_odds: list[float | None],
    labels: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate CLV for all outcomes in a market (e.g. 1X2, BTTS).

    Parameters
    ----------
    your_odds : list[float]
        Decimal odds at which bets were placed, one per outcome.
    closing_odds : list[float | None]
        Closing decimal odds, one per outcome. ``None`` = unavailable.
    labels : list[str], optional
        Outcome labels (e.g. ``["Home", "Draw", "Away"]``).

    Returns
    -------
    dict[str, Any]
        ``{
            "outcomes": [list of per-outcome CLV results],
            "avg_clv", "max_clv", "min_clv",
            "n_positive", "n_negative", "n_zero", "total",
        }``

    Examples
    --------
    >>> result = calculate_clv_market(
    ...     your_odds=[2.10, 3.40, 3.80],
    ...     closing_odds=[2.05, 3.50, 3.60],
    ...     labels=["H", "D", "A"],
    ... )
    >>> result["avg_clv"]
    0.0187
    >>> result["n_positive"]
    1
    """
    n = len(your_odds)
    if labels is None:
        labels = [f"outcome_{i}" for i in range(n)]

    # Pad labels if fewer than outcomes
    while len(labels) < n:
        labels.append(f"outcome_{len(labels) + 1}")

    outcomes: list[dict[str, Any]] = []
    clv_values: list[float] = []

    for i in range(n):
        odds = your_odds[i]
        close = closing_odds[i] if i < len(closing_odds) else None
        label = labels[i] if i < len(labels) else f"outcome_{i}"

        result = calculate_clv(odds, close, **kwargs)
        result["label"] = label
        result["index"] = i
        outcomes.append(result)
        clv_values.append(result["clv"])

    # Aggregate
    clv_arr = np.array(clv_values, dtype=np.float64)
    n_positive = int(np.sum(clv_arr > 0))
    n_negative = int(np.sum(clv_arr < 0))
    n_zero = int(np.sum(clv_arr == 0))

    return {
        "outcomes": outcomes,
        "avg_clv": round(float(np.mean(clv_arr)), 6),
        "max_clv": round(float(np.max(clv_arr)), 6) if n > 0 else 0.0,
        "min_clv": round(float(np.min(clv_arr)), 6) if n > 0 else 0.0,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "total": n,
    }


# ═══════════════════════════════════════════════════════════
#  Batch CLV (multiple independent bets)
# ═══════════════════════════════════════════════════════════


def calculate_clv_batch(
    your_odds_list: list[float],
    closing_odds_list: list[float | None],
    labels: list[str] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Calculate CLV for multiple independent bets.

    Parameters
    ----------
    your_odds_list : list[float]
        Decimal odds at which each bet was placed.
    closing_odds_list : list[float | None]
        Closing decimal odds for each bet.
    labels : list[str], optional
        Optional labels for each bet.

    Returns
    -------
    list[dict[str, Any]]
        List of ``calculate_clv()`` results, one per bet.
    """
    n = min(len(your_odds_list), len(closing_odds_list))
    if labels is None:
        labels = [f"bet_{i}" for i in range(n)]

    results = []
    for i in range(n):
        result = calculate_clv(your_odds_list[i], closing_odds_list[i], **kwargs)
        result["label"] = labels[i] if i < len(labels) else f"bet_{i}"
        result["index"] = i
        results.append(result)

    return results


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

    Examples
    --------
    >>> interpret_clv(0.0244)
    'Moderate — slight edge over the closing line'
    >>> interpret_clv(-0.10)
    'Poor — significantly below closing line'
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
#  Aggregation helpers
# ═══════════════════════════════════════════════════════════


def summarise_clv_batch(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate multiple CLV results into summary statistics.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of ``calculate_clv()`` results.

    Returns
    -------
    dict[str, Any]
        ``{
            "total_bets", "n_with_closing", "n_missing_closing",
            "avg_clv", "positive_clv_pct",
            "max_clv", "min_clv",
        }``
    """
    if not results:
        return {
            "total_bets": 0,
            "n_with_closing": 0,
            "n_missing_closing": 0,
            "avg_clv": 0.0,
            "positive_clv_pct": 0.0,
            "max_clv": 0.0,
            "min_clv": 0.0,
        }

    clv_values: list[float] = []
    n_with_closing = 0
    n_missing_closing = 0

    for r in results:
        clv_values.append(r.get("clv", 0.0))
        close = r.get("closing_odds")
        if close is not None and close > _MIN_ODDS:
            n_with_closing += 1
        else:
            n_missing_closing += 1

    total = len(results)
    clv_arr = np.array(clv_values, dtype=np.float64)
    n_positive = int(np.sum(clv_arr > 0))

    return {
        "total_bets": total,
        "n_with_closing": n_with_closing,
        "n_missing_closing": n_missing_closing,
        "avg_clv": round(float(np.mean(clv_arr)), 6),
        "positive_clv_pct": round(n_positive / total * 100, 2),
        "max_clv": round(float(np.max(clv_arr)), 6),
        "min_clv": round(float(np.min(clv_arr)), 6),
    }


# ═══════════════════════════════════════════════════════════
#  CLVTracker — track CLV over a historical period
# ═══════════════════════════════════════════════════════════


class CLVTracker:
    """Track and analyse Closing Line Value over a historical period.

    Maintains a running record of bets with their CLV values,
    computes summary metrics, and tracks trends over time.

    Parameters
    ----------
    bets : list[tuple[float, float | None, str]], optional
        Pre-populate with a list of ``(your_odds, closing_odds, date)`` tuples.
        Dates should be sortable strings (e.g. ``"2026-07-14"``).

    Examples
    --------
    >>> tracker = CLVTracker()
    >>> tracker.add_bet(2.10, 2.05, "2026-07-01")
    >>> tracker.add_bet(1.95, 2.00, "2026-07-02")
    >>> metrics = tracker.calculate_metrics()
    >>> metrics["avg_clv"]
    0.0122
    >>> metrics["clv_gt_0_pct"]
    50.0
    """

    def __init__(
        self,
        bets: list[tuple[float, float | None, str]] | None = None,
    ) -> None:
        self.bets: list[dict[str, Any]] = []
        if bets:
            for your_odds, closing_odds, date in bets:
                self.add_bet(your_odds, closing_odds, date)
            # Sort by date to ensure chronological trend analysis
            self.bets.sort(key=lambda x: x.get("date", ""))

    # ── Public API ────────────────────────────────────────

    def add_bet(
        self,
        your_odds: float,
        closing_odds: float | None,
        date: str,
    ) -> dict[str, Any]:
        """Add a single bet and return its CLV calculation.

        Parameters
        ----------
        your_odds : float
            Decimal odds at which the bet was placed.
        closing_odds : float | None
            Closing decimal odds for the bet.
        date : str
            Date of the bet (sortable format, e.g. ``"2026-07-14"``).

        Returns
        -------
        dict[str, Any]
            The CLV result dict with ``date`` and ``index`` added.
        """
        result = calculate_clv(your_odds, closing_odds)
        result["date"] = date
        result["index"] = len(self.bets)

        self.bets.append(result)
        return result

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate all CLV summary metrics.

        Returns
        -------
        dict[str, Any]
            ``{
                "total_bets",          # total number of bets tracked
                "n_with_closing",      # bets with valid closing odds
                "avg_clv",             # mean CLV across all bets
                "median_clv",          # median CLV
                "std_clv",             # standard deviation of CLV
                "min_clv",
                "max_clv",
                "clv_gt_0_pct",        # % of bets with CLV > 0
                "clv_gt_5_pct",        # % of bets with CLV > 5%
                "clv_gt_10_pct",       # % of bets with CLV > 10%
                "clv_lt_0_pct",        # % of bets with CLV < 0
                "n_positive",
                "n_negative",
                "n_zero",
                "n_gt_5",              # bets with CLV > 0.05
                "n_gt_10",             # bets with CLV > 0.10
                "trend",               # "Improving" / "Declining" / "Stable"
                "recent_avg_clv",      # avg CLV of last 10 bets (or all if <10)
                "recent_vs_overall",   # % change: recent vs overall
                "clv_by_month",        # average CLV per month
                "best_date",           # date of best CLV
                "worst_date",          # date of worst CLV
            }``
        """
        if not self.bets:
            return {
                "total_bets": 0,
                "n_with_closing": 0,
                "avg_clv": 0.0,
                "median_clv": 0.0,
                "std_clv": 0.0,
                "min_clv": 0.0,
                "max_clv": 0.0,
                "clv_gt_0_pct": 0.0,
                "clv_gt_5_pct": 0.0,
                "clv_gt_10_pct": 0.0,
                "clv_lt_0_pct": 0.0,
                "n_positive": 0,
                "n_negative": 0,
                "n_zero": 0,
                "n_gt_5": 0,
                "n_gt_10": 0,
                "trend": "Stable",
                "recent_avg_clv": 0.0,
                "recent_vs_overall": 0.0,
                "clv_by_month": {},
                "best_date": "",
                "worst_date": "",
            }

        clv_values = np.array([b.get("clv", 0.0) for b in self.bets], dtype=np.float64)
        total = len(clv_values)

        n_positive = int(np.sum(clv_values > 0))
        n_negative = int(np.sum(clv_values < 0))
        n_zero = int(np.sum(clv_values == 0))
        n_gt_5 = int(np.sum(clv_values > 0.05))
        n_gt_10 = int(np.sum(clv_values > 0.10))

        # Valid closing odds count
        n_with = sum(
            1 for b in self.bets
            if b.get("closing_odds") is not None
            and b.get("closing_odds", 0) > _MIN_ODDS
        )

        # Trend: compare first half vs second half
        trend_result = self._calculate_trend(clv_values)

        # Recent performance (last 10 bets or 30%, whichever is smaller)
        window = max(3, min(10, total // 3))
        recent = clv_values[-window:]
        recent_avg = float(np.mean(recent))
        overall_avg = float(np.mean(clv_values))

        recent_vs = (
            ((recent_avg - overall_avg) / abs(overall_avg)) * 100
            if overall_avg != 0
            else 0.0
        )

        # CLV by month
        clv_by_month: dict[str, list[float]] = {}
        for b in self.bets:
            date_str = b.get("date", "")
            if len(date_str) >= 7:
                month_key = date_str[:7]  # "2026-07"
            else:
                month_key = date_str
            if month_key not in clv_by_month:
                clv_by_month[month_key] = []
            clv_by_month[month_key].append(b.get("clv", 0.0))

        month_avgs = {
            k: round(float(np.mean(v)), 6)
            for k, v in sorted(clv_by_month.items())
        }

        # Best / worst dates
        best_idx = int(np.argmax(clv_values))
        worst_idx = int(np.argmin(clv_values))

        return {
            "total_bets": total,
            "n_with_closing": n_with,
            "avg_clv": round(overall_avg, 6),
            "median_clv": round(float(np.median(clv_values)), 6),
            "std_clv": round(float(np.std(clv_values, ddof=1)), 6) if total > 1 else 0.0,
            "min_clv": round(float(np.min(clv_values)), 6),
            "max_clv": round(float(np.max(clv_values)), 6),
            "clv_gt_0_pct": round(n_positive / total * 100, 2),
            "clv_gt_5_pct": round(n_gt_5 / total * 100, 2),
            "clv_gt_10_pct": round(n_gt_10 / total * 100, 2),
            "clv_lt_0_pct": round(n_negative / total * 100, 2),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "n_gt_5": n_gt_5,
            "n_gt_10": n_gt_10,
            "trend": trend_result["direction"],
            "recent_avg_clv": round(recent_avg, 6),
            "recent_vs_overall": round(recent_vs, 2),
            "clv_by_month": month_avgs,
            "best_date": self.bets[best_idx].get("date", ""),
            "worst_date": self.bets[worst_idx].get("date", ""),
            "best_clv": round(float(clv_values[best_idx]), 6),
            "worst_clv": round(float(clv_values[worst_idx]), 6),
        }

    def to_dataframe(self) -> "pd.DataFrame":
        """Return all bets as a pandas DataFrame.

        Requires pandas to be installed.

        Returns
        -------
        pd.DataFrame
            Columns: index, date, your_odds, closing_odds, clv, clv_pct, positive, error
        """
        import pandas as pd

        records = []
        for b in self.bets:
            records.append({
                "index": b.get("index"),
                "date": b.get("date", ""),
                "your_odds": b.get("your_odds"),
                "closing_odds": b.get("closing_odds"),
                "clv": b.get("clv", 0.0),
                "clv_pct": b.get("clv_pct", 0.0),
                "positive": b.get("positive", False),
                "error": b.get("error", ""),
            })
        return pd.DataFrame(records)

    def save(
        self,
        filepath: str | None = None,
        directory: str = "reports",
    ) -> str:
        """Save tracking results to a JSON file.

        Parameters
        ----------
        filepath : str, optional
            Full path to the output file. If omitted, auto-generates
            ``reports/clv_tracking_{timestamp}.json``.
        directory : str
            Output directory when ``filepath`` is not provided.

        Returns
        -------
        str
            Path to the saved file.

        Examples
        --------
        >>> tracker = CLVTracker([(2.10, 2.05, "2026-07-01")])
        >>> path = tracker.save()
        >>> print(path)
        reports/clv_tracking_20260715_120000.json
        """
        from datetime import datetime
        from pathlib import Path

        metrics = self.calculate_metrics()

        # Per-bet records (omit full dicts to keep file manageable)
        bet_records: list[dict[str, Any]] = []
        for b in self.bets:
            bet_records.append({
                "index": b.get("index"),
                "date": b.get("date", ""),
                "your_odds": b.get("your_odds"),
                "closing_odds": b.get("closing_odds"),
                "clv": b.get("clv"),
                "clv_pct": b.get("clv_pct"),
                "positive": b.get("positive"),
                "error": b.get("error", ""),
            })

        output = {
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "total_bets": metrics["total_bets"],
            "metrics": metrics,
            "bets": bet_records,
        }

        # Determine path
        if filepath is None:
            Path(directory).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = str(Path(directory) / f"clv_tracking_{ts}.json")

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info("Saved CLV tracking to %s", filepath)
        print(f"[CLVTracker] Saved to {filepath}")
        return filepath

    # ── Internal helpers ──────────────────────────────────

    def _calculate_trend(
        self,
        clv_values: np.ndarray | None = None,
    ) -> dict[str, str]:
        """Determine the CLV trend direction.

        Compares the average CLV of the first 40% of bets against the
        last 40% of bets. If the recent average is significantly higher,
        the trend is "Improving". If significantly lower, "Declining".
        Otherwise "Stable".

        Parameters
        ----------
        clv_values : np.ndarray, optional
            Pre-computed CLV values. If ``None``, computes from ``self.bets``.

        Returns
        -------
        dict[str, str]
            ``{"direction": "Improving" | "Declining" | "Stable",
            "first_avg": ..., "last_avg": ...}``
        """
        if clv_values is None:
            clv_values = np.array([b.get("clv", 0.0) for b in self.bets], dtype=np.float64)

        total = len(clv_values)
        if total < 5:
            return {
                "direction": "Stable",
                "first_avg": round(float(np.mean(clv_values)), 6) if total > 0 else 0.0,
                "last_avg": round(float(np.mean(clv_values)), 6) if total > 0 else 0.0,
            }

        # Compare first 40% vs last 40%
        split = int(total * 0.4)
        first_half = clv_values[:split]
        last_half = clv_values[-split:]

        first_avg = float(np.mean(first_half))
        last_avg = float(np.mean(last_half))

        # Use a threshold: 20% change relative to the larger absolute value
        threshold = 0.2 * max(abs(first_avg), abs(last_avg), 1e-6)
        diff = last_avg - first_avg

        if diff > threshold:
            direction = "Improving"
        elif diff < -threshold:
            direction = "Declining"
        else:
            direction = "Stable"

        return {
            "direction": direction,
            "first_avg": round(first_avg, 6),
            "last_avg": round(last_avg, 6),
        }

    def __len__(self) -> int:
        """Return the number of bets tracked."""
        return len(self.bets)

    def __repr__(self) -> str:
        return (
            f"CLVTracker(bets={len(self.bets)}, "
            f"avg_clv={self.calculate_metrics().get('avg_clv', 'N/A')})"
        )
