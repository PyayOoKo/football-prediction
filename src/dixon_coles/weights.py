"""
Tournament importance weights and recency weight calculations for Dixon-Coles model.

These functions compute match weights based on tournament importance and time decay.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
#  Tournament importance weights
# ═══════════════════════════════════════════════════════════

# Default tournament importance map. Keys are substrings matched
# (case-insensitive) against the league/competition column.
TOURNAMENT_IMPORTANCE: dict[str, float] = {
    # World Cup & equivalents
    "world cup": 2.5,
    "fifa world cup": 2.5,
    "wc": 2.5,
    # Continental championships
    "euro": 2.0,
    "european championship": 2.0,
    "copa america": 2.0,
    "africa cup": 2.0,
    "afcon": 2.0,
    "asian cup": 2.0,
    "gold cup": 2.0,
    "nations league": 1.5,
    "uefa nations league": 1.5,
    # World Cup qualifiers
    "world cup qualifying": 1.5,
    "world cup qualification": 1.5,
    "wc qual": 1.5,
    "wcq": 1.5,
    # Continental qualifiers
    "euro qualifying": 1.2,
    "euro qualification": 1.2,
    # League / club competitions (default)
    "uefa champions league": 1.3,
    "champions league": 1.3,
    "uefa europa league": 1.2,
    "europa league": 1.2,
    "premier league": 1.0,
    "la liga": 1.0,
    "serie a": 1.0,
    "bundesliga": 1.0,
    "ligue 1": 1.0,
    "eredivisie": 1.0,
    "primeira liga": 1.0,
    # Friendlies
    "friendly": 0.6,
    "international friendly": 0.6,
    "club friendly": 0.4,
}


def get_tournament_importance(league: str | None, round_str: str | None = None) -> float:
    """Return the importance weight for a given competition.

    Matches against known league/competition names (case-insensitive).
    If no match is found, returns 1.0 (neutral).

    Parameters
    ----------
    league : str | None
        League or competition name.
    round_str : str | None
        Optional round name for additional context (e.g. "Final" → bonus).

    Returns
    -------
    float
        Importance multiplier (0.0 to 3.0).
    """
    if not league:
        return 1.0

    league_lower = league.lower().strip()
    for pattern, weight in TOURNAMENT_IMPORTANCE.items():
        if pattern in league_lower:
            # Knockout bonus: +20% for knockout stages in important tournaments
            bonus = 1.0
            if round_str and weight >= 1.5:
                round_lower = round_str.lower()
                if any(kw in round_lower for kw in ["final", "semi", "quarter", "round of"]):
                    bonus = 1.2
            return weight * bonus

    return 1.0


def compute_recency_weight(
    match_date: pd.Timestamp | datetime,
    reference_date: pd.Timestamp | datetime,
    halflife_days: float = 1460.0,
) -> float:
    """Compute exponential time-decay weight for a match.

    Formula
    -------
    w = exp(-ln(2) × days_ago / halflife_days)

    Parameters
    ----------
    match_date : pd.Timestamp or datetime
        When the match was played.
    reference_date : pd.Timestamp or datetime
        The "current" date (usually the most recent match date + 1 day).
    halflife_days : float
        Number of days after which weight = 0.5. Default 1460 (~4 years).

    Returns
    -------
    float
        Recency weight between 0 and 1.
    """
    if halflife_days <= 0:
        return 1.0  # no decay

    days_ago = (pd.Timestamp(reference_date) - pd.Timestamp(match_date)).days
    if days_ago < 0:
        days_ago = 0  # future match shouldn't happen, but be safe

    return float(np.exp(-np.log(2) * days_ago / halflife_days))
