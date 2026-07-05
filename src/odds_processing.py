"""
Odds Processing — opening odds, closing odds, movement, market analysis, and CLV.

Purpose
-------
Bookmaker odds contain valuable market information.  The difference between
opening and closing odds reveals how sharp money moved the market, which
outcomes the market expects, and where the market is most confident.

This module processes raw decimal odds (opening and closing) for the three
match outcomes (Home, Draw, Away) and generates a rich set of features.

Equations
---------

**1. Implied Probability**
    IP_outcome = 1 / decimal_odds_outcome

    The bookmaker's raw probability estimate, including their margin.

**2. Bookmaker Margin (Overround)**
    margin = sum(IP_H, IP_D, IP_A) — 1

    The built-in commission.  Typical margins: 2-8%.
    Lower is better (tighter market).

**3. Normalized (Fair) Probability**
    fair_prob_outcome = IP_outcome / (1 + margin)

    Removes the bookmaker's edge to reveal the market's true probability.

**4. Odds Movement**
    movement_decimal = closing_odds — opening_odds

    Positive → odds drifted (outcome became less likely per the market).
    Negative → odds shortened (outcome became more likely per the market).

    movement_pct = movement_decimal / opening_odds × 100

**5. Market Favorite**
    The outcome with the **lowest** decimal odds (highest implied
    probability) at closing.  Returns ``"H"``, ``"D"``, or ``"A"``.

**6. Market Confidence**
    confidence = max(fair_prob_H, fair_prob_D, fair_prob_A)

    How certain the market is about the favorite.  A value of 0.55 means
    the market assigns 55% probability to the most likely outcome.
    Higher values (0.60+) indicate a lopsided match; lower values
    (0.35-0.40) indicate a very even contest.

**7. Closing Line Value (CLV)**
    CLV_outcome = fair_prob_closing — fair_prob_opening

    Positive CLV means the outcome's fair probability INCREASED from
    opening to closing (odds shortened → market moved toward it).
    Negative CLV means the market moved away from that outcome.

    CLV is widely considered the single best measure of betting skill.
    Professional bettors aim for consistently positive CLV on their bets.

**8. Consensus (across bookmakers)**
    When multiple bookmaker odds are available, the module can compute
    the consensus fair probability — the average fair probability across
    all provided bookmakers — as a more robust market estimate.

Column naming conventions
-------------------------
Football-Data.co.uk columns used (configurable):

    ====================  ===========================  =====================
    Role                  Default Opening Columns      Default Closing Cols
    ====================  ===========================  =====================
    Home odds             ``BbMxH`` (BetBrain max)     ``BbAvH`` (BetBrain avg)
    Draw odds             ``BbMxD``                     ``BbAvD``
    Away odds             ``BbMxA``                     ``BbAvA``
    ====================  ===========================  =====================

If none of these columns are found, the module falls back to ``B365H/D/A``
(Bet365 closing odds) for both opening AND closing, and logs a warning.

Usage
-----
::

    from src.odds_processing import add_odds_features

    # Auto-detect odds columns from Football-Data.co.uk format
    df = add_odds_features(df)

    # Custom column names
    df = add_odds_features(
        df,
        opening_odds_cols=("MaxH", "MaxD", "MaxA"),
        closing_odds_cols=("AvgH", "AvgD", "AvgA"),
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default column sets for Football-Data.co.uk ────────
# Opening: BetBrain maximum (widest line available early)
# Closing: BetBrain average (market consensus at kick-off)
_DEFAULT_OPENING = ("BbMxH", "BbMxD", "BbMxA")
_DEFAULT_CLOSING = ("BbAvH", "BbAvD", "BbAvA")

# Fallback: Bet365 (very commonly available)
_FALLBACK_CLOSING = ("B365H", "B365D", "B365A")

# Outcome labels
_OUTCOMES = ["home", "draw", "away"]
_OUTCOME_SHORT = ["H", "D", "A"]


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def add_odds_features(
    df: pd.DataFrame,
    opening_odds_cols: tuple[str, str, str] | None = None,
    closing_odds_cols: tuple[str, str, str] | None = None,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """Add odds-derived features to a match DataFrame.

    Processes opening and closing decimal odds for (Home, Draw, Away)
    and generates the following columns:

    ===============================  ==========================================
    Column                           Description
    ===============================  ==========================================
    ``odds_home_opening``            Opening decimal odds for home win
    ``odds_draw_opening``            Opening decimal odds for draw
    ``odds_away_opening``            Opening decimal odds for away win
    ``odds_home_closing``            Closing decimal odds for home win
    ``odds_draw_closing``            Closing decimal odds for draw
    ``odds_away_closing``            Closing decimal odds for away win
    ``fair_prob_home_opening``       Normalized home win prob (opening)
    ``fair_prob_draw_opening``       Normalized draw prob (opening)
    ``fair_prob_away_opening``       Normalized away win prob (opening)
    ``fair_prob_home_closing``       Normalized home win prob (closing)
    ``fair_prob_draw_closing``       Normalized draw prob (closing)
    ``fair_prob_away_closing``       Normalized away win prob (closing)
    ``odds_movement_home``           Change in home odds (closing - opening)
    ``odds_movement_draw``           Change in draw odds (closing - opening)
    ``odds_movement_away``           Change in away odds (closing - opening)
    ``odds_movement_pct_home``       % change in home odds
    ``odds_movement_pct_draw``       % change in draw odds
    ``odds_movement_pct_away``       % change in away odds
    ``market_favorite``              Shortest-odds outcome (H/D/A)
    ``market_confidence``            Fair prob of the favorite (0-1)
    ``clv_home``                     Closing line value for home win
    ``clv_draw``                     Closing line value for draw
    ``clv_away``                     Closing line value for away win
    ``bookmaker_margin_opening``     Overround in opening odds (fraction)
    ``bookmaker_margin_closing``     Overround in closing odds (fraction)
    ===============================  ==========================================

    Parameters
    ----------
    df : pd.DataFrame
        Match data with odds columns (Football-Data.co.uk format by default).
    opening_odds_cols : tuple[str, str, str], optional
        Column names for opening odds ``(home, draw, away)``.
        Default: ``("BbMxH", "BbMxD", "BbMxA")``.
    closing_odds_cols : tuple[str, str, str], optional
        Column names for closing odds ``(home, draw, away)``.
        Default: ``("BbAvH", "BbAvD", "BbAvA")``.
    home_team_col, away_team_col : str
        Team name columns.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with odds feature columns added.
    """
    df = df.copy()
    logger.info("Adding odds features on %d rows", len(df))

    # ── Resolve column names ──────────────────────────────
    open_cols = _resolve_odds_cols(df, opening_odds_cols, _DEFAULT_OPENING, "opening")
    close_cols = _resolve_odds_cols(df, closing_odds_cols, _DEFAULT_CLOSING, "closing")

    # If still missing both opening and closing, try Bet365 as a fallback
    if open_cols is None and close_cols is None:
        close_cols = _resolve_odds_cols(df, None, _FALLBACK_CLOSING, "closing")
        if close_cols is not None:
            open_cols = close_cols  # Use closing as opening too (no movement data)
            logger.warning(
                "No opening odds found — using closing odds as opening. "
                "Odds movement features will be zero."
            )

    if open_cols is None or close_cols is None:
        logger.warning(
            "No odds columns found in DataFrame. "
            "Creating zero-filled placeholders for all odds features."
        )
        return _add_placeholder_features(df)

    # ── Extract raw odds ──────────────────────────────────
    odds_open = df[list(open_cols)].values.astype(float)
    odds_close = df[list(close_cols)].values.astype(float)

    # Handle NaN odds gracefully
    odds_open = np.where(np.isfinite(odds_open), odds_open, np.nan)
    odds_close = np.where(np.isfinite(odds_close), odds_close, np.nan)

    # ── Compute per-match features ────────────────────────
    n = len(df)
    odds_home_open = odds_open[:, 0]
    odds_draw_open = odds_open[:, 1]
    odds_away_open = odds_open[:, 2]
    odds_home_close = odds_close[:, 0]
    odds_draw_close = odds_close[:, 1]
    odds_away_close = odds_close[:, 2]

    # Implied probabilities
    ip_home_open = 1.0 / odds_home_open
    ip_draw_open = 1.0 / odds_draw_open
    ip_away_open = 1.0 / odds_away_open
    ip_home_close = 1.0 / odds_home_close
    ip_draw_close = 1.0 / odds_draw_close
    ip_away_close = 1.0 / odds_away_close

    # Margins
    margin_open = ip_home_open + ip_draw_open + ip_away_open - 1.0
    margin_close = ip_home_close + ip_draw_close + ip_away_close - 1.0
    margin_open = np.where(margin_open > 0, margin_open, np.nan)
    margin_close = np.where(margin_close > 0, margin_close, np.nan)

    # Fair probabilities (no-margin)
    fair_home_open = np.where(
        np.isfinite(margin_open),
        ip_home_open / (1.0 + margin_open),
        np.nan,
    )
    fair_draw_open = np.where(
        np.isfinite(margin_open),
        ip_draw_open / (1.0 + margin_open),
        np.nan,
    )
    fair_away_open = np.where(
        np.isfinite(margin_open),
        ip_away_open / (1.0 + margin_open),
        np.nan,
    )
    fair_home_close = np.where(
        np.isfinite(margin_close),
        ip_home_close / (1.0 + margin_close),
        np.nan,
    )
    fair_draw_close = np.where(
        np.isfinite(margin_close),
        ip_draw_close / (1.0 + margin_close),
        np.nan,
    )
    fair_away_close = np.where(
        np.isfinite(margin_close),
        ip_away_close / (1.0 + margin_close),
        np.nan,
    )

    # Odds movement
    mov_home = odds_home_close - odds_home_open
    mov_draw = odds_draw_close - odds_draw_open
    mov_away = odds_away_close - odds_away_open

    mov_pct_home = np.where(
        odds_home_open > 0,
        mov_home / odds_home_open * 100,
        np.nan,
    )
    mov_pct_draw = np.where(
        odds_draw_open > 0,
        mov_draw / odds_draw_open * 100,
        np.nan,
    )
    mov_pct_away = np.where(
        odds_away_open > 0,
        mov_away / odds_away_open * 100,
        np.nan,
    )

    # Closing Line Value (change in fair probability)
    clv_home = fair_home_close - fair_home_open
    clv_draw = fair_draw_close - fair_draw_open
    clv_away = fair_away_close - fair_away_open

    # Market favorite and confidence
    # Use closing fair probs; fall back to closing implied if fair is NaN
    if np.any(np.isfinite(fair_home_close)):
        fav_probs = np.column_stack([fair_home_close, fair_draw_close, fair_away_close])
    else:
        fav_probs = np.column_stack([ip_home_close, ip_draw_close, ip_away_close])

    # Guard: if all probabilities are NaN, set default values
    all_nan = ~np.any(np.isfinite(fav_probs), axis=1)

    fav_indices = np.full(n, 0, dtype=int)
    fav_probs_max = np.full(n, np.nan)

    has_valid = ~all_nan
    if has_valid.any():
        valid_probs = fav_probs[has_valid]
        valid_idx = np.nanargmax(valid_probs, axis=1)
        valid_max = np.max(valid_probs, axis=1)
        fav_indices[has_valid] = valid_idx
        fav_probs_max[has_valid] = valid_max

    market_fav = [
        _OUTCOME_SHORT[idx] if np.isfinite(fav_probs_max[i]) else np.nan
        for i, idx in enumerate(fav_indices)
    ]
    market_conf = fav_probs_max

    # ── Assign columns ────────────────────────────────────
    df["odds_home_opening"] = odds_home_open
    df["odds_draw_opening"] = odds_draw_open
    df["odds_away_opening"] = odds_away_open
    df["odds_home_closing"] = odds_home_close
    df["odds_draw_closing"] = odds_draw_close
    df["odds_away_closing"] = odds_away_close

    df["fair_prob_home_opening"] = fair_home_open
    df["fair_prob_draw_opening"] = fair_draw_open
    df["fair_prob_away_opening"] = fair_away_open
    df["fair_prob_home_closing"] = fair_home_close
    df["fair_prob_draw_closing"] = fair_draw_close
    df["fair_prob_away_closing"] = fair_away_close

    df["odds_movement_home"] = mov_home
    df["odds_movement_draw"] = mov_draw
    df["odds_movement_away"] = mov_away
    df["odds_movement_pct_home"] = mov_pct_home
    df["odds_movement_pct_draw"] = mov_pct_draw
    df["odds_movement_pct_away"] = mov_pct_away

    df["market_favorite"] = market_fav
    df["market_confidence"] = market_conf

    df["clv_home"] = clv_home
    df["clv_draw"] = clv_draw
    df["clv_away"] = clv_away

    df["bookmaker_margin_opening"] = margin_open
    df["bookmaker_margin_closing"] = margin_close

    # Log summary
    has_opening = int(np.any(np.isfinite(odds_open)))
    has_closing = int(np.any(np.isfinite(odds_close)))
    logger.info(
        "Odds features added — opening=%s, closing=%s, "
        "avg margin open=%.2f%%, close=%.2f%%",
        "yes" if has_opening else "no",
        "yes" if has_closing else "no",
        np.nanmean(margin_open) * 100 if has_opening else 0,
        np.nanmean(margin_close) * 100 if has_closing else 0,
    )

    return df


# ═══════════════════════════════════════════════════════════
#  Multi-bookmaker consensus
# ═══════════════════════════════════════════════════════════


def add_consensus_features(
    df: pd.DataFrame,
    bookmaker_sets: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Compute consensus fair probabilities across multiple bookmakers.

    When multiple bookmaker odds columns are available, this function
    computes the average fair probability across all of them.  Consensus
    estimates are more robust than any single bookmaker's line.

    Parameters
    ----------
    df : pd.DataFrame
        Match data with odds columns.
    bookmaker_sets : list[tuple[str, str, str]], optional
        List of ``(home_col, draw_col, away_col)`` tuples for each
        bookmaker.  Default: common Football-Data.co.uk columns:
        ``BbAvH/D/A``, ``B365H/D/A``, ``BWH/D/A``, ``IWH/D/A``,
        ``LBH/D/A``, ``SBH/D/A``, ``WHH/D/A``, ``SJH/D/A``,
        ``VCH/D/A``.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with ``consensus_home``, ``consensus_draw``,
        ``consensus_away`` columns added (the mean fair probability
        across all available bookmakers).
    """
    df = df.copy()

    if bookmaker_sets is None:
        bookmaker_sets = [
            ("BbAvH", "BbAvD", "BbAvA"),
            ("B365H", "B365D", "B365A"),
            ("BWH", "BWD", "BWA"),
            ("IWH", "IWD", "IWA"),
            ("LBH", "LBD", "LBA"),
            ("SBH", "SBD", "SBA"),
            ("WHH", "WHD", "WHA"),
            ("SJH", "SJD", "SJA"),
            ("VCH", "VCD", "VCA"),
        ]

    all_fair_home: list[np.ndarray] = []
    all_fair_draw: list[np.ndarray] = []
    all_fair_away: list[np.ndarray] = []

    for h_col, d_col, a_col in bookmaker_sets:
        if h_col not in df.columns or d_col not in df.columns or a_col not in df.columns:
            continue

        odds = df[[h_col, d_col, a_col]].values.astype(float)
        odds = np.where(np.isfinite(odds), odds, np.nan)

        ip = 1.0 / odds
        margin = ip.sum(axis=1) - 1.0
        margin = np.where(margin > 0, margin, np.nan)

        fair = np.where(
            np.isfinite(margin)[:, None],
            ip / (1.0 + margin[:, None]),
            np.nan,
        )

        all_fair_home.append(fair[:, 0])
        all_fair_draw.append(fair[:, 1])
        all_fair_away.append(fair[:, 2])

    if not all_fair_home:
        logger.warning("No bookmaker odds columns found for consensus computation.")
        return df

    # Mean across all available bookmakers
    consensus_home = np.nanmean(np.column_stack(all_fair_home), axis=1)
    consensus_draw = np.nanmean(np.column_stack(all_fair_draw), axis=1)
    consensus_away = np.nanmean(np.column_stack(all_fair_away), axis=1)

    df["consensus_home"] = consensus_home
    df["consensus_draw"] = consensus_draw
    df["consensus_away"] = consensus_away

    logger.info(
        "Consensus features added — %d bookmakers, avg consensus: "
        "H=%.1f%%, D=%.1f%%, A=%.1f%%",
        len(all_fair_home),
        np.nanmean(consensus_home) * 100,
        np.nanmean(consensus_draw) * 100,
        np.nanmean(consensus_away) * 100,
    )

    return df


# ═══════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════


def _resolve_odds_cols(
    df: pd.DataFrame,
    user_cols: tuple[str, str, str] | None,
    default_cols: tuple[str, str, str],
    label: str,
) -> tuple[str, str, str] | None:
    """Resolve which odds columns to use.

    Checks user-provided columns first, then defaults.
    Returns ``None`` if none of the columns exist.
    """
    candidates = [user_cols, default_cols] if user_cols else [default_cols]

    for cols in candidates:
        if all(c in df.columns for c in cols):
            logger.debug("Using %s odds columns: %s", label, cols)
            return cols

    return None


def _add_placeholder_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add zero-filled placeholder odds features when no data is available."""
    placeholders = {
        "odds_home_opening": 0.0,
        "odds_draw_opening": 0.0,
        "odds_away_opening": 0.0,
        "odds_home_closing": 0.0,
        "odds_draw_closing": 0.0,
        "odds_away_closing": 0.0,
        "fair_prob_home_opening": 0.0,
        "fair_prob_draw_opening": 0.0,
        "fair_prob_away_opening": 0.0,
        "fair_prob_home_closing": 0.0,
        "fair_prob_draw_closing": 0.0,
        "fair_prob_away_closing": 0.0,
        "odds_movement_home": 0.0,
        "odds_movement_draw": 0.0,
        "odds_movement_away": 0.0,
        "odds_movement_pct_home": 0.0,
        "odds_movement_pct_draw": 0.0,
        "odds_movement_pct_away": 0.0,
        "market_favorite": np.nan,
        "market_confidence": 0.0,
        "clv_home": 0.0,
        "clv_draw": 0.0,
        "clv_away": 0.0,
        "bookmaker_margin_opening": 0.0,
        "bookmaker_margin_closing": 0.0,
    }
    for col, val in placeholders.items():
        df[col] = val
    return df


# ═══════════════════════════════════════════════════════════
#  Explanation guide
# ═══════════════════════════════════════════════════════════


def get_odds_guide() -> str:
    """Return a plain-text explanation of all odds processing calculations."""
    return """
ODDS PROCESSING — CALCULATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IMPLIED PROBABILITY (IP)
   ─────────────────────────
   Formula:    IP = 1 / decimal_odds

   The raw probability implied by the bookmaker's odds, including their
   built-in margin.  If a team is priced at 2.50 to win, the market
   implies a 40% chance (1/2.50 = 0.40).

   Because bookmakers build in an overround (margin), the three implied
   probabilities (H, D, A) sum to MORE than 1.0 (typically 1.03-1.08).

2. BOOKMAKER MARGIN (OVERROUND)
   ─────────────────────────────
   Formula:    margin = sum(IP_H + IP_D + IP_A) - 1

   The market's built-in commission.  A margin of 0.05 (5%) means the
   bookmaker expects to pay out £95 for every £100 staked.

   Typical margins by market:
     • Premier League: 2-4% (very competitive)
     • Lower leagues:  5-8% (less liquidity)
     • Obscure markets: 8-15% (wide spreads)

3. NORMALIZED (FAIR) PROBABILITY
   ──────────────────────────────
   Formula:    fair_prob = IP / (1 + margin)

   Strips the bookmaker's margin to reveal the market's TRUE probability.
   These three probabilities now sum to exactly 1.0.

   Example: IP = 0.40, margin = 0.05 → fair_prob = 0.40/1.05 = 0.381
   The market truly believes this outcome has a 38.1% chance.

4. ODDS MOVEMENT
   ──────────────
   Formula:    movement = closing_odds - opening_odds
               movement_pct = movement / opening_odds × 100

   Measures how the market's opinion changed between opening and closing:

     • Negative movement (odds shortened): smart money came in on this
       outcome → the market now thinks it's MORE likely.
     • Positive movement (odds drifted): the market now thinks this
       outcome is LESS likely.

   Example: opens at 2.00, closes at 1.80
     movement = 1.80 - 2.00 = -0.20
     movement_pct = -0.20/2.00 × 100 = -10% (shortened 10%)

5. MARKET FAVORITE
   ────────────────
   Formula:    favorite = argmin(decimal_odds_H, D, A)

   The outcome with the shortest odds (lowest decimal odds, highest
   implied probability) at closing time.  Returned as "H", "D", or "A".

6. MARKET CONFIDENCE
   ──────────────────
   Formula:    confidence = max(fair_prob_H, fair_prob_D, fair_prob_A)

   The fair probability of the market favorite.  Measures how one-sided
   the market thinks the match is:
     • 0.35-0.40: very even contest (both teams ~equal)
     • 0.45-0.55: moderate favorite
     • 0.60-0.75: strong favorite
     • 0.80+: overwhelming favorite (extremely rare in football)

7. CLOSING LINE VALUE (CLV)
   ─────────────────────────
   Formula:    CLV_outcome = fair_prob_closing - fair_prob_opening

   The change in fair probability from opening to closing.  Positive CLV
   means the market moved toward this outcome.

   CLV is the gold standard for measuring betting skill because:
     • The closing line is the most efficient price (all info priced in)
     • Consistently positive CLV means you're beating the market
     • It's a leading indicator — positive CLV predicts future profits

   Example: fair_prob_opening = 0.45, fair_prob_closing = 0.52
     CLV = 0.52 - 0.45 = +0.07 (+7 percentage points → strong move)

8. CONSENSUS (MULTI-BOOKMAKER)
   ────────────────────────────
   Formula:    consensus = mean(fair_prob_bookie_1, ..., fair_prob_bookie_N)

   Averages the fair probabilities across all available bookmakers for a
   more robust market estimate.  Reduces the influence of any single
   bookmaker's idiosyncratic pricing.

9. NORMALIZATION
   ──────────────
   "Normalizing odds" means removing the bookmaker's margin to convert
   raw decimal odds into valid probabilities that sum to 1.0.

   Step 1: Convert each odd to implied probability: IP = 1/odds
   Step 2: Compute margin: m = sum(IP) - 1
   Step 3: Remove margin: fair = IP / (1 + m)

   This is also called the "multiplicative method" and is the standard
   approach used by professional bettors and odds comparison sites.
"""
