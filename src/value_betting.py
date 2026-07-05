"""
Value Betting — identify betting opportunities where model probabilities exceed
bookmaker-implied probabilities.

Core workflow::

    from src.value_betting import compute_value_bets, plot_value_opportunities

    bets = compute_value_bets(
        odds=[[2.10, 3.40, 3.80], [1.95, 3.50, 4.00]],
        model_probs=[[0.52, 0.28, 0.20], [0.48, 0.30, 0.22]],
        team_matches=[("Arsenal", "Chelsea"), ("Liverpool", "Man City")],
    )

    # Positive EV bets only
    good_bets = bets[bets["positive_ev"]]

Calculations explained
----------------------
**1. Implied Probability**
    IP = 1 / decimal_odds
    The bookmaker's perceived probability of an outcome, before their margin.

**2. Bookmaker Margin (overround)**
    margin = sum(implied_probabilities) - 1
    The built-in advantage that guarantees the bookmaker a profit regardless
    of outcome.  Typical margins are 3-8%.

**3. No-margin (fair) probability**
    fair_prob = implied_prob / (1 + margin)
    The bookmaker's \"true\" probability after removing their margin.

**4. Expected Value (EV)**
    EV = (model_prob * decimal_odds) - 1
    Positive EV means the model believes the bet has value.  EV of +0.05
    means an expected 5% return on each unit staked.

**5. Kelly Criterion**
    kelly = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
    The fraction of bankroll to wager to maximise long-term growth.
    A conservative fraction (e.g. 25% Kelly) is often recommended.

**6. Confidence filter**
    Only bets where model_prob > fair_prob (i.e. positive EV) are highlighted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

# Outcome labels for a football match (index order must match model output)
OUTCOME_LABELS = ["Away Win", "Draw", "Home Win"]
OUTCOME_SHORT = ["A", "D", "H"]

# ── Public API ──────────────────────────────────────────


def compute_value_bets(
    odds: list[list[float]] | np.ndarray,
    model_probs: list[list[float]] | np.ndarray,
    team_matches: list[tuple[str, str]] | None = None,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_ev: float = 0.0,
) -> pd.DataFrame:
    """Compute value betting metrics for a set of fixtures.

    Parameters
    ----------
    odds : array-like of shape (n_matches, 3)
        Decimal odds for each match: ``[away_odds, draw_odds, home_odds]``.
    model_probs : array-like of shape (n_matches, 3)
        Model-predicted probabilities: ``[away_prob, draw_prob, home_prob]``.
        Should sum to 1.0 per match.
    team_matches : list[tuple[str, str]], optional
        Human-readable team names: ``[(home, away), ...]``.
    bankroll : float
        Total bankroll for Kelly stake calculation (default 1000).
    kelly_fraction : float
        Fraction of full Kelly to use (default 0.25 = 25% Kelly).
        0.25 is conservative; 0.5 is aggressive; 1.0 is full Kelly.
    min_ev : float
        Minimum EV threshold. Only bets with EV >= *min_ev* are flagged
        as ``positive_ev`` (default 0.0).

    Returns
    -------
    pd.DataFrame
        Columns: match, home_team, away_team, outcome, outcome_label,
        decimal_odds, implied_prob, fair_prob, bookmaker_margin,
        model_prob, ev, kelly_stake, kelly_pct, positive_ev, recommendation.

    Examples
    --------
    >>> odds = [[2.10, 3.40, 3.80], [1.95, 3.50, 4.00]]
    >>> model_probs = [[0.52, 0.28, 0.20], [0.48, 0.30, 0.22]]
    >>> df = compute_value_bets(odds, model_probs)
    >>> df[df["positive_ev"]][["match", "outcome_label", "ev", "kelly_pct"]]
    """
    odds = np.asarray(odds, dtype=float)
    model_probs = np.asarray(model_probs, dtype=float)
    n_matches = len(odds)

    if odds.shape != model_probs.shape:
        raise ValueError(
            f"odds shape {odds.shape} != model_probs shape {model_probs.shape} "
            f"— both must be (n_matches, 3)"
        )

    records: list[dict[str, Any]] = []

    for i in range(n_matches):
        match_odds = odds[i]
        match_probs = model_probs[i]

        # ── 1. Implied probabilities from bookmaker odds ──
        implied_probs = 1.0 / match_odds

        # ── 2. Bookmaker margin (overround) ──────────────
        margin = implied_probs.sum() - 1.0

        # ── 3. Fair (no-margin) probabilities ─────────────
        fair_probs = implied_probs / (1.0 + margin)

        # Home / away team names (with bounds guard)
        if team_matches and i < len(team_matches):
            home_team, away_team = team_matches[i]
            match_label = f"{home_team} vs {away_team}"
        else:
            home_team = away_team = ""
            match_label = f"Match {i + 1}"

        for j, (outcome, label) in enumerate(zip(OUTCOME_SHORT, OUTCOME_LABELS)):
            dec_odds = match_odds[j]
            imp_prob = implied_probs[j]
            fair_prob = fair_probs[j]
            mod_prob = match_probs[j]

            # ── 4. Expected Value ─────────────────────────
            ev = (mod_prob * dec_odds) - 1.0

            # ── 5. Kelly Criterion ────────────────────────
            if dec_odds > 1.0:
                full_kelly = (mod_prob * dec_odds - 1.0) / (dec_odds - 1.0)
            else:
                full_kelly = 0.0
            kelly_pct = max(full_kelly * kelly_fraction, 0.0)
            kelly_stake = bankroll * kelly_pct

            # ── 6. Positive EV filter ─────────────────────
            is_positive = bool(ev >= min_ev and mod_prob > fair_prob and kelly_pct > 0.0)

            records.append({
                "match": match_label,
                "home_team": home_team,
                "away_team": away_team,
                "outcome": outcome,
                "outcome_label": label,
                "decimal_odds": round(dec_odds, 4),
                "implied_prob": round(imp_prob, 4),
                "bookmaker_margin_pct": round(margin * 100, 2),
                "fair_prob": round(fair_prob, 4),
                "model_prob": round(mod_prob, 4),
                "prob_edge": round(mod_prob - fair_prob, 4),
                "ev": round(ev, 4),
                "kelly_fraction": round(kelly_fraction, 2),
                "kelly_pct": round(kelly_pct, 6),
                "kelly_stake": round(kelly_stake, 2),
                "positive_ev": is_positive,
                "recommendation": "✅ VALUE BET" if is_positive else "❌ No value",
            })

    df = pd.DataFrame(records)

    # Sort: positive EV first, then by EV descending
    df.sort_values(
        by=["positive_ev", "ev"],
        ascending=[False, False],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    n_positive = df["positive_ev"].sum()
    logger.info(
        "Found %d / %d value bets (%.1f%%)",
        n_positive, len(df), n_positive / len(df) * 100 if len(df) else 0,
    )
    return df


def compute_value_bets_from_dataframe(
    df: pd.DataFrame,
    odds_cols: tuple[str, str, str] = ("odds_home", "odds_draw", "odds_away"),
    prob_cols: tuple[str, str, str] = ("home_win_prob", "draw_prob", "away_win_prob"),
    home_col: str = "home_team",
    away_col: str = "away_team",
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper that reads odds and probabilities from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain odds columns and probability columns.
    odds_cols : tuple[str, str, str]
        Column names for ``(home_odds, draw_odds, away_odds)``.
    prob_cols : tuple[str, str, str]
        Column names for ``(home_prob, draw_prob, away_prob)``.
    home_col, away_col : str
        Team name columns.
    **kwargs
        Passed through to ``compute_value_bets``.

    Returns
    -------
    pd.DataFrame
        Value bets with all metrics.
    """
    required_odds = [c for c in odds_cols]
    required_probs = [c for c in prob_cols]
    missing = [c for c in required_odds + required_probs if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    odds_array = df[list(odds_cols)].values
    probs_array = df[list(prob_cols)].values
    team_matches = list(zip(df[home_col], df[away_col])) if home_col in df.columns else None

    return compute_value_bets(
        odds=odds_array,
        model_probs=probs_array,
        team_matches=team_matches,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════
#  Explanations (for display)
# ═══════════════════════════════════════════════════════════


def get_calculation_guide() -> str:
    """Return a plain-text explanation of all value-betting calculations."""
    return """
VALUE BETTING — CALCULATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.  IMPLIED PROBABILITY
    ────────────────────
    Formula:    IP = 1 / decimal_odds
    Example:    Odds of 2.10 → IP = 1 / 2.10 = 0.4762 (47.6%)

    What it means:
    The bookmaker's perceived probability of an outcome.  If odds are
    2.10, the bookmaker implies a 47.6% chance of that outcome occurring.
    Because bookmakers build in a margin, these probabilities sum to
    more than 100% across the three outcomes.

2.  BOOKMAKER MARGIN (OVERROUND)
    ─────────────────────────────
    Formula:    margin = sum(IP_home + IP_draw + IP_away) - 1
    Example:    0.476 + 0.294 + 0.263 = 1.033 → margin = 3.3%

    What it means:
    The bookmaker's built-in profit.  A 3.3% margin means the bookmaker
    expects to pay out £96.70 for every £100 staked.  Lower margins are
    better for bettors (more \"fair\" odds).

3.  FAIR (NO-MARGIN) PROBABILITY
    ─────────────────────────────
    Formula:    fair_prob = IP / (1 + margin)
    Example:    0.476 / 1.033 = 0.461 (46.1%)

    What it means:
    The \"true\" probability after stripping out the bookmaker's edge.
    This is our benchmark: if our model's probability exceeds the fair
    probability, we have a potential value bet.

4.  EXPECTED VALUE (EV)
    ────────────────────
    Formula:    EV = (model_prob × decimal_odds) - 1
    Example:    (0.52 × 2.10) - 1 = 0.092 → EV = +9.2%

    What it means:
    The expected return per unit staked.  An EV of +0.092 means for
    every £1 you bet, you expect to make £0.092 in the long run.
    Positive EV bets are profitable in expectation; negative EV bets
    are not, regardless of how \"sure\" they seem.

5.  PROBABILITY EDGE
    ─────────────────
    Formula:    edge = model_prob - fair_prob
    Example:    0.520 - 0.461 = 0.059 (5.9 pp edge)

    What it means:
    How much more likely our model thinks an outcome is compared to
    the bookmaker's fair estimate.  A 5.9 pp edge means we disagree
    with the market by nearly 6 percentage points.

6.  KELLY CRITERION
    ────────────────
    Formula:    kelly = (model_prob × decimal_odds - 1) / (decimal_odds - 1)
    Example:    (0.52 × 2.10 - 1) / (2.10 - 1) = 0.092 / 1.10 = 8.4%

    What it means:
    The optimal fraction of your bankroll to wager to maximise
    long-term growth.  Betting more than Kelly increases risk of ruin;
    betting less reduces growth but is safer.

    Conservative approach: use 25% Kelly (multiply by 0.25).
    For the example above: 8.4% × 0.25 = 2.1% of bankroll.

7.  POSITIVE EV FILTER
    ──────────────────
    A bet is flagged as a VALUE BET when ALL of:
        • EV >= min_ev threshold (default 0.0)
        • Model probability > fair (no-margin) probability
        • Kelly percentage > 0 (positive stake)
"""


def explain_row(row: pd.Series) -> str:
    """Return a sentence explaining why a specific bet is or isn't value."""
    if row["positive_ev"]:
        return (
            f"✅ {row['match']} — {row['outcome_label']} at {row['decimal_odds']:.2f}: "
            f"model sees {row['model_prob']:.1%} vs fair {row['fair_prob']:.1%} "
            f"(EV={row['ev']:+.1%}, edge={row['prob_edge']:+.1%}, "
            f"stake={row['kelly_pct']:.1%}% of bankroll)"
        )
    return (
        f"❌ {row['match']} — {row['outcome_label']} at {row['decimal_odds']:.2f}: "
        f"model {row['model_prob']:.1%} ≤ fair {row['fair_prob']:.1%} "
        f"(EV={row['ev']:+.1%}) — no value"
    )


def print_bets(df: pd.DataFrame, n: int | None = None) -> None:
    """Pretty-print value bets to the console.

    Parameters
    ----------
    df : pd.DataFrame
        Result from ``compute_value_bets``.
    n : int, optional
        Number of rows to print (default: all positive EV + top 5 negative).
    """
    pos = df[df["positive_ev"]]
    neg = df[~df["positive_ev"]]

    display_cols = [
        "match", "outcome_label", "decimal_odds",
        "model_prob", "fair_prob", "prob_edge",
        "ev", "kelly_pct", "recommendation",
    ]

    print("\n" + "=" * 90)
    print("  VALUE BETTING REPORT")
    print("=" * 90)

    if len(pos) > 0:
        print(f"\n  ✅ POSITIVE EV BETS ({len(pos)} found)\n")
        pd.set_option("display.max_columns", 12)
        pd.set_option("display.width", 140)
        print(pos[display_cols].to_string(index=False))
        print(f"\n  Total positive EV bets: {len(pos)}")

        # Print explanation for each positive bet
        print(f"\n  EXPLANATIONS:\n")
        for _, row in pos.iterrows():
            print(f"    • {explain_row(row)}")
    else:
        print("\n  No positive EV bets found.")

    if len(neg) > 0:
        n_show = min(n or 5, len(neg))
        print(f"\n  ❌ NEGATIVE EV BETS (showing top {n_show})\n")
        print(neg.head(n_show)[display_cols].to_string(index=False))

    print("=" * 90)
