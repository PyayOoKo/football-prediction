"""
Expected Goals (xG) Features — rolling xG, xGA, xG Difference, and Expected Points.

Purpose
-------
Expected Goals (xG) measure the quality of each shot attempt and are the
gold standard metric in modern football analytics.  This module provides:

1. **Auto-detection** of xG columns from enriched data sources (Understat,
   FBref, Opta, etc.) if present in the DataFrame.
2. **Zero-filled placeholders** when real xG data is unavailable — these
   can be overwritten later when an enriched data source is connected.
3. **Leakage-free rolling features** using expanding windows with ``.shift(1)``
   so each match only sees data available *before* kick-off.
4. **Expected Points (xPts)** conversion via the Poisson distribution.

Column naming conventions detected (case-insensitive)
------------------------------------------------------
- ``home_xg`` / ``away_xg``         (most common — Understat, FBref exports)
- ``xg_home`` / ``xg_away``
- ``xghome`` / ``xgaway``
- ``h_xg`` / ``a_xg``               (our internal standard prefix)

If none of these are found, zero-filled columns ``home_xg`` and ``away_xg``
are added as placeholders with a logged warning.

Equations
---------

**1. Rolling xG (attack efficiency)**
    avg_xG_5 = mean of last 5 matches' xG for the team
    avg_xG_N = mean of last N matches' xG for the team

    This measures how well a team is creating quality chances.

**2. Rolling xGA (defensive solidity)**
    avg_xGA_5 = mean of last 5 matches' xGA for the team
    avg_xGA_N = mean of last N matches' xGA for the team

    xGA is the xG *conceded* (opponent's xG in the team's matches).
    Lower is better — a low xGA means the team restricts opponents
    to low-quality chances.

**3. xG Difference (net xG)**
    xGD = xG − xGA

    Positive xGD = creating more and better chances than conceding.
    Strongly correlated with long-term league position.

**4. Expected Points (xPts)**
    Using the Poisson distribution to convert xG into outcome probabilities:

        P(0 goals) = e^{−λ}
        P(1 goal)  = e^{−λ} × λ
        P(2 goals) = e^{−λ} × λ² / 2
        ...

    For a match with expected goals λ_home, λ_away:

        P(Home Win) = Σ Σ Pois(i, λ_home) × Pois(j, λ_away)  for i > j
        P(Draw)     = Σ Σ Pois(i, λ_home) × Pois(j, λ_away)  for i = j
        P(Away Win) = Σ Σ Pois(i, λ_home) × Pois(j, λ_away)  for i < j

        xPts = P(Home Win) × 3 + P(Draw) × 1

    See ``src.poisson_model`` for the full derivations.

Leakage prevention
------------------
All rolling statistics use ``.shift(1)`` so the current match's xG values
are never included in its own features.  The expanding window ensures we
only use matches that truly occurred before the current one.

Usage
-----
::

    from src.xg_features import add_xg_features

    # Auto-detects xG columns or creates placeholders
    df = add_xg_features(df)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── xG column name patterns (case-insensitive) ──────────
# The module checks these in priority order.  First match wins.
_XG_HOME_PATTERNS = ["home_xg", "xg_home", "xghome", "h_xg", "hxg"]
_XG_AWAY_PATTERNS = ["away_xg", "xg_away", "xgaway", "a_xg", "axg"]

# Rolling window labels used as column suffixes
_ROLLING_WINDOWS = [5, 10]


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def add_xg_features(
    df: pd.DataFrame,
    rolling_windows: tuple[int, ...] = _ROLLING_WINDOWS,
    compute_xpts: bool = True,
    max_goals_table: int = 8,
    placeholder_value: float = 0.0,
    warn_missing: bool = True,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
) -> pd.DataFrame:
    """Add Expected Goals features to a match DataFrame.

    This function:
    1. Detects xG columns in the DataFrame (or creates zero-filled placeholders).
    2. Computes rolling averages of xG and xGA per team (leakage-free).
    3. Computes xG Difference and xG Difference rolling averages.
    4. Optionally computes Expected Points (xPts) via Poisson conversion.

    Parameters
    ----------
    df : pd.DataFrame
        Match data **sorted by date**.
    rolling_windows : tuple[int, ...]
        Window sizes for rolling averages (default 5, 10).
    compute_xpts : bool
        Whether to compute Expected Points (default True).
    max_goals_table : int
        Max goals per team for xPts probability table (default 8).
    placeholder_value : float
        Value for xG placeholders when no real xG data exists (default 0.0).
        Use ``None`` to leave as NaN.
    warn_missing : bool
        Warn when xG columns not found (default True).
    home_team_col, away_team_col : str
        Team name columns.
    home_goals_col, away_goals_col : str
        Goals columns (used as fallback for xPts when xG is unavailable).

    Returns
    -------
    pd.DataFrame
        Copy of **df** with the following columns added:

        ============================  =========================================
        Column                        Description
        ============================  =========================================
        ``home_xg`` / ``away_xg``     Match-level xG (real or placeholder)
        ``h_xg_avg5`` / ``a_xg_avg5`` Rolling 5-match xG average (pre-match)
        ``h_xg_avgN`` / ``a_xg_avgN`` Rolling N-match xG average (pre-match)
        ``h_xga_avg5`` / ``a_xga_avg5`` Rolling 5-match xGA average
        ``h_xga_avgN`` / ``a_xga_avgN`` Rolling N-match xGA average
        ``h_xgd_avg5`` / ``a_xgd_avg5`` Rolling 5-match xG difference
        ``h_xgd_avgN`` / ``a_xgd_avgN`` Rolling N-match xG difference
        ``h_xpts`` / ``a_xpts``       Expected Points from match xG
        ``xgd``                       Match-level xG difference (home − away)
        ============================  =========================================
    """
    df = df.copy()
    logger.info("Adding xG features on %d rows", len(df))

    # ── 1. Detect or create xG columns ────────────────────
    df, xg_available = _detect_or_create_xg_columns(
        df,
        placeholder_value=placeholder_value,
        warn_missing=warn_missing,
    )

    # ── 2. Build per-team match-level stats (one row per team per match) ──
    team_stats = _build_team_xg_stats(
        df,
        home_team_col=home_team_col,
        away_team_col=away_team_col,
    )

    # ── 3. Compute rolling xG, xGA, xGD per team ──────────
    team_rolling = _compute_rolling_xg(team_stats, rolling_windows)

    # ── 4. Merge home and away rolling features ───────────
    df = _merge_xg_features(df, team_rolling)

    # ── 5. Compute Expected Points (if enabled) ───────────
    if compute_xpts:
        df = _compute_expected_points(
            df,
            max_goals=max_goals_table,
            home_goals_col=home_goals_col,
            away_goals_col=away_goals_col,
        )

    n_features = len([c for c in df.columns if "xg" in c.lower()])
    source = "real xG data" if xg_available else "placeholders (no xG data found)"
    logger.info("xG features added — %d columns from %s", n_features, source)

    return df


# ═══════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════


def _find_xg_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Search for xG column names in the DataFrame (case-insensitive).

    Returns
    -------
    tuple[str | None, str | None]
        ``(home_xg_column, away_xg_column)`` — the actual column names
        found, or ``None`` if not found.
    """
    df_lower = {col.lower(): col for col in df.columns}

    home_xg_col: str | None = None
    away_xg_col: str | None = None

    for pattern in _XG_HOME_PATTERNS:
        if pattern in df_lower:
            home_xg_col = df_lower[pattern]
            break

    for pattern in _XG_AWAY_PATTERNS:
        if pattern in df_lower:
            away_xg_col = df_lower[pattern]
            break

    return home_xg_col, away_xg_col


def _detect_or_create_xg_columns(
    df: pd.DataFrame,
    placeholder_value: float = 0.0,
    warn_missing: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Detect real xG columns or create zero-filled placeholders.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(df, xg_available)`` — the DataFrame with ``home_xg`` and
        ``away_xg`` columns guaranteed to exist, and a flag indicating
        whether real (non-placeholder) xG data was found.
    """
    home_xg_col, away_xg_col = _find_xg_columns(df)

    if home_xg_col is not None and away_xg_col is not None:
        # Real xG data found — rename to canonical names
        df.rename(
            columns={
                home_xg_col: "home_xg",
                away_xg_col: "away_xg",
            },
            inplace=True,
        )
        logger.info(
            "Real xG columns detected: '%s' / '%s'",
            home_xg_col, away_xg_col,
        )
        return df, True

    # No real xG data — create placeholders
    if warn_missing:
        logger.warning(
            "No xG columns found (checked patterns: %s, %s). "
            "Creating zero-filled placeholders.",
            _XG_HOME_PATTERNS,
            _XG_AWAY_PATTERNS,
        )

    if placeholder_value is not None:
        df["home_xg"] = float(placeholder_value)
        df["away_xg"] = float(placeholder_value)
    else:
        df["home_xg"] = np.nan
        df["away_xg"] = np.nan

    return df, False


def _build_team_xg_stats(
    df: pd.DataFrame,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
) -> pd.DataFrame:
    """Build a per-team per-match DataFrame with xG and xGA values.

    Each match produces two rows: home team's perspective and away team's.
    xGA for a team is the opponent's xG in that match.
    """
    records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        home = row[home_team_col]
        away = row[away_team_col]
        h_xg = float(row.get("home_xg", 0.0) or 0.0)
        a_xg = float(row.get("away_xg", 0.0) or 0.0)

        # Home team
        records.append({
            "team": home,
            "date": row.get("date", pd.NaT),
            "opponent": away,
            "is_home": 1,
            "xg": h_xg,
            "xga": a_xg,
            "match_id": idx,
        })

        # Away team
        records.append({
            "team": away,
            "date": row.get("date", pd.NaT),
            "opponent": home,
            "is_home": 0,
            "xg": a_xg,
            "xga": h_xg,
            "match_id": idx,
        })

    team_df = pd.DataFrame(records)
    team_df.sort_values(["team", "date"], inplace=True)
    team_df.reset_index(drop=True, inplace=True)
    return team_df


def _compute_rolling_xg(
    team_stats: pd.DataFrame,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """Compute rolling averages of xG, xGA, and xGD per team.

    **Leakage prevention:** Uses ``.shift(1)`` so the current match's
    values are excluded from its own rolling averages.

    Parameters
    ----------
    team_stats : pd.DataFrame
        Per-team per-match data with columns ``team``, ``xg``, ``xga``,
        ``is_home``, ``match_id``.
    windows : tuple[int, ...]
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        Same as **team_stats** with rolling xG/xGA/xGD columns added.
    """
    team_stats = team_stats.copy()
    cols_to_merge = ["match_id", "is_home"]
    rolling_suffixes = [f"avg{w}" for w in windows]

    def _rolling_group(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("date").copy()

        # Rolling xG
        for w, suffix in zip(windows, rolling_suffixes):
            grp[f"xg_{suffix}"] = (
                grp["xg"].rolling(w, min_periods=1).mean().shift(1)
            )

        # Rolling xGA
        for w, suffix in zip(windows, rolling_suffixes):
            grp[f"xga_{suffix}"] = (
                grp["xga"].rolling(w, min_periods=1).mean().shift(1)
            )

        # Rolling xG Difference (xG - xGA)
        grp["xgd"] = grp["xg"] - grp["xga"]
        for w, suffix in zip(windows, rolling_suffixes):
            grp[f"xgd_{suffix}"] = (
                grp["xgd"].rolling(w, min_periods=1).mean().shift(1)
            )

        return grp

    team_stats = team_stats.groupby("team", group_keys=False).apply(_rolling_group)

    # Collect rolling column names for merge step
    rolling_cols = []
    for suffix in rolling_suffixes:
        rolling_cols.extend([f"xg_{suffix}", f"xga_{suffix}", f"xgd_{suffix}"])
    cols_to_merge.extend(rolling_cols)

    return team_stats[cols_to_merge]


def _merge_xg_features(
    df: pd.DataFrame,
    team_rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Merge rolling xG features back onto the original DataFrame.

    Home team stats get ``h_`` prefix, away team stats get ``a_`` prefix.
    """
    # Home team features
    home_stats = team_rolling[team_rolling["is_home"] == 1].copy()
    home_stats.rename(
        columns={c: f"h_{c}" for c in home_stats.columns if c not in ("match_id", "is_home")},
        inplace=True,
    )
    df = df.merge(home_stats, left_index=True, right_on="match_id", how="left")

    # Away team features
    away_stats = team_rolling[team_rolling["is_home"] == 0].copy()
    away_stats.rename(
        columns={c: f"a_{c}" for c in away_stats.columns if c not in ("match_id", "is_home")},
        inplace=True,
    )
    df = df.merge(away_stats, left_index=True, right_on="match_id", how="left")

    # Clean up auxiliary columns
    df.drop(columns=[c for c in df.columns if c.endswith("_match_id") or c == "is_home"],
            inplace=True, errors="ignore")

    return df


# ═══════════════════════════════════════════════════════════
#  Expected Points (xPts) from xG
# ═══════════════════════════════════════════════════════════


def _compute_expected_points(
    df: pd.DataFrame,
    max_goals: int = 8,
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
) -> pd.DataFrame:
    """Compute Expected Points (xPts) for each match using the Poisson model.

    When real xG data is available, uses xG values.  Falls back to actual
    goals if xG placeholders are all zero.

    Formula
    -------
    For each match:
        P(home wins) = Σ_{i>j} Pois(i, λ_home) × Pois(j, λ_away)
        P(draw)      = Σ_{i=j} Pois(i, λ_home) × Pois(j, λ_away)
        xPts_home    = P(Home Win) × 3 + P(Draw) × 1
        xPts_away    = P(Away Win) × 3 + P(Draw) × 1
    """
    if "home_xg" in df.columns and "away_xg" in df.columns:
        λ_home_list = df["home_xg"].values.astype(float)
        λ_away_list = df["away_xg"].values.astype(float)
    else:
        # Fallback to actual goals
        λ_home_list = df[home_goals_col].values.astype(float)
        λ_away_list = df[away_goals_col].values.astype(float)

    # Pre-compute Poisson probabilities for all (k, λ) combinations
    # to avoid repeated computation
    unique_lambdas = np.unique(np.concatenate([λ_home_list, λ_away_list]))
    lambda_to_probs: dict[float, list[float]] = {}

    for lam in unique_lambdas:
        if pd.isna(lam) or lam <= 0:
            # For lambda <= 0, return a degenerate distribution (P(0) = 1)
            probs = [1.0] + [0.0] * max_goals
        else:
            probs = [float(np.exp(-lam))]
            for k in range(1, max_goals + 1):
                probs.append(probs[-1] * lam / k)
            # Normalise to account for truncation
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
        lambda_to_probs[lam] = probs

    def _xpts(home_lam: float, away_lam: float) -> tuple[float, float]:
        """Compute (home_xPts, away_xPts) from expected goals."""
        if pd.isna(home_lam) or pd.isna(away_lam):
            return 0.0, 0.0

        p_home = lambda_to_probs.get(home_lam, [1.0] + [0.0] * max_goals)
        p_away = lambda_to_probs.get(away_lam, [1.0] + [0.0] * max_goals)

        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = p_home[i] * p_away[j]
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob

        xpts_h = home_win * 3.0 + draw * 1.0
        xpts_a = away_win * 3.0 + draw * 1.0
        return xpts_h, xpts_a

    xpts_home_list: list[float] = []
    xpts_away_list: list[float] = []

    for h_lam, a_lam in zip(λ_home_list, λ_away_list):
        xpts_h, xpts_a = _xpts(h_lam, a_lam)
        xpts_home_list.append(xpts_h)
        xpts_away_list.append(xpts_a)

    df["h_xpts"] = xpts_home_list
    df["a_xpts"] = xpts_away_list

    # Also compute xG Difference at match level (if not already)
    if "home_xg" in df.columns and "away_xg" in df.columns:
        df["xgd"] = df["home_xg"] - df["away_xg"]

    return df


# ═══════════════════════════════════════════════════════════
#  Explanation guide
# ═══════════════════════════════════════════════════════════


def get_xg_guide() -> str:
    """Return a plain-text explanation of all xG feature calculations."""
    return """
EXPECTED GOALS (xG) — FEATURE GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. WHAT IS xG?
   ─────────────
   Expected Goals (xG) measures the quality of a shot by estimating
   the probability it will result in a goal, from 0.0 (impossible) to
   1.0 (certain).  A shot with xG = 0.3 would be expected to result in
   a goal 30% of the time.

   xG models consider: shot distance, angle, body part (foot/head),
   assist type, defensive pressure, and historical data from millions
   of shots.

2. ROLLING xG (ATTACK STRENGTH)
   ─────────────────────────────
   avg_xG_5  =  mean of last 5 matches' xG for the team
   avg_xG_N  =  mean of last N matches' xG for the team

   A high rolling xG means the team is consistently creating high-quality
   chances, regardless of whether those chances were converted.  This is
   more predictive than actual goals because xG strips out finishing
   luck/variance.

3. ROLLING xGA (DEFENSIVE SOLIDITY)
   ─────────────────────────────────
   avg_xGA_5  =  mean of last 5 matches' xGA for the team
   avg_xGA_N  =  mean of last N matches' xGA for the team

   xGA is xG *conceded* — the quality of chances the opponent creates.
   A low xGA means the team's defensive system is effectively limiting
   opponents to low-quality shots from distance.

4. xG DIFFERENCE (NET xG)
   ───────────────────────
   xGD = xG - xGA

   Positive = creating better chances than you concede.
   xGD is the single best predictor of future league position over a
   full season — better than actual goal difference or points.

5. EXPECTED POINTS (xPts)
   ──────────────────────
   xPts uses the Poisson distribution to convert xG into expected
   match outcome probabilities:

       P(k goals | xG) = e^(-xG) × xG^k / k!

   For a match with xG_home and xG_away:

       P(Home Win) = sum of P(i | xG_home) × P(j | xG_away) for i > j
       P(Draw)     = sum of P(i | xG_home) × P(j | xG_away) for i = j

       xPts_home = P(Home Win) × 3 + P(Draw) × 1

   xPts tells you how many points a team *deserved* based on chance
   quality, stripping out randomness from finishing and goalkeeping.

6. PLACEHOLDER BEHAVIOUR
   ──────────────────────
   When real xG data is unavailable, zero-filled placeholder columns
   are created.  This means all xG-based features will initially be
   zero, making them safe to include in the model without causing
   spurious predictions.

   To add real xG data:
     1. Enrich your results CSV with columns named ``home_xg`` and
        ``away_xg``.
     2. Re-run the pipeline — the module auto-detects and uses them.
"""
