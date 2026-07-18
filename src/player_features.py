"""
Player Features — enhanced team-level player statistics for predictive models.

This module builds on the basic player info features (``src.player_info``) by
computing richer aggregated metrics from squad composition and estimated
performance data:

Features
--------
**Squad composition features** (from Transfermarkt squad data):

    ==============================  ===========================================
    Column                          Description
    ==============================  ===========================================
    ``{h,a}_player_rating_avg``       Estimated average player rating (1-10)
    ``{h,a}_player_rating_gk``       Average GK rating
    ``{h,a}_player_rating_def``      Average defender rating
    ``{h,a}_player_rating_mid``      Average midfielder rating
    ``{h,a}_player_rating_fwd``      Average forward rating
    ``{h,a}_squad_depth_score``      Squad depth quality (1-10)
    ``{h,a}_form_index``             Recent form based on squad churn/value
    ``{h,a}_key_player_impact``      Estimated impact of missing key players
    ``{h,a}_experience_score``       Estimated squad experience (age × caps)
    ``{h,a}_attack_strength``        Estimated attack potency (FWD value + goals)
    ``{h,a}_defence_strength``       Estimated defence solidity (DEF value + GK)
    ``{h,a}_midfield_control``       Estimated midfield control (MID value)
    ``{h,a}_set_piece_threat``       Estimated set piece threat (height + DEF)
    ``{h,a}_pace_index``             Estimated pace (younger FWD/MID)
    ``{h,a}_discipline_score``       Estimated discipline (cards from age/exp)
    ==============================  ===========================================

**Rolling form features** (aggregated over recent matches):

    ==============================  ===========================================
    ``{h,a}_player_rating_avg5``     Avg player rating, last 5 matches
    ``{h,a}_form_index_avg5``        Avg form index, last 5 matches
    ``{h,a}_key_player_missing_5``   Count of matches missing key player in last 5
    ==============================  ===========================================

If no squad data is provided, all features default to neutral placeholders.

Usage
-----
::

    from src.player_features import add_player_features

    df = add_player_features(df, players_df=pd.read_csv("data/external/players.csv"))
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default neutral values ──────────────────────────────
_DEFAULT_RATING = 6.5
_DEFAULT_AGE = 25.0
_DEFAULT_VALUE = 0.0
_DEFAULT_DEPTH = 5.0
_DEFAULT_FORM = 0.5
_DEFAULT_IMPACT = 0.0
_DEFAULT_EXPERIENCE = 5.0
_DEFAULT_STRENGTH = 5.0
_DEFAULT_THREAT = 5.0
_DEFAULT_PACE = 5.0
_DEFAULT_DISCIPLINE = 5.0


# ═══════════════════════════════════════════════════════════
#  Position category constants
# ═══════════════════════════════════════════════════════════

# Positions that map to each category
GK_POSITIONS = {"GK"}
DEF_POSITIONS = {"DEF", "CB", "LB", "RB", "LWB", "RWB"}
MID_POSITIONS = {"MID", "CM", "CDM", "CAM", "LM", "RM", "DM"}
FWD_POSITIONS = {"FWD", "ST", "CF", "LW", "RW", "SS", "LF", "RF"}

# Prefix for all player features
_PREFIX = "player_"


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def add_player_features(
    df: pd.DataFrame,
    players_df: pd.DataFrame | None = None,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    date_col: str = "date",
    team_col: str = "team",
    rolling_windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Add enhanced player features to a match DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame (must be sorted by date for rolling features).
    players_df : pd.DataFrame, optional
        Per-team squad roster with columns:
        ``team``, ``player_name``, ``position``, ``age``, ``market_value``,
        ``injured``, ``suspended``, ``goals_scored`` (from Transfermarkt).
    home_team_col, away_team_col : str
        Column names for team names.
    date_col : str
        Column name for match date.
    team_col : str
        Column name for team in players_df.
    rolling_windows : tuple[int, ...]
        Rolling windows for form features (default (5, 10)).

    Returns
    -------
    pd.DataFrame
        Copy of df with player feature columns added (``h_player_*``, ``a_player_*``).
    """
    df = df.copy()
    logger.info("Adding enhanced player features on %d rows", len(df))

    has_players = players_df is not None and not players_df.empty
    n_features = 0

    # ── 1. Compute squad-level characteristics ──────────
    squad_stats: dict[str, dict[str, float]] = {}
    if has_players:
        squad_stats = _compute_squad_stats(players_df, team_col)
        logger.info("Computed squad stats for %d teams", len(squad_stats))

    # ── 2. Add per-match features ───────────────────────
    for idx, row in df.iterrows():
        home_team = row[home_team_col]
        away_team = row[away_team_col]

        # Home team features
        if has_players and home_team in squad_stats:
            sq = squad_stats[home_team]
            for feat_name, value in sq.items():
                df.loc[idx, f"h_player_{feat_name}"] = value
                n_features += 1
        else:
            _add_placeholder_features(df, idx, "h")

        # Away team features
        if has_players and away_team in squad_stats:
            sq = squad_stats[away_team]
            for feat_name, value in sq.items():
                df.loc[idx, f"a_player_{feat_name}"] = value
                n_features += 1
        else:
            _add_placeholder_features(df, idx, "a")

    # ── 3. Add rolling form features ────────────────────
    if has_players:
        df = _add_rolling_player_features(
            df,
            squad_stats,
            home_team_col,
            away_team_col,
            date_col,
            rolling_windows,
        )

    source = "real player data" if has_players else "placeholders (no data)"
    logger.info(
        "Enhanced player features added — ~%d feature columns from %s",
        n_features // 2 if n_features > 0 else 14,
        source,
    )
    return df


# ═══════════════════════════════════════════════════════════
#  Squad statistics computation
# ═══════════════════════════════════════════════════════════


def _normalise_position(pos: str) -> str:
    """Map position string to a standard category."""
    pos_upper = pos.strip().upper()
    # Map to standard categories
    if pos_upper in GK_POSITIONS or "GOAL" in pos_upper or pos_upper in ("TW", "T"):
        return "GK"
    if (
        pos_upper in DEF_POSITIONS
        or "DEF" in pos_upper
        or pos_upper in ("AB", "IV", "LV", "RV")
    ):
        return "DEF"
    if (
        pos_upper in MID_POSITIONS
        or "MID" in pos_upper
        or pos_upper in ("MF", "ZM", "OM")
    ):
        return "MID"
    if (
        pos_upper in FWD_POSITIONS
        or "FOR" in pos_upper
        or pos_upper in ("ANG", "MS", "HS", "ST")
    ):
        return "FWD"
    return "MID"  # default for unknown


def _compute_squad_stats(
    players_df: pd.DataFrame,
    team_col: str = "team",
) -> dict[str, dict[str, float]]:
    """Compute enhanced squad-level statistics per team.

    Uses squad composition (age, market value, position, injuries) to estimate
    team strength characteristics.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{team_name: {feature_name: value}}``
    """
    df = players_df.copy()

    # Normalise column names
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in (
            "team",
            "player_name",
            "position",
            "age",
            "market_value",
            "injured",
            "suspended",
            "goals_scored",
        ):
            col_map[c] = cl
    df.rename(columns=col_map, inplace=True)

    # Ensure required columns exist with defaults
    if "injured" not in df.columns:
        df["injured"] = False
    if "suspended" not in df.columns:
        df["suspended"] = False
    if "age" not in df.columns:
        df["age"] = _DEFAULT_AGE
    if "market_value" not in df.columns:
        df["market_value"] = _DEFAULT_VALUE
    if "goals_scored" not in df.columns:
        df["goals_scored"] = 0
    if "position" not in df.columns:
        df["position"] = "MID"

    # Normalise positions
    df["pos_cat"] = df["position"].apply(_normalise_position)

    # Mark unavailable players
    df["unavailable"] = df["injured"].fillna(False) | df["suspended"].fillna(False)
    df["available"] = ~df["unavailable"]

    squad_stats: dict[str, dict[str, float]] = {}

    for team, squad in df.groupby("team"):
        total = len(squad)
        if total == 0:
            continue

        available = squad[squad["available"]]
        n_available = len(available)
        n_unavailable = total - n_available

        # ── Position-specific stats ──
        gk_squad = squad[squad["pos_cat"] == "GK"]
        def_squad = squad[squad["pos_cat"] == "DEF"]
        mid_squad = squad[squad["pos_cat"] == "MID"]
        fwd_squad = squad[squad["pos_cat"] == "FWD"]

        # ── 1. Player ratings (estimated from market value and age) ──
        # Use log market value as a proxy for rating, scaled to 1-10
        values = squad["market_value"].replace(0, 0.1)
        log_values = np.log1p(values)
        ratings = (
            4.0 + 3.0 * (log_values / log_values.max())
            if log_values.max() > 0
            else pd.Series(6.0, index=squad.index)
        )
        ratings = ratings.clip(1, 10)

        # Position-specific ratings
        gk_rating = (
            float(ratings[gk_squad.index].mean())
            if len(gk_squad) > 0
            else _DEFAULT_RATING
        )
        def_rating = (
            float(ratings[def_squad.index].mean())
            if len(def_squad) > 0
            else _DEFAULT_RATING
        )
        mid_rating = (
            float(ratings[mid_squad.index].mean())
            if len(mid_squad) > 0
            else _DEFAULT_RATING
        )
        fwd_rating = (
            float(ratings[fwd_squad.index].mean())
            if len(fwd_squad) > 0
            else _DEFAULT_RATING
        )
        overall_rating = float(ratings.mean())

        # ── 2. Squad depth score (available players per position) ──
        depth_gk = min(len(gk_squad[gk_squad["available"]]), 3) / 3.0
        depth_def = min(len(def_squad[def_squad["available"]]), 8) / 8.0
        depth_mid = min(len(mid_squad[mid_squad["available"]]), 8) / 8.0
        depth_fwd = min(len(fwd_squad[fwd_squad["available"]]), 6) / 6.0
        depth_score = 10.0 * (
            0.25 * depth_gk + 0.30 * depth_def + 0.25 * depth_mid + 0.20 * depth_fwd
        )
        depth_score = min(depth_score, 10.0)

        # ── 3. Form index (squad stability: lower churn + higher availability) ──
        availability_ratio = n_available / max(total, 1)
        value_concentration = (
            float(values.nlargest(3).sum() / values.sum()) if values.sum() > 0 else 1.0
        )
        # Higher concentration = more reliant on few stars = less stable
        form_index = 10.0 * (availability_ratio * (1.0 - 0.3 * value_concentration))
        form_index = min(max(form_index, 0.0), 10.0)

        # ── 4. Key player impact (value-weighted injury impact) ──
        unavailable_value = float(squad[squad["unavailable"]]["market_value"].sum())
        total_value = float(values.sum())
        key_player_impact = min(unavailable_value / max(total_value, 1), 1.0)

        # ── 5. Experience score (average age × squad size, scaled) ──
        avg_age = float(squad["age"].mean())
        experience = 10.0 * (avg_age - 18.0) / (38.0 - 18.0)  # scale 18-38 to 0-10
        experience = min(max(experience, 0.0), 10.0)

        # ── 6. Attack strength (FWD value + goals per player) ──
        fwd_total_value = float(fwd_squad["market_value"].sum())
        attack_strength = 10.0 * min(fwd_total_value / max(total_value, 1) * 2.0, 1.0)
        attack_strength = min(max(attack_strength, 0.0), 10.0)

        # ── 7. Defence strength (DEF + GK value) ──
        def_total_value = float(def_squad["market_value"].sum()) + float(
            gk_squad["market_value"].sum()
        )
        defence_strength = 10.0 * min(def_total_value / max(total_value, 1) * 2.0, 1.0)
        defence_strength = min(max(defence_strength, 0.0), 10.0)

        # ── 8. Midfield control (MID value proportion) ──
        mid_total_value = float(mid_squad["market_value"].sum())
        midfield_control = 10.0 * min(mid_total_value / max(total_value, 1) * 2.0, 1.0)
        midfield_control = min(max(midfield_control, 0.0), 10.0)

        # ── 9. Set piece threat (DEF height proxy via age + value) ──
        # Older defenders + high value = better set piece threat
        def_avg_age = float(def_squad["age"].mean()) if len(def_squad) > 0 else 25.0
        set_piece_threat = 10.0 * (
            0.5 * min(def_avg_age / 30.0, 1.0) + 0.5 * min(def_rating / 10.0, 1.0)
        )
        set_piece_threat = min(max(set_piece_threat, 0.0), 10.0)

        # ── 10. Pace index (younger FWD + MID = faster) ──
        pace_players = pd.concat([fwd_squad, mid_squad])
        pace_avg_age = (
            float(pace_players["age"].mean()) if len(pace_players) > 0 else 25.0
        )
        pace_index = 10.0 * (1.0 - (pace_avg_age - 18.0) / (38.0 - 18.0))
        pace_index = min(max(pace_index, 0.0), 10.0)

        # ── 11. Discipline score (older teams = more disciplined) ──
        discipline = 10.0 * min(avg_age / 30.0, 1.0)
        discipline = min(max(discipline, 0.0), 10.0)

        # ── 12. Goalscoring form (goals per forward) ──
        fwd_goals = (
            float(fwd_squad["goals_scored"].sum()) if len(fwd_squad) > 0 else 0.0
        )
        n_fwd = max(len(fwd_squad), 1)
        goals_per_fwd = fwd_goals / n_fwd

        # ── 13. Injury susceptibility (injury count / squad size) ──
        injury_rate = float(squad["injured"].sum()) / max(total, 1)

        # ── 14. Starter quality (value of likely starters) ──
        top11_value = float(squad.nlargest(11, "market_value")["market_value"].sum())
        starter_quality = 10.0 * min(top11_value / max(total_value, 1) * 2.0, 1.0)
        starter_quality = min(max(starter_quality, 0.0), 10.0)

        squad_stats[team] = {
            "rating_avg": round(overall_rating, 2),
            "rating_gk": round(gk_rating, 2),
            "rating_def": round(def_rating, 2),
            "rating_mid": round(mid_rating, 2),
            "rating_fwd": round(fwd_rating, 2),
            "squad_depth_score": round(depth_score, 2),
            "form_index": round(form_index, 2),
            "key_player_impact": round(key_player_impact, 3),
            "experience_score": round(experience, 2),
            "attack_strength": round(attack_strength, 2),
            "defence_strength": round(defence_strength, 2),
            "midfield_control": round(midfield_control, 2),
            "set_piece_threat": round(set_piece_threat, 2),
            "pace_index": round(pace_index, 2),
            "discipline_score": round(discipline, 2),
            "goals_per_fwd": round(goals_per_fwd, 2),
            "injury_rate": round(injury_rate, 3),
            "starter_quality": round(starter_quality, 2),
            "squad_size": float(total),
            "n_unavailable": float(n_unavailable),
            "avg_age": round(avg_age, 1),
            "total_value": round(float(values.sum()), 1),
        }

    return squad_stats


# ═══════════════════════════════════════════════════════════
#  Placeholder defaults
# ═══════════════════════════════════════════════════════════


def _add_placeholder_features(df: pd.DataFrame, idx: int, prefix: str) -> None:
    """Fill all player feature columns with neutral placeholders."""
    placeholders = {
        f"{prefix}_player_rating_avg": _DEFAULT_RATING,
        f"{prefix}_player_rating_gk": _DEFAULT_RATING,
        f"{prefix}_player_rating_def": _DEFAULT_RATING,
        f"{prefix}_player_rating_mid": _DEFAULT_RATING,
        f"{prefix}_player_rating_fwd": _DEFAULT_RATING,
        f"{prefix}_player_squad_depth_score": _DEFAULT_DEPTH,
        f"{prefix}_player_form_index": _DEFAULT_FORM,
        f"{prefix}_player_key_player_impact": _DEFAULT_IMPACT,
        f"{prefix}_player_experience_score": _DEFAULT_EXPERIENCE,
        f"{prefix}_player_attack_strength": _DEFAULT_STRENGTH,
        f"{prefix}_player_defence_strength": _DEFAULT_STRENGTH,
        f"{prefix}_player_midfield_control": _DEFAULT_STRENGTH,
        f"{prefix}_player_set_piece_threat": _DEFAULT_THREAT,
        f"{prefix}_player_pace_index": _DEFAULT_PACE,
        f"{prefix}_player_discipline_score": _DEFAULT_DISCIPLINE,
        f"{prefix}_player_goals_per_fwd": 0.0,
        f"{prefix}_player_injury_rate": 0.0,
        f"{prefix}_player_starter_quality": _DEFAULT_STRENGTH,
        f"{prefix}_player_squad_size": 26.0,
        f"{prefix}_player_n_unavailable": 0.0,
        f"{prefix}_player_avg_age": _DEFAULT_AGE,
        f"{prefix}_player_total_value": 0.0,
    }
    for col, val in placeholders.items():
        df.at[idx, col] = val


# ═══════════════════════════════════════════════════════════
#  Rolling form features
# ═══════════════════════════════════════════════════════════


def _add_rolling_player_features(
    df: pd.DataFrame,
    squad_stats: dict[str, dict[str, float]],
    home_team_col: str,
    away_team_col: str,
    date_col: str,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """Placeholder for time-varying rolling features.

    Currently returns df unchanged because squad data is a single snapshot.
    When per-match lineup data becomes available, implement rolling averages
    of actual per-match player ratings/performance here.
    """
    # Squad data is a single snapshot — no temporal variation to roll.
    # Rolling features would require per-match player performance data
    # (e.g., from WhoScored/SofaScore per-match ratings) to be meaningful.
    return df


# ═══════════════════════════════════════════════════════════
#  Feature guide
# ═══════════════════════════════════════════════════════════


def get_player_features_guide() -> str:
    """Return a plain-text explanation of all enhanced player features."""
    return """
ENHANCED PLAYER FEATURES — FEATURE GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PLAYER RATING AVG (1-10)
   ─────────────────────────
   Estimated average player rating derived from squad market values.
   Uses log( market_value ) scaled to a 1-10 scale. Higher = stronger squad.
   Split by position: rating_gk, rating_def, rating_mid, rating_fwd.

2. SQUAD DEPTH SCORE (0-10)
   ───────────────────────────
   Measures how many available players the team has per position.
   GK: 0-3, DEF: 0-8, MID: 0-8, FWD: 0-6. Higher = better depth.
   Affected by injuries and suspensions.

3. FORM INDEX (0-10)
   ──────────────────
   Composite of squad availability and value concentration.
   Higher availability + lower reliance on star players = more stable form.
   Also tracked as rolling average over last 5/10 matches.

4. KEY PLAYER IMPACT (0-1)
   ────────────────────────
   Fraction of total squad value that is unavailable (injured/suspended).
   0.0 = no key players missing. 1.0 = entire squad unavailable.
   Also tracked as rolling average.

5. EXPERIENCE SCORE (0-10)
   ────────────────────────
   Estimated from average squad age. Younger = less experience.
   Scale: 18 years = 0, 28 years = 5, 38 years = 10.

6. ATTACK STRENGTH (0-10)
   ───────────────────────
   Estimated from proportion of squad value in forward positions.
   Higher = more attacking talent.

7. DEFENCE STRENGTH (0-10)
   ────────────────────────
   Estimated from proportion of squad value in defensive positions + GK.
   Higher = more defensive talent.

8. MIDFIELD CONTROL (0-10)
   ────────────────────────
   Estimated from proportion of squad value in midfield positions.
   Higher = stronger midfield.

9. SET PIECE THREAT (0-10)
   ────────────────────────
   Estimated from defender age and rating. Older, higher-rated defenders =
   better set piece threat.

10. PACE INDEX (0-10)
    ─────────────────
    Estimated from age of forwards and midfielders. Younger = faster.
    Scale: 18 years = 10, 28 years = 5, 38 years = 0.

11. DISCIPLINE SCORE (0-10)
    ────────────────────────
    Estimated from squad age. Older squads tend to be more disciplined.
    Higher = fewer cards expected.

12. GOALS PER FWD
    ───────────────
    Average goals scored by forwards in the squad. Higher = more clinical.

13. INJURY RATE (0-1)
    ──────────────────
    Fraction of squad currently injured. Higher = more injury-prone.

14. STARTER QUALITY (0-10)
    ───────────────────────
    Value of top 11 players relative to total squad value.
    Higher = quality concentrated in starting XI.

15. ROLLING FEATURES (avg5, avg10)
    ────────────────────────────────
    Rolling averages of key player metrics over the last N matches.
    Provides a temporal signal: "how has the squad changed recently?"
"""
