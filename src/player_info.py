"""
Player Information — squad availability, squad characteristics, and rotation.

This module is **optional** — it gracefully handles missing player data by
creating zero-filled or neutral placeholders that can be updated later when
a player-data source is connected.

Features
--------

**Availability features** (require ``players_df`` with availability status):

    ==============================  ===========================================
    Column                          Description
    ==============================  ===========================================
    ``h_injured_count``               Number of injured players in home squad
    ``a_injured_count``               Number of injured players in away squad
    ``h_suspended_count``             Number of suspended players in home squad
    ``a_suspended_count``             Number of suspended players in away squad
    ``h_missing_gk``                 Starting goalkeeper unavailable (binary)
    ``a_missing_gk``                 Starting goalkeeper unavailable (binary)
    ``h_missing_top_scorer``         Top scorer unavailable (binary)
    ``a_missing_top_scorer``         Top scorer unavailable (binary)
    ``h_rotation_index``             Fraction of XI changed from last match
    ``a_rotation_index``             Fraction of XI changed from last match
    ==============================  ===========================================

**Squad characteristic features** (require ``players_df`` with age/value data):

    ==============================  ===========================================
    ``h_avg_age``                     Average age of home squad (years)
    ``a_avg_age``                     Average age of away squad (years)
    ``h_squad_value``                 Total market value of home squad (€m)
    ``a_squad_value``                 Total market value of away squad (€m)
    ==============================  ===========================================

If no player data is provided, all features default to zeros (counts,
flags) or neutral values (age = 25, rotation = 0.0, value = 0.0).

Expected input schemas
----------------------

**players_df** (per-team squad snapshot, one row per player):

    ================= ========  ================================================
    Column            Type      Description
    ================= ========  ================================================
    ``team``          str       Team name (must match match DataFrame)
    ``player_name``   str       Player name
    ``position``      str       ``GK``, ``DEF``, ``MID``, ``FWD``
    ``age``           float     Player age in years (optional)
    ``market_value``  float     Market value in €m (optional)
    ``is_starter``    bool      Part of preferred starting XI (optional)
    ``injured``       bool      Currently injured (optional, default False)
    ``suspended``     bool      Currently suspended (optional, default False)
    ``goals_scored``  int       Season goals so far (optional, for top scorer)
    ================= ========  ================================================

**lineups_df** (actual starting XIs for matchweeks, one row per player per match):

    ================= ========  ================================================
    Column            Type      Description
    ================= ========  ================================================
    ``team``          str       Team name
    ``date``          str/date  Match date
    ``player_name``   str       Player in the starting XI
    ================= ========  ================================================

Usage
-----
::

    # Standalone — auto-placeholder mode (no data needed)
    from src.player_info import add_player_features
    df = add_player_features(df)

    # With real player data
    df = add_player_features(
        df,
        players_df=pd.read_csv("data/external/players.csv"),
        lineups_df=pd.read_csv("data/external/lineups.csv"),
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default neutral values for placeholders ────────────
_DEFAULT_AGE = 25.0
_DEFAULT_VALUE = 0.0
_DEFAULT_ROTATION = 0.0

# Columns that indicate a player's availability status
_AVAIL_COLS = ["injured", "suspended", "is_starter", "position", "goals_scored"]


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def add_player_features(
    df: pd.DataFrame,
    players_df: pd.DataFrame | None = None,
    lineups_df: pd.DataFrame | None = None,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    date_col: str = "date",
    team_col: str = "team",
) -> pd.DataFrame:
    """Add player information features to a match DataFrame.

    This function checks for the presence of player data and computes:

    - **Injury/suspension counts** — from ``players_df['injured']`` and
      ``players_df['suspended']`` columns.
    - **Missing key players** — starting goalkeeper or top scorer unavailable.
    - **Squad age & market value** — from ``players_df['age']`` and
      ``players_df['market_value']``.
    - **Rotation index** — from ``lineups_df`` (changes in starting XI
      between consecutive matches).

    If any player data is missing, the corresponding features are filled
    with neutral placeholder values and a warning is logged.

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame (must be sorted by date for rotation index).
    players_df : pd.DataFrame, optional
        Per-team squad roster with player attributes (see module docstring
        for expected schema).
    lineups_df : pd.DataFrame, optional
        Actual starting XI per match per team (see module docstring).
    home_team_col, away_team_col : str
        Column names for team names in **df**.
    date_col : str
        Column name for match date in **df**.
    team_col : str
        Column name for team name in **players_df** / **lineups_df**.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with player feature columns added (prefixed ``h_``
        for home, ``a_`` for away).
    """
    df = df.copy()
    logger.info("Adding player information features on %d rows", len(df))

    has_players = players_df is not None and not players_df.empty
    has_lineups = lineups_df is not None and not lineups_df.empty

    # ── Determine which features we can compute ───────────
    has_avail = (
        _has_columns(players_df, ["injured", "suspended"]) if has_players else False
    )
    has_gk = _has_columns(players_df, ["position"]) if has_players else False
    has_scorer = _has_columns(players_df, ["goals_scored"]) if has_players else False
    has_age = _has_columns(players_df, ["age"]) if has_players else False
    has_value = _has_columns(players_df, ["market_value"]) if has_players else False

    if not has_players:
        logger.warning(
            "No players_df provided — "
            "injury/suspension/missing-player/age/value features will be "
            "filled with neutral placeholders (injured=0, suspended=0, "
            "age=%.0f, value=0.0, rotation=0.0).",
            _DEFAULT_AGE,
        )

    # ── Build per-match per-team feature dicts ────────────
    # We'll collect features for each (match_index, team) pair
    home_features: dict[int, dict[str, float]] = {}
    away_features: dict[int, dict[str, float]] = {}

    # Pre-process player squad data
    squad_info = (
        _build_squad_info(
            players_df,
            has_avail,
            has_gk,
            has_scorer,
            has_age,
            has_value,
            team_col,
        )
        if has_players
        else {}
    )

    # Pre-process rotation data
    rotation_info = (
        _build_rotation_info(
            lineups_df,
            date_col,
            team_col,
        )
        if has_lineups
        else {}
    )

    has_rotation = bool(rotation_info)

    for idx, row in df.iterrows():
        home_team = row[home_team_col]
        away_team = row[away_team_col]
        match_date = row.get(date_col, pd.NaT)

        # ── Home team features ──
        home_f: dict[str, float] = {}
        if has_players and home_team in squad_info:
            sq = squad_info[home_team]
            home_f["injured_count"] = float(sq["injured_count"])
            home_f["suspended_count"] = float(sq["suspended_count"])
            home_f["missing_gk"] = 1.0 if sq["missing_gk"] else 0.0
            home_f["missing_top_scorer"] = 1.0 if sq["missing_top_scorer"] else 0.0
            home_f["avg_age"] = sq["avg_age"]
            home_f["squad_value"] = sq["squad_value"]
        else:
            home_f["injured_count"] = 0.0
            home_f["suspended_count"] = 0.0
            home_f["missing_gk"] = 0.0
            home_f["missing_top_scorer"] = 0.0
            home_f["avg_age"] = _DEFAULT_AGE
            home_f["squad_value"] = _DEFAULT_VALUE

        # Rotation index (needs lineup data)
        if has_rotation and home_team in rotation_info:
            home_f["rotation_index"] = _get_rotation(
                rotation_info[home_team],
                match_date,
            )
        else:
            home_f["rotation_index"] = _DEFAULT_ROTATION

        home_features[idx] = home_f

        # ── Away team features (mirror) ──
        away_f: dict[str, float] = {}
        if has_players and away_team in squad_info:
            sq = squad_info[away_team]
            away_f["injured_count"] = float(sq["injured_count"])
            away_f["suspended_count"] = float(sq["suspended_count"])
            away_f["missing_gk"] = 1.0 if sq["missing_gk"] else 0.0
            away_f["missing_top_scorer"] = 1.0 if sq["missing_top_scorer"] else 0.0
            away_f["avg_age"] = sq["avg_age"]
            away_f["squad_value"] = sq["squad_value"]
        else:
            away_f["injured_count"] = 0.0
            away_f["suspended_count"] = 0.0
            away_f["missing_gk"] = 0.0
            away_f["missing_top_scorer"] = 0.0
            away_f["avg_age"] = _DEFAULT_AGE
            away_f["squad_value"] = _DEFAULT_VALUE

        if has_rotation and away_team in rotation_info:
            away_f["rotation_index"] = _get_rotation(
                rotation_info[away_team],
                match_date,
            )
        else:
            away_f["rotation_index"] = _DEFAULT_ROTATION

        away_features[idx] = away_f

    # ── Merge features onto DataFrame ─────────────────────
    _merge_features(df, home_features, "h_")
    _merge_features(df, away_features, "a_")

    source = (
        "real player data" if has_players else "placeholders (no player data available)"
    )
    logger.info(
        "Player info features added — %s columns from %s "
        "(injuries=%s, gk=%s, scorer=%s, age=%s, value=%s, rotation=%s)",
        len(home_features.get(0, {})),
        source,
        has_avail,
        has_gk,
        has_scorer,
        has_age,
        has_value,
        has_rotation,
    )

    return df


# ═══════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════


def _has_columns(df: pd.DataFrame | None, cols: list[str]) -> bool:
    """Check if a DataFrame has all the specified columns (case-insensitive)."""
    if df is None or df.empty:
        return False
    df_lower = {c.lower(): c for c in df.columns}
    return all(col.lower() in df_lower for col in cols)


def _build_squad_info(
    players_df: pd.DataFrame,
    has_avail: bool,
    has_gk: bool,
    has_scorer: bool,
    has_age: bool,
    has_value: bool,
    team_col: str = "team",
) -> dict[str, dict[str, Any]]:
    """Build per-team squad info from the players DataFrame.

    Returns a dict: ``{team_name: {injured_count, suspended_count, missing_gk,
    missing_top_scorer, avg_age, squad_value}}``
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
            "is_starter",
            "injured",
            "suspended",
            "goals_scored",
        ):
            col_map[c] = cl
    df.rename(columns=col_map, inplace=True)

    # Ensure availability columns exist with defaults
    if "injured" not in df.columns:
        df["injured"] = False
    if "suspended" not in df.columns:
        df["suspended"] = False
    if "is_starter" not in df.columns:
        df["is_starter"] = False
    if "goals_scored" not in df.columns:
        df["goals_scored"] = 0
    if "age" not in df.columns:
        df["age"] = _DEFAULT_AGE
    if "market_value" not in df.columns:
        df["market_value"] = _DEFAULT_VALUE

    squad_info: dict[str, dict[str, Any]] = {}

    for team, squad in df.groupby("team"):
        total = len(squad)
        if total == 0:
            continue

        injured = int(squad["injured"].sum()) if has_avail else 0
        suspended = int(squad["suspended"].sum()) if has_avail else 0
        avg_age = float(squad["age"].mean()) if has_age else _DEFAULT_AGE
        squad_value = (
            float(squad["market_value"].sum()) if has_value else _DEFAULT_VALUE
        )

        # Missing starting goalkeeper
        missing_gk = False
        if has_gk:
            gk_squad = squad[squad["position"].str.upper().str.contains("GK", na=False)]
            starters = gk_squad[gk_squad["is_starter"]]
            if not starters.empty:
                missing_gk = bool(
                    starters["injured"].any() or starters["suspended"].any()
                )
            elif not gk_squad.empty:
                # No designated starter — check if any GK is available
                available_gk = gk_squad[~gk_squad["injured"] & ~gk_squad["suspended"]]
                missing_gk = len(available_gk) == 0

        # Missing top scorer
        missing_top_scorer = False
        if has_scorer:
            # Top scorer = player with most goals this season
            top_scorer_idx = squad["goals_scored"].idxmax()
            top_scorer = squad.loc[top_scorer_idx]
            missing_top_scorer = bool(top_scorer["injured"] or top_scorer["suspended"])

        squad_info[team] = {
            "injured_count": injured,
            "suspended_count": suspended,
            "missing_gk": missing_gk,
            "missing_top_scorer": missing_top_scorer,
            "avg_age": round(avg_age, 1),
            "squad_value": round(squad_value, 1),
        }

    return squad_info


def _build_rotation_info(
    lineups_df: pd.DataFrame | None,
    date_col: str = "date",
    team_col: str = "team",
) -> dict[str, list[tuple[pd.Timestamp, set[str]]]]:
    """Build per-team chronological lineup history for rotation computation.

    Returns: ``{team: [(date, {player_names}), ...]}`` sorted by date.
    """
    if lineups_df is None or lineups_df.empty:
        return {}

    df = lineups_df.copy()

    # Normalise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("team", "date", "player_name"):
            col_map[c] = cl
    df.rename(columns=col_map, inplace=True)

    if "date" not in df.columns or "player_name" not in df.columns:
        logger.warning("lineups_df missing required columns 'date'/'player_name'")
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    rotation_info: dict[str, list[tuple[pd.Timestamp, set[str]]]] = {}

    for team, grp in df.groupby(team_col):
        # Collect unique players per match date
        match_rosters: dict[pd.Timestamp, set[str]] = {}
        for _, row in grp.iterrows():
            d = row["date"]
            if pd.isna(d):
                continue
            player = str(row["player_name"])
            if d not in match_rosters:
                match_rosters[d] = set()
            match_rosters[d].add(player)

        # Sort by date
        sorted_rosters = sorted(match_rosters.items(), key=lambda x: x[0])
        rotation_info[team] = sorted_rosters

    return rotation_info


def _get_rotation(
    lineup_history: list[tuple[pd.Timestamp, set[str]]],
    match_date: Any,
) -> float:
    """Compute rotation index for a team on a given match date.

    Formula
    -------
    rotation = changes_from_last_match / 11

    Where ``changes_from_last_match`` counts the number of players in the
    starting XI who were NOT in the previous match's starting XI. A value
    of 0.0 means an unchanged XI, 1.0 means a completely different XI.

    Returns
    -------
    float
        Rotation index between 0.0 and 1.0. Returns 0.0 if no previous
        lineup data is available for comparison.
    """
    if not lineup_history:
        return _DEFAULT_ROTATION

    match_dt = (
        pd.Timestamp(match_date)
        if not isinstance(match_date, pd.Timestamp)
        else match_date
    )
    if pd.isna(match_dt):
        return _DEFAULT_ROTATION

    # Find current match's lineup and the previous match's lineup
    current_players: set[str] | None = None
    previous_players: set[str] | None = None

    for date, players in lineup_history:
        if date <= match_dt:
            previous_players = current_players
            current_players = players
        else:
            break

    if current_players is None:
        return _DEFAULT_ROTATION

    if previous_players is None:
        # First match of the season — no comparison possible
        return _DEFAULT_ROTATION

    changes = len(current_players - previous_players)
    rotation = changes / 11.0
    return min(rotation, 1.0)


def _merge_features(
    df: pd.DataFrame,
    features: dict[int, dict[str, float]],
    prefix: str,
) -> None:
    """Merge feature dicts into the DataFrame with a prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame (modified in place).
    features : dict[int, dict[str, float]]
        ``{row_index: {feature_name: value}}``
    prefix : str
        Column prefix (``"h_"`` or ``"a_"``).
    """
    for idx, feat_dict in features.items():
        for feat_name, value in feat_dict.items():
            col = f"{prefix}{feat_name}"
            df.loc[idx, col] = value


# ═══════════════════════════════════════════════════════════
#  Explanation guide
# ═══════════════════════════════════════════════════════════


def get_player_info_guide() -> str:
    """Return a plain-text explanation of all player-information features."""
    return """
PLAYER INFORMATION — FEATURE GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. INJURED COUNT
   ──────────────
   The number of players currently injured in a team's squad.
   More injured players → weaker squad depth → lower expected performance.
   Typical range: 0–8 (varies by squad size and training load).

2. SUSPENDED COUNT
   ────────────────
   The number of players serving match bans (usually from yellow/red card
   accumulation or disciplinary issues).
   Differs from injuries in that suspension durations are fixed and known
   in advance.

3. MISSING STARTING GOALKEEPER
   ────────────────────────────
   Binary flag (0/1) indicating whether the team's primary goalkeeper
   (the ``is_starter = True`` GK) is unavailable due to injury or suspension.
   Losing the starting GK often forces a backup into the lineup and can
   significantly affect defensive organisation.

4. MISSING TOP SCORER
   ───────────────────
   Binary flag (0/1) indicating whether the team's top goalscorer is
   unavailable.  The top scorer is identified as the squad member with
   the highest ``goals_scored`` value.
   Losing the top scorer reduces goal-scoring threat, especially if the
   backup has significantly fewer goals.

5. AVERAGE SQUAD AGE
   ──────────────────
   The mean age (in years) of all players in the squad.
     - Younger squads (23–25): higher energy, worse decision-making
     - Older squads (27–29): more experience, lower stamina
     - Extreme values (<22 or >30) are rare but can indicate
       academy-heavy or ageing squads

6. SQUAD MARKET VALUE
   ───────────────────
   The total estimated market value of the squad in millions of euros (€m).
   Higher-valued squads are generally stronger (strong correlation with
   league position), but the relationship is not perfectly linear.
   Values vary enormously by league: Premier League squads often exceed
   €500m while lower-league squads may be under €10m.

7. ROTATION INDEX
   ───────────────
   The fraction of starting XI players who were NOT in the previous
   match's starting XI.

       rotation = number_of_changes / 11

   Range 0.0–1.0:
     - 0.00 = unchanged lineup from last match
     - 0.18 = 2 changes (common for most managers)
     - 0.45 = 5 changes (heavy rotation, e.g. before European matches)
     - 1.00 = completely different XI

   High rotation often signals:
     • An upcoming important match (manager resting key players)
     • A cup competition squad rotation
     • Multiple injuries forcing changes
     • A manager who favours squad rotation

   Low rotation signals a settled first XI with consistent partnerships.

8. PLACEHOLDER BEHAVIOUR
   ──────────────────────
   When no ``players_df`` or ``lineups_df`` is provided, all features
   default to neutral values:
     - Injury/suspension counts → 0
     - Missing GK/top scorer → 0 (not missing)
     - Average age → 25.0 years
     - Squad value → 0.0
     - Rotation index → 0.0

   To add real player data, provide a DataFrame with the expected schema
   documented at the top of this module.
"""
