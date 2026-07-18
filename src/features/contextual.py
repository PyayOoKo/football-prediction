"""
Contextual features — head-to-head stats, league position, competition importance,
extended H2H and form features from the feature framework.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import config as _global_config
from src.dixon_coles import TOURNAMENT_IMPORTANCE as DC_TOURNAMENT_IMPORTANCE

logger = logging.getLogger(__name__)

# Lazy import for extended feature transformers
_EXTENDED_FEATURES_AVAILABLE = False
try:
    from src.feature_framework.features.h2h import H2HTransformer
    from src.feature_framework.features.team_form import TeamFormTransformer

    _EXTENDED_FEATURES_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════
#  1.  Basic Head-to-head statistics
# ═══════════════════════════════════════════════════════════


def _add_h2h_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Append head-to-head features for each fixture pair.

    Features: ``h2h_home_points_avg``, ``h2h_away_points_avg``,
    ``h2h_total_goals_avg``, ``h2h_home_win_rate``,
    ``h2h_home_goals_avg``, ``h2h_away_goals_avg``, ``h2h_matches_played``.
    """
    logger.debug("Adding H2H features with window=%d", window)
    h2h = _compute_h2h_stats(df, window=window)
    df = df.merge(
        h2h.add_prefix("h2h_"),
        left_index=True,
        right_index=True,
        how="left",
    )
    n_features = len([c for c in df.columns if c.startswith("h2h_")])
    logger.debug("Added %d H2H feature columns", n_features)
    return df


def _compute_h2h_stats(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Compute head-to-head rolling stats for every (home_team, away_team) pair.

    **Leakage prevention:**
    - Uses ``rolling(window).mean().shift(1)`` so the current match's data
      is never used to compute its own features.
    - Only the last *window* meetings are included (not full expanding history).
    - Groups by (home_team, away_team) — a given ordered pair, e.g. (A, B).
      Home/away reversals (B vs A) are a separate group.
      The feature describes "when A hosts B, how have those fixtures gone?"

    Parameters
    ----------
    df : pd.DataFrame
        Match data with ``home_team``, ``away_team``, ``date``, ``result``,
        ``home_goals``, ``away_goals`` columns.
    window : int
        Maximum number of previous meetings to include (default 6).
    """
    df_sorted = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"}).copy()
    df_sorted["_sort_key"] = pd.to_datetime(df_sorted["date"])
    df_sorted.sort_values(["_sort_key", "_orig_idx"], inplace=True)
    records: dict[int, dict[str, Any]] = {}

    pair_groups = df_sorted.groupby(["home_team", "away_team"], sort=False)

    for (_home, _away), group in pair_groups:
        group = group.sort_values(["_sort_key", "_orig_idx"])
        res = group["result"]
        hg = group["home_goals"]
        ag = group["away_goals"]

        h_pts = res.map(lambda r: 3 if r == "H" else (1 if r == "D" else 0))
        a_pts = res.map(lambda r: 3 if r == "A" else (1 if r == "D" else 0))

        if window > 0:
            home_points = h_pts.rolling(window, min_periods=1).mean().shift(1)
            away_points = a_pts.rolling(window, min_periods=1).mean().shift(1)
            home_goals_stat = hg.rolling(window, min_periods=1).mean().shift(1)
            away_goals_stat = ag.rolling(window, min_periods=1).mean().shift(1)
            total_goals = (hg + ag).rolling(window, min_periods=1).mean().shift(1)
            home_win = (res == "H").rolling(window, min_periods=1).mean().shift(1)
        else:
            home_points = h_pts.expanding().mean().shift(1)
            away_points = a_pts.expanding().mean().shift(1)
            home_goals_stat = hg.expanding().mean().shift(1)
            away_goals_stat = ag.expanding().mean().shift(1)
            total_goals = (hg + ag).expanding().mean().shift(1)
            home_win = (res == "H").expanding().mean().shift(1)
        n_played = res.expanding().count().shift(1)

        for i, idx in enumerate(group.index):
            records[idx] = {
                "home_points_avg": home_points.iloc[i],
                "away_points_avg": away_points.iloc[i],
                "total_goals_avg": total_goals.iloc[i],
                "home_win_rate": home_win.iloc[i],
                "home_goals_avg": home_goals_stat.iloc[i],
                "away_goals_avg": away_goals_stat.iloc[i],
                "matches_played": n_played.iloc[i],
            }

    result_df = pd.DataFrame.from_dict(records, orient="index")
    result_df.index.name = None
    for drop_col in ["_sort_key", "_orig_idx"]:
        if drop_col in result_df.columns:
            result_df.drop(columns=[drop_col], inplace=True)
    return result_df


# ═══════════════════════════════════════════════════════════
#  2.  League position
# ═══════════════════════════════════════════════════════════


def _add_league_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append league-table position for each team before each match.

    Features: ``h_league_position``, ``a_league_position``, ``h_points_total``,
    ``a_points_total``, ``h_matches_played_league``, ``a_matches_played_league``,
    ``position_diff``.
    """
    logger.debug("Adding league position features")
    league_positions = _compute_league_positions(df)

    home_cols = ["league_position", "points_total", "matches_played_league"]
    home_feats = league_positions[home_cols].add_prefix("h_")
    df = df.merge(home_feats, left_index=True, right_index=True, how="left")

    away_cols = [
        "away_league_position",
        "away_points_total",
        "away_matches_played_league",
    ]
    away_feats = league_positions[away_cols].copy()
    away_feats.columns = [
        "a_league_position",
        "a_points_total",
        "a_matches_played_league",
    ]
    df = df.merge(away_feats, left_index=True, right_index=True, how="left")

    if "h_league_position" in df.columns and "a_league_position" in df.columns:
        df["position_diff"] = (df["h_league_position"] - df["a_league_position"]).abs()

    logger.debug(
        "Added league position features — range %.0f–%.0f",
        df.get("h_league_position", pd.Series()).min(),
        df.get("h_league_position", pd.Series()).max(),
    )
    return df


def _compute_league_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute running league table position for each team before each match.

    **Leakage prevention:**
    - Positions are computed using data from prior matches only.
    - The current match's result is applied AFTER recording the position.
    - Same-date matches are ordered by their original index to ensure
      deterministic results within a single day.
    """
    df_sorted = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"}).copy()
    df_sorted["_sort_key"] = pd.to_datetime(df_sorted["date"])
    df_sorted.sort_values(["_sort_key", "_orig_idx"], inplace=True)
    records: dict[int, dict[str, Any]] = {}

    group_keys = []
    if "season" in df_sorted.columns:
        group_keys.append("season")
    if "league" in df_sorted.columns:
        group_keys.append("league")

    if group_keys:
        groups = df_sorted.groupby(group_keys, sort=False)
    else:
        groups = [("", df_sorted)]

    for _, group in groups:
        group = group.sort_values(["_sort_key", "_orig_idx"])
        team_points: dict[str, float] = {}
        team_gd: dict[str, float] = {}
        team_matches: dict[str, int] = {}
        _get_pts = team_points.get
        _get_gd = team_gd.get
        _get_m = team_matches.get

        for row in group.itertuples(index=True):
            idx = row.Index
            home = row.home_team
            away = row.away_team

            home_pts = _get_pts(home, 0)
            away_pts = _get_pts(away, 0)
            home_m = _get_m(home, 0)
            away_m = _get_m(away, 0)

            all_teams = set(team_points.keys()) | {home, away}
            standings = sorted(
                ((t, _get_pts(t, 0), _get_gd(t, 0)) for t in all_teams),
                key=lambda x: (-x[1], -x[2], x[0]),
            )
            position_map = {t: i + 1 for i, (t, _, _) in enumerate(standings)}

            records[idx] = {
                "league_position": position_map.get(home, 0),
                "points_total": home_pts,
                "matches_played_league": home_m,
                "away_league_position": position_map.get(away, 0),
                "away_points_total": away_pts,
                "away_matches_played_league": away_m,
            }

            result = getattr(row, "result", None)
            hg_val = getattr(row, "home_goals", np.nan)
            hg = int(hg_val) if pd.notna(hg_val) else 0
            ag_val = getattr(row, "away_goals", np.nan)
            ag = int(ag_val) if pd.notna(ag_val) else 0

            if result == "H":
                team_points[home] = _get_pts(home, 0) + 3
                team_points[away] = _get_pts(away, 0)
            elif result == "A":
                team_points[home] = _get_pts(home, 0)
                team_points[away] = _get_pts(away, 0) + 3
            elif result == "D":
                team_points[home] = _get_pts(home, 0) + 1
                team_points[away] = _get_pts(away, 0) + 1
            else:
                team_points[home] = _get_pts(home, 0)
                team_points[away] = _get_pts(away, 0)

            team_gd[home] = _get_gd(home, 0) + (hg - ag)
            team_gd[away] = _get_gd(away, 0) + (ag - hg)
            team_matches[home] = _get_m(home, 0) + 1
            team_matches[away] = _get_m(away, 0) + 1

    result = pd.DataFrame.from_dict(records, orient="index")
    result.index.name = None
    for drop_col in ["_sort_key", "_orig_idx"]:
        if drop_col in result.columns:
            result.drop(columns=[drop_col], inplace=True)
    return result


# ═══════════════════════════════════════════════════════════
#  3.  Competition importance feature
# ═══════════════════════════════════════════════════════════


def _add_competition_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric competition importance feature column.

    Maps the ``league`` column to an importance weight (0.4–2.5)
    using the Dixon-Coles tournament importance map.
    """
    if "league" not in df.columns:
        df["competition_importance"] = 1.0
        return df

    def _importance(league_name: str) -> float:
        if not isinstance(league_name, str):
            return 1.0
        name_lower = league_name.lower().strip()
        for pattern, weight in DC_TOURNAMENT_IMPORTANCE.items():
            if pattern in name_lower:
                return weight
        return 1.0

    df["competition_importance"] = df["league"].apply(_importance)
    logger.debug(
        "Added competition_importance — range [%.1f, %.1f]",
        df["competition_importance"].min(),
        df["competition_importance"].max(),
    )
    return df


# ═══════════════════════════════════════════════════════════
#  4.  Extended H2H (multi-window, multi-context, xG-aware)
# ═══════════════════════════════════════════════════════════


def _add_extended_h2h_features(
    df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Add extended H2H features using the H2HTransformer.

    Provides richer H2H statistics:
    - Multiple windows (3, 5, 10 meetings)
    - Multiple contexts (overall, home, away)
    - More metrics (wins, draws, losses, goals, BTTS, xG)

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    if (
        not cfg.extended_features.enabled
        or not cfg.extended_features.include_extended_h2h
    ):
        return df

    try:
        if not _EXTENDED_FEATURES_AVAILABLE:
            logger.warning(
                "Extended H2H features require the feature framework modules."
            )
            return df

        windows = cfg.extended_features.h2h_windows
        transformer = H2HTransformer(
            windows=list(windows),
            contexts=["overall", "home", "away"],
            include_xg=True,
            sort_by_date=False,
        )
        transformer.init()
        df = transformer.transform(df)

        n_added = len([c for c in df.columns if c.startswith(("h_h2h_", "a_h2h_"))])
        logger.info(
            "Added %d extended H2H feature columns (windows=%s)", n_added, windows
        )

    except Exception as exc:
        logger.error("Failed to compute extended H2H features: %s", exc)

    return df


# ═══════════════════════════════════════════════════════════
#  5.  Extended form features (multi-context, multi-window)
# ═══════════════════════════════════════════════════════════


def _add_extended_form_features(
    df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Add extended team form features using the TeamFormTransformer.

    Provides richer rolling form features than the basic ``_add_rolling_features``:
    - More metrics: points, wins, draws, losses, goals, xG, shots
    - Per-venue context: overall, home, away
    - Multiple windows: 3, 5, 10, 20 matches

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    if (
        not cfg.extended_features.enabled
        or not cfg.extended_features.include_extended_form
    ):
        return df

    try:
        if not _EXTENDED_FEATURES_AVAILABLE:
            logger.warning(
                "Extended form features require the feature framework modules."
            )
            return df

        windows = cfg.extended_features.form_windows
        transformer = TeamFormTransformer(
            windows=list(windows),
            contexts=["overall", "home", "away"],
            include_xg=True,
            include_shots=True,
            include_possession=False,
            include_cards=False,
            league_specific=True,
            sort_by_date=False,
        )
        transformer.init()
        df = transformer.transform(df)

        n_added = len(
            [
                c
                for c in df.columns
                if c.startswith(
                    (
                        "h_overall_",
                        "a_overall_",
                        "h_home_",
                        "a_home_",
                        "h_away_",
                        "a_away_",
                    )
                )
                and "_avg" in c
            ]
        )
        logger.info(
            "Added %d extended form feature columns (windows=%s)", n_added, windows
        )

    except Exception as exc:
        logger.error("Failed to compute extended form features: %s", exc)

    return df
