"""
Rolling team features — form, goals, win rates, attack/defence ratios.

**Leakage prevention:** All features use ``.shift(1)`` so the current match's
data is never used to compute its own features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import config as _global_config
from src.features.helpers import _match_points

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Public entry point for this module
# ═══════════════════════════════════════════════════════════


def _add_rolling_features(
    df: pd.DataFrame,
    window: int,
    extra_windows: tuple[int, ...] = (),
    config: Any | None = None,
) -> pd.DataFrame:
    """Append rolling-average features for each team's recent performance.

    Parameters
    ----------
    window : int
        Primary rolling window size.
    extra_windows : tuple[int, ...]
        Additional window sizes to compute (e.g. (5, 10, 20)).
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).

    Features generated (*_home and *_away variants):
        - ``form_last5`` — points from last 5 matches
        - ``form_lastN`` — points from last *window* matches
        - ``goals_scored_avgN`` / ``goals_conceded_avgN``
        - ``goal_diff_avgN``
        - ``win_rate_home`` / ``win_rate_away`` / ``win_rate_overall``
        - ``days_since_last_match`` / ``matches_this_season``
    """
    cfg = config or _global_config
    windows = tuple(set([w for w in extra_windows if w != window] + [window]))
    logger.debug("Adding rolling features with windows=%s", windows)
    team_stats = _compute_team_stats(df)
    df = _merge_team_stats(df, team_stats, windows, cfg=cfg)
    return df


# ═══════════════════════════════════════════════════════════
#  Team stats computation
# ═══════════════════════════════════════════════════════════


def _compute_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-team per-match DataFrame with raw stats.

    Each match produces two rows: one for the home team's performance in that
    match, and one for the away team.
    """
    has_season = "season" in df.columns
    has_league = "league" in df.columns
    records: list[dict[str, Any]] = []
    append = records.append

    for row in df.itertuples(index=True):
        idx = row.Index
        home = row.home_team
        away = row.away_team
        hg = getattr(row, "home_goals", np.nan)
        ag = getattr(row, "away_goals", np.nan)
        result = getattr(row, "result", None)
        season = getattr(row, "season", None) if has_season else None
        league = getattr(row, "league", None) if has_league else None

        append(
            {
                "team": home,
                "date": row.date,
                "season": season,
                "league": league,
                "opponent": away,
                "goals_scored": hg,
                "goals_conceded": ag,
                "is_home": 1,
                "points": _match_points(result, True),
                "match_id": idx,
            }
        )
        append(
            {
                "team": away,
                "date": row.date,
                "season": season,
                "league": league,
                "opponent": home,
                "goals_scored": ag,
                "goals_conceded": hg,
                "is_home": 0,
                "points": _match_points(result, False),
                "match_id": idx,
            }
        )

    team_df = pd.DataFrame(records)
    # Stable ordering by date then match_id to handle same-day matches deterministically
    team_df.sort_values(["team", "date", "match_id"], inplace=True)
    team_df.reset_index(drop=True, inplace=True)
    return team_df


# ═══════════════════════════════════════════════════════════
#  Merge rolling stats onto original DF
# ═══════════════════════════════════════════════════════════


def _merge_team_stats(
    df: pd.DataFrame,
    team_stats: pd.DataFrame,
    windows: int | tuple[int, ...],
    cfg: Any | None = None,
) -> pd.DataFrame:
    """Compute rolling stats per team and merge home/away variants onto *df*."""
    _cfg = cfg or _global_config
    if isinstance(windows, int):
        windows = (5, windows)
    windows = tuple(sorted(set(windows)))
    n_windows = len(windows)

    team_stats.sort_values(["team", "date"], inplace=True)

    def _rolling_team_features(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("date").copy()
        halflife = getattr(_cfg.features, "time_decay_halflife", None)
        use_ewm = halflife is not None and halflife > 0

        for col, agg_func in [
            ("points", "mean"),
            ("goals_scored", "mean"),
            ("goals_conceded", "mean"),
        ]:
            for w in windows:
                name = _rolling_col_name(col, agg_func, w)
                if use_ewm:
                    grp[name] = (
                        grp[col].ewm(halflife=halflife, min_periods=1).mean().shift(1)
                    )
                else:
                    grp[name] = (
                        grp[col].rolling(w, min_periods=1).agg(agg_func).shift(1)
                    )

        grp["gd"] = grp["goals_scored"] - grp["goals_conceded"]
        for w in windows:
            name = f"goal_diff_avg{w}"
            if use_ewm:
                grp[name] = grp["gd"].ewm(halflife=w, min_periods=1).mean().shift(1)
            else:
                grp[name] = grp["gd"].rolling(w, min_periods=1).mean().shift(1)

        # ── Win rates — correct denominator ─────────────
        # win_rate_home = home wins / home matches (not / all matches)
        home_wins = (
            ((grp["is_home"] == 1) & (grp["points"] == 3)).expanding().sum().shift(1)
        )
        home_matches = (grp["is_home"] == 1).expanding().sum().shift(1)
        grp["win_rate_home"] = (home_wins / home_matches).fillna(0)

        away_wins = (
            ((grp["is_home"] == 0) & (grp["points"] == 3)).expanding().sum().shift(1)
        )
        away_matches = (grp["is_home"] == 0).expanding().sum().shift(1)
        grp["win_rate_away"] = (away_wins / away_matches).fillna(0)

        grp["win_rate_overall"] = (grp["points"] == 3).expanding().mean().shift(1)

        if "season" in grp.columns:
            # cumcount numbers rows within each season group;
            # shift(1) must also be WITHIN each season to avoid
            # leaking match counts across season boundaries.
            grp["matches_this_season"] = grp.groupby("season").cumcount() + 1
            grp["matches_this_season"] = grp.groupby("season")[
                "matches_this_season"
            ].shift(1)

        # ── Days since last match ────────────────────────
        # .diff() already gives days from previous match to THIS match.
        # No extra .shift(1) needed — that would push it one row further,
        # causing match 1 to show NaN instead of the correct gap from match 0.
        grp["days_since_last_match"] = grp["date"].diff().dt.days

        grp["home_matches"] = grp["is_home"].expanding().sum().shift(1)
        grp["away_matches"] = (1 - grp["is_home"]).expanding().sum().shift(1)
        return grp

    team_stats = team_stats.groupby("team", group_keys=False).apply(
        _rolling_team_features
    )

    feat_cols = [
        c
        for c in team_stats.columns
        if c
        not in [
            "match_id",
            "team",
            "date",
            "season",
            "league",
            "opponent",
            "goals_scored",
            "goals_conceded",
            "is_home",
            "points",
            "gd",
        ]
    ]

    home_raw = team_stats[team_stats["is_home"] == 1][["match_id"] + feat_cols].copy()
    home_raw.sort_values("match_id", inplace=True)
    home_raw.drop(columns=["match_id"], inplace=True)
    home_raw.columns = [f"h_{c}" for c in home_raw.columns]
    home_raw.reset_index(drop=True, inplace=True)

    away_raw = team_stats[team_stats["is_home"] == 0][["match_id"] + feat_cols].copy()
    away_raw.sort_values("match_id", inplace=True)
    away_raw.drop(columns=["match_id"], inplace=True)
    away_raw.columns = [f"a_{c}" for c in away_raw.columns]
    away_raw.reset_index(drop=True, inplace=True)

    df = df.reset_index(drop=True)

    # ── Prevent duplicate columns ────────────────────────
    # Schedule features (from _add_schedule_features) may have already added
    # columns with the same name (e.g. ``days_since_last_match``). Drop those
    # from the rolling output to avoid ``pd.concat`` creating duplicate columns.
    dup_home = [c for c in home_raw.columns if c in df.columns]
    dup_away = [c for c in away_raw.columns if c in df.columns]
    if dup_home or dup_away:
        logger.debug(
            "Rolling: dropping %d duplicate home + %d duplicate away "
            "columns to avoid concat collision",
            len(dup_home),
            len(dup_away),
        )
        home_raw.drop(columns=dup_home, inplace=True, errors="ignore")
        away_raw.drop(columns=dup_away, inplace=True, errors="ignore")

    df = pd.concat([df, home_raw, away_raw], axis=1)

    n_features = len([c for c in df.columns if c.startswith(("h_", "a_"))])
    logger.debug(
        "Added %d rolling feature columns across %d windows %s",
        n_features,
        n_windows,
        windows,
    )
    return df


def _rolling_col_name(metric: str, agg: str, window: int | None) -> str:
    """Generate a readable column name for a rolling feature."""
    if agg == "mean":
        return f"{metric}_avg{window or 'all'}"
    return f"{metric}_{agg}{window or 'all'}"


# ═══════════════════════════════════════════════════════════
#  Running league average
# ═══════════════════════════════════════════════════════════


def _add_running_league_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Add running league-average goals using an expanding window (no lookahead)."""
    team_stats = _compute_team_stats(df)
    team_stats.sort_values(["team", "date"], inplace=True)

    def _expanding_avg(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("date").copy()
        grp["goals_scored_cum"] = grp["goals_scored"].expanding().mean().shift(1)
        grp["goals_conceded_cum"] = grp["goals_conceded"].expanding().mean().shift(1)
        return grp

    team_stats = team_stats.groupby("team", group_keys=False).apply(_expanding_avg)

    league_avgs = (
        team_stats.groupby("match_id")
        .agg(
            {
                "goals_scored_cum": "mean",
                "goals_conceded_cum": "mean",
            }
        )
        .rename(
            columns={
                "goals_scored_cum": "league_avg_goals_scored",
                "goals_conceded_cum": "league_avg_goals_conceded",
            }
        )
    )

    df = df.join(league_avgs, how="left")
    for col in ["league_avg_goals_scored", "league_avg_goals_conceded"]:
        if col in df.columns:
            df[col] = df[col].fillna(1.0)
    return df


# ═══════════════════════════════════════════════════════════
#  Attack / defence strength ratios
# ═══════════════════════════════════════════════════════════


def _add_attack_defence_ratios(
    df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Add rolling-window attack/defence strength ratio features.

    For each rolling window already computed in ``_add_rolling_features``,
    generates ``h_attack_ratioN``, ``h_defence_ratioN``, etc.
    A ratio > 1.0 means the team scores/concedes more than average.

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    df = _add_running_league_avg(df)

    league_avg = (
        df["league_avg_goals_scored"].mean() + df["league_avg_goals_conceded"].mean()
    ) / 2.0
    if pd.isna(league_avg) or league_avg <= 0:
        league_avg = 1.0

    windows = getattr(cfg.features, "rolling_windows", (5, 10, 20))

    for w in windows:
        h_scored = f"h_goals_scored_avg{w}"
        h_conceded = f"h_goals_conceded_avg{w}"
        a_scored = f"a_goals_scored_avg{w}"
        a_conceded = f"a_goals_conceded_avg{w}"

        if h_scored in df.columns:
            df[f"h_attack_ratio{w}"] = df[h_scored] / league_avg
        if h_conceded in df.columns:
            df[f"h_defence_ratio{w}"] = df[h_conceded] / league_avg
        if a_scored in df.columns:
            df[f"a_attack_ratio{w}"] = df[a_scored] / league_avg
        if a_conceded in df.columns:
            df[f"a_defence_ratio{w}"] = df[a_conceded] / league_avg

    logger.debug("Added attack/defence ratios for windows %s", windows)
    return df
