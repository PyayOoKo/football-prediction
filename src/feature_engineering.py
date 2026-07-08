"""
Feature Engineering — transform clean match data into predictive features.

**Leakage prevention is the top priority.** Every rolling statistic is
computed with a **shift of 1** — the current match's data is never used to
compute its own features.

Pipeline stages:

1.  **Rolling team features** — form, goals scored/conceded averages, win
    rates, goal difference, days of rest.
2.  **Head-to-head statistics** — points, goals, and form from previous
    direct encounters between the same pair of teams.
3.  **League position** — running league-table rank derived from points
    accumulated before each match.
4.  **Categorical encoding** — team names → numeric feature vectors via
    label encoding, target encoding, or one-hot encoding.

Generated features (``h_`` = home team, ``a_`` = away team, ``h2h_`` = head-to-head):

    ┌──────────────────────────────────────┬─────────────────────────────────┐
    │ Feature column                       │ Description                     │
    ├──────────────────────────────────────┼─────────────────────────────────┤
    │ ``{h,a}_points_avg5``                │ Avg points per match, last 5    │
    │ ``{h,a}_points_avgN``                │ Avg points per match, last N    │
    │ ``{h,a}_goals_scored_avg5``          │ Avg goals scored, last 5        │
    │ ``{h,a}_goals_scored_avgN``          │ Avg goals scored, last N        │
    │ ``{h,a}_goals_conceded_avg5``        │ Avg goals conceded, last 5      │
    │ ``{h,a}_goals_conceded_avgN``        │ Avg goals conceded, last N      │
    │ ``{h,a}_goal_diff_avg5``             │ Avg goal difference, last 5     │
    │ ``{h,a}_goal_diff_avgN``             │ Avg goal difference, last N     │
    │ ``{h,a}_win_rate_home``              │ Proportion of home matches won  │
    │ ``{h,a}_win_rate_away``              │ Proportion of away matches won  │
    │ ``{h,a}_win_rate_overall``           │ Overall win rate (all matches)  │
    │ ``{h,a}_days_since_last_match``      │ Rest days since previous match  │
    │ ``{h,a}_matches_this_season``        │ Matches played so far this season│
    │ ``{h,a}_league_position``            │ League table rank before kickoff│
    │ ``{h,a}_points_total``               │ Total points before this match  │
    │ ``{h,a}_matches_played_league``      │ League matches played so far    │
    │ ``{h,a}_home_matches``               │ Home matches played so far      │
    │ ``{h,a}_away_matches``               │ Away matches played so far      │
    ├──────────────────────────────────────┼─────────────────────────────────┤
    │ ``h2h_home_points_avg``              │ Avg points at home vs opponent  │
    │ ``h2h_away_points_avg``              │ Avg points away vs opponent     │
    │ ``h2h_total_goals_avg``              │ Avg total goals in H2H matches  │
    │ ``h2h_home_win_rate``                │ Proportion of H2H home wins     │
    │ ``h2h_home_goals_avg``               │ Avg H2H goals scored by home    │
    │ ``h2h_away_goals_avg``               │ Avg H2H goals scored by away    │
    │ ``h2h_matches_played``               │ Number of previous H2H matches  │
    ├──────────────────────────────────────┼─────────────────────────────────┤
    │ ``position_diff``                    │ Abs diff in league positions    │
    │ ``home_team`` / ``away_team``        │ Encoded as int (label/onehot)   │
    └──────────────────────────────────────┴─────────────────────────────────┘

Typical usage::

    from src.feature_engineering import build_features, train_val_test_split

    X, y = build_features(df)
    splits = train_val_test_split(X, y)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import config
from src.elo import add_elo_features
from src.poisson_model import PoissonModel
from src.dixon_coles import DixonColesModel, TOURNAMENT_IMPORTANCE as DC_TOURNAMENT_IMPORTANCE
from src.xg_features import add_xg_features
from src.player_info import add_player_features
from src.odds_processing import add_odds_features, add_consensus_features

logger = logging.getLogger(__name__)

# ── Column name constants ───────────────────────────────
_PREFIX_HOME = "h_"
_PREFIX_AWAY = "a_"


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def build_features(
    df: pd.DataFrame,
    is_training: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run the full feature-engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Clean results DataFrame with columns ``date``, ``home_team``,
        ``away_team``, ``result`` or ``target``, ``home_goals``,
        ``away_goals``, and optionally ``season`` / ``league``.
    is_training : bool
        If ``True``, also separate the target column; otherwise assume it is
        already removed (prediction mode).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric only).
    y : pd.Series
        Target vector (0 = Away win, 1 = Draw, 2 = Home win).
    """
    logger.info("Building features on %d rows", len(df))
    df = df.copy()

    # ── Sort chronologically (essential for leakage-free rolling features) ──
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["date", "home_team"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 0a. Elo rating features (pre-match ratings, no leakage)
    # Host nations for World Cup home advantage bonus
    HOST_NATIONS = {2002: "South Korea", 2006: "Germany", 2010: "South Africa",
                    2014: "Brazil", 2018: "Russia", 2022: "Qatar", 2026: "USA"}

    df = add_elo_features(
        df,
        k=config.elo.k,
        home_advantage=config.elo.home_advantage,
        initial_rating=config.elo.initial_rating,
        regress_to_mean=config.elo.regress_to_mean,
        regress_factor=config.elo.regress_factor,
        use_goal_margin=config.elo.use_goal_margin,
        max_goal_margin=config.elo.max_goal_margin,
        home_col="home_team",
        away_col="away_team",
        result_col="result",
        home_goals_col="home_goals",
        away_goals_col="away_goals",
        season_col="season",
        home_xg_col="home_xg" if "home_xg" in df.columns else None,
        away_xg_col="away_xg" if "away_xg" in df.columns else None,
        host_nations=HOST_NATIONS,
    )

    # Apply manual Elo adjustments (e.g. user skepticism about a team)
    if config.elo.adjustments and "Home_Elo" in df.columns:
        for team, penalty in config.elo.adjustments.items():
            home_mask = df["home_team"] == team
            away_mask = df["away_team"] == team
            df.loc[home_mask, "Home_Elo"] -= penalty
            df.loc[away_mask, "Away_Elo"] -= penalty
            n_adjusted = home_mask.sum() + away_mask.sum()
            if n_adjusted > 0:
                logger.info(
                    "Elo adjustment: %s -%d pts (%d rows adjusted)",
                    team, penalty, n_adjusted,
                )
        # Recompute Elo difference after adjustments
        df["Elo_Difference"] = df["Home_Elo"] - df["Away_Elo"]

    # 0b. Odds processing features (opening/closing odds, movement, CLV, consensus)
    df = add_odds_features(
        df,
        opening_odds_cols=config.odds.opening_odds_cols,
        closing_odds_cols=config.odds.closing_odds_cols,
        home_team_col="home_team",
        away_team_col="away_team",
    )
    if config.odds.compute_consensus:
        df = add_consensus_features(df)

    # 0c. Player information features (injuries, squad, rotation — optional)
    if config.player_info.enabled:
        # Load player data from CSV if it exists
        _players_csv = config.paths.external / "players.csv"
        if _players_csv.exists():
            _players_df = pd.read_csv(_players_csv)
            logger.info("Loaded %d player records from %s", len(_players_df), _players_csv)
        else:
            _players_df = None
            logger.warning(
                "Player info enabled but %s not found — using placeholders. "
                "Run: python collect_player_data.py",
                _players_csv,
            )

        # Load lineup data from CSV if it exists
        _lineups_csv = config.paths.external / "lineups.csv"
        if _lineups_csv.exists():
            _lineups_df = pd.read_csv(_lineups_csv)
            logger.info("Loaded %d lineup records from %s", len(_lineups_df), _lineups_csv)
        else:
            _lineups_df = None

        df = add_player_features(
            df,
            players_df=_players_df,
            lineups_df=_lineups_df,
            home_team_col="home_team",
            away_team_col="away_team",
            date_col="date",
        )

    # 0c. xG features (rolling xG, xGA, xGD — uses placeholders if unavailable)
    df = add_xg_features(
        df,
        rolling_windows=config.xg.rolling_windows,
        compute_xpts=config.xg.compute_xpts,
        max_goals_table=config.xg.max_goals_table,
        placeholder_value=config.xg.placeholder_value,
        warn_missing=config.xg.warn_missing,
        home_team_col="home_team",
        away_team_col="away_team",
        home_goals_col="home_goals",
        away_goals_col="away_goals",
    )

    # 0c. Poisson-derived expected goals (leakage-free expanding window)
    _poisson_model = PoissonModel(
        min_matches=config.poisson.min_matches,
        max_goals=config.poisson.max_goals,
    )
    df = _poisson_model.add_poisson_features(df)

    # 0d. Dixon-Coles features (MLE with tau correction, recency, tournament importance)
    if config.dixon_coles.enabled:
        _dc_model = DixonColesModel(
            decay_halflife_days=config.dixon_coles.decay_halflife_days,
            use_importance=config.dixon_coles.use_importance,
            rho_fixed=config.dixon_coles.rho_fixed,
            regress_prior=config.dixon_coles.regress_prior,
            prior_strength=config.dixon_coles.prior_strength,
        )
        df = _dc_model.add_features(df, refit_every=config.dixon_coles.refit_every)

    # 0e. Competition importance as an explicit feature column
    df = _add_competition_importance(df)

    # 1. Rolling team features (form, goals, win rate, GD, rest days) — multiple windows
    windows = config.features.rolling_windows
    df = _add_rolling_features(df, window=config.features.form_window, extra_windows=windows)

    # 2. Head-to-head stats
    if config.features.include_h2h:
        df = _add_h2h_features(df, config.features.h2h_window)

    # 3. League position
    if config.features.include_league_position:
        df = _add_league_position_features(df)

        # 4. Categorical encoding
    df = _encode_categoricals(df)

    # 4b. Attack/defence strength ratios (rolling-window version)
    # Computed from the rolling goals averages that already exist.
    # Unlike PoissonModel's expanding-window strengths, these use
    # fixed rolling windows (last N matches) for more responsive form.
    df = _add_attack_defence_ratios(df)

    # 5. Separate features & target
    cols_to_drop = _get_target_columns(df)
    y: pd.Series
    if is_training and "target" in df.columns:
        y = df["target"].copy()
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    else:
        y = pd.Series(dtype=float)
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns],
                    errors="ignore")

    # Keep only numeric columns for the model
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # Sanitise column names: XGBoost/LightGBM forbid [ ] < > in names
    X.columns = [
        str(c).replace("<", "_lt_").replace(">", "_gt_").replace("[", "_lb_").replace("]", "_rb_")
        for c in X.columns
    ]

    # Drop any fully-NaN feature columns (early-season matches with no history)
    X.dropna(axis=1, how="all", inplace=True)

    logger.info(
        "Feature matrix: %d rows × %d columns  |  target distribution: %s",
        *X.shape,
        y.value_counts(normalize=True).to_dict() if not y.empty else "N/A",
    )
    return X, y


# ═══════════════════════════════════════════════════════════
#  1.  Rolling team features
# ═══════════════════════════════════════════════════════════


def _add_rolling_features(df: pd.DataFrame, window: int, extra_windows: tuple[int, ...] = ()) -> pd.DataFrame:
    """Append rolling-average features for each team's recent performance.

    **Leakage note:** All features are computed with ``.shift(1)`` so the
    current match outcome is never included.

    Parameters
    ----------
    window : int
        Primary rolling window size (used as the configurable N).
    extra_windows : tuple[int, ...]
        Additional window sizes to compute (e.g. (5, 10, 20)).
        The primary window is always included.

    Features generated (*_home and *_away variants):
        - ``form_last5`` — points from last 5 matches (W=3, D=1, L=0)
        - ``form_lastN`` — points from last *window* matches
        - ``goals_scored_avgN`` — average goals scored per match (window)
        - ``goals_conceded_avgN`` — average goals conceded (window)
        - ``goals_scored_avg5`` — average goals scored, last 5
        - ``goals_conceded_avg5`` — average goals conceded, last 5
        - ``goal_diff_avgN`` — avg goal difference (window)
        - ``win_rate_home`` — proportion of *home* matches won
        - ``win_rate_away`` — proportion of *away* matches won
        - ``win_rate_overall`` — overall win rate
        - ``days_since_last_match`` — rest days since team's previous match
        - ``matches_this_season`` — total matches played so far this season
    """
    windows = tuple(set([w for w in extra_windows if w != window] + [window]))
    logger.debug("Adding rolling features with windows=%s", windows)
    team_stats = _compute_team_stats(df)
    df = _merge_team_stats(df, team_stats, windows)
    return df


def _compute_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-team per-match DataFrame with raw stats.

    Each match produces two rows: one for the home team's performance in that
    match, and one for the away team.

    Columns
    -------
    team, date, season, league
    goals_scored, goals_conceded, is_home, points, match_id
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

        append({
            "team": home, "date": row.date, "season": season,
            "league": league, "opponent": away,
            "goals_scored": hg, "goals_conceded": ag,
            "is_home": 1, "points": _match_points(result, True), "match_id": idx,
        })
        append({
            "team": away, "date": row.date, "season": season,
            "league": league, "opponent": home,
            "goals_scored": ag, "goals_conceded": hg,
            "is_home": 0, "points": _match_points(result, False), "match_id": idx,
        })

    team_df = pd.DataFrame(records)
    team_df.sort_values(["team", "date"], inplace=True)
    team_df.reset_index(drop=True, inplace=True)
    return team_df


def _match_points(result: Any, is_home: bool) -> int:
    """Convert match result to points for a given team.

    Parameters
    ----------
    result : str
        ``"H"``, ``"D"``, or ``"A"``.
    is_home : bool
        Whether the team in question was the home team.

    Returns
    -------
    int
        3 for win, 1 for draw, 0 for loss.
    """
    if result == "H":
        return 3 if is_home else 0
    if result == "A":
        return 0 if is_home else 3
    if result == "D":
        return 1
    return 0


def _merge_team_stats(
    df: pd.DataFrame,
    team_stats: pd.DataFrame,
    windows: int | tuple[int, ...],
) -> pd.DataFrame:
    """Compute rolling stats per team and merge home/away variants onto *df*.

    This is the core leakage-free computation.  For each team, we compute
    expanding + rolling windows on historical data, then **shift by 1**
    so the current match is excluded.

    Parameters
    ----------
    windows : int | tuple[int, ...]
        Single window (legacy) or tuple of windows. Each window size
        produces separate rolling averages (e.g. 5, 10, 20).
    """
    if isinstance(windows, int):
        windows = (5, windows)  # legacy: 5-form + N-form
    windows = tuple(sorted(set(windows)))
    n_windows = len(windows)

    # ── Compute per-team rolling aggregates ─────────────
    team_stats.sort_values(["team", "date"], inplace=True)

    def _rolling_team_features(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("date").copy()

        halflife = getattr(config.features, "time_decay_halflife", None)
        use_ewm = halflife is not None and halflife > 0

        # Rolling features for ALL window sizes
        for col, agg_func in [("points", "mean"),
                               ("goals_scored", "mean"),
                               ("goals_conceded", "mean")]:
            for w in windows:
                name = _rolling_col_name(col, agg_func, w)
                if use_ewm:
                    grp[name] = (
                        grp[col].ewm(halflife=halflife, min_periods=1)
                        .mean().shift(1)
                    )
                else:
                    grp[name] = (
                        grp[col].rolling(w, min_periods=1).agg(agg_func).shift(1)
                    )

        # Goal difference (rolling average for each window)
        grp["gd"] = grp["goals_scored"] - grp["goals_conceded"]
        for w in windows:
            name = f"goal_diff_avg{w}"
            if use_ewm:
                grp[name] = grp["gd"].ewm(halflife=w, min_periods=1).mean().shift(1)
            else:
                grp[name] = grp["gd"].rolling(w, min_periods=1).mean().shift(1)

        # Win rates (expanding — all available history, no window)
        grp["win_rate_home"] = (
            (grp["is_home"] == 1) & (grp["points"] == 3)
        ).expanding().mean().shift(1)
        grp["win_rate_away"] = (
            (grp["is_home"] == 0) & (grp["points"] == 3)
        ).expanding().mean().shift(1)
        grp["win_rate_overall"] = (
            grp["points"] == 3
        ).expanding().mean().shift(1)

        # Matches played this season
        if "season" in grp.columns:
            if config.features.reset_per_season:
                # Reset rolling count per season
                season_start = grp.groupby("season").cumcount() == 0
                # For per-season, we still need to track matches within season
                grp["matches_this_season"] = (
                    grp.groupby("season").cumcount() + 1
                ).shift(1)
            else:
                grp["matches_this_season"] = (
                    grp.groupby("season").cumcount() + 1
                ).shift(1)

        # Days since last match
        grp["days_since_last_match"] = (
            grp["date"].diff().dt.days.shift(1)
        )

        # Home match count (for home/away win rate stability)
        grp["home_matches"] = grp["is_home"].expanding().sum().shift(1)
        grp["away_matches"] = (1 - grp["is_home"]).expanding().sum().shift(1)

        return grp

    team_stats = team_stats.groupby("team", group_keys=False).apply(
        _rolling_team_features
    )

    # ── Feature columns list (exclude original metadata) ─
    feat_cols = [
        c for c in team_stats.columns
        if c not in ["match_id", "team", "date", "season", "league",
                      "opponent", "goals_scored", "goals_conceded",
                      "is_home", "points", "gd"]
    ]

    # ── Build home feature columns ──────────────────────
    home_raw = team_stats[team_stats["is_home"] == 1][["match_id"] + feat_cols].copy()
    home_raw.sort_values("match_id", inplace=True)  # restore df row order
    home_raw.drop(columns=["match_id"], inplace=True)
    home_raw.columns = [f"h_{c}" for c in home_raw.columns]
    home_raw.reset_index(drop=True, inplace=True)

    # ── Build away feature columns ───────────────────────
    away_raw = team_stats[team_stats["is_home"] == 0][["match_id"] + feat_cols].copy()
    away_raw.sort_values("match_id", inplace=True)  # restore df row order
    away_raw.drop(columns=["match_id"], inplace=True)
    away_raw.columns = [f"a_{c}" for c in away_raw.columns]
    away_raw.reset_index(drop=True, inplace=True)

    # ── Concatenate all feature columns to df (rows aligned by match_id order) ─
    # Reset index to ensure uniqueness (prior feature steps may have altered it)
    df = df.reset_index(drop=True)
    df = pd.concat([df, home_raw, away_raw], axis=1)

    n_features = len([c for c in df.columns if c.startswith(("h_", "a_"))])
    logger.debug("Added %d rolling feature columns across %d windows %s",
                 n_features, n_windows, windows)
    return df


def _rolling_col_name(metric: str, agg: str, window: int | None) -> str:
    """Generate a readable column name for a rolling feature."""
    if agg == "mean":
        return f"{metric}_avg{window or 'all'}"
    return f"{metric}_{agg}{window or 'all'}"


# ═══════════════════════════════════════════════════════════
#  2.  Head-to-head statistics
# ═══════════════════════════════════════════════════════════


def _add_h2h_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Append head-to-head features for each fixture pair.

    **Leakage note:** Only matches *before* the current match are used.

    Features generated (for home team perspective):
        - ``h2h_home_points_avg`` — avg points per H2H match (home)
        - ``h2h_away_points_avg`` — avg points per H2H match (away)
        - ``h2h_total_goals_avg`` — avg total goals in H2H matches
        - ``h2h_home_win_rate`` — proportion of H2H matches home team won
        - ``h2h_home_goals_avg`` — avg goals scored by home team in H2H
        - ``h2h_away_goals_avg`` — avg goals scored by away team in H2H
        - ``h2h_matches_played`` — number of H2H matches in window
    """
    logger.debug("Adding H2H features with window=%d", window)
    h2h = _compute_h2h_stats(df)

    # Merge onto the original DataFrame
    df = df.merge(
        h2h.add_prefix("h2h_"),
        left_index=True,
        right_index=True,
        how="left",
    )

    n_features = len([c for c in df.columns if c.startswith("h2h_")])
    logger.debug("Added %d H2H feature columns", n_features)
    return df


def _compute_h2h_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head rolling stats for every (home_team, away_team) pair.

    Returns a DataFrame indexed identically to *df*.
    """
    df_sorted = df.sort_values("date").copy()
    records: dict[int, dict[str, Any]] = {}

    # Group by team-pair (direction matters)
    pair_groups = df_sorted.groupby(["home_team", "away_team"], sort=False)

    for (home, away), group in pair_groups:
        group = group.sort_values("date")
        res = group["result"]
        hg = group["home_goals"]
        ag = group["away_goals"]

        h_pts = res.map(lambda r: 3 if r == "H" else (1 if r == "D" else 0))
        a_pts = res.map(lambda r: 3 if r == "A" else (1 if r == "D" else 0))

        home_points = h_pts.expanding().mean().shift(1)
        away_points = a_pts.expanding().mean().shift(1)
        home_goals = hg.expanding().mean().shift(1)
        away_goals = ag.expanding().mean().shift(1)
        total_goals = (hg + ag).expanding().mean().shift(1)
        home_win = (res == "H").expanding().mean().shift(1)
        n_played = res.expanding().count().shift(1)

        for i, idx in enumerate(group.index):
            records[idx] = {
                "home_points_avg": home_points.iloc[i],
                "away_points_avg": away_points.iloc[i],
                "total_goals_avg": total_goals.iloc[i],
                "home_win_rate": home_win.iloc[i],
                "home_goals_avg": home_goals.iloc[i],
                "away_goals_avg": away_goals.iloc[i],
                "matches_played": n_played.iloc[i],
            }

    result_df = pd.DataFrame.from_dict(records, orient="index")
    result_df.index.name = None
    return result_df


# ═══════════════════════════════════════════════════════════
#  3.  League position
# ═══════════════════════════════════════════════════════════


def _add_league_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append league-table position for each team before each match.

    **Leakage note:** The position is computed from all matches *before* the
    current match's date, so it represents the table as it stood when the
    match kicked off.

    Features generated:
        - ``h_league_position`` — home team's position in the table
        - ``a_league_position`` — away team's position
        - ``h_points_total`` — home team's total points before this match
        - ``a_points_total`` — away team's total points
        - ``h_matches_played_league`` — home team's league matches played
        - ``a_matches_played_league`` — away team's league matches played
        - ``position_diff`` — abs(home_position - away_position)
    """
    logger.debug("Adding league position features")
    league_positions = _compute_league_positions(df)

    # Merge home team position (use the main columns which store home data)
    home_cols = [
        "league_position", "points_total", "matches_played_league"
    ]
    home_feats = league_positions[home_cols].add_prefix("h_")
    df = df.merge(home_feats, left_index=True, right_index=True, how="left")

    # Merge away team position (use the dedicated away columns)
    away_cols = [
        "away_league_position", "away_points_total", "away_matches_played_league"
    ]
    away_feats = league_positions[away_cols].copy()
    away_feats.columns = [
        "a_league_position", "a_points_total", "a_matches_played_league"
    ]
    df = df.merge(away_feats, left_index=True, right_index=True, how="left")

    # Position difference
    if "h_league_position" in df.columns and "a_league_position" in df.columns:
        df["position_diff"] = (
            (df["h_league_position"] - df["a_league_position"]).abs()
        )

    logger.debug(
        "Added league position features — range %.0f–%.0f",
        df.get("h_league_position", pd.Series()).min(),
        df.get("h_league_position", pd.Series()).max(),
    )
    return df


def _compute_league_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute running league table position for each team before each match.

    The algorithm:
    1. Sort matches chronologically.
    2. For each match, compute points so far for every team in the same
       season/league using only matches *before* the current one.
    3. Rank teams by points (descending) — this is the position as it stood.

    Returns a DataFrame indexed identically to *df* with columns:
    - ``league_position``, ``points_total``, ``matches_played_league``
      (home team's stats)
    - ``away_league_position``, ``away_points_total``,
      ``away_matches_played_league`` (away team's stats)
    """
    df_sorted = df.sort_values(["date", "home_team"]).copy()
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
        group = group.sort_values("date")
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
    return result


# ═══════════════════════════════════════════════════════════
#  4.  Categorical encoding


# ═══════════════════════════════════════════════════════════
#  4.  Categorical encoding
# ═══════════════════════════════════════════════════════════


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns per ``config.features.categorical_encoding``.

    Strategies:
    - ``"label"`` — simple integer label encoding (fast, memory-efficient)
    - ``"onehot"`` — one-hot encoding (preserves all information)
    - ``"target"`` — target encoding (mean target per category, leakage-free via
      expanding mean shifted by 1)
    """
    cat_cols = ["home_team", "away_team"]
    existing_cats = [c for c in cat_cols if c in df.columns]

    if not existing_cats:
        return df

    strategy = config.features.categorical_encoding
    logger.debug("Encoding categoricals via '%s'", strategy)

    if strategy == "label":
        df = _label_encode(df, existing_cats)
    elif strategy == "onehot":
        df = _onehot_encode(df, existing_cats)
    elif strategy == "target":
        df = _target_encode(df, existing_cats)
    else:
        logger.warning("Unknown encoding strategy '%s' — falling back to label", strategy)
        df = _label_encode(df, existing_cats)

    return df


def _label_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Label-encode categorical columns in place."""
    for col in cols:
        df[col] = df[col].astype("category").cat.codes
    # Also encode opponent columns if they exist from team stats
    for col in ["h_opponent", "a_opponent"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def _onehot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns and drop the originals."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, dtype="int8")
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)
    return df


def _target_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Target-encode categorical columns with expanding mean (shifted to avoid leakage).

    For each category, the encoded value is the mean target value from all
    *previous* occurrences of that category.
    """
    if "target" not in df.columns:
        logger.warning("Target encoding requires 'target' column — falling back to label")
        return _label_encode(df, cols)

    df_sorted = df.sort_values("date").copy()
    for col in cols:
        # Expanding mean of target per category, shifted by 1
        encoded = (
            df_sorted.groupby(col)["target"]
            .expanding()
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_encoded"] = encoded
        # Fill NaN (first occurrence) with global mean
        global_mean = df_sorted["target"].mean()
        df[f"{col}_encoded"].fillna(global_mean, inplace=True)
        df.drop(columns=[col], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


def _get_target_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that should be dropped from the feature matrix."""
    drop_cols = [
        "target",
        "result",
        "home_goals",
        "away_goals",
        "goal_diff",
        "total_goals",
        "date",
        "season",
        "league",
        "source",
        "downloaded_at",
    ]
    # Also drop any match_id auxiliary columns
    drop_cols.extend([c for c in df.columns if c.endswith("_match_id")])
    return drop_cols


# ═══════════════════════════════════════════════════════════
#  0e.  Competition importance feature
# ═══════════════════════════════════════════════════════════


def _add_competition_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a numeric competition importance feature column.

    Reuses the tournament importance map from ``src.dixon_coles``
    to avoid duplication.  Maps the ``league`` column to an importance
    weight (0.4–2.5):
    - World Cup                 → 2.5
    - Continental championships → 2.0
    - Qualifiers                → 1.2–1.5
    - Club competitions         → 1.0
    - Friendlies                → 0.4–0.6

    The Dixon-Coles model uses these weights internally for recency
    weighting, but exposing them as an explicit feature helps tree-based
    models learn competition-specific patterns directly.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with an added ``competition_importance`` column.
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
#  4a.  Attack / Defence strength ratios (rolling-window)
# ═══════════════════════════════════════════════════════════


def _add_attack_defence_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window attack/defence strength ratio features.

    For each rolling window already computed in ``_add_rolling_features``
    (e.g. 5, 10, 20), generates:

    - ``h_attack_ratioN``  = home_goals_scored_avgN / league_avg_goals
    - ``h_defence_ratioN`` = home_goals_conceded_avgN / league_avg_goals
    - ``a_attack_ratioN``  = away_goals_scored_avgN / league_avg_goals
    - ``a_defence_ratioN`` = away_goals_conceded_avgN / league_avg_goals

    A ratio > 1.0 means the team scores/concedes more than the
    tournament average (strong attack / weak defence).
    A ratio < 1.0 means the opposite.

    This differs from PoissonModel's expanding-window strengths in
    that it uses fixed-size rolling windows (most recent N matches),
    making it more responsive to recent form changes.

    **Leakage note:** relies on the (already shifted) rolling averages
    from ``_add_rolling_features``, so no additional leakage risk.
    """
    # Compute league-average goals per match from completed matches only
    completed = df[df["result"].notna()] if "result" in df.columns else df
    if len(completed) == 0:
        return df

    avg_home = float(completed["home_goals"].mean())
    avg_away = float(completed["away_goals"].mean())

    # Guard against division by zero (no goals ever scored)
    league_avg = (avg_home + avg_away) / 2.0
    if league_avg <= 0:
        league_avg = 1.0

    # Check which rolling windows are available
    windows = getattr(config.features, "rolling_windows", (5, 10, 20))

    for w in windows:
        h_scored_col = f"h_goals_scored_avg{w}"
        h_conceded_col = f"h_goals_conceded_avg{w}"
        a_scored_col = f"a_goals_scored_avg{w}"
        a_conceded_col = f"a_goals_conceded_avg{w}"

        if h_scored_col in df.columns:
            df[f"h_attack_ratio{w}"] = df[h_scored_col] / league_avg
        if h_conceded_col in df.columns:
            df[f"h_defence_ratio{w}"] = df[h_conceded_col] / league_avg
        if a_scored_col in df.columns:
            df[f"a_attack_ratio{w}"] = df[a_scored_col] / league_avg
        if a_conceded_col in df.columns:
            df[f"a_defence_ratio{w}"] = df[a_conceded_col] / league_avg

    n_new = len([c for c in df.columns if c.endswith("_ratio5") or c.endswith("_ratio10") or c.endswith("_ratio20")])
    logger.debug("Added %d attack/defence ratio features (league_avg=%.3f)", n_new, league_avg)
    return df


# ═══════════════════════════════════════════════════════════
#  Split
# ═══════════════════════════════════════════════════════════


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratios: tuple[float, float, float] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Split features and target into train / validation / test sets.

    **Important:** The split is chronological (``shuffle=False``) to prevent
    time-series leakage.  The oldest matches go to training, the most recent
    to testing.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    ratios : tuple[float, float, float], optional
        (train, val, test) split ratios.  Defaults to ``config.data.split_ratios``.
    seed : int, optional
        Random seed.  Defaults to ``config.data.seed``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``X_train``, ``X_val``, ``X_test``, ``y_train``,
        ``y_val``, ``y_test``.
    """
    if ratios is None:
        ratios = config.data.split_ratios
    if seed is None:
        seed = config.data.seed

    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # ── Chronological split (NO shuffle) ──────────
    test_ratio = ratios[2]
    val_ratio = ratios[1] / (ratios[0] + ratios[1])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, shuffle=False,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, shuffle=False,
    )

    logger.info(
        "Chronological split — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
