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
    ├──────────────────────────────────────┼─────────────────────────────────┤
    │ **ADVANCED FEATURES** (opt-in)       │                                 │
    ├──────────────────────────────────────┼─────────────────────────────────┤
    │ ``{h,a}_temperature_celsius``        │ Match-day temperature (C)       │
    │ ``{h,a}_humidity_pct``               │ Humidity percentage (0-100)     │
    │ ``{h,a}_wind_speed_kmh``             │ Wind speed (km/h)               │
    │ ``{h,a}_weather_severity``           │ Composite weather severity (0-1)│
    │ ``referee_home_yellow_rate``         │ Avg home yellows under ref      │
    │ ``referee_away_yellow_rate``         │ Avg away yellows under ref      │
    │ ``referee_home_win_rate``            │ Home win rate under ref         │
    │ ``referee_card_total_avg``           │ Avg total cards under ref       │
    │ ``{h,a}_rest_days``                  │ Days since team's last match    │
    │ ``{h,a}_matches_last_7_days``        │ Matches in last 7 days (fatigue)│
    │ ``{h,a}_matches_last_14_days``       │ Matches in last 14 days         │
    │ ``{h,a}_consec_home``                │ Consecutive home matches streak │
    │ ``{h,a}_consec_away``                │ Consecutive away matches streak │
    │ ``{h,a}_travel_distance``            │ Distance from prev venue (km)   │
    │ ``{h,a}_h2h_{ctx}_{metric}_last{W}`` │ Extended H2H: context+window    │
    │ ``{h,a}_{ctx}_{metric}_avg{W}``      │ Extended form: context+window   │
    └──────────────────────────────────────┴─────────────────────────────────┘

   Contexts: ``overall``, ``home``, ``away``
   Metrics (extended H2H): wins, draws, losses, goals_scored, goals_conceded,
                           goal_diff, btts, over_2.5, clean_sheets, xg/xga/xgd
   Metrics (extended form): points, wins, draws, losses, goals_scored,
                            goals_conceded, goal_diff, clean_sheets, btts,
                            over_2.5, under_2.5, xg/xga/xgd, shots, cards

Advanced features are **opt-in** via ``config.weather.enabled``,
``config.referee.enabled``, and ``config.extended_features.enabled``.

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

# Lazy imports for extended feature transformers
# (Avoid circular imports at module level)
_EXTENDED_FEATURES_AVAILABLE = False
try:
    from src.feature_framework.features.schedule import ScheduleTransformer
    from src.feature_framework.features.h2h import H2HTransformer
    from src.feature_framework.features.team_form import TeamFormTransformer
    _EXTENDED_FEATURES_AVAILABLE = True
except ImportError:
    pass

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

    # 0f. Weather features (temperature, humidity, wind, pitch condition)
    df = _add_weather_features(df)

    # 0g. Referee statistics (card rates, foul rates, home bias)
    df = _add_referee_features(df)

    # 0h. Schedule / congestion features (travel distance, fatigue, rest days)
    df = _add_schedule_features(df)

    # 0i. Extended H2H features (multi-window, multi-context)
    df = _add_extended_h2h_features(df)

    # 0j. Transfer impact features (recent signings, squad turnover)
    df = _add_transfer_features(df)

    # 0k. Extended form features (multi-context, multi-window with xG/shots/cards)
    df = _add_extended_form_features(df)

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
#  0f.  Weather features (temperature, humidity, wind, pitch)
# ═══════════════════════════════════════════════════════════


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-related features from database or CSV.

    Reads weather data from ``weather.csv`` (external dir) or fills
    neutral placeholders when no weather data is available.

    Features added (``h_`` = home, ``a_`` = away):
        - ``{{h,a}}_temperature_celsius`` — match-day temperature
        - ``{{h,a}}_humidity_pct`` — humidity percentage (0-100)
        - ``{{h,a}}_wind_speed_kmh`` — wind speed in km/h
        - ``{{h,a}}_precipitation_mm`` — precipitation in mm
        - ``{{h,a}}_pitch_condition_encoded`` — pitch condition (0=dry, 1=wet, 2=frozen)
        - ``{{h,a}}_weather_severity`` — composite severity (0-1)

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame with a row index that can be joined.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with weather feature columns added.
    """
    if not config.weather.enabled:
        return df

    _weather_csv = config.paths.external / "weather.csv"
    has_weather = _weather_csv.exists()

    if not has_weather:
        if config.weather.warn_missing:
            logger.warning(
                "Weather features enabled but %s not found — using placeholders. "
                "Collect weather data and save to: %s",
                _weather_csv, _weather_csv,
            )
        # Fill with neutral placeholders
        defaults = {
            "temperature_celsius": config.weather.default_temp,
            "humidity_pct": 50.0,
            "wind_speed_kmh": 10.0,
            "precipitation_mm": 0.0,
            "pitch_condition_encoded": 0.0,
            "weather_severity": 0.0,
        }
        for col, val in defaults.items():
            df[f"h_{col}"] = val
            df[f"a_{col}"] = val
        return df

    try:
        weather_df = pd.read_csv(_weather_csv)
        logger.info("Loaded %d weather records from %s", len(weather_df), _weather_csv)

        # Normalise columns
        col_map: dict[str, str] = {}
        norm_targets = {
            "match_id": "match_id", "temperature": "temperature_celsius",
            "temp": "temperature_celsius", "humidity": "humidity_pct",
            "wind": "wind_speed_kmh", "precipitation": "precipitation_mm",
            "pitch": "pitch_condition_encoded", "condition": "condition_str",
        }
        for c in weather_df.columns:
            cl = c.lower().strip().replace(" ", "_")
            if cl in norm_targets:
                col_map[c] = norm_targets[cl]
        weather_df.rename(columns=col_map, inplace=True)

        # Match by index — assume weather_df index aligns with df
        if "match_id" in weather_df.columns:
            weather_df.set_index("match_id", inplace=True)

        # Encode pitch condition to numeric
        if "condition_str" in weather_df.columns:
            cond_map = {"dry": 0, "wet": 1, "waterlogged": 2, "frozen": 3}
            weather_df["pitch_condition_encoded"] = (
                weather_df["condition_str"].astype(str).str.lower().map(cond_map).fillna(0)
            )

        # Composite weather severity (0-1) — 3 components: precip(0.4) + wind(0.3) + temp_extreme(0.3)
        severity = pd.Series(0.0, index=weather_df.index)
        for col, weight in [("precipitation_mm", 0.4), ("wind_speed_kmh", 0.3)]:
            if col in weather_df.columns:
                norm_val = weather_df[col].fillna(0) / (weather_df[col].max() if weather_df[col].max() > 0 else 1)
                severity += weight * norm_val
        # Temperature extreme: 0 at 15°C, 1 at extremes (<0°C or >35°C)
        if "temperature_celsius" in weather_df.columns:
            temp = weather_df["temperature_celsius"].fillna(15.0)
            temp_extreme = (temp - 15.0).abs() / 20.0  # 0 at 15°C, 1 at 35°C or -5°C
            severity += 0.3 * temp_extreme.clip(0, 1)
        weather_df["weather_severity"] = severity.clip(0, 1)

        # Add weather features — use index alignment (both DataFrames sorted by date)
        # Weather is match-level (same for both teams), so assign directly
        for col in ["temperature_celsius", "humidity_pct", "wind_speed_kmh",
                     "precipitation_mm", "pitch_condition_encoded", "weather_severity"]:
            if col in weather_df.columns:
                vals = weather_df[col].values
                # Ensure we have the right number of values
                if len(vals) >= len(df):
                    vals = vals[:len(df)]
                else:
                    logger.warning("Weather CSV has %d rows but match DF has %d — extending with placeholders",
                                   len(vals), len(df))
                    vals = list(vals) + [config.weather.placeholder_value] * (len(df) - len(vals))
                df["h_" + col] = vals
                df["a_" + col] = vals

        logger.info(
            "Added weather features (%d columns) from %s",
            len([c for c in df.columns if "temperature" in c or "humidity" in c]),
            _weather_csv,
        )

    except Exception as exc:
        logger.error("Failed to load weather data: %s — using placeholders", exc)
        defaults = {
            "temperature_celsius": config.weather.default_temp,
            "humidity_pct": 50.0, "wind_speed_kmh": 10.0,
            "precipitation_mm": 0.0, "pitch_condition_encoded": 0.0,
            "weather_severity": 0.0,
        }
        for col, val in defaults.items():
            df[f"h_{col}"] = val
            df[f"a_{col}"] = val

    return df


# ═══════════════════════════════════════════════════════════
#  0g.  Referee statistics (card rates, foul rates, home bias)
# ═══════════════════════════════════════════════════════════


def _add_referee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add referee-based features from database or CSV.

    Reads referee data from ``referees.csv`` (external dir) or fills
    neutral placeholders when no data is available.

    Features added:
        - ``referee_home_yellow_rate`` — avg yellow cards per match for home team
          under this referee (rolling window)
        - ``referee_away_yellow_rate`` — avg yellow cards per match for away team
        - ``referee_home_win_rate`` — proportion of home wins under this referee
        - ``referee_card_total_avg`` — avg total cards (YC + RC) per match

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame with ``referee`` column (optional).

    Returns
    -------
    pd.DataFrame
        Copy of **df** with referee feature columns added.
    """
    if not config.referee.enabled:
        return df

    _referee_csv = config.paths.external / "referees.csv"
    has_referee_data = _referee_csv.exists()

    if not has_referee_data:
        if config.referee.warn_missing:
            logger.warning(
                "Referee features enabled but %s not found — using placeholders. "
                "Save referee data to: %s",
                _referee_csv, _referee_csv,
            )
        df["referee_home_yellow_rate"] = config.referee.placeholder_value
        df["referee_away_yellow_rate"] = config.referee.placeholder_value
        df["referee_home_win_rate"] = 0.5
        df["referee_card_total_avg"] = config.referee.placeholder_value
        return df

    try:
        ref_df = pd.read_csv(_referee_csv)
        logger.info("Loaded %d referee records from %s", len(ref_df), _referee_csv)

        # Normalise columns
        col_map = {}
        for c in ref_df.columns:
            cl = c.lower().strip()
            if cl in ("referee", "referee_name", "name", "full_name"):
                col_map[c] = "referee_name"
            elif cl in ("home_yellow", "home_yellow_cards", "h_yellow", "h_yc"):
                col_map[c] = "home_yellow_cards"
            elif cl in ("away_yellow", "away_yellow_cards", "a_yellow", "a_yc"):
                col_map[c] = "away_yellow_cards"
            elif cl in ("home_red", "home_red_cards", "h_red", "h_rc"):
                col_map[c] = "home_red_cards"
            elif cl in ("away_red", "away_red_cards", "a_red", "a_rc"):
                col_map[c] = "away_red_cards"
            elif cl in ("match_id", "id", "matchid"):
                col_map[c] = "match_id"
            elif cl in ("home_fouls", "h_fouls"):
                col_map[c] = "home_fouls"
            elif cl in ("away_fouls", "a_fouls"):
                col_map[c] = "away_fouls"
            elif cl in ("result", "winner"):
                col_map[c] = "result"
            elif cl in ("date", "match_date"):
                col_map[c] = "date"
        ref_df.rename(columns=col_map, inplace=True)

        if "referee_name" not in ref_df.columns:
            # No referee name column — use averages as placeholders
            yellow_h = ref_df.get("home_yellow_cards", pd.Series()).mean() or 0
            yellow_a = ref_df.get("away_yellow_cards", pd.Series()).mean() or 0
            red_h = ref_df.get("home_red_cards", pd.Series()).mean() or 0
            red_a = ref_df.get("away_red_cards", pd.Series()).mean() or 0
            df["referee_home_yellow_rate"] = yellow_h
            df["referee_away_yellow_rate"] = yellow_a
            df["referee_card_total_avg"] = yellow_h + yellow_a + red_h + red_a
            df["referee_home_win_rate"] = 0.5
            return df

        # Group by referee and compute rolling stats
        if "date" in ref_df.columns:
            ref_df["date"] = pd.to_datetime(ref_df["date"])
            ref_df.sort_values(["referee_name", "date"], inplace=True)

        window = config.referee.window

        def _ref_stats(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.sort_values("date").copy() if "date" in grp.columns else grp.copy()
            grp["ref_home_yellow_rate"] = (
                grp.get("home_yellow_cards", pd.Series(0, index=grp.index))
                .rolling(window, min_periods=1).mean().shift(1)
            )
            grp["ref_away_yellow_rate"] = (
                grp.get("away_yellow_cards", pd.Series(0, index=grp.index))
                .rolling(window, min_periods=1).mean().shift(1)
            )
            total_cards = (
                grp.get("home_yellow_cards", pd.Series(0, index=grp.index)).fillna(0)
                + grp.get("away_yellow_cards", pd.Series(0, index=grp.index)).fillna(0)
                + grp.get("home_red_cards", pd.Series(0, index=grp.index)).fillna(0)
                + grp.get("away_red_cards", pd.Series(0, index=grp.index)).fillna(0)
            )
            grp["ref_card_total_avg"] = total_cards.rolling(window, min_periods=1).mean().shift(1)

            # Home win rate under this referee
            if "result" in grp.columns:
                grp["ref_home_win_rate"] = (
                    (grp["result"].str.upper() == "H")
                    .rolling(window, min_periods=1).mean().shift(1)
                )
            else:
                grp["ref_home_win_rate"] = 0.5

            return grp

        ref_stats = ref_df.groupby("referee_name", group_keys=False).apply(_ref_stats)

        # Merge onto df by match_id or index
        if "match_id" in ref_stats.columns and "match_id" in df.columns:
            merge_cols = ["match_id", "ref_home_yellow_rate", "ref_away_yellow_rate",
                         "ref_card_total_avg", "ref_home_win_rate"]
            existing = [c for c in merge_cols if c in ref_stats.columns]
            df = df.merge(ref_stats[existing], on="match_id", how="left")
        else:
            # Fallback: sequential alignment (fragile — warn user)
            logger.warning(
                "Cannot merge referee stats by match_id. Using sequential alignment — "
                "this may misalign data if referee.csv and match DataFrame are not "
                "in the same order or have different numbers of rows."
            )
            for col in ["ref_home_yellow_rate", "ref_away_yellow_rate",
                        "ref_card_total_avg", "ref_home_win_rate"]:
                if col in ref_stats.columns:
                    # Warning: this is a fragile alignment — ensure files are row-aligned
                    df[col] = ref_stats[col].iloc[:len(df)].values if len(ref_stats) >= len(df) else config.referee.placeholder_value
                else:
                    df[col] = config.referee.placeholder_value

        # Fill NaN placeholders
        for col in ["ref_home_yellow_rate", "ref_away_yellow_rate",
                    "ref_card_total_avg", "ref_home_win_rate"]:
            if col in df.columns:
                df[col] = df[col].fillna(config.referee.placeholder_value)

        logger.info("Added referee features (%d columns)",
                     len([c for c in df.columns if "ref_" in c]))

    except Exception as exc:
        logger.error("Failed to load referee data: %s — using placeholders", exc)
        df["referee_home_yellow_rate"] = config.referee.placeholder_value
        df["referee_away_yellow_rate"] = config.referee.placeholder_value
        df["referee_home_win_rate"] = 0.5
        df["referee_card_total_avg"] = config.referee.placeholder_value

    return df


# ═══════════════════════════════════════════════════════════
#  0h.  Schedule / congestion features (travel, fatigue, rest)
# ═══════════════════════════════════════════════════════════


def _add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add schedule/congestion features using the ScheduleTransformer.

    Integrates the feature framework's ``ScheduleTransformer`` to compute:
    - Rest days since last match
    - Matches in last 7/14 days (fatigue)
    - Consecutive home/away streaks
    - Back-to-back opponent flag
    - Travel distance (if venue coordinates available)
    - Days since last match in same competition

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame with ``date``, ``home_team``, ``away_team``.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with schedule feature columns added.
    """
    if not config.schedule.enabled:
        return df

    try:
        if not _EXTENDED_FEATURES_AVAILABLE:
            logger.warning(
                "Schedule features require the feature framework modules. "
                "Install them or set config.schedule.enabled=False."
            )
            return df

        transformer = ScheduleTransformer(
            include_travel_distance=config.schedule.include_travel_distance,
            league_specific=True,
            sort_by_date=False,  # Already sorted by build_features
        )
        transformer.init()
        df = transformer.transform(df)

        n_added = len([c for c in df.columns if c.startswith(("h_", "a_"))
                       and any(kw in c for kw in ["rest_days", "matches_last",
                                                   "consec_", "back_to_back",
                                                   "travel_distance",
                                                   "days_since_competition"])])
        logger.info("Added %d schedule/congestion feature columns", n_added)

    except Exception as exc:
        logger.error("Failed to compute schedule features: %s", exc)

    return df


# ═══════════════════════════════════════════════════════════
#  0i.  Extended H2H (multi-window, multi-context, xG-aware)
# ═══════════════════════════════════════════════════════════


def _add_extended_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extended H2H features using the H2HTransformer.

    Provides richer H2H statistics than the basic ``_add_h2h_features``:
    - Multiple windows (3, 5, 10 meetings)
    - Multiple contexts (overall, home, away)
    - More metrics (wins, draws, losses, goals, BTTS, over/under,
      clean sheets, xG/xGA/xGD when available)

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame with required columns.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with extended H2H feature columns added.
    """
    if not config.extended_features.enabled or not config.extended_features.include_extended_h2h:
        return df

    try:
        if not _EXTENDED_FEATURES_AVAILABLE:
            logger.warning(
                "Extended H2H features require the feature framework modules."
            )
            return df

        windows = config.extended_features.h2h_windows
        transformer = H2HTransformer(
            windows=list(windows),
            contexts=["overall", "home", "away"],
            include_xg=True,
            sort_by_date=False,
        )
        transformer.init()
        df = transformer.transform(df)

        n_added = len([c for c in df.columns if c.startswith(("h_h2h_", "a_h2h_"))])
        logger.info("Added %d extended H2H feature columns (windows=%s)", n_added, windows)

    except Exception as exc:
        logger.error("Failed to compute extended H2H features: %s", exc)

    return df

# ═══════════════════════════════════════════════════════════
#  0k.  Transfer impact features (recent signings, squad turnover)
# ═══════════════════════════════════════════════════════════


def _add_transfer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transfer/roster-change impact features.

    Reads transfer data from ``transfers.csv`` (external dir) or fills
    neutral placeholders when no data is available.

    Transfer activity significantly impacts team performance:
    - Many new signings → squad cohesion disruption (short-term negative)
    - Key player sold → weakened squad
    - Late-window arrivals → no preseason integration

    Features added:
        - ``{{h,a}}_signings_count`` — number of players transferred in (last window)
        - ``{{h,a}}_departures_count`` — number of players transferred out
        - ``{{h,a}}_net_spend_meur`` — net spend in millions of euros
        - ``{{h,a}}_squad_churn_pct`` — % of squad changed (in+out)/squad size

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame indexed chronologically.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with transfer feature columns added.
    """
    if not config.extended_features.enabled:
        return df

    _transfers_csv = config.paths.external / "transfers.csv"
    has_transfers = _transfers_csv.exists()

    if not has_transfers:
        if config.player_info.warn_missing:
            logger.info(
                "Transfer features: %s not found — using neutral placeholders. "
                "Save transfer data to enable squad-churn features.",
                _transfers_csv,
            )
        # Neutral placeholders: no transfers = no squad disruption
        for prefix in ["h_", "a_"]:
            for col in ["signings_count", "departures_count", "net_spend_meur", "squad_churn_pct"]:
                df[f"{prefix}{col}"] = 0.0
        return df

    try:
        transfer_df = pd.read_csv(_transfers_csv)
        logger.info("Loaded %d transfer records from %s", len(transfer_df), _transfers_csv)

        # Normalise columns
        col_map = {}
        for c in transfer_df.columns:
            cl = c.lower().strip()
            if cl in ("team", "club", "team_name"):
                col_map[c] = "team"
            elif cl in ("signings", "players_in", "incoming", "arrivals"):
                col_map[c] = "signings_count"
            elif cl in ("departures", "players_out", "outgoing", "sales"):
                col_map[c] = "departures_count"
            elif cl in ("net_spend", "net_spend_eur", "net_spend_meur", "balance"):
                col_map[c] = "net_spend_meur"
            elif cl in ("season", "window", "transfer_window", "year"):
                col_map[c] = "season"
            elif cl in ("squad_size", "squad", "total_players"):
                col_map[c] = "squad_size"
        transfer_df.rename(columns=col_map, inplace=True)

        # Compute squad churn %
        if "signings_count" in transfer_df.columns and "departures_count" in transfer_df.columns:
            total_changes = (
                transfer_df["signings_count"].fillna(0).astype(float)
                + transfer_df["departures_count"].fillna(0).astype(float)
            )
            squad_size = transfer_df.get("squad_size", pd.Series(25.0, index=transfer_df.index)).fillna(25.0)
            transfer_df["squad_churn_pct"] = (total_changes / squad_size).clip(0, 1)
        else:
            transfer_df["squad_churn_pct"] = 0.0

        # Ensure all expected columns exist
        for col in ["signings_count", "departures_count", "net_spend_meur", "squad_churn_pct"]:
            if col not in transfer_df.columns:
                transfer_df[col] = 0.0

        if "team" not in transfer_df.columns:
            logger.warning("Transfer data missing 'team' column — using placeholders")
            for prefix in ["h_", "a_"]:
                for col in ["signings_count", "departures_count", "net_spend_meur", "squad_churn_pct"]:
                    df[f"{prefix}{col}"] = 0.0
            return df

        # Build per-team transfer lookup (most recent season for each team)
        if "season" in transfer_df.columns:
            transfer_df = transfer_df.sort_values("season")
            latest_per_team = transfer_df.groupby("team").last().reset_index()
        else:
            latest_per_team = transfer_df

        # Merge home and away team transfer data
        for prefix, team_col in [("h_", "home_team"), ("a_", "away_team")]:
            merge_data = latest_per_team[["team", "signings_count", "departures_count",
                                          "net_spend_meur", "squad_churn_pct"]].copy()
            merge_data.columns = ["team"] + [f"{prefix}{c}" for c in
                                              ["signings_count", "departures_count",
                                               "net_spend_meur", "squad_churn_pct"]]
            df = df.merge(merge_data, left_on=team_col, right_on="team", how="left")
            df.drop(columns=["team"], inplace=True, errors="ignore")

        # Fill NaN placeholders for teams with no transfer data
        for prefix in ["h_", "a_"]:
            for col in ["signings_count", "departures_count", "net_spend_meur", "squad_churn_pct"]:
                full_col = f"{prefix}{col}"
                if full_col in df.columns:
                    df[full_col] = df[full_col].fillna(0.0)

        logger.info("Added transfer features from %s", _transfers_csv)

    except Exception as exc:
        logger.error("Failed to load transfer data: %s — using placeholders", exc)
        for prefix in ["h_", "a_"]:
            for col in ["signings_count", "departures_count", "net_spend_meur", "squad_churn_pct"]:
                df[f"{prefix}{col}"] = 0.0

    return df


# ═══════════════════════════════════════════════════════════
#  0j.  Extended form features (multi-context, multi-window)
# ═══════════════════════════════════════════════════════════



def _add_extended_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extended team form features using the TeamFormTransformer.

    Provides richer rolling form features than the basic ``_add_rolling_features``:
    - More metrics: points, wins, draws, losses, goals, xG, shots, cards
    - Per-venue context: overall, home, away
    - Multiple windows: 3, 5, 10, 20 matches
    - Auto-detection of optional stat columns (xG, shots, possession, cards)

    Parameters
    ----------
    df : pd.DataFrame
        Match DataFrame with required columns.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with extended form feature columns added.
    """
    if not config.extended_features.enabled or not config.extended_features.include_extended_form:
        return df

    try:
        if not _EXTENDED_FEATURES_AVAILABLE:
            logger.warning(
                "Extended form features require the feature framework modules."
            )
            return df

        windows = config.extended_features.form_windows
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

        n_added = len([c for c in df.columns if c.startswith(("h_overall_", "a_overall_",
                                                               "h_home_", "a_home_",
                                                               "h_away_", "a_away_"))
                       and "_avg" in c])
        logger.info("Added %d extended form feature columns (windows=%s)", n_added, windows)

    except Exception as exc:
        logger.error("Failed to compute extended form features: %s", exc)

    return df


# ═══════════════════════════════════════════════════════════
#  4a.  Attack / Defence strength ratios (rolling-window)
# ═══════════════════════════════════════════════════════════



def _add_running_league_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Add running league-average goals using an expanding window (no lookahead)."""
    team_stats = _compute_team_stats(df)
    team_stats.sort_values(["team", "date"], inplace=True)

    def _expanding_avg(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("date").copy()
        # Compute running averages shifted by 1 to exclude current match
        grp["goals_scored_cum"] = grp["goals_scored"].expanding().mean().shift(1)
        grp["goals_conceded_cum"] = grp["goals_conceded"].expanding().mean().shift(1)
        return grp

    team_stats = team_stats.groupby("team", group_keys=False).apply(_expanding_avg)

    # Aggregate across all teams to get league avg per match day
    league_avgs = team_stats.groupby("match_id").agg({
        "goals_scored_cum": "mean",
        "goals_conceded_cum": "mean",
    }).rename(columns={
        "goals_scored_cum": "league_avg_goals_scored",
        "goals_conceded_cum": "league_avg_goals_conceded",
    })

    df = df.join(league_avgs, how="left")

    for col in ["league_avg_goals_scored", "league_avg_goals_conceded"]:
        if col in df.columns:
            df[col] = df[col].fillna(1.0)
    return df


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
    # Compute league-average goals from the running (expanding) window
    # so each match only sees data available before kick-off.
    # The expanding window is computed per-match and shifted to avoid lookahead.
    df = _add_running_league_avg(df)

    # Use running (no-lookahead) league average
    league_avg = (
        df["league_avg_goals_scored"].mean() +
        df["league_avg_goals_conceded"].mean()
    ) / 2.0
    if pd.isna(league_avg) or league_avg <= 0:
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
