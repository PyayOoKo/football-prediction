"""
Feature Engineering — transform clean match data into predictive features.

**Leakage prevention is the top priority.** Every rolling statistic is
computed with a **shift of 1** — the current match's data is never used to
compute its own features.

This module is the public facade.  Internal implementations live in
the ``src.features`` sub-package.

Typical usage::

    from src.feature_engineering import build_features, train_val_test_split

    X, y = build_features(df)
    splits = train_val_test_split(X, y)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import config as _global_config
from src.elo import add_elo_features
from src.poisson_model import PoissonModel
from src.dixon_coles import DixonColesModel
from src.xg_features import add_xg_features
from src.player_info import add_player_features as add_basic_player_features
from src.player_features import add_player_features as add_enhanced_player_features
from src.odds_processing import add_odds_features, add_consensus_features

# ── Re-export all sub-module functions so they live in this module's namespace ──
from src.features.rolling import (
    _add_rolling_features,
    _add_attack_defence_ratios,
    _add_running_league_avg,
)
from src.features.contextual import (
    _add_h2h_features,
    _add_league_position_features,
    _add_competition_importance,
    _add_extended_h2h_features,
    _add_extended_form_features,
)
from src.features.opt_in import (
    _add_weather_features,
    _add_referee_features,
    _add_schedule_features,
    _add_transfer_features,
)
from src.features.encoding import _encode_categoricals
from src.features.helpers import _get_target_columns

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def build_features(
    df: pd.DataFrame,
    is_training: bool = True,
    config: Any | None = None,
    encoder: Any | None = None,
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
    config : Config, optional
        Config instance for DI.  Defaults to the global singleton.
    encoder : SafeTargetEncoder, optional
        Pre-fitted target encoder.  When provided, it is passed to
        ``_encode_categoricals`` so inference uses training-only priors.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric only).
    y : pd.Series
        Target vector (0 = Away win, 1 = Draw, 2 = Home win).
    """
    cfg = config or _global_config
    logger.info("Building features on %d rows", len(df))
    df = df.copy()

    # ── Sort chronologically (essential for leakage-free rolling features) ──
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["date", "home_team"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 0a. Elo rating features (pre-match ratings, no leakage)
    HOST_NATIONS = {
        2002: "South Korea",
        2006: "Germany",
        2010: "South Africa",
        2014: "Brazil",
        2018: "Russia",
        2022: "Qatar",
        2026: "USA",
    }

    df = add_elo_features(
        df,
        k=cfg.elo.k,
        home_advantage=cfg.elo.home_advantage,
        initial_rating=cfg.elo.initial_rating,
        regress_to_mean=cfg.elo.regress_to_mean,
        regress_factor=cfg.elo.regress_factor,
        use_goal_margin=cfg.elo.use_goal_margin,
        max_goal_margin=cfg.elo.max_goal_margin,
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

    # Apply manual Elo adjustments
    if cfg.elo.adjustments and "Home_Elo" in df.columns:
        for team, penalty in cfg.elo.adjustments.items():
            home_mask = df["home_team"] == team
            away_mask = df["away_team"] == team
            df.loc[home_mask, "Home_Elo"] -= penalty
            df.loc[away_mask, "Away_Elo"] -= penalty
            n_adjusted = home_mask.sum() + away_mask.sum()
            if n_adjusted > 0:
                logger.info(
                    "Elo adjustment: %s -%d pts (%d rows adjusted)",
                    team,
                    penalty,
                    n_adjusted,
                )
        df["Elo_Difference"] = df["Home_Elo"] - df["Away_Elo"]

    # 0b. Odds processing features
    df = add_odds_features(
        df,
        opening_odds_cols=cfg.odds.opening_odds_cols,
        closing_odds_cols=cfg.odds.closing_odds_cols,
        home_team_col="home_team",
        away_team_col="away_team",
    )
    if cfg.odds.compute_consensus:
        df = add_consensus_features(df)

    # 0c. Player information features (optional)
    _players_df = None
    _lineups_df = None
    if cfg.player_info.enabled or cfg.player_features.enabled:
        _players_csv = cfg.paths.external / "players.csv"
        if _players_csv.exists():
            _players_df = pd.read_csv(_players_csv)
            logger.info(
                "Loaded %d player records from %s", len(_players_df), _players_csv
            )
        else:
            logger.warning(
                "Player features enabled but %s not found — using placeholders. "
                "Run: python collect_player_data.py",
                _players_csv,
            )

        _lineups_csv = cfg.paths.external / "lineups.csv"
        if _lineups_csv.exists():
            _lineups_df = pd.read_csv(_lineups_csv)
            logger.info(
                "Loaded %d lineup records from %s", len(_lineups_df), _lineups_csv
            )

    if cfg.player_info.enabled:
        df = add_basic_player_features(
            df,
            players_df=_players_df,
            lineups_df=_lineups_df,
            home_team_col="home_team",
            away_team_col="away_team",
            date_col="date",
        )

    if cfg.player_features.enabled:
        df = add_enhanced_player_features(
            df,
            players_df=_players_df,
            home_team_col="home_team",
            away_team_col="away_team",
            date_col="date",
            rolling_windows=cfg.player_features.rolling_windows,
        )

    # 0c. xG features
    df = add_xg_features(
        df,
        rolling_windows=cfg.xg.rolling_windows,
        compute_xpts=cfg.xg.compute_xpts,
        max_goals_table=cfg.xg.max_goals_table,
        placeholder_value=cfg.xg.placeholder_value,
        warn_missing=cfg.xg.warn_missing,
        home_team_col="home_team",
        away_team_col="away_team",
        home_goals_col="home_goals",
        away_goals_col="away_goals",
    )

    # 0c. Poisson-derived expected goals
    _poisson_model = PoissonModel(
        min_matches=cfg.poisson.min_matches,
        max_goals=cfg.poisson.max_goals,
    )
    df = _poisson_model.add_poisson_features(df)

    # 0d. Dixon-Coles features
    if cfg.dixon_coles.enabled:
        _dc_model = DixonColesModel(
            decay_halflife_days=cfg.dixon_coles.decay_halflife_days,
            use_importance=cfg.dixon_coles.use_importance,
            rho_fixed=cfg.dixon_coles.rho_fixed,
            regress_prior=cfg.dixon_coles.regress_prior,
            prior_strength=cfg.dixon_coles.prior_strength,
        )
        df = _dc_model.add_features(df, refit_every=cfg.dixon_coles.refit_every)

    # 0e–0k. Internal feature builders (now imported from sub-package)
    df = _add_competition_importance(df)
    df = _add_weather_features(df)
    df = _add_referee_features(df)
    df = _add_schedule_features(df)
    df = _add_extended_h2h_features(df)
    df = _add_transfer_features(df)
    df = _add_extended_form_features(df)

    # 1. Rolling team features
    windows = cfg.features.rolling_windows
    df = _add_rolling_features(
        df, window=cfg.features.form_window, extra_windows=windows
    )

    # 2. Head-to-head stats
    if cfg.features.include_h2h:
        df = _add_h2h_features(df, cfg.features.h2h_window)

    # 3. League position
    if cfg.features.include_league_position:
        df = _add_league_position_features(df)

    # 4b. Attack/defence strength ratios (BEFORE categorical encoding — needs raw team names)
    df = _add_attack_defence_ratios(df)

    # 4. Categorical encoding (use pre-fitted encoder if available)
    df = _encode_categoricals(df, encoder=encoder)

    # 5. Preserve stable row ID before separating features & target
    _row_id_series: pd.Series | None = None
    if "_row_id" in df.columns:
        _row_id_series = df["_row_id"].copy()

    # 6. Separate features & target
    cols_to_drop = _get_target_columns(df)
    y: pd.Series
    if is_training and "target" in df.columns:
        y = df["target"].copy()
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    else:
        y = pd.Series(dtype=float)
        X = df.drop(
            columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
        )

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    # Re-attach _row_id (removed by _get_target_columns and numeric filter)
    if _row_id_series is not None:
        # The sort_values earlier shuffled row order, so align by original index
        X["_row_id"] = _row_id_series.values

    # Sanitise column names: XGBoost/LightGBM forbid [ ] < > in names
    X.columns = [
        str(c)
        .replace("<", "_lt_")
        .replace(">", "_gt_")
        .replace("[", "_lb_")
        .replace("]", "_rb_")
        for c in X.columns
    ]

    # Drop fully-NaN feature columns
    X.dropna(axis=1, how="all", inplace=True)

    logger.info(
        "Feature matrix: %d rows × %d columns  |  target distribution: %s",
        *X.shape,
        y.value_counts(normalize=True).to_dict() if not y.empty else "N/A",
    )
    return X, y


# ═══════════════════════════════════════════════════════════
#  Train / validation / test split
# ═══════════════════════════════════════════════════════════


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratios: tuple[float, float, float] | None = None,
    seed: int | None = None,
    config: Any | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
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
    config : Config, optional
        Config instance for dependency injection.  Defaults to the
        global singleton ``config``.

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series]
        Dictionary with keys ``X_train``, ``X_val``, ``X_test``, ``y_train``,
        ``y_val``, ``y_test``.
    """
    cfg = config or _global_config
    if ratios is None:
        ratios = cfg.data.split_ratios
    if seed is None:
        seed = cfg.data.seed

    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    test_ratio = ratios[2]
    val_ratio = ratios[1] / (ratios[0] + ratios[1])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=seed,
        shuffle=False,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=seed,
        shuffle=False,
    )

    logger.info(
        "Chronological split — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
