"""
Elo Rating System — dynamic team strength ratings for football.

Formulas
--------
**Expected Score (home team's expected points):**
    E_home = 1 / (1 + 10 ^ ((R_away - R_home - H) / 400))

    Where:
    - ``R_home``, ``R_away`` — teams' Elo ratings **before** the match
    - ``H`` — home advantage in Elo points (default 100)
    - ``400`` — standard Elo scaling factor

**Rating Update:**
    R_new = R_old + K_eff × (S - E)

    Where:
    - ``K_eff`` — effective K-factor, optionally adjusted by goal margin
    - ``S`` — actual score: 1.0 for win, 0.5 for draw, 0.0 for loss
    - ``E`` — expected score (from formula above)

**Goal Margin Adjustment (when enabled):**
    K_eff = K × ln(1 + min(goal_margin, max_margin))

    A 1-goal win → K_eff ≈ K × 0.69
    A 3-goal win → K_eff ≈ K × 1.39
    A 5-goal win → K_eff ≈ K × 1.79  (capped)

**Between-Season Regression (when enabled):**
    R_new = μ + (R_old − μ) × (1 − r)

    Where:
    - ``μ`` — mean rating across all teams (typically ~1500)
    - ``r`` — regression factor (default 1/3 or 33%)
    - Prevents ratings from drifting too far from the mean over many seasons

**Elo Difference (feature column):**
    Elo_Difference = Home_Elo − Away_Elo

Leakage prevention
------------------
All ratings recorded in ``Home_Elo`` and ``Away_Elo`` are the teams' ratings
**before** the match result is applied. The match outcome only updates the
ratings for the *next* fixture, so no future information leaks backwards.

Usage
-----
::

    from src.elo import EloSystem, add_elo_features

    # Standalone usage
    elo = EloSystem(k=32, home_advantage=100, initial_rating=1500)
    expected = elo.expected_score(1500, 1400)  # Home=1500, Away=1400
    elo.update_ratings("Arsenal", "Chelsea", "H")  # Home win

    # DataFrame integration
    df = add_elo_features(df, k=32, home_advantage=100)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Core Elo Engine
# ═══════════════════════════════════════════════════════════


class EloSystem:
    """Dynamic Elo rating system for football teams.

    Maintains an internal dictionary of team → rating and provides methods
    to compute expected scores, update ratings after matches, and regress
    ratings between seasons.

    Parameters
    ----------
    k : int
        Base K-factor (default 32).
    home_advantage : int
        Home advantage in Elo points (default 100).
    initial_rating : int
        Default rating for unseen teams (default 1500).
    regress_to_mean : bool
        Whether to regress ratings between seasons (default True).
    regress_factor : float
        Regression fraction (default 1/3).
    use_goal_margin : bool
        Whether to scale K-factor by goal margin (default True).
    max_goal_margin : int
        Cap on goal margin multiplier (default 5).
    """

    def __init__(
        self,
        k: int = 32,
        home_advantage: int = 100,
        initial_rating: int = 1500,
        regress_to_mean: bool = True,
        regress_factor: float = 1 / 3,
        use_goal_margin: bool = True,
        max_goal_margin: int = 5,
    ) -> None:
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.regress_to_mean = regress_to_mean
        self.regress_factor = regress_factor
        self.use_goal_margin = use_goal_margin
        self.max_goal_margin = max_goal_margin

        # Internal rating store {team_name: rating}
        self._ratings: dict[str, float] = {}

        # Track the most recent season seen (for between-season regression)
        self._current_season: str | None = None

    # ── Public helpers ────────────────────────────────────

    @property
    def ratings(self) -> dict[str, float]:
        """Return a copy of all current team ratings."""
        return dict(self._ratings)

    def get_rating(self, team: str) -> float:
        """Return a team's current Elo rating.

        If the team has not been seen before, initialise it with the
        configured ``initial_rating``.
        """
        if team not in self._ratings:
            self._ratings[team] = float(self.initial_rating)
        return self._ratings[team]

    def set_rating(self, team: str, rating: float) -> None:
        """Explicitly set a team's Elo rating."""
        self._ratings[team] = rating

    # ── Core Elo formulas ─────────────────────────────────

    def expected_score(self, rating_home: float, rating_away: float) -> float:
        """Compute the expected score for the **home** team.

        Formula
        -------
        E_home = 1 / (1 + 10 ^ ((R_away − R_home − H) / 400))

        The home-advantage term ``H`` shifts the curve right, meaning the
        home team is expected to perform better than its raw rating would
        suggest against an equal opponent.
        """
        return 1.0 / (
            1.0
            + 10.0
            ** ((rating_away - rating_home - self.home_advantage) / 400.0)
        )

    @staticmethod
    def _actual_score(result: str) -> float:
        """Convert match result to a numeric score for the home team.

        Returns
        -------
        float
            1.0 for home win (``"H"``),
            0.5 for draw (``"D"``),
            0.0 for away win (``"A"``).

        Raises
        ------
        ValueError
            If *result* is not one of ``"H"`` / ``"D"`` / ``"A"``.
        """
        if result == "H":
            return 1.0
        if result == "D":
            return 0.5
        if result == "A":
            return 0.0
        raise ValueError(f"Unknown result value: {result!r}  (expected H/D/A)")

    def _effective_k(self, margin: int, xg_margin: float | None = None) -> float:
        """Calculate the effective K-factor for a match.

        Uses xG-margin when available (more predictive), falls back to
        actual goal margin. When ``use_goal_margin`` is enabled:
            K_eff = K × ln(1 + min(effective_margin, max_margin))

        Parameters
        ----------
        margin : int
            Actual goal margin (fallback).
        xg_margin : float, optional
            Expected goals margin (preferred — more stable).

        Returns
        -------
        float
            Effective K-factor for this match.
        """
        if not self.use_goal_margin:
            return float(self.k)

        # Prefer xG-margin when available (more predictive, less noisy)
        if xg_margin is not None and xg_margin > 0:
            capped = min(xg_margin, float(self.max_goal_margin))
            return float(self.k) * np.log1p(capped)

        # Fall back to actual goal margin
        if margin <= 0:
            return float(self.k)
        capped = min(margin, self.max_goal_margin)
        return float(self.k) * np.log1p(float(capped))

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        result: str,
        home_goals: float | None = None,
        away_goals: float | None = None,
        home_xg: float | None = None,
        away_xg: float | None = None,
        is_host: bool = False,
    ) -> tuple[float, float, float]:
        """Update Elo ratings based on a single match result.

        Uses xG margin when available (more stable than actual goals),
        and applies host-nation bonus for international tournaments.

        Parameters
        ----------
        home_team : str
        away_team : str
        result : str
            ``"H"`` (home win), ``"D"`` (draw), or ``"A"`` (away win).
        home_goals : float | None
            Actual goals scored by home team (used for K-factor margin).
        away_goals : float | None
            Actual goals scored by away team.
        home_xg : float | None
            xG for home team (preferred for K-factor margin adjustment).
        away_xg : float | None
            xG for away team.
        is_host : bool
            Whether the home team is the tournament host — applies a
            bonus to the expected score calculation.

        Returns
        -------
        tuple[float, float, float]
            ``(home_rating_before, away_rating_before, elo_difference)``
        """
        # Pre-match ratings
        R_home = self.get_rating(home_team)
        R_away = self.get_rating(away_team)

        # Apply host-nation bonus: boost home rating for expected score
        effective_home_rating = R_home
        if is_host:
            effective_home_rating += 50  # host nation bonus in Elo points

        # Expected scores (using effective rating for host bonus)
        E_home = self.expected_score(effective_home_rating, R_away)
        E_away = 1.0 - E_home

        # Actual scores
        S_home = self._actual_score(result)
        S_away = 1.0 - S_home

        # Goal margin for K-factor adjustment (prefer xG margin)
        if home_goals is not None and away_goals is not None:
            margin = abs(int(home_goals) - int(away_goals))
        else:
            margin = 0

        if home_xg is not None and away_xg is not None:
            xg_margin = abs(home_xg - away_xg)
        else:
            xg_margin = None

        K_eff = self._effective_k(margin, xg_margin=xg_margin)

        # Update ratings (using ACTUAL ratings, not host-boosted, so bonus is temporary)
        new_R_home = R_home + K_eff * (S_home - E_home)
        new_R_away = R_away + K_eff * (S_away - E_away)

        self._ratings[home_team] = new_R_home
        self._ratings[away_team] = new_R_away

        elo_diff = R_home - R_away  # pre-match difference (unboosted)
        return R_home, R_away, elo_diff

    # ── Season management ─────────────────────────────────

    def regress_ratings(self) -> None:
        """Regress all ratings towards the population mean.

        This is applied **between seasons** to prevent ratings from
        accumulating extreme values over many years.  A team that was
        very strong regresses down; a weak team regresses up.

        Formula
        -------
        R_new = μ + (R_old − μ) × (1 − r)

        where ``μ`` is the mean rating across all active teams and ``r``
        is the regression factor (default 1/3).
        """
        if not self._ratings:
            return

        ratings_arr = np.array(list(self._ratings.values()))
        mean_rating = float(np.mean(ratings_arr))

        for team in self._ratings:
            self._ratings[team] = mean_rating + (
                self._ratings[team] - mean_rating
            ) * (1.0 - self.regress_factor)

        logger.debug(
            "Regressed %d ratings towards %.1f (factor=%.2f)",
            len(self._ratings),
            mean_rating,
            self.regress_factor,
        )

    def check_season_change(self, season: str | None) -> None:
        """Detect a season boundary and apply regression if needed.

        Call this once per match, **before** updating ratings, to ensure
        the regression applies at the start of a new season.
        """
        if season is None:
            return

        if self._current_season is not None and season != self._current_season:
            if self.regress_to_mean:
                self.regress_ratings()
                logger.debug(
                    "Season changed: %s → %s — ratings regressed",
                    self._current_season,
                    season,
                )
        self._current_season = season

    # ── Bulk processing ───────────────────────────────────

    def process_matches(
        self,
        df: pd.DataFrame,
        home_col: str = "home_team",
        away_col: str = "away_team",
        result_col: str = "result",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        season_col: str | None = "season",
        home_xg_col: str | None = "home_xg",
        away_xg_col: str | None = "away_xg",
        host_nations: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """Walk through a DataFrame of matches chronologically.

        For **each** match:
        1. Check for a season change (and regress ratings if so).
        2. Record the pre-match Elo ratings for both teams.
        3. Update the internal ratings using the match result.

        Parameters
        ----------
        df : pd.DataFrame
            Match data.  **Must be sorted by date** — the function does not
            sort internally because the caller (``build_features``) is
            responsible for chronological ordering.
        home_col, away_col : str
            Column names for team names.
        result_col : str
            Column with match outcomes (``"H"`` / ``"D"`` / ``"A"``).
        home_goals_col, away_goals_col : str
            Column names for goals scored (used for margin-of-victory
            adjustment to K-factor).
        season_col : str | None
            Optional season column for between-season regression.
        home_xg_col, away_xg_col : str | None
            Optional xG columns for xG-margin K-factor adjustment (more stable).
        host_nations : dict[int, str] | None
            Mapping of season -> host nation name for host bonus.

        Returns
        -------
        pd.DataFrame
            A copy of **df** with three new columns:
            - ``Home_Elo`` — home team's rating **before** the match
            - ``Away_Elo`` — away team's rating **before** the match
            - ``Elo_Difference`` — ``Home_Elo − Away_Elo``
        """
        df = df.copy()

        home_elo_list: list[float] = []
        away_elo_list: list[float] = []
        elo_diff_list: list[float] = []
        _append_elo = home_elo_list.append
        _append_away = away_elo_list.append
        _append_diff = elo_diff_list.append

        has_season = season_col is not None and season_col in df.columns
        has_xg = home_xg_col is not None and home_xg_col in df.columns
        has_hosts = host_nations is not None

        # Pre-extract columns for fast itertuples access
        _check_season = self.check_season_change
        _get_rating = self.get_rating
        _update = self.update_ratings

        for row in df.itertuples(index=False):
            home = getattr(row, home_col)
            away = getattr(row, away_col)
            result = getattr(row, result_col, None)

            if has_season:
                _check_season(str(getattr(row, season_col)))

            R_home = _get_rating(home)
            R_away = _get_rating(away)
            elo_diff = R_home - R_away

            if result is not None and result in ("H", "D", "A"):
                home_g = getattr(row, home_goals_col, None)
                away_g = getattr(row, away_goals_col, None)
                home_xg = float(getattr(row, home_xg_col, 0) or 0) if has_xg else None
                away_xg = float(getattr(row, away_xg_col, 0) or 0) if has_xg else None

                is_host = False
                if has_hosts and has_season:
                    season_val = getattr(row, season_col, None)
                    season_int = int(season_val) if season_val is not None else 0
                    host_team = host_nations.get(season_int)
                    if host_team and home == host_team:
                        is_host = True

                R_home, R_away, _ = _update(
                    home, away, str(result),
                    home_goals=home_g, away_goals=away_g,
                    home_xg=home_xg, away_xg=away_xg, is_host=is_host,
                )

            _append_elo(R_home)
            _append_away(R_away)
            _append_diff(elo_diff)

        df["Home_Elo"] = home_elo_list
        df["Away_Elo"] = away_elo_list
        df["Elo_Difference"] = elo_diff_list

        return df


# ═══════════════════════════════════════════════════════════
#  Convenience function (for feature_engineering integration)
# ═══════════════════════════════════════════════════════════


def add_elo_features(
    df: pd.DataFrame,
    k: int = 32,
    home_advantage: int = 100,
    initial_rating: int = 1500,
    regress_to_mean: bool = True,
    regress_factor: float = 1 / 3,
    use_goal_margin: bool = True,
    max_goal_margin: int = 5,
    home_col: str = "home_team",
    away_col: str = "away_team",
    result_col: str = "result",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
    season_col: str | None = "season",
    home_xg_col: str | None = "home_xg",
    away_xg_col: str | None = "away_xg",
    host_nations: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Add Elo rating features to a match results DataFrame.

    Creates an internal ``EloSystem``, processes every match (recording
    pre-match ratings), and appends three columns:

    - ``Home_Elo`` — home team's Elo rating **before** the match
    - ``Away_Elo`` — away team's Elo rating **before** the match
    - ``Elo_Difference`` — ``Home_Elo − Away_Elo``

    **Leakage-free:** Only data available *before* the match is used.
    The result of the match updates ratings for the **next** fixture.

    Parameters
    ----------
    df : pd.DataFrame
        Match data **already sorted by date**.
    k : int
        K-factor (default 32).
    home_advantage : int
        Home advantage in Elo points (default 100).
    initial_rating : int
        Starting Elo for unseen teams (default 1500).
    regress_to_mean : bool
        Regress ratings between seasons (default True).
    regress_factor : float
        Regression rate (default 1/3).
    use_goal_margin : bool
        Scale K-factor by goal margin (default True).
    max_goal_margin : int
        Cap on goal-margin multiplier (default 5).
    home_col, away_col : str
        Team name columns.
    result_col : str
        Result column (``"H"`` / ``"D"`` / ``"A"``).
    home_goals_col, away_goals_col : str
        Goals columns (for margin adjustment).
    season_col : str | None
        Season column for between-season regression.
    home_xg_col, away_xg_col : str | None
        Optional xG columns for xG-margin K-factor adjustment.
    host_nations : dict[int, str] | None
        Mapping of season -> host nation name for host bonus.

    Returns
    -------
    pd.DataFrame
        Copy of **df** with ``Home_Elo``, ``Away_Elo``, ``Elo_Difference``.

    Examples
    --------
    >>> df = add_elo_features(results_df)
    >>> df[["date", "home_team", "away_team", "Home_Elo", "Away_Elo", "Elo_Difference"]]
    """
    elo = EloSystem(
        k=k,
        home_advantage=home_advantage,
        initial_rating=initial_rating,
        regress_to_mean=regress_to_mean,
        regress_factor=regress_factor,
        use_goal_margin=use_goal_margin,
        max_goal_margin=max_goal_margin,
    )

    df_result = elo.process_matches(
        df,
        home_col=home_col,
        away_col=away_col,
        result_col=result_col,
        home_goals_col=home_goals_col,
        away_goals_col=away_goals_col,
        season_col=season_col,
        home_xg_col=home_xg_col,
        away_xg_col=away_xg_col,
        host_nations=host_nations,
    )

    n_teams = len(
        set(df[home_col].unique()) | set(df[away_col].unique())
    )
    logger.info(
        "Elo features added: %d teams, K=%d, home_adv=%d, host_nations=%s",
        n_teams,
        k,
        home_advantage,
        bool(host_nations),
    )

    return df_result
