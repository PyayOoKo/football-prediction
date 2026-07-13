"""
Elo Rating Engine — production-grade team strength ratings for football.

Architecture
------------
::

    ┌─────────────────────────────────────────────────────┐
    │                  EloEngine                           │
    │  Core rating engine with: home advantage, dynamic    │
    │  K-factor, goal-margin scaling, league strength,     │
    │  season regression, new-team handling, host bonus    │
    └───────────┬─────────────────────────────────────────┘
                │  wraps
    ┌───────────▼─────────────────────────────────────────┐
    │              EloTransformer                          │
    │  FeatureTransformer subclass for the framework       │
    │  Outputs: h_elo, a_elo, elo_diff (pre-match)        │
    └─────────────────────────────────────────────────────┘

Features
--------
1. **Home advantage** — configurable H points shift (default 100)
2. **Dynamic K-factor** — base K scaled by goal margin (ln(GD+1)),
   match importance, and league strength
3. **Goal margin adjustment** — bigger wins → bigger rating changes
4. **Newly promoted teams** — configurable starting rating (default 1300)
5. **Season carry-over** — regression to mean between seasons
6. **International competitions** — host nation bonus
7. **League strength adjustment** — tier-based K-factor multiplier

Formulas
--------
**Expected score (home team):**
    E_home = 1 / (1 + 10 ^ ((R_away − R_home − H) / 400))

**Rating update:**
    R_new = R_old + K_eff × (S − E)
    K_eff = K × log(1 + GD) × I × L

    Where:
    - K = base K-factor (default 20, like Club Elo)
    - GD = goal margin (capped at max_margin)
    - I = match importance multiplier (1.0–1.5)
    - L = league strength multiplier (0.8–1.2)
    - S = actual score (1.0 / 0.5 / 0.0)

**Season regression:**
    R_new = μ + (R_old − μ) × (1 − r)
    where r = regression factor (default 1/3)

Leakage prevention
------------------
All ratings recorded are **pre-match** ratings. The match outcome updates
ratings for the *next* fixture, so no future information leaks backwards.

Club Elo alignment
------------------
The default parameters align with the Club Elo methodology:
- K = 20 (standard), 30 (minor leagues)
- Goal margin: ln(GD + 1) multiplier
- Home advantage: 100 Elo points
- Season regression: 1/3 toward mean
- New teams start at 1300 (200 below default 1500)

Usage
-----
::

    from src.feature_framework.features.elo_rating import EloEngine

    # Standalone engine
    engine = EloEngine(k=20, home_advantage=100)
    engine.process_matches(df)

    # As a framework transformer
    from src.feature_framework import FeaturePipeline
    pipeline = FeaturePipeline(...)
    pipeline.plugins.register(EloTransformer)
    report = pipeline.run(entity_type="dataframe", df=df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.feature_framework.base import FeatureTransformer
from src.feature_framework.models import TransformContext

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Defaults
# ═══════════════════════════════════════════════════════════════

_DEFAULT_K: int = 20
_DEFAULT_HOME_ADVANTAGE: int = 100
_DEFAULT_INITIAL_RATING: float = 1500.0
_DEFAULT_NEW_TEAM_RATING: float = 1300.0  # Lower for promoted teams
_DEFAULT_REGRESSION_FACTOR: float = 1 / 3
_DEFAULT_MAX_MARGIN: int = 5

# League strength multipliers: tier → K multiplier
# Tier 1 (top leagues like PL, La Liga) get standard K
# Lower tiers get higher K (more volatile ratings)
_LEAGUE_STRENGTH: dict[int, float] = {
    1: 1.0,   # Top division
    2: 1.1,   # Second division
    3: 1.2,   # Third division
    4: 1.3,   # Fourth division
    5: 1.4,   # Lower divisions
}

# Match importance multipliers: competition type → K multiplier
# IMPORTANT: More specific patterns (qualifiers) must come BEFORE their
# parent types (world cup → world cup qualifier) so substring matching
# picks the correct multiplier.
_MATCH_IMPORTANCE: list[tuple[str, float]] = [
    ("world cup qualifier", 1.2),
    ("world cup", 1.5),
    ("continental qualifier", 1.1),
    ("continental championship", 1.3),
    ("domestic cup", 0.9),
    ("domestic league", 1.0),
    ("club friendly", 0.5),
    ("friendly", 0.6),
]

# Host nation bonus in Elo points
_HOST_BONUS: float = 50.0


# ═══════════════════════════════════════════════════════════════
#  Data models
# ═══════════════════════════════════════════════════════════════


@dataclass
class EloMatchRecord:
    """Record of a single match's Elo computation.

    Parameters
    ----------
    match_index : int
        Index in the original DataFrame.
    home_team : str
    away_team : str
    home_elo_before : float
        Home team's Elo *before* the match.
    away_elo_before : float
        Away team's Elo *before* the match.
    home_elo_after : float
        Home team's Elo *after* the match.
    away_elo_after : float
        Away team's Elo *after* the match.
    elo_diff : float
        ``home_elo_before - away_elo_before``.
    expected_home : float
        Expected score for home team.
    expected_away : float
        Expected score for away team (1 - expected_home).
    actual_home : float
        Actual score for home team (1.0/0.5/0.0).
    k_factor : float
        K-factor used for this match.
    home_elo_change : float
        Change in home team's rating (after - before).
    away_elo_change : float
        Change in away team's rating.
    """
    match_index: int
    home_team: str
    away_team: str
    home_elo_before: float
    away_elo_before: float
    home_elo_after: float
    away_elo_after: float
    elo_diff: float
    expected_home: float
    expected_away: float
    actual_home: float
    k_factor: float
    home_elo_change: float
    away_elo_change: float


@dataclass
class EloSnapshot:
    """Snapshot of the full Elo system at a point in time.

    Parameters
    ----------
    timestamp : datetime
    ratings : dict[str, float]
        All team ratings at this point.
    total_matches_processed : int
        Number of matches processed so far.
    """
    timestamp: datetime
    ratings: dict[str, float] = field(default_factory=dict)
    total_matches_processed: int = 0


# ═══════════════════════════════════════════════════════════════
#  EloEngine — core rating engine
# ═══════════════════════════════════════════════════════════════


class EloEngine:
    """Production-grade Elo rating engine for football teams.

    Extends the legacy ``EloSystem`` with:
    - Dynamic K-factor (goal margin, importance, league strength)
    - Newly promoted teams (lower starting rating)
    - League strength tier multiplier
    - Full match history recording
    - Historical reconstruction
    - Snapshot/checkpoint support
    - Visualisation helpers

    Parameters
    ----------
    k : int
        Base K-factor (default 20, Club Elo standard).
    home_advantage : int
        Home advantage in Elo points (default 100).
    initial_rating : float
        Default rating for completely new teams (default 1500).
    new_team_rating : float
        Rating for newly promoted/entered teams (default 1300).
    regression_factor : float
        Fraction of distance to mean to regress each season (default 1/3).
    use_goal_margin : bool
        Scale K-factor by goal margin (default True).
    max_margin : int
        Cap on goal margin for K-factor scaling (default 5).
    use_importance : bool
        Scale K-factor by match importance (default True).
    use_league_strength : bool
        Scale K-factor by league strength tier (default True).
    host_bonus : float
        Host nation bonus in Elo points (default 50).
    regress_to_mean : bool
        Regress ratings between seasons (default True).
    """

    def __init__(
        self,
        k: int = _DEFAULT_K,
        home_advantage: int = _DEFAULT_HOME_ADVANTAGE,
        initial_rating: float = _DEFAULT_INITIAL_RATING,
        new_team_rating: float = _DEFAULT_NEW_TEAM_RATING,
        regression_factor: float = _DEFAULT_REGRESSION_FACTOR,
        use_goal_margin: bool = True,
        max_margin: int = _DEFAULT_MAX_MARGIN,
        use_importance: bool = True,
        use_league_strength: bool = True,
        host_bonus: float = _HOST_BONUS,
        regress_to_mean: bool = True,
    ) -> None:
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.new_team_rating = new_team_rating
        self.regression_factor = regression_factor
        self.use_goal_margin = use_goal_margin
        self.max_margin = max_margin
        self.use_importance = use_importance
        self.use_league_strength = use_league_strength
        self.host_bonus = host_bonus
        self.regress_to_mean = regress_to_mean

        # Internal state
        self._ratings: dict[str, float] = {}
        self._history: list[EloMatchRecord] = []
        self._current_season: str | None = None
        self._match_count: int = 0
        self._season_team_counts: dict[str, int] = {}  # season -> number of teams seen
        self._known_teams: set[str] = set()

    # ── Public property access ───────────────────────────

    @property
    def ratings(self) -> dict[str, float]:
        """Current ratings for all known teams."""
        return dict(self._ratings)

    @property
    def match_count(self) -> int:
        """Total matches processed."""
        return self._match_count

    @property
    def history(self) -> list[EloMatchRecord]:
        """Full match history with Elo details."""
        return list(self._history)

    def get_rating(self, team: str, season: str | None = None) -> float:
        """Get a team's current Elo rating.

        If the team has not been seen before:
        - In its **first** season, use ``new_team_rating`` (lower, like promoted)
        - Otherwise, use ``initial_rating``
        """
        if team not in self._ratings:
            if season is not None and season not in self._season_team_counts:
                self._season_team_counts[season] = 0
            # First appearance: use lower rating (promoted/new team)
            self._ratings[team] = self.new_team_rating
        return self._ratings[team]

    def set_rating(self, team: str, rating: float) -> None:
        """Manually set a team's rating (for testing or manual adjustments)."""
        self._ratings[team] = rating

    def reset(self) -> None:
        """Reset the engine to its initial state (clears all ratings and history)."""
        self._ratings.clear()
        self._history.clear()
        self._current_season = None
        self._match_count = 0
        self._season_team_counts.clear()
        self._known_teams.clear()

    # ── Core Elo formulas ───────────────────────────────

    def expected_score(self, rating_home: float, rating_away: float) -> float:
        """Compute the expected score for the **home** team.

        Formula
        -------
        E_home = 1 / (1 + 10 ^ ((R_away − R_home − H) / 400))
        """
        return 1.0 / (
            1.0 + 10.0 ** ((rating_away - rating_home - self.home_advantage) / 400.0)
        )

    @staticmethod
    def _actual_score(result: str) -> float:
        """Convert match result to numeric score for the home team.

        Returns 1.0 for H, 0.5 for D, 0.0 for A.
        """
        if result == "H":
            return 1.0
        if result == "D":
            return 0.5
        if result == "A":
            return 0.0
        raise ValueError(f"Unknown result: {result!r} (expected H/D/A)")

    def _compute_k_factor(
        self,
        goal_margin: float,
        importance_mult: float = 1.0,
        league_mult: float = 1.0,
    ) -> float:
        """Compute the effective K-factor for a match.

        Formula
        -------
        K_eff = K × ln(1 + min(GD, max_margin)) × I × L

        Clamped to [K × 0.5 × I × L, K × 3.0 × I × L].

        Parameters
        ----------
        goal_margin : float
            Absolute goal difference (prefer xG margin if available).
        importance_mult : float
            Match importance multiplier (1.0 default).
        league_mult : float
            League strength multiplier (1.0 default).

        Returns
        -------
        float
            Effective K-factor for this match.
        """
        if not self.use_goal_margin:
            return float(self.k) * importance_mult * league_mult

        capped = min(goal_margin, float(self.max_margin))
        margin_mult = max(0.5, np.log1p(capped))  # At least 0.5x for 0-0 draws

        K_eff = float(self.k) * margin_mult * importance_mult * league_mult

        # Clamp to reasonable range
        min_K = float(self.k) * 0.5 * importance_mult * league_mult
        max_K = float(self.k) * 3.0 * importance_mult * league_mult
        return float(np.clip(K_eff, min_K, max_K))

    @staticmethod
    def _parse_importance(league_col: str | None) -> float:
        """Map a competition name to an importance multiplier.

        Uses substring matching against known competition types.
        More specific patterns (qualifiers) are checked first.
        """
        if not league_col or not isinstance(league_col, str):
            return 1.0
        name = league_col.lower().strip().replace("_", " ")

        for pattern, mult in _MATCH_IMPORTANCE:
            if pattern in name:
                return mult
        return 1.0

    @staticmethod
    def _parse_league_strength(league_col: str | None, tier: int | None = None) -> float:
        """Map a competition name or tier to a league strength multiplier.

        Uses the ``tier`` (level) if provided, otherwise tries to infer
        from the competition name.

        Keyword matching uses substring matching (case-insensitive).
        For example, "Premier League" matches tier 1, "Championship"
        matches tier 2 (English second division).
        """
        if tier is not None and tier in _LEAGUE_STRENGTH:
            return _LEAGUE_STRENGTH[tier]

        if league_col and isinstance(league_col, str):
            name = league_col.lower().strip().replace("_", " ")

            # Each tier lists keywords that indicate that division level
            if any(kw in name for kw in ["premier league", "la liga", "serie a",
                                          "bundesliga", "ligue 1", "primera division",
                                          "eredivisie", "primeira liga", "super lig",
                                          "mls", "jupiler", "premiership",
                                          "first division a"]):
                return _LEAGUE_STRENGTH.get(1, 1.0)
            if any(kw in name for kw in ["championship", "segunda division", "serie b",
                                          "2. bundesliga", "ligue 2", "league one",
                                          "second division"]):
                return _LEAGUE_STRENGTH.get(2, 1.1)
            if any(kw in name for kw in ["league two", "third division", "1. division"]):
                return _LEAGUE_STRENGTH.get(3, 1.2)

        return 1.0

    def _compute_goal_margin(
        self,
        home_goals: float | None,
        away_goals: float | None,
        home_xg: float | None = None,
        away_xg: float | None = None,
    ) -> float:
        """Compute goal margin, preferring xG margin when available.

        xG margin is more stable and predictive than actual goal margin.
        """
        if home_xg is not None and away_xg is not None:
            return abs(home_xg - away_xg)
        if home_goals is not None and away_goals is not None:
            return abs(int(home_goals) - int(away_goals))
        return 0.0

    # ── Season management ───────────────────────────────

    def check_season_change(self, season: str | None) -> None:
        """Detect a season boundary and apply regression if needed.

        Call this once per match, **before** updating ratings.
        """
        if season is None:
            return
        if self._current_season is not None and season != self._current_season:
            if self.regress_to_mean:
                self._regress_ratings()
                logger.debug(
                    "Season change: %s → %s — ratings regressed",
                    self._current_season, season,
                )
        self._current_season = season

    def _regress_ratings(self) -> None:
        """Regress all ratings towards the population mean.

        R_new = μ + (R_old − μ) × (1 − r)
        """
        if not self._ratings:
            return
        ratings_arr = np.array(list(self._ratings.values()))
        mean_rating = float(np.mean(ratings_arr))
        for team in self._ratings:
            self._ratings[team] = mean_rating + (
                self._ratings[team] - mean_rating
            ) * (1.0 - self.regression_factor)

    # ── Single match update ─────────────────────────────

    def update(
        self,
        home_team: str,
        away_team: str,
        result: str,
        season: str | None = None,
        league: str | None = None,
        league_tier: int | None = None,
        home_goals: float | None = None,
        away_goals: float | None = None,
        home_xg: float | None = None,
        away_xg: float | None = None,
        is_host: bool = False,
        match_index: int = 0,
    ) -> EloMatchRecord:
        """Process a single match and update ratings.

        Parameters
        ----------
        home_team, away_team : str
            Team names.
        result : str
            Match outcome (``H`` / ``D`` / ``A``).
        season : str, optional
            Season identifier (for regression detection).
        league : str, optional
            Competition name (for importance/strength scaling).
        league_tier : int, optional
            League tier (1 = top division).
        home_goals, away_goals : float, optional
            Actual goals (for margin adjustment).
        home_xg, away_xg : float, optional
            Expected goals (preferred for margin adjustment).
        is_host : bool
            Whether home team is tournament host.
        match_index : int
            Index for tracking in history.

        Returns
        -------
        EloMatchRecord
            Full record of the match's Elo computation.
        """
        self.check_season_change(season)

        # Pre-match ratings
        R_home = self.get_rating(home_team, season)
        R_away = self.get_rating(away_team, season)

        # Host bonus
        effective_home = R_home + (self.host_bonus if is_host else 0.0)

        # Expected scores
        E_home = self.expected_score(effective_home, R_away)
        E_away = 1.0 - E_home

        # Actual scores
        S_home = self._actual_score(result)
        S_away = 1.0 - S_home

        # K-factor computation
        goal_margin = self._compute_goal_margin(home_goals, away_goals, home_xg, away_xg)
        importance_mult = self._parse_importance(league) if self.use_importance else 1.0
        league_mult = self._parse_league_strength(league, league_tier) if self.use_league_strength else 1.0
        K_eff = self._compute_k_factor(goal_margin, importance_mult, league_mult)

        # Track seen teams
        self._known_teams.add(home_team)
        self._known_teams.add(away_team)

        # Update ratings (using unboosted ratings for the actual update)
        new_R_home = R_home + K_eff * (S_home - E_home)
        new_R_away = R_away + K_eff * (S_away - E_away)

        self._ratings[home_team] = new_R_home
        self._ratings[away_team] = new_R_away
        self._match_count += 1

        # Build record
        record = EloMatchRecord(
            match_index=match_index,
            home_team=home_team,
            away_team=away_team,
            home_elo_before=R_home,
            away_elo_before=R_away,
            home_elo_after=new_R_home,
            away_elo_after=new_R_away,
            elo_diff=R_home - R_away,
            expected_home=E_home,
            expected_away=E_away,
            actual_home=S_home,
            k_factor=K_eff,
            home_elo_change=new_R_home - R_home,
            away_elo_change=new_R_away - R_away,
        )
        self._history.append(record)
        return record

    # ── Batch processing ────────────────────────────────

    def process_matches(
        self,
        df: pd.DataFrame,
        home_col: str = "home_team",
        away_col: str = "away_team",
        result_col: str = "result",
        season_col: str | None = "season",
        league_col: str | None = "league",
        league_tier_col: str | None = None,
        home_goals_col: str | None = "home_goals",
        away_goals_col: str | None = "away_goals",
        home_xg_col: str | None = "home_xg",
        away_xg_col: str | None = "away_xg",
        host_nations: dict[int | str, str] | None = None,
        append: bool = False,
    ) -> pd.DataFrame:
        """Process a DataFrame of matches chronologically.

        Adds three columns to the DataFrame:
        - ``h_elo`` — home team's pre-match Elo
        - ``a_elo`` — away team's pre-match Elo
        - ``elo_diff`` — ``h_elo - a_elo``

        Parameters
        ----------
        df : pd.DataFrame
            Match data. **Must be sorted by date** externally or will be
            sorted if ``date`` column exists.
        home_col, away_col : str
            Team name columns.
        result_col : str
            Match outcome column.
        season_col : str, optional
            Season column (for between-season regression).
        league_col : str, optional
            League/competition column (for importance & strength).
        league_tier_col : str, optional
            League tier column (for strength multiplier).
        home_goals_col, away_goals_col : str, optional
            Goals columns.
        home_xg_col, away_xg_col : str, optional
            xG columns (preferred for margin adjustment).
        host_nations : dict[int | str, str], optional
            Season -> host nation mapping.
        append : bool
            If True, don't reset engine state before processing.

        Returns
        -------
        pd.DataFrame
            Copy of **df** with ``h_elo``, ``a_elo``, ``elo_diff`` columns.
        """
        df = df.copy()

        # Sort chronologically
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(["date", home_col], inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Optionally reset engine
        if not append:
            self.reset()

        home_elo_list: list[float] = []
        away_elo_list: list[float] = []
        elo_diff_list: list[float] = []
        k_factor_list: list[float] = []

        has_season = season_col is not None and season_col in df.columns
        has_league = league_col is not None and league_col in df.columns
        has_tier = league_tier_col is not None and league_tier_col in df.columns
        has_goals = home_goals_col is not None and home_goals_col in df.columns
        has_xg = home_xg_col is not None and home_xg_col in df.columns
        has_hosts = host_nations is not None

        for idx, row in df.iterrows():
            home = str(row[home_col])
            away = str(row[away_col])
            result = str(row[result_col]) if result_col in df.columns else "D"

            season: str | None = str(row[season_col]) if has_season else None
            league: str | None = str(row[league_col]) if has_league else None
            tier: int | None = int(row[league_tier_col]) if has_tier else None

            hg: float | None = float(row[home_goals_col]) if has_goals else None
            ag: float | None = float(row[away_goals_col]) if has_goals else None
            hxg: float | None = float(row[home_xg_col]) if has_xg else None
            axg: float | None = float(row[away_xg_col]) if has_xg else None

            is_host = False
            if has_hosts and has_season:
                host_team = host_nations.get(
                    int(season) if season and season.isdigit() else season
                )
                if host_team and home == host_team:
                    is_host = True

            record = self.update(
                home_team=home,
                away_team=away,
                result=result if result in ("H", "D", "A") else "D",
                season=season,
                league=league,
                league_tier=tier,
                home_goals=hg,
                away_goals=ag,
                home_xg=hxg,
                away_xg=axg,
                is_host=is_host,
                match_index=idx,
            )

            home_elo_list.append(record.home_elo_before)
            away_elo_list.append(record.away_elo_before)
            elo_diff_list.append(record.elo_diff)
            k_factor_list.append(record.k_factor)

        df["h_elo"] = home_elo_list
        df["a_elo"] = away_elo_list
        df["elo_diff"] = elo_diff_list
        df["elo_k"] = k_factor_list

        n_teams = len(self._known_teams)
        logger.info(
            "Elo: processed %d matches, %d teams, K=%d, H=%d",
            len(df), n_teams, self.k, self.home_advantage,
        )

        return df

    # ── History reconstruction ──────────────────────────

    def get_history_df(self) -> pd.DataFrame:
        """Return the full match history as a DataFrame.

        Useful for plotting rating trajectories over time.
        """
        if not self._history:
            return pd.DataFrame()

        records = [
            {
                "match_index": r.match_index,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "h_elo_before": r.home_elo_before,
                "a_elo_before": r.away_elo_before,
                "h_elo_after": r.home_elo_after,
                "a_elo_after": r.away_elo_after,
                "h_elo_change": r.home_elo_change,
                "a_elo_change": r.away_elo_change,
                "elo_diff": r.elo_diff,
                "expected_home": r.expected_home,
                "actual_home": r.actual_home,
                "k_factor": r.k_factor,
            }
            for r in self._history
        ]
        return pd.DataFrame(records)

    def team_trajectory(self, team: str) -> pd.DataFrame:
        """Get the Elo rating trajectory for a specific team.

        Returns a DataFrame with ``match_index`` and the team's
        ``elo_before`` and ``elo_after`` for each match they played.
        """
        if not self._history:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for r in self._history:
            if r.home_team == team:
                rows.append({
                    "match_index": r.match_index,
                    "opponent": r.away_team,
                    "side": "home",
                    "elo_before": r.home_elo_before,
                    "elo_after": r.home_elo_after,
                    "elo_change": r.home_elo_change,
                    "expected": r.expected_home,
                    "actual": r.actual_home,
                    "k_factor": r.k_factor,
                })
            elif r.away_team == team:
                rows.append({
                    "match_index": r.match_index,
                    "opponent": r.home_team,
                    "side": "away",
                    "elo_before": r.away_elo_before,
                    "elo_after": r.away_elo_after,
                    "elo_change": r.away_elo_change,
                    "expected": r.expected_away,
                    "actual": 1.0 - r.actual_home,  # away score = 1 - home score
                    "k_factor": r.k_factor,
                })
        return pd.DataFrame(rows).sort_values("match_index")

    def current_snapshot(self) -> EloSnapshot:
        """Return a snapshot of the current engine state."""
        return EloSnapshot(
            timestamp=datetime.now(timezone.utc),
            ratings=dict(self._ratings),
            total_matches_processed=self._match_count,
        )

    # ── Visualisation ──────────────────────────────────

    def plot_team_trajectory(
        self,
        team: str,
        figsize: tuple[int, int] = (12, 5),
        title: str | None = None,
    ) -> Any:
        """Plot a team's Elo rating trajectory over time.

        Requires ``matplotlib``.

        Parameters
        ----------
        team : str
            Team name.
        figsize : tuple
            Figure size (width, height).
        title : str, optional
            Plot title. Auto-generated if omitted.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib required for plotting. Install: pip install matplotlib")
            return None

        traj = self.team_trajectory(team)
        if traj.empty:
            logger.warning("No matches found for team: %s", team)
            return None

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            traj["match_index"], traj["elo_before"],
            marker="o", linestyle="-", linewidth=1.5,
            markersize=3, label=f"{team} Elo",
        )
        ax.axhline(y=self.initial_rating, color="gray", linestyle="--",
                   alpha=0.5, label="Initial rating")
        ax.fill_between(
            traj["match_index"], traj["elo_before"], self.initial_rating,
            where=(traj["elo_before"] >= self.initial_rating),
            color="green", alpha=0.1, label="Above average",
        )
        ax.fill_between(
            traj["match_index"], traj["elo_before"], self.initial_rating,
            where=(traj["elo_before"] < self.initial_rating),
            color="red", alpha=0.1, label="Below average",
        )

        ax.set_title(title or f"Elo Rating Trajectory — {team}")
        ax.set_xlabel("Match Index")
        ax.set_ylabel("Elo Rating")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return fig

    def plot_rating_distribution(
        self,
        figsize: tuple[int, int] = (10, 5),
    ) -> Any:
        """Plot the current distribution of Elo ratings across all teams.

        Requires ``matplotlib``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib required for plotting.")
            return None

        if not self._ratings:
            logger.warning("No ratings to plot.")
            return None

        ratings_arr = np.array(list(self._ratings.values()))
        teams = list(self._ratings.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(ratings_arr, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax1.axvline(x=self.initial_rating, color="red", linestyle="--",
                    alpha=0.7, label=f"Initial ({self.initial_rating})")
        ax1.axvline(x=float(np.mean(ratings_arr)), color="green", linestyle="--",
                    alpha=0.7, label=f"Mean ({np.mean(ratings_arr):.0f})")
        ax1.set_title("Elo Rating Distribution")
        ax1.set_xlabel("Elo Rating")
        ax1.set_ylabel("Number of Teams")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Top teams bar chart
        sorted_teams = sorted(self._ratings.items(), key=lambda x: -x[1])[:15]
        names = [t[0][:15] for t in sorted_teams]
        ratings_vals = [t[1] for t in sorted_teams]

        bars = ax2.barh(range(len(names)), ratings_vals, color="steelblue")
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.axvline(x=self.initial_rating, color="red", linestyle="--",
                    alpha=0.5, label=f"Initial ({self.initial_rating})")
        ax2.set_title("Top 15 Teams by Elo")
        ax2.set_xlabel("Elo Rating")
        ax2.invert_yaxis()

        for bar, val in zip(bars, ratings_vals):
            ax2.text(val + 5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}", va="center", fontsize=8)

        plt.tight_layout()
        return fig

    def print_standings(
        self,
        top_n: int = 20,
    ) -> None:
        """Print a ranked table of teams by current Elo rating.

        Parameters
        ----------
        top_n : int
            Number of top teams to display (default 20).
        """
        if not self._ratings:
            print("No ratings available.")
            return

        sorted_teams = sorted(self._ratings.items(), key=lambda x: -x[1])
        print(f"\n  ELO RATINGS — TOP {min(top_n, len(sorted_teams))}")
        print(f"  {'=' * 45}")
        print(f"  {'Rank':<6} {'Team':<25} {'Rating':>8}")
        print(f"  {'─' * 45}")
        for i, (team, rating) in enumerate(sorted_teams[:top_n], 1):
            arrow = "▲" if rating > self.initial_rating else "▼" if rating < self.initial_rating else "─"
            print(f"  {i:<6} {team:<25} {rating:>8.1f} {arrow}")
        print(f"  {'=' * 45}")
        print(f"  Teams: {len(self._ratings)}  |  Matches: {self._match_count}  |  K: {self.k}")

    # ── Benchmark alignment ────────────────────────────

    def benchmark_report(self) -> dict[str, Any]:
        """Return a report comparing this engine's parameters to Club Elo standard.

        Club Elo (FiveThirtyEight-style) parameters:
        - K = 20 (standard), 30 (minor leagues)
        - Home advantage: 100 Elo points
        - Goal margin: ln(GD + 1) multiplier
        - Season regression: 1/3 toward mean
        - Initial rating: 1500
        - New team rating: 1300

        Returns
        -------
        dict
            Comparison report with ``club_elo_aligned`` flag.
        """
        club_elo_params = {
            "k": 20,
            "home_advantage": 100,
            "initial_rating": 1500.0,
            "new_team_rating": 1300.0,
            "use_goal_margin": True,
            "regression_factor": 1 / 3,
        }

        actual = {
            "k": self.k,
            "home_advantage": self.home_advantage,
            "initial_rating": self.initial_rating,
            "new_team_rating": self.new_team_rating,
            "use_goal_margin": self.use_goal_margin,
            "regression_factor": self.regression_factor,
        }

        aligned = all(
            actual[k] == v for k, v in club_elo_params.items()
        )

        return {
            "club_elo_aligned": aligned,
            "club_elo_parameters": club_elo_params,
            "current_parameters": actual,
            "ratings_count": len(self._ratings),
            "matches_processed": self._match_count,
            "teams": len(self._known_teams),
        }


# ═══════════════════════════════════════════════════════════════
#  EloTransformer — FeatureTransformer for the framework
# ═══════════════════════════════════════════════════════════════


class EloTransformer(FeatureTransformer):
    """Elo rating features for the feature engineering framework.

    Wraps ``EloEngine`` as a ``FeatureTransformer`` so it integrates
    with ``FeaturePipeline``, plugin registry, and the YAML config
    system.

    Output columns
    --------------
    ``h_elo``
        Home team's pre-match Elo rating.
    ``a_elo``
        Away team's pre-match Elo rating.
    ``elo_diff``
        ``h_elo - a_elo`` (home advantage already factored into expected score).
    ``elo_k``
        K-factor used for the match (informational).

    Notes
    -----
    The transformer **preserves engine state** across calls — if you
    call ``transform`` multiple times, the ratings continue from where
    they left off.  Use ``reset_engine()`` to start fresh.
    """

    name: str = "elo_rating"
    version: int = 1
    description: str = (
        "Elo ratings for home and away teams, computed from "
        "historical match results with dynamic K-factor, home "
        "advantage, goal-margin scaling, and league adjustment."
    )
    dependencies: list[str] = []
    data_type: str = "float"
    computation_time: str = "medium"
    category: str = "elo_rating"
    author: str = "system"
    tags: list[str] = ["elo", "rating", "strength", "historical"]
    source: str = "derived"

    output_columns: list[str] = ["h_elo", "a_elo", "elo_diff"]

    _REQUIRED_COLS: frozenset[str] = frozenset({
        "date", "home_team", "away_team", "result",
    })

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        # Create engine from params
        self._engine = EloEngine(
            k=params.get("k", _DEFAULT_K),
            home_advantage=params.get("home_advantage", _DEFAULT_HOME_ADVANTAGE),
            initial_rating=params.get("initial_rating", _DEFAULT_INITIAL_RATING),
            new_team_rating=params.get("new_team_rating", _DEFAULT_NEW_TEAM_RATING),
            regression_factor=params.get("regression_factor", _DEFAULT_REGRESSION_FACTOR),
            use_goal_margin=params.get("use_goal_margin", True),
            max_margin=params.get("max_margin", _DEFAULT_MAX_MARGIN),
            use_importance=params.get("use_importance", True),
            use_league_strength=params.get("use_league_strength", True),
            host_bonus=params.get("host_bonus", _HOST_BONUS),
            regress_to_mean=params.get("regress_to_mean", True),
        )

    @property
    def engine(self) -> EloEngine:
        """Access the underlying EloEngine (for direct queries, plots, etc.)."""
        return self._engine

    def reset_engine(self) -> None:
        """Reset the Elo engine to its initial state."""
        self._engine.reset()

    # ── Input validation ───────────────────────────────

    def validate_input(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        for col in self._REQUIRED_COLS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        return errors

    # ── Transform ──────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        context: TransformContext | None = None,
    ) -> pd.DataFrame:
        """Compute Elo rating features and add them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: ``date``, ``home_team``, ``away_team``, ``result``.
            May contain: ``home_goals``, ``away_goals``, ``home_xg``,
            ``away_xg``, ``season``, ``league``, ``league_tier``.
        context : TransformContext, optional
            Pipeline context (ignored in this implementation).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``h_elo``, ``a_elo``, ``elo_diff`` added.
        """
        logger.debug("EloTransformer: processing %d rows", len(df))

        # Detect available columns
        has_goals = "home_goals" in df.columns and "away_goals" in df.columns
        has_xg = "home_xg" in df.columns and "away_xg" in df.columns
        has_season = "season" in df.columns
        has_league = "league" in df.columns
        has_tier = "league_tier" in df.columns

        result_df = self._engine.process_matches(
            df,
            home_col="home_team",
            away_col="away_team",
            result_col="result",
            season_col="season" if has_season else None,
            league_col="league" if has_league else None,
            league_tier_col="league_tier" if has_tier else None,
            home_goals_col="home_goals" if has_goals else None,
            away_goals_col="away_goals" if has_goals else None,
            home_xg_col="home_xg" if has_xg else None,
            away_xg_col="away_xg" if has_xg else None,
            host_nations=self._resolve_host_nations(),
            append=self.params.get("append_mode", False),
        )
        return result_df

    def _resolve_host_nations(self) -> dict[int | str, str] | None:
        """Return host nations map from params, or None."""
        host_nations = self.params.get("host_nations")
        if host_nations is not None:
            return host_nations
        # Check for a host_nations_file param to load from JSON
        host_file = self.params.get("host_nations_file")
        if host_file:
            import json
            from pathlib import Path
            p = Path(host_file)
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        return None

    # ── Validation ─────────────────────────────────────

    def validate_output(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        for col in self.output_columns:
            if col not in df.columns:
                errors.append(f"Missing output column: {col}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        # Add engine state summary
        d["engine_state"] = {
            "teams": len(self._engine.ratings),
            "matches_processed": self._engine.match_count,
        }
        return d

    def __repr__(self) -> str:
        return (
            f"<EloTransformer v{self.version}: "
            f"{self._engine.match_count} matches, "
            f"{len(self._engine.ratings)} teams>"
        )


# ═══════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════


def create_elo_transformer(
    k: int = _DEFAULT_K,
    home_advantage: int = _DEFAULT_HOME_ADVANTAGE,
    initial_rating: float = _DEFAULT_INITIAL_RATING,
    new_team_rating: float = _DEFAULT_NEW_TEAM_RATING,
    use_goal_margin: bool = True,
    use_importance: bool = True,
    use_league_strength: bool = True,
    **kwargs: Any,
) -> EloTransformer:
    """Create a pre-configured EloTransformer.

    Parameters
    ----------
    k : int
        Base K-factor (default 20, Club Elo standard).
    home_advantage : int
        Home advantage in Elo points (default 100).
    initial_rating : float
        Default rating for new teams (default 1500).
    new_team_rating : float
        Rating for promoted teams (default 1300).
    use_goal_margin : bool
        Scale K-factor by margin (default True).
    use_importance : bool
        Scale K-factor by match importance (default True).
    use_league_strength : bool
        Scale K-factor by league strength (default True).

    Returns
    -------
    EloTransformer
    """
    return EloTransformer(
        k=k,
        home_advantage=home_advantage,
        initial_rating=initial_rating,
        new_team_rating=new_team_rating,
        use_goal_margin=use_goal_margin,
        use_importance=use_importance,
        use_league_strength=use_league_strength,
        **kwargs,
    )
