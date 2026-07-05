"""
Poisson Football Model — predict match scores using independent Poisson distributions.

Core Assumption
---------------
Goals scored by each team in a match follow independent Poisson distributions:

    Home goals  ~ Pois(λ_home)
    Away goals  ~ Pois(λ_away)

The probability of a specific scoreline ``(i, j)`` is the product of the two
independent Poisson probabilities:

    P(i, j) = Pois(i, λ_home) × Pois(j, λ_away)
            = (e^{-λ_home} × λ_home^i / i!) × (e^{-λ_away} × λ_away^j / j!)

Equations (step by step)
------------------------

**1. League-average goals (baseline rates)**

    μ_home  = total_home_goals / total_matches
    μ_away  = total_away_goals / total_matches

    These are the average number of goals scored by *any* home/away team
    across the league.  They serve as the baseline from which individual
    team strengths are measured.

**2. Team attack strength**

    α_team  = (goals_scored_by_team / matches_played_by_team) / μ_league

    where  μ_league = (total_goals_all_matches) / (2 × total_matches)
                     = (μ_home + μ_away) / 2

    - α > 1.0 → team scores more than league average (strong attack)
    - α < 1.0 → team scores less than league average (weak attack)
    - α = 1.0 → team is exactly average

**3. Team defense strength**

    β_team  = (goals_conceded_by_team / matches_played_by_team) / μ_league

    - β > 1.0 → team concedes more than average (weak defense)
    - β < 1.0 → team concedes less than average (strong defense)
    - β = 1.0 → team is exactly average at defending

**4. Expected goals for a specific match**

    λ_home  = μ_home × α_home × β_away
    λ_away  = μ_away × α_away × β_home

    Intuition: the expected number of goals is the *baseline* (league avg)
    adjusted by how strong the *attacking team* is and how weak the
    *defending team* is.

    Example: if the home team attacks strongly (α_home = 1.2) and the away
    team defends weakly (β_away = 1.3), then λ_home = μ_home × 1.2 × 1.3.

**5. Scoreline probability**

    P(i, j) = Pois(i, λ_home) × Pois(j, λ_away)

    where  Pois(k, λ) = e^{-λ} × λ^k / k!

**6. Match outcome probabilities (derived from score table)**

    P(Home Win)  = sum of P(i, j) for all i > j
    P(Draw)      = sum of P(i, j) for all i == j
    P(Away Win)  = sum of P(i, j) for all i < j

**7. Over / Under**

    P(Over X.5)  = sum of P(i, j) for all (i, j) where i + j > X
    P(Under X.5) = 1 − P(Over X.5)

**8. Both Teams To Score (BTTS)**

    P(BTTS) = 1 − P(home = 0) − P(away = 0) + P(home = 0, away = 0)
            = 1 − Pois(0, λ_home) − Pois(0, λ_away) + Pois(0, λ_home) × Pois(0, λ_away)

Leakage prevention
------------------
Team strengths and league averages are computed using **only matches that
occurred before the current match**.  This is achieved by using expanding
(historical-only) windows, exactly like the rest of the feature-engineering
pipeline.

Usage
-----
::

    from src.poisson_model import PoissonModel

    model = PoissonModel()
    model.fit(df)                     # Compute all strengths from historical data

    # Predict a single fixture
    result = model.predict("Arsenal", "Chelsea")
    print(result["expected_home_goals"])   # 1.83
    print(result["most_likely_score"])     # "2-1"
    print(result["home_win_prob"])          # 0.482
    print(result["over_2_5_prob"])          # 0.534
    print(result["btts_prob"])              # 0.581

    # Full probability table
    table = model.scoreline_table("Arsenal", "Chelsea")

    # Generate features for the ML pipeline
    df = model.add_poisson_features(df)
"""

from __future__ import annotations

import logging
from math import factorial
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default max goals to compute in the probability table ─
_MAX_GOALS = 8


# ═══════════════════════════════════════════════════════════
#  Poisson Model — pure-python implementation
# ═══════════════════════════════════════════════════════════


class PoissonModel:
    """Poisson regression model for football match outcome prediction.

    The model estimates each team's attacking and defensive strength
    from historical match data and uses the Poisson distribution to
    compute the probability of every possible scoreline.

    Parameters
    ----------
    min_matches : int
        Minimum number of matches a team must have played before its
        strengths are used.  Teams with fewer matches get strength = 1.0
        (league average).  Default 0.
    max_goals : int
        Maximum number of goals per team to consider in the probability
        table (0 to *max_goals*).  Default 8.
    """

    def __init__(
        self,
        min_matches: int = 0,
        max_goals: int = _MAX_GOALS,
    ) -> None:
        self.min_matches = min_matches
        self.max_goals = max_goals

        # ── Data computed by ``fit()`` ──
        self._league_avg_home: float = 0.0
        self._league_avg_away: float = 0.0
        self._league_avg_overall: float = 0.0

        # Team strengths: {team_name: (attack_strength, defense_strength)}
        self._team_strengths: dict[str, tuple[float, float]] = {}

        # Rolling data (for leakage-free per-match computation)
        self._df: pd.DataFrame | None = None

        self._fitted: bool = False

    # ── Public properties ─────────────────────────────────

    @property
    def league_avg_home(self) -> float:
        """Average home goals per match across the league."""
        return self._league_avg_home

    @property
    def league_avg_away(self) -> float:
        """Average away goals per match across the league."""
        return self._league_avg_away

    @property
    def league_avg_overall(self) -> float:
        """Average goals per team per match across the league."""
        return self._league_avg_overall

    @property
    def team_strengths(self) -> dict[str, tuple[float, float]]:
        """Return a copy of all team strengths ``{team: (attack, defense)}``."""
        return dict(self._team_strengths)

    @property
    def fitted(self) -> bool:
        """Whether the model has been fitted to data."""
        return self._fitted

    # ── Fit (compute strengths) ───────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
    ) -> PoissonModel:
        """Fit the model by computing league averages and team strengths.

        Parameters
        ----------
        df : pd.DataFrame
            Historical match results.  Must contain home/away team names
            and goal columns.
        home_team_col, away_team_col : str
            Column names for team names.
        home_goals_col, away_goals_col : str
            Column names for goals scored.

        Returns
        -------
        PoissonModel
            Self (fitted) for method chaining.
        """
        self._df = df.copy()

        # ── League averages ───────────────────────────────
        all_home_goals = df[home_goals_col].values.astype(float)
        all_away_goals = df[away_goals_col].values.astype(float)

        n_matches = len(df)
        self._league_avg_home = float(np.nanmean(all_home_goals)) if n_matches > 0 else 0.0
        self._league_avg_away = float(np.nanmean(all_away_goals)) if n_matches > 0 else 0.0
        self._league_avg_overall = (self._league_avg_home + self._league_avg_away) / 2.0

        # ── Per-team strengths ────────────────────────────
        self._team_strengths = self._compute_team_strengths(
            df, home_team_col, away_team_col,
            home_goals_col, away_goals_col,
        )

        self._fitted = True
        logger.info(
            "PoissonModel fitted — %.0f home avg, %.0f away avg, %d teams",
            self._league_avg_home, self._league_avg_away, len(self._team_strengths),
        )
        return self

    def _compute_team_strengths(
        self,
        df: pd.DataFrame,
        home_team_col: str,
        away_team_col: str,
        home_goals_col: str,
        away_goals_col: str,
    ) -> dict[str, tuple[float, float]]:
        """Compute (attack, defense) for every team from all available data.

        Attack strength:
            α_team = (goals_scored_by_team / matches) / μ_overall

        Defense strength:
            β_team = (goals_conceded_by_team / matches) / μ_overall
        """
        μ = self._league_avg_overall
        if μ == 0.0:
            return {}

        # Build per-team aggregates
        goals_scored: dict[str, float] = {}
        goals_conceded: dict[str, float] = {}
        matches_played: dict[str, int] = {}

        for _, row in df.iterrows():
            home = row[home_team_col]
            away = row[away_team_col]
            hg = float(row.get(home_goals_col, 0) or 0)
            ag = float(row.get(away_goals_col, 0) or 0)

            goals_scored[home] = goals_scored.get(home, 0.0) + hg
            goals_scored[away] = goals_scored.get(away, 0.0) + ag
            goals_conceded[home] = goals_conceded.get(home, 0.0) + ag
            goals_conceded[away] = goals_conceded.get(away, 0.0) + hg
            matches_played[home] = matches_played.get(home, 0) + 1
            matches_played[away] = matches_played.get(away, 0) + 1

        strengths: dict[str, tuple[float, float]] = {}
        all_teams = set(goals_scored.keys()) | set(goals_conceded.keys())

        for team in all_teams:
            m = matches_played.get(team, 0)
            if m < self.min_matches:
                # Not enough data — use league average (strength = 1.0)
                strengths[team] = (1.0, 1.0)
                continue

            α = (goals_scored.get(team, 0.0) / m) / μ
            β = (goals_conceded.get(team, 0.0) / m) / μ
            strengths[team] = (α, β)

        return strengths

    # ── Expected goals ────────────────────────────────────

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """Return the expected number of goals for each team.

        Uses the fitted league averages and team strengths. For leakage-
        free per-match expected goals (expanding window), use
        ``add_poisson_features()`` instead.

        Parameters
        ----------
        home_team : str
        away_team : str

        Returns
        -------
        tuple[float, float]
            ``(expected_home_goals, expected_away_goals)``
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        μ_home = self._league_avg_home
        μ_away = self._league_avg_away

        α_home, β_home = self._team_strengths.get(home_team, (1.0, 1.0))
        α_away, β_away = self._team_strengths.get(away_team, (1.0, 1.0))

        λ_home = μ_home * α_home * β_away
        λ_away = μ_away * α_away * β_home

        return λ_home, λ_away

    # ── Scoreline probability table ───────────────────────

    def scoreline_table(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
    ) -> pd.DataFrame:
        """Generate a full probability table for all possible scorelines.

        Parameters
        ----------
        home_team : str
        away_team : str
        max_goals : int, optional
            Maximum goals per team (default ``self.max_goals``).

        Returns
        -------
        pd.DataFrame
            Columns: home_goals, away_goals, probability, total_goals,
            scoreline.  Rows sorted by descending probability.
        """
        max_g = max_goals or self.max_goals
        λ_home, λ_away = self.expected_goals(home_team, away_team)

        records: list[dict[str, Any]] = []

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                prob = self._poisson(i, λ_home) * self._poisson(j, λ_away)
                records.append({
                    "home_goals": i,
                    "away_goals": j,
                    "probability": prob,
                    "total_goals": i + j,
                    "scoreline": f"{i}-{j}",
                })

        table = pd.DataFrame(records)
        # Normalise so probabilities sum to 1.0 (accounts for truncated tail)
        total_prob = table["probability"].sum()
        if total_prob > 0:
            table["probability"] /= total_prob

        # Add derived flags
        table["exact_score_flag"] = False  # will compute outside
        table.sort_values("probability", ascending=False, inplace=True)
        table.reset_index(drop=True, inplace=True)

        return table

    # ── Single-match prediction ───────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
        over_under_threshold: float = 2.5,
    ) -> dict[str, Any]:
        """Full prediction for a single match.

        Parameters
        ----------
        home_team : str
        away_team : str
        max_goals : int, optional
            Max goals per team for the probability table.
        over_under_threshold : float
            Threshold for over/under (default 2.5).

        Returns
        -------
        dict[str, Any]
            Keys:
            - ``home_team``, ``away_team``
            - ``expected_home_goals``, ``expected_away_goals``
            - ``most_likely_score`` (string like ``"2-1"``)
            - ``most_likely_prob`` (probability of the most likely score)
            - ``home_win_prob``, ``draw_prob``, ``away_win_prob``
            - ``over_X_prob`` (e.g. ``over_2_5_prob``)
            - ``under_X_prob`` (e.g. ``under_2_5_prob``)
            - ``btts_prob`` (both teams to score)
            - ``btts_no_prob`` (one or both teams fail to score)
            - ``scoreline_table`` (DataFrame of all scorelines)
        """
        λ_home, λ_away = self.expected_goals(home_team, away_team)
        table = self.scoreline_table(home_team, away_team, max_goals=max_goals)

        # ── Most likely exact score ──
        best_row = table.iloc[0]
        most_likely = str(best_row["scoreline"])
        most_likely_prob = float(best_row["probability"])

        # ── Match outcome probabilities ──
        home_win = table[table["home_goals"] > table["away_goals"]]["probability"].sum()
        draw = table[table["home_goals"] == table["away_goals"]]["probability"].sum()
        away_win = table[table["home_goals"] < table["away_goals"]]["probability"].sum()

        # ── Over / Under ──
        over_mask = table["total_goals"] > over_under_threshold
        over_prob = table.loc[over_mask, "probability"].sum()
        under_prob = 1.0 - over_prob

        # ── Both Teams To Score ──
        p_home_zero = self._poisson(0, λ_home)
        p_away_zero = self._poisson(0, λ_away)
        btts_prob = 1.0 - p_home_zero - p_away_zero + (p_home_zero * p_away_zero)

        over_key = f"over_{over_under_threshold:.1f}_prob".replace(".", "_")
        under_key = f"under_{over_under_threshold:.1f}_prob".replace(".", "_")

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_goals": λ_home,
            "expected_away_goals": λ_away,
            "most_likely_score": most_likely,
            "most_likely_prob": round(most_likely_prob, 4),
            "home_win_prob": round(home_win, 4),
            "draw_prob": round(draw, 4),
            "away_win_prob": round(away_win, 4),
            over_key: round(over_prob, 4),
            under_key: round(under_prob, 4),
            "btts_prob": round(btts_prob, 4),
            "btts_no_prob": round(1.0 - btts_prob, 4),
            "scoreline_table": table,
        }

    # ── Predict all fixtures in a DataFrame ────────────────

    def predict_matches(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        max_goals: int | None = None,
        over_under_threshold: float = 2.5,
    ) -> pd.DataFrame:
        """Predict outcomes for every fixture in a DataFrame.

        Returns a summary DataFrame with one row per match containing
        all key prediction metrics (no scoreline tables).
        """
        records: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            home = row[home_team_col]
            away = row[away_team_col]

            result = self.predict(
                home, away,
                max_goals=max_goals,
                over_under_threshold=over_under_threshold,
            )

            # Flatten the result dict, dropping the large table
            flat = {k: v for k, v in result.items() if k != "scoreline_table"}
            records.append(flat)

        return pd.DataFrame(records)

    # ── Feature engineering integration ───────────────────

    def add_poisson_features(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
    ) -> pd.DataFrame:
        """Add Poisson-derived features to a match DataFrame (leakage-free).

        For each match chronologically:
        1. Compute league averages and team strengths from **all previous
           matches** (expanding window).
        2. Compute expected goals using ONLY pre-match data.
        3. Update the running aggregates with the current match's result.

        Parameters
        ----------
        df : pd.DataFrame
            Match data **sorted by date**.
        home_team_col, away_team_col : str
        home_goals_col, away_goals_col : str

        Returns
        -------
        pd.DataFrame
            Copy of **df** with columns:
            - ``Expected_Home_Goals``
            - ``Expected_Away_Goals``
            - ``Expected_Total_Goals``
            - ``Expected_Goal_Difference``
            - ``Home_Attack_Strength``
            - ``Home_Defense_Strength``
            - ``Away_Attack_Strength``
            - ``Away_Defense_Strength``
        """
        df = df.copy()

        expected_home: list[float] = []
        expected_away: list[float] = []
        home_attack_str: list[float] = []
        home_defense_str: list[float] = []
        away_attack_str: list[float] = []
        away_defense_str: list[float] = []

        # Running aggregates for expanding-window computation
        # {team: [total_goals_scored, total_goals_conceded, matches]}
        team_stats: dict[str, list[float]] = {}
        total_home_goals = 0.0
        total_away_goals = 0.0
        total_matches = 0

        for _idx, row in df.iterrows():
            home = row[home_team_col]
            away = row[away_team_col]
            hg = float(row.get(home_goals_col, 0) or 0)
            ag = float(row.get(away_goals_col, 0) or 0)

            # ── Compute current league averages ──────────────
            μ_home = total_home_goals / total_matches if total_matches > 0 else 0.0
            μ_away = total_away_goals / total_matches if total_matches > 0 else 0.0
            μ_overall = (μ_home + μ_away) / 2.0 if total_matches > 0 else 0.0

            # ── Compute current team strengths ───────────────
            def _strength(team: str, stat_type: str) -> float:
                """Helper: compute attack (1) or defense (2) strength for a team."""
                if μ_overall == 0.0 or team not in team_stats:
                    return 1.0  # No data yet → league average
                s = team_stats[team]
                matches = s[2]
                if matches == 0:
                    return 1.0
                avg = s[0 if stat_type == "attack" else 1] / matches
                return avg / μ_overall

            α_home = _strength(home, "attack")
            β_home = _strength(home, "defense")
            α_away = _strength(away, "attack")
            β_away = _strength(away, "defense")

            λ_home = μ_home * α_home * β_away
            λ_away = μ_away * α_away * β_home

            expected_home.append(λ_home)
            expected_away.append(λ_away)
            home_attack_str.append(α_home)
            home_defense_str.append(β_home)
            away_attack_str.append(α_away)
            away_defense_str.append(β_away)

            # ── Update aggregates (for next match) ───────────
            for team_key, scored, conceded in [
                (home, hg, ag),
                (away, ag, hg),
            ]:
                if team_key not in team_stats:
                    team_stats[team_key] = [0.0, 0.0, 0.0]
                team_stats[team_key][0] += scored
                team_stats[team_key][1] += conceded
                team_stats[team_key][2] += 1.0

            total_home_goals += hg
            total_away_goals += ag
            total_matches += 1

        df["Expected_Home_Goals"] = expected_home
        df["Expected_Away_Goals"] = expected_away
        df["Expected_Total_Goals"] = [
            e_h + e_a for e_h, e_a in zip(expected_home, expected_away)
        ]
        df["Expected_Goal_Difference"] = [
            e_h - e_a for e_h, e_a in zip(expected_home, expected_away)
        ]
        df["Home_Attack_Strength"] = home_attack_str
        df["Home_Defense_Strength"] = home_defense_str
        df["Away_Attack_Strength"] = away_attack_str
        df["Away_Defense_Strength"] = away_defense_str

        # Also store the final state so predict() can use rolling data
        self._league_avg_home = μ_home if total_matches > 0 else 0.0
        self._league_avg_away = μ_away if total_matches > 0 else 0.0
        self._league_avg_overall = (self._league_avg_home + self._league_avg_away) / 2.0

        # Recompute global strengths from final state
        for team, (scored, conceded, matches) in team_stats.items():
            if self._league_avg_overall > 0 and matches > 0:
                α = (scored / matches) / self._league_avg_overall
                β = (conceded / matches) / self._league_avg_overall
                self._team_strengths[team] = (α, β)

        self._df = df
        self._fitted = True

        logger.info(
            "Poisson features added — μ_home=%.3f, μ_away=%.3f, %d teams",
            self._league_avg_home, self._league_avg_away, len(self._team_strengths),
        )

        return df

    # ── Static helpers ────────────────────────────────────

    @staticmethod
    def _poisson(k: int, lam: float) -> float:
        """Compute the Poisson probability mass function.

        Formula
        -------
        Pois(k, λ) = e^{-λ} × λ^k / k!

        Where:
        - ``k`` is the number of events (goals)
        - ``λ`` (lambda) is the expected (average) number of events

        Implementation notes
        --------------------
        - Uses ``math.factorial`` for small integer k (k ≤ 8 — standard for
          football).  For larger k, we would use the log-gamma approach,
          but football goals almost never exceed 8.
        - Handles λ = 0 gracefully (returns 0 for k > 0, 1 for k = 0).
        """
        if lam == 0.0:
            return 1.0 if k == 0 else 0.0
        return (np.exp(-lam) * (lam ** k)) / factorial(k)

    @staticmethod
    def _poisson_cdf(k: int, lam: float) -> float:
        """Compute the Poisson cumulative distribution function.

        ``P(X ≤ k) = sum_{i=0}^{k} Pois(i, λ)``
        """
        return float(np.sum([PoissonModel._poisson(i, lam) for i in range(k + 1)]))

    # ── Explanation guide ─────────────────────────────────

    @staticmethod
    def equation_guide() -> str:
        """Return a plain-text explanation of all Poisson model equations."""
        return """
POISSON MODEL — EQUATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LEAGUE-AVERAGE GOALS (BASELINE RATES)
   ─────────────────────────────────────
   μ_home  =  total_home_goals / total_matches
   μ_away  =  total_away_goals / total_matches
   μ_all   =  (μ_home + μ_away) / 2

   These are the average goals scored by a generic home/away team.
   μ_home is always higher than μ_away due to home advantage.
   Example: μ_home = 1.53, μ_away = 1.19 in the Premier League.

2. TEAM ATTACK STRENGTH (α)
   ────────────────────────
   α_team  =  (goals_scored_by_team / matches) / μ_all

   Measures how much better/worse a team is at scoring compared to
   the league average.
     α > 1.0  →  stronger-than-average attack
     α < 1.0  →  weaker-than-average attack
     α = 1.0  →  exactly league average

   Example: α = 1.25 means the team scores 25% more than average.

3. TEAM DEFENSE STRENGTH (β)
   ─────────────────────────
   β_team  =  (goals_conceded_by_team / matches) / μ_all

   Measures how much better/worse a team is at preventing goals.
   Note: lower is better for defense.
     β < 1.0  →  stronger-than-average defense (concedes less)
     β > 1.0  →  weaker-than-average defense (concedes more)
     β = 1.0  →  exactly league average

   Example: β = 0.75 means the team concedes 25% fewer than average.

4. EXPECTED GOALS FOR A MATCH (λ)
   ──────────────────────────────
   λ_home  =  μ_home  ×  α_home  ×  β_away
   λ_away  =  μ_away  ×  α_away  ×  β_home

   Interpretation:
   • Start with the baseline (league avg goals for home/away)
   • Multiply by the attacking team's attack strength (how good they are
     at scoring)
   • Multiply by the defending team's defense strength (how bad/good they
     are at conceding)

   Example: Liverpool (α=1.30) vs Derby (β=1.45) at home:
     λ_home = 1.53 × 1.30 × 1.45 = 2.88 expected goals

5. POISSON DISTRIBUTION
   ─────────────────────
   P(X = k)  =  e⁻^λ  ×  λ^k  /  k!

   The probability of exactly k goals given expected λ.

   Key property: the variance of a Poisson distribution EQUALS its mean.
   This means the spread of possible scores grows with the expected
   number of goals.

   Example with λ = 1.5:
     P(0 goals) = e⁻¹·⁵ = 22.3%
     P(1 goal)  = e⁻¹·⁵ × 1.5 / 1 = 33.5%
     P(2 goals) = e⁻¹·⁵ × 1.5² / 2 = 25.1%
     P(3 goals) = 12.6%, P(4+) = 6.5%

6. SCORELINE PROBABILITY
   ──────────────────────
   P(i, j)  =  Pois(i, λ_home) × Pois(j, λ_away)

   Since home and away goals are assumed independent, the joint
   probability is simply the product of the two marginals.

   Example: λ_home = 1.5, λ_away = 1.0
     P(1-1)  = Pois(1, 1.5) × Pois(1, 1.0)
             = 0.335 × 0.368
             = 12.3%

7. MATCH OUTCOME PROBABILITIES
   ───────────────────────────
   P(Home)  = Σ_{i>j} P(i, j)
   P(Draw)  = Σ_{i=j} P(i, j)
   P(Away)  = Σ_{i<j} P(i, j)

   Sum the scoreline probabilities for all (i,j) where i>j (home win),
   i=j (draw), or i<j (away win).

8. OVER / UNDER
   ──────────────
   P(Over X.5)  =  Σ_{i+j > X} P(i, j)
   P(Under X.5) =  1 - P(Over X.5)

   For X=2.5: sum all scoreline probabilities where total goals ≥ 3.
   The maximum-over threshold is 6.5; the minimum is 0.5.

9. BOTH TEAMS TO SCORE (BTTS)
   ───────────────────────────
   P(BTTS) = 1 - P(home=0) - P(away=0) + P(both=0)
           = 1 - e^{-λ_home} - e^{-λ_away} + e^{-λ_home} × e^{-λ_away}
           = 1 - e^{-λ_home} - e^{-λ_away} + e^{-(λ_home + λ_away)}

   Intuition: Start with 1 (certainty). Subtract the cases where home
   doesn't score, subtract the cases where away doesn't score, then add
   back the double-counted case where BOTH don't score.
"""
