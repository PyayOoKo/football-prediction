"""
Dixon-Coles MLE Model — Maximum Likelihood Estimation for football match prediction.

This model extends the standard independent Poisson model with three key innovations:

1. **Tau (ρ) correction** — corrects the systematic underestimation of low-scoring
   results (0-0, 1-0, 0-1, 1-1) by introducing a dependence parameter ρ.
   
2. **Recency weighting** — older matches contribute less to the likelihood via
   exponential time decay, so recent form matters more than results from years ago.
   
3. **Tournament importance weighting** — World Cup matches matter more than
   friendlies; each competition type gets a weight multiplier.

Theoretical Foundation
----------------------
Dixon & Coles (1997), "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", *Applied Statistics*, 46(2), 265-280.

Core Equations
--------------
**Expected goals (log-linear model):**

    λ_home = exp(α_home + β_away + γ)
    λ_away = exp(α_away + β_home)

    where:
    - α_i = attack strength of team i
    - β_i = defense weakness of team i  (higher = weaker defence)
    - γ   = home advantage parameter

**Tau correction for low-scoring matches:**

    τ(x, y; λ, μ, ρ) = | 1 - λ·μ·ρ        if x=0, y=0
                        | 1 + λ·ρ          if x=0, y=1
                        | 1 + μ·ρ          if x=1, y=0
                        | 1 - ρ            if x=1, y=1
                        | 1                otherwise

**Match probability (with correction):**

    P(x, y) = τ(x, y; λ, μ, ρ) × Pois(x; λ) × Pois(y; μ)

**Weighted log-likelihood (maximised via MLE):**

    L = ∑ w_k × ln(P(x_k, y_k; λ_k, μ_k, ρ))

    where w_k = recency_k × importance_k

**Recency weight:**

    w_recency(t) = exp(-ln(2) × days_ago / halflife_days)
    
    Default halflife = 1460 days (~4 years): a match 4 years ago has 50% weight.

**Tournament importance weights:**

    World Cup                    : 2.5
    Continental Championships    : 2.0
    World Cup Qualifiers         : 1.5
    Continental Qualifiers       : 1.2
    League / Club Competitions   : 1.0
    International Friendlies     : 0.6
    Club Friendlies / Other      : 0.4

Usage
-----
::

    from src.dixon_coles import DixonColesModel
    
    # Fit the model to historical match data
    model = DixonColesModel(decay_halflife_days=1460)
    model.fit(df)
    
    # Predict a single match
    result = model.predict("Brazil", "Argentina")
    print(result.home_win_prob)   # 0.42
    print(result.draw_prob)       # 0.28
    print(result.away_win_prob)   # 0.30
    
    # Add Dixon-Coles features to a DataFrame (leakage-free)
    df = model.add_features(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Tournament importance weights
# ═══════════════════════════════════════════════════════════

# Default tournament importance map. Keys are substrings matched
# (case-insensitive) against the league/competition column.
TOURNAMENT_IMPORTANCE: dict[str, float] = {
    # World Cup & equivalents
    "world cup": 2.5,
    "fifa world cup": 2.5,
    "wc": 2.5,
    # Continental championships
    "euro": 2.0,
    "european championship": 2.0,
    "copa america": 2.0,
    "africa cup": 2.0,
    "afcon": 2.0,
    "asian cup": 2.0,
    "gold cup": 2.0,
    "nations league": 1.5,
    "uefa nations league": 1.5,
    # World Cup qualifiers
    "world cup qualifying": 1.5,
    "world cup qualification": 1.5,
    "wc qual": 1.5,
    "wcq": 1.5,
    # Continental qualifiers
    "euro qualifying": 1.2,
    "euro qualification": 1.2,
    # League / club competitions (default)
    "uefa champions league": 1.3,
    "champions league": 1.3,
    "uefa europa league": 1.2,
    "europa league": 1.2,
    "premier league": 1.0,
    "la liga": 1.0,
    "serie a": 1.0,
    "bundesliga": 1.0,
    "ligue 1": 1.0,
    "eredivisie": 1.0,
    "primeira liga": 1.0,
    # Friendlies
    "friendly": 0.6,
    "international friendly": 0.6,
    "club friendly": 0.4,
}


def get_tournament_importance(league: str | None, round_str: str | None = None) -> float:
    """Return the importance weight for a given competition.

    Matches against known league/competition names (case-insensitive).
    If no match is found, returns 1.0 (neutral).

    Parameters
    ----------
    league : str | None
        League or competition name.
    round_str : str | None
        Optional round name for additional context (e.g. "Final" → bonus).

    Returns
    -------
    float
        Importance multiplier (0.0 to 3.0).
    """
    if not league:
        return 1.0

    league_lower = league.lower().strip()
    for pattern, weight in TOURNAMENT_IMPORTANCE.items():
        if pattern in league_lower:
            # Knockout bonus: +20% for knockout stages in important tournaments
            bonus = 1.0
            if round_str and weight >= 1.5:
                round_lower = round_str.lower()
                if any(kw in round_lower for kw in ["final", "semi", "quarter", "round of"]):
                    bonus = 1.2
            return weight * bonus

    return 1.0


def compute_recency_weight(
    match_date: pd.Timestamp | datetime,
    reference_date: pd.Timestamp | datetime,
    halflife_days: float = 1460.0,
) -> float:
    """Compute exponential time-decay weight for a match.

    Formula
    -------
    w = exp(-ln(2) × days_ago / halflife_days)

    Parameters
    ----------
    match_date : pd.Timestamp or datetime
        When the match was played.
    reference_date : pd.Timestamp or datetime
        The \"current\" date (usually the most recent match date + 1 day).
    halflife_days : float
        Number of days after which weight = 0.5. Default 1460 (~4 years).

    Returns
    -------
    float
        Recency weight between 0 and 1.
    """
    if halflife_days <= 0:
        return 1.0  # no decay

    days_ago = (pd.Timestamp(reference_date) - pd.Timestamp(match_date)).days
    if days_ago < 0:
        days_ago = 0  # future match shouldn't happen, but be safe

    return float(np.exp(-np.log(2) * days_ago / halflife_days))


# ═══════════════════════════════════════════════════════════
#  Tau correction function
# ═══════════════════════════════════════════════════════════


def dixon_coles_tau(
    x: int | np.ndarray,
    y: int | np.ndarray,
    lam: float | np.ndarray,
    mu: float | np.ndarray,
    rho: float,
) -> float | np.ndarray:
    """Dixon-Coles tau correction factor for low-scoring matches.

    The correction adjusts for the empirical observation that low-scoring
    results (0-0, 1-0, 0-1, 1-1) occur more frequently than the independent
    Poisson model predicts.

    Parameters
    ----------
    x : int or array-like
        Home goals (can be array for vectorised computation).
    y : int or array-like
        Away goals.
    lam : float or array-like
        Expected home goals.
    mu : float or array-like
        Expected away goals.
    rho : float
        Dependence parameter. rho > 0 means low scores are more likely
        than independent model; rho < 0 means less likely; rho = 0 means
        independent Poisson.

    Returns
    -------
    float or np.ndarray
        Tau correction factor.
    """
    # Handle scalar case
    if np.isscalar(x):
        if x == 0 and y == 0:
            return 1.0 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1.0 + lam * rho
        elif x == 1 and y == 0:
            return 1.0 + mu * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        else:
            return 1.0

    # Vectorised case
    tau = np.ones_like(lam, dtype=float)
    tau[(x == 0) & (y == 0)] = 1.0 - lam[(x == 0) & (y == 0)] * mu[(x == 0) & (y == 0)] * rho
    tau[(x == 0) & (y == 1)] = 1.0 + lam[(x == 0) & (y == 1)] * rho
    tau[(x == 1) & (y == 0)] = 1.0 + mu[(x == 1) & (y == 0)] * rho
    tau[(x == 1) & (y == 1)] = 1.0 - rho
    return tau


# ═══════════════════════════════════════════════════════════
#  Main Model Class
# ═══════════════════════════════════════════════════════════

@dataclass
class DixonColesResult:
    """Container for a single match prediction from Dixon-Coles model."""

    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    rho_used: float
    most_likely_score: str
    most_likely_prob: float
    over_2_5_prob: float
    under_2_5_prob: float
    btts_prob: float
    btts_no_prob: float


class DixonColesModel:
    """Dixon-Coles MLE model for football match prediction.

    Estimates team attack/defence strengths using maximum likelihood
    estimation (MLE) with tau correction for low-scoring matches,
    recency weighting, and tournament importance weighting.

    Parameters
    ----------
    decay_halflife_days : float
        Recency decay halflife in days. A match this many days in the past
        gets 50% weight. Default 1460 (~4 years).
        Set to 0 to disable recency weighting.
    use_importance : bool
        Whether to apply tournament importance weighting (default True).
    rho_fixed : float | None
        If set, fixes ρ to this value instead of estimating it via MLE.
        Set to 0.0 for standard independent Poisson (no tau correction).
    max_goals_table : int
        Maximum goals per team in the probability table (default 8).
    regress_prior : bool
        Whether to apply a weak L2 prior on attack/defence parameters
        to prevent extreme values for teams with few matches (default True).
    prior_strength : float
        Strength of the L2 prior (default 0.01). Higher = more shrinkage.
    """

    def __init__(
        self,
        decay_halflife_days: float = 1460.0,
        use_importance: bool = True,
        rho_fixed: float | None = None,
        max_goals_table: int = 8,
        regress_prior: bool = True,
        prior_strength: float = 0.01,
    ) -> None:
        self.decay_halflife_days = decay_halflife_days
        self.use_importance = use_importance
        self.rho_fixed = rho_fixed
        self.max_goals_table = max_goals_table
        self.regress_prior = regress_prior
        self.prior_strength = prior_strength

        # Fitted parameters
        self._alpha: dict[str, float] = {}  # attack strengths
        self._beta: dict[str, float] = {}   # defence weaknesses
        self._gamma: float = 0.0            # home advantage
        self._rho: float = 0.0              # tau correction
        self._team_list: list[str] = []     # ordered team names (index = param position)
        self._reference_date: pd.Timestamp | None = None
        self._n_matches: int = 0
        self._optimise_success: bool = False
        self._fitted: bool = False

        # Convergence diagnostics
        self._convergence_message: str = ""
        self._final_neg_ll: float = 0.0

    # ── Properties ────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def team_attack(self) -> dict[str, float]:
        return dict(self._alpha)

    @property
    def team_defence(self) -> dict[str, float]:
        return dict(self._beta)

    @property
    def home_advantage(self) -> float:
        return self._gamma

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def n_matches(self) -> int:
        return self._n_matches

    # ── Weight computation ────────────────────────────────

    def _compute_weights(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        league_col: str = "league",
        round_col: str = "round",
    ) -> np.ndarray:
        """Compute combined (recency × importance) weights for each match.

        Parameters
        ----------
        df : pd.DataFrame
            Match data with date column.
        date_col : str
            Column with match dates.
        league_col : str
            Column with league/competition name.
        round_col : str
            Column with round/stage name.

        Returns
        -------
        np.ndarray
            Weight for each match (between 0 and 2.5 typically).
        """
        if self._reference_date is None:
            max_date = pd.to_datetime(df[date_col]).max()
            self._reference_date = max_date + pd.Timedelta(days=1)

        n = len(df)
        weights = np.ones(n, dtype=float)

        # Recency weighting
        if self.decay_halflife_days > 0:
            for i in range(n):
                match_date = pd.to_datetime(df.iloc[i][date_col])
                weights[i] *= compute_recency_weight(
                    match_date, self._reference_date, self.decay_halflife_days,
                )

        # Tournament importance weighting
        if self.use_importance:
            has_league = league_col in df.columns
            has_round = round_col in df.columns
            for i in range(n):
                league = df.iloc[i][league_col] if has_league else None
                rnd = df.iloc[i][round_col] if has_round else None
                weights[i] *= get_tournament_importance(league, rnd)

        # Ensure minimum weight to avoid zero-gradient issues
        weights = np.clip(weights, 1e-6, None)
        return weights

    # ── Fit (MLE) ─────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        date_col: str = "date",
        league_col: str = "league",
        round_col: str = "round",
        reference_date: pd.Timestamp | datetime | None = None,
        verbose: bool = True,
    ) -> DixonColesModel:
        """Fit the Dixon-Coles model via MLE.

        Parameters
        ----------
        df : pd.DataFrame
            Historical match results **sorted by date**.
        home_team_col, away_team_col : str
            Team name columns.
        home_goals_col, away_goals_col : str
            Goals columns.
        date_col : str
            Date column.
        league_col : str
            League/competition column (for importance weighting).
        round_col : str
            Round/stage column (for knockout bonus).
        reference_date : datetime, optional
            The \"current\" date for recency computation. Defaults to
            max(date) + 1 day.
        verbose : bool
            Whether to log progress.

        Returns
        -------
        DixonColesModel
            Self (fitted).
        """
        df = df.copy()
        n = len(df)

        if n == 0:
            raise ValueError("Cannot fit Dixon-Coles model on empty DataFrame.")

        # Set reference date
        if reference_date is not None:
            self._reference_date = pd.Timestamp(reference_date)
        else:
            self._reference_date = pd.to_datetime(df[date_col].max()) + pd.Timedelta(days=1)

        # Collect all unique teams
        all_teams = sorted(
            set(df[home_team_col].unique()) | set(df[away_team_col].unique())
        )
        self._team_list = all_teams
        n_teams = len(all_teams)
        team_to_idx = {team: i for i, team in enumerate(all_teams)}

        if n_teams < 2:
            raise ValueError(f"Need at least 2 teams, got {n_teams}")

        # Prepare data arrays
        home_goals = df[home_goals_col].values.astype(float)
        away_goals = df[away_goals_col].values.astype(float)
        home_idx = np.array([team_to_idx[t] for t in df[home_team_col]], dtype=int)
        away_idx = np.array([team_to_idx[t] for t in df[away_team_col]], dtype=int)

        # Compute weights
        weights = self._compute_weights(df, date_col, league_col, round_col)

        # ── Build objective function ──────────────────────
        # Parameter layout: [α_0..α_{n-1}, β_0..β_{n-1}, γ, ρ]
        # Identification: α_0 and β_0 are fixed to 0 (reference team).
        # We estimate α_1..α_{n-1} and β_1..β_{n-1}.

        n_est = n_teams - 1  # number of estimable attack params (team 0 is reference)
        n_param_attack = n_est
        n_param_defence = n_est
        n_param_gamma = 1
        n_param_rho = 0 if self.rho_fixed is not None else 1
        n_total = n_param_attack + n_param_defence + n_param_gamma + n_param_rho

        # Helper: build full (n_teams,) arrays from estimable params
        def _full_attack(est_params: np.ndarray) -> np.ndarray:
            full = np.zeros(n_teams)
            full[1:] = est_params[:n_est]
            return full

        def _full_defence(est_params: np.ndarray) -> np.ndarray:
            full = np.zeros(n_teams)
            full[1:] = est_params[n_est:2 * n_est]
            return full

        if self.regress_prior:
            # Add L2 prior term: λ * sum(α² + β²) to shrink extreme values
            def objective(p: np.ndarray) -> float:
                alpha_full = _full_attack(p)
                beta_full = _full_defence(p)
                lam = np.exp(alpha_full[home_idx] + beta_full[away_idx] + p[-2])
                mu = np.exp(alpha_full[away_idx] + beta_full[home_idx])
                rho = p[-1] if self.rho_fixed is None else self.rho_fixed

                log_p = (
                    poisson.logpmf(home_goals, lam)
                    + poisson.logpmf(away_goals, mu)
                )
                tau = dixon_coles_tau(home_goals, away_goals, lam, mu, rho)
                _tau_min = float(np.min(tau))
                if _tau_min <= 0:
                    logger.debug("tau <= 0 (min=%.4f, ρ=%.4f)", _tau_min, rho)
                tau = np.clip(tau, 1e-10, None)
                log_p += np.log(tau)

                nll = -float(np.sum(weights * log_p))

                # L2 prior (shrinkage)
                prior = self.prior_strength * float(
                    np.sum(alpha_full ** 2) + np.sum(beta_full ** 2)
                )
                return nll + prior
        else:
            def objective(p: np.ndarray) -> float:
                alpha_full = _full_attack(p)
                beta_full = _full_defence(p)
                lam = np.exp(alpha_full[home_idx] + beta_full[away_idx] + p[-2])
                mu = np.exp(alpha_full[away_idx] + beta_full[home_idx])
                rho = p[-1] if self.rho_fixed is None else self.rho_fixed

                log_p = (
                    poisson.logpmf(home_goals, lam)
                    + poisson.logpmf(away_goals, mu)
                )
                tau = dixon_coles_tau(home_goals, away_goals, lam, mu, rho)
                _tau_min = float(np.min(tau))
                if _tau_min <= 0:
                    logger.debug("tau <= 0 (min=%.4f, ρ=%.4f)", _tau_min, rho)
                tau = np.clip(tau, 1e-10, None)
                log_p += np.log(tau)

                return -float(np.sum(weights * log_p))

        # ── Initial values ────────────────────────────────
        # α initial: mild deviations from 0 (log(avg_goals) style)
        avg_home = float(np.nanmean(home_goals))
        avg_away = float(np.nanmean(away_goals))
        gamma_init = np.log(max(avg_home, 0.1)) - np.log(max(avg_away, 0.1))

        x0 = np.zeros(n_total)
        # Attack params: start at 0 (reference = 0, all others small random)
        x0[:n_est] = np.random.default_rng(42).normal(0, 0.1, size=n_est)
        # Defence params
        x0[n_est:2 * n_est] = np.random.default_rng(43).normal(0, 0.1, size=n_est)
        # Home advantage
        x0[-2] = gamma_init * 0.5  # conservative initialisation
        # Rho
        if self.rho_fixed is None:
            x0[-1] = 0.05  # small positive correction

        # ── Optimise ──────────────────────────────────────
        bounds = []
        for _ in range(n_est):
            bounds.append((-3.0, 3.0))  # attack
        for _ in range(n_est):
            bounds.append((-3.0, 3.0))  # defence
        bounds.append((-2.0, 2.0))  # gamma
        if self.rho_fixed is None:
            bounds.append((-0.5, 0.5))  # rho

        if verbose:
            logger.info(
                "Fitting Dixon-Coles MLE: %d teams, %d matches, %d parameters",
                n_teams, n, n_total,
            )
            if self.decay_halflife_days > 0:
                logger.info(
                    "  Recency decay halflife: %.0f days", self.decay_halflife_days,
                )
            logger.info("  Weight range: [%.4f, %.4f]", weights.min(), weights.max())

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5000,
                "ftol": 1e-8,
                "gtol": 1e-6,
                "maxfun": 25000,
            },
        )

        self._optimise_success = result.success
        self._convergence_message = result.message
        self._final_neg_ll = result.fun

        if not result.success:
            logger.warning(
                "Dixon-Coles MLE did not fully converge: %s", result.message,
            )

        # ── Extract parameters ────────────────────────────
        full_alpha = np.zeros(n_teams)
        full_alpha[1:] = result.x[:n_est]
        full_beta = np.zeros(n_teams)
        full_beta[1:] = result.x[n_est:2 * n_est]

        self._alpha = {team: float(full_alpha[i]) for i, team in enumerate(all_teams)}
        self._beta = {team: float(full_beta[i]) for i, team in enumerate(all_teams)}
        self._gamma = float(result.x[-2])
        self._rho = float(result.x[-1]) if self.rho_fixed is None else self.rho_fixed
        self._n_matches = n
        self._fitted = True

        if verbose:
            logger.info(
                "Dixon-Coles fitted: γ=%.3f, ρ=%.3f, neg-LL=%.1f, %s",
                self._gamma, self._rho, result.fun, "converged" if result.success else "partial",
            )

        return self

    # ── Expected goals ────────────────────────────────────

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """Return expected goals for a match.

        Parameters
        ----------
        home_team : str
        away_team : str

        Returns
        -------
        tuple[float, float]
            (expected_home_goals, expected_away_goals)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        alpha_h = self._alpha.get(home_team, 0.0)
        beta_a = self._beta.get(away_team, 0.0)
        alpha_a = self._alpha.get(away_team, 0.0)
        beta_h = self._beta.get(home_team, 0.0)

        lam = np.exp(alpha_h + beta_a + self._gamma)
        mu = np.exp(alpha_a + beta_h)

        return float(lam), float(mu)

    # ── Scoreline probability table ───────────────────────

    def scoreline_table(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
    ) -> pd.DataFrame:
        """Generate a probability table for all scorelines (0..max_goals).

        Uses the tau-corrected Dixon-Coles joint distribution.

        Parameters
        ----------
        home_team : str
        away_team : str
        max_goals : int, optional
            Max goals per team (default self.max_goals_table).

        Returns
        -------
        pd.DataFrame
            Columns: home_goals, away_goals, probability, total_goals, scoreline.
        """
        max_g = max_goals or self.max_goals_table
        lam, mu = self.expected_goals(home_team, away_team)

        records: list[dict[str, Any]] = []
        for i in range(max_g + 1):
            p_i = poisson.pmf(i, lam)
            for j in range(max_g + 1):
                p_j = poisson.pmf(j, mu)
                tau = dixon_coles_tau(i, j, lam, mu, self._rho)
                prob = float(p_i * p_j * tau)
                records.append({
                    "home_goals": i,
                    "away_goals": j,
                    "probability": max(prob, 0.0),
                    "total_goals": i + j,
                    "scoreline": f"{i}-{j}",
                })

        table = pd.DataFrame(records)
        total = table["probability"].sum()
        if total > 0:
            table["probability"] /= total
        table.sort_values("probability", ascending=False, inplace=True)
        table.reset_index(drop=True, inplace=True)
        return table

    # ── Single match prediction ───────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        max_goals: int | None = None,
        over_under_threshold: float = 2.5,
    ) -> DixonColesResult:
        """Full Dixon-Coles prediction for a single match.

        Parameters
        ----------
        home_team : str
        away_team : str
        max_goals : int, optional
            Max goals per team for probability table.
        over_under_threshold : float
            Threshold for over/under (default 2.5).

        Returns
        -------
        DixonColesResult
        """
        lam, mu = self.expected_goals(home_team, away_team)
        table = self.scoreline_table(home_team, away_team, max_goals=max_goals)

        # Most likely exact score
        best = table.iloc[0]
        most_likely = str(best["scoreline"])
        most_likely_prob = float(best["probability"])

        # Match outcome probabilities
        home_win = table[table["home_goals"] > table["away_goals"]]["probability"].sum()
        draw = table[table["home_goals"] == table["away_goals"]]["probability"].sum()
        away_win = table[table["home_goals"] < table["away_goals"]]["probability"].sum()

        # Over/Under
        over = table[table["total_goals"] > over_under_threshold]["probability"].sum()
        under = 1.0 - over

        # BTTS
        p_h0 = poisson.pmf(0, lam)
        p_a0 = poisson.pmf(0, mu)
        btts = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)

        return DixonColesResult(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=lam,
            expected_away_goals=mu,
            home_win_prob=round(home_win, 4),
            draw_prob=round(draw, 4),
            away_win_prob=round(away_win, 4),
            rho_used=round(self._rho, 4),
            most_likely_score=most_likely,
            most_likely_prob=round(most_likely_prob, 4),
            over_2_5_prob=round(over, 4),
            under_2_5_prob=round(under, 4),
            btts_prob=round(btts, 4),
            btts_no_prob=round(1.0 - btts, 4),
        )

    # ── Batch prediction ──────────────────────────────────

    def predict_matches(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        max_goals: int | None = None,
    ) -> pd.DataFrame:
        """Predict outcomes for all fixtures in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Fixtures with home_team, away_team columns.
        home_team_col, away_team_col : str
            Column names.
        max_goals : int, optional
            Max goals per team.

        Returns
        -------
        pd.DataFrame
            One row per match with prediction columns.
        """
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            home = row[home_team_col]
            away = row[away_team_col]
            try:
                result = self.predict(home, away, max_goals=max_goals)
                records.append({
                    "home_team": home,
                    "away_team": away,
                    "expected_home_goals": result.expected_home_goals,
                    "expected_away_goals": result.expected_away_goals,
                    "home_win_prob": result.home_win_prob,
                    "draw_prob": result.draw_prob,
                    "away_win_prob": result.away_win_prob,
                    "most_likely_score": result.most_likely_score,
                })
            except Exception as e:
                logger.warning("Prediction failed for %s vs %s: %s", home, away, e)

        return pd.DataFrame(records)

    # ── Feature engineering integration ───────────────────

    def add_features(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        date_col: str = "date",
        league_col: str = "league",
        round_col: str = "round",
        refit_every: int = 10,
    ) -> pd.DataFrame:
        """Add Dixon-Coles derived features to a DataFrame (leakage-free).

        Uses a warm-start approach: refits the MLE model every ``refit_every``
        matches and uses the latest parameters for intermediate rows. This is
        drastically faster than refitting from scratch for every match.

        Parameters
        ----------
        df : pd.DataFrame
            Match data **sorted by date**.
        home_team_col, away_team_col : str
        home_goals_col, away_goals_col : str
        date_col : str
        league_col : str
        round_col : str
        refit_every : int
            How often to refit the MLE model (default 10). Higher = faster
            but slightly less responsive to recent form.

        Returns
        -------
        pd.DataFrame
            Copy of df with DC_* columns.
        """
        df = df.copy()
        n = len(df)

        exp_home = np.full(n, np.nan)
        exp_away = np.full(n, np.nan)
        home_attack = np.full(n, 0.0)
        home_def = np.full(n, 0.0)
        away_attack = np.full(n, 0.0)
        away_def = np.full(n, 0.0)
        hw_prob = np.full(n, np.nan)
        d_prob = np.full(n, np.nan)
        aw_prob = np.full(n, np.nan)
        rho_vals = np.full(n, 0.0)

        # Identify completed matches only (those with goals)
        # Use positional indices (0..n-1) throughout to avoid label/iloc confusion
        completed_pos = [
            i for i in range(n)
            if pd.notna(df.iloc[i][home_goals_col]) and pd.notna(df.iloc[i][away_goals_col])
        ]

        if len(completed_pos) < 5:
            # Not enough data — use simple averages
            avg_h = float(df[home_goals_col].mean()) if len(completed_pos) > 0 else 1.0
            avg_a = float(df[away_goals_col].mean()) if len(completed_pos) > 0 else 1.0
            df["DC_Expected_Home_Goals"] = avg_h
            df["DC_Expected_Away_Goals"] = avg_a
            df["DC_Expected_Total_Goals"] = avg_h + avg_a
            df["DC_Expected_Goal_Difference"] = avg_h - avg_a
            df["DC_Home_Attack_Strength"] = 0.0
            df["DC_Home_Defence_Weakness"] = 0.0
            df["DC_Away_Attack_Strength"] = 0.0
            df["DC_Away_Defence_Weakness"] = 0.0
            return df

        # ── Batch-rolling approach: refit every N matches ───────
        def _get_train_df(up_to_pos: int) -> pd.DataFrame:
            """Get all completed matches up to (and including) a positional index."""
            mask = [
                i <= up_to_pos and pd.notna(df.iloc[i][home_goals_col]) and pd.notna(df.iloc[i][away_goals_col])
                for i in range(n)
            ]
            return df.iloc[mask].copy()

        # First fit on the first chunk of completed matches
        first_chunk_end_pos = min(refit_every - 1, len(completed_pos) - 1)
        first_cutoff_pos = completed_pos[first_chunk_end_pos]

        train_df = _get_train_df(first_cutoff_pos)
        ref_date = (
            pd.to_datetime(df.iloc[first_cutoff_pos + 1][date_col])
            if first_cutoff_pos + 1 < n and pd.notna(df.iloc[first_cutoff_pos + 1][date_col])
            else None
        )

        current_model = DixonColesModel(
            decay_halflife_days=self.decay_halflife_days,
            use_importance=self.use_importance,
            rho_fixed=self.rho_fixed,
            max_goals_table=self.max_goals_table,
        )
        current_model.fit(train_df, date_col=date_col, league_col=league_col, round_col=round_col,
                          reference_date=ref_date, verbose=False)

        # Fill features for rows up to the first cutoff
        for i in range(first_cutoff_pos):  # Exclude cutoff to prevent self-leakage
            _fill_dc_row(df, current_model, i, home_team_col, away_team_col,
                         exp_home, exp_away, home_attack, home_def,
                         away_attack, away_def, hw_prob, d_prob, aw_prob, rho_vals)

        # ── Iterate through remaining completed-match chunks ──
        # Progressively double refit_every: early fits are cheap (few matches),
        # late fits are expensive (thousands of matches, hundreds of params).
        last_filled_pos = first_cutoff_pos
        _current_step = refit_every
        chunk_idx = first_chunk_end_pos + 1
        while chunk_idx < len(completed_pos):
            chunk_end_idx = min(chunk_idx + _current_step, len(completed_pos))
            cutoff_pos = completed_pos[chunk_end_idx - 1]

            # Fit on ALL completed data up to this cutoff
            train_df = _get_train_df(cutoff_pos)
            ref_date = (
                pd.to_datetime(df.iloc[cutoff_pos + 1][date_col])
                if cutoff_pos + 1 < n and pd.notna(df.iloc[cutoff_pos + 1][date_col])
                else None
            )

            current_model = DixonColesModel(
                decay_halflife_days=self.decay_halflife_days,
                use_importance=self.use_importance,
                rho_fixed=self.rho_fixed,
                max_goals_table=self.max_goals_table,
            )
            current_model.fit(train_df, date_col=date_col, league_col=league_col, round_col=round_col,
                              reference_date=ref_date, verbose=False)

            # Fill rows between last_filled_pos+1 and cutoff_pos
            for i in range(last_filled_pos + 1, cutoff_pos):  # Exclude cutoff to prevent self-leakage
                _fill_dc_row(df, current_model, i, home_team_col, away_team_col,
                             exp_home, exp_away, home_attack, home_def,
                             away_attack, away_def, hw_prob, d_prob, aw_prob, rho_vals)

            last_filled_pos = cutoff_pos
            chunk_idx = chunk_end_idx
            _current_step = min(_current_step * 2, 2000)  # cap at 2000

        # Fill any remaining rows (future predictions after last completed match)
        if last_filled_pos < n - 1:
            for i in range(last_filled_pos + 1, n):
                _fill_dc_row(df, current_model, i, home_team_col, away_team_col,
                             exp_home, exp_away, home_attack, home_def,
                             away_attack, away_def, hw_prob, d_prob, aw_prob, rho_vals)

        df["DC_Expected_Home_Goals"] = exp_home
        df["DC_Expected_Away_Goals"] = exp_away
        df["DC_Expected_Total_Goals"] = exp_home + exp_away
        df["DC_Expected_Goal_Difference"] = exp_home - exp_away
        df["DC_Home_Attack_Strength"] = home_attack
        df["DC_Home_Defence_Weakness"] = home_def
        df["DC_Away_Attack_Strength"] = away_attack
        df["DC_Away_Defence_Weakness"] = away_def
        df["DC_Home_Win_Prob"] = hw_prob
        df["DC_Draw_Prob"] = d_prob
        df["DC_Away_Win_Prob"] = aw_prob
        df["DC_Rho"] = rho_vals

        logger.info(
            "Dixon-Coles features added: %d matches, refit every %d, last ρ=%.3f",
            n, refit_every, rho_vals[completed_pos[-1]] if completed_pos else 0.0,
        )

        return df


# ═══════════════════════════════════════════════════════════
#  Rolling helper
# ═══════════════════════════════════════════════════════════


def _fill_dc_row(
    df: pd.DataFrame,
    model: DixonColesModel,
    i: int,
    home_team_col: str,
    away_team_col: str,
    exp_home: np.ndarray,
    exp_away: np.ndarray,
    home_attack: np.ndarray,
    home_def: np.ndarray,
    away_attack: np.ndarray,
    away_def: np.ndarray,
    hw_prob: np.ndarray,
    d_prob: np.ndarray,
    aw_prob: np.ndarray,
    rho_vals: np.ndarray,
) -> None:
    """Fill DC feature arrays for a single row using a fitted model."""
    try:
        home = df.iloc[i][home_team_col]
        away = df.iloc[i][away_team_col]
        lam, mu = model.expected_goals(home, away)
        result = model.predict(home, away)
        exp_home[i] = lam
        exp_away[i] = mu
        home_attack[i] = model._alpha.get(home, 0.0)
        home_def[i] = model._beta.get(home, 0.0)
        away_attack[i] = model._alpha.get(away, 0.0)
        away_def[i] = model._beta.get(away, 0.0)
        hw_prob[i] = result.home_win_prob
        d_prob[i] = result.draw_prob
        aw_prob[i] = result.away_win_prob
        rho_vals[i] = model._rho
    except Exception as e:
        logger.warning("DC fill failed for row %d (%s vs %s): %s", i,
                       df.iloc[i][home_team_col], df.iloc[i][away_team_col], e)
        exp_home[i] = 1.0
        exp_away[i] = 1.0
        hw_prob[i] = 0.45
        d_prob[i] = 0.25
        aw_prob[i] = 0.30


# ═══════════════════════════════════════════════════════════
#  Convenience function (single-call fit + predict)
# ═══════════════════════════════════════════════════════════


def fit_dixon_coles_predict(
    df_train: pd.DataFrame,
    home_team: str,
    away_team: str,
    decay_halflife_days: float = 1460.0,
    use_importance: bool = True,
) -> DixonColesResult:
    """Fit Dixon-Coles on training data and predict one match.

    Parameters
    ----------
    df_train : pd.DataFrame
        Historical match data.
    home_team, away_team : str
        Teams to predict.
    decay_halflife_days : float
        Recency decay halflife (default 1460).
    use_importance : bool
        Apply tournament importance weighting (default True).

    Returns
    -------
    DixonColesResult
    """
    model = DixonColesModel(
        decay_halflife_days=decay_halflife_days,
        use_importance=use_importance,
    )
    model.fit(df_train)
    return model.predict(home_team, away_team)
