"""
Dixon-Coles model fitting logic extracted from the main model class.

This module contains the fit method and related optimization logic for the Dixon-Coles model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.dixon_coles.tau import dixon_coles_tau

if TYPE_CHECKING:
    from src.dixon_coles.model import DixonColesModel

logger = logging.getLogger(__name__)


def fit_dixon_coles_model(
    model: "DixonColesModel",
    df: pd.DataFrame,
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
    date_col: str = "date",
    league_col: str = "league",
    round_col: str = "round",
    reference_date: pd.Timestamp | None = None,
    verbose: bool = True,
) -> "DixonColesModel":
    """Fit the Dixon-Coles model via MLE.

    Parameters
    ----------
    model : DixonColesModel
        The model instance to fit.
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
        The "current" date for recency computation. Defaults to
        max(date) + 1 day.
    verbose : bool
        Whether to log progress.

    Returns
    -------
    DixonColesModel
        Self (fitted).
    """
    from src.dixon_coles.weights import compute_recency_weight, get_tournament_importance

    df = df.copy()
    n = len(df)

    if n == 0:
        raise ValueError("Cannot fit Dixon-Coles model on empty DataFrame.")

    # Set reference date
    if reference_date is not None:
        model._reference_date = pd.Timestamp(reference_date)
    else:
        model._reference_date = pd.to_datetime(df[date_col].max()) + pd.Timedelta(days=1)

    # Collect all unique teams
    all_teams = sorted(
        set(df[home_team_col].unique()) | set(df[away_team_col].unique())
    )
    model._team_list = all_teams
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
    weights = model._compute_weights(df, date_col, league_col, round_col)

    # ── Build objective function ──────────────────────
    # Parameter layout: [α_0..α_{n-1}, β_0..β_{n-1}, γ, ρ]
    # Identification: α_0 and β_0 are fixed to 0 (reference team).
    # We estimate α_1..α_{n-1} and β_1..β_{n-1}.

    n_est = n_teams - 1  # number of estimable attack params (team 0 is reference)
    n_param_attack = n_est
    n_param_defence = n_est
    n_param_gamma = 1
    n_param_rho = 0 if model.rho_fixed is not None else 1
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

    if model.regress_prior:
        # Add L2 prior term: λ * sum(α² + β²) to shrink extreme values
        def objective(p: np.ndarray) -> float:
            alpha_full = _full_attack(p)
            beta_full = _full_defence(p)
            lam = np.exp(alpha_full[home_idx] + beta_full[away_idx] + p[-2])
            mu = np.exp(alpha_full[away_idx] + beta_full[home_idx])
            rho = p[-1] if model.rho_fixed is None else model.rho_fixed

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
            prior = model.prior_strength * float(
                np.sum(alpha_full ** 2) + np.sum(beta_full ** 2)
            )
            return nll + prior
    else:
        def objective(p: np.ndarray) -> float:
            alpha_full = _full_attack(p)
            beta_full = _full_defence(p)
            lam = np.exp(alpha_full[home_idx] + beta_full[away_idx] + p[-2])
            mu = np.exp(alpha_full[away_idx] + beta_full[home_idx])
            rho = p[-1] if model.rho_fixed is None else model.rho_fixed

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
    if model.rho_fixed is None:
        x0[-1] = 0.05  # small positive correction

    # ── Optimise ──────────────────────────────────────
    bounds = []
    for _ in range(n_est):
        bounds.append((-3.0, 3.0))  # attack
    for _ in range(n_est):
        bounds.append((-3.0, 3.0))  # defence
    bounds.append((-2.0, 2.0))  # gamma
    if model.rho_fixed is None:
        bounds.append((-0.5, 0.5))  # rho

    if verbose:
        logger.info(
            "Fitting Dixon-Coles MLE: %d teams, %d matches, %d parameters",
            n_teams, n, n_total,
        )
        if model.decay_halflife_days > 0:
            logger.info(
                "  Recency decay halflife: %.0f days", model.decay_halflife_days,
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

    model._optimise_success = result.success
    model._convergence_message = result.message
    model._final_neg_ll = result.fun

    if not result.success:
        logger.warning(
            "Dixon-Coles MLE did not fully converge: %s", result.message,
        )

    # ── Extract parameters ────────────────────────────
    full_alpha = np.zeros(n_teams)
    full_alpha[1:] = result.x[:n_est]
    full_beta = np.zeros(n_teams)
    full_beta[1:] = result.x[n_est:2 * n_est]

    model._alpha = {team: float(full_alpha[i]) for i, team in enumerate(all_teams)}
    model._beta = {team: float(full_beta[i]) for i, team in enumerate(all_teams)}
    model._gamma = float(result.x[-2])
    model._rho = float(result.x[-1]) if model.rho_fixed is None else model.rho_fixed
    model._n_matches = n
    model._fitted = True

    if verbose:
        logger.info(
            "Dixon-Coles fitted: γ=%.3f, ρ=%.3f, neg-LL=%.1f, %s",
            model._gamma, model._rho, result.fun, "converged" if result.success else "partial",
        )

    return model


# Convenience function for standalone usage
def fit_dixon_coles_predict(
    df_train: pd.DataFrame,
    home_team: str,
    away_team: str,
    decay_halflife_days: float = 1460.0,
    use_importance: bool = True,
) -> "DixonColesResult":
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
    from src.dixon_coles.model import DixonColesModel
    
    model = DixonColesModel(
        decay_halflife_days=decay_halflife_days,
        use_importance=use_importance,
    )
    model = fit_dixon_coles_model(model, df_train)
    return model.predict(home_team, away_team)
