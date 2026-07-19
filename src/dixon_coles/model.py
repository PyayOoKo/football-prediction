"""
Dixon-Coles Model - Main model class for football match prediction.

This module contains the DixonColesModel class and DixonColesResult dataclass,
providing Maximum Likelihood Estimation for football match prediction with
tau correction, recency weighting, and tournament importance weighting.
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

from src.dixon_coles.tau import dixon_coles_tau
from src.dixon_coles.weights import (
    TOURNAMENT_IMPORTANCE,
    compute_recency_weight,
    get_tournament_importance,
)

logger = logging.getLogger(__name__)


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
    ) -> "DixonColesModel":
        """Fit the Dixon-Coles model via MLE."""
        from src.dixon_coles.fit import fit_dixon_coles_model
        return fit_dixon_coles_model(
            self, df, home_team_col, away_team_col, home_goals_col, away_goals_col,
            date_col, league_col, round_col, reference_date, verbose,
        )

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return expected goals for a match."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        alpha_h = self._alpha.get(home_team, 0.0)
        beta_a = self._beta.get(away_team, 0.0)
        alpha_a = self._alpha.get(away_team, 0.0)
        beta_h = self._beta.get(home_team, 0.0)
        lam = np.exp(alpha_h + beta_a + self._gamma)
        mu = np.exp(alpha_a + beta_h)
        return float(lam), float(mu)

    def scoreline_table(self, home_team: str, away_team: str, max_goals: int | None = None) -> pd.DataFrame:
        """Generate a probability table for all scorelines."""
        from src.dixon_coles.tau import dixon_coles_tau
        max_g = max_goals or self.max_goals_table
        lam, mu = self.expected_goals(home_team, away_team)
        records: list[dict[str, Any]] = []
        for i in range(max_g + 1):
            p_i = poisson.pmf(i, lam)
            for j in range(max_g + 1):
                p_j = poisson.pmf(j, mu)
                tau = dixon_coles_tau(i, j, lam, mu, self._rho)
                prob = float(p_i * p_j * tau)
                records.append({"home_goals": i, "away_goals": j, "probability": max(prob, 0.0), "total_goals": i + j, "scoreline": f"{i}-{j}"})
        table = pd.DataFrame(records)
        total = table["probability"].sum()
        if total > 0:
            table["probability"] /= total
        table.sort_values("probability", ascending=False, inplace=True)
        table.reset_index(drop=True, inplace=True)
        return table

    def predict(self, home_team: str, away_team: str, max_goals: int | None = None, over_under_threshold: float = 2.5) -> DixonColesResult:
        """Full Dixon-Coles prediction for a single match."""
        lam, mu = self.expected_goals(home_team, away_team)
        table = self.scoreline_table(home_team, away_team, max_goals=max_goals)
        best = table.iloc[0]
        most_likely = str(best["scoreline"])
        most_likely_prob = float(best["probability"])
        home_win = table[table["home_goals"] > table["away_goals"]]["probability"].sum()
        draw = table[table["home_goals"] == table["away_goals"]]["probability"].sum()
        away_win = table[table["home_goals"] < table["away_goals"]]["probability"].sum()
        over = table[table["total_goals"] > over_under_threshold]["probability"].sum()
        under = 1.0 - over
        p_h0 = poisson.pmf(0, lam)
        p_a0 = poisson.pmf(0, mu)
        btts = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)
        return DixonColesResult(
            home_team=home_team, away_team=away_team, expected_home_goals=lam, expected_away_goals=mu,
            home_win_prob=round(home_win, 4), draw_prob=round(draw, 4), away_win_prob=round(away_win, 4),
            rho_used=round(self._rho, 4), most_likely_score=most_likely, most_likely_prob=round(most_likely_prob, 4),
            over_2_5_prob=round(over, 4), under_2_5_prob=round(under, 4), btts_prob=round(btts, 4), btts_no_prob=round(1.0 - btts, 4),
        )

    def predict_matches(self, df: pd.DataFrame, home_team_col: str = "home_team", away_team_col: str = "away_team", max_goals: int | None = None) -> pd.DataFrame:
        """Predict outcomes for all fixtures in a DataFrame."""
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            home = row[home_team_col]
            away = row[away_team_col]
            try:
                result = self.predict(home, away, max_goals=max_goals)
                records.append({"home_team": home, "away_team": away, "expected_home_goals": result.expected_home_goals, "expected_away_goals": result.expected_away_goals, "home_win_prob": result.home_win_prob, "draw_prob": result.draw_prob, "away_win_prob": result.away_win_prob, "most_likely_score": result.most_likely_score, "over_2_5_prob": result.over_2_5_prob, "under_2_5_prob": result.under_2_5_prob, "btts_prob": result.btts_prob, "btts_no_prob": result.btts_no_prob})
            except Exception as e:
                logger.warning("Prediction failed for %s vs %s: %s", home, away, e)
        return pd.DataFrame(records)

    def predict_proba(self, df: pd.DataFrame, home_team_col: str = "home_team", away_team_col: str = "away_team") -> np.ndarray:
        """Return match outcome probabilities as a (n, 3) array."""
        preds_df = self.predict_matches(df, home_team_col=home_team_col, away_team_col=away_team_col)
        n = len(preds_df)
        probs = np.zeros((n, 3))
        if "away_win_prob" in preds_df.columns:
            probs[:, 0] = preds_df["away_win_prob"].values
        if "draw_prob" in preds_df.columns:
            probs[:, 1] = preds_df["draw_prob"].values
        if "home_win_prob" in preds_df.columns:
            probs[:, 2] = preds_df["home_win_prob"].values
        row_sums = probs.sum(axis=1)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        probs = probs / row_sums[:, np.newaxis]
        return probs

    def evaluate(self, df_test: pd.DataFrame, home_team_col: str = "home_team", away_team_col: str = "away_team", home_goals_col: str = "home_goals", away_goals_col: str = "away_goals", over_under_threshold: float = 2.5) -> dict[str, Any]:
        """Evaluate the Dixon-Coles model on test data."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluating.")
        from sklearn.metrics import log_loss as sk_log_loss
        actual_hg = df_test[home_goals_col].values.astype(float)
        actual_ag = df_test[away_goals_col].values.astype(float)
        actual_result = df_test["result"].map({"A": 0, "D": 1, "H": 2}).values
        preds_df = self.predict_matches(df_test, home_team_col=home_team_col, away_team_col=away_team_col)
        probs = np.column_stack([preds_df["away_win_prob"].values, preds_df["draw_prob"].values, preds_df["home_win_prob"].values])
        pred_labels = np.argmax(probs, axis=1)
        accuracy = float(np.mean(pred_labels == actual_result))
        ll = sk_log_loss(actual_result, probs)
        y_onehot = np.zeros((len(actual_result), 3))
        for i, v in enumerate(actual_result):
            if not np.isnan(v) and 0 <= v <= 2:
                y_onehot[i, int(v)] = 1
        brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
        actual_btts = ((actual_hg > 0) & (actual_ag > 0)).astype(float)
        pred_btts_probs = preds_df["btts_prob"].values
        pred_btts = (pred_btts_probs > 0.5).astype(float)
        btts_accuracy = float(np.mean(pred_btts == actual_btts)) if len(actual_btts) > 0 else 0.0
        btts_brier = float(np.mean((pred_btts_probs - actual_btts) ** 2)) if len(actual_btts) > 0 else 0.0
        actual_total = actual_hg + actual_ag
        actual_ou = (actual_total > over_under_threshold).astype(float)
        ou_col = f"over_{over_under_threshold:.1f}_prob".replace(".", "_")
        pred_ou_probs = preds_df.get(ou_col, pd.Series([0.5] * len(df_test))).values
        pred_ou = (pred_ou_probs > 0.5).astype(float)
        ou_accuracy = float(np.mean(pred_ou == actual_ou)) if len(actual_ou) > 0 else 0.0
        ou_brier = float(np.mean((pred_ou_probs - actual_ou) ** 2)) if len(actual_ou) > 0 else 0.0
        ou_key = f"over_under_{over_under_threshold:.1f}_accuracy".replace(".", "_")
        ou_brier_key = f"over_under_{over_under_threshold:.1f}_brier".replace(".", "_")
        return {"accuracy": round(accuracy, 4), "log_loss": round(ll, 4), "brier_score": round(brier, 4), "btts_accuracy": round(btts_accuracy, 4), "btts_brier": round(btts_brier, 4), ou_key: round(ou_accuracy, 4), ou_brier_key: round(ou_brier, 4), "n_test": len(df_test)}

    def add_features(self, df: pd.DataFrame, home_team_col: str = "home_team", away_team_col: str = "away_team", home_goals_col: str = "home_goals", away_goals_col: str = "away_goals", date_col: str = "date", league_col: str = "league", round_col: str = "round", refit_every: int = 10) -> pd.DataFrame:
        """Add Dixon-Coles derived features to a DataFrame (leakage-free)."""
        from src.dixon_coles.fit import _fill_dc_row
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
        completed_pos = [i for i in range(n) if pd.notna(df.iloc[i][home_goals_col]) and pd.notna(df.iloc[i][away_goals_col])]
        if len(completed_pos) < 5:
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
        def _get_train_df(up_to_pos: int) -> pd.DataFrame:
            mask = [i <= up_to_pos and pd.notna(df.iloc[i][home_goals_col]) and pd.notna(df.iloc[i][away_goals_col]) for i in range(n)]
            return df.iloc[mask].copy()
        first_chunk_end_pos = min(refit_every - 1, len(completed_pos) - 1)
        first_cutoff_pos = completed_pos[first_chunk_end_pos]
        train_df = _get_train_df(first_cutoff_pos)
        ref_date = pd.to_datetime(df.iloc[first_cutoff_pos + 1][date_col]) if first_cutoff_pos + 1 < n and pd.notna(df.iloc[first_cutoff_pos + 1][date_col]) else None
        current_model = DixonColesModel(decay_halflife_days=self.decay_halflife_days, use_importance=self.use_importance, rho_fixed=self.rho_fixed, max_goals_table=self.max_goals_table)
        current_model.fit(train_df, date_col=date_col, league_col=league_col, round_col=round_col, reference_date=ref_date, verbose=False)
        for i in range(first_cutoff_pos):
            _fill_dc_row(df, current_model, i, home_team_col, away_team_col, exp_home, exp_away, home_attack, home_def, away_attack, away_def, hw_prob, d_prob, aw_prob, rho_vals)
        last_filled_pos = first_cutoff_pos
        _current_step = refit_every
        chunk_idx = first_chunk_end_pos + 1
        while chunk_idx < len(completed_pos):
            chunk_end_idx = min(chunk_idx + _current_step, len(completed_pos))
            cutoff_pos = completed_pos[chunk_end_idx - 1]
            train_df = _get_train_df(cutoff_pos)
            ref_date = pd.to_datetime(df.iloc[cutoff_pos + 1][date_col]) if cutoff_pos + 1 < n and pd.notna(df.iloc[cutoff_pos + 1][date_col]) else None
            current_model = DixonColesModel(decay_halflife_days=self.decay_halflife_days, use_importance=self.use_importance, rho_fixed=self.rho_fixed, max_goals_table=self.max_goals_table)
            current_model.fit(train_df, date_col=date_col, league_col=league_col, round_col=round_col, reference_date=ref_date, verbose=False)
            for i in range(last_filled_pos + 1, cutoff_pos):
                _fill_dc_row(df, current_model, i, home_team_col, away_team_col, exp_home, exp_away, home_attack, home_def, away_attack, away_def, hw_prob, d_prob, aw_prob, rho_vals)
            last_filled_pos = cutoff_pos
            chunk_idx = chunk_end_idx
            _current_step = min(_current_step * 2, 2000)
        if last_filled_pos < n - 1:
            for i in range(last_filled_pos + 1, n):
                _fill_dc_row(df, current_model, i, home_team_col, away_team_col, exp_home, exp_away, home_attack, home_def, away_attack, away_def, hw_prob, d_prob, aw_prob, rho_vals)
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
        logger.info("Dixon-Coles features added: %d matches, refit every %d, last ρ=%.3f", n, refit_every, rho_vals[completed_pos[-1]] if completed_pos else 0.0)
        return df
