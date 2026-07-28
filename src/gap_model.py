"""
GAP (Goal-Adjusted Performance) Ratings for Over/Under 2.5 Prediction.

Based on Wheatcroft (2020) "A Profitable Model for Predicting the Over/Under
Market in Football" — International Journal of Forecasting, 36(3), 916–932.

Core idea
---------
Instead of modelling goals directly (which are rare and noisy), GAP ratings
model the *precursors* to goals — higher-frequency match statistics like
shots on target, total shots, or corners.  Team strength is expressed as
four time-varying ratings per team:

    home_attack, home_defence, away_attack, away_defence

Ratings are updated **iteratively** after each match (like Elo), using the
prediction error between expected and observed statistics.  The predicted
statistics are then fed into a logistic regression to estimate P(Over 2.5).

Usage
-----
    from src.gap_model import GAPModel

    model = GAPModel(k=32, stat_column="home_shots_target")
    model.fit(match_data)
    probs = model.predict_over_25(fixtures_df)

    # Or for a single match:
    prob = model.predict_single("Team A", "Team B")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class GAPModel:
    """Goal-Adjusted Performance ratings for Over/Under 2.5 prediction.

    Each team maintains four ratings (home_attack, home_defence,
    away_attack, away_defence) updated iteratively using observed
    match statistics.

    Parameters
    ----------
    k : float
        Learning rate — how quickly ratings adjust to new information.
        Higher = more responsive (default 32, same as standard Elo).
    initial_rating : float
        Starting rating for new teams (default 1500).
    stat_column : str
        The column name for the match statistic to rate teams on.
        Must match column names in the DataFrame passed to ``fit()``.
        The column is used for **both** home and away stats.
        Examples: ``home_shots_target``, ``home_shots``, ``home_corners``.
    home_adv : float
        Home advantage bonus added to home attack rating (default 50).
        Captures the fact that teams perform better at home.
    decay_halflife_days : float
        Time-decay halflife for rating updates.  A match this many days
        ago gets 50 % weight in the rating update.  0 = no decay.
        (default 365 — 1 year).
    """

    k: float = 8.0  # Lower K for stat-based ratings (K=32 was for goals/elo)
    initial_rating: float = 5.0  # Centre rating for expected stat value
    stat_column: str = "home_shots_target"
    home_adv: float = 0.5  # Home advantage in stat units (e.g. +0.5 shots on target)
    decay_halflife_days: float = 365.0

    # ── Internal state ────────────────────────────────────
    # team -> {home_attack, home_defence, away_attack, away_defence}
    # Ratings represent expected stat contribution (not Elo points)
    _ratings: dict[str, dict[str, float]] | None = None
    # Fitted logistic regression: expected_stats -> P(Over 2.5)
    _logistic: LogisticRegression | None = None
    # Track league-average stat for normalisation
    _stat_mean: float = 0.0
    _stat_std: float = 1.0
    _fitted: bool = False
    _n_matches: int = 0

    # ── Rating helpers ────────────────────────────────────

    def _init_team(self, team: str) -> None:
        """Ensure a team has initialised ratings."""
        if self._ratings is None:
            self._ratings = {}
        if team not in self._ratings:
            # Ratings are in standardised stat units (z-score-like)
            # Home ratings get a small home advantage bonus;
            # away ratings do not (Wheatcroft 2020).
            self._ratings[team] = {
                "home_attack": self.initial_rating + self.home_adv,
                "home_defence": self.initial_rating + self.home_adv,
                "away_attack": self.initial_rating,
                "away_defence": self.initial_rating,
            }

    def _get_rating(self, team: str, key: str) -> float:
        self._init_team(team)
        return self._ratings[team][key]  # type: ignore[index]

    # ── Expected statistics ───────────────────────────────

    def _expected_stat_std(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return expected stats in standardised units (internal use).

        Formula (Wheatcroft 2020):
            E[home_stat] = (home_attack_home + away_defence_away) / 2
            E[away_stat] = (away_attack_away + home_defence_home) / 2

        Returns values in standardised z-score units (ratings are stored in
        standardised units internally).
        """
        ha = self._get_rating(home_team, "home_attack")
        ad = self._get_rating(away_team, "away_defence")
        aa = self._get_rating(away_team, "away_attack")
        hd = self._get_rating(home_team, "home_defence")

        exp_home = (ha + ad) / 2.0
        exp_away = (aa + hd) / 2.0
        return exp_home, exp_away

    def expected_stat(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return expected (home_stat, away_stat) in raw stat units.

        Converts internal standardised units to raw stat units for external
        use (e.g. logistic regression features).
        """
        exp_home_std, exp_away_std = self._expected_stat_std(home_team, away_team)

        exp_home_raw = exp_home_std * self._stat_std + self._stat_mean
        exp_away_raw = exp_away_std * self._stat_std + self._stat_mean

        # Ensure non-negative (stats like shots/corners can't be negative)
        exp_home_raw = max(exp_home_raw, 0.01)
        exp_away_raw = max(exp_away_raw, 0.01)

        return exp_home_raw, exp_away_raw

    # ── Rating update ─────────────────────────────────────

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        observed_home_std: float,
        observed_away_std: float,
        weight: float = 1.0,
    ) -> None:
        """Update GAP ratings based on observed vs expected stats.

        All values must be in STANDARDISED units.

        Formula:
            new_rating = old_rating + k * weight * (observed - expected)
        """
        exp_home_std, exp_away_std = self._expected_stat_std(home_team, away_team)
        k_weighted = self.k * weight

        # Home team: attack updated by home stat, defence by away stat
        ha = self._get_rating(home_team, "home_attack")
        hd = self._get_rating(home_team, "home_defence")
        self._ratings[home_team]["home_attack"] = ha + k_weighted * (observed_home_std - exp_home_std)  # type: ignore[index]
        self._ratings[home_team]["home_defence"] = hd + k_weighted * (observed_away_std - exp_away_std)  # type: ignore[index]

        # Away team: attack updated by away stat, defence by home stat
        aa = self._get_rating(away_team, "away_attack")
        ad = self._get_rating(away_team, "away_defence")
        self._ratings[away_team]["away_attack"] = aa + k_weighted * (observed_away_std - exp_away_std)  # type: ignore[index]
        self._ratings[away_team]["away_defence"] = ad + k_weighted * (observed_home_std - exp_home_std)  # type: ignore[index]

    # ── Fit (iterative rating computation + logistic regression) ──

    def fit(
        self,
        df: pd.DataFrame,
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        date_col: str = "date",
    ) -> GAPModel:
        """Fit the GAP model on historical match data.

        Two-stage process:
            1. Iterate chronologically, updating GAP ratings after each match
            2. Train logistic regression: expected_stats -> P(Over 2.5)

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: home_team_col, away_team_col,
            {stat_column} (home stat), {stat_column.replace('home_', 'away_')}
            (away stat), ``date``, ``home_goals``, ``away_goals``.
        """
        df = df.copy()
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)

        # Resolve stat column names
        home_stat = self.stat_column
        away_stat = home_stat.replace("home_", "away_")
        if away_stat == home_stat:
            raise ValueError(
                f"stat_column '{home_stat}' must start with 'home_' "
                f"so that 'away_' variant can be derived"
            )

        required = [home_team_col, away_team_col, home_stat, away_stat,
                    "home_goals", "away_goals"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        n = len(df)
        self._ratings = {}
        self._n_matches = n

        # Pre-compute stat mean/std for normalisation
        home_stat_vals = df[home_stat].values.astype(float)
        away_stat_vals = df[away_stat].values.astype(float)
        all_vals = np.concatenate([home_stat_vals, away_stat_vals])
        all_vals = all_vals[~np.isnan(all_vals)]
        self._stat_mean = float(np.mean(all_vals)) if len(all_vals) > 0 else 5.0
        self._stat_std = float(np.std(all_vals)) if len(all_vals) > 0 else 3.0
        if self._stat_std < 0.1:
            self._stat_std = 1.0

        logger.info(
            "Stat normalisation: mean=%.2f, std=%.2f (%s)",
            self._stat_mean, self._stat_std, home_stat,
        )

        # Stage 1: Iterative rating updates
        features: list[dict[str, float]] = []
        labels: list[int] = []

        # For time-decay weight, use a reference date (last match date + 1 day)
        if date_col in df.columns:
            max_date = pd.to_datetime(df[date_col]).max()
            reference_date = max_date + pd.Timedelta(days=1)
        else:
            reference_date = None

        for i in range(n):
            row = df.iloc[i]
            ht = str(row[home_team_col])
            at = str(row[away_team_col])

            # Normalise observed stats to standardised units
            obs_home_raw = float(row[home_stat])
            obs_away_raw = float(row[away_stat])
            if np.isnan(obs_home_raw) or np.isnan(obs_away_raw):
                continue

            # Standardise: z-score for rating update (ratings are in std units)
            obs_home_std = (obs_home_raw - self._stat_mean) / self._stat_std
            obs_away_std = (obs_away_raw - self._stat_mean) / self._stat_std
            obs_home_std = np.clip(obs_home_std, -5.0, 5.0)
            obs_away_std = np.clip(obs_away_std, -5.0, 5.0)

            # Get pre-match expected stats in STANDARDISED units
            exp_home_std, exp_away_std = self._expected_stat_std(ht, at)

            # Store features for logistic regression (in RAW stat units)
            exp_home_raw = exp_home_std * self._stat_std + self._stat_mean
            exp_away_raw = exp_away_std * self._stat_std + self._stat_mean
            exp_home_raw = max(exp_home_raw, 0.01)
            exp_away_raw = max(exp_away_raw, 0.01)

            total_exp = exp_home_raw + exp_away_raw
            features.append({
                "exp_home_stat": exp_home_raw,
                "exp_away_stat": exp_away_raw,
                "exp_total_stat": total_exp,
                "exp_diff": exp_home_raw - exp_away_raw,
            })

            # Label: was the total goals > 2.5?
            total_goals = float(row["home_goals"]) + float(row["away_goals"])
            labels.append(1 if total_goals > 2.5 else 0)

            # Compute recency weight from reference date (standard Elo-style decay)
            weight = 1.0
            if self.decay_halflife_days > 0 and reference_date is not None and date_col in df.columns:
                try:
                    match_date = pd.to_datetime(row[date_col])
                    days_ago = (reference_date - match_date).days
                    days_ago = max(days_ago, 0)
                    weight = np.exp(-np.log(2) * days_ago / self.decay_halflife_days)
                    weight = max(weight, 0.01)
                except Exception:
                    weight = 1.0

            # Update ratings using STANDARDISED stats
            self.update_ratings(ht, at, obs_home_std, obs_away_std, weight=weight)

        # Stage 2: Logistic regression
        X = np.array([[f["exp_home_stat"], f["exp_away_stat"],
                       f["exp_total_stat"], f["exp_diff"]]
                      for f in features])
        y = np.array(labels)

        self._logistic = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced",
        )
        self._logistic.fit(X, y)
        self._fitted = True

        # Report
        train_score = self._logistic.score(X, y)
        logger.info(
            "GAP model fitted on %d matches (%s). "
            "LR train accuracy: %.1f%%",
            n, self.stat_column, train_score * 100,
        )
        return self

    # ── Prediction ────────────────────────────────────────

    def predict_single(self, home_team: str, away_team: str) -> float:
        """Return P(Over 2.5) for a single match."""
        if not self._fitted or self._logistic is None:
            raise RuntimeError("Model must be fitted before predicting")

        exp_home, exp_away = self.expected_stat(home_team, away_team)
        X = np.array([[exp_home, exp_away,
                       exp_home + exp_away, exp_home - exp_away]])
        prob = self._logistic.predict_proba(X)[0, 1]
        return float(prob)

    def predict(self, df: pd.DataFrame,
                home_team_col: str = "home_team",
                away_team_col: str = "away_team") -> pd.DataFrame:
        """Return DataFrame with P(Over 2.5) for each fixture row."""
        if not self._fitted or self._logistic is None:
            raise RuntimeError("Model must be fitted before predicting")

        results = []
        for _, row in df.iterrows():
            ht = str(row[home_team_col])
            at = str(row[away_team_col])
            prob = self.predict_single(ht, at)
            results.append({
                "home_team": ht,
                "away_team": at,
                "over_2_5_prob": prob,
                "under_2_5_prob": 1.0 - prob,
            })
        return pd.DataFrame(results)

    def predict_proba(self, df: pd.DataFrame,
                      home_team_col: str = "home_team",
                      away_team_col: str = "away_team") -> np.ndarray:
        """Return P(Over 2.5) as a 1D numpy array aligned to df rows."""
        preds = self.predict(df, home_team_col, away_team_col)
        return preds["over_2_5_prob"].values

    # ── Evaluation ────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame,
                 home_team_col: str = "home_team",
                 away_team_col: str = "away_team") -> dict[str, float]:
        """Evaluate GAP model on test data.

        Returns Brier score, accuracy, log loss for O/U 2.5.
        """
        from sklearn.metrics import log_loss as sk_ll

        probs = self.predict_proba(df, home_team_col, away_team_col)
        actual = ((df["home_goals"].values + df["away_goals"].values) > 2.5).astype(float)

        brier = float(np.mean((probs - actual) ** 2))
        acc = float(np.mean((probs > 0.5).astype(float) == actual))
        ll = sk_ll(actual, np.column_stack([1 - probs, probs]))

        return {
            "brier": round(brier, 4),
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "n_matches": len(df),
            "over_rate": float(actual.mean()),
        }
