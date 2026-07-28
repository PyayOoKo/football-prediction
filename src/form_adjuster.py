"""
RecentFormAdjuster — adjusts Elo ratings based on recent form.

Elo ratings are inherently slow to react — a team on a 5-game losing streak
still carries a high rating from past success. This module computes a
**form score** from the last N matches and translates it into a temporary
Elo adjustment (bonus for good form, penalty for bad) that gets applied
during prediction without modifying the underlying Elo ratings.

Usage
-----
    from src.form_adjuster import RecentFormAdjuster

    adjuster = RecentFormAdjuster(n_matches=6, form_weight=50.0)
    adjuster.fit(historical_df)

    # Then during prediction:
    adj = adjuster.get_form_adjustment("Helsingborgs IF")
    adjusted_rating = elo_rating + adj
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RecentFormAdjuster:
    """Adjusts Elo ratings based on recent match form.

    Computes a form score for each team based on their last N matches,
    then converts it to an Elo rating adjustment.

    Parameters
    ----------
    n_matches : int
        Number of recent matches to consider (default 6).
    form_weight : float
        Maximum Elo adjustment in Elo points (default 50.0).
        A team with perfect form (6 wins) gets +form_weight.
        A team with terrible form (6 losses) gets -form_weight.
        A team with league-average form gets 0.
    min_matches : int
        Minimum matches required to compute a meaningful form score.
        Teams with fewer matches get no adjustment (default 3).
    """

    def __init__(
        self,
        n_matches: int = 6,
        form_weight: float = 50.0,
        min_matches: int = 3,
    ) -> None:
        if n_matches < 1:
            raise ValueError(f"n_matches must be >= 1, got {n_matches}")
        if form_weight < 0:
            raise ValueError(f"form_weight must be >= 0, got {form_weight}")

        self.n_matches = n_matches
        self.form_weight = form_weight
        self.min_matches = min_matches
        self._form_cache: dict[str, float] = {}
        self._fitted: bool = False

    # ── Properties ────────────────────────────────────────

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def form_scores(self) -> dict[str, float]:
        """Return a copy of all computed form scores (-1 to +1)."""
        return dict(self._form_cache)

    # ── Fit ──────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> RecentFormAdjuster:
        """Compute form scores for all teams from historical match data.

        For each team, looks at their last N matches (time-ordered) and
        computes:
            form_score = (actual_points - expected_points) / max_possible_points

        Where:
        - actual_points = 3 per win, 1 per draw, 0 per loss
        - expected_points = 1.5 per match (league-average expectation)
        - max_possible_points = 3 * n_matches

        This form_score ranges from -1 (all losses) to +1 (all wins),
        centred at 0 (league-average form).

        Parameters
        ----------
        df : pd.DataFrame
            Match results. Must have columns: date, home_team, away_team, result.
            Should be sorted by date chronologically.

        Returns
        -------
        RecentFormAdjuster
            Self, with form scores computed.
        """
        if df.empty:
            logger.warning("Empty DataFrame — no form scores computed")
            return self
        if "result" not in df.columns:
            logger.warning("DataFrame has no 'result' column — no form scores computed")
            return self

        # Extract all teams
        all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        # Sort by date if not already
        if "date" in df.columns:
            df_sorted = df.sort_values("date")
        else:
            df_sorted = df

        for team in all_teams:
            # Get all matches involving this team, most recent first
            team_matches = df_sorted[
                (df_sorted["home_team"] == team) | (df_sorted["away_team"] == team)
            ].tail(self.n_matches)

            if len(team_matches) < self.min_matches:
                self._form_cache[team] = 0.0
                continue

            # Compute points from these matches
            total_points = 0.0
            for _, row in team_matches.iterrows():
                result = row["result"]
                is_home = row["home_team"] == team
                if result == "H":
                    total_points += 3.0 if is_home else 0.0
                elif result == "A":
                    total_points += 0.0 if is_home else 3.0
                elif result == "D":
                    total_points += 1.0

            max_possible = len(team_matches) * 3.0
            expected = len(team_matches) * 1.5  # league-average expectation

            # Form score: -1 to +1
            form_score = (total_points / max_possible) * 2.0 - 1.0
            form_score = float(np.clip(form_score, -1.0, 1.0))

            self._form_cache[team] = form_score

        self._fitted = True
        logger.debug(
            "Form scores computed for %d teams (n=%d, weight=%.1f)",
            len(self._form_cache), self.n_matches, self.form_weight,
        )
        return self

    # ── Query ────────────────────────────────────────────

    def get_form_adjustment(self, team: str) -> float:
        """Return Elo rating adjustment for a team based on recent form.

        Returns
        -------
        float
            Elo points to add (positive) or subtract (negative).
            Range: [-form_weight, +form_weight].
            Returns 0.0 if team has insufficient match history.
        """
        score = self._form_cache.get(team, 0.0)
        return score * self.form_weight

    def get_form_score(self, team: str) -> float:
        """Return the raw form score (-1 to +1) for a team."""
        return self._form_cache.get(team, 0.0)

    def adjust_rating(self, rating: float, team: str) -> float:
        """Apply form adjustment to an Elo rating.

        Parameters
        ----------
        rating : float
            Current Elo rating for the team.
        team : str
            Team name.

        Returns
        -------
        float
            Form-adjusted Elo rating.
        """
        return rating + self.get_form_adjustment(team)

    # ── Convenience ──────────────────────────────────────

    def get_form_report(self, teams: list[str] | None = None) -> pd.DataFrame:
        """Generate a readable form report for debugging.

        Parameters
        ----------
        teams : list[str], optional
            Specific teams to report. If None, shows all.

        Returns
        -------
        pd.DataFrame
            Columns: team, form_score, elo_adjustment, form_label
        """
        records = []
        target_teams = teams or list(self._form_cache.keys())
        for team in target_teams:
            score = self._form_cache.get(team, 0.0)
            adj = score * self.form_weight

            if score > 0.5:
                label = "Hot"
            elif score > 0.15:
                label = "Good"
            elif score > -0.15:
                label = "Neutral"
            elif score > -0.5:
                label = "Cold"
            else:
                label = "Ice Cold"

            records.append({
                "team": team,
                "form_score": round(score, 3),
                "elo_adjustment": round(adj, 1),
                "form_label": label,
            })

        return pd.DataFrame(records).sort_values("form_score", ascending=False)
