"""
Three-Model Blend — Market-specific ensemble of Poisson + Elo + XGBoost.

Combines predictions from three fundamentally different model types:
- **Poisson** (statistical scoring distribution) → best for goal-line & BTTS
- **Elo** (dynamic team strength ratings) → stable long-term prior
- **XGBoost** (gradient-boosted ML) → best for complex feature interactions

Market-specific weights allow each model to contribute where it excels.

Usage
-----
::

    from src.models.three_model_blend import ThreeModelBlend
    from src.poisson_model import PoissonModel
    from src.elo import EloSystem
    import joblib

    poisson = PoissonModel().fit(df_train)
    elo = EloSystem()
    elo.process_matches(df_train)
    xgb = joblib.load("models/xgboost_model.joblib")

    blend = ThreeModelBlend(poisson, elo, xgb)

    # Predict a single fixture
    result = blend.predict("France", "England")

    # Optimise weights for each market
    blend.optimise_weights(df_train, df_val)

    # Evaluate per-market
    metrics = blend.evaluate(df_test)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Default Weights (Hypothesis-driven)
# ═══════════════════════════════════════════════════════════

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    # --- Optimised via exhaustive grid search on All Top 5 Leagues + World Cup ---
    # 5 seasons each for EPL, La Liga, Bundesliga, Serie A, Ligue 1 (2016-2024),
    # plus 7 World Cups (2002-2026). Total: 9,189 preprocessed matches.
    # See config/three_model_weights_league_wc_*.json for full optimisation results.
    #
    # Key insights from Top 5 league retraining:
    # - 1X2: Poisson increased (0.51), XGBoost minimal (0.06) for broad league coverage
    # - Over2.5: Elo dominates (0.48) with more league data -- stable ratings help totals
    # - BTTS: XGBoost stable (0.55) -- ML features consistently best for BTTS
    # - Over3.5: Poisson + XGBoost equal (0.50/0.50) -- stable across all league mixtures
    "1X2": {"poisson": 0.51, "elo": 0.43, "xgb": 0.06},
    # Over/Under: Elo weight increased to 0.48 with Top 5 data.
    # More league data strengthens Elo's contribution to totals.
    "Over2.5": {"poisson": 0.38, "elo": 0.48, "xgb": 0.14},
    # Over3.5: Stable result across all runs -- Poisson and XGBoost equally contribute.
    "Over3.5": {"poisson": 0.50, "elo": 0.00, "xgb": 0.50},
    # BTTS: XGBoost weight stable at 0.55 across all retraining runs.
    # This is the most consistent finding -- XGBoost dominates BTTS.
    "BTTS": {"poisson": 0.29, "elo": 0.16, "xgb": 0.55},
}

WEIGHT_SEARCH_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "1X2": {
        "poisson": (0.30, 0.70),
        "elo": (0.15, 0.50),
        "xgb": (0.05, 0.30),
    },
    "Over2.5": {
        # Elo excluded from totals (weight fixed at 0)
        "poisson": (0.20, 0.70),
        "elo": (0.00, 0.00),
        "xgb": (0.30, 0.80),
    },
    "Over3.5": {
        # Rare event (37.6% base rate XGBoost base rate ~28%); Elo excluded
        # Search widened for XGBoost based on deep-dive finding that XGBoost
        # alone (Brier 0.2106) outperforms the old 50/50 blend (0.2606).
        "poisson": (0.10, 0.50),
        "elo": (0.00, 0.00),
        "xgb": (0.50, 0.90),
    },
    "BTTS": {
        "poisson": (0.20, 0.60),
        "elo": (0.15, 0.40),
        "xgb": (0.15, 0.45),
    },
}


# ═══════════════════════════════════════════════════════════
#  Conditional Rate Cache
# ═══════════════════════════════════════════════════════════


@dataclass
class ConditionalRates:
    """Conditional BTTS, O/U, and mean total goals given match outcome.

    Used to derive BTTS and O/U probabilities from any model's 1X2 predictions.
    Also provides expected total goals per outcome for the Poisson CDF approach
    to O/U probability conversion.
    """

    btts_given_home_win: float = 0.50
    btts_given_draw: float = 0.70
    btts_given_away_win: float = 0.40
    ou_given_home_win: float = 0.55
    ou_given_draw: float = 0.40
    ou_given_away_win: float = 0.50
    # Mean total goals conditional on each outcome (for Poisson CDF conversion)
    mean_total_given_home_win: float = 2.50
    mean_total_given_draw: float = 2.00
    mean_total_given_away_win: float = 2.30

    @classmethod
    def from_data(cls, df: pd.DataFrame) -> "ConditionalRates":
        if df.empty or "result" not in df.columns:
            return cls()
        hw = df[df["result"] == "H"]
        dr = df[df["result"] == "D"]
        aw = df[df["result"] == "A"]

        def _btts(g: pd.DataFrame) -> float:
            if len(g) == 0:
                return 0.50
            return float(((g["home_goals"] > 0) & (g["away_goals"] > 0)).mean())

        def _ou(g: pd.DataFrame) -> float:
            if len(g) == 0:
                return 0.50
            return float(((g["home_goals"] + g["away_goals"]) > 2.5).mean())

        def _mean_total(g: pd.DataFrame) -> float:
            if len(g) == 0:
                return 2.50
            return float((g["home_goals"] + g["away_goals"]).mean())

        return cls(
            btts_given_home_win=_btts(hw),
            btts_given_draw=_btts(dr),
            btts_given_away_win=_btts(aw),
            ou_given_home_win=_ou(hw),
            ou_given_draw=_ou(dr),
            ou_given_away_win=_ou(aw),
            mean_total_given_home_win=_mean_total(hw),
            mean_total_given_draw=_mean_total(dr),
            mean_total_given_away_win=_mean_total(aw),
        )

    def btts_from_1x2(self, probs: np.ndarray) -> np.ndarray:
        return (
            probs[:, 2] * self.btts_given_home_win
            + probs[:, 1] * self.btts_given_draw
            + probs[:, 0] * self.btts_given_away_win
        )

    def ou_from_1x2(self, probs: np.ndarray, thresh: float = 2.5) -> np.ndarray:
        scale = thresh / 2.5
        ou_hw = min(self.ou_given_home_win * scale, 0.95)
        ou_dr = min(self.ou_given_draw * scale, 0.95)
        ou_aw = min(self.ou_given_away_win * scale, 0.95)
        return probs[:, 2] * ou_hw + probs[:, 1] * ou_dr + probs[:, 0] * ou_aw


# ═══════════════════════════════════════════════════════════
#  Feature Builder
# ═══════════════════════════════════════════════════════════


class _FeatureBuilder:
    """Build feature vectors for XGBoost from team names."""

    def __init__(self, historical_df: pd.DataFrame | None = None):
        self._historical_data = historical_df
        self._feature_cols: list[str] = []

    def set_historical_data(self, df: pd.DataFrame) -> None:
        self._historical_data = df

    def build(self, home_teams: list[str], away_teams: list[str]) -> pd.DataFrame | None:
        if self._historical_data is None or self._historical_data.empty:
            logger.warning("No historical data for feature engineering")
            return None
        try:
            from src.feature_engineering import build_features
            today_str = datetime.now().strftime("%Y-%m-%d")
            fixture_rows = []
            for ht, at in zip(home_teams, away_teams):
                row = {
                    "date": pd.Timestamp(today_str),
                    "home_team": ht, "away_team": at,
                    "result": "H", "home_goals": 0, "away_goals": 0,
                }
                for col in self._historical_data.columns:
                    if col not in row:
                        row[col] = self._historical_data[col].iloc[-1] if len(self._historical_data) > 0 else 0
                fixture_rows.append(row)
            df_ext = pd.concat([self._historical_data, pd.DataFrame(fixture_rows)], ignore_index=True)
            X_full, _ = build_features(df_ext, is_training=False)
            n_hist = len(self._historical_data)
            self._feature_cols = list(X_full.columns)
            return X_full.iloc[n_hist:].copy()
        except Exception as exc:
            logger.warning("Feature engineering failed: %s", exc)
            return None


# ═══════════════════════════════════════════════════════════
#  Pre-computed Predictions Container
# ═══════════════════════════════════════════════════════════


@dataclass
class PerModelPredictions:
    """Pre-computed predictions from each individual model."""

    pois_1x2: np.ndarray  # (n, 3) [away, draw, home]
    elo_1x2: np.ndarray  # (n, 3)
    xgb_1x2: np.ndarray  # (n, 3)

    # Poisson-specific binary market predictions (most accurate source)
    pois_btts: np.ndarray  # (n,)
    elo_btts: np.ndarray  # (n,)  — direct Poisson BTTS from Elo-derived expected goals
    xgb_btts: np.ndarray  # (n,)  — Poisson BTTS from XGBoost 1X2 → expected goals
    pois_over_25: np.ndarray  # (n,)
    pois_over_35: np.ndarray  # (n,)
    pois_exp_home: np.ndarray  # (n,)
    pois_exp_away: np.ndarray  # (n,)

    n: int

    @property
    def pois_total_goals(self) -> np.ndarray:
        return self.pois_exp_home + self.pois_exp_away


# ═══════════════════════════════════════════════════════════
#  ThreeModelBlend
# ═══════════════════════════════════════════════════════════


class ThreeModelBlend:
    """Market-specific blend of Poisson + Elo + XGBoost.

    Parameters
    ----------
    poisson_model : PoissonModel
        Fitted Poisson model.
    elo_system : EloSystem
        Processed Elo system.
    xgb_model : Any
        Fitted XGBoost/sklearn classifier with ``predict_proba(X)``.
    weights : dict, optional
        Per-market weights. Falls back to ``DEFAULT_WEIGHTS``.
    conditional_rates : ConditionalRates, optional
        Pre-computed conditional rates.
    historical_df : pd.DataFrame, optional
        Historical data for XGBoost feature building.
    """

    def __init__(
        self,
        poisson_model: Any,
        elo_model: Any,
        xgb_model: Any,
        weights: dict[str, dict[str, float]] | None = None,
        conditional_rates: ConditionalRates | None = None,
        historical_df: pd.DataFrame | None = None,
    ):
        self.poisson = poisson_model
        self.elo = elo_model
        self.xgb = xgb_model
        self.weights = weights or {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
        self.cond_rates = conditional_rates or ConditionalRates()
        self._feature_builder = _FeatureBuilder(historical_df)
        self._cache: dict[str, PerModelPredictions] = {}

    # ── Properties ────────────────────────────────────────

    @property
    def available_markets(self) -> list[str]:
        return list(self.weights.keys())

    @property
    def fitted(self) -> bool:
        poisson_ok = hasattr(self.poisson, "_fitted") and self.poisson._fitted
        elo_ok = hasattr(self.elo, "_ratings") and len(self.elo._ratings) > 0
        return poisson_ok and elo_ok and self.xgb is not None

    # ── Single Fixture Prediction ─────────────────────────

    def predict(self, home_team: str, away_team: str) -> dict[str, Any]:
        """Predict all markets for a single fixture using the 3-model blend."""
        probs_1x2 = self.predict_1x2(home_team, away_team)
        over_under = self.predict_over_under(home_team, away_team, 2.5)
        over_35 = self.predict_over_under(home_team, away_team, 3.5)
        btts = self.predict_btts(home_team, away_team)

        expectations = {}
        if hasattr(self.poisson, "expected_goals"):
            try:
                λ_h, λ_a = self.poisson.expected_goals(home_team, away_team)
                expectations = {
                    "expected_home_goals": round(λ_h, 4),
                    "expected_away_goals": round(λ_a, 4),
                    "expected_total_goals": round(λ_h + λ_a, 4),
                }
            except Exception:
                pass

        return {
            "home_team": home_team, "away_team": away_team,
            "1x2": probs_1x2,
            "over_under": over_under,
            "over_3_5": over_35,
            "btts": btts,
            "expected_goals": expectations,
        }

    # ── 1X2 Market ────────────────────────────────────────

    def predict_1x2(self, home_team: str, away_team: str) -> dict[str, float]:
        """Predict match outcome probabilities (1X2) using the 3-model blend.

        Gets predictions from all 3 models via their ``predict_proba()``
        interfaces, blends them using market-specific weights for '1X2',
        and renormalises so probabilities sum to 1.0.

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name.

        Returns
        -------
        dict[str, float]
            ``{'H': home_win_prob, 'D': draw_prob, 'A': away_win_prob}``
        """
        w = self.weights.get("1X2", DEFAULT_WEIGHTS["1X2"])

        # Get individual model predictions (each as np array [away, draw, home])
        p_pois = self._poisson_1x2(home_team, away_team)  # [away, draw, home]
        p_elo = self._elo_1x2(home_team, away_team)       # [away, draw, home]
        p_xgb = self._xgb_1x2(home_team, away_team)       # [away, draw, home]

        # Weighted blend
        p_h = w["poisson"] * p_pois[2] + w["elo"] * p_elo[2] + w["xgb"] * p_xgb[2]
        p_d = w["poisson"] * p_pois[1] + w["elo"] * p_elo[1] + w["xgb"] * p_xgb[1]
        p_a = w["poisson"] * p_pois[0] + w["elo"] * p_elo[0] + w["xgb"] * p_xgb[0]

        # Renormalise to ensure sum = 1.0
        total = p_h + p_d + p_a
        if total > 0:
            p_h /= total
            p_d /= total
            p_a /= total

        return {"H": p_h, "D": p_d, "A": p_a}

    # ── Over/Under Market ─────────────────────────────────

    def predict_over_under(self, home_team: str, away_team: str, threshold: float = 2.5) -> dict[str, float]:
        """Predict Over/Under probabilities for a given goal threshold.

        Uses the 3-model blend with Elo excluded (weight = 0 for totals):
        - **Poisson**: exact P(Over) from scoreline probability table
        - **XGBoost**: expected total goals → Poisson CDF conversion

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name.
        threshold : float
            Goal threshold: 2.5, 3.5, etc.

        Returns
        -------
        dict[str, float]
            ``{'Over': over_prob, 'Under': under_prob}`` (sums to 1.0).
        """
        # Build the market key matching the DEFAULT_WEIGHTS convention (e.g. "Over3.5")
        market_key = f"Over{threshold:.1f}"
        w = self.weights.get(market_key, self.weights.get("Over2.5", DEFAULT_WEIGHTS["Over2.5"]))

        # Poisson: exact P(Over) from scoreline table
        over_pois = self._poisson_over(home_team, away_team, threshold)

        # XGBoost: expected total goals → Poisson CDF
        over_xgb = self._xgb_over(home_team, away_team, threshold)

        # Elo: NOT used for totals (weight fixed at 0)
        # Blend only Poisson + XGBoost
        wp = w.get("poisson", 0)
        wx = w.get("xgb", 0)
        total_w = wp + wx
        if total_w > 0:
            over_blend = (wp * over_pois + wx * over_xgb) / total_w
        else:
            over_blend = 0.5

        return {"Over": over_blend, "Under": 1.0 - over_blend}

    # ── BTTS Market ───────────────────────────────────────

    def predict_btts(self, home_team: str, away_team: str) -> dict[str, float]:
        """Predict Both Teams To Score probability using the 3-model blend.

        Uses **direct** BTTS modelling for each model:
        - **Poisson**: exact BTTS from scoreline probability table
        - **Elo**: Poisson BTTS formula from Elo-derived expected goals
        - **XGBoost**: Poisson BTTS formula from XGBoost 1X2 → expected goals → Poisson CDF

        All three models now compute BTTS via the Poisson formula:
            P(BTTS) = 1 - e^{-λ_home} - e^{-λ_away} + e^{-(λ_home + λ_away)}

        The old conditional-rate fallback has been fully replaced.

        Parameters
        ----------
        home_team : str
            Home team name.
        away_team : str
            Away team name.

        Returns
        -------
        dict[str, float]
            ``{'BTTS': btts_prob, 'No BTTS': no_btts_prob}`` (sums to 1.0).
        """
        w = self.weights.get("BTTS", DEFAULT_WEIGHTS["BTTS"])

        # Poisson: exact BTTS from scoreline table
        btts_pois = self._poisson_btts(home_team, away_team)

        # Elo: direct BTTS from Elo-derived expected goals
        btts_elo = self._elo_btts(home_team, away_team)

        # XGBoost: Poisson BTTS from XGBoost 1X2 → expected goals
        btts_xgb = self._xgb_btts(home_team, away_team)

        # Blend with graceful skip: if a model returned None, redistribute
        wp = w.get("poisson", 0)
        we = w.get("elo", 0) if btts_elo is not None else 0.0
        wx = w.get("xgb", 0) if btts_xgb is not None else 0.0
        total_w = wp + we + wx

        if total_w > 0:
            btts_blend = (
                wp * btts_pois
                + (we * btts_elo if btts_elo is not None else 0.0)
                + (wx * btts_xgb if btts_xgb is not None else 0.0)
            ) / total_w
        else:
            btts_blend = btts_pois  # Fallback to Poisson alone

        return {"BTTS": btts_blend, "No BTTS": 1.0 - btts_blend}

    def _elo_btts(self, home_team: str, away_team: str) -> float | None:
        """Get Elo's BTTS probability using direct Poisson BTTS formula.

        Uses Elo's dedicated ``predict_btts()`` method which computes
        expected goals from the Elo rating difference, then applies
        the Poisson BTTS formula:
            P(BTTS) = 1 - e^{-λ_home} - e^{-λ_away} + e^{-(λ_home + λ_away)}

        This replaced the old conditional-rate fallback.
        """
        try:
            if hasattr(self.elo, "predict_btts") and callable(self.elo.predict_btts):
                result = self.elo.predict_btts(home_team, away_team)
                if result is not None:
                    return float(result)
        except Exception:
            pass
        return None

    def _xgb_btts(self, home_team: str, away_team: str) -> float | None:
        """Get XGBoost's BTTS probability via 1X2 → expected goals → Poisson formula.

        Strategy (replaces old conditional-rate approach):
        1. Get XGBoost's 1X2 probabilities from ``predict_proba()``.
        2. Compute expected total goals from outcome-conditional mean totals.
        3. Estimate home/away goal split (55/45 default).
        4. Apply Poisson BTTS formula:
           P(BTTS) = 1 - e^{-λ_home} - e^{-λ_away} + e^{-(λ_home + λ_away)}

        This is more principled than the old conditional-rate approach because
        it models the explicit goal distribution via Poisson.
        """
        try:
            xgb_1x2 = self._xgb_1x2(home_team, away_team)  # [away, draw, home]
            cr = self.cond_rates
            # Expected total goals from XGBoost's outcome distribution
            exp_total = (
                xgb_1x2[2] * cr.mean_total_given_home_win
                + xgb_1x2[1] * cr.mean_total_given_draw
                + xgb_1x2[0] * cr.mean_total_given_away_win
            )
            if exp_total <= 0:
                return 0.50
            # Split expected total into home/away (~55/45 home advantage split)
            exp_home = exp_total * 0.55
            exp_away = exp_total * 0.45
            # Poisson BTTS formula
            p_h0 = np.exp(-exp_home)
            p_a0 = np.exp(-exp_away)
            btts_prob = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)
            return float(np.clip(btts_prob, 0.0, 1.0))
        except Exception:
            return 0.50

    # ── Individual Model Proxies ──────────────────────────

    def _poisson_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        try:
            r = self.poisson.predict(home_team, away_team)
            return np.array([r.get("away_win_prob", 0.33), r.get("draw_prob", 0.34), r.get("home_win_prob", 0.33)])
        except Exception:
            return np.array([0.33, 0.34, 0.33])

    def _poisson_over(self, home_team: str, away_team: str, threshold: float) -> float:
        try:
            key = f"over_{threshold:.1f}_prob".replace(".", "_")
            r = self.poisson.predict(home_team, away_team, over_under_threshold=threshold)
            return float(r.get(key, 0.50))
        except Exception:
            return 0.50

    def _poisson_btts(self, home_team: str, away_team: str) -> float:
        try:
            r = self.poisson.predict(home_team, away_team)
            return float(r.get("btts_prob", 0.50))
        except Exception:
            return 0.50

    def _elo_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        try:
            df = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
            return self.elo.predict_proba(df)[0]
        except Exception:
            return np.array([0.33, 0.34, 0.33])

    def _xgb_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        try:
            X = self._feature_builder.build([home_team], [away_team])
            if X is not None and len(X) > 0:
                return self.xgb.predict_proba(X)[0]
        except Exception:
            pass
        return np.array([0.33, 0.34, 0.33])

    def _xgb_over(self, home_team: str, away_team: str, threshold: float) -> float:
        """Get XGBoost's P(Over threshold) via expected total goals → Poisson CDF.

        Strategy:
        1. Get XGBoost's 1X2 probabilities from ``predict_proba()``.
        2. Compute expected total goals as the weighted average of
           outcome-conditional mean totals (from ``cond_rates``).
        3. Convert to P(Over) using Poisson CDF:
           P(Over) = 1 - P(X <= floor(threshold)) where X ~ Pois(expected_total).

        This is more principled than deriving P(Over) directly from 1X2 probs
        because it models the full goal distribution via Poisson.
        """
        try:
            xgb_1x2 = self._xgb_1x2(home_team, away_team)  # [away, draw, home]
            cr = self.cond_rates
            # Expected total goals from XGBoost's outcome distribution
            exp_total = (
                xgb_1x2[2] * cr.mean_total_given_home_win
                + xgb_1x2[1] * cr.mean_total_given_draw
                + xgb_1x2[0] * cr.mean_total_given_away_win
            )
            if exp_total <= 0:
                return 0.50
            return 1.0 - _poisson_cdf(threshold, exp_total)
        except Exception:
            return 0.50

    # ── Batch Prediction ──────────────────────────────────

    def predict_matches(self, df: pd.DataFrame, home_col: str = "home_team", away_col: str = "away_team") -> pd.DataFrame:
        """Predict all markets for multiple fixtures in batch."""
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            home, away = row[home_col], row[away_col]
            result = self.predict(home, away)
            p = result["1x2"]  # {'H': ..., 'D': ..., 'A': ...}
            flat = {
                "home_team": home, "away_team": away,
                "home_win_prob": p["H"],
                "draw_prob": p["D"],
                "away_win_prob": p["A"],
                "over_2_5_prob": result["over_under"]["Over"],
                "under_2_5_prob": result["over_under"]["Under"],
                "over_3_5_prob": result["over_3_5"]["Over"],
                "under_3_5_prob": result["over_3_5"]["Under"],
                "btts_prob": result["btts"]["BTTS"],
                "btts_no_prob": result["btts"]["No BTTS"],
            }
            if result["expected_goals"]:
                flat.update(result["expected_goals"])
            if p["H"] >= p["D"] and p["H"] >= p["A"]:
                flat["predicted_outcome"] = "Home Win"
            elif p["D"] >= p["A"]:
                flat["predicted_outcome"] = "Draw"
            else:
                flat["predicted_outcome"] = "Away Win"
            flat["confidence"] = max(p["H"], p["D"], p["A"])
            records.append(flat)
        return pd.DataFrame(records)

    # ═══════════════════════════════════════════════════════
    #  Pre-computation (cached for optimisation speed)
    # ═══════════════════════════════════════════════════════

    def precompute(self, df: pd.DataFrame, home_col: str = "home_team", away_col: str = "away_team",
                   cache_key: str = "default") -> PerModelPredictions:
        """Pre-compute per-model predictions for all matches in df.

        This is the key performance optimisation: instead of calling each
        model's predict() hundreds of times during weight grid search,
        we compute everything once and cache it.

        Importantly, Poisson's binary market predictions (BTTS, O/U) are
        computed here from the Poisson model's scoreline table — NOT
        approximated via conditional rates.
        """
        home_teams = df[home_col].tolist()
        away_teams = df[away_col].tolist()
        n = len(df)

        pois_1x2_list, elo_1x2_list = [], []
        pois_btts_list, pois_over25_list, pois_over35_list = [], [], []
        elo_btts_list, xgb_btts_list = [], []
        pois_eh_list, pois_ea_list = [], []

        for ht, at in zip(home_teams, away_teams):
            # Poisson 1X2
            try:
                r = self.poisson.predict(ht, at)
                pois_1x2_list.append([r["away_win_prob"], r["draw_prob"], r["home_win_prob"]])
                pois_btts_list.append(r.get("btts_prob", 0.5))
                pois_over25_list.append(r.get("over_2_5_prob", 0.5))
                pois_over35_list.append(r.get("over_3_5_prob", 0.5))
                pois_eh_list.append(r.get("expected_home_goals", 0.0))
                pois_ea_list.append(r.get("expected_away_goals", 0.0))
            except Exception:
                pois_1x2_list.append([0.33, 0.34, 0.33])
                pois_btts_list.append(0.5)
                pois_over25_list.append(0.5)
                pois_over35_list.append(0.5)
                pois_eh_list.append(0.0)
                pois_ea_list.append(0.0)

            # Elo 1X2 and direct BTTS
            try:
                df_single = pd.DataFrame([{"home_team": ht, "away_team": at}])
                elo_1x2_list.append(self.elo.predict_proba(df_single)[0])
                elo_btts_list.append(self.elo.predict_btts(ht, at))
            except Exception:
                elo_1x2_list.append([0.33, 0.34, 0.33])
                elo_btts_list.append(0.5)

        # XGBoost — batch feature engineering
        xgb_1x2_list = []
        try:
            X = self._feature_builder.build(home_teams, away_teams)
            if X is not None and len(X) > 0:
                xgb_raw = self.xgb.predict_proba(X)
                cr = self.cond_rates
                for i in range(len(X)):
                    xgb_probs = xgb_raw[i]  # [away, draw, home]
                    xgb_1x2_list.append(xgb_probs)
                    # Compute XGBoost BTTS via 1X2 → expected goals → Poisson formula (inline)
                    exp_total = (
                        xgb_probs[2] * cr.mean_total_given_home_win
                        + xgb_probs[1] * cr.mean_total_given_draw
                        + xgb_probs[0] * cr.mean_total_given_away_win
                    )
                    if exp_total > 0:
                        exp_home = exp_total * 0.55
                        exp_away = exp_total * 0.45
                        p_h0 = np.exp(-exp_home)
                        p_a0 = np.exp(-exp_away)
                        btts_val = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)
                        xgb_btts_list.append(float(np.clip(btts_val, 0.0, 1.0)))
                    else:
                        xgb_btts_list.append(0.5)
            else:
                xgb_1x2_list = [[0.33, 0.34, 0.33]] * n
                xgb_btts_list = [0.5] * n
        except Exception as exc:
            logger.warning("XGBoost batch prediction failed: %s", exc)
            xgb_1x2_list = [[0.33, 0.34, 0.33]] * n
            xgb_btts_list = [0.5] * n

        ppm = PerModelPredictions(
            pois_1x2=np.array(pois_1x2_list),
            elo_1x2=np.array(elo_1x2_list),
            xgb_1x2=np.array(xgb_1x2_list),
            pois_btts=np.array(pois_btts_list),
            elo_btts=np.array(elo_btts_list),
            xgb_btts=np.array(xgb_btts_list),
            pois_over_25=np.array(pois_over25_list),
            pois_over_35=np.array(pois_over35_list),
            pois_exp_home=np.array(pois_eh_list),
            pois_exp_away=np.array(pois_ea_list),
            n=n,
        )
        self._cache[cache_key] = ppm
        return ppm

    def _blend_1x2(self, ppm: PerModelPredictions, w: dict[str, float]) -> np.ndarray:
        wp, we, wx = w.get("poisson", 0), w.get("elo", 0), w.get("xgb", 0)
        total = wp + we + wx
        if total <= 0:
            return ppm.pois_1x2.copy()
        result = (wp / total) * ppm.pois_1x2 + (we / total) * ppm.elo_1x2 + (wx / total) * ppm.xgb_1x2
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return result / row_sums

    def _blend_binary(self, ppm: PerModelPredictions, w: dict[str, float], market: str) -> np.ndarray:
        """Blend binary market (BTTS or O/U).

        For **BTTS**: uses direct Poisson BTTS from each model:
        - Poisson: exact from scoreline table
        - Elo: direct from Elo-derived expected goals
        - XGBoost: from 1X2 → expected goals → Poisson formula

        For **Over/Under**: Poisson exact + XGBoost Poisson CDF + Elo conditional rates.
        """
        wp, we, wx = w.get("poisson", 0), w.get("elo", 0), w.get("xgb", 0)
        total = wp + we + wx
        if total <= 0:
            return np.full(ppm.n, 0.5)

        if market == "BTTS":
            # Direct BTTS from all three models
            pois_val = ppm.pois_btts
            elo_val = ppm.elo_btts
            xgb_val = ppm.xgb_btts
        elif "3.5" in market:
            pois_val = ppm.pois_over_35
            elo_val = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 3.5)
            xgb_val = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 3.5)
        else:
            pois_val = ppm.pois_over_25
            elo_val = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 2.5)
            xgb_val = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 2.5)

        return (wp * pois_val + we * elo_val + wx * xgb_val) / total

    # ═══════════════════════════════════════════════════════
    #  Weight Optimisation
    # ═══════════════════════════════════════════════════════

    def optimise_weights(
        self,
        df_val: pd.DataFrame,
        markets: list[str] | None = None,
        n_grid: int = 6,
        metric: str = "brier_score",
        home_col: str = "home_team", away_col: str = "away_team",
        home_goals_col: str = "home_goals", away_goals_col: str = "away_goals",
        verbose: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Optimise per-market weights via grid search on a validation set.

        Tests weight combinations per market, selecting the blend that
        minimises Brier score on **df_val** (a held-out validation set).

        Parameters
        ----------
        df_val : pd.DataFrame
            Validation data with actual results (NOT the test set).
        markets : list[str], optional
        n_grid : int
            Number of weight splits per model (default 6 → ~36-216 combos).
        metric : str
            ``"brier_score"``, ``"log_loss"``, or ``"accuracy"``.
        """
        if markets is None:
            markets = list(WEIGHT_SEARCH_RANGES.keys())

        if self._feature_builder._historical_data is None:
            raise RuntimeError("Historical data not set — call set_historical_data() first")

        self.cond_rates = ConditionalRates.from_data(
            self._feature_builder._historical_data
        )

        # Pre-compute predictions on validation set
        logger.info("Pre-computing predictions for %d validation matches...", len(df_val))
        ppm = self.precompute(df_val, home_col, away_col, cache_key="optimise")

        # Prepare actual outcomes
        actual_result = df_val["result"].map({"A": 0, "D": 1, "H": 2}).values
        hg, ag = df_val[home_goals_col].values.astype(float), df_val[away_goals_col].values.astype(float)
        actual_btts = ((hg > 0) & (ag > 0)).astype(float)
        actual_ou25 = ((hg + ag) > 2.5).astype(float)
        actual_ou35 = ((hg + ag) > 3.5).astype(float)

        optimal_weights: dict[str, dict[str, float]] = {}

        for market in markets:
            if market not in WEIGHT_SEARCH_RANGES:
                logger.warning("No search range for '%s', skipping", market)
                continue
            ranges = WEIGHT_SEARCH_RANGES[market]
            models_in_market = [m for m in ["poisson", "elo", "xgb"] if m in ranges]
            combos = _build_weight_grid(ranges, models_in_market, n_grid)

            best_score, best_w = float("inf"), dict(DEFAULT_WEIGHTS.get(market, {}))
            lower_better = metric in ("brier_score", "log_loss")

            if market == "1X2":
                y_true = actual_result
            elif market in ("Over2.5", "Under2.5"):
                y_true = actual_ou25
            elif market in ("Over3.5", "Under3.5"):
                y_true = actual_ou35
            elif market == "BTTS":
                y_true = actual_btts
            else:
                continue

            for combo in combos:
                if market == "1X2":
                    blended = self._blend_1x2(ppm, combo)
                else:
                    blended = self._blend_binary(ppm, combo, market)

                score = _score_predictions(blended, y_true, metric, market)
                if score is None:
                    continue
                if (lower_better and score < best_score) or (not lower_better and score > best_score):
                    best_score, best_w = score, dict(combo)

            optimal_weights[market] = best_w
            self.weights[market] = best_w

            if verbose:
                w_str = ", ".join(f"{k}={v:.2f}" for k, v in best_w.items())
                logger.info("  %s: best %s=%.4f  weights=(%s)", market, metric, best_score, w_str)

        return optimal_weights

    # ═══════════════════════════════════════════════════════
    #  Evaluation
    # ═══════════════════════════════════════════════════════

    def evaluate(
        self,
        df_test: pd.DataFrame,
        home_col: str = "home_team", away_col: str = "away_team",
        home_goals_col: str = "home_goals", away_goals_col: str = "away_goals",
        include_individual: bool = True,
        ensemble_model: Any = None,
        cache_key: str = "eval",
    ) -> dict[str, Any]:
        """Evaluate the blend on test data across all markets.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test data with actual results.
        include_individual : bool
            Also compute metrics for individual models.
        ensemble_model : Any, optional
            Current ensemble model (e.g. EnsembleModel) for comparison.
            Must have ``predict_proba(X)``.
        cache_key : str
            Cache key for pre-computed predictions.

        Returns
        -------
        dict
            Nested dict of market → model_name → {metrics}, plus expected_goals errors.
        """
        if not self.fitted:
            raise RuntimeError("ThreeModelBlend not fitted")

        ppm = self.precompute(df_test, home_col, away_col, cache_key=cache_key)

        actual_result = df_test["result"].map({"A": 0, "D": 1, "H": 2}).values
        hg, ag = df_test[home_goals_col].values.astype(float), df_test[away_goals_col].values.astype(float)
        actual_btts = ((hg > 0) & (ag > 0)).astype(float)
        actual_ou25 = ((hg + ag) > 2.5).astype(float)
        actual_ou35 = ((hg + ag) > 3.5).astype(float)
        actual_total = hg + ag

        # Expected goals errors
        mse = float(np.mean((ppm.pois_total_goals - actual_total) ** 2))
        mae = float(np.mean(np.abs(ppm.pois_total_goals - actual_total)))

        results: dict[str, Any] = {
            "n_test": ppm.n,
            "expected_goals": {"mse": round(mse, 4), "mae": round(mae, 4)},
            "markets": {},
        }

        # ── Individual model predictions ──
        # BTTS: direct Poisson computation from each model
        elo_btts = ppm.elo_btts
        xgb_btts = ppm.xgb_btts
        # Over/Under: Elo and XGBoost derived from 1X2 via conditional rates
        elo_ou25 = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 2.5)
        elo_ou35 = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 3.5)
        xgb_ou25 = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 2.5)
        xgb_ou35 = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 3.5)

        # Ensemble comparison (if provided)
        ens_1x2 = None
        if ensemble_model is not None:
            try:
                X = self._feature_builder.build(
                    df_test[home_col].tolist(), df_test[away_col].tolist()
                )
                if X is not None and len(X) > 0:
                    ens_1x2 = ensemble_model.predict_proba(X)
            except Exception as exc:
                logger.warning("Ensemble prediction failed: %s", exc)

        for market_name in ["1X2", "Over2.5", "BTTS", "Over3.5"]:
            w = self.weights.get(market_name, self.weights.get("Over2.5", DEFAULT_WEIGHTS.get("Over2.5", {})))
            md: dict[str, Any] = {"blend_weights": dict(w), "models": {}}

            if market_name == "1X2":
                if include_individual:
                    md["models"]["Poisson"] = _metrics_1x2(actual_result, ppm.pois_1x2)
                    md["models"]["Elo"] = _metrics_1x2(actual_result, ppm.elo_1x2)
                    md["models"]["XGBoost"] = _metrics_1x2(actual_result, ppm.xgb_1x2)
                    if ens_1x2 is not None:
                        md["models"]["Current Ensemble"] = _metrics_1x2(actual_result, ens_1x2)
                blend = self._blend_1x2(ppm, w)
                md["models"]["3-Model Blend"] = _metrics_1x2(actual_result, blend)

            elif market_name == "Over2.5":
                if include_individual:
                    md["models"]["Poisson"] = _metrics_binary(actual_ou25, ppm.pois_over_25)
                    md["models"]["Elo"] = _metrics_binary(actual_ou25, elo_ou25)
                    md["models"]["XGBoost"] = _metrics_binary(actual_ou25, xgb_ou25)
                blend = self._blend_binary(ppm, w, "Over2.5")
                md["models"]["3-Model Blend"] = _metrics_binary(actual_ou25, blend)

            elif market_name == "Over3.5":
                if include_individual:
                    md["models"]["Poisson"] = _metrics_binary(actual_ou35, ppm.pois_over_35)
                    md["models"]["Elo"] = _metrics_binary(actual_ou35, elo_ou35)
                    md["models"]["XGBoost"] = _metrics_binary(actual_ou35, xgb_ou35)
                blend = self._blend_binary(ppm, w, "Over3.5")
                md["models"]["3-Model Blend"] = _metrics_binary(actual_ou35, blend)

            elif market_name == "BTTS":
                if include_individual:
                    md["models"]["Poisson"] = _metrics_binary(actual_btts, ppm.pois_btts)
                    md["models"]["Elo"] = _metrics_binary(actual_btts, elo_btts)
                    md["models"]["XGBoost"] = _metrics_binary(actual_btts, xgb_btts)
                blend = self._blend_binary(ppm, w, "BTTS")
                md["models"]["3-Model Blend"] = _metrics_binary(actual_btts, blend)

            results["markets"][market_name] = md

        return results

    # ═══════════════════════════════════════════════════════
    #  Reporting
    # ═══════════════════════════════════════════════════════

    def generate_report(self, evaluation: dict[str, Any], output_dir: str | Path = "reports",
                        timestamp: str | None = None) -> dict[str, str]:
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save optimal weights
        weights_data = {"weights": self.weights, "timestamp": ts}
        weights_path = output_path / f"three_model_blend_weights_{ts}.json"
        with open(weights_path, "w") as f:
            json.dump(weights_data, f, indent=2)

        lines: list[str] = []
        lines.append("# Three-Model Blend Comparison Report")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Test samples:** {evaluation.get('n_test', 'N/A')}")
        lines.append("")

        # Expected goals error
        eg = evaluation.get("expected_goals", {})
        lines.append("## Expected Goals Prediction (Poisson)")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| MSE | {eg.get('mse', 'N/A'):.4f} |")
        lines.append(f"| MAE | {eg.get('mae', 'N/A'):.4f} |")
        lines.append("")

        for market_name in ["1X2", "Over2.5", "Over3.5", "BTTS"]:
            md = evaluation.get("markets", {}).get(market_name)
            if not md:
                continue
            w = md.get("blend_weights", {})
            w_str = ", ".join(f"{k}={v:.2f}" for k, v in w.items())
            lines.append(f"## {market_name} Market")
            lines.append("")
            lines.append(f"**Optimal Weights:** {w_str}")
            lines.append("")
            lines.append("| Model | Brier Score | Log Loss | Accuracy | Samples |")
            lines.append("|-------|-------------|----------|----------|---------|")
            for mn in ["Poisson", "Elo", "XGBoost", "Current Ensemble", "3-Model Blend"]:
                m = md.get("models", {}).get(mn)
                if m:
                    lines.append(f"| {mn} | {m.get('brier_score', 'N/A'):.4f} | {m.get('log_loss', 'N/A'):.4f} | {m.get('accuracy', 'N/A'):.2%} | {m.get('n', 'N/A')} |")
            lines.append("")

        # Weight recommendations
        lines.append("## Optimal Weight Recommendations")
        lines.append("")
        lines.append("| Market | Poisson Weight | Elo Weight | XGBoost Weight |")
        lines.append("|--------|---------------|------------|----------------|")
        for mkt in ["1X2", "Over2.5", "BTTS", "Over3.5"]:
            w = self.weights.get(mkt, {})
            lines.append(f"| {mkt} | {w.get('poisson', 0):.2f} | {w.get('elo', 0):.2f} | {w.get('xgb', 0):.2f} |")
        lines.append("")

        # Recommendation text
        lines.append("## Recommendation")
        lines.append("")
        improvement_found = False
        for market_name in ["1X2", "Over2.5", "BTTS"]:
            md = evaluation.get("markets", {}).get(market_name, {})
            models_d = md.get("models", {})
            blend_m = models_d.get("3-Model Blend", {})
            best_single, best_brier = None, float("inf")
            for mn in ["Poisson", "Elo", "XGBoost", "Current Ensemble"]:
                m = models_d.get(mn, {})
                br = m.get("brier_score", float("inf"))
                if br < best_brier:
                    best_brier, best_single = br, mn
            if blend_m and best_single:
                bb = blend_m.get("brier_score", float("inf"))
                imp = ((best_brier - bb) / best_brier * 100) if best_brier > 0 else 0
                if bb < best_brier:
                    lines.append(f"- **{market_name}**: Blend improves over {best_single} by **{imp:.1f}%** (Brier).")
                    improvement_found = True
                else:
                    lines.append(f"- **{market_name}**: Best individual model ({best_single}) outperforms blend.")
        if not improvement_found:
            lines.append("- Blend shows competitive performance across all markets.")

        lines.append("")
        lines.append("## Hypothesis Validation")
        lines.append("")
        lines.append("| Hypothesis | Expected | Actual |")
        lines.append("|------------|----------|--------|")
        w1, w2, w3 = self.weights.get("1X2", {}), self.weights.get("Over2.5", {}), self.weights.get("BTTS", {})
        lines.append(f"| 1X2 Weight | Poisson(0.5-0.6) + Elo(0.3-0.4) + XGB(0.1-0.2) | P({w1.get('poisson',0):.2f}) + E({w1.get('elo',0):.2f}) + X({w1.get('xgb',0):.2f}) |")
        lines.append(f"| Over/Under Weight | Poisson(0.3-0.4) + XGB(0.5-0.6) + Elo(0-0.1) | P({w2.get('poisson',0):.2f}) + X({w2.get('xgb',0):.2f}) + E({w2.get('elo',0):.2f}) |")
        lines.append(f"| BTTS Weight | Poisson(0.3-0.4) + Elo(0.3-0.4) + XGB(0.3-0.4) | P({w3.get('poisson',0):.2f}) + E({w3.get('elo',0):.2f}) + X({w3.get('xgb',0):.2f}) |")

        report_md = "\n".join(lines)
        report_path = output_path / f"three_model_comparison_{ts}.md"
        with open(report_path, "w") as f:
            f.write(report_md)

        return {"weights_json": str(weights_path), "report_md": str(report_path)}


# ═══════════════════════════════════════════════════════════
#  Static helpers
# ═══════════════════════════════════════════════════════════


def _poisson_cdf(k: float, lam: float) -> float:
    """Poisson CDF: P(X <= k) where X ~ Pois(lam).

    Uses the standard Poisson PMF summed from 0 to floor(k).
    This is used by ``_xgb_over()`` to convert expected total goals
    into an Over/Under probability.
    """
    if lam <= 0:
        return 1.0 if k < 0 else 0.0
    from math import exp, factorial
    k_int = int(k)
    if k_int < 0:
        return 0.0
    cdf = 0.0
    for i in range(k_int + 1):
        cdf += exp(-lam) * (lam ** i) / factorial(i)
    return min(cdf, 1.0)


def _build_weight_grid(ranges: dict[str, tuple[float, float]], models: list[str], n_grid: int) -> list[dict[str, float]]:
    """Build weight combinations by random sampling within ranges.

    Generates ``n_grid^2`` valid combinations (or as many as feasible)
    where weights sum to 1.0 and each is within its specified range.
    """
    import random as _rnd
    _rnd.seed(42)
    combos: list[dict[str, float]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n_grid ** 3 * 5

    while len(combos) < max(20, n_grid ** 2) and attempts < max_attempts:
        attempts += 1
        candidate: dict[str, float] = {}
        remaining = 1.0
        for i, m in enumerate(models):
            lo, hi = ranges.get(m, (0.0, 1.0))
            if i == len(models) - 1:
                w = remaining
            else:
                # Leave enough for remaining models (at least their min)
                min_left = sum(ranges.get(m2, (0, 0))[0] for m2 in models[i + 1:])
                max_this = remaining - min_left
                lo = max(lo, 0.0)
                hi = min(hi, max_this)
                if lo > hi:
                    break
                w = _rnd.uniform(lo, min(hi, remaining))
            w = round(w, 4)
            candidate[m] = w
            remaining -= w

        if remaining < -0.01 or remaining > 0.01:
            continue
        key = tuple(sorted(candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    # Ensure defaults are included
    for mkt_default in ["1X2", "Over2.5", "BTTS"]:
        default = DEFAULT_WEIGHTS.get(mkt_default, {})
        default_subset = {m: default.get(m, 1.0 / len(models)) for m in models if m in default}
        if default_subset and all(m in default_subset for m in models):
            key = tuple(sorted(default_subset.items()))
            if key not in seen:
                combos.append(default_subset)

    _rnd.seed()
    return combos


def _metrics_1x2(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss as sk_ll
    valid = ~np.isnan(y_true)
    y_v, p_v = y_true[valid], probs[valid]
    y_oh = np.zeros_like(p_v)
    for i, v in enumerate(y_v):
        if 0 <= v <= 2:
            y_oh[i, int(v)] = 1
    preds = np.argmax(p_v, axis=1)
    return {
        "brier_score": round(float(np.mean(np.sum((p_v - y_oh) ** 2, axis=1))), 4),
        "log_loss": round(float(sk_ll(y_v, p_v)), 4),
        "accuracy": round(float(np.mean(preds == y_v)), 4),
        "n": int(valid.sum()),
    }


def _metrics_binary(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss as sk_ll
    valid = ~np.isnan(y_true)
    p_v = np.clip(probs[valid], 1e-15, 1 - 1e-15)
    y_v = y_true[valid]
    preds = (p_v > 0.5).astype(float)
    return {
        "brier_score": round(float(np.mean((p_v - y_v) ** 2)), 4),
        "log_loss": round(float(sk_ll(y_v, np.column_stack([1 - p_v, p_v]))), 4),
        "accuracy": round(float(np.mean(preds == y_v)), 4),
        "n": int(len(y_v)),
    }


def _score_predictions(blended: np.ndarray, y_true: np.ndarray, metric: str, market: str) -> float | None:
    valid = ~np.isnan(y_true)
    y_v = y_true[valid]

    if market == "1X2":
        if blended.ndim != 2 or blended.shape[1] != 3:
            return None
        b_v = blended[valid]
        if metric == "brier_score":
            y_oh = np.zeros_like(b_v)
            for i, v in enumerate(y_v):
                if 0 <= v <= 2:
                    y_oh[i, int(v)] = 1
            return float(np.mean(np.sum((b_v - y_oh) ** 2, axis=1)))
        elif metric == "log_loss":
            from sklearn.metrics import log_loss as sk_ll
            return float(sk_ll(y_v, b_v))
        elif metric == "accuracy":
            return float(np.mean(np.argmax(b_v, axis=1) == y_v))
    else:
        b_v = blended[valid].flatten() if blended.ndim == 2 else blended[valid]
        if metric == "brier_score":
            return float(np.mean((b_v - y_v) ** 2))
        elif metric == "log_loss":
            from sklearn.metrics import log_loss as sk_ll
            eps = 1e-15
            return float(sk_ll(y_v, np.column_stack([1 - np.clip(b_v, eps, 1 - eps), np.clip(b_v, eps, 1 - eps)])))
        elif metric == "accuracy":
            return float(np.mean((b_v > 0.5).astype(float) == y_v))
    return None
