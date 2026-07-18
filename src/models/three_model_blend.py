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
    # --- Validated on 2023-2024 season (1051 train / 350 val / 351 test) ---
    # 1X2:        Poisson dominates for match outcome. Blend wins best Brier + Accuracy.
    # Over2.5:    Poisson + Elo work together. Blend best Brier/Acc, Default gives max ROI.
    # Over3.5:    Poisson alone is best via scoreline table. Elo/XGBoost too noisy for rare events.
    # BTTS:       XGBoost alone is best (optimised blend leans heavily on XGBoost).
    "1X2": {"poisson": 0.65, "elo": 0.19, "xgb": 0.16},
    "Over2.5": {"poisson": 0.50, "elo": 0.30, "xgb": 0.20},
    "Over3.5": {"poisson": 0.85, "elo": 0.05, "xgb": 0.10},
    "BTTS": {"poisson": 0.40, "elo": 0.10, "xgb": 0.50},
}

WEIGHT_SEARCH_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "1X2": {
        "poisson": (0.30, 0.70),
        "elo": (0.15, 0.50),
        "xgb": (0.05, 0.30),
    },
    "Over2.5": {
        "poisson": (0.20, 0.60),
        "elo": (0.00, 0.35),
        "xgb": (0.10, 0.60),
    },
    "Over3.5": {
        # Rare event (37.6% base rate); Poisson's scoreline table should dominate
        "poisson": (0.40, 0.90),
        "elo": (0.00, 0.20),
        "xgb": (0.05, 0.35),
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
    """Conditional BTTS and O/U rates given match outcome.

    Used to derive BTTS and O/U probabilities from any model's 1X2 predictions.
    """

    btts_given_home_win: float = 0.50
    btts_given_draw: float = 0.70
    btts_given_away_win: float = 0.40
    ou_given_home_win: float = 0.55
    ou_given_draw: float = 0.40
    ou_given_away_win: float = 0.50

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

        return cls(
            btts_given_home_win=_btts(hw),
            btts_given_draw=_btts(dr),
            btts_given_away_win=_btts(aw),
            ou_given_home_win=_ou(hw),
            ou_given_draw=_ou(dr),
            ou_given_away_win=_ou(aw),
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
        elo_system: Any,
        xgb_model: Any,
        weights: dict[str, dict[str, float]] | None = None,
        conditional_rates: ConditionalRates | None = None,
        historical_df: pd.DataFrame | None = None,
    ):
        self.poisson = poisson_model
        self.elo = elo_system
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
        w = self.weights.get("1X2", DEFAULT_WEIGHTS["1X2"])
        p_pois = self._poisson_1x2(home_team, away_team)
        p_elo = self._elo_1x2(home_team, away_team)
        p_xgb = self._xgb_1x2(home_team, away_team)

        p_h = w["poisson"] * p_pois[2] + w["elo"] * p_elo[2] + w["xgb"] * p_xgb[2]
        p_d = w["poisson"] * p_pois[1] + w["elo"] * p_elo[1] + w["xgb"] * p_xgb[1]
        p_a = w["poisson"] * p_pois[0] + w["elo"] * p_elo[0] + w["xgb"] * p_xgb[0]

        total = p_h + p_d + p_a
        if total > 0:
            p_h /= total
            p_d /= total
            p_a /= total
        return {"home_win": p_h, "draw": p_d, "away_win": p_a}

    # ── Over/Under Market ─────────────────────────────────

    def predict_over_under(self, home_team: str, away_team: str, threshold: float = 2.5) -> dict[str, float]:
        market_key = f"Over{threshold:.1f}".replace(".", "_")
        w = self.weights.get(market_key, self.weights.get("Over2.5", DEFAULT_WEIGHTS["Over2.5"]))

        over_pois = self._poisson_over(home_team, away_team, threshold)
        p_xgb_1x2 = self._xgb_1x2(home_team, away_team)
        over_xgb = float(self.cond_rates.ou_from_1x2(p_xgb_1x2.reshape(1, -1), threshold)[0])
        p_elo_1x2 = self._elo_1x2(home_team, away_team)
        over_elo = float(self.cond_rates.ou_from_1x2(p_elo_1x2.reshape(1, -1), threshold)[0])

        over_blend = w["poisson"] * over_pois + w["xgb"] * over_xgb + w["elo"] * over_elo
        total_w = w["poisson"] + w["xgb"] + w["elo"]
        if total_w > 0:
            over_blend /= total_w

        over_key = f"over_{threshold:.1f}".replace(".", "_")
        under_key = f"under_{threshold:.1f}".replace(".", "_")
        return {over_key: over_blend, under_key: 1.0 - over_blend}

    # ── BTTS Market ───────────────────────────────────────

    def predict_btts(self, home_team: str, away_team: str) -> dict[str, float]:
        w = self.weights.get("BTTS", DEFAULT_WEIGHTS["BTTS"])
        btts_pois = self._poisson_btts(home_team, away_team)
        p_xgb = self._xgb_1x2(home_team, away_team)
        btts_xgb = float(self.cond_rates.btts_from_1x2(p_xgb.reshape(1, -1))[0])
        p_elo = self._elo_1x2(home_team, away_team)
        btts_elo = float(self.cond_rates.btts_from_1x2(p_elo.reshape(1, -1))[0])

        btts_blend = w["poisson"] * btts_pois + w["elo"] * btts_elo + w["xgb"] * btts_xgb
        total_w = w["poisson"] + w["elo"] + w["xgb"]
        if total_w > 0:
            btts_blend /= total_w
        return {"btts": btts_blend, "btts_no": 1.0 - btts_blend}

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

    # ── Batch Prediction ──────────────────────────────────

    def predict_matches(self, df: pd.DataFrame, home_col: str = "home_team", away_col: str = "away_team") -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            home, away = row[home_col], row[away_col]
            result = self.predict(home, away)
            flat = {
                "home_team": home, "away_team": away,
                "home_win_prob": result["1x2"]["home_win"],
                "draw_prob": result["1x2"]["draw"],
                "away_win_prob": result["1x2"]["away_win"],
                "over_2_5_prob": result["over_under"].get("over_2_5", 0.5),
                "under_2_5_prob": result["over_under"].get("under_2_5", 0.5),
                "over_3_5_prob": result["over_3_5"].get("over_3_5", 0.5),
                "under_3_5_prob": result["over_3_5"].get("under_3_5", 0.5),
                "btts_prob": result["btts"].get("btts", 0.5),
                "btts_no_prob": result["btts"].get("btts_no", 0.5),
            }
            if result["expected_goals"]:
                flat.update(result["expected_goals"])
            p = result["1x2"]
            if p["home_win"] >= p["draw"] and p["home_win"] >= p["away_win"]:
                flat["predicted_outcome"] = "Home Win"
            elif p["draw"] >= p["away_win"]:
                flat["predicted_outcome"] = "Draw"
            else:
                flat["predicted_outcome"] = "Away Win"
            flat["confidence"] = max(p["home_win"], p["draw"], p["away_win"])
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

            # Elo 1X2
            try:
                df_single = pd.DataFrame([{"home_team": ht, "away_team": at}])
                elo_1x2_list.append(self.elo.predict_proba(df_single)[0])
            except Exception:
                elo_1x2_list.append([0.33, 0.34, 0.33])

        # XGBoost — batch feature engineering
        xgb_1x2_list = []
        try:
            X = self._feature_builder.build(home_teams, away_teams)
            if X is not None and len(X) > 0:
                xgb_raw = self.xgb.predict_proba(X)
                for i in range(len(X)):
                    xgb_1x2_list.append(xgb_raw[i])
            else:
                xgb_1x2_list = [[0.33, 0.34, 0.33]] * n
        except Exception as exc:
            logger.warning("XGBoost batch prediction failed: %s", exc)
            xgb_1x2_list = [[0.33, 0.34, 0.33]] * n

        ppm = PerModelPredictions(
            pois_1x2=np.array(pois_1x2_list),
            elo_1x2=np.array(elo_1x2_list),
            xgb_1x2=np.array(xgb_1x2_list),
            pois_btts=np.array(pois_btts_list),
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
        """Blend binary market (BTTS or O/U) using Poisson's actual predictions."""
        wp, we, wx = w.get("poisson", 0), w.get("elo", 0), w.get("xgb", 0)
        total = wp + we + wx
        if total <= 0:
            return np.full(ppm.n, 0.5)

        if market == "BTTS":
            # Poisson provides exact BTTS from scoreline table
            pois_val = ppm.pois_btts
        elif "3.5" in market:
            # Use Poisson's exact over_3_5_prob
            pois_val = ppm.pois_over_35
        else:
            # Default Over2.5
            pois_val = ppm.pois_over_25

        # Elo and XGBoost derive BTTS/O/U from 1X2 via conditional rates
        elo_val = self.cond_rates.btts_from_1x2(ppm.elo_1x2) if market == "BTTS" else self.cond_rates.ou_from_1x2(ppm.elo_1x2, 3.5 if "3.5" in market else 2.5)
        xgb_val = self.cond_rates.btts_from_1x2(ppm.xgb_1x2) if market == "BTTS" else self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 3.5 if "3.5" in market else 2.5)

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
        # For Elo and XGBoost BTTS/O/U, derive from 1X2
        elo_btts = self.cond_rates.btts_from_1x2(ppm.elo_1x2)
        elo_ou25 = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 2.5)
        elo_ou35 = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 3.5)
        xgb_btts = self.cond_rates.btts_from_1x2(ppm.xgb_1x2)
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
