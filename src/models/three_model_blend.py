"""
Multi-Model Blend — Market-specific ensemble of Dixon-Coles + Elo + XGBoost + LightGBM + CatBoost.

Combines predictions from five fundamentally different model types, each
contributing where it excels, with weights optimised per market:

- **Dixon-Coles** (MLE with tau-correction, recency + importance weighting)
  → exact scoreline distribution, BTTS & O/U — replaces independent Poisson
- **Elo** (dynamic team strength ratings) → stable long-term prior
- **XGBoost** (gradient-boosted trees, 146 features) → complex interactions
- **LightGBM** (gradient-boosted trees, leaf-wise) → different tree structure,
  better categorical handling
- **CatBoost** (gradient-boosted trees, ordered boosting) → native categorical,
  robust to noisy features

Key design decisions
--------------------
1. **Direct Poisson BTTS formula** — P(BTTS) = 1 - e^{-λ_home} - e^{-λ_away}
   + e^{-(λ_home + λ_away)} for DC/Elo, and 1X2 → expected goals → BTTS for
   tree models.
2. **Poisson CDF for tree model O/U** — 1X2 probs → expected total goals →
   Poisson CDF P(X ≤ t) to get P(Over).
3. **ConditionalRates** — fallback for converting any model's 1X2 predictions
   into BTTS/O/U probabilities when direct methods are unavailable.
4. **Feature pipeline** — `_FeatureBuilder` appends fixture rows to historical
   data and runs the full ``build_features()`` pipeline.
5. **Pre-compute cache** — `precompute()` runs all models once for a dataset
   and caches results for fast weight grid search.
6. **Models are optional** — pass only the models you have; blend skips
   missing models gracefully.

Usage
-----
::

    from src.models.three_model_blend import ThreeModelBlend
    from src.dixon_coles import DixonColesModel
    from src.elo import EloSystem
    import joblib

    dc = DixonColesModel().fit(df_train)
    elo = EloSystem()
    elo.process_matches(df_train)
    xgb = joblib.load("models/xgboost_model.joblib")
    lgb = joblib.load("models/lightgbm_model.joblib")
    cat = joblib.load("models/catboost_model.joblib")

    blend = ThreeModelBlend(dc_model=dc, elo_model=elo, xgb_model=xgb,
                            lgb_model=lgb, cat_model=cat)

    # Predict a single fixture (all markets)
    result = blend.predict("France", "England")
    # → {'1x2': {'H': ..., 'D': ..., 'A': ...},
    #    'over_under': {'Over': ..., 'Under': ...},
    #    'over_3_5': {'Over': ..., 'Under': ...},
    #    'btts': {'BTTS': ..., 'No BTTS': ...},
    #    'expected_goals': {...}}

    # Batch predict
    df_preds = blend.predict_matches(df_fixtures)

    # Optimise weights for each market
    blend.optimise_weights(df_val)

    # Evaluate per-market
    metrics = blend.evaluate(df_test)

    # Persist / restore
    blend.save("models/three_model_blend.joblib")
    blend = ThreeModelBlend.load("models/three_model_blend.joblib", historical_df=df)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Default Weights (Hypothesis-driven)
# ═══════════════════════════════════════════════════════════

DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    # ═══════════════════════════════════════════════════════════════
    #  Per-Market Model Selection (research-validated 2026-07-25)
    # ═══════════════════════════════════════════════════════════════
    #
    # 1X2: 5-model blend (DC + Elo + XGB + LGB + Cat)
    #   → Tree models dominate on league data (XGB Brier=0.460, 72.6% acc)
    #
    # Over2.5 / Over3.5 / BTTS: DC-only
    #   → DC-only comparison vs DC+Market Trees on F1 (387 test matches):
    #     OU Brier:   DC 0.2488 vs Trees 0.2545 (trees worse)
    #     OU Acc:     DC 54.52% vs Trees 48.84% (trees -5.68pp)
    #     BTTS Brier: DC 0.2487 vs Trees 0.2523 (trees worse)
    #     BTTS Acc:   DC 54.78% vs Trees 47.29% (trees -7.49pp)
    #     Yield:      DC +1.18% vs Trees -11.06% (trees -12.24pp)
    #   → Conclusion: DC-only strictly dominates trees for binary markets.
    #     Trees are reserved for 1X2 only.
    #
    # Weights are initial defaults; grid-search optimisation via optimise_weights()
    # will find the optimal blend for each dataset.
    "1X2": {"dc": 0.35, "elo": 0.25, "xgb": 0.15, "lgb": 0.15, "cat": 0.10},
    "Over2.5": {"dc": 1.00},
    "Over3.5": {"dc": 1.00},
    "BTTS": {"dc": 1.00},
}

WEIGHT_SEARCH_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    # All markets: per-market model selection based on research
    # 1X2: full 5-model search
    # Binary markets: DC-only (no tree search — research shows trees degrade performance)
    "1X2": {
        "dc": (0.15, 0.50),
        "elo": (0.10, 0.40),
        "xgb": (0.05, 0.25),
        "lgb": (0.05, 0.25),
        "cat": (0.05, 0.20),
    },
    "Over2.5": {
        "dc": (1.00, 1.00),
    },
    "Over3.5": {
        "dc": (1.00, 1.00),
    },
    "BTTS": {
        "dc": (1.00, 1.00),
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
    """Build feature vectors for XGBoost from team names.

    Uses the same ``build_features()`` pipeline as the main training
    pipeline to ensure consistent feature computation.  Fixture rows are
    appended with ``None`` goals/result so rolling statistics are computed
    from real historical data only (never leaked from the fixture itself).
    """

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
                row: dict[str, object] = {
                    "date": pd.Timestamp(today_str),
                    "home_team": ht,
                    "away_team": at,
                    # Null result/goals prevent pollution of rolling features.
                    # The fixture is appended AFTER all historical matches, so
                    # rolling stats for the fixture look backward at real data.
                    # Use np.nan instead of None to avoid ufunc 'isnan' errors
                    # when numpy tries to process these values in feature pipeline.
                    "result": np.nan,
                    "home_goals": np.nan,
                    "away_goals": np.nan,
                }
                # Carry forward essential context columns (static identifiers).
                # These affect encoding (e.g. league-specific target encoding)
                # but NOT rolling features (which are driven by results/goals).
                for col in ("season", "league", "country"):
                    if col in self._historical_data.columns:
                        val = self._historical_data[col].iloc[-1]
                        row[col] = val if pd.notna(val) else None
                fixture_rows.append(row)

            df_ext = pd.concat(
                [self._historical_data, pd.DataFrame(fixture_rows)],
                ignore_index=True,
            )
            # is_training=False ensures build_features does not try to extract
            # a target from fixture rows (which have result=None).
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

    dc_1x2: np.ndarray  # (n, 3) [away, draw, home]
    elo_1x2: np.ndarray  # (n, 3)
    xgb_1x2: np.ndarray  # (n, 3)
    lgb_1x2: np.ndarray  # (n, 3)
    cat_1x2: np.ndarray  # (n, 3)

    # Dixon-Coles binary market predictions
    dc_btts: np.ndarray  # (n,)
    elo_btts: np.ndarray  # (n,)  — direct BTTS from Elo-derived expected goals
    xgb_btts: np.ndarray  # (n,)  — BTTS from XGBoost 1X2 → expected goals
    lgb_btts: np.ndarray  # (n,)  — BTTS from LightGBM 1X2 → expected goals
    cat_btts: np.ndarray  # (n,)  — BTTS from CatBoost 1X2 → expected goals
    dc_over_25: np.ndarray  # (n,)
    dc_over_35: np.ndarray  # (n,)
    dc_exp_home: np.ndarray  # (n,)
    dc_exp_away: np.ndarray  # (n,)

    n: int

    @property
    def dc_total_goals(self) -> np.ndarray:
        return cast(np.ndarray, self.dc_exp_home + self.dc_exp_away)


# ═══════════════════════════════════════════════════════════
#  ThreeModelBlend
# ═══════════════════════════════════════════════════════════


class ThreeModelBlend:
    """Market-specific blend of Dixon-Coles + Elo + XGBoost + LightGBM + CatBoost.

    Parameters
    ----------
    dc_model : DixonColesModel, optional
        Fitted Dixon-Coles model (replaces PoissonModel). If not provided,
        falls back to ``poisson_model`` for backward compatibility.
    elo_model : EloSystem
        Processed Elo system.
    xgb_model : Any, optional
        Fitted XGBoost/sklearn classifier with ``predict_proba(X)``.
    lgb_model : Any, optional
        Fitted LightGBM classifier with ``predict_proba(X)``.
    cat_model : Any, optional
        Fitted CatBoost classifier with ``predict_proba(X)``.
    poisson_model : Any, optional
        Deprecated — use ``dc_model`` instead. Falls back if ``dc_model``
        is not provided.
    weights : dict, optional
        Per-market weights. Falls back to ``DEFAULT_WEIGHTS``.
    conditional_rates : ConditionalRates, optional
        Pre-computed conditional rates.
    historical_df : pd.DataFrame, optional
        Historical data for ML model feature building.
    """

    def __init__(
        self,
        dc_model: Any = None,
        elo_model: Any = None,
        xgb_model: Any = None,
        lgb_model: Any = None,
        cat_model: Any = None,
        poisson_model: Any = None,
        weights: dict[str, dict[str, float]] | None = None,
        conditional_rates: ConditionalRates | None = None,
        historical_df: pd.DataFrame | None = None,
        away_fix_enabled: bool = False,
        away_fix_elo_threshold: float = -100.0,
        draw_fix_enabled: bool = False,
        draw_fix_max_elo_diff: float = 100.0,
        draw_fix_min_prob: float = 0.20,
    ):
        # Dixon-Coles (or Poisson fallback for backward compat)
        self.dc = dc_model or poisson_model
        self.elo = elo_model
        self.xgb = xgb_model
        self.lgb = lgb_model
        self.cat = cat_model
        self.weights = weights or {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
        self.cond_rates = conditional_rates or ConditionalRates()
        self._feature_builder = _FeatureBuilder(historical_df)
        self._cache: dict[str, PerModelPredictions] = {}
        self._calibrator: Any = None

        # Market-specific tree models for O/U and BTTS (trained directly on those targets)
        # Keyed by model type: {"xgb": model, "lgb": model, "cat": model}
        # When loaded, these replace the derived-from-1X2 approach in predict_over_under/predict_btts
        self.ou_models: dict[str, Any] = {}
        self.btts_models: dict[str, Any] = {}
        self.form_adjuster: Any = None  # RecentFormAdjuster instance

        # Per-league Dixon-Coles models for league-specific O/U and BTTS predictions.
        # Keyed by league code (e.g. "E0", "SE1"), each is a fitted DixonColesModel.
        # When a per-league model covers both teams in a fixture, it is preferred
        # over the global ``self.dc`` model for O/U and BTTS (1X2 still uses global).
        # Fitted in ``run_pipeline.py`` via ``fit_per_league_dc_models()``.
        self.per_league_dc: dict[str, Any] = {}
        # Away-fix: override blend to predict Away when Elo strongly favours away team
        self.away_fix_enabled = away_fix_enabled
        self.away_fix_elo_threshold = away_fix_elo_threshold  # e.g. -100 means diff < -100 triggers fix
        self.away_fix_applied: int = 0  # debug counter
        # Draw-fix: ensure a minimum draw probability when Elo gap is small
        self.draw_fix_enabled = draw_fix_enabled
        self.draw_fix_max_elo_diff = draw_fix_max_elo_diff  # max absolute Elo diff to trigger fix
        self.draw_fix_min_prob = draw_fix_min_prob  # minimum draw probability to enforce
        self.draw_fix_applied: int = 0  # debug counter

    # ── Away Fix Logic ─────────────────────────────────────

    def _away_fix(self, probs: dict[str, float], home_team: str, away_team: str) -> dict[str, float]:
        """Override 1X2 probabilities to predict Away when Elo strongly favours the away team.

        This compensates for a known calibration issue where the blend systematically
        underestimates away win probabilities (most pronounced in lower-division leagues
        like SE1 where the model almost never predicts away wins).

        The override only triggers when ``away_fix_enabled`` is ``True`` and the Elo
        difference (home Elo - away Elo) is below ``away_fix_elo_threshold``.

        Parameters
        ----------
        probs : dict
            Current 1X2 probabilities ``{'H': ..., 'D': ..., 'A': ...}``
        home_team : str
        away_team : str

        Returns
        -------
        dict
            Updated probabilities (may be unchanged if Elo diff is above threshold).
        """
        if not self.away_fix_enabled or self.elo is None:
            return probs

        try:
            R_h = self.elo._ratings.get(home_team, 1500.0)
            R_a = self.elo._ratings.get(away_team, 1500.0)
            elo_diff = R_h - R_a
        except Exception:
            return probs

        if elo_diff >= self.away_fix_elo_threshold:
            return probs

        self.away_fix_applied += 1

        # Override probabilities based on how extreme the Elo gap is
        p = np.array([probs["A"], probs["D"], probs["H"]])
        if elo_diff < self.away_fix_elo_threshold * 1.5:
            # Strong away favourite: give away 55%, draw 25%, home 20%
            p[0] = 0.55
            p[1] = 0.25
            p[2] = 0.20
        else:
            # Moderate away favourite: give away 50%, draw 28%, home 22%
            p[0] = 0.50
            p[1] = 0.28
            p[2] = 0.22
        # Renormalise
        total = p.sum()
        if total > 0:
            p /= total

        return {"H": float(p[2]), "D": float(p[1]), "A": float(p[0])}

    # ── Draw Fix Logic ────────────────────────────────────

    def _draw_fix(self, probs: dict[str, float], home_team: str, away_team: str) -> dict[str, float]:
        """Ensure a minimum draw probability when the Elo gap between teams is small.

        This compensates for a known calibration issue where the blend systematically
        underestimates draw probabilities in top-tier leagues (most pronounced in D1
        Bundesliga where the model predicts 0% draws despite an actual rate of ~25%).

        The override only triggers when ``draw_fix_enabled`` is ``True`` and the
        absolute Elo difference between home and away is below ``draw_fix_max_elo_diff``.
        When triggered, the draw probability is raised to at least ``draw_fix_min_prob``,
        with the excess taken from home and away proportionally.

        Parameters
        ----------
        probs : dict
            Current 1X2 probabilities ``{'H': ..., 'D': ..., 'A': ...}``
        home_team : str
        away_team : str

        Returns
        -------
        dict
            Updated probabilities (may be unchanged if Elo diff is above threshold).
        """
        if not self.draw_fix_enabled or self.elo is None:
            return probs

        try:
            R_h = self.elo._ratings.get(home_team, 1500.0)
            R_a = self.elo._ratings.get(away_team, 1500.0)
            elo_diff = abs(R_h - R_a)
        except Exception:
            return probs

        if elo_diff >= self.draw_fix_max_elo_diff:
            return probs

        p_h = probs["H"]
        p_d = probs["D"]
        p_a = probs["A"]

        if p_d >= self.draw_fix_min_prob:
            return probs  # draw already meets minimum

        self.draw_fix_applied += 1

        # Raise draw to min probability, reducing home and away proportionally
        new_d = self.draw_fix_min_prob
        remaining = 1.0 - new_d
        total_ha = p_h + p_a
        if total_ha > 0:
            new_h = remaining * (p_h / total_ha)
            new_a = remaining * (p_a / total_ha)
        else:
            new_h = remaining * 0.5
            new_a = remaining * 0.5

        return {"H": new_h, "D": new_d, "A": new_a}

    # ── Properties ────────────────────────────────────────

    @property
    def available_markets(self) -> list[str]:
        return list(self.weights.keys())

    @property
    def fitted(self) -> bool:
        dc_ok = False
        if self.dc is not None:
            if hasattr(self.dc, "_fitted"):
                dc_ok = self.dc._fitted
            elif hasattr(self.dc, "fitted"):
                dc_ok = self.dc.fitted if callable(self.dc.fitted) else self.dc.fitted
        elo_ok = hasattr(self.elo, "_ratings") and len(self.elo._ratings) > 0
        return dc_ok and elo_ok

    @property
    def calibrated(self) -> bool:
        """Whether a probability calibrator is loaded and will be applied."""
        return self._calibrator is not None

    # ── Per-League DC Resolution ───────────────────────────

    def _resolve_dc(self, home_team: str, away_team: str, league: str | None = None) -> Any:
        """Resolve the best DC model for a given fixture.

        Priority:
        1. If ``league`` is provided and exists in ``per_league_dc``, use it.
        2. Scan all per_league_dc models for one that has BOTH teams in its
           ``_team_list`` (teams only play in one league).
        3. Fall back to the global ``self.dc`` model.

        Parameters
        ----------
        home_team : str
        away_team : str
        league : str or None
            League code, if known (e.g. from a ``league`` column in the
            fixtures DataFrame).

        Returns
        -------
        DixonColesModel or None
            The best DC model for this fixture, or ``None`` if no DC model
            is available.
        """
        # Priority 1: league parameter provided
        if league and league in self.per_league_dc:
            dc = self.per_league_dc[league]
            if hasattr(dc, "_fitted") and dc._fitted:
                return dc

        # Priority 2: scan per-league models by team membership
        if self.per_league_dc:
            for league_code, dc in self.per_league_dc.items():
                if not hasattr(dc, "_fitted") or not dc._fitted:
                    continue
                if not hasattr(dc, "_team_list"):
                    continue
                teams = set(dc._team_list)
                if home_team in teams and away_team in teams:
                    return dc

        # Priority 3: fall back to global DC
        return self.dc

    # ── Calibrator Loading ────────────────────────────────

    def load_calibrator(self, path: str | Path | None = None) -> bool:
        """Load a previously-fitted probability calibrator for 1X2 calibration.

        Searches for ``blend_calibrator_*.joblib`` in the ``models/`` directory
        if no explicit path is given.  When loaded, ``predict()`` automatically
        applies the calibrator to the blend's 1X2 probabilities (Over/Under and
        BTTS are unaffected).

        Parameters
        ----------
        path : str or Path, optional
            Explicit path to a calibrator file.  If ``None``, auto-detects the
            most recent ``blend_calibrator_*.joblib`` in ``models/``.

        Returns
        -------
        bool
            ``True`` if a calibrator was successfully loaded.
        """
        import joblib

        if path is None:
            models_dir = Path("models")
            candidates = sorted(models_dir.glob("blend_calibrator_*.joblib"), reverse=True)
            if not candidates:
                logger.info("No calibrator file found in models/ (searched blend_calibrator_*.joblib)")
                return False
            path = candidates[0]

        p = Path(path)
        if not p.exists():
            logger.warning("Calibrator not found: %s", p)
            return False

        self._calibrator = joblib.load(p)
        logger.info("Calibrator loaded from %s (type=%s)", p, type(self._calibrator).__name__)
        return True

    # ── Single Fixture Prediction ─────────────────────────

    def predict(self, home_team: str, away_team: str) -> dict[str, Any]:
        """Predict all markets for a single fixture using the multi-model blend.

        When a probability calibrator is loaded (via ``load_calibrator()``),
        the 1X2 predictions are automatically calibrated.  Over/Under and
        BTTS are always taken directly from the blend.
        """
        probs_1x2 = self.predict_1x2(home_team, away_team)

        # Auto-calibrate 1X2 if a calibrator is loaded
        if self._calibrator is not None:
            try:
                raw = np.array([[probs_1x2["A"], probs_1x2["D"], probs_1x2["H"]]])
                cal = self._calibrator.transform(raw)[0]
                probs_1x2 = {
                    "H": float(cal[2]),
                    "D": float(cal[1]),
                    "A": float(cal[0]),
                }
            except Exception as exc:
                logger.warning("1X2 calibration failed: %s — using raw blend", exc)

        # Away fix applied AFTER calibration so it sticks (not overwritten by calibrator)
        if self.away_fix_enabled:
            probs_1x2 = self._away_fix(probs_1x2, home_team, away_team)

        # Draw fix applied after away fix so they compose correctly
        if self.draw_fix_enabled:
            probs_1x2 = self._draw_fix(probs_1x2, home_team, away_team)

        over_under = self.predict_over_under(home_team, away_team, 2.5)
        over_35 = self.predict_over_under(home_team, away_team, 3.5)
        btts = self.predict_btts(home_team, away_team)

        expectations = {}
        if self.dc is not None and hasattr(self.dc, "expected_goals"):
            try:
                λ_h, λ_a = self.dc.expected_goals(home_team, away_team)
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
        """Predict match outcome probabilities (1X2) using the multi-model blend.

        Gets predictions from all available models via their ``predict_proba()``
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

        models = {
            "dc": self._dc_1x2(home_team, away_team) if self.dc else None,
            "elo": self._elo_1x2(home_team, away_team),
            "xgb": self._xgb_1x2(home_team, away_team) if self.xgb else None,
            "lgb": self._lgb_1x2(home_team, away_team) if self.lgb else None,
            "cat": self._cat_1x2(home_team, away_team) if self.cat else None,
        }

        p_h = 0.0
        p_d = 0.0
        p_a = 0.0
        total_w = 0.0

        for key, probs in models.items():
            if probs is not None and key in w:
                weight = w.get(key, 0.0)
                if weight > 0:
                    p_h += weight * probs[2]
                    p_d += weight * probs[1]
                    p_a += weight * probs[0]
                    total_w += weight

        if total_w > 0:
            p_h /= total_w
            p_d /= total_w
            p_a /= total_w
        else:
            total = p_h + p_d + p_a
            if total > 0:
                p_h /= total
                p_d /= total
                p_a /= total

        # Note: away fix is NOT applied here. It's applied in `predict()` AFTER
        # calibration so the calibrator sees unfixed probabilities.
        # Direct callers of predict_1x2 (backtesting scripts) apply the fix themselves.
        return {"H": p_h, "D": p_d, "A": p_a}

    # ── Over/Under Market ─────────────────────────────────

    def predict_over_under(self, home_team: str, away_team: str, threshold: float = 2.5) -> dict[str, float]:
        """Predict Over/Under probabilities for a given goal threshold.

        Blends all available models:
        - **Dixon-Coles**: exact P(Over) from scoreline probability table
        - **XGBoost/LightGBM/CatBoost**: expected total goals → Poisson CDF

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
        market_key = f"Over{threshold:.1f}"
        w = self.weights.get(market_key, self.weights.get("Over2.5", DEFAULT_WEIGHTS["Over2.5"]))

        models = {
            "dc": self._dc_over(home_team, away_team, threshold) if self.dc else None,
            "xgb": self._xgb_over(home_team, away_team, threshold) if self.xgb else None,
            "lgb": self._lgb_over(home_team, away_team, threshold) if self.lgb else None,
            "cat": self._cat_over(home_team, away_team, threshold) if self.cat else None,
        }

        over_blend = 0.0
        total_w = 0.0

        for key, prob in models.items():
            if prob is not None and key in w:
                weight = w.get(key, 0.0)
                if weight > 0:
                    over_blend += weight * prob
                    total_w += weight

        if total_w > 0:
            over_blend /= total_w
        else:
            over_blend = 0.5

        return {"Over": over_blend, "Under": 1.0 - over_blend}

    # ── BTTS Market ───────────────────────────────────────

    def predict_btts(self, home_team: str, away_team: str) -> dict[str, float]:
        """Predict Both Teams To Score probability using the multi-model blend.

        Uses **direct** BTTS modelling for each model:
        - **Dixon-Coles**: exact BTTS from scoreline probability table
        - **Elo**: BTTS formula from Elo-derived expected goals
        - **XGBoost/LightGBM/CatBoost**: BTTS formula from 1X2 → expected goals

        All models compute BTTS via the Poisson formula:
            P(BTTS) = 1 - e^{-λ_home} - e^{-λ_away} + e^{-(λ_home + λ_away)}

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

        models = {
            "dc": self._dc_btts(home_team, away_team) if self.dc else None,
            "elo": self._elo_btts(home_team, away_team),
            "xgb": self._xgb_btts(home_team, away_team) if self.xgb else None,
            "lgb": self._lgb_btts(home_team, away_team) if self.lgb else None,
            "cat": self._cat_btts(home_team, away_team) if self.cat else None,
        }

        btts_blend = 0.0
        total_w = 0.0

        for key, prob in models.items():
            if prob is not None and key in w:
                weight = w.get(key, 0.0)
                if weight > 0:
                    btts_blend += weight * prob
                    total_w += weight

        if total_w > 0:
            btts_blend /= total_w
        else:
            btts_blend = 0.5

        return {"BTTS": btts_blend, "No BTTS": 1.0 - btts_blend}

    def _elo_btts(self, home_team: str, away_team: str) -> float | None:
        try:
            if hasattr(self.elo, "predict_btts") and callable(self.elo.predict_btts):
                result = self.elo.predict_btts(home_team, away_team)
                if result is not None:
                    return float(result)
        except Exception:
            pass
        return None

    def _ml_btts(self, probs_1x2: np.ndarray) -> float | None:
        """Compute BTTS from a tree model's 1X2 probs → expected goals → Poisson."""
        try:
            cr = self.cond_rates
            exp_total = (
                probs_1x2[2] * cr.mean_total_given_home_win
                + probs_1x2[1] * cr.mean_total_given_draw
                + probs_1x2[0] * cr.mean_total_given_away_win
            )
            if exp_total <= 0:
                return 0.50
            exp_home = exp_total * 0.55
            exp_away = exp_total * 0.45
            p_h0 = np.exp(-exp_home)
            p_a0 = np.exp(-exp_away)
            return float(np.clip(1.0 - p_h0 - p_a0 + (p_h0 * p_a0), 0.0, 1.0))
        except Exception:
            return 0.50

    def _ml_over(self, probs_1x2: np.ndarray, threshold: float) -> float:
        """Compute P(Over) from a tree model's 1X2 probs → exp. total → Poisson CDF."""
        try:
            cr = self.cond_rates
            exp_total = (
                probs_1x2[2] * cr.mean_total_given_home_win
                + probs_1x2[1] * cr.mean_total_given_draw
                + probs_1x2[0] * cr.mean_total_given_away_win
            )
            if exp_total <= 0:
                return 0.50
            return 1.0 - _poisson_cdf(threshold, exp_total)
        except Exception:
            return 0.50

    # ── Individual Model Proxies ──────────────────────────

    def _align_features(self, X: pd.DataFrame, model: Any) -> pd.DataFrame | None:
        """Align feature columns to match what a tree model expects.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix from ``_FeatureBuilder.build()``.
        model : Any
            Fitted model with ``feature_names_in_`` or ``feature_name_``.

        Returns
        -------
        pd.DataFrame or None
            Aligned feature matrix, or ``None`` if alignment fails.
        """
        if X is None or len(X) == 0:
            return None
        try:
            expected = None
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
            elif hasattr(model, "feature_name_"):
                expected = list(model.feature_name_)
            elif hasattr(model, "feature_names_"):
                expected = list(model.feature_names_)
            if expected is None:
                return X
            missing = set(expected) - set(X.columns)
            for col in missing:
                X[col] = np.nan
            return X[expected]
        except Exception as exc:
            logger.warning("Feature alignment failed for %s: %s", type(model).__name__, exc)
            return None

    def _dc_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        try:
            if hasattr(self.dc, "predict_proba"):
                df = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
                return cast(np.ndarray, self.dc.predict_proba(df)[0])
            return np.array([0.33, 0.34, 0.33])
        except Exception:
            return np.array([0.33, 0.34, 0.33])

    def _resolve_dc_for_fixture(self, home_team: str, away_team: str, league: str | None = None) -> Any:
        """Resolve the best DC model for O/U and BTTS on a single fixture.

        Uses ``_resolve_dc()`` to pick per-league model or global fallback.
        If the resolved model has both teams, returns it; otherwise falls
        back to ``self.dc``.
        """
        dc = self._resolve_dc(home_team, away_team, league=league)
        if dc is not None and hasattr(dc, "_fitted") and dc._fitted:
            return dc
        return self.dc

    def _dc_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        try:
            # 1X2 always uses global DC (tree models dominate 1X2)
            if hasattr(self.dc, "predict_proba"):
                df = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
                return cast(np.ndarray, self.dc.predict_proba(df)[0])
            return np.array([0.33, 0.34, 0.33])
        except Exception:
            return np.array([0.33, 0.34, 0.33])

    def _dc_over(self, home_team: str, away_team: str, threshold: float, league: str | None = None) -> float:
        """Dixon-Coles Over/Under probability, using per-league DC if available."""
        dc = self._resolve_dc_for_fixture(home_team, away_team, league=league)
        try:
            if dc is not None and hasattr(dc, "predict"):
                r = dc.predict(home_team, away_team, over_under_threshold=threshold)
                return float(getattr(r, "over_2_5_prob" if threshold == 2.5 else "over_3_5_prob", 0.50))
            return 0.50
        except Exception:
            return 0.50

    def _dc_btts(self, home_team: str, away_team: str, league: str | None = None) -> float:
        """Dixon-Coles BTTS probability, using per-league DC if available."""
        dc = self._resolve_dc_for_fixture(home_team, away_team, league=league)
        try:
            if dc is not None and hasattr(dc, "predict"):
                r = dc.predict(home_team, away_team)
                return float(getattr(r, "btts_prob", 0.50))
            return 0.50
        except Exception:
            return 0.50

    def _elo_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        try:
            # If a form adjuster is loaded, compute form-adjusted Elo probs
            if self.form_adjuster is not None and hasattr(self.elo, "expected_score"):
                R_home_raw = self.elo._ratings.get(home_team, float(self.elo.initial_rating))
                R_away_raw = self.elo._ratings.get(away_team, float(self.elo.initial_rating))
                R_home = self.form_adjuster.adjust_rating(R_home_raw, home_team)
                R_away = self.form_adjuster.adjust_rating(R_away_raw, away_team)
                E_home = self.elo.expected_score(R_home, R_away)
                # Reuse the EloSystem's own probability conversion
                if hasattr(self.elo, "_expected_to_probs"):
                    return np.array(self.elo._expected_to_probs(E_home))
                # Fallback if method not available
                E_away = 1.0 - E_home
                diff = abs(E_home - E_away)
                p_draw = 0.25 * (1.0 - diff)
                p_home = E_home - p_draw / 2.0
                p_away = 1.0 - E_home - p_draw / 2.0
                return np.array([p_away, p_draw, p_home])
            # Standard Elo prediction (no form adjustment)
            df = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
            return cast(np.ndarray, self.elo.predict_proba(df)[0])
        except Exception:
            return np.array([0.33, 0.34, 0.33])

    def _ml_predict_proba(self, model: Any, home_team: str, away_team: str) -> np.ndarray | None:
        """Generic predict_proba for any tree model via feature builder."""
        try:
            X = self._feature_builder.build([home_team], [away_team])
            if X is not None and len(X) > 0:
                X_aligned = self._align_features(X, model)
                if X_aligned is not None and len(X_aligned) > 0:
                    return cast(np.ndarray, model.predict_proba(X_aligned)[0])
        except Exception:
            pass
        return None

    def _xgb_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        if self.xgb is None:
            return np.array([0.33, 0.34, 0.33])
        result = self._ml_predict_proba(self.xgb, home_team, away_team)
        return result if result is not None else np.array([0.33, 0.34, 0.33])

    def _xgb_btts(self, home_team: str, away_team: str) -> float | None:
        # Try market-specific BTTS model first
        if "xgb" in self.btts_models:
            result = self._ml_binary_predict_proba(self.btts_models["xgb"], home_team, away_team)
            if result is not None:
                return float(result[1])  # binary: [P(No), P(BTTS)]
        if self.xgb is None:
            return None
        probs = self._xgb_1x2(home_team, away_team)
        if np.array_equal(probs, np.array([0.33, 0.34, 0.33])):
            return None
        return self._ml_btts(probs)

    def _xgb_over(self, home_team: str, away_team: str, threshold: float) -> float:
        # Try market-specific O/U model first
        if "xgb" in self.ou_models and threshold == 2.5:
            result = self._ml_binary_predict_proba(self.ou_models["xgb"], home_team, away_team)
            if result is not None:
                return float(result[1])  # binary: [P(Under), P(Over)]
        if self.xgb is None:
            return 0.50
        probs = self._xgb_1x2(home_team, away_team)
        if np.array_equal(probs, np.array([0.33, 0.34, 0.33])):
            return 0.50
        return self._ml_over(probs, threshold)

    def _lgb_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        if self.lgb is None:
            return np.array([0.33, 0.34, 0.33])
        result = self._ml_predict_proba(self.lgb, home_team, away_team)
        return result if result is not None else np.array([0.33, 0.34, 0.33])

    def _lgb_btts(self, home_team: str, away_team: str) -> float | None:
        # Try market-specific BTTS model first
        if "lgb" in self.btts_models:
            result = self._ml_binary_predict_proba(self.btts_models["lgb"], home_team, away_team)
            if result is not None:
                return float(result[1])
        if self.lgb is None:
            return None
        probs = self._lgb_1x2(home_team, away_team)
        if np.array_equal(probs, np.array([0.33, 0.34, 0.33])):
            return None
        return self._ml_btts(probs)

    def _lgb_over(self, home_team: str, away_team: str, threshold: float) -> float:
        # Try market-specific O/U model first
        if "lgb" in self.ou_models and threshold == 2.5:
            result = self._ml_binary_predict_proba(self.ou_models["lgb"], home_team, away_team)
            if result is not None:
                return float(result[1])
        if self.lgb is None:
            return 0.50
        probs = self._lgb_1x2(home_team, away_team)
        if np.array_equal(probs, np.array([0.33, 0.34, 0.33])):
            return 0.50
        return self._ml_over(probs, threshold)

    def _cat_1x2(self, home_team: str, away_team: str) -> np.ndarray:
        if self.cat is None:
            return np.array([0.33, 0.34, 0.33])
        result = self._ml_predict_proba(self.cat, home_team, away_team)
        return result if result is not None else np.array([0.33, 0.34, 0.33])

    def _cat_btts(self, home_team: str, away_team: str) -> float | None:
        # Try market-specific BTTS model first
        if "cat" in self.btts_models:
            result = self._ml_binary_predict_proba(self.btts_models["cat"], home_team, away_team)
            if result is not None:
                return float(result[1])
        if self.cat is None:
            return None
        probs = self._cat_1x2(home_team, away_team)
        if np.array_equal(probs, np.array([0.33, 0.34, 0.33])):
            return None
        return self._ml_btts(probs)

    def _cat_over(self, home_team: str, away_team: str, threshold: float) -> float:
        # Try market-specific O/U model first
        if "cat" in self.ou_models and threshold == 2.5:
            result = self._ml_binary_predict_proba(self.ou_models["cat"], home_team, away_team)
            if result is not None:
                return float(result[1])
        if self.cat is None:
            return 0.50
        probs = self._cat_1x2(home_team, away_team)
        if np.array_equal(probs, np.array([0.33, 0.34, 0.33])):
            return 0.50
        return self._ml_over(probs, threshold)

    def _batch_ml_predict(self, model: Any, X: pd.DataFrame | None) -> np.ndarray | None:
        """Run predict_proba on a pre-built feature matrix (batched).

        Parameters
        ----------
        model : Any
            Fitted tree model with ``predict_proba(X)``.
        X : pd.DataFrame or None
            Pre-built feature matrix for ALL fixtures.

        Returns
        -------
        np.ndarray or None
            ``(n_fixtures, 3)`` probability array, or ``None`` if model/X
            are unavailable.
        """
        if model is None or X is None or len(X) == 0:
            return None
        X_aligned = self._align_features(X, model)
        if X_aligned is None or len(X_aligned) == 0:
            return None
        try:
            return cast(np.ndarray, model.predict_proba(X_aligned))
        except Exception as exc:
            logger.warning("Batch predict failed for %s: %s", type(model).__name__, exc)
            return None

    def _batch_ml_predict_binary(self, model: Any, X: pd.DataFrame | None) -> np.ndarray | None:
        """Run predict_proba on a pre-built feature matrix (batched) — binary output.

        Returns the **positive class** (index 1) probability for each fixture,
        suitable for O/U or BTTS market-specific models.

        Parameters
        ----------
        model : Any
            Fitted binary tree model with ``predict_proba(X)`` returning (n, 2).
        X : pd.DataFrame or None
            Pre-built feature matrix for ALL fixtures.

        Returns
        -------
        np.ndarray or None
            ``(n_fixtures,)`` array of P(positive) probabilities, or ``None``.
        """
        if model is None or X is None or len(X) == 0:
            return None
        X_aligned = self._align_features(X, model)
        if X_aligned is None or len(X_aligned) == 0:
            return None
        try:
            probs = cast(np.ndarray, model.predict_proba(X_aligned))
            return probs[:, 1]  # binary: [P(0), P(1)]
        except Exception as exc:
            logger.warning("Batch binary predict failed for %s: %s", type(model).__name__, exc)
            return None

    def _ml_binary_predict_proba(self, model: Any, home_team: str, away_team: str) -> np.ndarray | None:
        """Single-fixture predict_proba for a binary market-specific model.

        Parameters
        ----------
        model : Any
            Fitted binary tree model.
        home_team : str
        away_team : str

        Returns
        -------
        np.ndarray or None
            ``(2,)`` array with ``[P(negative), P(positive)]``, or ``None``.
        """
        try:
            X = self._feature_builder.build([home_team], [away_team])
            if X is not None and len(X) > 0:
                X_aligned = self._align_features(X, model)
                if X_aligned is not None and len(X_aligned) > 0:
                    return cast(np.ndarray, model.predict_proba(X_aligned)[0])
        except Exception:
            pass
        return None

    # ── Load Market-Specific Models ───────────────────────

    def load_market_models(self, league: str | None = None, models_dir: str | Path | None = None) -> dict[str, int]:
        """Load market-specific O/U and BTTS tree models from disk.

        Searches for files matching ``{model_type}_ou.joblib`` and
        ``{model_type}_btts.joblib`` (e.g. ``xgboost_ou.joblib``,
        ``lightgbm_btts.joblib``) and loads them into ``self.ou_models``
        and ``self.btts_models`` dicts.

        Parameters
        ----------
        league : str, optional
            League code (e.g. ``"F1"``).  If provided, looks in
            ``models/per_league/{league}/``.
        models_dir : str or Path, optional
            Explicit directory to search.  Supersedes ``league``.

        Returns
        -------
        dict[str, int]
            Summary of loaded models: ``{"ou": n, "btts": n}``.
        """
        import joblib

        if models_dir is None:
            if league:
                models_dir = Path("models") / "per_league" / league
            else:
                models_dir = Path("models")
        models_dir = Path(models_dir)

        if not models_dir.exists():
            logger.warning("Market models directory not found: %s", models_dir)
            return {"ou": 0, "btts": 0}

        model_type_map = {
            "xgboost": "xgb",
            "lightgbm": "lgb",
            "catboost": "cat",
        }

        loaded = {"ou": 0, "btts": 0}

        for pattern in ["*_ou.joblib", "*_btts.joblib"]:
            for fpath in sorted(models_dir.glob(pattern)):
                stem = fpath.stem  # e.g. "xgboost_ou"
                parts = stem.split("_")
                if len(parts) < 2:
                    continue
                # The last part is the market ("ou" or "btts")
                market = parts[-1]
                # Everything before the last part is the model name
                model_name = "_".join(parts[:-1])
                # Map to short names
                short_key = model_type_map.get(model_name, model_name)

                try:
                    model = joblib.load(fpath)
                    if market == "ou":
                        self.ou_models[short_key] = model
                        loaded["ou"] += 1
                        logger.info("Loaded O/U model: %s → ou_models[%s]", fpath.name, short_key)
                    elif market == "btts":
                        self.btts_models[short_key] = model
                        loaded["btts"] += 1
                        logger.info("Loaded BTTS model: %s → btts_models[%s]", fpath.name, short_key)
                except Exception as exc:
                    logger.warning("Failed to load market model %s: %s", fpath.name, exc)

        logger.info("Market models loaded: %d O/U, %d BTTS from %s", loaded["ou"], loaded["btts"], models_dir)
        return loaded

    # ── Batch Prediction ──────────────────────────────────

    def predict_matches(self, df: pd.DataFrame, home_col: str = "home_team", away_col: str = "away_team") -> pd.DataFrame:
        """Predict all markets for multiple fixtures in **batched** fashion.

        **Performance:** Feature engineering is done **once** for all fixtures
        instead of once per fixture per tree model.  This reduces the cost
        from ``N × M`` ``build_features()`` calls to just **1** (where N =
        number of fixtures and M = number of tree models).

        For a typical 50-fixture, 3-tree-model batch this cuts prediction time
        from ~25 minutes to ~10–15 seconds.
        """
        home_teams = df[home_col].tolist()
        away_teams = df[away_col].tolist()
        n = len(df)

        # ── Batch feature engineering ONCE for all tree models ──
        X = self._feature_builder.build(home_teams, away_teams)

        # ── Batch tree model predict_proba (once per model) ──
        xgb_probs = self._batch_ml_predict(self.xgb, X)
        lgb_probs = self._batch_ml_predict(self.lgb, X)
        cat_probs = self._batch_ml_predict(self.cat, X)

        # ── Batch binary predictions for market-specific models ──
        # Only compute if their model keys have non-zero weight
        w_ou = self.weights.get("Over2.5", DEFAULT_WEIGHTS["Over2.5"])
        w_btts = self.weights.get("BTTS", DEFAULT_WEIGHTS["BTTS"])
        need_ou_trees = any(k in w_ou and w_ou[k] > 0 for k in ("xgb", "lgb", "cat"))
        need_btts_trees = any(k in w_btts and w_btts[k] > 0 for k in ("xgb", "lgb", "cat"))

        xgb_ou_preds = self._batch_ml_predict_binary(self.ou_models.get("xgb"), X) if ("xgb" in self.ou_models and need_ou_trees) else None
        lgb_ou_preds = self._batch_ml_predict_binary(self.ou_models.get("lgb"), X) if ("lgb" in self.ou_models and need_ou_trees) else None
        cat_ou_preds = self._batch_ml_predict_binary(self.ou_models.get("cat"), X) if ("cat" in self.ou_models and need_ou_trees) else None
        xgb_btts_preds = self._batch_ml_predict_binary(self.btts_models.get("xgb"), X) if ("xgb" in self.btts_models and need_btts_trees) else None
        lgb_btts_preds = self._batch_ml_predict_binary(self.btts_models.get("lgb"), X) if ("lgb" in self.btts_models and need_btts_trees) else None
        cat_btts_preds = self._batch_ml_predict_binary(self.btts_models.get("cat"), X) if ("cat" in self.btts_models and need_btts_trees) else None

        w_1x2 = self.weights.get("1X2", DEFAULT_WEIGHTS["1X2"])
        w_ou = self.weights.get("Over2.5", DEFAULT_WEIGHTS["Over2.5"])
        w_ou35 = self.weights.get("Over3.5", DEFAULT_WEIGHTS["Over3.5"])
        w_btts = self.weights.get("BTTS", DEFAULT_WEIGHTS["BTTS"])

        records: list[dict[str, Any]] = []
        for i, (home, away) in enumerate(zip(home_teams, away_teams)):
            # ── DC predictions (fast per-fixture — no feature build) ──
            # Use league column if available for per-league DC routing
            league_ctx = None
            if "league" in df.columns:
                try:
                    league_ctx = str(df.iloc[i].get("league", ""))
                    if not league_ctx or league_ctx == "nan":
                        league_ctx = None
                except (IndexError, ValueError):
                    pass
            dc_1x2 = self._dc_1x2(home, away) if self.dc else None
            dc_btts_val = self._dc_btts(home, away, league=league_ctx) if self.dc else None
            dc_over25 = self._dc_over(home, away, 2.5, league=league_ctx) if self.dc else None
            dc_over35 = self._dc_over(home, away, 3.5, league=league_ctx) if self.dc else None

            # ── Elo predictions (fast per-fixture — no feature build) ──
            elo_1x2 = self._elo_1x2(home, away)
            elo_btts_val = self._elo_btts(home, away)

            # ── Tree model predictions from batched compute ──
            xgb_1x2_i = xgb_probs[i] if xgb_probs is not None else None
            lgb_1x2_i = lgb_probs[i] if lgb_probs is not None else None
            cat_1x2_i = cat_probs[i] if cat_probs is not None else None

            # ═══════════════════════════════════════════════════
            #  Blend 1X2
            # ═══════════════════════════════════════════════════
            models_1x2: list[tuple[str, Any]] = [
                ("dc", dc_1x2), ("elo", elo_1x2),
                ("xgb", xgb_1x2_i), ("lgb", lgb_1x2_i), ("cat", cat_1x2_i),
            ]
            p_h = p_d = p_a = 0.0
            total_w = 0.0
            for key, probs in models_1x2:
                if probs is not None and key in w_1x2:
                    weight = w_1x2.get(key, 0.0)
                    if weight > 0:
                        p_h += weight * probs[2]
                        p_d += weight * probs[1]
                        p_a += weight * probs[0]
                        total_w += weight
            if total_w > 0:
                p_h /= total_w
                p_d /= total_w
                p_a /= total_w
            else:
                total = p_h + p_d + p_a
                if total > 0:
                    p_h /= total
                    p_d /= total
                    p_a /= total

            # ═══════════════════════════════════════════════════
            #  Blend BTTS — market-specific model preferred, fallback to derived-from-1X2
            # ═══════════════════════════════════════════════════
            xgb_btts_i = xgb_btts_preds[i] if xgb_btts_preds is not None else (self._ml_btts(xgb_1x2_i) if xgb_1x2_i is not None else None)
            lgb_btts_i = lgb_btts_preds[i] if lgb_btts_preds is not None else (self._ml_btts(lgb_1x2_i) if lgb_1x2_i is not None else None)
            cat_btts_i = cat_btts_preds[i] if cat_btts_preds is not None else (self._ml_btts(cat_1x2_i) if cat_1x2_i is not None else None)
            models_btts: list[tuple[str, float | None]] = [
                ("dc", dc_btts_val),
                ("elo", elo_btts_val),
                ("xgb", xgb_btts_i),
                ("lgb", lgb_btts_i),
                ("cat", cat_btts_i),
            ]
            btts_blend = 0.0
            total_w = 0.0
            for key, val in models_btts:
                if val is not None and key in w_btts:
                    weight = w_btts.get(key, 0.0)
                    if weight > 0:
                        btts_blend += weight * val
                        total_w += weight
            if total_w > 0:
                btts_blend /= total_w
            else:
                btts_blend = 0.5

            # ═══════════════════════════════════════════════════
            #  Blend Over/Under 2.5 — market-specific model preferred, fallback to derived
            # ═══════════════════════════════════════════════════
            xgb_ou_i = xgb_ou_preds[i] if xgb_ou_preds is not None else (self._ml_over(xgb_1x2_i, 2.5) if xgb_1x2_i is not None else None)
            lgb_ou_i = lgb_ou_preds[i] if lgb_ou_preds is not None else (self._ml_over(lgb_1x2_i, 2.5) if lgb_1x2_i is not None else None)
            cat_ou_i = cat_ou_preds[i] if cat_ou_preds is not None else (self._ml_over(cat_1x2_i, 2.5) if cat_1x2_i is not None else None)
            models_ou25: list[tuple[str, float | None]] = [
                ("dc", dc_over25),
                ("xgb", xgb_ou_i),
                ("lgb", lgb_ou_i),
                ("cat", cat_ou_i),
            ]
            over25 = 0.0
            total_w = 0.0
            for key, val in models_ou25:
                if val is not None and key in w_ou:
                    weight = w_ou.get(key, 0.0)
                    if weight > 0:
                        over25 += weight * val
                        total_w += weight
            if total_w > 0:
                over25 /= total_w
            else:
                over25 = 0.5

            # ═══════════════════════════════════════════════════
            #  Blend Over/Under 3.5  (always derived — no market model for 3.5)
            # ═══════════════════════════════════════════════════
            models_ou35: list[tuple[str, float | None]] = [
                ("dc", dc_over35),
                ("xgb", self._ml_over(xgb_1x2_i, 3.5) if xgb_1x2_i is not None else None),
                ("lgb", self._ml_over(lgb_1x2_i, 3.5) if lgb_1x2_i is not None else None),
                ("cat", self._ml_over(cat_1x2_i, 3.5) if cat_1x2_i is not None else None),
            ]
            over35 = 0.0
            total_w = 0.0
            for key, val in models_ou35:
                if val is not None and key in w_ou35:
                    weight = w_ou35.get(key, 0.0)
                    if weight > 0:
                        over35 += weight * val
                        total_w += weight
            if total_w > 0:
                over35 /= total_w
            else:
                over35 = 0.5

            # ═══════════════════════════════════════════════════
            #  Expected goals
            # ═══════════════════════════════════════════════════
            expectations: dict[str, float] = {}
            if self.dc is not None and hasattr(self.dc, "expected_goals"):
                try:
                    eh, ea = self.dc.expected_goals(home, away)
                    expectations = {
                        "expected_home_goals": round(float(eh), 4),
                        "expected_away_goals": round(float(ea), 4),
                        "expected_total_goals": round(float(eh + ea), 4),
                    }
                except Exception:
                    pass

            # ═══════════════════════════════════════════════════
            #  Build flat record
            # ═══════════════════════════════════════════════════
            flat: dict[str, Any] = {
                "home_team": home,
                "away_team": away,
                "home_win_prob": round(p_h, 4),
                "draw_prob": round(p_d, 4),
                "away_win_prob": round(p_a, 4),
                "over_2_5_prob": round(over25, 4),
                "under_2_5_prob": round(1.0 - over25, 4),
                "over_3_5_prob": round(over35, 4),
                "under_3_5_prob": round(1.0 - over35, 4),
                "btts_prob": round(btts_blend, 4),
                "btts_no_prob": round(1.0 - btts_blend, 4),
            }
            if expectations:
                flat.update(expectations)

            # Apply away-fix override in batch mode too
            if self.away_fix_enabled and self.elo is not None:
                try:
                    R_h = self.elo._ratings.get(home, 1500.0)
                    R_a = self.elo._ratings.get(away, 1500.0)
                    elo_diff = R_h - R_a
                    if elo_diff < self.away_fix_elo_threshold:
                        self.away_fix_applied += 1
                        if elo_diff < self.away_fix_elo_threshold * 1.5:
                            p_a, p_d, p_h = 0.55, 0.25, 0.20
                        else:
                            p_a, p_d, p_h = 0.50, 0.28, 0.22
                        # Sync flat record with fixed probabilities
                        flat["home_win_prob"] = round(p_h, 4)
                        flat["draw_prob"] = round(p_d, 4)
                        flat["away_win_prob"] = round(p_a, 4)
                except Exception:
                    pass

            # Apply draw-fix override in batch mode too
            if self.draw_fix_enabled and self.elo is not None:
                try:
                    R_h = self.elo._ratings.get(home, 1500.0)
                    R_a = self.elo._ratings.get(away, 1500.0)
                    if abs(R_h - R_a) < self.draw_fix_max_elo_diff and p_d < self.draw_fix_min_prob:
                        self.draw_fix_applied += 1
                        new_d = self.draw_fix_min_prob
                        remaining = 1.0 - new_d
                        total_ha = p_h + p_a
                        if total_ha > 0:
                            p_h = remaining * (p_h / total_ha)
                            p_a = remaining * (p_a / total_ha)
                        else:
                            p_h = remaining * 0.5
                            p_a = remaining * 0.5
                        p_d = new_d
                        # Sync flat record
                        flat["home_win_prob"] = round(p_h, 4)
                        flat["draw_prob"] = round(p_d, 4)
                        flat["away_win_prob"] = round(p_a, 4)
                except Exception:
                    pass

            # Predicted outcome
            if p_h >= p_d and p_h >= p_a:
                flat["predicted_outcome"] = "Home Win"
            elif p_d >= p_a:
                flat["predicted_outcome"] = "Draw"
            else:
                flat["predicted_outcome"] = "Away Win"
            flat["confidence"] = round(max(p_h, p_d, p_a), 4)

            records.append(flat)

        return pd.DataFrame(records)

    # ═══════════════════════════════════════════════════════
    #  Pre-computation (cached for optimisation speed)
    # ═══════════════════════════════════════════════════════

    def precompute(self, df: pd.DataFrame, home_col: str = "home_team", away_col: str = "away_team",
                   cache_key: str = "default") -> PerModelPredictions:
        """Pre-compute per-model predictions for all matches in df.

        Computes all model predictions once and caches them for fast weight
        grid search.  DC binary markets (BTTS, O/U) come from the DC model's
        scoreline table.  Tree model binary markets are derived from 1X2 probs
        via expected total goals → Poisson CDF / Poisson BTTS formula.
        """
        home_teams = df[home_col].tolist()
        away_teams = df[away_col].tolist()
        n = len(df)

        dc_1x2_list, elo_1x2_list = [], []
        dc_btts_list, dc_over25_list, dc_over35_list = [], [], []
        elo_btts_list = []
        dc_eh_list, dc_ea_list = [], []
        xgb_1x2_list: list[float | np.ndarray | list[float]] = []
        lgb_1x2_list: list[float | np.ndarray | list[float]] = []
        cat_1x2_list: list[float | np.ndarray | list[float]] = []
        xgb_btts_list: list[float] = []
        lgb_btts_list: list[float] = []
        cat_btts_list: list[float] = []

        for ht, at in zip(home_teams, away_teams):
            # Dixon-Coles 1X2 + binary markets
            if self.dc is not None:
                try:
                    if hasattr(self.dc, "predict"):
                        r = self.dc.predict(ht, at)
                        dc_1x2_list.append([r.away_win_prob, r.draw_prob, r.home_win_prob])
                        dc_btts_list.append(r.btts_prob)
                        dc_over25_list.append(r.over_2_5_prob)
                        dc_over35_list.append(r.over_3_5_prob)
                        dc_eh_list.append(r.expected_home_goals)
                        dc_ea_list.append(r.expected_away_goals)
                    elif hasattr(self.dc, "predict_proba"):
                        df_single = pd.DataFrame([{"home_team": ht, "away_team": at}])
                        dc_1x2_list.append(self.dc.predict_proba(df_single)[0])
                        dc_btts_list.append(0.5)
                        dc_over25_list.append(0.5)
                        dc_over35_list.append(0.5)
                        dc_eh_list.append(0.0)
                        dc_ea_list.append(0.0)
                    else:
                        raise AttributeError("DC model has no predict or predict_proba")
                except Exception:
                    dc_1x2_list.append([0.33, 0.34, 0.33])
                    dc_btts_list.append(0.5)
                    dc_over25_list.append(0.5)
                    dc_over35_list.append(0.5)
                    dc_eh_list.append(0.0)
                    dc_ea_list.append(0.0)
            else:
                dc_1x2_list.append([0.33, 0.34, 0.33])
                dc_btts_list.append(0.5)
                dc_over25_list.append(0.5)
                dc_over35_list.append(0.5)
                dc_eh_list.append(0.0)
                dc_ea_list.append(0.0)

            # Elo 1X2 and direct BTTS
            try:
                df_single = pd.DataFrame([{"home_team": ht, "away_team": at}])
                elo_1x2_list.append(self.elo.predict_proba(df_single)[0])
                elo_btts_list.append(self.elo.predict_btts(ht, at))
            except Exception:
                elo_1x2_list.append([0.33, 0.34, 0.33])
                elo_btts_list.append(0.5)

        # Tree models — batch feature engineering
        cr = self.cond_rates
        X = self._feature_builder.build(home_teams, away_teams)

        _tree_batch_predict(
            self.xgb, xgb_1x2_list, xgb_btts_list, n, X, cr, self
        )
        _tree_batch_predict(
            self.lgb, lgb_1x2_list, lgb_btts_list, n, X, cr, self
        )
        _tree_batch_predict(
            self.cat, cat_1x2_list, cat_btts_list, n, X, cr, self
        )

        ppm = PerModelPredictions(
            dc_1x2=np.array(dc_1x2_list),
            elo_1x2=np.array(elo_1x2_list),
            xgb_1x2=np.array(xgb_1x2_list),
            lgb_1x2=np.array(lgb_1x2_list),
            cat_1x2=np.array(cat_1x2_list),
            dc_btts=np.array(dc_btts_list),
            elo_btts=np.array(elo_btts_list),
            xgb_btts=np.array(xgb_btts_list),
            lgb_btts=np.array(lgb_btts_list),
            cat_btts=np.array(cat_btts_list),
            dc_over_25=np.array(dc_over25_list),
            dc_over_35=np.array(dc_over35_list),
            dc_exp_home=np.array(dc_eh_list),
            dc_exp_away=np.array(dc_ea_list),
            n=n,
        )
        self._cache[cache_key] = ppm
        return ppm

    def _blend_1x2(self, ppm: PerModelPredictions, w: dict[str, float]) -> np.ndarray:
        model_map = {
            "dc": ppm.dc_1x2,
            "elo": ppm.elo_1x2,
            "xgb": ppm.xgb_1x2,
            "lgb": ppm.lgb_1x2,
            "cat": ppm.cat_1x2,
        }
        total_w = 0.0
        result = np.zeros_like(ppm.dc_1x2)
        for key, probs in model_map.items():
            weight = w.get(key, 0.0)
            if weight > 0:
                result += weight * probs
                total_w += weight
        if total_w <= 0:
            return ppm.dc_1x2.copy()
        result /= total_w
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cast(np.ndarray, result / row_sums)

    def _blend_binary(self, ppm: PerModelPredictions, w: dict[str, float], market: str) -> np.ndarray:
        """Blend binary market (BTTS or O/U) across all available models.

        For **BTTS**: direct from each model's BTTS computation.
        For **Over/Under**: DC exact + tree model Poisson CDF + Elo conditional rates.
        """
        model_map: dict[str, np.ndarray] = {"elo": ppm.elo_btts}
        if market == "BTTS":
            model_map["dc"] = ppm.dc_btts
            model_map["xgb"] = ppm.xgb_btts
            model_map["lgb"] = ppm.lgb_btts
            model_map["cat"] = ppm.cat_btts
        elif "3.5" in market:
            model_map["dc"] = ppm.dc_over_35
            model_map["xgb"] = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 3.5)
            model_map["lgb"] = self.cond_rates.ou_from_1x2(ppm.lgb_1x2, 3.5)
            model_map["cat"] = self.cond_rates.ou_from_1x2(ppm.cat_1x2, 3.5)
        else:
            model_map["dc"] = ppm.dc_over_25
            model_map["xgb"] = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 2.5)
            model_map["lgb"] = self.cond_rates.ou_from_1x2(ppm.lgb_1x2, 2.5)
            model_map["cat"] = self.cond_rates.ou_from_1x2(ppm.cat_1x2, 2.5)

        total_w = 0.0
        result = np.zeros(ppm.n)
        for key, vals in model_map.items():
            weight = w.get(key, 0.0)
            if weight > 0:
                result += weight * vals
                total_w += weight

        if total_w <= 0:
            return np.full(ppm.n, 0.5)
        return result / total_w

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
            w = self.weights.get(market, {})
            models_in_market = [m for m in ranges.keys() if w.get(m, 0) > 0 or ranges[m][1] > 0]
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
        mse = float(np.mean((ppm.dc_total_goals - actual_total) ** 2))
        mae = float(np.mean(np.abs(ppm.dc_total_goals - actual_total)))

        results: dict[str, Any] = {
            "n_test": ppm.n,
            "expected_goals": {"mse": round(mse, 4), "mae": round(mae, 4)},
            "markets": {},
        }

        # ── Individual model predictions ──
        elo_btts = ppm.elo_btts
        xgb_btts = ppm.xgb_btts
        lgb_btts = ppm.lgb_btts
        cat_btts = ppm.cat_btts
        # Over/Under: derived from 1X2 via conditional rates for tree models
        elo_ou25 = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 2.5)
        elo_ou35 = self.cond_rates.ou_from_1x2(ppm.elo_1x2, 3.5)
        xgb_ou25 = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 2.5)
        xgb_ou35 = self.cond_rates.ou_from_1x2(ppm.xgb_1x2, 3.5)
        lgb_ou25 = self.cond_rates.ou_from_1x2(ppm.lgb_1x2, 2.5)
        lgb_ou35 = self.cond_rates.ou_from_1x2(ppm.lgb_1x2, 3.5)
        cat_ou25 = self.cond_rates.ou_from_1x2(ppm.cat_1x2, 2.5)
        cat_ou35 = self.cond_rates.ou_from_1x2(ppm.cat_1x2, 3.5)

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
                    md["models"]["Dixon-Coles"] = _metrics_1x2(actual_result, ppm.dc_1x2)
                    md["models"]["Elo"] = _metrics_1x2(actual_result, ppm.elo_1x2)
                    md["models"]["XGBoost"] = _metrics_1x2(actual_result, ppm.xgb_1x2)
                    md["models"]["LightGBM"] = _metrics_1x2(actual_result, ppm.lgb_1x2)
                    md["models"]["CatBoost"] = _metrics_1x2(actual_result, ppm.cat_1x2)
                    if ens_1x2 is not None:
                        md["models"]["Current Ensemble"] = _metrics_1x2(actual_result, ens_1x2)
                blend = self._blend_1x2(ppm, w)
                md["models"]["Multi-Model Blend"] = _metrics_1x2(actual_result, blend)

            elif market_name == "Over2.5":
                if include_individual:
                    md["models"]["Dixon-Coles"] = _metrics_binary(actual_ou25, ppm.dc_over_25)
                    md["models"]["Elo"] = _metrics_binary(actual_ou25, elo_ou25)
                    md["models"]["XGBoost"] = _metrics_binary(actual_ou25, xgb_ou25)
                    md["models"]["LightGBM"] = _metrics_binary(actual_ou25, lgb_ou25)
                    md["models"]["CatBoost"] = _metrics_binary(actual_ou25, cat_ou25)
                blend = self._blend_binary(ppm, w, "Over2.5")
                md["models"]["Multi-Model Blend"] = _metrics_binary(actual_ou25, blend)

            elif market_name == "Over3.5":
                if include_individual:
                    md["models"]["Dixon-Coles"] = _metrics_binary(actual_ou35, ppm.dc_over_35)
                    md["models"]["Elo"] = _metrics_binary(actual_ou35, elo_ou35)
                    md["models"]["XGBoost"] = _metrics_binary(actual_ou35, xgb_ou35)
                    md["models"]["LightGBM"] = _metrics_binary(actual_ou35, lgb_ou35)
                    md["models"]["CatBoost"] = _metrics_binary(actual_ou35, cat_ou35)
                blend = self._blend_binary(ppm, w, "Over3.5")
                md["models"]["Multi-Model Blend"] = _metrics_binary(actual_ou35, blend)

            elif market_name == "BTTS":
                if include_individual:
                    md["models"]["Dixon-Coles"] = _metrics_binary(actual_btts, ppm.dc_btts)
                    md["models"]["Elo"] = _metrics_binary(actual_btts, elo_btts)
                    md["models"]["XGBoost"] = _metrics_binary(actual_btts, xgb_btts)
                    md["models"]["LightGBM"] = _metrics_binary(actual_btts, lgb_btts)
                    md["models"]["CatBoost"] = _metrics_binary(actual_btts, cat_btts)
                blend = self._blend_binary(ppm, w, "BTTS")
                md["models"]["Multi-Model Blend"] = _metrics_binary(actual_btts, blend)

            results["markets"][market_name] = md

        return results

    # ═══════════════════════════════════════════════════════
    #  Save / Load (pipeline persistence)
    # ═══════════════════════════════════════════════════════

    def save(self, path: str | Path) -> str:
        """Persist the blend (models + weights + config) to disk via joblib.

        Parameters
        ----------
        path : str or Path
            Output path (e.g. ``models/three_model_blend.joblib``).

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        import joblib

        payload = {
            "dc": self.dc,
            "elo": self.elo,
            "xgb": self.xgb,
            "lgb": self.lgb,
            "cat": self.cat,
            "weights": self.weights,
            "cond_rates": self.cond_rates,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, p)
        logger.info("ThreeModelBlend saved to %s", p)
        return str(p.absolute())

    @classmethod
    def load(cls, path: str | Path, historical_df: pd.DataFrame | None = None) -> "ThreeModelBlend":
        """Load a persisted blend from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved blend payload.
        historical_df : pd.DataFrame, optional
            Historical data for feature building.  Required for prediction.

        Returns
        -------
        ThreeModelBlend
            Reconstructed blend with all models and weights restored.
        """
        import joblib

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ThreeModelBlend not found: {p}")

        payload = joblib.load(p)
        # Backward compat: old payloads used "poisson" key
        dc = payload.get("dc", payload.get("poisson", None))
        blend = cls(
            dc_model=dc,
            elo_model=payload.get("elo"),
            xgb_model=payload.get("xgb"),
            lgb_model=payload.get("lgb"),
            cat_model=payload.get("cat"),
            weights=payload.get("weights"),
            conditional_rates=payload.get("cond_rates"),
            historical_df=historical_df,
        )
        logger.info(
            "ThreeModelBlend loaded from %s (%d markets)",
            p, len(blend.available_markets),
        )

        # Auto-load calibrator if available
        calibrator_path = p.parent / "blend_calibrator_hybrid.joblib"
        if not calibrator_path.exists():
            calibrator_path = p.parent / "blend_calibrator_platt.joblib"
        if calibrator_path.exists():
            try:
                blend._calibrator = joblib.load(calibrator_path)
                logger.info("  Calibrator auto-loaded from %s", calibrator_path.name)
            except Exception as exc:
                logger.warning("  Calibrator load failed: %s", exc)

        return blend

    # ═══════════════════════════════════════════════════════
    #  Calibrated prediction (HybridTail via src.calibration)
    # ═══════════════════════════════════════════════════════

    def predict_with_calibrated_probs(
        self,
        home_team: str,
        away_team: str,
        cal_probs: np.ndarray | None = None,
        method: str = "hybrid",
    ) -> dict[str, Any]:
        """Predict all markets, optionally overriding 1X2 with calibrated probs.

        ``cal_probs`` must be computed by the caller (e.g. via
        ``CalibratedModel``).  When provided, the blend's raw 1X2
        prediction is replaced with these calibrated probabilities,
        while Over/Under and BTTS continue to use the blend.

        Parameters
        ----------
        home_team : str
        away_team : str
        cal_probs : np.ndarray, optional
            Pre-calibrated ``[away, draw, home]`` probabilities.
        method : str
            Calibration method label for metadata (default ``"hybrid"``).

        Returns
        -------
        dict[str, Any]
            Same structure as ``predict()`` with ``calibration_method``
            set when ``cal_probs`` is provided.
        """
        result = self.predict(home_team, away_team)

        if cal_probs is not None and len(cal_probs) == 3:
            result["1x2"] = {
                "A": float(cal_probs[0]),
                "D": float(cal_probs[1]),
                "H": float(cal_probs[2]),
            }
            result["calibration_method"] = method

        return result

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
            for mn in ["Dixon-Coles", "Elo", "XGBoost", "LightGBM", "CatBoost", "Current Ensemble", "Multi-Model Blend"]:
                m = md.get("models", {}).get(mn)
                if m:
                    lines.append(f"| {mn} | {m.get('brier_score', 'N/A'):.4f} | {m.get('log_loss', 'N/A'):.4f} | {m.get('accuracy', 'N/A'):.2%} | {m.get('n', 'N/A')} |")
            lines.append("")

        # Weight recommendations
        lines.append("## Optimal Weight Recommendations")
        lines.append("")
        lines.append("| Market | DC | Elo | XGB | LGB | Cat |")
        lines.append("|--------|----|-----|-----|-----|-----|")
        for mkt in ["1X2", "Over2.5", "BTTS", "Over3.5"]:
            w = self.weights.get(mkt, {})
            lines.append(f"| {mkt} | {w.get('dc', 0):.2f} | {w.get('elo', 0):.2f} | {w.get('xgb', 0):.2f} | {w.get('lgb', 0):.2f} | {w.get('cat', 0):.2f} |")
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
            for mn in ["Dixon-Coles", "Elo", "XGBoost", "LightGBM", "CatBoost", "Current Ensemble"]:
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
        lines.append(f"| 1X2 Weight | DC+Elo dominate, tree models small | DC({w1.get('dc',0):.2f}) E({w1.get('elo',0):.2f}) X({w1.get('xgb',0):.2f}) L({w1.get('lgb',0):.2f}) C({w1.get('cat',0):.2f}) |")
        lines.append(f"| Over/Under Weight | DC+tree models dominate | DC({w2.get('dc',0):.2f}) X({w2.get('xgb',0):.2f}) L({w2.get('lgb',0):.2f}) C({w2.get('cat',0):.2f}) |")
        lines.append(f"| BTTS Weight | Tree models dominate, DC+Elo support | DC({w3.get('dc',0):.2f}) E({w3.get('elo',0):.2f}) X({w3.get('xgb',0):.2f}) L({w3.get('lgb',0):.2f}) C({w3.get('cat',0):.2f}) |")

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


def _tree_batch_predict(model: Any, xgb_list: list[Any], btts_list: list[float], n: int, X: pd.DataFrame | None, cr: ConditionalRates, blend: ThreeModelBlend) -> None:
    if model is None:
        xgb_list.extend([[0.33, 0.34, 0.33]] * n)
        btts_list.extend([0.5] * n)
        return
    try:
        if X is not None and len(X) > 0:
            X_aligned = blend._align_features(X.copy(), model)
            if X_aligned is not None and len(X_aligned) > 0:
                raw = model.predict_proba(X_aligned)
                for i in range(len(X_aligned)):
                    probs = raw[i]
                    xgb_list.append(probs)
                    exp_total = (
                        probs[2] * cr.mean_total_given_home_win
                        + probs[1] * cr.mean_total_given_draw
                        + probs[0] * cr.mean_total_given_away_win
                    )
                    if exp_total > 0:
                        exp_h = exp_total * 0.55
                        exp_a = exp_total * 0.45
                        p_h0 = np.exp(-exp_h)
                        p_a0 = np.exp(-exp_a)
                        btts_val = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)
                        btts_list.append(float(np.clip(btts_val, 0.0, 1.0)))
                    else:
                        btts_list.append(0.5)
            else:
                xgb_list.extend([[0.33, 0.34, 0.33]] * n)
                btts_list.extend([0.5] * n)
        else:
            xgb_list.extend([[0.33, 0.34, 0.33]] * n)
            btts_list.extend([0.5] * n)
    except Exception as exc:
        logger.warning("%s batch prediction failed: %s", type(model).__name__, exc)
        xgb_list.extend([[0.33, 0.34, 0.33]] * n)
        btts_list.extend([0.5] * n)


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
        key = str(sorted(candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    # Ensure defaults are included (subset to active models)
    for mkt_default in ["1X2", "Over2.5", "BTTS"]:
        default = DEFAULT_WEIGHTS.get(mkt_default, {})
        default_subset = {m: default.get(m, 1.0 / len(models)) for m in models if m in default}
        if default_subset and all(m in default_subset for m in models):
            key = str(sorted(default_subset.items()))
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
