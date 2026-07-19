"""
PredictionEngine — Unified, reusable prediction library.

Provides a single interface for loading any model type, generating match
outcome predictions, and producing bet recommendations, regardless of
the underlying model implementation (Phase 4 sklearn, Phase 3 statistical,
EnsembleModel, WeightedEnsemble, or arbitrary pickle/joblib).

Usage
-----
::

    from src.prediction_engine import PredictionEngine

    engine = PredictionEngine()
    engine.load_model()                          # auto-detect best model
    # engine.load_model("models/xgboost.pkl")    # or explicit path

    # Predict
    proba = engine.predict_proba("Brazil", "Argentina")
    # → {"away_win": 0.25, "draw": 0.20, "home_win": 0.55}

    # Bet recommendations
    bets = engine.get_bet_recommendations([
        {"home_team": "Brazil", "away_team": "Argentina",
         "home_odds": 1.8, "draw_odds": 3.5, "away_odds": 4.5},
    ])
    # → [{"fixture": ..., "ev": 0.12, "kelly_fraction": 0.05, ...}]
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════


@dataclass
class PredictionResult:
    """Outcome of a single match prediction.

    Attributes
    ----------
    home_team : str
    away_team : str
    prob_home_win : float
        Probability of a home win (0.0–1.0).
    prob_draw : float
        Probability of a draw (0.0–1.0).
    prob_away_win : float
        Probability of an away win (0.0–1.0).
    predicted_outcome : str
        ``"Home Win"``, ``"Draw"``, or ``"Away Win"``.
    confidence : float
        Highest probability (max of the three).
    model_name : str
        Name of the model used.
    processing_time_ms : float
        Time taken to generate this prediction.
    over_2_5_prob : float | None
        Probability of Over 2.5 goals (from 3-model blend, if enabled).
    under_2_5_prob : float | None
        Probability of Under 2.5 goals.
    over_3_5_prob : float | None
        Probability of Over 3.5 goals.
    under_3_5_prob : float | None
        Probability of Under 3.5 goals.
    btts_prob : float | None
        Probability of Both Teams To Score (from 3-model blend, if enabled).
    btts_no_prob : float | None
        Probability of no BTTS.
    metadata : dict
        Additional info (feature count, calibration status, etc.).
    """

    home_team: str = ""
    away_team: str = ""
    prob_home_win: float = 0.0
    prob_draw: float = 0.0
    prob_away_win: float = 0.0
    predicted_outcome: str = ""
    confidence: float = 0.0
    model_name: str = ""
    processing_time_ms: float = 0.0
    over_2_5_prob: float | None = None
    under_2_5_prob: float | None = None
    over_3_5_prob: float | None = None
    under_3_5_prob: float | None = None
    btts_prob: float | None = None
    btts_no_prob: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def probabilities(self) -> dict[str, float]:
        return {
            "home_win": self.prob_home_win,
            "draw": self.prob_draw,
            "away_win": self.prob_away_win,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prob_home_win": round(self.prob_home_win, 4),
            "prob_draw": round(self.prob_draw, 4),
            "prob_away_win": round(self.prob_away_win, 4),
            "predicted_outcome": self.predicted_outcome,
            "confidence": round(self.confidence, 4),
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "over_2_5_prob": round(self.over_2_5_prob, 4) if self.over_2_5_prob is not None else None,
            "under_2_5_prob": round(self.under_2_5_prob, 4) if self.under_2_5_prob is not None else None,
            "over_3_5_prob": round(self.over_3_5_prob, 4) if self.over_3_5_prob is not None else None,
            "under_3_5_prob": round(self.under_3_5_prob, 4) if self.under_3_5_prob is not None else None,
            "btts_prob": round(self.btts_prob, 4) if self.btts_prob is not None else None,
            "btts_no_prob": round(self.btts_no_prob, 4) if self.btts_no_prob is not None else None,
        }


@dataclass
class BetRecommendation:
    """A value-bet recommendation for a single fixture outcome.

    Attributes
    ----------
    fixture : dict
        Original fixture data.
    outcome : str
        The recommended bet outcome (``"home_win"`` / ``"draw"`` / ``"away_win"``).
    model_probability : float
        Our model's estimated probability for this outcome.
    decimal_odds : float
        Bookmaker decimal odds for this outcome.
    implied_probability : float
        Bookmaker-implied probability (1/odds).
    expected_value : float
        EV = (model_prob * odds) - 1.
    kelly_fraction : float
        Full Kelly fraction (clamped to [0, 1]).
    edge : float
        Percentage edge over bookmaker (model_prob - implied_prob).
    confidence : float
        Our model's confidence in the overall match prediction.
    recommended : bool
        Whether this bet is recommended (EV > 0 and confidence > threshold).
    """

    fixture: dict = field(default_factory=dict)
    outcome: str = ""
    model_probability: float = 0.0
    decimal_odds: float = 0.0
    implied_probability: float = 0.0
    expected_value: float = 0.0
    kelly_fraction: float = 0.0
    edge: float = 0.0
    confidence: float = 0.0
    recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture,
            "outcome": self.outcome,
            "model_probability": round(self.model_probability, 4),
            "decimal_odds": round(self.decimal_odds, 4),
            "implied_probability": round(self.implied_probability, 4),
            "expected_value": round(self.expected_value, 6),
            "kelly_fraction": round(self.kelly_fraction, 6),
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 4),
            "recommended": self.recommended,
        }


# ═══════════════════════════════════════════════════════════
#  Model Loader
# ═══════════════════════════════════════════════════════════


class ModelLoader:
    """Handles model discovery, loading, and type detection.

    Searches known model paths in priority order and can load any
    supported model type (sklearn, XGBoost, EnsembleModel,
    WeightedEnsemble, PoissonModel, etc.).
    """

    # Default search paths in priority order
    DEFAULT_SEARCH_PATHS: list[str] = [
        "models/ensemble.pkl",
        "models/ensemble_model.joblib",
        "models/weighted_ensemble.joblib",
        "models/xgboost_model.pkl",
        "models/model.pkl",
        "models/xgboost.pkl",
    ]

    @staticmethod
    def detect_model_type(model: Any) -> str:
        """Detect the type of a loaded model object.

        Returns one of ``"phase4"`` (sklearn-compatible), ``"phase3"``
        (statistical with ``predict_matches``), ``"ensemble_model"``
        (``EnsembleModel``), ``"weighted_ensemble"`` (``WeightedEnsemble``),
        or ``"unknown"``.
        """
        modname = type(model).__module__
        clsname = type(model).__name__

        if clsname == "EnsembleModel":
            return "ensemble_model"
        if clsname == "WeightedEnsemble":
            return "weighted_ensemble"
        if hasattr(model, "predict_matches"):
            return "phase3"
        if hasattr(model, "predict_proba"):
            return "phase4"
        return "unknown"

    @staticmethod
    def load(path: str | Path | None = None) -> tuple[Any, dict]:
        """Load a model from disk.

        Parameters
        ----------
        path : str or Path, optional
            Explicit path to a model file. If ``None``, searches
            ``DEFAULT_SEARCH_PATHS`` in order and returns the first hit.

        Returns
        -------
        (model, metadata)
            model : Any — the loaded model object (or ``None``).
            metadata : dict — ``{"path", "model_type", "name", "loaded"}``.
        """
        import joblib

        metadata: dict[str, Any] = {
            "path": "",
            "model_type": "none",
            "name": "none",
            "loaded": False,
        }

        # Collect candidate paths
        candidates: list[Path] = []
        if path is not None:
            candidates = [Path(path)]
        else:
            candidates = [Path(p) for p in ModelLoader.DEFAULT_SEARCH_PATHS]

        for candidate in candidates:
            if not candidate.exists():
                logger.debug("Model not found at: %s", candidate)
                continue

            try:
                model = joblib.load(candidate)
                mtype = ModelLoader.detect_model_type(model)

                metadata = {
                    "path": str(candidate.resolve()),
                    "model_type": mtype,
                    "name": candidate.stem,
                    "loaded": True,
                }

                logger.info(
                    "Loaded model: %s (type=%s)", candidate.name, mtype,
                )
                return model, metadata

            except Exception as exc:
                logger.warning("Failed to load %s: %s", candidate, exc)
                continue

        logger.warning("No model found at any search path")
        return None, metadata

    @staticmethod
    def get_model_name(model: Any) -> str:
        """Get a human-readable name for any model object."""
        if hasattr(model, "model_name"):
            return model.model_name
        if hasattr(model, "name"):
            return model.name
        return type(model).__name__


# ═══════════════════════════════════════════════════════════
#  Feature Builder
# ═══════════════════════════════════════════════════════════


class FeatureBuilder:
    """Constructs feature vectors for fixture(s) using the project's
    ``src.feature_engineering`` pipeline, with graceful fallbacks."""

    def __init__(self) -> None:
        self._historical_data: pd.DataFrame | None = None
        self._feature_cols: list[str] = []

    def load_historical_data(self) -> pd.DataFrame | None:
        """Load historical match data for feature engineering."""
        if self._historical_data is not None:
            return self._historical_data

        # Try multiple sources
        from src.data_loader import load_clean_data

        df = load_clean_data()
        if df is not None and not df.empty:
            self._historical_data = df
            return df

        # Fallback: try processed CSV
        processed = Path("data/processed/results_clean.csv")
        if processed.exists():
            df = pd.read_csv(processed, low_memory=False)
            if not df.empty:
                self._historical_data = df
                return df

        # Fallback: try raw worldcup data
        raw = Path("data/raw/worldcup_all.csv")
        if raw.exists():
            df = pd.read_csv(raw, low_memory=False)
            if not df.empty:
                self._historical_data = df
                return df

        return None

    def build_features(
        self,
        fixtures: list[dict[str, Any]],
    ) -> pd.DataFrame | None:
        """Build feature matrix for a list of fixtures.

        Parameters
        ----------
        fixtures : list[dict]
            Each dict must have ``home_team`` and ``away_team`` keys.
            Optionally ``match_date``.

        Returns
        -------
        pd.DataFrame or None
            Feature matrix, or ``None`` if feature engineering fails.
        """
        historical = self.load_historical_data()
        if historical is None:
            logger.warning("No historical data for feature engineering")
            return None

        try:
            from src.feature_engineering import build_features

            fixture_rows = []
            for fix in fixtures:
                row = {
                    "date": pd.Timestamp(fix.get("match_date", datetime.now().strftime("%Y-%m-%d"))),
                    "home_team": fix["home_team"],
                    "away_team": fix["away_team"],
                    "result": "H",
                    "home_goals": 0,
                    "away_goals": 0,
                }
                # Fill in missing columns from historical
                for col in historical.columns:
                    if col not in row:
                        row[col] = historical[col].iloc[-1] if len(historical) > 0 else 0
                fixture_rows.append(row)

            df_ext = pd.concat(
                [historical, pd.DataFrame(fixture_rows)],
                ignore_index=True,
            )
            X_full, _ = build_features(df_ext, is_training=False)
            n_hist = len(historical)
            X_fixtures = X_full.iloc[n_hist:]
            self._feature_cols = list(X_full.columns)
            return X_fixtures

        except Exception as exc:
            logger.warning("Feature engineering failed: %s", exc)
            return None


# ═══════════════════════════════════════════════════════════
#  Prediction Engine
# ═══════════════════════════════════════════════════════════


class PredictionEngine:
    """Unified prediction engine — load models, predict matches, get bets.

    Parameters
    ----------
    model_path : str or Path, optional
        Explicit path to a model file. Auto-detects if not provided.
    min_confidence : float
        Minimum confidence threshold for bet recommendations (default 0.35).
    min_ev : float
        Minimum expected value for bet recommendations (default 0.0).
    kelly_fraction : float
        Fraction of full Kelly to use (default 0.25).
    bankroll : float
        Default bankroll for stake calculations (default 1000.0).
    use_blend : bool
        Whether to load and use the 3-model blend for Over/Under and BTTS
        markets (default True). When enabled, Over2.5, Over3.5, BTTS
        probabilities come from the blend while 1X2 continues to use the
        current ensemble model.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_confidence: float = 0.35,
        min_ev: float = 0.0,
        kelly_fraction: float = 0.25,
        bankroll: float = 1000.0,
        use_blend: bool = True,
        blend_config: Any | None = None,
    ) -> None:
        self.model: Any = None
        self.model_metadata: dict = {"model_type": "none", "name": "none", "loaded": False}

        self.min_confidence = min_confidence
        self.min_ev = min_ev
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll

        self._feature_builder = FeatureBuilder()
        self._blend_model: Any = None
        self._blend_loaded: bool = False

        # Load model immediately if path given, or auto-detect
        if model_path is not None:
            self.load_model(str(model_path))
        else:
            self.load_model()  # auto-detect

        # Optionally load the 3-model blend for secondary markets
        if use_blend:
            self.load_three_model_blend(config=blend_config)

    # ── Model Loading ──────────────────────────────────────

    def load_model(self, path: str | None = None) -> bool:
        """Load a prediction model from disk.

        Parameters
        ----------
        path : str, optional
            Path to model file. Auto-detects if ``None``.

        Returns
        -------
        bool
            ``True`` if a model was successfully loaded.
        """
        self.model, self.model_metadata = ModelLoader.load(path)
        loaded = self.model_metadata.get("loaded", False)
        if loaded:
            logger.info(
                "PredictionEngine ready: %s (type=%s)",
                self.model_metadata["name"],
                self.model_metadata["model_type"],
            )
        else:
            logger.warning("PredictionEngine: no model loaded — predictions will use fallback")
        return loaded

    def load_three_model_blend(
        self,
        config: Any | None = None,
    ) -> bool:
        """Load the 3-model blend for Over/Under and BTTS market predictions.

        The blend combines Poisson + Elo + XGBoost with market-specific
        optimised weights. Falls back gracefully if any component is
        unavailable.

        Parameters
        ----------
        config : Config, optional
            Config instance with blend settings. Uses global config if
            ``None``.

        Returns
        -------
        bool
            ``True`` if the blend was loaded successfully.
        """
        if config is None:
            from config import config as _cfg
            config = _cfg

        blend_cfg = config.blend

        if not blend_cfg.enabled:
            logger.info("3-model blend disabled via config.blend.enabled")
            self._blend_loaded = False
            self._blend_model = None
            return False

        try:
            from src.models.three_model_blend import (
                ConditionalRates, ThreeModelBlend, DEFAULT_WEIGHTS,
            )
            from src.poisson_model import PoissonModel
            from src.elo import EloSystem
            import joblib

            # Load historical data for conditional rates and feature building
            historical = self._feature_builder.load_historical_data()
            if historical is None or historical.empty:
                logger.warning(
                    "No historical data for 3-model blend — blend disabled"
                )
                self._blend_loaded = False
                return False

            # Fit Poisson model
            logger.info("Fitting Poisson model for 3-model blend...")
            poisson = PoissonModel()
            poisson.fit(historical)

            # Fit Elo system
            logger.info("Fitting Elo system for 3-model blend...")
            elo = EloSystem()
            elo.process_matches(historical)

            # Load XGBoost model (try multiple paths)
            xgb = None
            for candidate in [
                config.paths.models / "xgboost_model.joblib",
                config.paths.models / "worldcup_lightgbm.joblib",
                Path("models/xgboost_model.joblib"),
                Path("models/worldcup_lightgbm.joblib"),
            ]:
                if candidate.exists():
                    xgb = joblib.load(candidate)
                    logger.info("Loaded XGBoost model: %s", candidate.name)
                    break

            if xgb is None:
                logger.warning(
                    "No XGBoost model found for 3-model blend — "
                    "blend will use Poisson + Elo only"
                )

            # Compute conditional rates from historical data
            cond_rates = ConditionalRates.from_data(historical)

            # Load optimised weights from config file if available
            weights = None
            if blend_cfg.weights_path:
                w_path = Path(blend_cfg.weights_path)
                if w_path.exists():
                    try:
                        with open(w_path) as f:
                            w_data = json.load(f)
                        weights = w_data.get("weights", None)
                        logger.info(
                            "Loaded optimised blend weights from %s",
                            w_path.name,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to load blend weights from %s: %s",
                            w_path, exc,
                        )

            if weights is None:
                weights = dict(DEFAULT_WEIGHTS)
                logger.info("Using default blend weights")            # 1X2 routing: when use_blend_for_1x2=False (default),
            # the blend is still loaded but predict_matches() only
            # uses it for binary markets (Over/Under, BTTS).
            # 1X2 predictions continue to use the current ensemble model.
            self._blend_model = ThreeModelBlend(
                poisson_model=poisson,
                elo_model=elo,
                xgb_model=xgb,
                weights=weights,
                conditional_rates=cond_rates,
                historical_df=historical,
            )
            self._blend_loaded = True
            logger.info(
                "3-model blend loaded: %d markets, %s",
                len(self._blend_model.available_markets),
                f"Poisson={'yes'}, Elo={'yes'}, XGBoost={'yes' if xgb else 'no'}",
            )
            return True

        except Exception as exc:
            logger.warning(
                "Failed to load 3-model blend: %s — blend disabled", exc,
            )
            self._blend_loaded = False
            self._blend_model = None
            return False

    @property
    def model_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        return self.model_metadata.get("loaded", False) and self.model is not None

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return self.model_metadata.get("name", "none")

    @property
    def model_type(self) -> str:
        """Detected model type string."""
        return self.model_metadata.get("model_type", "none")

    @property
    def supports_predict_proba(self) -> bool:
        """Whether the loaded model supports probability predictions."""
        if self.model is None:
            return False
        mtype = self.model_type
        if mtype in ("phase4", "ensemble_model", "weighted_ensemble"):
            return True
        if mtype == "phase3":
            return hasattr(self.model, "predict_matches")
        return hasattr(self.model, "predict_proba") or hasattr(self.model, "predict")

    @property
    def blend_loaded(self) -> bool:
        """Whether the 3-model blend is loaded."""
        return self._blend_loaded and self._blend_model is not None

    # ── Market-Specific Blend Predictions ──────────────────

    def predict_over_under(
        self,
        home_team: str,
        away_team: str,
        threshold: float = 2.5,
    ) -> dict[str, float] | None:
        """Predict Over/Under for a goal threshold using the 3-model blend.

        Parameters
        ----------
        home_team : str
        away_team : str
        threshold : float
            Goal threshold (2.5, 3.5, etc.). Default 2.5.

        Returns
        -------
        dict or None
            ``{"Over": prob, "Under": prob}`` or ``None`` if blend not loaded.
        """
        if not self._blend_loaded or self._blend_model is None:
            return None
        try:
            return self._blend_model.predict_over_under(home_team, away_team, threshold)
        except Exception as exc:
            logger.debug("Blend OU prediction failed: %s", exc)
            return None

    def predict_btts(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float] | None:
        """Predict Both Teams To Score probability using the 3-model blend.

        Parameters
        ----------
        home_team : str
        away_team : str

        Returns
        -------
        dict or None
            ``{"BTTS": prob, "No BTTS": prob}`` or ``None`` if blend not loaded.
        """
        if not self._blend_loaded or self._blend_model is None:
            return None
        try:
            return self._blend_model.predict_btts(home_team, away_team)
        except Exception as exc:
            logger.debug("Blend BTTS prediction failed: %s", exc)
            return None

    # ── Single Match Prediction ────────────────────────────

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        match_date: str | None = None,
        use_fallback: bool = True,
    ) -> dict[str, float]:
        """Predict match outcome probabilities.

        Parameters
        ----------
        home_team : str
        away_team : str
        match_date : str, optional
            Date in ``YYYY-MM-DD`` format.
        use_fallback : bool
            Whether to use the deterministic fallback if the model fails
            (default ``True``).

        Returns
        -------
        dict
            ``{"away_win": float, "draw": float, "home_win": float}``.
        """
        start = time.perf_counter()

        # Try feature-based prediction
        if self.model_loaded:
            try:
                fixture = [{"home_team": home_team, "away_team": away_team, "match_date": match_date or ""}]
                X = self._feature_builder.build_features(fixture)
                if X is not None and len(X) > 0:
                    probs = self._predict_with_model(X)
                    if probs is not None:
                        return self._normalise_probs(probs, home_team, away_team, start)
            except Exception as exc:
                logger.debug("Feature prediction failed: %s", exc)

        # Try model-only prediction (no features — for phase3 models)
        if self.model_loaded:
            try:
                probs = self._predict_direct(home_team, away_team)
                if probs is not None:
                    return self._normalise_probs(probs, home_team, away_team, start)
            except Exception as exc:
                logger.debug("Direct prediction failed: %s", exc)

        # Fallback
        if use_fallback:
            probs = self._fallback_prediction(home_team, away_team)
            return self._normalise_probs(probs, home_team, away_team, start)

        return {"away_win": 0.0, "draw": 0.0, "home_win": 0.0}

    def predict(self, home_team: str, away_team: str, match_date: str | None = None) -> str:
        """Predict hard match outcome.

        Returns
        -------
        str
            ``"Home Win"``, ``"Draw"``, or ``"Away Win"``.
        """
        probs = self.predict_proba(home_team, away_team, match_date)
        outcomes = ["Away Win", "Draw", "Home Win"]
        return outcomes[np.argmax([probs["away_win"], probs["draw"], probs["home_win"]])]

    # ── Batch Prediction ───────────────────────────────────

    def predict_matches(
        self,
        fixtures: list[dict[str, Any]],
        use_fallback: bool = True,
        include_blend_markets: bool = True,
    ) -> list[PredictionResult]:
        """Predict outcomes for multiple fixtures.

        When the 3-model blend is loaded, also computes Over/Under and
        BTTS probabilities for each fixture and attaches them to the
        ``PredictionResult``.

        Parameters
        ----------
        fixtures : list[dict]
            Each dict must contain ``home_team`` and ``away_team``.
        use_fallback : bool
            Whether to fall back to deterministic estimation.
        include_blend_markets : bool
            Whether to enrich results with Over/Under/BTTS from the
            3-model blend (default ``True``).

        Returns
        -------
        list[PredictionResult]
        """
        results: list[PredictionResult] = []
        X_batch = None
        if self.model_loaded:
            try:
                X_batch = self._feature_builder.build_features(fixtures)
            except Exception as exc:
                logger.warning(
                    "Batch feature engineering failed for %d fixtures: %s",
                    len(fixtures), exc,
                )

        if X_batch is not None and len(X_batch) > 0:
            # Batch prediction with features
            for i, fixture in enumerate(fixtures):
                start = time.perf_counter()
                try:
                    row = X_batch.iloc[i:i+1]
                    probs = self._predict_with_model(row)
                    if probs is not None:
                        result = self._make_result(probs, fixture, start)
                        results.append(result)
                        continue
                except Exception as exc:
                    logger.warning(
                        "Batch prediction failed for %s vs %s (idx=%d): %s",
                        fixture.get("home_team", "?"),
                        fixture.get("away_team", "?"),
                        i, exc,
                    )
                # Fallback per fixture
                if use_fallback:
                    start = time.perf_counter()
                    probs = self._fallback_prediction(
                        fixture.get("home_team", ""),
                        fixture.get("away_team", ""),
                    )
                    results.append(self._make_result(probs, fixture, start))
                else:
                    start = time.perf_counter()
                    results.append(self._make_result(
                        [0.33, 0.34, 0.33], fixture, start,
                    ))
        else:
            # Sequential fallback per fixture
            for fixture in fixtures:
                start = time.perf_counter()
                probs = self.predict_proba(
                    fixture.get("home_team", ""),
                    fixture.get("away_team", ""),
                    fixture.get("match_date"),
                    use_fallback=use_fallback,
                )
                probs_list = [probs["away_win"], probs["draw"], probs["home_win"]]
                results.append(self._make_result(probs_list, fixture, time.perf_counter() - start))

        # Enrich with 3-model blend markets (Over/Under, BTTS)
        if include_blend_markets and self._blend_loaded and self._blend_model is not None:
            for result in results:
                ht, at = result.home_team, result.away_team
                try:
                    ou = self._blend_model.predict_over_under(ht, at, 2.5)
                    if ou:
                        result.over_2_5_prob = ou["Over"]
                        result.under_2_5_prob = ou["Under"]

                    ou35 = self._blend_model.predict_over_under(ht, at, 3.5)
                    if ou35:
                        result.over_3_5_prob = ou35["Over"]
                        result.under_3_5_prob = ou35["Under"]

                    btts = self._blend_model.predict_btts(ht, at)
                    if btts:
                        result.btts_prob = btts["BTTS"]
                        result.btts_no_prob = btts["No BTTS"]
                except Exception as exc:
                    logger.debug(
                        "Blend enrichment failed for %s vs %s: %s", ht, at, exc,
                    )

        return results

    # ── Bet Recommendations ────────────────────────────────

    def get_bet_recommendations(
        self,
        fixtures_with_odds: list[dict[str, Any]],
        kelly_fraction: float | None = None,
        min_ev: float | None = None,
        min_confidence: float | None = None,
    ) -> list[BetRecommendation]:
        """Generate bet recommendations from fixtures with bookmaker odds.

        Each fixture dict must include keys:
        ``home_team``, ``away_team``, ``home_odds``, ``draw_odds``, ``away_odds``.

        Parameters
        ----------
        fixtures_with_odds : list[dict]
            Fixtures with odds.
        kelly_fraction : float, optional
            Override default Kelly fraction.
        min_ev : float, optional
            Override default minimum EV.
        min_confidence : float, optional
            Override default minimum confidence.

        Returns
        -------
        list[BetRecommendation]
        """
        kelly_frac = kelly_fraction if kelly_fraction is not None else self.kelly_fraction
        min_ev_val = min_ev if min_ev is not None else self.min_ev
        min_conf = min_confidence if min_confidence is not None else self.min_confidence

        # Get predictions first
        predictions = self.predict_matches(fixtures_with_odds)

        recommendations: list[BetRecommendation] = []

        for i, fixture in enumerate(fixtures_with_odds):
            pred = predictions[i] if i < len(predictions) else None
            if pred is None:
                continue

            model_probs = pred.probabilities
            odds_map = {
                "home_win": float(fixture.get("home_odds", 0)),
                "draw": float(fixture.get("draw_odds", 0)),
                "away_win": float(fixture.get("away_odds", 0)),
            }

            for outcome in ["home_win", "draw", "away_win"]:
                model_prob = model_probs.get(outcome, 0)
                decimal_odds = odds_map.get(outcome, 0)

                if decimal_odds <= 1 or model_prob <= 0:
                    continue

                # Calculate EV
                ev = (model_prob * decimal_odds) - 1

                # Calculate Kelly
                kelly_raw = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
                kelly = max(0.0, min(kelly_raw * kelly_frac, 1.0))

                # Edge
                implied_prob = 1.0 / decimal_odds
                edge = model_prob - implied_prob

                rec = BetRecommendation(
                    fixture=fixture,
                    outcome=outcome,
                    model_probability=model_prob,
                    decimal_odds=decimal_odds,
                    implied_probability=implied_prob,
                    expected_value=ev,
                    kelly_fraction=kelly,
                    edge=edge,
                    confidence=pred.confidence,
                    recommended=(
                        ev > min_ev_val
                        and pred.confidence >= min_conf
                        and kelly > 0
                    ),
                )
                recommendations.append(rec)

        # Sort by EV descending
        recommendations.sort(key=lambda r: r.expected_value, reverse=True)
        return recommendations

    def get_best_bet(
        self,
        fixtures_with_odds: list[dict[str, Any]],
    ) -> BetRecommendation | None:
        """Get the single best bet recommendation across all fixtures.

        Returns
        -------
        BetRecommendation or None
        """
        recs = self.get_bet_recommendations(fixtures_with_odds)
        recommended = [r for r in recs if r.recommended]
        if recommended:
            return recommended[0]  # Already sorted by EV descending
        return recs[0] if recs else None

    # ── Internal Prediction Methods ────────────────────────

    def _predict_with_model(self, X: pd.DataFrame) -> np.ndarray | None:
        """Predict probabilities using the loaded model on feature matrix X.

        Returns shape ``(n, 3)`` as ``[away_prob, draw_prob, home_prob]``,
        or ``None`` on failure.
        """
        if self.model is None:
            return None

        mtype = self.model_type

        try:
            if mtype == "phase4":
                probs = self.model.predict_proba(X)
                return self._align_proba_order(probs)

            elif mtype == "ensemble_model":
                probs = self.model.predict_proba(X)  # Already [away, draw, home]
                return np.asarray(probs, dtype=np.float64)

            elif mtype == "weighted_ensemble":
                probs = self.model.predict_proba(X)
                return np.asarray(probs, dtype=np.float64)

            elif mtype == "phase3":
                # phase3 models need raw data — pass through
                return None

            else:
                # Generic: try predict_proba then predict
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)
                    return self._align_proba_order(probs)
                return None

        except Exception as exc:
            logger.debug("Model prediction failed: %s", exc)
            return None

    def _predict_direct(self, home_team: str, away_team: str) -> np.ndarray | None:
        """Direct prediction for phase3 models (no feature matrix needed)."""
        if self.model is None:
            return None

        if self.model_type != "phase3":
            return None

        try:
            # Create a minimal DataFrame for predict_matches
            df = pd.DataFrame([{
                "home_team": home_team,
                "away_team": away_team,
                "date": datetime.now().strftime("%Y-%m-%d"),
            }])
            result = self.model.predict_matches(df)
            if result is not None and not result.empty:
                row = result.iloc[0]
                return np.array([
                    [float(row.get("away_win_prob", 0.33)),
                     float(row.get("draw_prob", 0.34)),
                     float(row.get("home_win_prob", 0.33))],
                ])
        except Exception as exc:
            logger.debug("Direct prediction failed: %s", exc)

        return None

    @staticmethod
    def _align_proba_order(probs: np.ndarray) -> np.ndarray:
        """Ensure probability array is in ``[away, draw, home]`` order.

        Scikit-learn models output probabilities in sorted class order.
        Since the codebase uses integer targets (0=Away, 1=Draw, 2=Home),
        sklearn's sorted classes_ will be [0, 1, 2] which matches our
        expected [away, draw, home] order automatically. No reordering needed.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if probs.shape[1] != 3:
            # Can't align — pad with zeros
            padded = np.zeros((probs.shape[0], 3))
            n = min(probs.shape[1], 3)
            padded[:, :n] = probs[:, :n]
            return padded
        return probs

    @staticmethod
    def _fallback_prediction(home_team: str, away_team: str) -> np.ndarray:
        """Deterministic fallback using team-name hashing."""
        import hashlib
        import random as rnd

        seed_str = f"{home_team}|{away_team}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        rng = rnd.Random(seed)

        home_str = rng.uniform(0.30, 0.55)
        away_str = rng.uniform(0.20, 0.45)
        draw_str = rng.uniform(0.20, 0.35)
        total = home_str + draw_str + away_str

        return np.array([[away_str / total, draw_str / total, home_str / total]])

    @staticmethod
    def _normalise_probs(
        probs: np.ndarray, home_team: str, away_team: str, start: float,
    ) -> dict[str, float]:
        """Convert probability array to named dict with renormalisation."""
        arr = np.asarray(probs).flatten()
        if arr.shape[0] != 3:
            arr = np.array([0.33, 0.34, 0.33])
        total = arr.sum()
        if total <= 0:
            arr = np.array([0.33, 0.34, 0.33])
        else:
            arr = arr / total

        return {
            "away_win": float(arr[0]),
            "draw": float(arr[1]),
            "home_win": float(arr[2]),
        }

    def _make_result(
        self, probs: np.ndarray | list, fixture: dict, start_time: float,
    ) -> PredictionResult:
        """Build a PredictionResult from a probability array and fixture data."""
        probs_arr = np.asarray(probs).flatten()
        if len(probs_arr) != 3:
            probs_arr = np.array([0.33, 0.34, 0.33])

        total = probs_arr.sum()
        if total > 0:
            probs_arr = probs_arr / total

        outcomes = ["Away Win", "Draw", "Home Win"]
        pred_idx = int(np.argmax(probs_arr))
        confidence = float(probs_arr[pred_idx])
        elapsed = (time.perf_counter() - start_time) * 1000

        homet = fixture.get("home_team", fixture.get("home", ""))
        awayt = fixture.get("away_team", fixture.get("away", ""))

        return PredictionResult(
            home_team=homet,
            away_team=awayt,
            prob_away_win=float(probs_arr[0]),
            prob_draw=float(probs_arr[1]),
            prob_home_win=float(probs_arr[2]),
            predicted_outcome=outcomes[pred_idx],
            confidence=confidence,
            model_name=self.model_name,
            processing_time_ms=elapsed,
            metadata={
                "model_type": self.model_type,
                "model_loaded": self.model_loaded,
            },
        )

    # ── Convenience Methods ────────────────────────────────

    def predict_from_csv(
        self,
        csv_path: str | Path,
        home_col: str = "home_team",
        away_col: str = "away_team",
    ) -> list[PredictionResult]:
        """Predict from a CSV file of fixtures.

        Parameters
        ----------
        csv_path : str or Path
            Path to CSV with at least home/away team columns.
        home_col : str
            Column name for home team.
        away_col : str
            Column name for away team.

        Returns
        -------
        list[PredictionResult]
        """
        df = pd.read_csv(csv_path, low_memory=False)
        if home_col not in df.columns or away_col not in df.columns:
            raise ValueError(
                f"CSV must contain '{home_col}' and '{away_col}' columns. "
                f"Found: {list(df.columns)}"
            )

        fixtures = []
        for _, row in df.iterrows():
            fix = {
                "home_team": row[home_col],
                "away_team": row[away_col],
            }
            if "date" in df.columns:
                fix["match_date"] = str(row["date"])
            fixtures.append(fix)

        return self.predict_matches(fixtures)

    def save_predictions(
        self,
        results: list[PredictionResult],
        output_path: str | Path,
    ) -> str:
        """Save predictions to CSV or JSON.

        Parameters
        ----------
        results : list[PredictionResult]
        output_path : str or Path
            ``.csv`` or ``.json`` file path.

        Returns
        -------
        str
            Path to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in results]

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            pd.DataFrame(data).to_csv(path, index=False)

        logger.info("Saved %d predictions to %s", len(results), path)
        return str(path)

    def health_check(self) -> dict[str, Any]:
        """Run a health check on the engine.

        Returns
        -------
        dict
            Status info: model loaded, model name, type, etc.
        """
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "supports_proba": self.supports_predict_proba,
            "blend_loaded": self.blend_loaded,
            "blend_markets": list(self._blend_model.available_markets) if self._blend_model else [],
            "kelly_fraction": self.kelly_fraction,
            "min_confidence": self.min_confidence,
            "min_ev": self.min_ev,
        }

    def summary(self) -> str:
        """Print a human-readable summary of the engine state."""
        blend_market_count = (
            len(self._blend_model.available_markets)
            if self._blend_model
            else 0
        )
        lines = [
            "=" * 55,
            "  PREDICTION ENGINE SUMMARY",
            "=" * 55,
            f"  Model loaded:   {'YES' if self.model_loaded else 'NO'}",
            f"  Model name:     {self.model_name}",
            f"  Model type:     {self.model_type}",
            f"  Supports proba: {self.supports_predict_proba}",
            f"  Blend loaded:   {'YES' if self.blend_loaded else 'NO'}",
            f"  Blend markets:  {blend_market_count}",
            f"  Kelly frac:     {self.kelly_fraction}",
            f"  Min confidence: {self.min_confidence}",
            f"  Min EV:         {self.min_ev}",
            "=" * 55,
        ]
        return "\n".join(lines)
