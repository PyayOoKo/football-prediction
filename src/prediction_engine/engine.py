"""PredictionEngine — unified prediction engine for match outcomes and bets."""

from __future__ import annotations

import hashlib
import json
import logging
import random as rnd
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.prediction_engine.features import FeatureBuilder
from src.prediction_engine.loader import ModelLoader
from src.prediction_engine.models import BetRecommendation, PredictionResult

logger = logging.getLogger(__name__)


class PredictionEngine:
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
        self.model_metadata: dict[str, Any] = {
            "model_type": "none",
            "name": "none",
            "loaded": False,
        }

        self.min_confidence = min_confidence
        self.min_ev = min_ev
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll

        self._feature_builder = FeatureBuilder()
        self._blend_model: Any = None
        self._blend_loaded: bool = False

        if model_path is not None:
            self.load_model(str(model_path))
        else:
            self.load_model()

        if use_blend:
            self.load_three_model_blend(config=blend_config)

    # ── Model Loading ──────────────────────────────────────

    def load_model(self, path: str | None = None) -> bool:
        self.model, self.model_metadata = ModelLoader.load(path)
        loaded = self.model_metadata.get("loaded", False)
        if loaded:
            logger.info(
                "PredictionEngine ready: %s (type=%s)",
                self.model_metadata["name"],
                self.model_metadata["model_type"],
            )
        else:
            logger.warning(
                "PredictionEngine: no model loaded — predictions will use fallback"
            )
        return bool(loaded)

    def load_three_model_blend(
        self,
        config: Any | None = None,
    ) -> bool:
        if config is None:
            from src.config import config as _cfg

            config = _cfg

        blend_cfg = config.blend

        if not blend_cfg.enabled:
            logger.info("3-model blend disabled via config.blend.enabled")
            self._blend_loaded = False
            self._blend_model = None
            return False

        try:
            import joblib

            from src.dixon_coles import DixonColesModel
            from src.elo import EloSystem
            from src.models.three_model_blend import (
                DEFAULT_WEIGHTS,
                ConditionalRates,
                ThreeModelBlend,
            )

            historical = self._feature_builder.load_historical_data()
            if historical is None or historical.empty:
                logger.warning("No historical data for 3-model blend — blend disabled")
                self._blend_loaded = False
                return False

            logger.info("Fitting Dixon-Coles model for 3-model blend...")
            dc = DixonColesModel(decay_halflife_days=1460)
            dc.fit(historical)

            logger.info("Fitting Elo system for 3-model blend...")
            elo = EloSystem()
            elo.process_matches(historical)

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
                    "No XGBoost model found for 3-model blend — blend will use DC + Elo only"
                )

            cond_rates = ConditionalRates.from_data(historical)

            weights = None
            if blend_cfg.weights_path:
                w_path = Path(blend_cfg.weights_path)
                if w_path.exists():
                    try:
                        with open(w_path) as f:
                            w_data = json.load(f)
                        weights = w_data.get("weights", None)
                        logger.info(
                            "Loaded optimised blend weights from %s", w_path.name
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to load blend weights from %s: %s", w_path, exc
                        )

            if weights is None:
                weights = dict(DEFAULT_WEIGHTS)
                logger.info("Using default blend weights")

            self._blend_model = ThreeModelBlend(
                dc_model=dc,
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
                f"DC={'yes'}, Elo={'yes'}, XGBoost={'yes' if xgb else 'no'}",
            )
            return True

        except Exception as exc:
            logger.warning("Failed to load 3-model blend: %s — blend disabled", exc)
            self._blend_loaded = False
            self._blend_model = None
            return False

    @property
    def model_loaded(self) -> bool:
        return self.model_metadata.get("loaded", False) and self.model is not None

    @property
    def model_name(self) -> str:
        return str(self.model_metadata.get("name", "none"))

    @property
    def model_type(self) -> str:
        return str(self.model_metadata.get("model_type", "none"))

    @property
    def supports_predict_proba(self) -> bool:
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
        return self._blend_loaded and self._blend_model is not None

    # ── Market-Specific Blend Predictions ──────────────────

    def predict_over_under(
        self,
        home_team: str,
        away_team: str,
        threshold: float = 2.5,
    ) -> dict[str, float] | None:
        if not self._blend_loaded or self._blend_model is None:
            return None
        try:
            return cast(
                "dict[str, float] | None",
                self._blend_model.predict_over_under(home_team, away_team, threshold),
            )
        except Exception as exc:
            logger.debug("Blend OU prediction failed: %s", exc)
            return None

    def predict_btts(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, float] | None:
        if not self._blend_loaded or self._blend_model is None:
            return None
        try:
            return cast(
                "dict[str, float] | None",
                self._blend_model.predict_btts(home_team, away_team),
            )
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
        start = time.perf_counter()

        if self.model_loaded:
            try:
                fixture = [
                    {
                        "home_team": home_team,
                        "away_team": away_team,
                        "match_date": match_date or "",
                    }
                ]
                X = self._feature_builder.build_features(fixture)
                if X is not None and len(X) > 0:
                    probs = self._predict_with_model(X)
                    if probs is not None:
                        return self._normalise_probs(probs, home_team, away_team, start)
            except Exception as exc:
                logger.debug("Feature prediction failed: %s", exc)

        if self.model_loaded:
            try:
                probs = self._predict_direct(home_team, away_team)
                if probs is not None:
                    return self._normalise_probs(probs, home_team, away_team, start)
            except Exception as exc:
                logger.debug("Direct prediction failed: %s", exc)

        if use_fallback:
            probs = self._fallback_prediction(home_team, away_team)
            return self._normalise_probs(probs, home_team, away_team, start)

        return {"away_win": 0.0, "draw": 0.0, "home_win": 0.0}

    def predict(
        self, home_team: str, away_team: str, match_date: str | None = None
    ) -> str:
        probs = self.predict_proba(home_team, away_team, match_date)
        outcomes = ["Away Win", "Draw", "Home Win"]
        return outcomes[
            np.argmax([probs["away_win"], probs["draw"], probs["home_win"]])
        ]

    # ── Batch Prediction ───────────────────────────────────

    def predict_matches(
        self,
        fixtures: list[dict[str, Any]],
        use_fallback: bool = True,
        include_blend_markets: bool = True,
    ) -> list[PredictionResult]:
        results: list[PredictionResult] = []
        X_batch = None
        if self.model_loaded:
            try:
                X_batch = self._feature_builder.build_features(fixtures)
            except Exception as exc:
                logger.warning(
                    "Batch feature engineering failed for %d fixtures: %s",
                    len(fixtures),
                    exc,
                )

        if X_batch is not None and len(X_batch) > 0:
            for i, fixture in enumerate(fixtures):
                start = time.perf_counter()
                try:
                    row = X_batch.iloc[i : i + 1]
                    probs_arr = self._predict_with_model(row)
                    if probs_arr is not None:
                        result = self._make_result(probs_arr, fixture, start)
                        results.append(result)
                        continue
                except Exception as exc:
                    logger.warning(
                        "Batch prediction failed for %s vs %s (idx=%d): %s",
                        fixture.get("home_team", "?"),
                        fixture.get("away_team", "?"),
                        i,
                        exc,
                    )
                if use_fallback:
                    start = time.perf_counter()
                    probs_arr = self._fallback_prediction(
                        fixture.get("home_team", ""),
                        fixture.get("away_team", ""),
                    )
                    results.append(self._make_result(probs_arr, fixture, start))
                else:
                    start = time.perf_counter()
                    results.append(
                        self._make_result([0.33, 0.34, 0.33], fixture, start)
                    )
        else:
            for fixture in fixtures:
                start = time.perf_counter()
                probs = self.predict_proba(
                    fixture.get("home_team", ""),
                    fixture.get("away_team", ""),
                    fixture.get("match_date"),
                    use_fallback=use_fallback,
                )
                probs_list = [probs["away_win"], probs["draw"], probs["home_win"]]
                results.append(
                    self._make_result(probs_list, fixture, time.perf_counter() - start)
                )

        if (
            include_blend_markets
            and self._blend_loaded
            and self._blend_model is not None
        ):
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
                        "Blend enrichment failed for %s vs %s: %s", ht, at, exc
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
        kelly_frac = (
            kelly_fraction if kelly_fraction is not None else self.kelly_fraction
        )
        min_ev_val = min_ev if min_ev is not None else self.min_ev
        min_conf = min_confidence if min_confidence is not None else self.min_confidence

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

                ev = (model_prob * decimal_odds) - 1
                kelly_raw = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
                kelly = max(0.0, min(kelly_raw * kelly_frac, 1.0))
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
                        ev > min_ev_val and pred.confidence >= min_conf and kelly > 0
                    ),
                )
                recommendations.append(rec)

        recommendations.sort(key=lambda r: r.expected_value, reverse=True)
        return recommendations

    def get_best_bet(
        self,
        fixtures_with_odds: list[dict[str, Any]],
    ) -> BetRecommendation | None:
        recs = self.get_bet_recommendations(fixtures_with_odds)
        recommended = [r for r in recs if r.recommended]
        if recommended:
            return recommended[0]
        return recs[0] if recs else None

    # ── Internal Prediction Methods ────────────────────────

    def _predict_with_model(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.model is None:
            return None

        mtype = self.model_type

        try:
            if mtype == "phase4":
                probs = self.model.predict_proba(X)
                return self._align_proba_order(probs)

            elif mtype == "ensemble_model" or mtype == "weighted_ensemble":
                probs = self.model.predict_proba(X)
                return np.asarray(probs, dtype=np.float64)

            elif mtype == "phase3":
                return None

            else:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)
                    return self._align_proba_order(probs)
                return None

        except Exception as exc:
            logger.debug("Model prediction failed: %s", exc)
            return None

    def _predict_direct(self, home_team: str, away_team: str) -> np.ndarray | None:
        if self.model is None:
            return None

        if self.model_type != "phase3":
            return None

        try:
            df = pd.DataFrame(
                [
                    {
                        "home_team": home_team,
                        "away_team": away_team,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                    }
                ]
            )
            result = self.model.predict_matches(df)
            if result is not None and not result.empty:
                row = result.iloc[0]
                return np.array(
                    [
                        [
                            float(row.get("away_win_prob", 0.33)),
                            float(row.get("draw_prob", 0.34)),
                            float(row.get("home_win_prob", 0.33)),
                        ],
                    ]
                )
        except Exception as exc:
            logger.debug("Direct prediction failed: %s", exc)

        return None

    @staticmethod
    def _align_proba_order(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if probs.shape[1] != 3:
            padded = np.zeros((probs.shape[0], 3))
            n = min(probs.shape[1], 3)
            padded[:, :n] = probs[:, :n]
            return padded
        return probs

    @staticmethod
    def _fallback_prediction(home_team: str, away_team: str) -> np.ndarray:
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
        probs: np.ndarray,
        home_team: str,
        away_team: str,
        start: float,
    ) -> dict[str, float]:
        arr = np.asarray(probs).flatten()
        if arr.shape[0] != 3:
            arr = np.array([0.33, 0.34, 0.33])
        total = arr.sum()
        arr = np.array([0.33, 0.34, 0.33]) if total <= 0 else arr / total

        return {
            "away_win": float(arr[0]),
            "draw": float(arr[1]),
            "home_win": float(arr[2]),
        }

    def _make_result(
        self,
        probs: np.ndarray | list[Any],
        fixture: dict[str, Any],
        start_time: float,
    ) -> PredictionResult:
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
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "supports_proba": self.supports_predict_proba,
            "blend_loaded": self.blend_loaded,
            "blend_markets": list(self._blend_model.available_markets)
            if self._blend_model
            else [],
            "kelly_fraction": self.kelly_fraction,
            "min_confidence": self.min_confidence,
            "min_ev": self.min_ev,
        }

    def summary(self) -> str:
        blend_market_count = (
            len(self._blend_model.available_markets) if self._blend_model else 0
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
