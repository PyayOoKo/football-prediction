"""
Football Prediction REST API — FastAPI application.

Provides endpoints for match outcome prediction, model management,
and health monitoring with API key authentication and rate limiting.

Usage:
    uvicorn api.main:app --reload --port 8000
    python -m api.main
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import config

from api.auth import optional_auth, rate_limiter, verify_api_key
from api.models import (
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    OutcomeProbabilities,
    PredictWithOddsRequest,
    PredictResponse,
    MatchPrediction,
)

logger = logging.getLogger(__name__)

# ── Application state ──────────────────────────────────────
class AppState:
    """Shared application state, populated at startup."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_name: str = "none"
        self.model_registry: Any = None
        self.start_time: float = time.time()
        self.feature_columns: list[str] = []


state = AppState()


# ── Lifespan handler ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Football Prediction API")
    _load_model()
    yield
    logger.info("Shutting down Football Prediction API")


# ── App creation ───────────────────────────────────────────
app = FastAPI(
    title="Football Prediction API",
    description=(
        "REST API for AI-powered football match outcome prediction. "
        "Accepts fixture data, returns probabilities, expected value, "
        "and Kelly stake recommendations."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    contact={
        "name": "Football Prediction System",
        "url": "https://github.com/football-prediction",
    },
)

# ── CORS ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID middleware ──────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next: Any) -> Any:
    """Add a unique request ID and rate limiting to every request."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Rate limiting for all requests
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client_ip):
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                detail="Rate limit exceeded. Max 100 requests per 60 seconds per IP.",
                status_code=429,
                request_id=request_id,
            ).model_dump(),
        )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handler ────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions and return a clean error response."""
    logger.error(
        "Unhandled %s: %s [request_id=%s]",
        type(exc).__name__, exc,
        getattr(request.state, "request_id", "unknown"),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An internal error occurred. Please try again later.",
            status_code=500,
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


# ═══════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════


def _load_model() -> None:
    """Load the best available prediction model from disk."""
    try:
        import joblib

        model_paths = [
            config.paths.models / "ensemble.pkl",
            config.paths.models / "xgboost_model.pkl",
            config.paths.models / "weighted_ensemble.joblib",
            config.paths.models / "model.pkl",
            Path("models/ensemble.pkl"),
            Path("models/xgboost_model.pkl"),
        ]

        for mp in model_paths:
            if mp.exists():
                state.model = joblib.load(mp)
                state.model_name = mp.name
                logger.info("Loaded model: %s", mp)
                break

        if state.model is None:
            logger.warning("No trained model found. Predictions will use fallback.")

        # Try to load the model registry
        try:
            from src.models.registry import ModelRegistry
            state.model_registry = ModelRegistry(use_db=False)
            # Register any models found on disk
            for mp in model_paths:
                if mp.exists():
                    try:
                        m = joblib.load(mp)
                        if hasattr(m, "model_name"):
                            state.model_registry.register(m, force=True)
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception as exc:
        logger.error("Failed to load model: %s", exc)


# ═══════════════════════════════════════════════════════════
#  Prediction Logic
# ═══════════════════════════════════════════════════════════


def _predict_match(
    home_team: str,
    away_team: str,
    match_date: str | None = None,
) -> dict[str, Any]:
    """Generate a prediction for a single match.

    Strategy (in order of preference):
    1. Loaded model + feature engineering pipeline → real predictions
    2. Deterministic fallback based on team-name hashing → estimated probs

    Currently uses strategy 2 (deterministic fallback) because the
    feature engineering pipeline requires real match data (DataFrame with
    all expected columns) and the model must be trained on those features.

    To enable real predictions in production:
    1. Train a model with ``train_worldcup.py`` or ``run_combined_pipeline.py``
    2. Load feature-engineered data into the API (via POST /predict with
       a pre-processed DataFrame)
    3. The model will automatically use the feature pipeline

    Returns a dict with probabilities, confidence, and model info.
    """
    # Try real predictions with feature engineering
    if state.model is not None:
        try:
            return _predict_with_features(home_team, away_team)
        except Exception as exc:
            logger.debug("Feature-based prediction unavailable: %s", exc)

    # Deterministic fallback based on team names
    return _fallback_prediction(home_team, away_team)


def _predict_with_features(home_team: str, away_team: str) -> dict[str, Any]:
    """Generate a prediction using the loaded model with feature engineering.

    Attempts to use the existing feature engineering pipeline from
    ``src.feature_engineering`` to generate real feature vectors
    from team names. Falls back gracefully if the pipeline is
    unavailable.
    """
    # Try to use the real feature engineering pipeline
    try:
        from src.data_loader import load_results
        from src.feature_engineering import build_features

        # Load historical match data
        df = load_results()
        if df is None or df.empty:
            raise ValueError("No historical data available for feature engineering")

        # Create a synthetic fixture row
        synthetic = {
            "date": pd.Timestamp.now(),
            "home_team": home_team,
            "away_team": away_team,
            "result": "H",  # placeholder
            "home_goals": 0,
            "away_goals": 0,
        }

        # Fill in missing columns from the data
        for col in df.columns:
            if col not in synthetic:
                synthetic[col] = df[col].iloc[-1] if len(df) > 0 else 0

        df_ext = pd.concat([df, pd.DataFrame([synthetic])], ignore_index=True)
        X_full, _ = build_features(df_ext, is_training=False)
        feature_row = X_full.iloc[-1:]

        model = state.model
        probs = model.predict_proba(feature_row)[0]
        pred_class = int(model.predict(feature_row)[0])

        # Determine class ordering (sklearn convention: sorted by class)
        if hasattr(model, "classes_"):
            classes = sorted(model.classes_)
            # Map class indices to [away, draw, home]
            probs_out = [0.0, 0.0, 0.0]
            for i, cls in enumerate(classes):
                if cls == 0:
                    probs_out[0] = float(probs[i])
                elif cls == 1:
                    probs_out[1] = float(probs[i])
                elif cls == 2:
                    probs_out[2] = float(probs[i])
            if sum(probs_out) > 0:
                total = sum(probs_out)
                probs_out = [p / total for p in probs_out]
            else:
                probs_out = [float(probs[0]), float(probs[1]), float(probs[2])]
        else:
            # Assume [home, draw, away] or [away, draw, home] order
            probs = np.asarray(probs)
            if len(probs) == 3:
                probs_out = [float(probs[2]), float(probs[1]), float(probs[0])]
            else:
                probs_out = [0.33, 0.34, 0.33]

        # Normalise to sum to 1.0
        total = sum(probs_out)
        if total > 0:
            probs_out = [p / total for p in probs_out]

        labels = ["Away Win", "Draw", "Home Win"]
        pred_idx = int(np.argmax(probs_out))
        confidence = probs_out[pred_idx]

        logger.info(
            "Feature-based prediction: %s vs %s → %s (%.1f%%)",
            home_team, away_team, labels[pred_idx], confidence * 100,
        )

        return {
            "probabilities": {
                "home_win": round(probs_out[2], 4),
                "draw": round(probs_out[1], 4),
                "away_win": round(probs_out[0], 4),
            },
            "predicted_outcome": labels[pred_idx],
            "confidence": round(confidence, 4),
            "model": state.model_name,
        }
    except Exception as exc:
        logger.debug(
            "Feature pipeline failed for %s vs %s: %s. Using fallback.",
            home_team, away_team, exc,
        )
        raise  # Re-raise to let _predict_match handle the fallback


def _fallback_prediction(home_team: str, away_team: str) -> dict[str, Any]:
    """Generate a deterministic fallback prediction using team name hashing."""
    import hashlib
    import random as rnd

    seed = int(hashlib.md5(f"{home_team}|{away_team}".encode()).hexdigest()[:8], 16)
    rng = rnd.Random(seed)

    home_str = rng.uniform(0.30, 0.55)
    away_str = rng.uniform(0.20, 0.45)
    draw_str = rng.uniform(0.20, 0.35)
    total = home_str + draw_str + away_str

    probs = {
        "away_win": round(away_str / total, 4),
        "draw": round(draw_str / total, 4),
        "home_win": round(home_str / total, 4),
    }

    outcome = max(probs, key=probs.get)
    label_map = {
        "home_win": "Home Win",
        "draw": "Draw",
        "away_win": "Away Win",
    }

    return {
        "probabilities": probs,
        "predicted_outcome": label_map[outcome],
        "confidence": probs[outcome],
        "model": "fallback_estimation",
    }


def _calculate_ev_and_kelly(
    probabilities: dict[str, float],
    odds: dict[str, float],
) -> tuple[float | None, float | None]:
    """Calculate Expected Value and Kelly stake for given probabilities and odds.

    Returns (ev, kelly_fraction).
    """
    if not odds:
        return None, None

    outcomes = {
        "home_win": ("home_odds", probabilities.get("home_win", 0)),
        "draw": ("draw_odds", probabilities.get("draw", 0)),
        "away_win": ("away_odds", probabilities.get("away_win", 0)),
    }

    evs = {}
    for outcome, (odds_key, prob) in outcomes.items():
        decimal_odds = odds.get(odds_key, 0)
        if decimal_odds > 0 and prob > 0:
            ev = round((prob * decimal_odds) - 1, 6)
            evs[outcome] = ev

    # Return the EV for the most likely outcome (or the highest EV)
    if not evs:
        return None, None

    best_outcome = max(evs, key=evs.get)
    best_ev = evs[best_outcome]

    # Calculate Kelly for the best outcome
    odds_key = {
        "home_win": "home_odds",
        "draw": "draw_odds",
        "away_win": "away_odds",
    }[best_outcome]

    decimal_odds = odds.get(odds_key, 0)
    prob = probabilities[best_outcome]

    kelly = None
    if decimal_odds > 1 and prob > 0:
        kelly_raw = (prob * decimal_odds - 1) / (decimal_odds - 1)
        kelly = round(max(0, min(kelly_raw, 1)), 6)  # clamp to [0, 1]

    return best_ev, kelly


# ═══════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════


# ── Health ──────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns the current health status of the API.",
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=state.model is not None,
        uptime_seconds=round(time.time() - state.start_time, 2),
    )


# ── List Models ─────────────────────────────────────────────
@app.get(
    "/models",
    response_model=ModelListResponse,
    tags=["Models"],
    summary="List available models",
    description="Returns a list of all loaded prediction models with metadata.",
)
async def list_models(
    auth: str | None = Depends(optional_auth),
) -> ModelListResponse:
    """List all available prediction models."""
    models_list: list[ModelInfo] = []

    # If we have a loaded model
    if state.model is not None:
        model_info = ModelInfo(
            name=state.model_name,
            model_type=type(state.model).__name__,
            version=getattr(state.model, "model_version", "0.1.0"),
            fitted=getattr(state.model, "_fitted", True) or getattr(state.model, "trained", False),
            calibrated=getattr(state.model, "_calibrated", False),
            features=getattr(state.model, "n_features_in_", 0)
            or getattr(state.model, "_n_features", 0),
        )

        # Get metrics
        metrics = {}
        if hasattr(state.model, "_training_metrics"):
            metrics = state.model._training_metrics
        elif hasattr(state.model, "_val_log_loss"):
            metrics["val_log_loss"] = state.model._val_log_loss

        if metrics:
            model_info.metrics = metrics

        # Try to add trained_at
        if hasattr(state.model, "_fit_completed_at"):
            model_info.trained_at = str(state.model._fit_completed_at)

        models_list.append(model_info)

    # If we have a registry, query it
    if state.model_registry is not None:
        for name, model in getattr(state.model_registry, "_models", {}).items():
            # Avoid duplicates
            if not any(m.name == name for m in models_list):
                models_list.append(ModelInfo(
                    name=name,
                    model_type=getattr(model, "model_type", "unknown"),
                    version=getattr(model, "model_version", "0.1.0"),
                    fitted=getattr(model, "_fitted", False),
                    calibrated=getattr(model, "_calibrated", False),
                ))

    return ModelListResponse(
        models=models_list,
        total=len(models_list),
    )


# ── Predict ─────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict match outcomes",
    description=(
        "Accept a list of football fixtures and return AI-powered predictions "
        "with probabilities, expected value, and Kelly stake recommendations."
    ),
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid request (validation error)"},
        401: {"description": "Missing or invalid API key"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def predict(
    request: Request,
    body: PredictWithOddsRequest,
    auth: str | None = Depends(optional_auth),
    _rate_limit: None = None,  # Placeholder — rate limiting applied in middleware
) -> PredictResponse:
    """Predict match outcomes for the given fixtures."""
    start_time = time.time()

    predictions: list[MatchPrediction] = []

    for i, fixture in enumerate(body.fixtures):
        # Generate prediction
        result = _predict_match(
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            match_date=fixture.match_date,
        )

        probs = OutcomeProbabilities(**result["probabilities"])

        # Calculate EV and Kelly if odds are provided
        ev: float | None = None
        kelly: float | None = None
        implied_probs: dict[str, float] | None = None

        if body.odds and i < len(body.odds):
            odds_dict = {
                "home_odds": body.odds[i].home_odds,
                "draw_odds": body.odds[i].draw_odds,
                "away_odds": body.odds[i].away_odds,
            }
            ev, kelly = _calculate_ev_and_kelly(result["probabilities"], odds_dict)

            # Calculate implied probabilities from odds
            if body.odds[i].home_odds > 0:
                ip_home = 1.0 / body.odds[i].home_odds
                ip_draw = 1.0 / body.odds[i].draw_odds
                ip_away = 1.0 / body.odds[i].away_odds
                ip_total = ip_home + ip_draw + ip_away
                if ip_total > 0:
                    implied_probs = {
                        "home_win": round(ip_home / ip_total, 4),
                        "draw": round(ip_draw / ip_total, 4),
                        "away_win": round(ip_away / ip_total, 4),
                    }

        predictions.append(MatchPrediction(
            fixture=fixture,
            predicted_outcome=result["predicted_outcome"],
            probabilities=probs,
            confidence=result["confidence"],
            model=result["model"],
            expected_value=ev,
            kelly_stake=kelly,
            implied_probabilities=OutcomeProbabilities(**implied_probs) if implied_probs else None,
        ))

    processing_time = round((time.time() - start_time) * 1000, 2)

    return PredictResponse(
        status="success",
        predictions=predictions,
        model_info={
            "name": state.model_name,
            "loaded": state.model is not None,
        },
        processing_time_ms=processing_time,
    )


# ── Catch-all route for docs ────────────────────────────────
@app.get("/", tags=["System"], include_in_schema=False)
async def root() -> dict[str, Any]:
    """Root endpoint — redirects to API docs."""
    return {
        "message": "Football Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "GET /health": "Health check",
            "GET /models": "List available models",
            "POST /predict": "Predict match outcomes",
        },
    }


# ═══════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════


def main() -> None:
    """Run the API server directly."""
    import uvicorn

    port = int(os.environ.get("API_PORT", "8000"))
    host = os.environ.get("API_HOST", "0.0.0.0")

    print("=" * 60)
    print("  FOOTBALL PREDICTION API")
    print("=" * 60)
    print(f"\n  Server: http://{host}:{port}")
    print(f"  Docs:   http://{host}:{port}/docs")
    print(f"  API Key: {os.environ.get('PREDICTION_API_KEY', 'dev-key-change-in-production')[:8]}...")
    print("\n  Press Ctrl+C to stop.\n")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
