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
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env before any other imports that read env vars
load_dotenv(dotenv_path=_project_root / ".env")

from api.auth import IS_DEV_MODE, rate_limiter, verify_api_key
from api.models import (
    ErrorResponse,
    HealthResponse,
    MatchPrediction,
    ModelInfo,
    ModelListResponse,
    OutcomeProbabilities,
    PredictResponse,
    PredictWithOddsRequest,
)
from config import config

logger = logging.getLogger(__name__)


# ── Application state ──────────────────────────────────────
class AppState:
    """Shared application state, populated at startup."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_name: str = "none"
        self.model_type: str = "none"
        self.model_path: str = ""
        self.model_trained_at: str | None = None
        self.model_feature_count: int = 0
        self.model_feature_names: list[str] = []
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
# In production, specify explicit allowed origins.
# Wildcard "*" with allow_credentials=True is invalid and rejected by browsers.
_ALLOWED_ORIGINS = (
    os.environ.get("API_ALLOWED_ORIGINS", "").split(",")
    if os.environ.get("API_ALLOWED_ORIGINS")
    else None
)
if _ALLOWED_ORIGINS:
    _ALLOWED_ORIGINS = [o.strip() for o in _ALLOWED_ORIGINS if o.strip()]

if _ALLOWED_ORIGINS:
    _cors_origins = _ALLOWED_ORIGINS
    _cors_credentials = True
elif IS_DEV_MODE:
    _cors_origins = ["*"]
    _cors_credentials = False  # Cannot combine * with credentials
    logger.warning("CORS: allow_origins=['*'] in dev mode (credentials disabled)")
else:
    _cors_origins = ["*"]  # Fallback — safe because credentials=False
    _cors_credentials = False
    logger.warning(
        "CORS: No API_ALLOWED_ORIGINS set. Defaulting to '*'. Set API_ALLOWED_ORIGINS in production."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
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
                detail=(
                    f"Rate limit exceeded. Max {rate_limiter.max_requests} requests "
                    f"per {rate_limiter.window_seconds} seconds per IP."
                ),
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
        type(exc).__name__,
        exc,
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
    """Load the best available prediction model from disk.

    Searches the models/ directory for the actual filenames produced by
    the training service: {model_type}_model.joblib (e.g. lightgbm_model.joblib,
    xgboost_model.joblib, random_forest_model.joblib, logistic_regression_model.joblib).
    Falls back to any .joblib file in the models directory.

    Validates that the loaded artifact has predict() and predict_proba().
    Does NOT silently replace a missing model with fake predictions.
    """
    try:

        model_dir = config.paths.models

        # 1. Explicit configured model path (highest priority)
        configured_path = getattr(config, "api_model_path", None)
        if configured_path:
            p = Path(configured_path)
            if p.exists():
                _try_load(p)
                if state.model is not None:
                    return

        # 2. Search for filenames matching training service output patterns
        training_patterns = [
            "lightgbm_model.joblib",
            "xgboost_model.joblib",
            "random_forest_model.joblib",
            "logistic_regression_model.joblib",
            "worldcup_xgboost.joblib",
            "worldcup_lightgbm.joblib",
        ]
        for name in training_patterns:
            p = model_dir / name
            if p.exists():
                _try_load(p)
                if state.model is not None:
                    return

        # 3. Fallback: most recently modified .joblib file
        candidates = sorted(
            model_dir.glob("*.joblib"),
            key=lambda p: p.stat().st_mtime,
        )
        for mp in candidates:
            _try_load(mp)
            if state.model is not None:
                return

        if state.model is None:
            logger.warning(
                "No trained model found in %s. "
                "The API will return 503 Service Unavailable for /predict. "
                "Train a model first via TrainingService.train().",
                model_dir,
            )

    except Exception as exc:
        logger.error("Failed to load model: %s", exc)


def _try_load(path: Path) -> None:
    """Try to load a model from *path*, capturing metadata into AppState."""
    from datetime import datetime

    import joblib

    try:
        model = joblib.load(path)
        # Validate predict and predict_proba exist
        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            logger.warning("Skipping %s: missing predict() or predict_proba()", path)
            return
        # Validate they're callable
        if not callable(model.predict) or not callable(model.predict_proba):
            logger.warning(
                "Skipping %s: predict() or predict_proba() not callable", path
            )
            return
        state.model = model
        state.model_name = path.name
        state.model_type = type(model).__name__
        state.model_path = str(path.absolute())

        # Capture training timestamp from file modification time
        try:
            state.model_trained_at = datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat()
        except Exception:
            state.model_trained_at = None

        # Capture expected feature count
        if hasattr(model, "n_features_in_"):
            state.model_feature_count = int(model.n_features_in_)
        elif hasattr(model, "_n_features"):
            state.model_feature_count = int(model._n_features)
        else:
            state.model_feature_count = 0

        # Capture feature names
        if hasattr(model, "feature_names_in_"):
            state.model_feature_names = list(model.feature_names_in_)
        elif hasattr(model, "get_booster"):
            try:
                names = model.get_booster().feature_names
                if names:
                    state.model_feature_names = list(names)
            except Exception:
                pass

        logger.info(
            "Loaded model: %s (%s, %d features)",
            path.name,
            state.model_type,
            state.model_feature_count,
        )
    except Exception as exc:
        logger.warning("Failed to load model %s: %s", path, exc)


# ═══════════════════════════════════════════════════════════
#  Prediction Logic
# ═══════════════════════════════════════════════════════════


def _predict_match(
    home_team: str,
    away_team: str,
    match_date: str | None = None,
) -> dict[str, Any]:
    """Generate a prediction for a single match using the loaded model.

    Raises HTTPException(503) if no valid model is loaded.
    Never returns fake/fallback predictions in production.

    In development mode, if ``ALLOW_DEV_FALLBACK`` is set, returns a
    simulated prediction clearly marked as such.
    """
    if state.model is None:
        # In dev mode with explicit flag, allow simulated predictions
        if IS_DEV_MODE and os.environ.get("ALLOW_DEV_FALLBACK", "").lower() in (
            "true",
            "1",
        ):
            return _simulated_prediction(home_team, away_team)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No prediction model is loaded. Train a model first. "
                f"Checked in: {config.paths.models}"
            ),
        )

    try:
        return _predict_with_features(home_team, away_team)
    except HTTPException:
        raise  # Pass through alignment/validation errors cleanly
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction pipeline failed: {exc}",
        )


def _validate_feature_alignment(feature_row: pd.DataFrame) -> None:
    """Validate that the pipeline's feature columns match model expectations.

    Checks both feature count and column names against the metadata
    captured when the model was loaded.  Raises
    ``HTTPException(503)`` with a clear diagnostic message on
    mismatch.

    Parameters
    ----------
    feature_row : pd.DataFrame
        The feature row(s) produced by the pipeline.
    """
    if state.model is None:
        return

    actual_cols = set(feature_row.columns)
    actual_count = len(feature_row.columns)

    # 1. Feature count check
    if state.model_feature_count > 0 and actual_count != state.model_feature_count:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Feature count mismatch: model expects "
                f"{state.model_feature_count} features, but pipeline produced "
                f"{actual_count}. Model: {state.model_name}. "
                "Retrain the model with the current pipeline."
            ),
        )

    # 2. Feature name check (if model exposes them)
    if state.model_feature_names:
        expected = set(state.model_feature_names)
        missing = expected - actual_cols
        extra = actual_cols - expected

        if missing or extra:
            msg_parts = [f"Feature column mismatch for {state.model_name}:"]
            if missing:
                sorted_missing = sorted(missing)[:15]
                msg_parts.append(f"{len(missing)} missing: {sorted_missing}")
            if extra:
                sorted_extra = sorted(extra)[:15]
                msg_parts.append(f"{len(extra)} extra: {sorted_extra}")
            msg_parts.append("Retrain the model to align with the current pipeline.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=" | ".join(msg_parts),
            )

    # 3. Log success for debugging
    logger.debug(
        "Feature alignment OK: %d features match model '%s'",
        actual_count,
        state.model_name,
    )


def _predict_with_features(home_team: str, away_team: str) -> dict[str, Any]:
    """Generate a prediction using the loaded model with feature engineering.

    Attempts to use the existing feature engineering pipeline from
    ``src.feature_engineering`` to generate real feature vectors
    from team names. Validates feature alignment before inference.

    Raises HTTPException(503) with clear details on feature mismatch
    or pipeline failure.
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

        # ── Feature alignment validation ───────────────────
        _validate_feature_alignment(feature_row)

        # Enforce training-time column ordering
        if state.model_feature_names:
            feature_row = feature_row[list(state.model_feature_names)]

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
            home_team,
            away_team,
            labels[pred_idx],
            confidence * 100,
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
            home_team,
            away_team,
            exc,
        )
        raise  # Re-raise to let _predict_match handle the fallback


def _simulated_prediction(home_team: str, away_team: str) -> dict[str, Any]:
    """Generate a simulated prediction for development/testing only.

    This is clearly marked as simulated and must NOT be used for
    actual betting decisions. Only available in dev mode with
    ALLOW_DEV_FALLBACK=true.
    """
    # Uniform probabilities — no real prediction
    probs = {"home_win": 0.34, "draw": 0.33, "away_win": 0.33}
    return {
        "probabilities": probs,
        "predicted_outcome": "Home Win",
        "confidence": 0.34,
        "model": "simulated-dev-mode",
        "simulated": True,
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
        model_name=state.model_name,
        model_type=state.model_type,
        model_features=state.model_feature_count,
        model_trained_at=state.model_trained_at,
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
    auth: str = Depends(verify_api_key),
) -> ModelListResponse:
    """List all available prediction models."""
    models_list: list[ModelInfo] = []

    # If we have a loaded model (use state metadata)
    if state.model is not None:
        model_info = ModelInfo(
            name=state.model_name,
            model_type=state.model_type,
            version=getattr(state.model, "model_version", "0.1.0"),
            fitted=getattr(state.model, "_fitted", True)
            or getattr(state.model, "trained", False),
            calibrated=getattr(state.model, "_calibrated", False),
            features=state.model_feature_count,
            model_path=state.model_path,
            trained_at=state.model_trained_at,
        )

        # Get metrics
        metrics = {}
        if hasattr(state.model, "_training_metrics"):
            metrics = state.model._training_metrics
        elif hasattr(state.model, "_val_log_loss"):
            metrics["val_log_loss"] = state.model._val_log_loss

        if metrics:
            model_info.metrics = metrics

        models_list.append(model_info)

    # If we have a registry, query it
    if state.model_registry is not None:
        for reg_name, reg_model in getattr(state.model_registry, "_models", {}).items():
            # Avoid duplicates
            if not any(m.name == reg_name for m in models_list):
                models_list.append(
                    ModelInfo(
                        name=reg_name,
                        model_type=getattr(reg_model, "model_type", "unknown"),
                        version=getattr(reg_model, "model_version", "0.1.0"),
                        fitted=getattr(reg_model, "_fitted", False),
                        calibrated=getattr(reg_model, "_calibrated", False),
                        features=(
                            getattr(reg_model, "n_features_in_", 0)
                            or getattr(reg_model, "_n_features", 0)
                        ),
                        model_path="",
                        trained_at=(getattr(reg_model, "_fit_completed_at", None)),
                    )
                )

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
    auth: str = Depends(verify_api_key),
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

        predictions.append(
            MatchPrediction(
                fixture=fixture,
                predicted_outcome=result["predicted_outcome"],
                probabilities=probs,
                confidence=result["confidence"],
                model=result["model"],
                expected_value=ev,
                kelly_stake=kelly,
                implied_probabilities=(
                    OutcomeProbabilities(**implied_probs) if implied_probs else None
                ),
            )
        )

    processing_time = round((time.time() - start_time) * 1000, 2)

    return PredictResponse(
        status="success",
        predictions=predictions,
        model_info={
            "name": state.model_name,
            "type": state.model_type,
            "path": state.model_path,
            "feature_count": state.model_feature_count,
            "trained_at": state.model_trained_at,
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
    api_key_status = "configured" if os.environ.get("PREDICTION_API_KEY") else "not set"
    print(f"  API Key: {api_key_status}")
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
