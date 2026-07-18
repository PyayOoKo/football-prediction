"""
Pydantic models for the Football Prediction REST API.

Defines request and response schemas for all endpoints with
validation rules, field descriptions, and example values.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ───────────────────────────────────────────────────
class MarketEnum(str, Enum):
    """Supported betting markets."""

    h2h = "h2h"
    btts = "btts"
    over_under = "over_under"


class OutcomeEnum(str, Enum):
    """Match outcome labels."""

    home_win = "Home Win"
    draw = "Draw"
    away_win = "Away Win"


# ── Fixture Input ───────────────────────────────────────────
class FixtureInput(BaseModel):
    """A single football fixture to predict."""

    home_team: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Home team name",
        examples=["Brazil"],
    )
    away_team: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Away team name",
        examples=["Argentina"],
    )
    match_date: str | None = Field(
        None,
        description="Match date (YYYY-MM-DD). Defaults to today.",
        examples=["2026-07-15"],
    )
    competition: str | None = Field(
        None,
        description="Competition name (optional, for context)",
        examples=["World Cup 2026"],
    )
    home_goals: float | None = Field(
        None,
        ge=0,
        description="Historical home goals (for H2H stats, optional)",
    )
    away_goals: float | None = Field(
        None,
        ge=0,
        description="Historical away goals (for H2H stats, optional)",
    )

    @field_validator("home_team", "away_team")
    @classmethod
    def teams_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Team name cannot be empty")
        return v

    @model_validator(mode="after")
    def home_not_away(self) -> FixtureInput:
        if self.home_team.strip().lower() == self.away_team.strip().lower():
            raise ValueError(
                f"Home team '{self.home_team}' and away team '{self.away_team}' must be different"
            )
        return self


# ── Odds Input (optional, for EV calculation) ───────────────
class OddsInput(BaseModel):
    """Optional odds input for EV calculation."""

    home_odds: float = Field(..., ge=1.0, description="Home win decimal odds")
    draw_odds: float = Field(..., ge=1.0, description="Draw decimal odds")
    away_odds: float = Field(..., ge=1.0, description="Away win decimal odds")


class PredictRequest(BaseModel):
    """Request body for the prediction endpoint."""

    fixtures: list[FixtureInput] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of fixtures to predict (1-50 matches)",
    )
    market: MarketEnum = Field(
        default=MarketEnum.h2h,
        description="Betting market to predict",
    )
    include_expected_value: bool = Field(
        default=True,
        description="Include expected value (EV) and Kelly stake calculations in response",
    )
    include_features: bool = Field(
        default=False,
        description="Include the feature vector in the response (verbose)",
    )

    @field_validator("fixtures")
    @classmethod
    def no_duplicate_fixtures(cls, v: list[FixtureInput]) -> list[FixtureInput]:
        seen: set[tuple[str, str]] = set()
        for f in v:
            key = (f.home_team.lower(), f.away_team.lower())
            if key in seen:
                raise ValueError(f"Duplicate fixture: {f.home_team} vs {f.away_team}")
            seen.add(key)
        return v


# ── Prediction Response ─────────────────────────────────────
class OutcomeProbabilities(BaseModel):
    """Probabilities for each match outcome."""

    home_win: float = Field(..., ge=0, le=1, description="Home win probability")
    draw: float = Field(..., ge=0, le=1, description="Draw probability")
    away_win: float = Field(..., ge=0, le=1, description="Away win probability")


class MatchPrediction(BaseModel):
    """Prediction result for a single fixture."""

    fixture: FixtureInput
    predicted_outcome: OutcomeEnum = Field(..., description="Predicted match result")
    probabilities: OutcomeProbabilities = Field(
        ..., description="Per-outcome probabilities"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Model confidence in prediction"
    )
    model: str = Field(..., description="Model name used for prediction")
    expected_value: float | None = Field(
        None, description="Expected Value (if odds available)"
    )
    kelly_stake: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Kelly Criterion fraction of bankroll to stake",
    )
    implied_probabilities: OutcomeProbabilities | None = Field(
        None, description="Bookmaker-implied probabilities (if odds provided)"
    )


class PredictResponse(BaseModel):
    """Response from the prediction endpoint."""

    status: str = Field("success", description="Response status")
    predictions: list[MatchPrediction] = Field(
        ...,
        description="List of predictions, one per fixture",
    )
    model_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Model metadata (name, version, fitted date)",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
    )


# ── Model Info Response ─────────────────────────────────────
class ModelInfo(BaseModel):
    """Information about a single model."""

    name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (xgboost, ensemble, etc.)")
    version: str = Field("0.1.0", description="Model version")
    fitted: bool = Field(False, description="Whether the model is fitted")
    calibrated: bool = Field(False, description="Whether probabilities are calibrated")
    features: int = Field(0, description="Number of features the model expects")
    trained_at: str | None = Field(None, description="When the model was trained")
    model_path: str = Field("", description="Absolute path to the saved model file")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Evaluation metrics (accuracy, log_loss, etc.)",
    )


class ModelListResponse(BaseModel):
    """Response listing available models."""

    models: list[ModelInfo] = Field(..., description="List of available models")
    total: int = Field(..., description="Total number of models")


# ── Health Response ─────────────────────────────────────────
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("healthy", description="Service health status")
    version: str = Field("2.0.0", description="API version")
    model_loaded: bool = Field(False, description="Whether a model is loaded")
    model_name: str = Field("none", description="Loaded model filename")
    model_type: str = Field("none", description="Loaded model class name")
    model_features: int = Field(0, description="Number of features model expects")
    model_trained_at: str | None = Field(
        None, description="Training timestamp (file mtime)"
    )
    uptime_seconds: float = Field(0.0, description="Server uptime in seconds")


# ── Error Response ──────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error description")
    status_code: int = Field(..., description="HTTP status code")
    request_id: str | None = Field(None, description="Unique request identifier")


class PredictWithOddsRequest(PredictRequest):
    """Prediction request with optional bookmaker odds."""

    odds: list[OddsInput] | None = Field(
        None,
        description="Bookmaker odds for each fixture (for EV calculation)",
        max_length=50,
    )

    @field_validator("odds")
    @classmethod
    def odds_match_fixtures(
        cls, v: list[OddsInput] | None, info: Any
    ) -> list[OddsInput] | None:
        if v is not None and info.data.get("fixtures"):
            if len(v) != len(info.data["fixtures"]):
                raise ValueError(
                    f"Number of odds entries ({len(v)}) must match "
                    f"number of fixtures ({len(info.data['fixtures'])})"
                )
        return v
