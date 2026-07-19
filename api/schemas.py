"""
Pydantic models for API request/response validation.
Ensures type safety and data integrity at the API boundary.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for match prediction."""
    match_id: Optional[int] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    competition: Optional[str] = None
    date: Optional[datetime] = None
    
    @validator('home_team')
    def validate_teams(cls, v, values):
        if v is None and values.get('match_id') is None:
            raise ValueError("Either match_id or home_team must be provided")
        return v

class PredictionResponse(BaseModel):
    """Standardized response model for predictions."""
    match_id: int
    home_team: str
    away_team: str
    home_win_prob: float = Field(..., ge=0.0, le=1.0)
    draw_prob: float = Field(..., ge=0.0, le=1.0)
    away_win_prob: float = Field(..., ge=0.0, le=1.0)
    predicted_outcome: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    expected_goals_home: Optional[float] = None
    expected_goals_away: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('predicted_outcome')
    def validate_outcome(cls, v):
        allowed = ['HOME', 'DRAW', 'AWAY']
        if v not in allowed:
            raise ValueError(f"Outcome must be one of {allowed}")
        return v

class ValueBetRequest(BaseModel):
    """Request model for value bet calculation."""
    predictions: List[PredictionResponse]
    odds_home: float = Field(..., gt=0.0)
    odds_draw: float = Field(..., gt=0.0)
    odds_away: float = Field(..., gt=0.0)
    min_ev: float = Field(default=0.05, ge=0.0)  # Minimum 5% EV by default

class ValueBetResponse(BaseModel):
    """Response model for value bet opportunities."""
    match_id: int
    recommendation: str  # HOME, DRAW, AWAY, or NONE
    expected_value: float
    kelly_fraction: float
    odds_used: Dict[str, float]
    probabilities: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database_connected: bool
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
