"""Data structures for prediction results and bet recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionResult:
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
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "over_2_5_prob": round(self.over_2_5_prob, 4)
            if self.over_2_5_prob is not None
            else None,
            "under_2_5_prob": round(self.under_2_5_prob, 4)
            if self.under_2_5_prob is not None
            else None,
            "over_3_5_prob": round(self.over_3_5_prob, 4)
            if self.over_3_5_prob is not None
            else None,
            "under_3_5_prob": round(self.under_3_5_prob, 4)
            if self.under_3_5_prob is not None
            else None,
            "btts_prob": round(self.btts_prob, 4)
            if self.btts_prob is not None
            else None,
            "btts_no_prob": round(self.btts_no_prob, 4)
            if self.btts_no_prob is not None
            else None,
        }


@dataclass
class BetRecommendation:
    fixture: dict[str, Any] = field(default_factory=dict)
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
