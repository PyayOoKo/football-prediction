"""Data structures — OddsSnapshot and LivePrediction dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

OUTCOME_NAMES = ["Away Win", "Draw", "Home Win"]
OUTCOME_SHORT = ["A", "D", "H"]

_ODDS_TO_MODEL: dict[str, int] = {
    "home_odds": 2,
    "draw_odds": 1,
    "away_odds": 0,
}


@dataclass
class OddsSnapshot:
    home_team: str
    away_team: str
    sport_key: str
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmaker: str
    timestamp: str
    match_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "sport_key": self.sport_key,
            "home_odds": self.home_odds,
            "draw_odds": self.draw_odds,
            "away_odds": self.away_odds,
            "bookmaker": self.bookmaker,
            "timestamp": self.timestamp,
            "match_date": self.match_date,
        }


@dataclass
class LivePrediction:
    home_team: str
    away_team: str
    match_date: str
    sport_key: str
    home_prob: float
    draw_prob: float
    away_prob: float
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmaker: str
    home_ev: float
    draw_ev: float
    away_ev: float
    home_clv: float
    draw_clv: float
    away_clv: float
    home_kelly: float
    draw_kelly: float
    away_kelly: float
    prev_home_odds: float | None = None
    prev_draw_odds: float | None = None
    prev_away_odds: float | None = None
    timestamp: str = ""
    confidence_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
            "sport_key": self.sport_key,
            "home_prob": round(self.home_prob, 4),
            "draw_prob": round(self.draw_prob, 4),
            "away_prob": round(self.away_prob, 4),
            "home_odds": self.home_odds,
            "draw_odds": self.draw_odds,
            "away_odds": self.away_odds,
            "bookmaker": self.bookmaker,
            "home_ev": round(self.home_ev, 4),
            "draw_ev": round(self.draw_ev, 4),
            "away_ev": round(self.away_ev, 4),
            "home_clv": round(self.home_clv, 4),
            "draw_clv": round(self.draw_clv, 4),
            "away_clv": round(self.away_clv, 4),
            "home_kelly": round(self.home_kelly, 4),
            "draw_kelly": round(self.draw_kelly, 4),
            "away_kelly": round(self.away_kelly, 4),
            "predicted_outcome": self.predicted_outcome,
            "best_value": self.best_value_outcome,
            "best_value_ev": round(self.best_value_ev, 4),
            "confidence_score": round(self.confidence_score, 2),
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "n_value_bets": self.n_value_bets,
            "value_outcomes": self.value_outcomes,
        }

    @property
    def predicted_outcome(self) -> str:
        idx = int(np.argmax([self.away_prob, self.draw_prob, self.home_prob]))
        return OUTCOME_NAMES[idx]

    @property
    def best_value_outcome(self) -> str:
        idx = int(np.argmax([self.away_ev, self.draw_ev, self.home_ev]))
        return OUTCOME_NAMES[idx]

    @property
    def best_value_ev(self) -> float:
        return max(self.home_ev, self.draw_ev, self.away_ev)

    @property
    def n_value_bets(self) -> int:
        return sum(1 for ev in [self.home_ev, self.draw_ev, self.away_ev] if ev > 0)

    @property
    def value_outcomes(self) -> list[str]:
        return [
            outcome for outcome, ev in zip(OUTCOME_NAMES, [self.away_ev, self.draw_ev, self.home_ev])
            if ev > 0
        ]
