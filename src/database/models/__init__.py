"""
SQLAlchemy ORM models — fully normalised football analytics schema.

All models inherit from ``src.database.base.Base``.
Import this module in Alembic's ``env.py`` for autogeneration::

    from src.database.models import *   # noqa: F401, F403

Tables
------
Core entities:
- Country        — ISO-coded country reference (alpha2, alpha3, FIFA code)
- Competition    — League/cup/tournament (replaces old League model)
- Season         — Time-bound grouping within a competition
- Team           — Club or national team (country_id FK)
- Stadium        — Venue (city, capacity, surface, country_id FK)
- Referee        — Match official (country_id FK)
- Match          — Central fact table (7 FKs, 6 CHECK constraints)

Match detail (1:1):
- MatchStatistics — Shots, possession, cards, corners
- Weather         — Temperature, humidity, wind, pitch condition

Match detail (1:N):
- Odds            — Multi-bookmaker, multi-timestamp decimal odds
- Lineup          — Formation, starting XI JSON, substitutes

Team analytics (computed after each match):
- TeamForm        — Pre-computed rolling form (last 5/10/20)
- TeamEloHistory  — Elo rating snapshot before/after
- TeamXgHistory   — xG, xA, shot counts per source

Player analytics:
- Player              — Personal info, position, market value
- PlayerMatchStats    — Per-match performance (goals, xG, rating)
- Injury              — Injury tracking (type, severity, return)
- Transfer            — Transfer fees, loans, dates

Betting & predictions:
- Prediction          — Model probabilities, confidence (from previous arch)
- ExpectedValueBet    — EV calculations per match+bookmaker
- ClosingLineValue    — Opening-to-closing line movement
- BettingResult       — Actual bet outcomes and P&L tracking
"""

from src.database.models.betting_result import BettingResult
from src.database.models.closing_line_value import ClosingLineValue
from src.database.models.competition import Competition
from src.database.models.country import Country
from src.database.models.expected_value_bet import ExpectedValueBet
from src.database.models.injury import Injury
from src.database.models.lineup import Lineup
from src.database.models.match import Match
from src.database.models.match_statistics import MatchStatistics
from src.database.models.odds import Odds
from src.database.models.player import Player
from src.database.models.player_match_stats import PlayerMatchStats
from src.database.models.prediction import Prediction
from src.database.models.referee import Referee
from src.database.models.season import Season
from src.database.models.stadium import Stadium
from src.database.models.team import Team
from src.database.models.team_elo_history import TeamEloHistory
from src.database.models.team_form import TeamForm
from src.database.models.team_xg_history import TeamXgHistory
from src.database.models.transfer import Transfer
from src.database.models.weather import Weather

__all__ = [
    "BettingResult",
    "ClosingLineValue",
    "Competition",
    "Country",
    "ExpectedValueBet",
    "Injury",
    "Lineup",
    "Match",
    "MatchStatistics",
    "Odds",
    "Player",
    "PlayerMatchStats",
    "Prediction",
    "Referee",
    "Season",
    "Stadium",
    "Team",
    "TeamEloHistory",
    "TeamForm",
    "TeamXgHistory",
    "Transfer",
    "Weather",
]
