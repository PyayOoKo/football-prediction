"""
Repository pattern — abstract data access layer.

Repositories encapsulate database query logic, keeping service
code clean and making unit tests easier (repositories can be
mocked or swapped).
"""

from src.database.repositories.base import BaseRepository
from src.database.repositories.match_repository import MatchRepository
from src.database.repositories.team_repository import TeamRepository

__all__ = [
    "BaseRepository",
    "MatchRepository",
    "TeamRepository",
]
