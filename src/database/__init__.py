"""
Database package — SQLAlchemy ORM setup and repository pattern.

Sub-modules
-----------
base
    Declarative base class and metadata.
session
    Session factory, engine, and dependency helpers.
models
    SQLAlchemy ORM model definitions (Match, Team, League, Prediction).
repositories
    Repository pattern for database access.
"""

from src.database.base import Base
from src.database.session import (
    create_engine_from_config,
    create_session_factory,
    get_session,
    init_db,
)

__all__ = [
    "Base",
    "create_engine_from_config",
    "create_session_factory",
    "get_session",
    "init_db",
]
