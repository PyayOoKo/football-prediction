"""
Football Prediction — ML pipeline for predicting match outcomes.

Package layout
--------------
src.config
    Application configuration and logging setup.
src.database
    SQLAlchemy ORM, models, repositories, and Alembic migrations.
src.data
    Data loading, preprocessing, cleaning, and feature engineering.
src.models
    ML model implementations (XGBoost, Poisson, ensemble).
src.scrapers
    Data collection from football-data.co.uk, Transfermarkt, etc.
src.services
    Business logic orchestration (prediction, training, betting).
src.utils
    Cross-cutting utilities (exceptions, helpers, validators).

Quick start
-----------
::

    from src.config import configure_logging
    configure_logging()

    from src.database import get_session
    with get_session() as session:
        ...
"""

from __future__ import annotations

__version__ = "0.1.0"
