"""
Alembic environment configuration.

Handles migration context for both offline and online migrations.
Supports autogeneration by importing all ORM models.
"""

from __future__ import annotations

import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Alembic Config object ──────────────────────────────
config = context.config

# ── Logging setup ─────────────────────────────────────
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# ── Import all models so Alembic can detect changes ────
from src.database.base import Base  # noqa: E402

# Import models to register them with the metadata
from src.database.models import *  # noqa: E402, F401, F403

# Import Feature Store models so Alembic can detect them
from src.feature_store.models import (  # noqa: E402, F401, F403
    FeatureComputationBatch,
    FeatureDefinition,
    FeatureDependency,
    FeatureValue,
    FeatureVersion,
)
from src.feature_store.lineage import FeatureLineageEntry  # noqa: E402, F401

target_metadata = Base.metadata

# ── Helper: get the database URL ───────────────────────
def get_url() -> str:
    """Return the database URL from config or environment.

    Priority:
    1. ``sqlalchemy.url`` from ``alembic.ini``
    2. ``DATABASE_URL`` environment variable
    3. Application config (``settings.py``)

    Raises
    ------
    ConfigurationError
        If no database URL can be resolved.
    """
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url

    import os

    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url

    try:
        from src.config.settings import config as app_config
        return app_config.db.sa_url
    except ImportError:
        pass

    raise SystemExit(
        "ERROR: No database URL configured.\n\n"
        "Set DATABASE_URL in your .env file or configure\n"
        "sqlalchemy.url in alembic.ini."
    )


# ── Online migration ───────────────────────────────────
def run_migrations_online() -> None:
    """Run migrations in 'online' mode (with a live connection)."""
    url = get_url()
    logger.info("Connecting to: %s", url)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        url=url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ── Offline migration ──────────────────────────────────
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL script)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
