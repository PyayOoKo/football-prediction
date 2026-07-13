"""
SQLAlchemy declarative base and metadata.

All ORM models inherit from ``Base``, which provides:
- A ``metadata`` object for Alembic autogeneration.
- Common ``__tablename__`` conventions.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    # ── Naming convention for constraints ──────────────
    # These are used by Alembic so that migrations are
    # consistent across environments.
    metadata_naming_convention = {
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
