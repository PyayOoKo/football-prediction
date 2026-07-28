"""
SQLAlchemy declarative base and metadata.

All ORM models inherit from ``Base``, which provides:
- A ``metadata`` object for Alembic autogeneration.
- Common ``__tablename__`` conventions.

Soft Deletes
------------
Models that need **soft delete** support should inherit ``SoftDeleteMixin``
in addition to ``Base``::

    from src.database.base import Base, SoftDeleteMixin

    class Match(Base, SoftDeleteMixin):
        __tablename__ = "matches"
        ...

The mixin adds a nullable ``deleted_at`` timestamp column. Queries
should filter on ``deleted_at.is_(None)``
(see :func:`~src.database.base.with_active`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Self

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


# ═══════════════════════════════════════════════════════════
#  Soft Delete Mixin
# ═══════════════════════════════════════════════════════════


class SoftDeleteMixin:
    """Adds soft delete support to an ORM model.

    Adds a nullable ``deleted_at`` :class:`~datetime.datetime` column.
    When a record is "deleted", this timestamp is set instead of the
    row being removed from the database.  Active (non-deleted) records
    have ``deleted_at IS NULL``.

    Usage
    -----
    ::

        from src.database.base import Base, SoftDeleteMixin

        class MyModel(Base, SoftDeleteMixin):
            __tablename__ = "my_model"
            ...

        # Delete
        record.soft_delete()

        # Query active only
        active = session.query(MyModel).filter(MyModel.deleted_at.is_(None))

        # Query including deleted (admin / audit only)
        all_incl = session.query(MyModel)
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    @property
    def is_deleted(self) -> bool:
        """``True`` if this record has been soft-deleted."""
        return self.deleted_at is not None

    def soft_delete(self) -> Self:
        """Mark this record as deleted by setting ``deleted_at`` to now.

        Returns ``self`` so calls can be chained:
        ``record.soft_delete()``.
        """
        self.deleted_at = datetime.now(timezone.utc)
        return self
