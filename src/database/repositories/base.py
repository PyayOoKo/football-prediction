"""
Base repository — generic CRUD operations.

Provides a reusable base class for all entity-specific repositories.
Subclasses only need to define domain-specific query methods.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic repository with common CRUD operations.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy ORM session.
    model : type[ModelT]
        The ORM model class this repository manages.
    """

    def __init__(self, session: Session, model: type[ModelT]) -> None:
        self._session = session
        self._model = model

    # ── Read ───────────────────────────────────────────

    def get_by_id(self, entity_id: int) -> ModelT | None:
        return self._session.get(self._model, entity_id)

    def get_all(self) -> list[ModelT]:
        stmt = select(self._model)
        return list(self._session.scalars(stmt).all())

    def find(self, **filters: Any) -> list[ModelT]:
        stmt = select(self._model).filter_by(**filters)
        return list(self._session.scalars(stmt).all())

    def find_one(self, **filters: Any) -> ModelT | None:
        stmt = select(self._model).filter_by(**filters).limit(1)
        return self._session.scalars(stmt).first()

    def count(self, **filters: Any) -> int:
        from sqlalchemy.functions import count as sa_count

        stmt = select(sa_count()).select_from(self._model).filter_by(**filters)
        return self._session.scalar(stmt) or 0

    # ── Write ──────────────────────────────────────────

    def add(self, entity: ModelT) -> ModelT:
        self._session.add(entity)
        self._session.flush()
        return entity

    def add_all(self, entities: list[ModelT]) -> list[ModelT]:
        self._session.add_all(entities)
        self._session.flush()
        return entities

    def delete(self, entity: ModelT) -> None:
        self._session.delete(entity)
        self._session.flush()

    def delete_by_id(self, entity_id: int) -> bool:
        entity = self.get_by_id(entity_id)
        if entity is not None:
            self.delete(entity)
            return True
        return False
