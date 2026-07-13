"""
Storage stage — persists transformed data to databases or files.

Supports:
- PostgreSQL via SQLAlchemy (upsert with on_conflict_do_nothing)
- CSV files
- Parquet files

All stores inherit from ``DataStore`` and implement ``write()``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.database.session import get_session
from src.etl.models import PipelineStage, StageResult, StageStatus
from src.etl.progress import ProgressReporter

logger = logging.getLogger(__name__)


class DataStore(ABC):
    """Abstract base for storage backends."""

    @abstractmethod
    def write(
        self,
        data: list[dict[str, Any]],
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> StageResult:
        """Persist data to the target.

        Parameters
        ----------
        data : list[dict]
            Data to store.
        batch_size : int
            Rows per batch (default 1000).
        **kwargs
            Store-specific options.

        Returns
        -------
        StageResult
        """
        ...


class FileStore(DataStore):
    """Write data to CSV or Parquet files.

    Parameters
    ----------
    output_dir : str | Path
        Directory for output files.
    format : str
        Output format: ``csv`` or ``parquet`` (default ``csv``).
    filename : str, optional
        Output filename. Auto-generated if not provided.
    """

    def __init__(
        self,
        output_dir: str | Path,
        format: str = "csv",
        filename: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()
        self.filename = filename

    def write(
        self,
        data: list[dict[str, Any]],
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> StageResult:
        stage = PipelineStage.STORE
        result = StageResult(stage=stage, status=StageStatus.RUNNING)
        start = time.perf_counter()

        result.records_in = len(data)

        if not data:
            result.data = data
            result.records_out = 0
            result.status = StageStatus.WARNING
            result.errors.append("No data to write")
            result.duration_seconds = time.perf_counter() - start
            return result

        try:
            self.filename = kwargs.get("filename", self.filename)
            filepath = self._resolve_path(kwargs.get("source", "etl_output"))

            df = pd.DataFrame(data)

            if self.format == "csv":
                df.to_csv(filepath, index=False)
            elif self.format == "parquet":
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {self.format}")

            result.data = data
            result.records_out = len(data)
            result.metrics["file_size_mb"] = round(
                filepath.stat().st_size / (1024 * 1024), 2
            )
            result.status = StageStatus.SUCCESS

            logger.info(
                "Wrote %d rows to %s (%.1f MB)",
                len(data),
                filepath,
                result.metrics["file_size_mb"],
            )

        except Exception as exc:
            logger.exception("File write failed: %s", exc)
            result.status = StageStatus.FAILED
            result.errors.append(str(exc))

        result.duration_seconds = time.perf_counter() - start
        return result

    def _resolve_path(self, source: str) -> Path:
        """Generate output file path."""
        if self.filename:
            return self.output_dir / self.filename
        import datetime
        import hashlib
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(source.encode()).hexdigest()[:8]
        return self.output_dir / f"{source}_{ts}_{h}.{self.format}"


class DatabaseStore(DataStore):
    """Upsert data into a database table via SQLAlchemy.

    Uses PostgreSQL ``INSERT ... ON CONFLICT DO NOTHING`` when
    ``unique_columns`` are provided, otherwise falls back to a
    plain parameterised ``INSERT`` — never raw SQL string formatting.

    Parameters
    ----------
    model_class : type
        SQLAlchemy ORM model class.
    unique_columns : list[str]
        Columns that define a unique constraint (for upsert).
    batch_size : int
        Rows per batch commit (default 1000).
    """

    def __init__(
        self,
        model_class: type,
        unique_columns: list[str] | None = None,
        batch_size: int = 1000,
    ) -> None:
        self.model_class = model_class
        self.unique_columns = unique_columns or []
        self.batch_size = batch_size
        self._progress = ProgressReporter()

    def write(
        self,
        data: list[dict[str, Any]],
        batch_size: int = 0,
        **kwargs: Any,
    ) -> StageResult:
        stage = PipelineStage.STORE
        result = StageResult(stage=stage, status=StageStatus.RUNNING)
        start = time.perf_counter()

        result.records_in = len(data)

        if not data:
            result.status = StageStatus.WARNING
            result.errors.append("No data to store")
            result.duration_seconds = time.perf_counter() - start
            return result

        bs = batch_size or self.batch_size
        total = len(data)
        inserted = 0
        updated = 0
        errors = 0

        try:
            with get_session() as session:
                batches = [data[i:i + bs] for i in range(0, total, bs)]

                for batch_idx, batch in enumerate(batches):
                    try:
                        cnt = self._insert_batch(session, batch)
                        inserted += cnt
                    except Exception as exc:
                        errors += len(batch)
                        logger.error("Batch %d failed: %s", batch_idx, exc)
                        result.errors.append(f"Batch {batch_idx}: {exc}")
                        session.rollback()
                        continue

                    logger.info(
                        "Batch %d/%d: %d rows stored",
                        batch_idx + 1,
                        len(batches),
                        len(batch),
                    )

            result.data = data
            result.records_out = inserted
            result.metrics["inserted"] = float(inserted)
            result.metrics["updated"] = float(updated)
            result.metrics["errors"] = float(errors)

            if errors > 0:
                result.status = (
                    StageStatus.WARNING
                    if result.records_out > 0
                    else StageStatus.FAILED
                )
            else:
                result.status = StageStatus.SUCCESS

        except Exception as exc:
            logger.exception("Database store failed: %s", exc)
            result.status = StageStatus.FAILED
            result.errors.append(str(exc))

        result.duration_seconds = time.perf_counter() - start
        logger.info(
            "DB store: %d inserted, %d errors in %.1fs",
            inserted,
            errors,
            result.duration_seconds,
        )
        return result

    def _insert_batch(
        self, session: Session, batch: list[dict[str, Any]]
    ) -> int:
        """Insert or do-nothing on conflict using parameterised SQL."""
        table = self.model_class.__table__

        if self.unique_columns:
            stmt = pg_insert(self.model_class).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=self.unique_columns
            )
        else:
            # Parameterised insert via SQLAlchemy core — no string formatting
            stmt = table.insert().values(batch)

        session.execute(stmt)
        session.flush()
        return len(batch)
