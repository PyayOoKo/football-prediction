"""
ModelRegistry — best-performing model tracking and leaderboards.

Features:
- Register a run as best for a given metric
- Automatic ranking (insert with rank, re-rank others)
- Query leaderboards by metric, experiment, or model type
- Promote/demote models for production
- History of promotions
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.experiment_tracking.models import BestModel, Experiment, Run

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Best model registry with leaderboard and promotion support.

    Parameters
    ----------
    session : Session
        SQLAlchemy ORM session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Registration ──────────────────────────────────────

    def register(
        self,
        experiment_id: str,
        run_id: str,
        metric_name: str,
        metric_value: float,
        *,
        rank: int | None = None,
        notes: str | None = None,
    ) -> BestModel:
        """Register a run as best for a given metric.

        If ``rank`` is not specified, the model is inserted at the
        correct rank based on ``metric_value`` (lower is better for
        loss metrics like log_loss; higher is better for accuracy/F1).

        Parameters
        ----------
        experiment_id : str
        run_id : str
        metric_name : str
            e.g. ``val_log_loss``, ``test_accuracy``, ``f1_score``.
        metric_value : float
        rank : int, optional
            Explicit rank. Auto-computed if not provided.
        notes : str, optional

        Returns
        -------
        BestModel
        """
        # Determine sort direction based on metric name
        lower_is_better = any(
            name in metric_name.lower()
            for name in ["loss", "error", "brier", "mse", "mae", "rmse"]
        )

        if rank is None:
            rank = self._compute_rank(
                metric_name, metric_value, lower_is_better,
            )

        # Check for existing entry at this rank
        existing = self._get_by_rank(experiment_id, metric_name, rank)
        if existing is not None:
            logger.warning(
                "Replacing existing best model at rank %d for metric %s "
                "(experiment %s)",
                rank, metric_name, experiment_id[:8],
            )
            self._session.delete(existing)
            self._session.flush()

        entry = BestModel(
            experiment_id=experiment_id,
            run_id=run_id,
            metric_name=metric_name,
            metric_value=metric_value,
            rank=rank,
            notes=notes,
        )
        self._session.add(entry)
        self._session.flush()
        logger.info(
            "Registered best model: rank=%d metric=%s value=%.4f run=%s",
            rank, metric_name, metric_value, run_id[:8],
        )
        return entry

    def _compute_rank(
        self,
        metric_name: str,
        metric_value: float,
        lower_is_better: bool,
    ) -> int:
        """Compute insertion rank based on existing entries.

        Finds all existing entries for the same metric, determines
        where the new value would slot in, and shifts existing
        entries down if necessary.
        """
        existing_entries = self.get_leaderboard(
            metric_name=metric_name,
            limit=1000,  # All entries for ranking
        )

        if not existing_entries:
            return 1

        # Extract existing values and find insertion point
        existing_values = [(e.metric_value, e.rank) for e in existing_entries]

        if lower_is_better:
            existing_values.sort(key=lambda x: x[0])  # Ascending
            for i, (val, _) in enumerate(existing_values):
                if metric_value < val:
                    new_rank = i + 1
                    break
            else:
                new_rank = len(existing_values) + 1
        else:
            existing_values.sort(key=lambda x: -x[0])  # Descending
            for i, (val, _) in enumerate(existing_values):
                if metric_value > val:
                    new_rank = i + 1
                    break
            else:
                new_rank = len(existing_values) + 1

        # Shift ranks of existing entries that are now lower
        self._shift_ranks(
            metric_name, new_rank, lower_is_better,
        )

        return new_rank

    def _shift_ranks(
        self,
        metric_name: str,
        insert_rank: int,
        lower_is_better: bool,
    ) -> None:
        """Shift existing entries down by one rank starting from insert_rank."""
        stmt = select(BestModel).where(
            BestModel.metric_name == metric_name,
            BestModel.rank >= insert_rank,
        )
        entries = self._session.execute(stmt).scalars().all()
        for entry in entries:
            entry.rank += 1
        if entries:
            self._session.flush()

    def _get_by_rank(
        self,
        experiment_id: str,
        metric_name: str,
        rank: int,
    ) -> BestModel | None:
        stmt = select(BestModel).where(
            BestModel.experiment_id == experiment_id,
            BestModel.metric_name == metric_name,
            BestModel.rank == rank,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    # ── Leaderboard ───────────────────────────────────────

    def get_leaderboard(
        self,
        *,
        metric_name: str | None = None,
        experiment_id: str | None = None,
        model_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[BestModel]:
        """Get the leaderboard with optional filters.

        Parameters
        ----------
        metric_name : str, optional
            Filter by metric.
        experiment_id : str, optional
            Filter by experiment.
        model_type : str, optional
            Filter by model type (joins Run).
        limit : int
            Max results (default 20).
        offset : int
            Pagination offset.

        Returns
        -------
        list[BestModel]
        """
        stmt = (
            select(BestModel)
            .order_by(BestModel.metric_name, BestModel.rank, BestModel.metric_value)
            .limit(limit)
            .offset(offset)
        )

        if metric_name is not None:
            stmt = stmt.where(BestModel.metric_name == metric_name)
        if experiment_id is not None:
            stmt = stmt.where(BestModel.experiment_id == experiment_id)
        if model_type is not None:
            # Join with Run to filter by model_type
            stmt = stmt.join(Run).where(Run.model_type == model_type)

        return list(self._session.execute(stmt).scalars().all())

    def get_best(
        self,
        metric_name: str,
        *,
        experiment_id: str | None = None,
        model_type: str | None = None,
    ) -> BestModel | None:
        """Get the single best model for a given metric.

        Parameters
        ----------
        metric_name : str
        experiment_id : str, optional
        model_type : str, optional

        Returns
        -------
        BestModel | None
        """
        results = self.get_leaderboard(
            metric_name=metric_name,
            experiment_id=experiment_id,
            model_type=model_type,
            limit=1,
        )
        return results[0] if results else None

    # ── Promotion ─────────────────────────────────────────

    def promote(
        self,
        entry_id: str,
        notes: str | None = None,
    ) -> BestModel:
        """Mark a best model entry as promoted to production.

        Parameters
        ----------
        entry_id : str
            BestModel entry ID.
        notes : str, optional

        Returns
        -------
        BestModel
        """
        entry = self._session.get(BestModel, entry_id)
        if entry is None:
            raise ValueError(f"BestModel entry {entry_id!r} not found.")

        entry.is_promoted = True
        entry.promoted_at = datetime.now(timezone.utc)
        if notes:
            entry.notes = (entry.notes or "") + ("\n" + notes if entry.notes else notes)
        self._session.flush()
        logger.info(
            "Promoted best model %s to production (metric=%s, value=%.4f)",
            entry_id[:8], entry.metric_name, entry.metric_value,
        )
        return entry

    def demote(self, entry_id: str) -> BestModel:
        """Remove promotion status from a best model entry.

        Parameters
        ----------
        entry_id : str

        Returns
        -------
        BestModel
        """
        entry = self._session.get(BestModel, entry_id)
        if entry is None:
            raise ValueError(f"BestModel entry {entry_id!r} not found.")

        entry.is_promoted = False
        entry.promoted_at = None
        self._session.flush()
        logger.info("Demoted best model %s", entry_id[:8])
        return entry

    def get_promoted(self) -> list[BestModel]:
        """Get all currently promoted models."""
        stmt = (
            select(BestModel)
            .where(BestModel.is_promoted == True)
            .order_by(BestModel.metric_name)
        )
        return list(self._session.execute(stmt).scalars().all())

    # ── Convenience ───────────────────────────────────────

    def to_dataframe(
        self,
        *,
        metric_name: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> Any:
        """Get the leaderboard as a pandas DataFrame.

        Parameters
        ----------
        metric_name : str, optional
        experiment_id : str, optional
        limit : int

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd

        entries = self.get_leaderboard(
            metric_name=metric_name,
            experiment_id=experiment_id,
            limit=limit,
        )
        rows = []
        for e in entries:
            run = self._session.get(Run, e.run_id)
            rows.append({
                "rank": e.rank,
                "metric_name": e.metric_name,
                "metric_value": e.metric_value,
                "model_type": run.model_type if run else "?",
                "run_id": e.run_id[:8],
                "experiment_id": e.experiment_id[:8],
                "is_promoted": e.is_promoted,
                "promoted_at": e.promoted_at,
                "notes": e.notes,
            })
        return pd.DataFrame(rows)
