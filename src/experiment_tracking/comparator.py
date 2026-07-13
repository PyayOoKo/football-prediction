"""
ExperimentComparator — compare experiments, runs, and models side-by-side.

Features:
- Compare multiple runs within an experiment
- Compare the same model type across experiments
- Rank runs by selected metric
- Identify best run for each metric
- Export comparison table as dict, DataFrame, or HTML
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.experiment_tracking.models import Experiment, Run

logger = logging.getLogger(__name__)


class ExperimentComparator:
    """Compare experiments, runs, and models.

    Parameters
    ----------
    session : Session
        SQLAlchemy ORM session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Run comparison ────────────────────────────────────

    def compare_runs(
        self,
        run_ids: list[str],
    ) -> dict[str, Any]:
        """Compare specific runs by their IDs.

        Returns a dict with per-run metrics, params, and a summary
        identifying the best run for each metric.

        Parameters
        ----------
        run_ids : list[str]
            Run IDs to compare.

        Returns
        -------
        dict[str, Any]
            ``{"best_by_metric": {...}, "runs": {...}}``
        """
        runs = []
        for rid in run_ids:
            run = self._session.get(Run, rid)
            if run is not None:
                runs.append(run)

        result = self._compare_run_objects(runs)
        return result

    def compare_runs_in_experiment(
        self,
        experiment_id: str,
        *,
        model_type: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Compare all completed runs within an experiment.

        Parameters
        ----------
        experiment_id : str
        model_type : str, optional
            Filter by model type.
        status : str, optional
            Filter by status (default ``completed``).

        Returns
        -------
        dict[str, Any]
        """
        stmt = select(Run).where(Run.experiment_id == experiment_id)

        if model_type is not None:
            stmt = stmt.where(Run.model_type == model_type)
        if status is not None:
            stmt = stmt.where(Run.status == status)
        else:
            stmt = stmt.where(Run.status == "completed")

        runs = list(self._session.execute(stmt).scalars().all())
        return self._compare_run_objects(runs)

    def compare_across_experiments(
        self,
        experiment_ids: list[str],
        *,
        model_type: str | None = None,
        metric: str | None = None,
    ) -> dict[str, Any]:
        """Compare runs across multiple experiments.

        Parameters
        ----------
        experiment_ids : list[str]
        model_type : str, optional
            Filter by model type.
        metric : str, optional
            Only include this metric in the comparison.

        Returns
        -------
        dict[str, Any]
        """
        stmt = select(Run).where(
            Run.experiment_id.in_(experiment_ids),
            Run.status == "completed",
        )
        if model_type is not None:
            stmt = stmt.where(Run.model_type == model_type)

        runs = list(self._session.execute(stmt).scalars().all())
        return self._compare_run_objects(runs, metric_filter=metric)

    def _compare_run_objects(
        self,
        runs: list[Run],
        metric_filter: str | None = None,
    ) -> dict[str, Any]:
        """Compare a list of Run objects, returning structured results.

        Parameters
        ----------
        runs : list[Run]
        metric_filter : str, optional

        Returns
        -------
        dict[str, Any]
        """
        if not runs:
            return {"best_by_metric": {}, "runs": {}}

        # Collect all metrics across all runs
        all_metrics: set[str] = set()
        run_data: dict[str, dict[str, Any]] = {}

        for run in runs:
            metrics = run.metrics or {}
            if metric_filter:
                metrics = {k: v for k, v in metrics.items() if k == metric_filter}
            all_metrics.update(metrics.keys())

            run_data[run.id] = {
                "id": run.id,
                "experiment_id": run.experiment_id,
                "experiment_name": self._get_experiment_name(run.experiment_id),
                "run_name": run.run_name,
                "model_type": run.model_type,
                "model_version": run.model_version,
                "status": run.status,
                "hyperparameters": run.hyperparameters or {},
                "metrics": metrics,
                "duration_seconds": run.training_duration_seconds,
                "random_seed": run.random_seed,
                "started_at": run.started_at.isoformat() if run.started_at else None,
            }

        # Determine best run for each metric
        best_by_metric: dict[str, dict[str, Any]] = {}
        for metric in sorted(all_metrics):
            best_run_id = None
            best_value = None
            lower_is_better = any(
                name in metric.lower()
                for name in ["loss", "error", "brier", "mse", "mae", "rmse"]
            )

            for run_id, data in run_data.items():
                val = data["metrics"].get(metric)
                if val is None:
                    continue
                if best_value is None:
                    best_value = val
                    best_run_id = run_id
                elif lower_is_better and val < best_value:
                    best_value = val
                    best_run_id = run_id
                elif not lower_is_better and val > best_value:
                    best_value = val
                    best_run_id = run_id

            if best_run_id is not None:
                best_by_metric[metric] = {
                    "run_id": best_run_id,
                    "value": best_value,
                    "model_type": run_data[best_run_id]["model_type"],
                    "run_name": run_data[best_run_id]["run_name"],
                }

        return {
            "best_by_metric": best_by_metric,
            "runs": run_data,
        }

    # ── Ranking ────────────────────────────────────────────

    def rank_by_metric(
        self,
        *,
        experiment_id: str | None = None,
        model_type: str | None = None,
        metric: str = "val_log_loss",
        ascending: bool | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Rank runs by a specific metric.

        Parameters
        ----------
        experiment_id : str, optional
        model_type : str, optional
        metric : str
            Metric to rank by.
        ascending : bool, optional
            Sort order. Auto-detected from metric name if not provided.
        limit : int

        Returns
        -------
        list[dict[str, Any]]
            Runs sorted by metric value, ranked 1..N.
        """
        if ascending is None:
            ascending = any(
                name in metric.lower()
                for name in ["loss", "error", "brier", "mse", "mae", "rmse"]
            )

        stmt = select(Run).where(Run.status == "completed")
        if experiment_id is not None:
            stmt = stmt.where(Run.experiment_id == experiment_id)
        if model_type is not None:
            stmt = stmt.where(Run.model_type == model_type)

        runs = list(self._session.execute(stmt).scalars().all())

        # Filter to runs that have the metric
        scored = [
            r for r in runs
            if r.metrics and metric in r.metrics and r.metrics[metric] is not None
        ]
        scored.sort(key=lambda r: r.metrics[metric], reverse=not ascending)

        results: list[dict[str, Any]] = []
        for i, run in enumerate(scored[:limit], 1):
            results.append({
                "rank": i,
                "run_id": run.id[:8],
                "experiment_id": run.experiment_id[:8],
                "run_name": run.run_name,
                "model_type": run.model_type,
                "metric_value": run.metrics[metric],
                "duration_seconds": run.training_duration_seconds,
            })

        return results

    # ── Export helpers ────────────────────────────────────

    def to_dataframe(
        self,
        comparison_result: dict[str, Any],
    ) -> Any:
        """Convert a comparison result to a pandas DataFrame.

        Parameters
        ----------
        comparison_result : dict
            Result from ``compare_runs()``, etc.

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd

        rows = []
        for run_id, data in comparison_result.get("runs", {}).items():
            row = {
                "run_id": data.get("id", run_id)[:8],
                "experiment": data.get("experiment_name", ""),
                "run_name": data.get("run_name", ""),
                "model_type": data.get("model_type", ""),
                "duration": data.get("duration_seconds"),
                "seed": data.get("random_seed"),
            }
            # Flatten metrics into columns
            for metric, value in data.get("metrics", {}).items():
                row[metric] = value
            rows.append(row)

        return pd.DataFrame(rows)

    def best_summary(self, comparison_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract a summary of the best run for each metric.

        Parameters
        ----------
        comparison_result : dict
            Result from ``compare_runs()``, etc.

        Returns
        -------
        list[dict[str, Any]]
        """
        return [
            {
                "metric": metric,
                "best_value": info["value"],
                "model_type": info["model_type"],
                "run_id": info["run_id"],
            }
            for metric, info in sorted(
                comparison_result.get("best_by_metric", {}).items(),
            )
        ]

    def _get_experiment_name(self, experiment_id: str) -> str:
        exp = self._session.get(Experiment, experiment_id)
        return exp.name if exp else "?"
