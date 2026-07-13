"""
ExperimentTracker — primary interface for experiment tracking.

Features:
- Create, list, and manage experiments
- Start, finish, fail individual training runs
- Log hyperparameters, metrics, and artifacts per run
- Resume interrupted runs
- Context manager support for safe run lifecycle
- Auto-capture hardware info, git commit, and timing
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.experiment_tracking.models import (
    BestModel,
    Experiment,
    ModelArtifact,
    Run,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════


def _capture_hardware() -> dict[str, Any]:
    """Capture a snapshot of the current hardware environment.

    Returns
    -------
    dict
        Keys: ``cpu``, ``cpu_count``, ``ram_gb``, ``platform``,
        ``python_version``, ``gpu``.
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
    }

    # CPU info (platform-dependent)
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu"] = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                info["cpu"] = result.stdout.strip()
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
                if len(lines) > 1:
                    info["cpu"] = lines[1]
    except Exception:
        info["cpu"] = platform.processor() or "unknown"

    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        info["ram_gb"] = 0.0

    # GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpus = [g.strip() for g in result.stdout.strip().split("\n") if g.strip()]
            info["gpu"] = gpus
    except Exception:
        info["gpu"] = []

    return info


def _capture_git_commit() -> str | None:
    """Capture the current git commit hash.

    Returns
    -------
    str or None
        The short (7-char) commit hash, or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
#  ExperimentTracker
# ═══════════════════════════════════════════════════════════


class ExperimentTracker:
    """Primary interface for tracking ML experiments and runs.

    Usage
    -----
    ::

        tracker = ExperimentTracker(session)

        # Create experiment
        exp = tracker.create_experiment("test_elo_v2")

        # Run training with context manager
        run = tracker.start_run(
            exp.id, model_type="xgboost",
            hyperparameters={"n_estimators": 100, "max_depth": 5},
            random_seed=42,
        )
        try:
            # ... train model ...
            tracker.finish_run(run.id, metrics={"val_log_loss": 0.58, "val_acc": 0.72})
        except Exception as exc:
            tracker.fail_run(run.id, error=str(exc))
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Experiment lifecycle ───────────────────────────────

    def create_experiment(
        self,
        name: str,
        *,
        description: str | None = None,
        dataset_version: str | None = None,
        feature_version: str | None = None,
        model_version: str | None = None,
        git_commit: str | None = None,
        notes: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Experiment:
        """Create a new experiment.

        Parameters
        ----------
        name : str
            Human-readable experiment name.
        description : str, optional
        dataset_version : str, optional
        feature_version : str, optional
        model_version : str, optional
        git_commit : str, optional
            Auto-captured from ``git rev-parse --short HEAD`` if not provided.
        notes : str, optional
        tags : dict, optional

        Returns
        -------
        Experiment
        """
        if not name or not name.strip():
            raise ValueError("Experiment name cannot be empty.")

        if git_commit is None:
            git_commit = _capture_git_commit()

        experiment = Experiment(
            name=name.strip(),
            description=description,
            dataset_version=dataset_version,
            feature_version=feature_version,
            model_version=model_version,
            git_commit=git_commit,
            notes=notes,
            tags=tags or {},
        )
        self._session.add(experiment)
        self._session.flush()
        logger.info("Created experiment: %s (id=%s)", name, experiment.id)
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        return self._session.get(Experiment, experiment_id)

    def list_experiments(
        self,
        *,
        tag_key: str | None = None,
        tag_value: str | None = None,
        limit: int = 50,
    ) -> list[Experiment]:
        """List experiments, optionally filtered by tag.

        Parameters
        ----------
        tag_key : str, optional
            Filter by tag key.
        tag_value : str, optional
            Filter by tag value (requires ``tag_key``).
        limit : int
            Max results (default 50).

        Returns
        -------
        list[Experiment]
        """
        stmt = select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)

        if tag_key is not None:
            # Filter by tag using JSON contains (PostgreSQL: -> operator)
            # For SQLite compatibility, we filter in Python
            experiments = list(self._session.execute(stmt).scalars().all())
            if tag_value is not None:
                return [
                    e for e in experiments
                    if e.tags and e.tags.get(tag_key) == tag_value
                ]
            return [
                e for e in experiments
                if e.tags and tag_key in e.tags
            ]

        return list(self._session.execute(stmt).scalars().all())

    def update_experiment(
        self,
        experiment_id: str,
        **updates: Any,
    ) -> Experiment:
        """Update experiment fields.

        Parameters
        ----------
        experiment_id : str
        **updates : Any
            Fields to update (e.g. ``notes``, ``tags``, ``description``).

        Returns
        -------
        Experiment
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id!r} not found.")

        for key, value in updates.items():
            if hasattr(experiment, key):
                setattr(experiment, key, value)

        self._session.flush()
        logger.info("Updated experiment %s: %s", experiment_id, set(updates.keys()))
        return experiment

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all associated runs/artifacts.

        Parameters
        ----------
        experiment_id : str

        Returns
        -------
        bool
            True if deleted.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return False
        self._session.delete(experiment)
        self._session.flush()
        logger.info("Deleted experiment %s", experiment_id)
        return True

    # ── Run lifecycle ─────────────────────────────────────

    def start_run(
        self,
        experiment_id: str,
        model_type: str,
        *,
        run_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        random_seed: int | None = None,
        git_commit: str | None = None,
        notes: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Run:
        """Start a new training run.

        Parameters
        ----------
        experiment_id : str
        model_type : str
        run_name : str, optional
        hyperparameters : dict, optional
        random_seed : int, optional
        git_commit : str, optional
            Auto-captured if not provided.
        notes : str, optional
        tags : dict, optional

        Returns
        -------
        Run
        """
        if git_commit is None:
            git_commit = _capture_git_commit()

        run = Run.create(
            experiment_id=experiment_id,
            model_type=model_type,
            run_name=run_name,
            hyperparameters=hyperparameters,
            random_seed=random_seed,
            git_commit=git_commit,
            notes=notes,
            tags=tags,
        )
        # Capture hardware on start
        run.hardware = _capture_hardware()

        self._session.add(run)
        self._session.flush()
        logger.info(
            "Started run %s: model=%s experiment=%s",
            run.id[:8], model_type, experiment_id[:8],
        )
        return run

    def finish_run(
        self,
        run_id: str,
        *,
        metrics: dict[str, float] | None = None,
        duration_seconds: float | None = None,
        notes: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Run:
        """Mark a run as completed with metrics and timing.

        Parameters
        ----------
        run_id : str
        metrics : dict, optional
            All computed metrics (e.g. ``val_log_loss``, ``test_accuracy``).
        duration_seconds : float, optional
            Auto-computed if not provided (elapsed from ``started_at``).
        notes : str, optional
        tags : dict, optional

        Returns
        -------
        Run
        """
        run = self._get_run(run_id)
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        if metrics is not None:
            run.metrics = metrics
        if duration_seconds is not None:
            run.training_duration_seconds = duration_seconds
        elif run.started_at:
            run.training_duration_seconds = (
                datetime.now(timezone.utc) - run.started_at
            ).total_seconds()
        if notes:
            run.notes = (run.notes or "") + ("\n" + notes if run.notes else notes)
        if tags:
            run.tags = {**(run.tags or {}), **tags}

        self._session.flush()
        logger.info(
            "Finished run %s: metrics=%s duration=%.2fs",
            run_id[:8], metrics, run.training_duration_seconds or 0,
        )
        return run

    def fail_run(
        self,
        run_id: str,
        error: str,
        *,
        duration_seconds: float | None = None,
    ) -> Run:
        """Mark a run as failed with an error message.

        Parameters
        ----------
        run_id : str
        error : str
            Error description or traceback.
        duration_seconds : float, optional
            Auto-computed if not provided.

        Returns
        -------
        Run
        """
        run = self._get_run(run_id)
        run.status = "failed"
        run.error_message = error
        run.finished_at = datetime.now(timezone.utc)
        if duration_seconds is not None:
            run.training_duration_seconds = duration_seconds
        elif run.started_at:
            run.training_duration_seconds = (
                datetime.now(timezone.utc) - run.started_at
            ).total_seconds()

        self._session.flush()
        logger.warning(
            "Failed run %s: %s (duration=%.2fs)",
            run_id[:8], error, run.training_duration_seconds or 0,
        )
        return run

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        return self._session.get(Run, run_id)

    def list_runs(
        self,
        *,
        experiment_id: str | None = None,
        model_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Run]:
        """List runs with optional filters.

        Parameters
        ----------
        experiment_id : str, optional
        model_type : str, optional
        status : str, optional
        limit : int

        Returns
        -------
        list[Run]
        """
        stmt = select(Run).order_by(Run.started_at.desc()).limit(limit)

        if experiment_id is not None:
            stmt = stmt.where(Run.experiment_id == experiment_id)
        if model_type is not None:
            stmt = stmt.where(Run.model_type == model_type)
        if status is not None:
            stmt = stmt.where(Run.status == status)

        return list(self._session.execute(stmt).scalars().all())

    def resume_run(self, run_id: str) -> Run:
        """Resume a run that was previously started but not finished.

        Resets the run's ``status`` to ``running`` and ``started_at``
        to now, preserving all other metadata.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        Run

        Raises
        ------
        ValueError
            If the run is already completed.
        """
        run = self._get_run(run_id)
        if run.status == "completed":
            raise ValueError(
                f"Cannot resume completed run {run_id[:8]}. "
                f"Create a new run instead."
            )
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        run.finished_at = None
        run.error_message = None
        self._session.flush()
        logger.info("Resumed run %s", run_id[:8])
        return run

    # ── Artifacts ─────────────────────────────────────────

    def log_artifact(
        self,
        run_id: str,
        name: str,
        uri: str,
        *,
        file_size_bytes: int | None = None,
        artifact_type: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifact:
        """Log a model artifact for a run.

        Parameters
        ----------
        run_id : str
        name : str
            Artifact name (e.g. ``model.joblib``).
        uri : str
            Path or URI to the artifact.
        file_size_bytes : int, optional
        artifact_type : str
            ``model``, ``preprocessor``, ``encoder``, ``report``, etc.
        metadata : dict, optional

        Returns
        -------
        ModelArtifact
        """
        run = self._get_run(run_id)
        artifact = ModelArtifact(
            run_id=run.id,
            name=name,
            uri=uri,
            file_size_bytes=file_size_bytes,
            artifact_type=artifact_type,
            extra_metadata=metadata or {},
        )
        self._session.add(artifact)
        self._session.flush()
        logger.info(
            "Logged artifact %s for run %s (type=%s)",
            name, run_id[:8], artifact_type,
        )
        return artifact

    def list_artifacts(self, run_id: str) -> list[ModelArtifact]:
        """List all artifacts for a run."""
        run = self._get_run(run_id)
        return run.artifacts if run.artifacts else []

    # ── Context manager for runs ──────────────────────────

    def run(
        self,
        experiment_id: str,
        model_type: str,
        *,
        run_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        random_seed: int | None = None,
        notes: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> _RunContext:
        """Context manager for a safe run lifecycle.

        Usage
        -----
        ::

            with tracker.run(exp.id, "xgboost", hyperparameters=params) as run:
                model = train_model(...)
                tracker.finish_run(run.id, metrics=metrics)
        """
        return _RunContext(
            tracker=self,
            experiment_id=experiment_id,
            model_type=model_type,
            run_name=run_name,
            hyperparameters=hyperparameters,
            random_seed=random_seed,
            notes=notes,
            tags=tags,
        )

    # ── Extended metrics ─────────────────────────────────

    def update_extended_metrics(
        self,
        run_id: str,
        *,
        cross_validation_metrics: dict[str, dict[str, float]] | None = None,
        calibration_metrics: dict[str, float] | None = None,
        profit_metrics: dict[str, float] | None = None,
        confusion_matrix: dict[str, int] | None = None,
        feature_importance: dict[str, float] | None = None,
        shap_values: dict[str, Any] | None = None,
    ) -> Run:
        """Update extended metrics for a completed run.

        Parameters
        ----------
        run_id : str
        cross_validation_metrics : dict, optional
            Per-fold CV metrics: ``{"fold_1": {"val_log_loss": 0.58}, ...}``.
        calibration_metrics : dict, optional
            Calibration metrics: ``{"expected_calibration_error": 0.02, "brier_score": 0.15}``.
        profit_metrics : dict, optional
            Betting profit metrics: ``{"roi": 0.15, "yield": 0.12, "clv": 25.50}``.
        confusion_matrix : dict, optional
            ``{"tp": 45, "fp": 8, "tn": 120, "fn": 12}``.
        feature_importance : dict, optional
            ``{"elo_rating": 0.25, "home_advantage": 0.18}``.
        shap_values : dict, optional
            SHAP summary: ``{"mean_abs_shap": {"elo_rating": 0.15}}``.

        Returns
        -------
        Run
        """
        run = self._get_run(run_id)
        if cross_validation_metrics is not None:
            run.cross_validation_metrics = cross_validation_metrics
        if calibration_metrics is not None:
            run.calibration_metrics = calibration_metrics
        if profit_metrics is not None:
            run.profit_metrics = profit_metrics
        if confusion_matrix is not None:
            run.confusion_matrix = confusion_matrix
        if feature_importance is not None:
            run.feature_importance = feature_importance
        if shap_values is not None:
            run.shap_values = shap_values
        self._session.flush()
        logger.info(
            "Updated extended metrics for run %s (CV=%s, Profit=%s, FI=%s)",
            run_id[:8],
            "yes" if cross_validation_metrics else "no",
            "yes" if profit_metrics else "no",
            "yes" if feature_importance else "no",
        )
        return run

    def _get_run(self, run_id: str) -> Run:
        run = self._session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run {run_id!r} not found.")
        return run


# ═══════════════════════════════════════════════════════════
#  Context manager helper
# ═══════════════════════════════════════════════════════════


class _RunContext:
    """Context manager that ensures run lifecycle is handled safely."""

    def __init__(
        self,
        tracker: ExperimentTracker,
        experiment_id: str,
        model_type: str,
        run_name: str | None,
        hyperparameters: dict[str, Any] | None,
        random_seed: int | None,
        notes: str | None,
        tags: dict[str, str] | None,
    ) -> None:
        self._tracker = tracker
        self._experiment_id = experiment_id
        self._model_type = model_type
        self._run_name = run_name
        self._hyperparameters = hyperparameters
        self._random_seed = random_seed
        self._notes = notes
        self._tags = tags
        self.run: Run | None = None

    def __enter__(self) -> Run:
        self.run = self._tracker.start_run(
            self._experiment_id,
            self._model_type,
            run_name=self._run_name,
            hyperparameters=self._hyperparameters,
            random_seed=self._random_seed,
            notes=self._notes,
            tags=self._tags,
        )
        return self.run

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.run is None:
            return
        if exc_type is not None:
            try:
                self._tracker.fail_run(
                    self.run.id,
                    error=f"{exc_type.__name__}: {exc_val}",
                )
            except Exception:
                pass
