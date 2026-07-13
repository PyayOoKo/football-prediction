"""
SQLAlchemy ORM models for ML experiment tracking.

Tables
------
experiments
    High-level experiment grouping (dataset, features, notes, git commit).
runs
    Individual training runs with params, metrics, duration, hardware info.
best_models
    Registry of best-performing models across experiments.
model_artifacts
    File paths to saved model files (supports multiple versions).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.config.settings import config
from src.database.base import Base


# ── Helper: generate string UUID PK ───────────────────────
def _uuid_pk() -> Any:
    return mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))


# ═══════════════════════════════════════════════════════════
#  Experiment
# ═══════════════════════════════════════════════════════════


class Experiment(Base):
    """
    A high-level experiment that groups related training runs.

    Each experiment represents a hypothesis or investigation — e.g.
    "test Elo features v2" or "compare LR vs XGBoost on 2025 data".
    All runs within an experiment share the same dataset and feature
    versions but may vary model types, hyperparameters, and seeds.

    Columns
    -------
    id : str (UUID)
        Primary key.
    name : str
        Human-readable experiment name.
    description : str, optional
        Detailed description of the experiment's goals.
    dataset_version : str, optional
        Version identifier for the dataset used.
    feature_version : str, optional
        Version of the feature store / feature engineering pipeline.
    model_version : str, optional
        Version of the model code / architecture.
    git_commit : str, optional
        Git commit hash at experiment start.
    notes : str, optional
        Free-form notes about the experiment.
    tags : dict, optional
        Arbitrary key-value tags for filtering and search.
    created_at : datetime
        When the experiment was created.
    updated_at : datetime
        When the experiment was last updated.
    """

    __tablename__ = "experiments"

    id: Mapped[str] = _uuid_pk()
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True,
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    dataset_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
    )
    feature_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
    )
    model_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
    )
    git_commit: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    tags: Mapped[dict[str, str] | None] = mapped_column(
        JSON, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=func.now(),
    )

    # Relationships
    runs: Mapped[list[Run]] = relationship(
        "Run", back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="Run.started_at.desc()",
    )
    best_models: Mapped[list[BestModel]] = relationship(
        "BestModel", back_populates="experiment",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Experiment {self.name!r} ({len(self.runs)} runs)>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "git_commit": self.git_commit,
            "notes": self.notes,
            "tags": self.tags or {},
            "run_count": len(self.runs) if self.runs else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ═══════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════


class Run(Base):
    """
    A single model training run with full metadata.

    Records everything needed to reproduce a training run:
    hyperparameters, random seed, all computed metrics, training
    duration, hardware information, and a pointer to the saved
    model file.

    Columns
    -------
    id : str (UUID)
        Primary key.
    experiment_id : str
        FK to ``experiments``.
    run_name : str, optional
        Short name for this run.
    model_type : str
        The model algorithm (xgboost, logistic_regression, etc.).
    model_version : str, optional
        Version of the model code.
    hyperparameters : dict
        Full hyperparameter dictionary (JSON).
    random_seed : int, optional
        Random seed used for reproducibility.
    status : str
        Current status: ``running``, ``completed``, ``failed``.
    metrics : dict, optional
        All computed metrics (JSON) — log_loss, accuracy, f1, etc.
    cross_validation_metrics : dict, optional
        Per-fold CV metrics (JSON) — ``{"fold_1": {"val_log_loss": 0.58, ...}, ...}``.
    calibration_metrics : dict, optional
        Calibration metrics — ``{"expected_calibration_error": 0.02, "brier_score": 0.15}``.
    profit_metrics : dict, optional
        Betting profit metrics — ``{"roi": 0.15, "yield": 0.12, "clv": 25.50}``.
    confusion_matrix : dict, optional
        Confusion matrix — ``{"tp": 45, "fp": 8, "tn": 120, "fn": 12}``.
    feature_importance : dict, optional
        Feature importance scores — ``{"elo_rating": 0.25, "home_advantage": 0.18}``.
    shap_values : dict, optional
        SHAP summary — ``{"mean_abs_shap": {"elo_rating": 0.15, ...}, ...}``.
    training_duration_seconds : float, optional
        Wall-clock training time.
    hardware : dict, optional
        Hardware info snapshot (CPU, RAM, GPU, OS).
    git_commit : str, optional
        Git commit hash at run start.
    notes : str, optional
        Free-form notes about this run.
    tags : dict, optional
        Arbitrary key-value tags.
    error_message : str, optional
        Error details if the run failed.
    started_at : datetime
        When this run started.
    finished_at : datetime, optional
        When this run finished.
    created_at : datetime
        Row creation timestamp.
    """

    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint("experiment_id", "run_name", name="uq_run_name_per_experiment"),
    )

    id: Mapped[str] = _uuid_pk()
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    run_name: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    model_type: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True,
    )
    model_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
    )
    hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    random_seed: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="running",
    )
    metrics: Mapped[dict[str, float] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    cross_validation_metrics: Mapped[dict[str, dict[str, float]] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    calibration_metrics: Mapped[dict[str, float] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    profit_metrics: Mapped[dict[str, float] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    confusion_matrix: Mapped[dict[str, int] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    feature_importance: Mapped[dict[str, float] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    shap_values: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    training_duration_seconds: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    hardware: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True, default=dict,
    )
    git_commit: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    tags: Mapped[dict[str, str] | None] = mapped_column(
        JSON, nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    # Relationships
    experiment: Mapped[Experiment] = relationship(
        "Experiment", back_populates="runs",
    )
    artifacts: Mapped[list[ModelArtifact]] = relationship(
        "ModelArtifact", back_populates="run",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Run {self.model_type} status={self.status} "
            f"experiment={self.experiment_id[:8]}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "run_name": self.run_name,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "hyperparameters": self.hyperparameters or {},
            "random_seed": self.random_seed,
            "status": self.status,
            "metrics": self.metrics or {},
            "cross_validation_metrics": self.cross_validation_metrics or {},
            "calibration_metrics": self.calibration_metrics or {},
            "profit_metrics": self.profit_metrics or {},
            "confusion_matrix": self.confusion_matrix or {},
            "feature_importance": self.feature_importance or {},
            "shap_values": self.shap_values or {},
            "training_duration_seconds": self.training_duration_seconds,
            "hardware": self.hardware or {},
            "git_commit": self.git_commit,
            "notes": self.notes,
            "tags": self.tags or {},
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def create(
        cls,
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
        """Create a new Run in 'running' status."""
        return cls(
            experiment_id=experiment_id,
            run_name=run_name,
            model_type=model_type,
            hyperparameters=hyperparameters or {},
            random_seed=random_seed,
            status="running",
            git_commit=git_commit,
            notes=notes,
            tags=tags or {},
        )


# ═══════════════════════════════════════════════════════════
#  BestModel
# ═══════════════════════════════════════════════════════════


class BestModel(Base):
    """
    Registry of the best-performing models across experiments.

    Each row links a specific run to its rank for a given metric.
    Supports multiple metric-based leaderboards (e.g. best by
    log_loss, best by accuracy, best by ROC-AUC).

    Columns
    -------
    id : str (UUID)
        Primary key.
    experiment_id : str
        FK to ``experiments``.
    run_id : str
        FK to ``runs``.
    metric_name : str
        Which metric this ranking is based on (e.g. ``val_log_loss``).
    metric_value : float
        The actual metric value.
    rank : int
        Rank (1 = best).
    is_promoted : bool
        Whether this model has been promoted to production.
    promoted_at : datetime, optional
        When the model was promoted.
    notes : str, optional
        Notes about why this model was selected.
    created_at : datetime
        Row creation timestamp.
    """

    __tablename__ = "best_models"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id", "metric_name", "rank",
            name="uq_best_model_rank",
        ),
    )

    id: Mapped[str] = _uuid_pk()
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )
    metric_value: Mapped[float] = mapped_column(
        Float, nullable=False,
    )
    rank: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
    )
    is_promoted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    # Relationships
    experiment: Mapped[Experiment] = relationship(
        "Experiment", back_populates="best_models",
    )
    run: Mapped[Run] = relationship("Run")

    def __repr__(self) -> str:
        return (
            f"<BestModel rank={self.rank} metric={self.metric_name} "
            f"value={self.metric_value:.4f}>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "rank": self.rank,
            "is_promoted": self.is_promoted,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ═══════════════════════════════════════════════════════════
#  ModelArtifact
# ═══════════════════════════════════════════════════════════


class ModelArtifact(Base):
    """
    A saved model file associated with a training run.

    Supports multiple artifacts per run (e.g. model.joblib,
    preprocessor.pkl, encoder.pkl).

    Columns
    -------
    id : str (UUID)
        Primary key.
    run_id : str
        FK to ``runs``.
    name : str
        Artifact name (e.g. ``model.joblib``, ``preprocessor.pkl``).
    uri : str
        URI or path to the artifact (file path, S3 URI, etc.).
    file_size_bytes : int, optional
        File size in bytes.
    artifact_type : str
        Type hint (e.g. ``model``, ``preprocessor``, ``encoder``, ``report``).
    metadata : dict, optional
        Arbitrary metadata about this artifact.
    created_at : datetime
        Row creation timestamp.
    """

    __tablename__ = "model_artifacts"

    id: Mapped[str] = _uuid_pk()
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    uri: Mapped[str] = mapped_column(
        String(1024), nullable=False,
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    artifact_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="model",
    )
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True, default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
    )

    # Relationships
    run: Mapped[Run] = relationship(
        "Run", back_populates="artifacts",
    )

    def __repr__(self) -> str:
        return f"<ModelArtifact {self.name!r} type={self.artifact_type}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "name": self.name,
            "uri": self.uri,
            "file_size_bytes": self.file_size_bytes,
            "artifact_type": self.artifact_type,
            "metadata": self.extra_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
