"""
ML Experiment Management Platform — track, compare, export, and serve
ML experiments via CLI, REST API, and third-party integrations.

Track every training run with full metadata: hyperparameters, metrics,
hardware, git commit, random seeds, cross-validation metrics,
calibration metrics, profit metrics (ROI/Yield/CLV), confusion matrices,
feature importance, and SHAP values.

Compare runs side-by-side, maintain best-model leaderboards, export to
JSON/CSV/HTML with Plotly charts, and integrate with MLflow, Weights &
Biases, and TensorBoard.

Quick start
-----------
::

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.experiment_tracking import ExperimentTracker, ModelRegistry

    engine = create_engine("sqlite:///experiments.db")
    from src.experiment_tracking.models import Base
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        tracker = ExperimentTracker(session)
        exp = tracker.create_experiment("test_elo_v2")
        run = tracker.start_run(exp.id, model_type="xgboost")
        # ... train model ...
        tracker.finish_run(run.id, metrics={"val_log_loss": 0.58})
        # Also log extended metrics
        tracker.update_extended_metrics(run.id,
            cross_validation_metrics={"fold_1": {"val_log_loss": 0.58}},
            profit_metrics={"roi": 0.15, "yield": 0.12, "clv": 25.50},
            confusion_matrix={"tp": 45, "fp": 8, "tn": 120, "fn": 12},
            feature_importance={"elo_rating": 0.25},
        )

    # REST API
    # uvicorn src.experiment_tracking.api:app --reload

    # Export to MLflow
    from src.experiment_tracking.integrations import export_to_mlflow
    export_to_mlflow(session, tracking_uri="http://localhost:5000")

Package layout
--------------
models.py
    SQLAlchemy ORM models: Experiment, Run (18+ metric fields), BestModel, ModelArtifact.
tracker.py
    ExperimentTracker — experiment and run lifecycle management.
registry.py
    ModelRegistry — best-model leaderboard, auto-rank, promotion.
comparator.py
    ExperimentComparator — compare runs and experiments side-by-side.
export.py
    Export to JSON, CSV, and self-contained HTML with Plotly charts.
cli.py
    CLI with 10 subcommands (list, show, compare, leaderboard, export,
    promote, api, mlflow, wandb, tensorboard).
api.py
    FastAPI REST API (20+ endpoints for experiments, runs, artifacts,
    leaderboards, comparison, and export).
integrations/
    mlflow_adapter.py — export/import with MLflow Tracking Server.
    wandb_adapter.py — export/import with Weights & Biases.
    tensorboard_adapter.py — export to TensorBoard event files.
"""

from __future__ import annotations

from src.experiment_tracking.comparator import ExperimentComparator
from src.experiment_tracking.export import export_csv, export_html, export_json
from src.experiment_tracking.models import (
    BestModel,
    Experiment,
    ModelArtifact,
    Run,
)
from src.experiment_tracking.registry import ModelRegistry
from src.experiment_tracking.tracker import ExperimentTracker

__all__ = [
    # Models
    "Experiment",
    "Run",
    "BestModel",
    "ModelArtifact",
    # Core services
    "ExperimentTracker",
    "ModelRegistry",
    "ExperimentComparator",
    # Export
    "export_json",
    "export_csv",
    "export_html",
]
