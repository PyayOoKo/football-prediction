"""
FastAPI REST API for experiment tracking.

Provides CRUD endpoints for experiments, runs, best models, artifacts,
leaderboards, and comparisons. Includes filtering, pagination, and export.

Run with::

    uvicorn src.experiment_tracking.api:app --reload

Or via the CLI::

    python -m src.experiment_tracking.cli api --port 8000
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.experiment_tracking.comparator import ExperimentComparator
from src.experiment_tracking.export import export_html, export_json
from src.experiment_tracking.models import Base
from src.experiment_tracking.registry import ModelRegistry
from src.experiment_tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

# ── OpenAPI metadata ────────────────────────────────────

app = FastAPI(
    title="Experiment Tracking API",
    description="REST API for ML experiment management — track runs, metrics, models, and artifacts.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Database setup (lazy) ───────────────────────────────

_engine = None


def _get_db_url() -> str:
    import os
    return os.environ.get(
        "EXPERIMENT_DB_URL",
        os.environ.get("DATABASE_URL", "sqlite:///experiments.db"),
    )


def _get_engine():
    global _engine
    if _engine is None:
        db_url = _get_db_url()
        _engine = create_engine(db_url, echo=False, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
    return _engine


def get_db():
    """FastAPI dependency: yield a database session per request."""
    engine = _get_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ═══════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ═══════════════════════════════════════════════════════════
#  Experiments
# ═══════════════════════════════════════════════════════════


@app.get("/experiments")
def list_experiments(
    tag_key: str | None = Query(None),
    tag_value: str | None = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    """List all experiments with optional tag filtering."""
    tracker = ExperimentTracker(session)
    exps = tracker.list_experiments(tag_key=tag_key, tag_value=tag_value, limit=limit)
    return {"experiments": [e.to_dict() for e in exps], "total": len(exps)}


@app.post("/experiments")
def create_experiment(
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Create a new experiment."""
    tracker = ExperimentTracker(session)
    try:
        exp = tracker.create_experiment(
            name=body.get("name", ""),
            description=body.get("description"),
            dataset_version=body.get("dataset_version"),
            feature_version=body.get("feature_version"),
            model_version=body.get("model_version"),
            notes=body.get("notes"),
            tags=body.get("tags"),
        )
        return exp.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/experiments/{experiment_id}")
def get_experiment(
    experiment_id: str,
    session: Session = Depends(get_db),
):
    """Get a single experiment with runs."""
    tracker = ExperimentTracker(session)
    exp = tracker.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    data = exp.to_dict()
    data["runs"] = [r.to_dict() for r in (exp.runs or [])]
    return data


@app.patch("/experiments/{experiment_id}")
def update_experiment(
    experiment_id: str,
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Update an experiment's fields."""
    tracker = ExperimentTracker(session)
    try:
        exp = tracker.update_experiment(experiment_id, **body)
        return exp.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/experiments/{experiment_id}")
def delete_experiment(
    experiment_id: str,
    session: Session = Depends(get_db),
):
    """Delete an experiment and all associated runs."""
    tracker = ExperimentTracker(session)
    if tracker.delete_experiment(experiment_id):
        return {"deleted": True, "id": experiment_id}
    raise HTTPException(status_code=404, detail="Experiment not found")


# ═══════════════════════════════════════════════════════════
#  Runs
# ═══════════════════════════════════════════════════════════


@app.get("/runs")
def list_runs(
    experiment_id: str | None = Query(None),
    model_type: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    session: Session = Depends(get_db),
):
    """List runs with optional filters."""
    tracker = ExperimentTracker(session)
    runs = tracker.list_runs(
        experiment_id=experiment_id,
        model_type=model_type,
        status=status,
        limit=limit,
    )
    return {"runs": [r.to_dict() for r in runs], "total": len(runs)}


@app.post("/runs")
def start_run(
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Start a new training run."""
    tracker = ExperimentTracker(session)
    try:
        run = tracker.start_run(
            experiment_id=body.get("experiment_id", ""),
            model_type=body.get("model_type", ""),
            run_name=body.get("run_name"),
            hyperparameters=body.get("hyperparameters"),
            random_seed=body.get("random_seed"),
            notes=body.get("notes"),
            tags=body.get("tags"),
        )
        return run.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}")
def get_run(
    run_id: str,
    session: Session = Depends(get_db),
):
    """Get a single run with artifacts."""
    tracker = ExperimentTracker(session)
    run = tracker.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    data = run.to_dict()
    data["artifacts"] = [a.to_dict() for a in tracker.list_artifacts(run_id)]
    return data


@app.post("/runs/{run_id}/finish")
def finish_run(
    run_id: str,
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Finish a run with metrics (including CV, calibration, profit metrics)."""
    tracker = ExperimentTracker(session)
    try:
        run = tracker.finish_run(
            run_id,
            metrics=body.get("metrics"),
            duration_seconds=body.get("duration_seconds"),
            notes=body.get("notes"),
            tags=body.get("tags"),
        )
        # Set extended metrics if provided (via the tracker's new method)
        tracker.update_extended_metrics(
            run_id,
            cross_validation_metrics=body.get("cross_validation_metrics"),
            calibration_metrics=body.get("calibration_metrics"),
            profit_metrics=body.get("profit_metrics"),
            confusion_matrix=body.get("confusion_matrix"),
            feature_importance=body.get("feature_importance"),
            shap_values=body.get("shap_values"),
        )
        run = tracker.get_run(run_id)  # Refresh
        return run.to_dict()  # type: ignore[union-attr]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/runs/{run_id}/fail")
def fail_run(
    run_id: str,
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Mark a run as failed."""
    tracker = ExperimentTracker(session)
    try:
        run = tracker.fail_run(
            run_id,
            error=body.get("error", "Unknown error"),
            duration_seconds=body.get("duration_seconds"),
        )
        return run.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/runs/{run_id}/resume")
def resume_run(
    run_id: str,
    session: Session = Depends(get_db),
):
    """Resume an interrupted run."""
    tracker = ExperimentTracker(session)
    try:
        run = tracker.resume_run(run_id)
        return run.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════
#  Artifacts
# ═══════════════════════════════════════════════════════════


@app.post("/runs/{run_id}/artifacts")
def log_artifact(
    run_id: str,
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Log an artifact for a run."""
    tracker = ExperimentTracker(session)
    try:
        artifact = tracker.log_artifact(
            run_id,
            name=body.get("name", ""),
            uri=body.get("uri", ""),
            file_size_bytes=body.get("file_size_bytes"),
            artifact_type=body.get("artifact_type", "model"),
            metadata=body.get("metadata"),
        )
        return artifact.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/runs/{run_id}/artifacts")
def list_artifacts(
    run_id: str,
    session: Session = Depends(get_db),
):
    """List all artifacts for a run."""
    tracker = ExperimentTracker(session)
    run = tracker.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"artifacts": [a.to_dict() for a in tracker.list_artifacts(run_id)]}


# ═══════════════════════════════════════════════════════════
#  Best Models & Leaderboard
# ═══════════════════════════════════════════════════════════


@app.get("/leaderboard")
def get_leaderboard(
    metric_name: str | None = Query(None),
    experiment_id: str | None = Query(None),
    model_type: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_db),
):
    """Get the model leaderboard."""
    registry = ModelRegistry(session)
    entries = registry.get_leaderboard(
        metric_name=metric_name,
        experiment_id=experiment_id,
        model_type=model_type,
        limit=limit,
    )
    return {
        "leaderboard": [e.to_dict() for e in entries],
        "total": len(entries),
    }


@app.post("/leaderboard")
def register_best_model(
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Register a run as best for a given metric."""
    registry = ModelRegistry(session)
    try:
        entry = registry.register(
            experiment_id=body.get("experiment_id", ""),
            run_id=body.get("run_id", ""),
            metric_name=body.get("metric_name", ""),
            metric_value=body.get("metric_value", 0.0),
            rank=body.get("rank"),
            notes=body.get("notes"),
        )
        return entry.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/leaderboard/{entry_id}/promote")
def promote_model(
    entry_id: str,
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Promote a model to production."""
    registry = ModelRegistry(session)
    try:
        entry = registry.promote(entry_id, notes=body.get("notes"))
        return entry.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/leaderboard/{entry_id}/demote")
def demote_model(
    entry_id: str,
    session: Session = Depends(get_db),
):
    """Remove production status from a model."""
    registry = ModelRegistry(session)
    try:
        entry = registry.demote(entry_id)
        return entry.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ═══════════════════════════════════════════════════════════
#  Comparison
# ═══════════════════════════════════════════════════════════


@app.post("/compare")
def compare_runs(
    body: dict[str, Any],
    session: Session = Depends(get_db),
):
    """Compare runs by IDs or within an experiment."""
    comparator = ExperimentComparator(session)
    if body.get("run_ids"):
        result = comparator.compare_runs(body["run_ids"])
    elif body.get("experiment_id"):
        result = comparator.compare_runs_in_experiment(
            body["experiment_id"],
            model_type=body.get("model_type"),
        )
    else:
        raise HTTPException(status_code=400, detail="Provide run_ids or experiment_id")
    return result


@app.get("/rank")
def rank_runs(
    experiment_id: str | None = Query(None),
    model_type: str | None = Query(None),
    metric: str = Query("val_log_loss"),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_db),
):
    """Rank runs by a specific metric."""
    comparator = ExperimentComparator(session)
    ranked = comparator.rank_by_metric(
        experiment_id=experiment_id,
        model_type=model_type,
        metric=metric,
        limit=limit,
    )
    return {"rankings": ranked}


# ═══════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════


@app.get("/export/json")
def export_experiments_json(
    experiment_id: str | None = Query(None),
    session: Session = Depends(get_db),
):
    """Export experiments as JSON."""
    data = export_json(session, experiment_id=experiment_id)
    import json as json_mod
    return json_mod.loads(data)


@app.get("/export/html")
def export_experiments_html(
    experiment_id: str | None = Query(None),
    title: str = Query("ML Experiment Report"),
    session: Session = Depends(get_db),
):
    """Export experiments as self-contained HTML report."""
    html_str = export_html(
        session,
        experiment_id=experiment_id,
        title=title,
    )
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_str, media_type="text/html")
