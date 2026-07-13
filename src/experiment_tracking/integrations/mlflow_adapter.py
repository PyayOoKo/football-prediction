"""
MLflow integration adapter — export/import experiments between the local
store and MLflow Tracking Server.

Requires ``mlflow``::

    pip install mlflow

Usage
-----
::

    from src.experiment_tracking.integrations import export_to_mlflow, import_from_mlflow

    # Export all experiments from the local store to MLflow
    export_to_mlflow(session, tracking_uri="http://localhost:5000")

    # Import runs from an MLflow experiment into the local store
    import_from_mlflow(session, mlflow_experiment_name="my_experiment")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.experiment_tracking.models import Experiment, Run
from src.experiment_tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def export_to_mlflow(
    session: Session,
    *,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
) -> int:
    """Export experiments/runs from the local store to MLflow.

    Creates an MLflow experiment for each local experiment and logs
    parameters, metrics, tags, and artifacts for each run.

    Parameters
    ----------
    session : Session
        Local database session.
    tracking_uri : str, optional
        MLflow Tracking URI. Defaults to ``mlflow.get_tracking_uri()``.
    experiment_id : str, optional
        If specified, only export this experiment.

    Returns
    -------
    int
        Number of runs exported.
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "MLflow integration requires the 'mlflow' package. "
            "Install it with: pip install mlflow"
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    tracker = ExperimentTracker(session)
    experiments = (
        [tracker.get_experiment(experiment_id)]
        if experiment_id
        else tracker.list_experiments(limit=1000)
    )
    experiments = [e for e in experiments if e is not None]

    exported = 0
    for exp in experiments:
        # Create or get MLflow experiment
        mlflow_exp = mlflow.get_experiment_by_name(exp.name)
        if mlflow_exp is None:
            mlflow_exp_id = mlflow.create_experiment(
                exp.name,
                tags={"source": "football-prediction"},
            )
        else:
            mlflow_exp_id = mlflow_exp.experiment_id

        # Export each run
        for run in (exp.runs or []):
            with mlflow.start_run(
                experiment_id=mlflow_exp_id,
                run_name=run.run_name or run.model_type,
            ) as mlflow_run:
                # Log parameters
                params = dict(run.hyperparameters or {})
                params["model_type"] = run.model_type
                params["random_seed"] = str(run.random_seed) if run.random_seed else None
                params["model_version"] = run.model_version or ""
                params = {k: str(v) for k, v in params.items() if v is not None}
                mlflow.log_params(params)

                # Log metrics
                metrics = dict(run.metrics or {})
                if run.cross_validation_metrics:
                    for fold, fold_metrics in run.cross_validation_metrics.items():
                        for m_key, m_val in fold_metrics.items():
                            metrics[f"cv_{fold}_{m_key}"] = m_val
                if run.calibration_metrics:
                    for m_key, m_val in run.calibration_metrics.items():
                        metrics[f"calib_{m_key}"] = m_val
                if run.profit_metrics:
                    for m_key, m_val in run.profit_metrics.items():
                        metrics[f"profit_{m_key}"] = m_val
                for m_key, m_val in metrics.items():
                    if isinstance(m_val, (int, float)):
                        mlflow.log_metric(m_key, m_val)

                # Log tags
                tags = dict(run.tags or {})
                tags["status"] = run.status
                tags["git_commit"] = run.git_commit or ""
                tags["source_id"] = run.id
                if run.confusion_matrix:
                    tags["confusion_matrix"] = str(run.confusion_matrix)
                mlflow.set_tags({k: str(v) for k, v in tags.items() if v})

                # Log artifact references
                for artifact in (run.artifacts or []):
                    try:
                        mlflow.log_artifact(artifact.uri)
                    except Exception:
                        logger.warning("Could not log artifact %s to MLflow", artifact.uri)

            exported += 1

    logger.info("Exported %d runs to MLflow", exported)
    return exported


def import_from_mlflow(
    session: Session,
    mlflow_experiment_name: str,
    *,
    tracking_uri: str | None = None,
    local_experiment_name: str | None = None,
) -> int:
    """Import runs from an MLflow experiment into the local store.

    Parameters
    ----------
    session : Session
        Local database session.
    mlflow_experiment_name : str
        Name of the MLflow experiment to import.
    tracking_uri : str, optional
        MLflow Tracking URI.
    local_experiment_name : str, optional
        Name for the local experiment. Defaults to the MLflow name.

    Returns
    -------
    int
        Number of runs imported.
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "MLflow integration requires the 'mlflow' package."
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow_exp = mlflow.get_experiment_by_name(mlflow_experiment_name)
    if mlflow_exp is None:
        raise ValueError(f"MLflow experiment '{mlflow_experiment_name}' not found.")

    tracker = ExperimentTracker(session)
    local_name = local_experiment_name or mlflow_experiment_name

    # Create local experiment
    exp = tracker.create_experiment(
        local_name,
        description=f"Imported from MLflow: {mlflow_experiment_name}",
        tags={"source": "mlflow", "mlflow_experiment_id": mlflow_exp.experiment_id},
    )

    # Search MLflow runs
    runs = mlflow.search_runs(
        experiment_ids=[mlflow_exp.experiment_id],
        order_by=["start_time ASC"],
    )

    imported = 0
    for _, row in runs.iterrows():
        run_dict = row.to_dict()

        # Extract params
        params = {}
        model_type = "unknown"
        for key, value in run_dict.items():
            if key.startswith("params."):
                param_key = key.replace("params.", "")
                params[param_key] = value

        model_type = params.pop("model_type", model_type)
        random_seed_str = params.pop("random_seed", None)
        random_seed = int(random_seed_str) if random_seed_str and random_seed_str.isdigit() else None

        # Extract metrics
        metrics = {}
        for key, value in run_dict.items():
            if key.startswith("metrics."):
                metric_key = key.replace("metrics.", "")
                try:
                    metrics[metric_key] = float(value)
                except (ValueError, TypeError):
                    pass

        # Start and finish run
        local_run = tracker.start_run(
            exp.id,
            model_type=model_type,
            run_name=run_dict.get("tags.mlflow.runName", ""),
            hyperparameters=params,
            random_seed=random_seed,
            tags={"mlflow_run_id": run_dict.get("run_id", "")},
        )

        tracker.finish_run(
            local_run.id,
            metrics=metrics,
            duration_seconds=run_dict.get("metrics.training_duration_seconds"),
        )
        imported += 1

    logger.info("Imported %d runs from MLflow experiment '%s'", imported, mlflow_experiment_name)
    return imported
