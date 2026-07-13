"""
Weights & Biases integration adapter — export experiments to W&B.

Requires ``wandb``::

    pip install wandb

Usage
-----
::

    from src.experiment_tracking.integrations import export_to_wandb

    export_to_wandb(session, project="football-prediction", entity="my-team")
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from src.experiment_tracking.models import Experiment, Run
from src.experiment_tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def export_to_wandb(
    session: Session,
    *,
    project: str = "football-prediction",
    entity: str | None = None,
    experiment_id: str | None = None,
    notes: str | None = None,
) -> int:
    """Export experiment runs to Weights & Biases.

    Creates a W&B run for each local run, logging hyperparameters as
    config and metrics as W&B metrics.

    Parameters
    ----------
    session : Session
        Local database session.
    project : str
        W&B project name (default ``football-prediction``).
    entity : str, optional
        W&B team/username.
    experiment_id : str, optional
        If specified, only export this experiment.
    notes : str, optional
        Notes for the W&B run group.

    Returns
    -------
    int
        Number of runs exported.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "W&B integration requires the 'wandb' package. "
            "Install it with: pip install wandb"
        )

    tracker = ExperimentTracker(session)
    experiments = (
        [tracker.get_experiment(experiment_id)]
        if experiment_id
        else tracker.list_experiments(limit=1000)
    )
    experiments = [e for e in experiments if e is not None]

    exported = 0
    for exp in experiments:
        # Initialize W&B run for the experiment group
        wandb.init(
            project=project,
            entity=entity,
            group=exp.name,
            notes=notes or exp.description,
            tags=["football-prediction", "exported"],
            config={
                "dataset_version": exp.dataset_version,
                "feature_version": exp.feature_version,
                "model_version": exp.model_version,
                "git_commit": exp.git_commit,
                "experiment_id": exp.id,
                "experiment_name": exp.name,
            },
            reinit=True,
        )

        for run in (exp.runs or []):
            metrics = dict(run.metrics or {})

            # Log extended metrics
            if run.cross_validation_metrics:
                for fold, fold_metrics in run.cross_validation_metrics.items():
                    for m_key, m_val in fold_metrics.items():
                        if isinstance(m_val, (int, float)):
                            metrics[f"cv/{fold}/{m_key}"] = m_val
            if run.calibration_metrics:
                for m_key, m_val in run.calibration_metrics.items():
                    if isinstance(m_val, (int, float)):
                        metrics[f"calib/{m_key}"] = m_val
            if run.profit_metrics:
                for m_key, m_val in run.profit_metrics.items():
                    if isinstance(m_val, (int, float)):
                        metrics[f"profit/{m_key}"] = m_val

            # Log as a nested run
            wandb_run = wandb.run  # Use the current run
            wandb_run.name = run.run_name or run.model_type
            wandb_run.config.update({
                "model_type": run.model_type,
                "hyperparameters": run.hyperparameters or {},
                "random_seed": run.random_seed,
                "status": run.status,
                "training_duration": run.training_duration_seconds,
            })

            # Log metrics
            if metrics:
                wandb_run.log(metrics)

            # Log confusion matrix as W&B Table
            if run.confusion_matrix:
                cm = run.confusion_matrix
                cm_table = wandb.Table(
                    columns=["", "Predicted Positive", "Predicted Negative"],
                    data=[
                        ["Actual Positive", cm.get("tp", 0), cm.get("fn", 0)],
                        ["Actual Negative", cm.get("fp", 0), cm.get("tn", 0)],
                    ],
                )
                wandb_run.log({"confusion_matrix": cm_table})

            # Log feature importance as bar chart
            if run.feature_importance:
                feat_names = list(run.feature_importance.keys())[:20]
                feat_values = [run.feature_importance[n] for n in feat_names]
                fi_table = wandb.Table(
                    columns=["feature", "importance"],
                    data=list(zip(feat_names, feat_values)),
                )
                wandb_run.log({"feature_importance": wandb.plot.bar(
                    fi_table, "feature", "importance",
                    title="Top Feature Importance",
                )})

            exported += 1

        wandb.finish()

    logger.info("Exported %d runs to W&B project '%s'", exported, project)
    return exported


def import_from_wandb(
    session: Session,
    *,
    project: str = "football-prediction",
    entity: str | None = None,
    run_ids: list[str] | None = None,
) -> int:
    """Import runs from W&B into the local experiment store.

    Parameters
    ----------
    session : Session
    project : str
    entity : str, optional
    run_ids : list[str], optional
        Specific W&B run IDs to import. If None, imports all.

    Returns
    -------
    int
        Number of runs imported.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "W&B integration requires the 'wandb' package."
        )

    api = wandb.Api()
    runs = api.runs(f"{entity or ''}/{project}" if entity else project)

    tracker = ExperimentTracker(session)
    imported = 0

    for wandb_run in runs:
        if run_ids and wandb_run.id not in run_ids:
            continue

        # Get or create experiment
        exp_name = wandb_run.group or f"wandb_import_{project}"
        existing = tracker.list_experiments(limit=1000)
        exp = next((e for e in existing if e.name == exp_name), None)
        if exp is None:
            exp = tracker.create_experiment(
                exp_name,
                description=f"Imported from W&B: {wandb_run.name}",
                tags={"source": "wandb", "wandb_run_id": wandb_run.id},
            )

        # Create run
        config = dict(wandb_run.config)
        local_run = tracker.start_run(
            exp.id,
            model_type=config.pop("model_type", "unknown"),
            run_name=wandb_run.name,
            hyperparameters=config.pop("hyperparameters", config),
            tags={"wandb_run_id": wandb_run.id},
        )

        # Get metrics
        metrics = {}
        for metric_key in wandb_run.history().columns:
            if metric_key in ["_step", "_runtime", "_timestamp"]:
                continue
            try:
                values = wandb_run.history(keys=[metric_key])[metric_key]
                final_val = values.dropna().iloc[-1] if not values.dropna().empty else None
                if final_val is not None:
                    metrics[metric_key] = float(final_val)
            except (ValueError, TypeError, IndexError):
                pass

        tracker.finish_run(
            local_run.id,
            metrics=metrics,
            duration_seconds=wandb_run.summary.get("_runtime"),
        )
        imported += 1

    logger.info("Imported %d runs from W&B", imported)
    return imported
