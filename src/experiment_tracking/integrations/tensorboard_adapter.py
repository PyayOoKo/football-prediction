"""
TensorBoard integration adapter — export experiment metrics to TensorBoard.

Requires ``torch`` or ``tensorboard``::

    pip install tensorboard

Usage
-----
::

    from src.experiment_tracking.integrations import export_to_tensorboard

    export_to_tensorboard(session, log_dir="./runs/tensorboard")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from src.experiment_tracking.models import Experiment, Run
from src.experiment_tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def export_to_tensorboard(
    session: Session,
    *,
    log_dir: str = "./runs/tensorboard",
    experiment_id: str | None = None,
) -> int:
    """Export experiment runs to TensorBoard event files.

    Creates a TensorBoard ``SummaryWriter`` per experiment, logging
    scalars (metrics), text (hyperparameters), and histograms where
    available.

    Parameters
    ----------
    session : Session
    log_dir : str
        Root directory for TensorBoard logs.
    experiment_id : str, optional

    Returns
    -------
    int
        Number of runs exported.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard integration requires 'torch' or 'tensorboardX'. "
                "Install with: pip install torch"
            )

    tracker = ExperimentTracker(session)
    experiments = (
        [tracker.get_experiment(experiment_id)]
        if experiment_id
        else tracker.list_experiments(limit=1000)
    )
    experiments = [e for e in experiments if e is not None]

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    exported = 0
    for exp in experiments:
        exp_dir = log_path / exp.name.replace(" ", "_")
        exp_dir.mkdir(exist_ok=True)

        for run in (exp.runs or []):
            run_dir = exp_dir / (run.run_name or run.model_type)
            writer = SummaryWriter(str(run_dir))

            # Log hyperparameters as text
            hparams = dict(run.hyperparameters or {})
            hparams["model_type"] = run.model_type
            hparams["random_seed"] = run.random_seed
            hparams_str = "\n".join(f"{k}: {v}" for k, v in hparams.items())

            writer.add_text("hyperparameters", hparams_str, global_step=0)
            writer.add_text("status", run.status, global_step=0)

            # Log git commit
            if run.git_commit:
                writer.add_text("git_commit", run.git_commit, global_step=0)

            # Log standard metrics as scalars
            metrics = dict(run.metrics or {})
            for m_key, m_val in metrics.items():
                if isinstance(m_val, (int, float)):
                    writer.add_scalar(f"metrics/{m_key}", m_val, global_step=0)

            # Log CV metrics
            if run.cross_validation_metrics:
                for fold, fold_metrics in run.cross_validation_metrics.items():
                    for m_key, m_val in fold_metrics.items():
                        if isinstance(m_val, (int, float)):
                            writer.add_scalar(
                                f"cv/{fold}/{m_key}", m_val, global_step=0,
                            )

            # Log calibration metrics
            if run.calibration_metrics:
                for m_key, m_val in run.calibration_metrics.items():
                    if isinstance(m_val, (int, float)):
                        writer.add_scalar(f"calibration/{m_key}", m_val, global_step=0)

            # Log profit metrics
            if run.profit_metrics:
                for m_key, m_val in run.profit_metrics.items():
                    if isinstance(m_val, (int, float)):
                        writer.add_scalar(f"profit/{m_key}", m_val, global_step=0)

            # Log training duration
            if run.training_duration_seconds:
                writer.add_scalar(
                    "training/duration_seconds",
                    run.training_duration_seconds,
                    global_step=0,
                )

            # Log hardware info as text
            if run.hardware:
                hw_str = "\n".join(f"{k}: {v}" for k, v in run.hardware.items())
                writer.add_text("hardware", hw_str, global_step=0)

            # Log feature importance as histograms/scalars
            if run.feature_importance:
                for feat_name, importance in run.feature_importance.items():
                    if isinstance(importance, (int, float)):
                        writer.add_scalar(
                            f"feature_importance/{feat_name}",
                            importance,
                            global_step=0,
                        )

            # Log confusion matrix as text
            if run.confusion_matrix:
                cm = run.confusion_matrix
                cm_str = (
                    f"TP: {cm.get('tp', 0)}, FP: {cm.get('fp', 0)}, "
                    f"TN: {cm.get('tn', 0)}, FN: {cm.get('fn', 0)}"
                )
                writer.add_text("confusion_matrix", cm_str, global_step=0)

            writer.close()
            exported += 1

    logger.info("Exported %d runs to TensorBoard at %s", exported, log_path)
    return exported
