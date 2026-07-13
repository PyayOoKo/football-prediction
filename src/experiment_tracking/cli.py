"""
CLI commands for ML experiment tracking.

Provides subcommands to list experiments, compare runs, export results,
view leaderboards, and manage best models.

Usage
-----
::

    python -m src.experiment_tracking list
    python -m src.experiment_tracking compare --experiment <id>
    python -m src.experiment_tracking export --format html
    python -m src.experiment_tracking leaderboard --metric val_log_loss
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.experiment_tracking.comparator import ExperimentComparator
from src.experiment_tracking.export import export_csv, export_html, export_json
from src.experiment_tracking.models import Base, Run
from src.experiment_tracking.registry import ModelRegistry
from src.experiment_tracking.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def _get_session(db_url: str | None = None) -> Session:
    """Create a database session for the CLI."""
    url = db_url or "sqlite:///experiments.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return Session(engine)


def cmd_list(args: argparse.Namespace) -> int:
    """List experiments."""
    session = _get_session(args.db_url)
    tracker = ExperimentTracker(session)

    experiments = tracker.list_experiments(limit=args.limit)
    if not experiments:
        print("No experiments found.")
        return 0

    print(f"\n  {'ID':<10} {'Name':<30} {'Runs':<6} {'Model Version':<16} {'Created'}")
    print(f"  {'─' * 10} {'─' * 30} {'─' * 6} {'─' * 16} {'─' * 24}")
    for exp in experiments:
        print(
            f"  {exp.id[:8]:<10} {exp.name:<30} {len(exp.runs) if exp.runs else 0:<6} "
            f"{exp.model_version or '—':<16} "
            f"{exp.created_at.strftime('%Y-%m-%d %H:%M') if exp.created_at else '—'}"
        )
    print(f"\n  Total: {len(experiments)} experiment(s)")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show experiment details with runs."""
    session = _get_session(args.db_url)
    tracker = ExperimentTracker(session)

    exp = tracker.get_experiment(args.experiment_id)
    if exp is None:
        print(f"Experiment {args.experiment_id!r} not found.")
        return 1

    print(f"\n  Experiment: {exp.name}")
    print(f"  ID:         {exp.id}")
    print(f"  Dataset:    {exp.dataset_version or '—'}")
    print(f"  Features:   {exp.feature_version or '—'}")
    print(f"  Model:      {exp.model_version or '—'}")
    print(f"  Git:        {exp.git_commit or '—'}")
    print(f"  Notes:      {exp.notes or '—'}")
    print(f"  Created:    {exp.created_at.strftime('%Y-%m-%d %H:%M:%S') if exp.created_at else '—'}")

    runs = tracker.list_runs(experiment_id=exp.id, limit=args.limit)
    if runs:
        print(f"\n  Runs ({len(runs)}):")
        print(f"  {'ID':<10} {'Model':<22} {'Status':<12} {'Duration':<10} {'Seed':<6}")
        print(f"  {'─' * 10} {'─' * 22} {'─' * 12} {'─' * 10} {'─' * 6}")
        for run in runs:
            dur = run.training_duration_seconds
            dur_str = f"{dur:.1f}s" if dur else "—"
            print(
                f"  {run.id[:8]:<10} {run.model_type:<22} {run.status:<12} "
                f"{dur_str:<10} {str(run.random_seed or '—'):<6}"
            )

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare runs or experiments."""
    session = _get_session(args.db_url)
    comparator = ExperimentComparator(session)

    if args.run_ids:
        run_ids = [r.strip() for r in args.run_ids.split(",")]
        result = comparator.compare_runs(run_ids)
    elif args.experiment_id:
        result = comparator.compare_runs_in_experiment(
            args.experiment_id,
            model_type=args.model_type,
        )
    else:
        print("Specify --run-ids or --experiment-id")
        return 1

    runs_data = result.get("runs", {})
    if not runs_data:
        print("No runs to compare.")
        return 0

    best = result.get("best_by_metric", {})
    print(f"\n  Comparison: {len(runs_data)} run(s)")
    print(f"  {'─' * 50}")

    for run_id, data in runs_data.items():
        print(f"\n  Run: {data.get('id', run_id)[:8]}")
        print(f"    Model:  {data.get('model_type', '?')}")
        print(f"    Status: {data.get('status', '?')}")
        print(f"    Duration: {data.get('duration_seconds', '?'):.2f}s" if data.get("duration_seconds") else "")
        metrics = data.get("metrics", {})
        if metrics:
            for m, v in sorted(metrics.items()):
                suffix = " ★" if m in best and best[m].get("run_id") == run_id else ""
                print(f"    {m}: {v:.4f}{suffix}")

    if best:
        print(f"\n  Best by metric:")
        for metric, info in sorted(best.items()):
            print(f"    {metric}: {info['value']:.4f} ({info['model_type']})")

    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """Show the best-models leaderboard."""
    session = _get_session(args.db_url)
    registry = ModelRegistry(session)

    metric = args.metric or "val_log_loss"
    entries = registry.get_leaderboard(metric_name=metric, limit=args.limit)

    if not entries:
        print(f"No entries found for metric '{metric}'.")
        return 0

    print(f"\n  Leaderboard: {metric}")
    print(f"  {'Rank':<6} {'Value':<12} {'Model Type':<22} {'Promoted':<10}")
    print(f"  {'─' * 6} {'─' * 12} {'─' * 22} {'─' * 10}")
    for entry in entries:
        promoted = "✓" if entry.is_promoted else ""
        run = session.get(Run, entry.run_id)
        model_type = run.model_type if run else "?"
        print(
            f"  #{entry.rank:<4} {entry.metric_value:<12.4f} "
            f"{model_type:<22} {promoted:<10}"
        )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export experiment data in the specified format."""
    session = _get_session(args.db_url)
    fmt = args.format or "json"

    output_path = args.output
    if not output_path:
        output_path = f"experiments_export.{fmt}"

    if fmt == "json":
        export_json(session, experiment_id=args.experiment_id, output_path=output_path)
    elif fmt == "csv":
        out_dir = args.output or "experiments_csv_export"
        export_csv(session, experiment_id=args.experiment_id, output_dir=out_dir)
    elif fmt == "html":
        export_html(
            session,
            experiment_id=args.experiment_id,
            output_path=output_path,
            title=args.title or "ML Experiment Report",
        )
    else:
        print(f"Unsupported format: {fmt}. Use json, csv, or html.")
        return 1

    print(f"Exported to {output_path}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """Promote a best-model entry to production."""
    session = _get_session(args.db_url)
    registry = ModelRegistry(session)

    try:
        entry = registry.promote(args.entry_id, notes="Promoted via CLI")
        print(f"Promoted entry {args.entry_id[:8]} to production.")
        print(f"  Metric: {entry.metric_name} = {entry.metric_value:.4f}")
        return 0
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1


# ═══════════════════════════════════════════════════════════
#  New CLI commands
# ═══════════════════════════════════════════════════════════


def _cmd_api(args: argparse.Namespace) -> int:
    """Start the REST API server."""
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn is required. Install it with: pip install uvicorn[standard]")
        return 1
    print(f"Starting API server at http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print(f"  ReDoc: http://{args.host}:{args.port}/redoc")
    uvicorn.run(
        "src.experiment_tracking.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _cmd_mlflow(args: argparse.Namespace) -> int:
    """Export/import experiments to/from MLflow."""
    session = _get_session(args.db_url)
    from src.experiment_tracking.integrations import export_to_mlflow, import_from_mlflow

    if args.action == "export":
        count = export_to_mlflow(
            session,
            tracking_uri=args.tracking_uri,
            experiment_id=args.experiment_id,
        )
        print(f"Exported {count} runs to MLflow.")
    elif args.action == "import":
        if not args.mlflow_experiment:
            print("Error: --mlflow-experiment is required for import.")
            return 1
        count = import_from_mlflow(
            session,
            args.mlflow_experiment,
            tracking_uri=args.tracking_uri,
        )
        print(f"Imported {count} runs from MLflow.")
    else:
        print(f"Unknown action: {args.action}")
        return 1
    return 0


def _cmd_wandb(args: argparse.Namespace) -> int:
    """Export experiments to Weights & Biases."""
    session = _get_session(args.db_url)
    from src.experiment_tracking.integrations import export_to_wandb

    count = export_to_wandb(
        session,
        project=args.project,
        entity=args.entity,
        experiment_id=args.experiment_id,
    )
    print(f"Exported {count} runs to W&B project '{args.project}'.")
    return 0


def _cmd_tensorboard(args: argparse.Namespace) -> int:
    """Export experiments to TensorBoard."""
    session = _get_session(args.db_url)
    from src.experiment_tracking.integrations import export_to_tensorboard

    count = export_to_tensorboard(
        session,
        log_dir=args.log_dir,
        experiment_id=args.experiment_id,
    )
    print(f"Exported {count} runs to TensorBoard at {args.log_dir}.")
    return 0


# ═══════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ML Experiment Tracking — manage experiments, runs, and models.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database URL (default: sqlite:///experiments.db)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # list
    p_list = subparsers.add_parser("list", help="List experiments")
    p_list.add_argument("--limit", type=int, default=20, help="Max experiments")

    # show
    p_show = subparsers.add_parser("show", help="Show experiment details")
    p_show.add_argument("experiment_id", help="Experiment ID")
    p_show.add_argument("--limit", type=int, default=50, help="Max runs to show")

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare runs or experiments")
    p_comp.add_argument("--run-ids", help="Comma-separated run IDs")
    p_comp.add_argument("--experiment-id", help="Experiment ID to compare within")
    p_comp.add_argument("--model-type", help="Filter by model type")

    # leaderboard
    p_lead = subparsers.add_parser("leaderboard", help="View model leaderboard")
    p_lead.add_argument("--metric", default="val_log_loss", help="Metric name")
    p_lead.add_argument("--limit", type=int, default=20)

    # export
    p_export = subparsers.add_parser("export", help="Export experiment data")
    p_export.add_argument("--format", choices=["json", "csv", "html"], default="json")
    p_export.add_argument("--output", help="Output path or directory")
    p_export.add_argument("--experiment-id", help="Filter by experiment ID")
    p_export.add_argument("--title", default="ML Experiment Report", help="HTML report title")

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote a model to production")
    p_promote.add_argument("entry_id", help="BestModel entry ID")

    # api — start REST API server
    p_api = subparsers.add_parser("api", help="Start the REST API server")
    p_api.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    p_api.add_argument("--host", default="127.0.0.1", help="Host (default 127.0.0.1)")
    p_api.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    # mlflow — export to MLflow
    p_mlflow = subparsers.add_parser("mlflow", help="Export/import experiments to/from MLflow")
    p_mlflow.add_argument("--action", choices=["export", "import"], required=True)
    p_mlflow.add_argument("--tracking-uri", help="MLflow Tracking URI")
    p_mlflow.add_argument("--experiment-id", help="Local experiment ID to export")
    p_mlflow.add_argument("--mlflow-experiment", help="MLflow experiment name (for import)")

    # wandb — export to W&B
    p_wandb = subparsers.add_parser("wandb", help="Export experiments to Weights & Biases")
    p_wandb.add_argument("--project", default="football-prediction", help="W&B project")
    p_wandb.add_argument("--entity", help="W&B team/username")
    p_wandb.add_argument("--experiment-id", help="Local experiment ID")

    # tensorboard — export to TensorBoard
    p_tb = subparsers.add_parser("tensorboard", help="Export experiments to TensorBoard")
    p_tb.add_argument("--log-dir", default="./runs/tensorboard", help="TensorBoard log directory")
    p_tb.add_argument("--experiment-id", help="Local experiment ID")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    command_map = {
        "list": cmd_list,
        "show": cmd_show,
        "compare": cmd_compare,
        "leaderboard": cmd_leaderboard,
        "export": cmd_export,
        "promote": cmd_promote,
        "api": _cmd_api,
        "mlflow": _cmd_mlflow,
        "wandb": _cmd_wandb,
        "tensorboard": _cmd_tensorboard,
    }

    cmd_fn = command_map.get(args.command)
    if cmd_fn is None:
        parser.print_help()
        return 1

    return cmd_fn(args)


if __name__ == "__main__":
    sys.exit(main())
