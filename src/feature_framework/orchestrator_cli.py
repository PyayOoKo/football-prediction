"""
Feature Pipeline Orchestrator CLI — build, validate, recompute, list, and
get status of features through the FeatureOrchestrator.

Usage
-----
::

    # Build all features (DataFrame mode)
    python -m src.feature_framework.orchestrator_cli build-features \\
        --entity-type dataframe --input matches.csv \\
        --output features.csv

    # Validate computed features
    python -m src.feature_framework.orchestrator_cli validate-features \\
        --input features.csv

    # Recompute a single feature
    python -m src.feature_framework.orchestrator_cli recompute-feature \\
        elo_rating --input matches.csv

    # List all configured features
    python -m src.feature_framework.orchestrator_cli list-features

    # Get status of a specific feature
    python -m src.feature_framework.orchestrator_cli feature-status elo_rating
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_framework.orchestrator import (
    FeatureOrchestrator,
    OrchestratorReport,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Command handlers
# ═══════════════════════════════════════════════════════════


def cmd_build_features(args: argparse.Namespace) -> int:
    """Run the full feature pipeline to build features."""
    config_dict = _load_config(args.config)

    orchestrator = _build_orchestrator(args, config_dict)

    # Load input data
    df = _load_input(args.input, args.entity_type)

    # Run build
    report = orchestrator.run(
        entity_type=args.entity_type,
        df=df,
        trigger=args.trigger,
        force_recompute=args.force,
    )

    # Save output
    if args.output and report.success:
        _save_output(args.output, report, df)

    return _print_report(report, args.verbose)


def cmd_validate_features(args: argparse.Namespace) -> int:
    """Validate computed features via the FeatureValidator."""
    from src.feature_framework.validation import FeatureValidator

    df = _load_input(args.input, "dataframe")

    validator = FeatureValidator(
        verbose=not args.quiet,
        checks=args.checks.split(",") if args.checks else None,
    )

    result = validator.validate_for_pipeline(df, step_name="cli_validate")

    print(f"\n{'═' * 50}")
    print(f"  FEATURE VALIDATION REPORT")
    print(f"{'═' * 50}")
    print(f"  Passed:          {'✅ YES' if result['passed'] else '❌ NO'}")
    print(f"  Total checks:    {result['total_checks']}")
    print(f"  Failed checks:   {result['failed_checks']}")
    print(f"  Total violations: {result['total_violations']}")

    if args.verbose and result.get("details"):
        print(f"\n  Violations per check:")
        for detail in result["details"]:
            if detail.get("violations"):
                print(f"    • {detail['check']}: {detail['violations']} violations")
                if detail.get("columns"):
                    print(f"      Columns: {', '.join(detail['columns'][:10])}")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Report saved: {output_path}")

    print(f"{'═' * 50}\n")

    return 0 if result["passed"] else 1


def cmd_recompute_feature(args: argparse.Namespace) -> int:
    """Force recompute a single feature by name."""
    config_dict = _load_config(args.config)
    orchestrator = _build_orchestrator(args, config_dict)

    df = _load_input(args.input, "dataframe") if args.input else None

    record = orchestrator.recompute_feature(
        feature_name=args.name,
        entity_type=args.entity_type,
        df=df,
    )

    print(f"\n{'═' * 50}")
    print(f"  RECOMPUTE FEATURE: {args.name}")
    print(f"{'═' * 50}")
    print(f"  Status:    {record.status}")
    print(f"  Duration:  {record.duration:.4f}s")
    if record.error:
        print(f"  Error:     {record.error}")
    if record.output_columns:
        print(f"  Columns:   {', '.join(record.output_columns)}")
    print(f"{'═' * 50}\n")

    return 0 if record.status == "completed" else 1


def cmd_list_features(args: argparse.Namespace) -> int:
    """List all configured features with metadata."""
    config_dict = _load_config(args.config)
    orchestrator = _build_orchestrator(args, config_dict)

    features = orchestrator.list_features()

    if not features:
        print("No features configured.")
        return 0

    # Filter
    if args.type:
        features = [f for f in features if f.get("type") == args.type]
    if args.category:
        features = [f for f in features if f.get("category") == args.category]
    if args.enabled_only:
        features = [f for f in features if f.get("enabled", True)]

    print(f"\n{'═' * 100}")
    print(f"  FEATURES ({len(features)} total)")
    print(f"{'═' * 100}")
    print(
        f"  {'Name':<25s} {'Type':<18s} {'Category':<18s} "
        f"{'Version':<8s} {'Enabled':<8s} {'Transformer':<12s}"
    )
    print(f"  {'─' * 89}")

    for feat in features:
        transformer_ok = "✅" if feat.get("has_transformer") else "❌"
        enabled = "✅" if feat.get("enabled", True) else "❌"
        print(
            f"  {feat['name']:<25s} {feat.get('type', '?'):<18s} "
            f"{feat.get('category', '?'):<18s} "
            f"v{feat.get('version', 1):<5d} "
            f"{enabled:<8s} {transformer_ok:<12s}"
        )

    if args.verbose:
        for feat in features:
            if feat.get("dependencies"):
                deps = ", ".join(feat["dependencies"])
                print(f"    └── {feat['name']} depends on: {deps}")
            if feat.get("output_columns"):
                cols = feat["output_columns"]
                print(f"    └── Output columns ({len(cols)}): {', '.join(cols[:5])}")
                if len(cols) > 5:
                    print(f"        ... and {len(cols) - 5} more")

    print(f"{'═' * 100}\n")
    return 0


def cmd_feature_status(args: argparse.Namespace) -> int:
    """Get detailed status for a single feature."""
    config_dict = _load_config(args.config)
    orchestrator = _build_orchestrator(args, config_dict)

    status = orchestrator.feature_status(args.name)

    if status.get("status") == "not_found":
        print(f"Feature '{args.name}' not found in configuration.")
        return 1

    print(f"\n{'═' * 60}")
    print(f"  FEATURE STATUS: {args.name}")
    print(f"{'═' * 60}")
    print(f"  Name:           {status.get('name', '?')}")
    print(f"  Type:           {status.get('type', '?')}")
    print(f"  Version:        v{status.get('version', 1)}")
    print(f"  Enabled:        {'✅' if status.get('enabled', True) else '❌'}")
    print(f"  Status:         {status.get('status', '?')}")
    print(f"  Initialized:    {'✅' if status.get('initialized', False) else '❌'}")

    deps = status.get("dependencies", [])
    if deps:
        print(f"  Dependencies:   {', '.join(deps)}")
    else:
        print(f"  Dependencies:   (none)")

    output_cols = status.get("output_columns", [])
    if output_cols:
        print(f"  Output columns: {len(output_cols)}")
        for col in output_cols[:10]:
            print(f"    • {col}")
        if len(output_cols) > 10:
            print(f"    ... and {len(output_cols) - 10} more")
    else:
        print(f"  Output columns: (none)")

    print(f"{'═' * 60}\n")
    return 0


# ═══════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════


def _build_orchestrator(
    args: argparse.Namespace,
    config_dict: dict[str, Any] | None,
) -> FeatureOrchestrator:
    """Build a FeatureOrchestrator from CLI args."""
    kwargs: dict[str, Any] = {
        "config_dict": config_dict,
        "show_progress": not args.quiet,
        "parallel": not args.no_parallel,
        "max_workers": args.max_workers,
        "incremental": not args.force,
    }

    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    if args.checkpoint_dir:
        kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.max_retries is not None:
        kwargs["max_retries"] = args.max_retries
    if args.log_to_file:
        kwargs["log_to_file"] = True
        kwargs["log_dir"] = args.log_dir or ".logs/"

    return FeatureOrchestrator(**kwargs)


def _load_config(config_path: str | None) -> dict[str, Any] | None:
    """Load optional YAML/JSON config file."""
    if config_path is None:
        return None

    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    try:
        with open(path) as f:
            if suffix in (".yaml", ".yml"):
                import yaml
                return yaml.safe_load(f)
            elif suffix == ".json":
                return json.load(f)
            else:
                print(f"Unsupported config format: {suffix}", file=sys.stderr)
                sys.exit(1)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_input(
    input_path: str | None,
    entity_type: str,
) -> pd.DataFrame:
    """Load input data from CSV, JSON, or Parquet."""
    if input_path is None:
        # Create a minimal empty DataFrame for operations that don't need input
        return pd.DataFrame()

    path = Path(input_path)
    if not path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif suffix == ".json":
            return pd.read_json(path)
        elif suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        else:
            print(f"Unsupported input format: {suffix}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"Error loading input: {exc}", file=sys.stderr)
        sys.exit(1)


def _save_output(
    output_path: str,
    report: OrchestratorReport,
    df: pd.DataFrame,
) -> None:
    """Save the output DataFrame and report metadata."""
    path = Path(output_path)

    # Save DataFrame
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df.to_csv(path, index=False)
        elif suffix == ".json":
            df.to_json(path, orient="records", indent=2)
        elif suffix in (".parquet", ".pq"):
            df.to_parquet(path, index=False)
        else:
            # Default to CSV
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"  Output saved: {csv_path} (unsupported format '{suffix}', used .csv)")
            return
    except Exception as exc:
        print(f"  Error saving output: {exc}", file=sys.stderr)
        return

    print(f"  Output saved: {path}")

    # Save report metadata alongside
    report_path = path.with_suffix(".report.json")
    try:
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
    except Exception:
        pass


def _print_report(report: OrchestratorReport, verbose: bool) -> int:
    """Print pipeline report and return exit code."""
    print()
    print(report.summary())
    print()

    if verbose and report.validation:
        val = report.validation
        print(f"  Validation: {val['total_violations']} violations, "
              f"{val['failed_checks']} failed checks")
        print()

    return 0 if report.success else 1


# ═══════════════════════════════════════════════════════════
#  Argument parser
# ═══════════════════════════════════════════════════════════


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common orchestrator args to a subparser."""
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML/JSON config file with feature definitions",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress bars and verbose output",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose debug output and detailed reports",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force recompute all features (skip cache)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/features/",
        help="Cache directory (default: .cache/features/)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=".checkpoints/",
        help="Checkpoint directory (default: .checkpoints/)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel computation",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Max parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2,
        help="Max retry attempts per feature (default: 2)",
    )
    parser.add_argument(
        "--log-to-file", action="store_true",
        help="Write structured JSON logs to file",
    )
    parser.add_argument(
        "--log-dir", default=".logs/",
        help="Log directory (default: .logs/)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature Pipeline Orchestrator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build features from CSV data
  python -m src.feature_framework.orchestrator_cli build-features \\
      --input matches.csv --output features.csv

  # Validate computed features
  python -m src.feature_framework.orchestrator_cli validate-features \\
      --input features.csv --checks nan_values,constant_features

  # Recompute a specific feature
  python -m src.feature_framework.orchestrator_cli recompute-feature \\
      elo_rating --input matches.csv

  # List all features
  python -m src.feature_framework.orchestrator_cli list-features

  # Get feature status
  python -m src.feature_framework.orchestrator_cli feature-status elo_rating
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True,
        help="Available commands",
    )

    # build-features
    p_build = subparsers.add_parser(
        "build-features",
        help="Run the full feature pipeline to build features",
    )
    _add_common_args(p_build)
    p_build.add_argument(
        "--input", "-i", required=True,
        help="Input data file (CSV, JSON, or Parquet)",
    )
    p_build.add_argument(
        "--output", "-o", required=True,
        help="Output file for computed features (CSV, JSON, or Parquet)",
    )
    p_build.add_argument(
        "--entity-type", default="dataframe",
        help="Entity type (default: dataframe)",
    )
    p_build.add_argument(
        "--trigger", default="cli",
        help="Computation trigger (default: cli)",
    )
    p_build.set_defaults(func=cmd_build_features)

    # validate-features
    p_val = subparsers.add_parser(
        "validate-features",
        help="Validate computed features",
    )
    _add_common_args(p_val)
    p_val.add_argument(
        "--input", "-i", required=True,
        help="Input feature file (CSV, JSON, or Parquet)",
    )
    p_val.add_argument(
        "--output", "-o",
        help="Output path for validation report JSON",
    )
    p_val.add_argument(
        "--checks",
        help="Comma-separated list of checks (e.g. nan_values,constant_features)",
    )
    p_val.set_defaults(func=cmd_validate_features)

    # recompute-feature
    p_rec = subparsers.add_parser(
        "recompute-feature",
        help="Force recompute a single feature",
    )
    _add_common_args(p_rec)
    p_rec.add_argument(
        "name",
        help="Name of the feature to recompute",
    )
    p_rec.add_argument(
        "--input", "-i",
        help="Input data file (required for DataFrame-mode features)",
    )
    p_rec.add_argument(
        "--entity-type", default="dataframe",
        help="Entity type (default: dataframe)",
    )
    p_rec.set_defaults(func=cmd_recompute_feature)

    # list-features
    p_list = subparsers.add_parser(
        "list-features",
        help="List all configured features with metadata",
    )
    _add_common_args(p_list)
    p_list.add_argument(
        "--type",
        help="Filter by feature type",
    )
    p_list.add_argument(
        "--category",
        help="Filter by feature category",
    )
    p_list.add_argument(
        "--enabled-only", action="store_true",
        help="Show only enabled features",
    )
    p_list.set_defaults(func=cmd_list_features)

    # feature-status
    p_stat = subparsers.add_parser(
        "feature-status",
        help="Get detailed status for a single feature",
    )
    _add_common_args(p_stat)
    p_stat.add_argument(
        "name",
        help="Name of the feature",
    )
    p_stat.set_defaults(func=cmd_feature_status)

    return parser


# ═══════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
