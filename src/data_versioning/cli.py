"""
CLI — command-line interface for the data versioning system.

Commands
--------
::

    # Create a version from a CSV file
    python -m src.data_versioning.cli create-version \\
        --file data/raw/results.csv --source football-data --league E0

    # List all dataset versions
    python -m src.data_versioning.cli list-versions

    # Compare two versions
    python -m src.data_versioning.cli compare v001 v002

    # Rollback to a previous version
    python -m src.data_versioning.cli rollback v003

    # Verify data integrity
    python -m src.data_versioning.cli verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.data_versioning import VersionManager

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data_versioning.cli",
        description="Dataset Versioning — track, compare, and rollback data imports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "versions"),
        help="Version storage directory (default: data/versions)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ── create-version ─────────────────────────────────
    cv_parser = subparsers.add_parser(
        "create-version",
        help="Create a new dataset version from a file",
        description="Load data from CSV or Parquet and save as a new version.",
    )
    cv_parser.add_argument("--file", "-f", required=True, help="Path to input data file (CSV or Parquet)")
    cv_parser.add_argument("--source", "-s", default="", help="Data source identifier")
    cv_parser.add_argument("--league", "-l", default="", help="League code")
    cv_parser.add_argument("--season", default="", help="Season identifier")
    cv_parser.add_argument("--schema-version", default="", help="Schema version identifier")
    cv_parser.add_argument("--pipeline-version", default="", help="Pipeline version identifier")
    cv_parser.add_argument("--user", "-u", default="cli", help="User creating the version")
    cv_parser.add_argument("--notes", "-n", default="", help="Optional notes")
    cv_parser.add_argument("--tag", action="append", help="Key=value tags (can be repeated)")

    # ── list-versions ──────────────────────────────────
    subparsers.add_parser(
        "list-versions",
        help="List all dataset versions",
        description="Show version history with metadata.",
    )

    # ── compare ────────────────────────────────────────
    comp_parser = subparsers.add_parser(
        "compare",
        help="Compare two dataset versions",
        description="Show inserted, updated, and deleted rows between two versions.",
    )
    comp_parser.add_argument("base", help="Base (older) version ID")
    comp_parser.add_argument("target", help="Target (newer) version ID")
    comp_parser.add_argument(
        "--no-samples", action="store_true",
        help="Don't include sample row data in output",
    )

    # ── rollback ───────────────────────────────────────
    rb_parser = subparsers.add_parser(
        "rollback",
        help="Rollback to a previous version",
        description="Restore a previous dataset version as the current one.",
    )
    rb_parser.add_argument("version_id", help="Version to rollback to")
    rb_parser.add_argument(
        "--no-backup", action="store_true",
        help="Don't create a backup before rolling back",
    )
    rb_parser.add_argument(
        "--user", default="cli",
        help="User performing the rollback",
    )

    # ── verify ─────────────────────────────────────────
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify data integrity of versions",
        description="Check that stored data hashes match metadata.",
    )
    verify_parser.add_argument(
        "--version", default=None,
        help="Specific version to verify (default: all)",
    )

    # ── info ───────────────────────────────────────────
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed info about a version",
    )
    info_parser.add_argument("version_id", help="Version ID")

    return parser


def _parse_tags(tag_args: list[str] | None) -> dict[str, str]:
    """Parse --tag key=value arguments into a dict."""
    tags: dict[str, str] = {}
    if not tag_args:
        return tags
    for t in tag_args:
        if "=" in t:
            k, v = t.split("=", 1)
            tags[k.strip()] = v.strip()
        else:
            tags[t] = "true"
    return tags


def cmd_create_version(args: argparse.Namespace, mgr: VersionManager) -> int:
    """Execute the ``create-version`` command."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return 1

    tags = _parse_tags(args.tag)

    try:
        if file_path.suffix.lower() in (".csv",):
            info = mgr.create_version_from_csv(
                csv_path=file_path,
                source=args.source or file_path.stem,
                league=args.league,
                season=args.season,
                user=args.user,
                notes=args.notes,
                tags=tags,
                schema_version=args.schema_version,
                pipeline_version=args.pipeline_version,
            )
        elif file_path.suffix.lower() in (".parquet", ".pq"):
            import pandas as pd
            df = pd.read_parquet(file_path)
            info = mgr.create_version(
                df=df,
                source=args.source or file_path.stem,
                league=args.league,
                season=args.season,
                schema_version=args.schema_version,
                pipeline_version=args.pipeline_version,
                user=args.user,
                notes=args.notes,
                tags=tags,
            )
        else:
            print(f"  ✗ Unsupported file format: {file_path.suffix}")
            print("    Supported: .csv, .parquet")
            return 1

        print(f"\n  ✅ Version created: {info.version_id}")
        print(f"     Rows:      {info.n_rows:,}")
        print(f"     Columns:   {info.n_columns}")
        print(f"     Source:    {info.source}")
        print(f"     League:    {info.league}")
        print(f"     Season:    {info.season}")
        print(f"     Schema:    {info.schema_version or '—'}")
        print(f"     Pipeline:  {info.pipeline_version or '—'}")
        print(f"     Hash:      {info.hash[:16]}...")
        print(f"     Path:      {info.data_path}")
        return 0

    except Exception as exc:
        print(f"  ✗ Failed to create version: {exc}")
        logger.exception("create-version failed")
        return 1


def cmd_list_versions(args: argparse.Namespace, mgr: VersionManager) -> int:
    """Execute the ``list-versions`` command."""
    versions = mgr.list_versions()

    if not versions:
        print("  No versions found.")
        print(f"  Version directory: {mgr.storage.base_dir}")
        print("  Create one with: python -m src.data_versioning.cli create-version --file <path>")
        return 0

    current_id = mgr.storage.get_current_version_id()

    print(f"\n  {'ID':<8} {'Created':<22} {'Source':<20} {'League':<6} {'Season':<8} {'Rows':>10}  {'Hash'}")
    print(f"  {'─' * 84}")
    for v in versions:
        marker = "  ← CURRENT" if v.version_id == current_id else ""
        print(f"  {v}{marker}")

    print(f"\n  {len(versions)} version(s) total")
    if current_id:
        print(f"  Active: {current_id}")
    return 0


def cmd_compare(args: argparse.Namespace, mgr: VersionManager) -> int:
    """Execute the ``compare`` command."""
    try:
        diff = mgr.compare(
            args.base,
            args.target,
            include_samples=not args.no_samples,
        )
    except ValueError as exc:
        print(f"  ✗ {exc}")
        return 1

    print(f"\n  {'=' * 70}")
    print(f"  DIFF: {args.base} → {args.target}")
    print(f"  {'=' * 70}")
    print(f"\n  {'Metric':<20} {'Count':>12}")
    print(f"  {'─' * 32}")
    print(f"  {'Unchanged':<20} {diff.n_unchanged:>12,}")
    print(f"  {'Inserted':<20} {diff.n_inserted:>12,}")
    print(f"  {'Updated':<20} {diff.n_updated:>12,}")
    print(f"  {'Deleted':<20} {diff.n_deleted:>12,}")
    print(f"  {'Total changed':<20} {diff.total_changed:>12,}")

    if diff.changed_columns:
        print(f"\n  Changed columns: {', '.join(diff.changed_columns)}")

    if diff.metadata_changes:
        print(f"\n  Metadata changes:")
        for field, (old, new) in diff.metadata_changes.items():
            print(f"    {field:<15}: {old} → {new}")

    if diff.inserted_rows and not args.no_samples:
        print(f"\n  Sample inserted rows (first {len(diff.inserted_rows)}):")
        for r in diff.inserted_rows[:5]:
            print(f"    + {r}")

    if diff.deleted_rows and not args.no_samples:
        print(f"\n  Sample deleted rows (first {len(diff.deleted_rows)}):")
        for r in diff.deleted_rows[:5]:
            print(f"    - {r}")

    return 0


def cmd_rollback(args: argparse.Namespace, mgr: VersionManager) -> int:
    """Execute the ``rollback`` command."""
    try:
        info = mgr.rollback(
            target_version_id=args.version_id,
            create_backup=not args.no_backup,
            user=args.user,
        )
        print(f"\n  ✅ Rolled back to {args.version_id}")
        print(f"     Rows:   {info.n_rows:,}")
        print(f"     Source: {info.source}")
        print(f"     Date:   {info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    except ValueError as exc:
        print(f"  ✗ {exc}")
        return 1
    except Exception as exc:
        print(f"  ✗ Rollback failed: {exc}")
        logger.exception("rollback failed")
        return 1


def cmd_verify(args: argparse.Namespace, mgr: VersionManager) -> int:
    """Execute the ``verify`` command."""
    try:
        results = mgr.verify(version_id=args.version)
    except Exception as exc:
        print(f"  ✗ Verification failed: {exc}")
        return 1

    all_valid = all(r["valid"] for r in results.values())
    for vid, result in sorted(results.items()):
        status = "✅" if result["valid"] else "❌"
        print(f"  {status} {vid:<8} rows={result['n_rows']:>10,}  hash={result['hash']}  source={result['source']}")

    if all_valid:
        print(f"\n  ✅ All {len(results)} version(s) passed integrity check.")
    else:
        print(f"\n  ❌ Some versions failed integrity check!")
        return 1
    return 0


def cmd_info(args: argparse.Namespace, mgr: VersionManager) -> int:
    """Execute the ``info`` command."""
    info = mgr.get_version(args.version_id)
    if info is None:
        print(f"  ✗ Version '{args.version_id}' not found.")
        return 1

    print(f"\n  {'─' * 50}")
    print(f"  Version: {info.version_id}")
    print(f"  {'─' * 50}")
    print(f"  Created:        {info.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Source:         {info.source}")
    print(f"  League:         {info.league}")
    print(f"  Season:         {info.season}")
    print(f"  Schema version: {info.schema_version or '—'}")
    print(f"  Pipeline v.:    {info.pipeline_version or '—'}")
    print(f"  Git commit:     {info.git_commit[:12] if info.git_commit else '—'}")
    print(f"  Rows:           {info.n_rows:,}")
    print(f"  Columns:        {info.n_columns}")
    print(f"  Hash:           {info.hash[:16]}...")
    print(f"  User:           {info.user}")
    print(f"  Import time:    {info.import_duration:.2f}s")
    print(f"  Added records:  {info.added_records:,}")
    print(f"  Deleted recs:   {info.deleted_records:,}")
    print(f"  Modified recs:  {info.modified_records:,}")
    print(f"  Data path:      {info.data_path}")
    if info.delta_path:
        print(f"  Delta path:     {info.delta_path}")
    print(f"  Prev version:   {info.previous_version or '(none)'}")
    if info.notes:
        print(f"  Notes:          {info.notes}")
    if info.tags:
        print(f"  Tags:           {info.tags}")
    print(f"  {'─' * 50}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Initialise version manager
    mgr = VersionManager(data_dir=args.data_dir)

    # Dispatch
    commands = {
        "create-version": cmd_create_version,
        "list-versions": cmd_list_versions,
        "compare": cmd_compare,
        "rollback": cmd_rollback,
        "verify": cmd_verify,
        "info": cmd_info,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args, mgr)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        return 1
    except Exception as exc:
        print(f"  Error: {exc}")
        logger.exception("Command failed: %s", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
