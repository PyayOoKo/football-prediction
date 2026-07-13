"""
Feature Store CLI — manage feature definitions, values, validation, and computation.

Usage
-----
::

    # Feature definitions
    python -m src.feature_store.cli register <name> --type elo --category elo_rating --entity team
    python -m src.feature_store.cli list [--status active] [--category elo_rating]
    python -m src.feature_store.cli show <name>
    python -m src.feature_store.cli activate <name>
    python -m src.feature_store.cli deprecate <name> --reason "..."
    python -m src.feature_store.cli retire <name> --reason "..."

    # Feature values
    python -m src.feature_store.cli get <name> --match-id 42
    python -m src.feature_store.cli set <name> --match-id 42 --numeric 0.85

    # Validation
    python -m src.feature_store.cli validate <name> [--match-id 42]

    # Computation
    python -m src.feature_store.cli compute <names> --entity-ids 1,2,3 --entity-type match
    python -m src.feature_store.cli resume <batch-id>

    # Lineage
    python -m src.feature_store.cli provenance <name> [--match-id 42]
    python -m src.feature_store.cli lineage-summary

    # Cache
    python -m src.feature_store.cli cache-stats
    python -m src.feature_store.cli cache-warm <name> --entity-ids 1,2,3
    python -m src.feature_store.cli cache-clear

    # Batch tracking
    python -m src.feature_store.cli batches [--limit 10] [--trigger scheduled]

    # Export
    python -m src.feature_store.cli export [--format json|csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine

from src.database.base import Base
from src.feature_store import (
    ComputerRegistry,
    FeatureCategory,
    FeatureComputationEngine,
    FeatureRegistry,
    FeatureStore,
    FeatureStatus,
    FeatureValidator,
)
from src.feature_store.cache import FeatureCache
from src.feature_store.lineage import FeatureLineage
from src.feature_store.store import FeatureStore

# Import models so tables get created
from src.feature_store.models import (
    FeatureComputationBatch,
    FeatureDependency,
    FeatureValue,
    FeatureVersion,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Database connection helpers
# ═══════════════════════════════════════════════════════════


def _get_session(db_url: str = "sqlite:///data/feature_store.db"):
    """Create a session for the CLI.

    Uses SQLite by default for CLI usage; can override with ``--db-url``.
    """
    from sqlalchemy.orm import Session, sessionmaker

    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# ═══════════════════════════════════════════════════════════
#  Category mappings
# ═══════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "rolling_stat": FeatureCategory.ROLLING_STAT,
    "team_form": FeatureCategory.TEAM_FORM,
    "elo_rating": FeatureCategory.ELO_RATING,
    "attack_strength": FeatureCategory.ATTACK_STRENGTH,
    "defense_strength": FeatureCategory.DEFENSE_STRENGTH,
    "home_advantage": FeatureCategory.HOME_ADVANTAGE,
    "away_advantage": FeatureCategory.AWAY_ADVANTAGE,
    "rest_days": FeatureCategory.REST_DAYS,
    "fixture_congestion": FeatureCategory.FIXTURE_CONGESTION,
    "league_strength": FeatureCategory.LEAGUE_STRENGTH,
    "team_momentum": FeatureCategory.TEAM_MOMENTUM,
    "market_movement": FeatureCategory.MARKET_MOVEMENT,
    "h2h_stat": FeatureCategory.H2H_STAT,
    "xg_feature": FeatureCategory.XG_FEATURE,
    "odds_feature": FeatureCategory.ODDS_FEATURE,
    "composite": FeatureCategory.COMPOSITE,
}

STATUS_MAP = {
    "draft": FeatureStatus.DRAFT,
    "active": FeatureStatus.ACTIVE,
    "deprecated": FeatureStatus.DEPRECATED,
    "retired": FeatureStatus.RETIRED,
}


# ═══════════════════════════════════════════════════════════
#  Subcommand handlers
# ═══════════════════════════════════════════════════════════


def cmd_register(args: argparse.Namespace) -> None:
    """Register a new feature definition."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    category = CATEGORY_MAP.get(args.category)
    if category is None:
        print(f"Unknown category: {args.category}")
        print(f"Available: {', '.join(CATEGORY_MAP.keys())}")
        sys.exit(1)

    status = STATUS_MAP.get(args.status, FeatureStatus.DRAFT)

    try:
        fd = registry.register(
            name=args.name,
            feature_type=args.type,
            category=category,
            entity_type=args.entity_type,
            description=args.description,
            computation_params=json.loads(args.params) if args.params else None,
            validation_rules=json.loads(args.validation) if args.validation else None,
            dependencies=args.dependencies.split(",") if args.dependencies else None,
            status=status,
            changelog=args.changelog or "Initial registration",
        )
        print(f"✅ Registered: {fd.name} v{fd.version} (id={fd.id})")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        session.close()


def cmd_list(args: argparse.Namespace) -> None:
    """List feature definitions."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    category = CATEGORY_MAP.get(args.category) if args.category else None
    status = STATUS_MAP.get(args.status) if args.status else None

    features = registry.list(
        category=category,
        feature_type=args.type,
        entity_type=args.entity_type,
        status=status,
        is_active=args.active,
    )

    if not features:
        print("No features found.")
        session.close()
        return

    print(f"{'Name':<35s} {'Version':<8s} {'Type':<20s} {'Category':<20s} {'Status':<12s} {'Entity':<10s}")
    print("-" * 105)
    for fd in features:
        print(
            f"{fd.name:<35s} {fd.version:<8d} {fd.feature_type:<20s} "
            f"{fd.category.value if fd.category else '?':<20s} "
            f"{fd.status.value if fd.status else '?':<12s} "
            f"{fd.entity_type.value if fd.entity_type else '?':<10s}"
        )
    print(f"\n{len(features)} feature(s)")

    session.close()


def cmd_show(args: argparse.Namespace) -> None:
    """Show details for a feature definition."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    fd = registry.latest(args.name)
    if fd is None:
        print(f"Feature '{args.name}' not found.")
        session.close()
        sys.exit(1)

    d = fd.to_dict()
    print(f"Name:          {d['name']}")
    print(f"Version:       {d['version']}")
    print(f"Type:          {d['feature_type']}")
    print(f"Category:      {d['category']}")
    print(f"Entity Type:   {d['entity_type']}")
    print(f"Status:        {d['status']}")
    print(f"Active:        {d['is_active']}")
    print(f"Description:   {d['description'] or '(none)'}")
    print(f"ID:            {d['id']}")
    print(f"Created:       {d['created_at']}")
    print(f"Updated:       {d['updated_at']}")

    if d['computation_params']:
        print(f"\nComputation Params:")
        print(f"  {json.dumps(d['computation_params'], indent=2)}")

    if d['validation_rules']:
        print(f"\nValidation Rules:")
        print(f"  {json.dumps(d['validation_rules'], indent=2)}")

    if d['dependencies']:
        deps = registry.get_dependencies(fd.id)
        print(f"\nDependencies ({len(deps)}):")
        for dep in deps:
            print(f"  - {dep.name} v{dep.version}")

    # Show history
    history = registry.get_history(args.name)
    if history:
        print(f"\nVersion History ({len(history)}):")
        for v in history:
            print(f"  v{v.version:3d} | {'current' if v.is_current else '':8s} | {v.changelog or ''}")

    # Show lineage
    try:
        lineage = FeatureLineage(session)
        upstream = lineage.get_upstream(fd)
        if upstream:
            print(f"\nUpstream Sources/Transforms ({len(upstream)}):")
            for u in upstream:
                print(f"  [{u['type']:10s}] {u['name']} (v{u['version'] or '?'})")
    except Exception:
        pass

    session.close()


def cmd_activate(args: argparse.Namespace) -> None:
    """Activate a feature definition."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    try:
        fd = registry.activate(args.name, version=args.version)
        print(f"✅ Activated: {fd.name} v{fd.version}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        session.close()


def cmd_deprecate(args: argparse.Namespace) -> None:
    """Deprecate a feature definition."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    try:
        fd = registry.deprecate(args.name, reason=args.reason or "")
        print(f"✅ Deprecated: {fd.name} v{fd.version}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        session.close()


def cmd_retire(args: argparse.Namespace) -> None:
    """Retire a feature definition."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    try:
        fd = registry.retire(args.name, reason=args.reason or "")
        print(f"✅ Retired: {fd.name} v{fd.version}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        session.close()


def cmd_new_version(args: argparse.Namespace) -> None:
    """Create a new version of a feature definition."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    updates: dict[str, Any] = {}
    if args.description:
        updates["description"] = args.description
    if args.params:
        updates["computation_params"] = json.loads(args.params)
    if args.validation:
        updates["validation_rules"] = json.loads(args.validation)

    try:
        fd = registry.new_version(
            args.name,
            changelog=args.changelog or "",
            **updates,
        )
        print(f"✅ Created {fd.name} v{fd.version}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        session.close()


def cmd_get_value(args: argparse.Namespace) -> None:
    """Get a feature value."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)
    store = FeatureStore(session)

    fd = registry.latest(args.name)
    if fd is None:
        print(f"Feature '{args.name}' not found.")
        session.close()
        sys.exit(1)

    kwargs = {}
    if args.match_id is not None:
        kwargs["match_id"] = args.match_id
    if args.team_id is not None:
        kwargs["team_id"] = args.team_id
    if args.league_id is not None:
        kwargs["league_id"] = args.league_id

    value = store.get_value(fd.id, **kwargs)  # type: ignore[arg-type]
    if value is None:
        print(f"No value found for {args.name} with given entity.")
    else:
        print(f"{args.name} = {value}")
        print(f"  (definition: {fd.id})")

    session.close()


def cmd_set_value(args: argparse.Namespace) -> None:
    """Set a feature value."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)
    store = FeatureStore(session)

    fd = registry.latest(args.name)
    if fd is None:
        print(f"Feature '{args.name}' not found.")
        session.close()
        sys.exit(1)

    kwargs: dict[str, Any] = {"computed_by": args.computed_by or "cli"}
    if args.match_id is not None:
        kwargs["match_id"] = args.match_id
    if args.team_id is not None:
        kwargs["team_id"] = args.team_id
    if args.numeric is not None:
        kwargs["numeric_value"] = args.numeric
    if args.text is not None:
        kwargs["text_value"] = args.text
    if args.json_val is not None:
        kwargs["json_value"] = json.loads(args.json_val)

    fv = store.set(definition_id=fd.id, **kwargs)  # type: ignore[arg-type]
    print(f"✅ Set {args.name} = {fv.numeric_value or fv.text_value or '(json)'}")
    print(f"   Value ID: {fv.id}")

    session.close()


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a feature value."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)
    store = FeatureStore(session)

    fd = registry.latest(args.name)
    if fd is None:
        print(f"Feature '{args.name}' not found.")
        session.close()
        sys.exit(1)

    kwargs = {}
    if args.match_id is not None:
        kwargs["match_id"] = args.match_id
    if args.team_id is not None:
        kwargs["team_id"] = args.team_id

    value = store.get(fd.id, **kwargs)  # type: ignore[arg-type]

    validator = FeatureValidator()
    result = validator.validate_one(fd, value)

    print(f"Validation for '{args.name}':")
    print(f"  Passed:    {result.passed}")
    if result.errors:
        print(f"  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    ❌ {err}")
    if result.warnings:
        print(f"  Warnings ({len(result.warnings)}):")
        for warn in result.warnings:
            print(f"    ⚠ {warn}")
    if result.metadata:
        print(f"  Metadata: {json.dumps(result.metadata, indent=4)}")

    session.close()


def cmd_compute(args: argparse.Namespace) -> None:
    """Compute features for entities."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)
    store = FeatureStore(session)
    computer_registry = ComputerRegistry()

    feature_names = args.names.split(",")
    entity_ids = [int(x) for x in args.entity_ids.split(",")]

    engine = FeatureComputationEngine(
        registry=computer_registry,
        store=store,
        registry_service=registry,
        show_progress=not args.no_progress,
    )

    report = engine.compute_features(
        feature_names=feature_names,
        entity_ids=entity_ids,
        entity_type=args.entity_type,
        trigger=args.trigger,
        incremental=not args.force,
        force_recompute=args.force,
    )

    print(f"\n{'═' * 50}")
    print(f"Batch:     {report.batch_label}")
    print(f"Batch ID:  {report.batch_id}")
    print(f"Duration:  {report.duration_seconds:.2f}s")
    print(f"Success:   {'✅' if report.success else '❌'}")

    if report.error:
        print(f"Error:     {report.error}")

    print(f"\n{'Feature':<30s} {'C':>4s} {'S':>4s} {'F':>4s} {'Duration':>10s}")
    print(f"{'─' * 52}")
    for fname, stats in sorted(report.per_feature_stats.items()):
        print(
            f"{fname:<30s} {stats['computed']:>4d} {stats['skipped']:>4d} "
            f"{stats['failed']:>4d} {stats['duration']:>8.2f}s"
        )
    print(f"{'─' * 52}")
    print(f"{'TOTAL':<30s} {report.computed_count:>4d} {report.skipped_count:>4d} "
          f"{report.failed_count:>4d} {report.duration_seconds:>8.2f}s")

    session.close()


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume a failed computation batch."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)
    store = FeatureStore(session)
    computer_registry = ComputerRegistry()

    engine = FeatureComputationEngine(
        registry=computer_registry,
        store=store,
        registry_service=registry,
        show_progress=not args.no_progress,
    )

    try:
        report = engine.resume(
            batch_id=args.batch_id,
            force_recompute=args.force,
        )
        print(f"\nResumed batch {args.batch_id}")
        print(f"  Computed: {report.computed_count}")
        print(f"  Skipped:  {report.skipped_count}")
        print(f"  Failed:   {report.failed_count}")
        print(f"  Duration: {report.duration_seconds:.2f}s")
        print(f"  Status:   {'✅' if report.success else '❌'}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        session.close()


def cmd_provenance(args: argparse.Namespace) -> None:
    """Show provenance for a feature value."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)
    lineage = FeatureLineage(session)

    fd = registry.latest(args.name)
    if fd is None:
        print(f"Feature '{args.name}' not found.")
        session.close()
        sys.exit(1)

    provenance = lineage.get_provenance(
        fd,
        match_id=args.match_id,
        team_id=args.team_id,
    )

    print(f"Provenance for '{args.name}':")
    print(f"  Version:    {provenance.feature_version}")
    print(f"  Value:      {provenance.value}")
    if provenance.computed_at:
        print(f"  Computed:   {provenance.computed_at}")
    print(f"  Computed By: {provenance.computed_by or '(not recorded)'}")

    if provenance.source_chain:
        print(f"\n  Source Chain ({len(provenance.source_chain)} steps):")
        for i, source in enumerate(provenance.source_chain):
            print(
                f"    {i}. [{source['type']:10s}] {source['name']}"
                f"{' v' + (source['version'] or '?') if source.get('version') else ''}"
            )

    if provenance.model_consumers:
        print(f"\n  Consumed By Models ({len(provenance.model_consumers)}):")
        for model in provenance.model_consumers:
            print(f"    - {model}")

    session.close()


def cmd_lineage_summary(args: argparse.Namespace) -> None:
    """Show lineage summary."""
    session = _get_session(args.db_url)
    lineage = FeatureLineage(session)

    summary = lineage.get_source_summary()
    if not summary:
        print("No lineage entries recorded yet.")
        session.close()
        return

    print(f"{'Source':<30s} {'Version':<12s} {'Features':<10s} {'Downstream':<10s}")
    print("-" * 62)
    for s in summary:
        print(
            f"{s['source_name']:<30s} {str(s['source_version'] or '?'):<12s} "
            f"{s['feature_count']:<10d} {s['total_downstream_entries']:<10d}"
        )

    total = lineage.to_dict()
    print(f"\nTotal: {total['total_entries']} lineage entries "
          f"({total['sources']} sources, {total['transforms']} transforms, "
          f"{total['features']} features, {total['models']} models)")

    session.close()


def cmd_batches(args: argparse.Namespace) -> None:
    """List computation batches."""
    session = _get_session(args.db_url)
    store = FeatureStore(session)

    batches = store.list_batches(
        trigger=args.trigger,
        success=args.success,
        limit=args.limit,
    )

    if not batches:
        print("No batches found.")
        session.close()
        return

    print(f"{'Label':<35s} {'ID':<10s} {'Trigger':<12s} {'Status':<8s} {'Entities':<10s} {'Duration':<10s}")
    print("-" * 85)
    for b in batches:
        status = "✅" if b.success else "❌"
        dur = f"{b.duration_seconds:.1f}s" if b.duration_seconds else "-"
        print(
            f"{b.batch_label:<35s} {b.id[:8]:<10s} {b.trigger:<12s} {status:<8s} "
            f"{b.entity_count:<10d} {dur:<10s}"
        )

    session.close()


def cmd_export(args: argparse.Namespace) -> None:
    """Export feature definitions or values."""
    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)

    if args.data == "definitions":
        data = registry.to_dict()
        output = args.output or f"feature_definitions_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    elif args.data == "lineage":
        lineage = FeatureLineage(session)
        data = lineage.to_dict()
        output = args.output or f"feature_lineage_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    else:
        print("Unknown export type. Use 'definitions' or 'lineage'.")
        session.close()
        return

    fmt = args.format or "json"

    if fmt == "json":
        output_path = f"{output}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif fmt == "csv":
        output_path = f"{output}.csv"
        if isinstance(data, list):
            with open(output_path, "w", newline="") as f:
                if data:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        else:
            print("CSV export only supported for lists.")
            session.close()
            return

    print(f"✅ Exported {len(data) if isinstance(data, list) else 'data'} to {output_path}")

    session.close()


def cmd_cache_stats(args: argparse.Namespace) -> None:
    """Show cache statistics."""
    from src.cache import CacheManager, SQLiteBackend

    session = _get_session(args.db_url)
    store = FeatureStore(session)

    # Create a cache wrapping the store
    cache_backend = SQLiteBackend("data/cache/feature_cache.db")
    cache_manager = CacheManager(cache_backend, namespace="feature")
    feature_cache = FeatureCache(store, cache_manager)

    import asyncio
    stats = asyncio.run(feature_cache.cache_stats())

    print(f"Cache Statistics:")
    print(f"  Namespace:    feature")
    print(f"  Hits:         {stats.hits}")
    print(f"  Misses:       {stats.misses}")
    print(f"  Hit Ratio:    {stats.hit_ratio:.1%}")
    print(f"  Entries:      {stats.entries}")
    print(f"  Size:         {stats.size_bytes:,} bytes")

    session.close()


def cmd_cache_warm(args: argparse.Namespace) -> None:
    """Warm the cache for a feature."""
    from src.cache import CacheManager, SQLiteBackend

    session = _get_session(args.db_url)
    registry = FeatureRegistry(session)
    store = FeatureStore(session)

    fd = registry.latest(args.name)
    if fd is None:
        print(f"Feature '{args.name}' not found.")
        session.close()
        sys.exit(1)

    entity_ids = [int(x) for x in args.entity_ids.split(",")]

    cache_backend = SQLiteBackend("data/cache/feature_cache.db")
    cache_manager = CacheManager(cache_backend, namespace="feature")
    feature_cache = FeatureCache(store, cache_manager)

    import asyncio
    warmed = asyncio.run(
        feature_cache.warm(fd, entity_ids, entity_type=args.entity_type)
    )

    print(f"✅ Warmed {warmed}/{len(entity_ids)} cache entries for {args.name}")

    session.close()


def cmd_cache_clear(args: argparse.Namespace) -> None:
    """Clear all cached feature values."""
    from src.cache import CacheManager, SQLiteBackend

    session = _get_session(args.db_url)
    store = FeatureStore(session)

    cache_backend = SQLiteBackend("data/cache/feature_cache.db")
    cache_manager = CacheManager(cache_backend, namespace="feature")
    feature_cache = FeatureCache(store, cache_manager)

    import asyncio
    cleared = asyncio.run(feature_cache.clear_cache())

    print(f"✅ Cleared {cleared} cache entries")
    session.close()


# ═══════════════════════════════════════════════════════════
#  Argument parser
# ═══════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature Store CLI — manage features, values, and pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a new feature
  python -m src.feature_store.cli register elo_rating --type elo --category elo_rating --entity team --status active
  
  # List active features
  python -m src.feature_store.cli list --status active
  
  # Show feature details
  python -m src.feature_store.cli show elo_rating
  
  # Set a feature value
  python -m src.feature_store.cli set elo_rating --team-id 42 --numeric 1500.0
  
  # Show provenance
  python -m src.feature_store.cli provenance elo_rating --team-id 42
        """,
    )
    parser.add_argument("--db-url", default="sqlite:///data/feature_store.db",
                        help="Database URL (default: sqlite:///data/feature_store.db)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # register
    p_register = subparsers.add_parser("register", help="Register a new feature definition")
    p_register.add_argument("name", help="Feature name")
    p_register.add_argument("--type", required=True, help="Feature type (e.g. elo, rolling_stat)")
    p_register.add_argument("--category", required=True, help="Feature category")
    p_register.add_argument("--entity-type", default="match", help="Entity type (match/team/league/player/global)")
    p_register.add_argument("--description", help="Feature description")
    p_register.add_argument("--params", help="Computation params as JSON")
    p_register.add_argument("--validation", help="Validation rules as JSON")
    p_register.add_argument("--dependencies", help="Comma-separated dependency names")
    p_register.add_argument("--status", default="draft", help="Initial status (draft/active)")
    p_register.add_argument("--changelog", help="Change description")

    # list
    p_list = subparsers.add_parser("list", help="List feature definitions")
    p_list.add_argument("--category", help="Filter by category")
    p_list.add_argument("--type", help="Filter by feature type")
    p_list.add_argument("--entity-type", help="Filter by entity type")
    p_list.add_argument("--status", help="Filter by status")
    p_list.add_argument("--active", action="store_true", default=None, help="Filter by active (use --no-active)")

    # show
    p_show = subparsers.add_parser("show", help="Show feature definition details")
    p_show.add_argument("name", help="Feature name")

    # activate / deprecate / retire
    for cmd_name in ("activate", "deprecate", "retire"):
        p = subparsers.add_parser(cmd_name, help=f"{cmd_name.capitalize()} a feature definition")
        p.add_argument("name", help="Feature name")
        if cmd_name in ("deprecate", "retire"):
            p.add_argument("--reason", help="Reason")
        if cmd_name == "activate":
            p.add_argument("--version", type=int, help="Specific version to activate")

    # new-version
    p_nv = subparsers.add_parser("new-version", help="Create a new version of a feature")
    p_nv.add_argument("name", help="Feature name")
    p_nv.add_argument("--changelog", required=True, help="What changed")
    p_nv.add_argument("--description", help="New description")
    p_nv.add_argument("--params", help="New computation params (JSON)")
    p_nv.add_argument("--validation", help="New validation rules (JSON)")

    # get
    p_get = subparsers.add_parser("get", help="Get a feature value")
    p_get.add_argument("name", help="Feature name")
    p_get.add_argument("--match-id", type=int, help="Match entity ID")
    p_get.add_argument("--team-id", type=int, help="Team entity ID")
    p_get.add_argument("--league-id", type=int, help="League entity ID")

    # set
    p_set = subparsers.add_parser("set", help="Set a feature value")
    p_set.add_argument("name", help="Feature name")
    p_set.add_argument("--match-id", type=int, help="Match entity ID")
    p_set.add_argument("--team-id", type=int, help="Team entity ID")
    p_set.add_argument("--numeric", type=float, help="Numeric value")
    p_set.add_argument("--text", help="Text value")
    p_set.add_argument("--json-val", help="JSON value")
    p_set.add_argument("--computed-by", default="cli", help="Computer identifier")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate a feature value")
    p_val.add_argument("name", help="Feature name")
    p_val.add_argument("--match-id", type=int, help="Match entity ID")
    p_val.add_argument("--team-id", type=int, help="Team entity ID")

    # compute
    p_comp = subparsers.add_parser("compute", help="Compute features for entities")
    p_comp.add_argument("names", help="Comma-separated feature names")
    p_comp.add_argument("--entity-ids", required=True, help="Comma-separated entity IDs")
    p_comp.add_argument("--entity-type", default="match", help="Entity type (match/team)")
    p_comp.add_argument("--trigger", default="manual", help="Computation trigger")
    p_comp.add_argument("--force", action="store_true", help="Force recompute all entities")
    p_comp.add_argument("--no-progress", action="store_true", help="Hide progress bars")

    # resume
    p_res = subparsers.add_parser("resume", help="Resume a computation batch")
    p_res.add_argument("batch_id", help="Batch ID to resume")
    p_res.add_argument("--force", action="store_true", help="Recompute all entities")
    p_res.add_argument("--no-progress", action="store_true", help="Hide progress bars")

    # provenance
    p_prov = subparsers.add_parser("provenance", help="Show feature provenance")
    p_prov.add_argument("name", help="Feature name")
    p_prov.add_argument("--match-id", type=int, help="Match entity ID")
    p_prov.add_argument("--team-id", type=int, help="Team entity ID")

    # lineage-summary
    subparsers.add_parser("lineage-summary", help="Show lineage summary")

    # batches
    p_batch = subparsers.add_parser("batches", help="List computation batches")
    p_batch.add_argument("--limit", type=int, default=20, help="Max results")
    p_batch.add_argument("--trigger", help="Filter by trigger type")
    p_batch.add_argument("--success", action="store_true", default=None, help="Filter by success")

    # export
    p_exp = subparsers.add_parser("export", help="Export feature data")
    p_exp.add_argument("data", choices=["definitions", "lineage"], help="What to export")
    p_exp.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    p_exp.add_argument("--output", help="Output file path (without extension)")

    # cache commands
    subparsers.add_parser("cache-stats", help="Show cache statistics")

    p_cw = subparsers.add_parser("cache-warm", help="Warm cache for a feature")
    p_cw.add_argument("name", help="Feature name")
    p_cw.add_argument("--entity-ids", required=True, help="Comma-separated entity IDs")
    p_cw.add_argument("--entity-type", default="match", help="Entity type")

    subparsers.add_parser("cache-clear", help="Clear cached feature values")

    return parser


# ═══════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════


COMMAND_MAP: dict[str, Any] = {
    "register": cmd_register,
    "list": cmd_list,
    "show": cmd_show,
    "activate": cmd_activate,
    "deprecate": cmd_deprecate,
    "retire": cmd_retire,
    "new-version": cmd_new_version,
    "get": cmd_get_value,
    "set": cmd_set_value,
    "validate": cmd_validate,
    "compute": cmd_compute,
    "resume": cmd_resume,
    "provenance": cmd_provenance,
    "lineage-summary": cmd_lineage_summary,
    "batches": cmd_batches,
    "export": cmd_export,
    "cache-stats": cmd_cache_stats,
    "cache-warm": cmd_cache_warm,
    "cache-clear": cmd_cache_clear,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command is None:
        parser.print_help()
        return 1

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        return 1

    try:
        handler(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
