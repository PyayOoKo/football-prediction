"""
Verify Feature Store — check all expected features exist,
validate NaN thresholds, and generate a summary report.

Usage:
    python scripts/verify_feature_store.py
    python scripts/verify_feature_store.py --max-nan-pct 10
    python scripts/verify_feature_store.py --export-json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import Base
from src.feature_store import FeatureRegistry, FeatureStore
from src.feature_store.models import FeatureComputationBatch, FeatureValue

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_session(db_url: str = "sqlite:///data/feature_store.db") -> Session:
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def verify_feature_store(
    db_url: str = "sqlite:///data/feature_store.db",
    max_nan_pct: float = 5.0,
    export_json: str | None = None,
) -> dict:
    start = time.time()
    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db_url": db_url,
        "passed": True,
        "summary": {},
        "details": {},
        "warnings": [],
        "errors": [],
    }

    session = _get_session(db_url)
    try:
        registry = FeatureRegistry(session)
        store = FeatureStore(session)

        # ── 1. Feature definitions count ────────────────────
        all_features = registry.list()
        total_defs = len(all_features)
        active_features = registry.list(status="active") if hasattr(registry, "list") else all_features
        report["summary"]["definitions_total"] = total_defs
        report["summary"]["definitions_active"] = len(active_features)
        report["details"]["definitions"] = [
            {
                "name": f.name,
                "version": f.version,
                "type": f.feature_type,
                "category": f.category.value if f.category else None,
                "status": f.status.value if f.status else None,
                "entity_type": f.entity_type.value if f.entity_type else None,
                "created": str(f.created_at) if f.created_at else None,
            }
            for f in all_features
        ]

        if total_defs == 0:
            msg = "No feature definitions found in database"
            report["errors"].append(msg)
            report["passed"] = False
            report["summary"]["result"] = "FAIL"
            return report

        # ── 2. Feature values count ────────────────────────
        total_values = session.query(FeatureValue).count()
        report["summary"]["values_total"] = total_values
        report["summary"]["values_per_definition"] = round(total_values / max(total_defs, 1), 1)

        if total_values == 0:
            msg = "No feature values stored in database"
            report["warnings"].append(msg)

        # ── 3. Per-feature NaN / value stats ───────────────
        nan_stats = {}
        value_stats = {}

        for fd in all_features:
            def_id = fd.id
            values = (
                session.query(FeatureValue)
                .filter(FeatureValue.feature_definition_id == def_id)
                .all()
            )
            total = len(values)
            numeric_vals = [v.numeric_value for v in values if v.numeric_value is not None]
            nan_count = sum(1 for v in numeric_vals if v is None)
            null_count = total - len(numeric_vals)

            stats = {
                "definition_name": fd.name,
                "definition_version": fd.version,
                "total_values": total,
                "numeric_values": len(numeric_vals),
                "null_values": null_count,
                "nan_pct": round(null_count / max(total, 1) * 100, 2),
            }

            if numeric_vals:
                stats.update({
                    "min": round(float(min(numeric_vals)), 4),
                    "max": round(float(max(numeric_vals)), 4),
                    "mean": round(float(sum(numeric_vals) / len(numeric_vals)), 4),
                    "std": round(
                        float(
                            (sum((x - sum(numeric_vals) / len(numeric_vals)) ** 2 for x in numeric_vals) / len(numeric_vals))
                            ** 0.5
                        ),
                        4,
                    ),
                    "zeros": sum(1 for v in numeric_vals if v == 0),
                })

            nan_stats[fd.name] = stats
            value_stats[fd.name] = stats

            # Check NaN threshold
            if null_count > 0 and total > 0:
                actual_nan_pct = (null_count / total) * 100
                if actual_nan_pct > max_nan_pct:
                    report["warnings"].append(
                        f"Feature '{fd.name}' has {actual_nan_pct:.1f}% null values "
                        f"(threshold: {max_nan_pct}%)"
                    )

        report["details"]["per_feature"] = value_stats

        # ── 4. Computation batches ─────────────────────────
        batches = store.list_batches(limit=10)
        report["summary"]["batches_total"] = len(batches)
        report["details"]["batches"] = [
            {
                "id": b.id,
                "label": b.batch_label,
                "trigger": b.trigger,
                "success": b.success,
                "entity_count": b.entity_count,
                "duration_seconds": b.duration_seconds,
                "features_computed": b.features_computed,
                "created": str(b.created_at) if b.created_at else None,
            }
            for b in batches
        ]

        # Determine overall status
        errors = report["errors"]
        warnings = report["warnings"]
        severe = [w for w in warnings if "80%" in w or "severe" in w.lower()]
        if errors:
            report["passed"] = False
            report["summary"]["result"] = "FAIL"
        elif severe:
            report["passed"] = False
            report["summary"]["result"] = "SEVERE_WARNINGS"
        elif len(warnings) > 10:
            report["passed"] = False
            report["summary"]["result"] = "MANY_WARNINGS"
        elif warnings:
            report["passed"] = True
            report["summary"]["result"] = "PASS_WITH_WARNINGS"
        else:
            report["passed"] = True
            report["summary"]["result"] = "PASS"

        report["summary"]["duration_seconds"] = round(time.time() - start, 2)

        # ── 5. Print summary ───────────────────────────────
        print(f"\n{'=' * 55}")
        print(f"  Feature Store Verification Report")
        print(f"{'=' * 55}")
        print(f"  Timestamp:     {report['timestamp']}")
        print(f"  Result:        {report['summary']['result']}")
        print(f"  Duration:      {report['summary']['duration_seconds']:.2f}s")
        print()
        print(f"  Definitions:   {report['summary']['definitions_total']} total, "
              f"{report['summary']['definitions_active']} active")
        print(f"  Values:        {report['summary']['values_total']:,} total "
              f"({report['summary']['values_per_definition']:.1f} avg/def)")
        print(f"  Batches:       {report['summary']['batches_total']}")
        print()
        print(f"  Errors:        {len(errors)}")
        print(f"  Warnings:      {len(warnings)}")

        if warnings:
            print(f"\n  ⚠ Warnings (first 15):")
            for w in warnings[:15]:
                print(f"    • {w}")
            if len(warnings) > 15:
                print(f"    ... and {len(warnings) - 15} more")

        if errors:
            print(f"\n  ❌ Errors:")
            for e in errors:
                print(f"    • {e}")

        # Top features by null rate
        if value_stats:
            sorted_by_null = sorted(
                value_stats.values(),
                key=lambda x: x["null_values"],
                reverse=True,
            )[:5]
            if any(s["null_values"] > 0 for s in sorted_by_null):
                print(f"\n  Top features by null values:")
                head = "  Name".ljust(30) + "Nulls".rjust(8) + "Total".rjust(8) + "Null%".rjust(8)
                print(head)
                print("  " + "-" * 54)
                for s in sorted_by_null:
                    if s["null_values"] > 0:
                        print(
                            f"  {s['definition_name']:<30s}"
                            f"{s['null_values']:>8d}"
                            f"{s['total_values']:>8d}"
                            f"{s['nan_pct']:>7.1f}%"
                        )

        # Features with all zeros (likely placeholders)
        all_zeros_features = [
            s for s in value_stats.values()
            if s.get("zeros", 0) == s["total_values"] and s["total_values"] > 0
        ]
        if all_zeros_features:
            print(f"\n  ℹ All-zero features ({len(all_zeros_features)}): "
                  f"likely placeholders")
            for s in all_zeros_features[:5]:
                print(f"    {s['definition_name']}")
            if len(all_zeros_features) > 5:
                print(f"    ... and {len(all_zeros_features) - 5} more")

        print(f"{'=' * 55}\n")

        # Export JSON if requested
        if export_json:
            out_path = Path(export_json)
            out_path.write_text(json.dumps(report, indent=2, default=str))
            print(f"Report exported to {out_path.resolve()}")

    finally:
        session.close()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Feature Store data integrity"
    )
    parser.add_argument("--db-url", default="sqlite:///data/feature_store.db")
    parser.add_argument("--max-nan-pct", type=float, default=5.0,
                        help="Max allowed NaN pct per feature (default: 5.0)")
    parser.add_argument("--export-json", help="Export report as JSON")
    args = parser.parse_args()

    report = verify_feature_store(
        db_url=args.db_url,
        max_nan_pct=args.max_nan_pct,
        export_json=args.export_json,
    )

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
