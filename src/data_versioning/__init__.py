"""
Data Versioning — track, compare, and roll back dataset imports.

Every import operation creates an immutable version snapshot. The system
maintains a full version history with metadata, enables diffing between
any two versions, and supports rollback to any previous state.

Architecture
------------
::

    ┌─────────────────────────────────────────────┐
    │              VersionManager                  │
    │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
    │  │ storage  │  │  differ  │  │ metadata  │  │
    │  │ .parquet │  │ .delta   │  │ .json     │  │
    │  └──────────┘  └──────────┘  └───────────┘  │
    └─────────────────────────────────────────────┘

Storage
-------
- **Parquet** format for version snapshots (compressed, columnar, fast
  for 10M+ rows).
- **JSON manifest** per version with metadata (hash, row count, source,
  timestamp).
- **Delta files** record inserted/updated/deleted rows between versions.
- **Hard-link** or symlink to the "current" dataset pointer.

Usage
-----
::

    from src.data_versioning import VersionManager

    mgr = VersionManager(data_dir="data/versions")

    # Create a version from a DataFrame
    v = mgr.create_version(
        df=match_data,
        source="football-data-co-uk",
        league="E0",
        season="2425",
        user="pipeline",
    )
    print(f"Created version {v.version_id}")

    # List all versions
    for v in mgr.list_versions():
        print(f"  {v.version_id}  {v.created_at}  {v.n_rows} rows")

    # Compare two versions
    diff = mgr.compare("v001", "v002")
    print(f"Inserted: {diff.n_inserted}, Updated: {diff.n_updated}, "
          f"Deleted: {diff.n_deleted}")

    # Rollback
    mgr.rollback("v001")

CLI
---
::

    # Create a version from a CSV file
    python -m src.data_versioning.cli create-version \\
        --file data/raw/results.csv --source football-data --league E0

    # List all versions
    python -m src.data_versioning.cli list-versions

    # Rollback to a specific version
    python -m src.data_versioning.cli rollback v003

    # Compare two versions
    python -m src.data_versioning.cli compare v001 v002
"""

from __future__ import annotations

from src.data_versioning.manager import VersionManager
from src.data_versioning.models import (
    VersionInfo,
    VersionDiff,
    ChangeType,
    VersionSummary,
)

__all__ = [
    "VersionManager",
    "VersionInfo",
    "VersionDiff",
    "ChangeType",
    "VersionSummary",
]
