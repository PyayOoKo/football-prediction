"""
Data models for the dataset versioning system.

Defines the core types used throughout the versioning module:
version metadata, change records, and diff summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ChangeType(str, Enum):
    """Type of change detected between two dataset versions."""

    INSERTED = "inserted"
    UPDATED = "updated"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class VersionInfo:
    """Metadata for a single dataset version.

    Attributes
    ----------
    version_id : str
        Unique version identifier (e.g. ``v001``, ``v002``).
    created_at : datetime
        When this version was created (UTC).
    source : str
        Data source identifier (e.g. ``football-data-co-uk``).
    league : str
        League code (e.g. ``E0``, ``WC``).
    season : str
        Season identifier (e.g. ``2425``, ``2026``).
    n_rows : int
        Number of rows (records) in this version.
    n_columns : int
        Number of columns in this version.
    schema_version : str
        Version of the data schema (e.g. ``v2``, ``2026-07``).
    pipeline_version : str
        Version of the ETL pipeline that produced this version.
    git_commit : str
        Git commit hash when this version was created.
    hash : str
        SHA256 hash of the full dataset content.
    data_path : str
        Path to the Parquet snapshot file.
    delta_path : str | None
        Path to the delta file vs the previous version (None for first).
    previous_version : str | None
        Version ID of the previous version (None for first).
    import_duration : float
        Time in seconds taken to import this version.
    user : str
        User or process that created this version.
    notes : str
        Optional notes about this version.
    tags : dict[str, str]
        Arbitrary key-value tags for filtering/lookup.
    added_records : int
        Number of records added compared to previous version.
    deleted_records : int
        Number of records deleted compared to previous version.
    modified_records : int
        Number of records modified compared to previous version.
    """

    version_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    league: str = ""
    season: str = ""
    schema_version: str = ""
    pipeline_version: str = ""
    git_commit: str = ""
    n_rows: int = 0
    n_columns: int = 0
    hash: str = ""
    data_path: str = ""
    delta_path: str | None = None
    previous_version: str | None = None
    import_duration: float = 0.0
    user: str = "system"
    notes: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    added_records: int = 0
    deleted_records: int = 0
    modified_records: int = 0

    @property
    def is_first(self) -> bool:
        return self.previous_version is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "league": self.league,
            "season": self.season,
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "git_commit": self.git_commit,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "hash": self.hash,
            "data_path": self.data_path,
            "delta_path": self.delta_path,
            "previous_version": self.previous_version,
            "import_duration": round(self.import_duration, 3),
            "user": self.user,
            "notes": self.notes,
            "tags": self.tags,
            "added_records": self.added_records,
            "deleted_records": self.deleted_records,
            "modified_records": self.modified_records,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VersionInfo:
        """Reconstruct from a dictionary (e.g. loaded from JSON)."""
        data = dict(d)
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    def __str__(self) -> str:
        return (
            f"Version {self.version_id} | {self.n_rows:,} rows x {self.n_columns} cols "
            f"| {self.source}/{self.league}/{self.season} "
            f"| {self.created_at.strftime('%Y-%m-%d %H:%M:%S')} "
            f"| hash={self.hash[:12]}..."
        )


@dataclass
class VersionDiff:
    """Result of comparing two dataset versions.

    Attributes
    ----------
    base_version : str
        The older (base) version ID.
    target_version : str
        The newer (target) version ID.
    n_inserted : int
        Number of newly added rows.
    n_updated : int
        Number of rows with changed content (same key, different values).
    n_deleted : int
        Number of rows removed.
    n_unchanged : int
        Number of rows identical in both versions.
    inserted_rows : list[dict[str, Any]] | None
        Sample of inserted rows (or None for large diffs).
    updated_rows : list[tuple[dict, dict]] | None
        Sample of (old, new) updated rows.
    deleted_rows : list[dict[str, Any]] | None
        Sample of deleted rows.
    changed_columns : list[str]
        Columns that have different values in any row.
    fingerprint_column : str
        The column used as row fingerprint.
    metadata_changes : dict[str, tuple[Any, Any]]
        Changes in metadata fields between versions.
    """

    base_version: str
    target_version: str
    n_inserted: int = 0
    n_updated: int = 0
    n_deleted: int = 0
    n_unchanged: int = 0
    inserted_rows: list[dict[str, Any]] | None = None
    updated_rows: list[tuple[dict[str, Any], dict[str, Any]]] | None = None
    deleted_rows: list[dict[str, Any]] | None = None
    changed_columns: list[str] = field(default_factory=list)
    fingerprint_column: str = "_fingerprint"
    metadata_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def total_changed(self) -> int:
        return self.n_inserted + self.n_updated + self.n_deleted

    @property
    def total_rows_target(self) -> int:
        return self.n_unchanged + self.n_inserted + self.n_updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_version": self.base_version,
            "target_version": self.target_version,
            "n_inserted": self.n_inserted,
            "n_updated": self.n_updated,
            "n_deleted": self.n_deleted,
            "n_unchanged": self.n_unchanged,
            "total_changed": self.total_changed,
            "changed_columns": self.changed_columns,
            "fingerprint_column": self.fingerprint_column,
            "metadata_changes": {
                k: [str(v[0][:80]) if isinstance(v[0], str) else v[0],
                    str(v[1][:80]) if isinstance(v[1], str) else v[1]]
                for k, v in self.metadata_changes.items()
            },
        }

    def __str__(self) -> str:
        lines = [
            f"DIFF: {self.base_version} → {self.target_version}",
            f"  Unchanged:  {self.n_unchanged:>10,}",
            f"  Inserted:   {self.n_inserted:>10,}",
            f"  Updated:    {self.n_updated:>10,}",
            f"  Deleted:    {self.n_deleted:>10,}",
            f"  Changed cols: {self.changed_columns or '(none)'}",
        ]
        return "\n".join(lines)


@dataclass
class VersionSummary:
    """Lightweight summary of a version for list displays.

    Attributes
    ----------
    version_id : str
    created_at : datetime
    source : str
    league : str
    season : str
    n_rows : int
    hash_prefix : str
        First 12 characters of the content hash.
    user : str
    """

    version_id: str
    created_at: datetime
    source: str
    league: str
    season: str
    schema_version: str
    pipeline_version: str
    git_commit: str
    n_rows: int
    hash_prefix: str
    user: str
    added_records: int
    deleted_records: int
    modified_records: int

    def __str__(self) -> str:
        return (
            f"{self.version_id:<8} "
            f"{self.created_at.strftime('%Y-%m-%d %H:%M:%S'):<22} "
            f"{self.source:<16} "
            f"{self.league:<6} "
            f"{self.season:<8} "
            f"{self.n_rows:>10,}  "
            f"{self.hash_prefix}"
        )

    @classmethod
    def from_version(cls, v: VersionInfo) -> VersionSummary:
        return cls(
            version_id=v.version_id,
            created_at=v.created_at,
            source=v.source,
            league=v.league,
            season=v.season,
            schema_version=v.schema_version,
            pipeline_version=v.pipeline_version,
            git_commit=v.git_commit,
            n_rows=v.n_rows,
            hash_prefix=v.hash[:12] if v.hash else "?" * 12,
            user=v.user,
            added_records=v.added_records,
            deleted_records=v.deleted_records,
            modified_records=v.modified_records,
        )
