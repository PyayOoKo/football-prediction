"""
Storage — Parquet-based version snapshot management.

Handles reading, writing, and managing version snapshots in Parquet format.
Optimised for datasets exceeding 10 million rows using chunked I/O,
compression, and column pruning.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_versioning.models import VersionInfo

logger = logging.getLogger(__name__)

# ── Default chunk size for large datasets ──────────────
_DEFAULT_CHUNK_SIZE = 500_000  # rows per chunk when processing 10M+
_PARQUET_COMPRESSION = "zstd"  # good compression / speed trade-off
_PARQUET_ROW_GROUP_SIZE = 100_000


def compute_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of a DataFrame's content.

    Uses streaming hashing to avoid materialising the full CSV
    string in memory — safe for 10M+ rows.

    The algorithm:
    1. Sort columns for determinism.
    2. Hash column names and types first.
    3. Iterate through sorted row chunks, hashing each row's
       CSV representation incrementally.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to hash.

    Returns
    -------
    str
        Hex digest of the SHA256 hash.
    """
    hasher = hashlib.sha256()

    # Sort columns for determinism
    sorted_cols = sorted(df.columns)
    df_sorted = df[sorted_cols]

    # Hash column names (schema)
    schema_bytes = "|".join(sorted_cols).encode("utf-8")
    hasher.update(schema_bytes)
    hasher.update(b"\n")

    # Hash column dtypes
    dtype_bytes = ",".join(str(df_sorted[c].dtype) for c in sorted_cols).encode("utf-8")
    hasher.update(dtype_bytes)
    hasher.update(b"\n")

    # Hash rows in chunks to limit memory
    chunk_size = 50_000
    for start in range(0, len(df_sorted), chunk_size):
        end = min(start + chunk_size, len(df_sorted))
        chunk = df_sorted.iloc[start:end]
        # Convert chunk to deterministic CSV bytes
        chunk_bytes = chunk.to_csv(
            index=False,
            header=False,  # no header for chunks after first
            sep=",",
            date_format="%Y-%m-%dT%H:%M:%S",
        ).encode("utf-8")
        hasher.update(chunk_bytes)

    return hasher.hexdigest()


def compute_fingerprint(df: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    """Compute a fingerprint string for each row based on key columns.

    Uses vectorised string concatenation + ``pd.util.hash_pandas_object``
    for speed on large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    key_columns : list[str]
        Column names that uniquely identify a row (e.g. ``[\"match_id\"]``
        or ``[\"date\", \"home_team\", \"away_team\"]``).

    Returns
    -------
    pd.Series
        Fingerprint strings, one per row.
    """
    if not key_columns:
        raise ValueError("At least one key column is required for fingerprinting.")

    # Ensure key columns exist
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Key columns not found in DataFrame: {missing}")

    # Vectorised concatenation of key columns
    concat = df[key_columns].astype(str).apply("|".join, axis=1)

    # Compute deterministic SHA256 hashes row by row
    # (pd.util.hash_pandas_object is non-deterministic between calls)
    def _sha256_hex(x: str) -> str:
        return hashlib.sha256(x.encode("utf-8")).hexdigest()

    return concat.apply(_sha256_hex)


class VersionStorage:
    """Manages version snapshots on disk using Parquet format.

    Directory structure::

        data/versions/
        ├── v001/
        │   ├── snapshot.parquet     # Full dataset snapshot
        │   ├── metadata.json        # VersionInfo metadata
        │   └── delta.parquet        # Changes vs previous version (opt.)
        ├── v002/
        │   ├── snapshot.parquet
        │   ├── metadata.json
        │   └── delta.parquet
        └── current -> v002/         # Symlink to "current" version

    Parameters
    ----------
    base_dir : str | Path
        Root directory for version storage (e.g. ``data/versions``).
    chunk_size : int
        Number of rows to process per chunk for large datasets (default 500k).
    compression : str
        Parquet compression codec (default ``zstd``).
    fingerprint_columns : list[str]
        Column names used for row fingerprinting. Defaults to
        ``[\"date\", \"home_team\", \"away_team\"]`` if None.
    """

    def __init__(
        self,
        base_dir: str | Path = "data/versions",
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        compression: str = _PARQUET_COMPRESSION,
        fingerprint_columns: list[str] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.chunk_size = chunk_size
        self.compression = compression
        self.fingerprint_columns = fingerprint_columns or [
            "date", "home_team", "away_team",
        ]
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Version path helpers ────────────────────────────

    def _version_dir(self, version_id: str) -> Path:
        return self.base_dir / version_id

    def _snapshot_path(self, version_id: str) -> Path:
        return self._version_dir(version_id) / "snapshot.parquet"

    def _metadata_path(self, version_id: str) -> Path:
        return self._version_dir(version_id) / "metadata.json"

    def _delta_path(self, version_id: str) -> Path:
        return self._version_dir(version_id) / "delta.parquet"

    def _current_link(self) -> Path:
        return self.base_dir / "current"

    # ── Write ───────────────────────────────────────────

    def save_version(
        self,
        df: pd.DataFrame,
        version_id: str,
        prev_version: VersionInfo | None = None,
        source: str = "",
        league: str = "",
        season: str = "",
        schema_version: str = "",
        pipeline_version: str = "",
        user: str = "system",
        notes: str = "",
        import_duration: float = 0.0,
        tags: dict[str, str] | None = None,
        git_commit: str = "",
    ) -> VersionInfo:
        """Save a DataFrame as a new version snapshot.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to save.
        version_id : str
            Unique version identifier (e.g. ``v001``).
        prev_version : VersionInfo, optional
            Previous version for delta computation.
        source, league, season : str
            Metadata identifying the data source.
        schema_version : str
            Version of the data schema.
        pipeline_version : str
            Version of the ETL pipeline.
        user : str
            User/process creating this version.
        notes : str
            Optional notes.
        import_duration : float
            Seconds taken to import the data.
        tags : dict, optional
            Arbitrary key-value tags.
        git_commit : str
            Git commit hash at version creation.

        Returns
        -------
        VersionInfo
            Populated metadata for the saved version.
        """
        start = datetime.now(timezone.utc)
        version_dir = self._version_dir(version_id)
        version_dir.mkdir(parents=True, exist_ok=True)

        n_rows = len(df)
        n_cols = len(df.columns)
        logger.info(
            "Saving version %s (%d rows x %d cols) → %s",
            version_id, n_rows, n_cols, version_dir,
        )

        # Write Parquet snapshot (chunked for large datasets)
        # Use partitioned parquet for datasets > chunk_size * 2
        if n_rows > self.chunk_size * 2:
            self._write_partitioned_parquet(df, self._snapshot_path(version_id))
        else:
            self._write_parquet_chunked(df, self._snapshot_path(version_id))

        # Compute hash
        content_hash = compute_hash(df)

        # Compute delta vs previous version and track change counts
        added = 0
        deleted = 0
        modified = 0
        delta_path: str | None = None
        if prev_version is not None:
            delta_path, added, deleted, modified = self._compute_and_save_delta(
                df, version_id, prev_version,
            )

        # Build metadata
        info = VersionInfo(
            version_id=version_id,
            created_at=datetime.now(timezone.utc),
            source=source,
            league=league,
            season=season,
            schema_version=schema_version,
            pipeline_version=pipeline_version,
            git_commit=git_commit,
            n_rows=n_rows,
            n_columns=n_cols,
            hash=content_hash,
            data_path=str(self._snapshot_path(version_id)),
            delta_path=delta_path,
            previous_version=prev_version.version_id if prev_version else None,
            import_duration=import_duration,
            user=user,
            notes=notes,
            tags=tags or {},
            added_records=added,
            deleted_records=deleted,
            modified_records=modified,
        )

        # Write metadata JSON
        self._write_metadata(info)

        # Update current symlink
        self._update_current_link(version_id)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            "Version %s saved: %d rows, hash=%s... added=%d deleted=%d modified=%d (%.2fs)",
            version_id, n_rows, content_hash[:12],
            added, deleted, modified, elapsed,
        )

        return info

    def _write_parquet_chunked(
        self,
        df: pd.DataFrame,
        path: Path,
    ) -> None:
        """Write a DataFrame to Parquet, chunking if very large.

        For datasets larger than ``chunk_size``, writes are done in
        chunks to avoid high memory usage.
        """
        n = len(df)
        if n <= self.chunk_size:
            # Single write
            df.to_parquet(
                path,
                compression=self.compression,
                row_group_size=_PARQUET_ROW_GROUP_SIZE,
                index=False,
            )
        else:
            # Chunked write — write first chunk normally, then append
            first = True
            for start in range(0, n, self.chunk_size):
                end = min(start + self.chunk_size, n)
                chunk = df.iloc[start:end]
                chunk.to_parquet(
                    path,
                    compression=self.compression,
                    row_group_size=_PARQUET_ROW_GROUP_SIZE,
                    index=False,
                    append=not first,
                )
                first = False
                logger.debug("  Wrote chunk [%d:%d) to %s", start, end, path)

    def _write_partitioned_parquet(
        self,
        df: pd.DataFrame,
        path: Path,
    ) -> None:
        """Write a DataFrame to partitioned Parquet for datasets > 1M rows.

        Uses Hive-style partitioning (``column=value/`` directories)
        for efficient predicate pushdown and parallel reads. The first
        column is used as the partition key.

        For 50M+ row datasets, this provides:
        - Parallel read/write of partitions
        - Predicate pushdown (only read relevant partitions)
        - Smaller individual file sizes
        - Efficient column pruning
        """
        try:
            # Choose partition column — use first non-numeric column or first column
            partition_col = None
            for col in df.columns:
                if df[col].dtype in ("object", "category", "string"):
                    partition_col = col
                    break

            if partition_col is None or len(df) < self.chunk_size * 2:
                # Fall back to chunked write if no good partition column
                self._write_parquet_chunked(df, path)
                return

            # Write with Hive partitioning
            df.to_parquet(
                path,
                compression=self.compression,
                row_group_size=_PARQUET_ROW_GROUP_SIZE,
                index=False,
                partition_cols=[partition_col],
            )
            n_partitions = df[partition_col].nunique()
            logger.info(
                "  Partitioned by '%s' → %d partitions",
                partition_col, n_partitions,
            )
        except Exception as exc:
            logger.warning(
                "Partitioned write failed, falling back to chunked: %s", exc,
            )
            self._write_parquet_chunked(df, path)

    def _compute_and_save_delta(
        self,
        new_df: pd.DataFrame,
        version_id: str,
        prev_version: VersionInfo,
    ) -> tuple[str, int, int, int]:
        """Compute and save delta between new data and previous version.

        Returns
        -------
        tuple[str, int, int, int]
            (delta_path, added_count, deleted_count, modified_count)
        """
        from src.data_versioning.differ import compute_delta

        # Load previous snapshot
        old_df = self.load_snapshot(prev_version.version_id)

        delta_df = compute_delta(
            old_df, new_df,
            key_columns=self.fingerprint_columns,
        )

        delta_path = self._delta_path(version_id)
        added = 0
        deleted = 0
        modified = 0

        if len(delta_df) > 0:
            change_counts = delta_df["_change_type"].value_counts()
            added = int(change_counts.get("inserted", 0))
            deleted = int(change_counts.get("deleted", 0))
            modified = int(change_counts.get("updated", 0))

            delta_df.to_parquet(
                delta_path,
                compression=self.compression,
                index=False,
            )
            logger.info(
                "Delta saved: %s → %s: +%d -%d ~%d",
                prev_version.version_id, version_id,
                added, deleted, modified,
            )
        else:
            # Write empty delta with schema
            cols = list(new_df.columns) + ["_change_type", "_fingerprint"]
            pd.DataFrame(columns=cols).to_parquet(
                delta_path, compression=self.compression, index=False,
            )

        return str(delta_path), added, deleted, modified

    def _write_metadata(self, info: VersionInfo) -> None:
        """Write version metadata to JSON."""
        path = self._metadata_path(info.version_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info.to_dict(), f, indent=2, default=str)

    def _update_current_link(self, version_id: str) -> None:
        """Update the 'current' symlink to point to the latest version."""
        current = self._current_link()
        target = self._version_dir(version_id)

        try:
            if current.exists() or current.is_symlink():
                current.unlink()
            current.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Windows may not support symlinks without admin; fall back
            # to a marker file
            current.write_text(version_id, encoding="utf-8")

    # ── Read ────────────────────────────────────────────

    def load_snapshot(
        self,
        version_id: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load a version's full snapshot.

        Parameters
        ----------
        version_id : str
            Version to load.
        columns : list[str], optional
            Only load specific columns (faster for large datasets).

        Returns
        -------
        pd.DataFrame
        """
        path = self._snapshot_path(version_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Snapshot not found for version '{version_id}' at {path}"
            )

        return pd.read_parquet(path, columns=columns)

    def load_delta(self, version_id: str) -> pd.DataFrame:
        """Load the delta file for a version.

        Parameters
        ----------
        version_id : str

        Returns
        -------
        pd.DataFrame
            Empty DataFrame if no delta exists.
        """
        path = self._delta_path(version_id)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_metadata(self, version_id: str) -> VersionInfo | None:
        """Load version metadata from disk.

        Parameters
        ----------
        version_id : str

        Returns
        -------
        VersionInfo or None
        """
        path = self._metadata_path(version_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return VersionInfo.from_dict(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load metadata for %s: %s", version_id, exc)
            return None

    def get_current_version_id(self) -> str | None:
        """Return the version ID pointed to by the 'current' link.

        Returns None if no current version is set.
        """
        current = self._current_link()
        if current.is_symlink():
            return current.resolve().name
        if current.exists():
            return current.read_text(encoding="utf-8").strip()
        return None

    def iter_chunks(
        self,
        version_id: str,
        chunk_size: int | None = None,
    ) -> Any:
        """Iterate over a version's data in chunks (memory-efficient).

        Parameters
        ----------
        version_id : str
        chunk_size : int, optional
            Chunk size (defaults to storage's chunk_size).

        Yields
        ------
        pd.DataFrame
            Chunks of data.
        """
        path = self._snapshot_path(version_id)
        if not path.exists():
            return

        chunk_size = chunk_size or self.chunk_size
        for chunk in pd.read_parquet(
            path,
            chunksize=chunk_size,
        ):
            yield chunk

    # ── List / Discover ─────────────────────────────────

    def list_versions(self) -> list[VersionInfo]:
        """List all available versions in order.

        Returns
        -------
        list[VersionInfo]
            Sorted by creation time (oldest first).
        """
        versions: list[VersionInfo] = []
        for dir_path in sorted(self.base_dir.iterdir()):
            if not dir_path.is_dir():
                continue
            version_id = dir_path.name
            if version_id == "current":
                continue
            info = self.load_metadata(version_id)
            if info is not None:
                versions.append(info)

        versions.sort(key=lambda v: v.created_at)
        return versions

    # ── Delete / Rollback ───────────────────────────────

    def delete_version(self, version_id: str) -> bool:
        """Delete a version's data from disk.

        Parameters
        ----------
        version_id : str

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        version_dir = self._version_dir(version_id)
        if not version_dir.exists():
            return False

        import shutil
        shutil.rmtree(version_dir)
        logger.info("Deleted version %s", version_id)
        return True

    def rollback(
        self,
        target_version_id: str,
        create_backup: bool = True,
        user: str = "system",
    ) -> VersionInfo:
        """Rollback the current dataset to a previous version.

        If *create_backup* is True (default), the current state is first
        saved as a backup version before rolling back.

        Parameters
        ----------
        target_version_id : str
            Version to rollback to.
        create_backup : bool
            Save current state as a backup before rolling back.
        user : str
            User performing the rollback.

        Returns
        -------
        VersionInfo
            The target version metadata (now the "current" version).
        """
        target_info = self.load_metadata(target_version_id)
        if target_info is None:
            raise ValueError(
                f"Cannot rollback: version '{target_version_id}' not found."
            )

        current_id = self.get_current_version_id()

        # Backup current state
        if create_backup and current_id is not None:
            current_info = self.load_metadata(current_id)
            if current_info is not None:
                backup_id = f"backup_before_rollback_{target_version_id}"
                self.save_version(
                    df=self.load_snapshot(current_id),
                    version_id=backup_id,
                    prev_version=None,
                    source=current_info.source,
                    league=current_info.league,
                    season=current_info.season,
                    user=user,
                    notes=f"Auto-backup before rollback to {target_version_id}",
                )
                logger.info(
                    "Created backup version %s before rollback", backup_id,
                )

        # Switch current link to target
        self._update_current_link(target_version_id)

        logger.info(
            "Rolled back from %s → %s",
            current_id or "(none)", target_version_id,
        )

        return target_info

    # ── Integrity ───────────────────────────────────────

    def verify_integrity(self, version_id: str) -> bool:
        """Verify that a version's data hash matches its metadata.

        Parameters
        ----------
        version_id : str

        Returns
        -------
        bool
        """
        info = self.load_metadata(version_id)
        if info is None:
            logger.warning("No metadata for version %s", version_id)
            return False

        try:
            df = self.load_snapshot(version_id)
            actual_hash = compute_hash(df)
            matches = actual_hash == info.hash
            if not matches:
                logger.error(
                    "Integrity check FAILED for %s: "
                    "expected %s, got %s",
                    version_id, info.hash[:16], actual_hash[:16],
                )
            return matches
        except Exception as exc:
            logger.error("Integrity check error for %s: %s", version_id, exc)
            return False
