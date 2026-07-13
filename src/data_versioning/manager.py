"""
VersionManager — high-level orchestrator for dataset versioning.

Coordinates storage, diffing, and metadata for the full version lifecycle:
create, list, compare, rollback, and verify.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_versioning.differ import compare_versions
from src.data_versioning.models import VersionDiff, VersionInfo, VersionSummary
from src.data_versioning.storage import VersionStorage

logger = logging.getLogger(__name__)


class VersionManager:
    """High-level orchestrator for the data versioning system.

    Parameters
    ----------
    data_dir : str | Path
        Root directory for version storage (default ``data/versions``).
    fingerprint_columns : list[str], optional
        Columns used for row fingerprinting (default: date, home_team,
        away_team).
    chunk_size : int
        Rows per chunk for large dataset processing (default 500k).
    """

    def __init__(
        self,
        data_dir: str | Path = "data/versions",
        fingerprint_columns: list[str] | None = None,
        chunk_size: int = 500_000,
        pipeline_version: str = "",
        schema_version: str = "",
    ) -> None:
        self.storage = VersionStorage(
            base_dir=data_dir,
            chunk_size=chunk_size,
            fingerprint_columns=fingerprint_columns,
        )
        self.pipeline_version = pipeline_version
        self.schema_version = schema_version
        self._version_counter = 0

    @staticmethod
    def _capture_git_commit() -> str:
        """Capture the current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    # ── Version ID generation ───────────────────────────

    def _next_version_id(self) -> str:
        """Generate the next version ID (e.g. v001, v002, ...)."""
        existing = self.storage.list_versions()
        if existing:
            last_id = existing[-1].version_id
            try:
                next_num = int(last_id.lstrip("v")) + 1
            except ValueError:
                next_num = len(existing) + 1
        else:
            next_num = 1
        return f"v{next_num:03d}"

    # ── Create Version ──────────────────────────────────

    def create_version(
        self,
        df: pd.DataFrame,
        source: str = "",
        league: str = "",
        season: str = "",
        user: str = "system",
        notes: str = "",
        tags: dict[str, str] | None = None,
        version_id: str | None = None,
        import_duration: float = 0.0,
        schema_version: str = "",
        pipeline_version: str = "",
        git_commit: str = "",
    ) -> VersionInfo:
        """Create a new dataset version from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to version.
        source : str
            Data source identifier.
        league : str
            League code.
        season : str
            Season identifier.
        user : str
            User/process creating the version.
        notes : str
            Optional notes.
        tags : dict, optional
            Arbitrary key-value tags.
        version_id : str, optional
            Explicit version ID (auto-generated if omitted).
        import_duration : float
            Seconds taken to prepare the data.
        schema_version : str
            Version of the data schema.
        pipeline_version : str
            Version of the ETL pipeline.
        git_commit : str
            Git commit hash. Auto-captured if not provided.

        Returns
        -------
        VersionInfo
            Metadata for the newly created version.
        """
        if df.empty:
            raise ValueError("Cannot create version from empty DataFrame.")

        if not git_commit:
            git_commit = self._capture_git_commit()
        if not schema_version:
            schema_version = self.schema_version
        if not pipeline_version:
            pipeline_version = self.pipeline_version

        vid = version_id or self._next_version_id()
        prev = self._get_previous_version()

        info = self.storage.save_version(
            df=df,
            version_id=vid,
            prev_version=prev,
            source=source,
            league=league,
            season=season,
            schema_version=schema_version,
            pipeline_version=pipeline_version,
            user=user,
            notes=notes,
            tags=tags,
            import_duration=import_duration,
            git_commit=git_commit,
        )

        logger.info(
            "Created version %s (%d rows, schema=%s, pipeline=%s, git=%s)",
            vid, len(df), schema_version or "?",
            pipeline_version or "?", git_commit[:8] if git_commit else "?",
        )
        return info

    def create_version_from_csv(
        self,
        csv_path: str | Path,
        source: str = "",
        league: str = "",
        season: str = "",
        user: str = "system",
        notes: str = "",
        tags: dict[str, str] | None = None,
        parse_dates: list[str] | None = None,
        schema_version: str = "",
        pipeline_version: str = "",
        git_commit: str = "",
    ) -> VersionInfo:
        """Create a new version by loading data from a CSV file.

        Parameters
        ----------
        csv_path : str | Path
            Path to the CSV file.
        source, league, season, user, notes, tags :
            See ``create_version``.
        parse_dates : list[str], optional
            Date columns to parse.
        schema_version, pipeline_version, git_commit :
            Version/pipeline/git metadata — see ``create_version``.

        Returns
        -------
        VersionInfo
        """
        import time

        start = time.perf_counter()
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        logger.info("Loading CSV: %s", csv_path)
        kwargs: dict[str, Any] = {"low_memory": False}
        if parse_dates:
            kwargs["parse_dates"] = parse_dates
        df = pd.read_csv(csv_path, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("Loaded %d rows from %s (%.2fs)", len(df), csv_path, elapsed)

        return self.create_version(
            df=df,
            source=source or csv_path.stem,
            league=league,
            season=season,
            user=user,
            notes=notes,
            tags=tags,
            import_duration=elapsed,
            schema_version=schema_version,
            pipeline_version=pipeline_version,
            git_commit=git_commit,
        )

    # ── List Versions ───────────────────────────────────

    def list_versions(self) -> list[VersionSummary]:
        """List all versions as lightweight summaries.

        Returns
        -------
        list[VersionSummary]
            Sorted oldest-first.
        """
        return [
            VersionSummary.from_version(v)
            for v in self.storage.list_versions()
        ]

    def get_version(self, version_id: str) -> VersionInfo | None:
        """Get full metadata for a specific version.

        Parameters
        ----------
        version_id : str

        Returns
        -------
        VersionInfo or None
        """
        return self.storage.load_metadata(version_id)

    def get_current_version(self) -> VersionInfo | None:
        """Get the current (active) version's metadata.

        Returns
        -------
        VersionInfo or None
        """
        current_id = self.storage.get_current_version_id()
        if current_id is None:
            return None
        return self.storage.load_metadata(current_id)

    # ── Compare ─────────────────────────────────────────

    def compare(
        self,
        base_version_id: str,
        target_version_id: str,
        include_samples: bool = True,
    ) -> VersionDiff:
        """Compare two versions and return a structured diff.

        Parameters
        ----------
        base_version_id : str
            The older version.
        target_version_id : str
            The newer version.
        include_samples : bool
            Include sample rows in the diff output (default True).

        Returns
        -------
        VersionDiff
        """
        if base_version_id == target_version_id:
            info = self.storage.load_metadata(base_version_id)
            return VersionDiff(
                base_version=base_version_id,
                target_version=target_version_id,
                n_unchanged=info.n_rows if info else 0,
                fingerprint_column=self.storage.fingerprint_columns[0],
            )

        old_info = self.storage.load_metadata(base_version_id)
        new_info = self.storage.load_metadata(target_version_id)

        if old_info is None:
            raise ValueError(f"Base version '{base_version_id}' not found.")
        if new_info is None:
            raise ValueError(f"Target version '{target_version_id}' not found.")

        logger.info(
            "Comparing %s (%d rows) → %s (%d rows)",
            base_version_id, old_info.n_rows,
            target_version_id, new_info.n_rows,
        )

        old_df = self.storage.load_snapshot(base_version_id)
        new_df = self.storage.load_snapshot(target_version_id)

        diff = compare_versions(
            old_df=old_df,
            new_df=new_df,
            old_info=old_info,
            new_info=new_info,
            key_columns=self.storage.fingerprint_columns,
            include_samples=include_samples,
        )

        logger.info(
            "Diff: %d unchanged, %d inserted, %d updated, %d deleted",
            diff.n_unchanged, diff.n_inserted,
            diff.n_updated, diff.n_deleted,
        )

        return diff

    # ── Rollback ────────────────────────────────────────

    def rollback(
        self,
        target_version_id: str,
        create_backup: bool = True,
        user: str = "system",
    ) -> VersionInfo:
        """Rollback to a previous dataset version.

        Parameters
        ----------
        target_version_id : str
            Version to restore.
        create_backup : bool
            Backup current state before rolling back (default True).
        user : str
            User performing the rollback.

        Returns
        -------
        VersionInfo
        """
        return self.storage.rollback(
            target_version_id=target_version_id,
            create_backup=create_backup,
            user=user,
        )

    # ── Verify ──────────────────────────────────────────

    def verify(self, version_id: str | None = None) -> dict[str, Any]:
        """Verify data integrity for one or all versions.

        Parameters
        ----------
        version_id : str, optional
            Specific version to verify. If None, verifies all.

        Returns
        -------
        dict[str, Any]
            ``{version_id: {"valid": bool, "n_rows": int, "hash": str}}``
        """
        versions = [version_id] if version_id else [
            v.version_id for v in self.storage.list_versions()
        ]

        results: dict[str, Any] = {}
        for vid in versions:
            info = self.storage.load_metadata(vid)
            valid = self.storage.verify_integrity(vid)
            results[vid] = {
                "valid": valid,
                "n_rows": info.n_rows if info else 0,
                "hash": info.hash[:16] if info else "?",
                "source": info.source if info else "",
            }

        return results

    # ── Load Data ───────────────────────────────────────

    def load_current_data(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load the current version's dataset.

        Parameters
        ----------
        columns : list[str], optional
            Only load specific columns (column pruning for performance).

        Returns
        -------
        pd.DataFrame
        """
        current_id = self.storage.get_current_version_id()
        if current_id is None:
            raise ValueError(
                "No current version set. Create a version first."
            )
        return self.storage.load_snapshot(current_id, columns=columns)

    # ── Internal ────────────────────────────────────────

    def _get_previous_version(self) -> VersionInfo | None:
        """Get the current version (used as previous for deltas)."""
        current_id = self.storage.get_current_version_id()
        if current_id is None:
            return None
        return self.storage.load_metadata(current_id)
