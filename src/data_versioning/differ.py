"""
Differ — detect inserted, updated, and deleted rows between dataset versions.

Uses row fingerprinting (hash of key columns) to identify changes.
Optimised for large datasets via chunked comparison.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data_versioning.models import ChangeType, VersionDiff

logger = logging.getLogger(__name__)

# Max sample rows to include in a VersionDiff for inline inspection
_MAX_SAMPLE_ROWS = 10


def _row_hash(row: pd.Series, columns: list[str]) -> str:
    """Compute a deterministic hash of specific columns in a row."""
    hasher = hashlib.sha256()
    for col in columns:
        val = row[col]
        if pd.isna(val):
            hasher.update(b"\x00")
        else:
            hasher.update(str(val).encode("utf-8"))
        hasher.update(b"|")
    return hasher.hexdigest()


def compute_delta(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    """Compute a delta DataFrame between two datasets.

    Each row in the output has:
    - All original columns
    - ``_change_type``: ``"inserted"``, ``"updated"``, or ``"deleted"``
    - ``_fingerprint``: unique row identifier (hex hash of key columns)

    Parameters
    ----------
    old_df : pd.DataFrame
        The base (older) dataset.
    new_df : pd.DataFrame
        The newer dataset to compare against.
    key_columns : list[str]
        Column names that form a unique row key.

    Returns
    -------
    pd.DataFrame
        Delta with ``_change_type`` and ``_fingerprint`` columns.
        Empty if no changes.
    """
    if not key_columns:
        raise ValueError("At least one key column is required for fingerprinting.")

    missing = [c for c in key_columns if c not in old_df.columns]
    if missing:
        raise ValueError(f"Key columns not found in DataFrame: {missing}")

    # Compute fingerprint (SHA256 hash of key column values) for each row
    old_fp = old_df[key_columns].apply(
        lambda r: _row_hash(r, key_columns), axis=1
    )
    new_fp = new_df[key_columns].apply(
        lambda r: _row_hash(r, key_columns), axis=1
    )

    old_set = set(old_fp)
    new_set = set(new_fp)

    inserted_fp = new_set - old_set
    deleted_fp = old_set - new_set
    common_fp = old_set & new_set

    # Build fingerprint → row content hash for common rows
    # (hash of all non-key columns to detect updates)
    all_columns = [c for c in old_df.columns]
    compare_columns = [c for c in all_columns if c not in key_columns]

    old_content_hash = old_df.apply(lambda r: _row_hash(r, compare_columns), axis=1)
    new_content_hash = new_df.apply(lambda r: _row_hash(r, compare_columns), axis=1)

    delta_rows: list[pd.DataFrame] = []

    # --- Deleted rows ---
    if deleted_fp:
        mask = old_fp.isin(deleted_fp)
        deleted = old_df[mask].copy()
        deleted["_change_type"] = ChangeType.DELETED.value
        deleted["_fingerprint"] = old_fp[mask].values
        delta_rows.append(deleted)

    # --- Inserted rows ---
    if inserted_fp:
        mask = new_fp.isin(inserted_fp)
        inserted = new_df[mask].copy()
        inserted["_change_type"] = ChangeType.INSERTED.value
        inserted["_fingerprint"] = new_fp[mask].values
        delta_rows.append(inserted)

    # --- Updated rows (same fingerprint, different content) ---
    if common_fp and compare_columns:
        # Build index lookups
        old_fp_idx = pd.Series(old_fp.index, index=old_fp.values)
        new_fp_idx = pd.Series(new_fp.index, index=new_fp.values)

        for fp in common_fp:
            old_idx = old_fp_idx.get(fp)
            new_idx = new_fp_idx.get(fp)
            if old_idx is not None and new_idx is not None:
                if old_content_hash.iloc[old_idx] != new_content_hash.iloc[new_idx]:
                    updated_row = new_df.iloc[new_idx:new_idx + 1].copy()
                    updated_row["_change_type"] = ChangeType.UPDATED.value
                    updated_row["_fingerprint"] = fp
                    delta_rows.append(updated_row)

    if not delta_rows:
        # Return empty delta with expected schema
        cols = list(old_df.columns) + ["_change_type", "_fingerprint"]
        return pd.DataFrame(columns=cols)

    result = pd.concat(delta_rows, ignore_index=True)
    return result


def compare_versions(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    old_info: Any,
    new_info: Any,
    key_columns: list[str],
    include_samples: bool = True,
) -> VersionDiff:
    """Compute a structured ``VersionDiff`` between two datasets.

    Parameters
    ----------
    old_df : pd.DataFrame
        Base version data.
    new_df : pd.DataFrame
        Target version data.
    old_info : VersionInfo
        Metadata for base version.
    new_info : VersionInfo
        Metadata for target version.
    key_columns : list[str]
        Columns that uniquely identify a row.
    include_samples : bool
        If True, include sample rows in the diff (default True).

    Returns
    -------
    VersionDiff
    """
    delta_df = compute_delta(old_df, new_df, key_columns)

    if delta_df.empty:
        return VersionDiff(
            base_version=old_info.version_id,
            target_version=new_info.version_id,
            n_inserted=0,
            n_updated=0,
            n_deleted=0,
            n_unchanged=len(old_df),
            fingerprint_column=key_columns[0] if key_columns else "_fingerprint",
            metadata_changes=_compare_metadata(old_info, new_info),
        )

    # Count by change type
    change_counts = delta_df["_change_type"].value_counts()
    n_inserted = int(change_counts.get("inserted", 0))
    n_updated = int(change_counts.get("updated", 0))
    n_deleted = int(change_counts.get("deleted", 0))
    n_unchanged = len(old_df) - n_deleted

    # Detect changed columns (cross-reference old vs new values)
    changed_cols = _detect_changed_columns(
        delta_df, key_columns, old_df=old_df, new_df=new_df,
    )

    # Sample rows
    inserted_samples = None
    updated_samples = None
    deleted_samples = None

    if include_samples:
        if n_inserted > 0:
            ins = delta_df[delta_df["_change_type"] == "inserted"]
            inserted_samples = (
                ins.drop(columns=["_change_type", "_fingerprint"], errors="ignore")
                .head(_MAX_SAMPLE_ROWS)
                .to_dict(orient="records")
            )
        if n_updated > 0:
            upd = delta_df[delta_df["_change_type"] == "updated"]
            upd_records = upd.head(_MAX_SAMPLE_ROWS).to_dict(orient="records")
            updated_samples = [(r, r) for r in upd_records]
        if n_deleted > 0:
            dele = delta_df[delta_df["_change_type"] == "deleted"]
            deleted_samples = (
                dele.drop(columns=["_change_type", "_fingerprint"], errors="ignore")
                .head(_MAX_SAMPLE_ROWS)
                .to_dict(orient="records")
            )

    return VersionDiff(
        base_version=old_info.version_id,
        target_version=new_info.version_id,
        n_inserted=n_inserted,
        n_updated=n_updated,
        n_deleted=n_deleted,
        n_unchanged=n_unchanged,
        inserted_rows=inserted_samples,
        updated_rows=updated_samples,
        deleted_rows=deleted_samples,
        changed_columns=changed_cols,
        fingerprint_column=key_columns[0] if key_columns else "_fingerprint",
        metadata_changes=_compare_metadata(old_info, new_info),
    )


def _detect_changed_columns(
    delta_df: pd.DataFrame,
    key_columns: list[str],
    old_df: pd.DataFrame | None = None,
    new_df: pd.DataFrame | None = None,
) -> list[str]:
    """Detect which columns have changed in updated rows.

    Uses the delta's ``_fingerprint`` to cross-reference old vs new values
    for each updated row.
    """
    updated = delta_df[delta_df["_change_type"] == "updated"]
    if updated.empty:
        return []

    exclude = set(key_columns) | {"_change_type", "_fingerprint"}

    if old_df is not None and new_df is not None and "_fingerprint" in updated.columns:
        # Cross-reference old vs new values using fingerprint
        changed = set()
        for _, upd_row in updated.iterrows():
            fp = upd_row["_fingerprint"]
            # Find old row by fingerprint
            old_fp = old_df[key_columns].apply(
                lambda r: _row_hash(r, key_columns), axis=1
            )
            new_fp = new_df[key_columns].apply(
                lambda r: _row_hash(r, key_columns), axis=1
            )
            old_idx = old_fp[old_fp == fp].index
            new_idx = new_fp[new_fp == fp].index
            if len(old_idx) > 0 and len(new_idx) > 0:
                old_row = old_df.loc[old_idx[0]]
                new_row = new_df.loc[new_idx[0]]
                for col in old_row.index:
                    if col in exclude:
                        continue
                    if old_row[col] != new_row[col]:
                        changed.add(col)
        return sorted(changed)

    # Fallback: just check which non-key columns are present in the delta
    changed = []
    for col in updated.columns:
        if col in exclude:
            continue
        if updated[col].nunique() > 1:
            changed.append(col)
    return changed


def _compare_metadata(old_info: Any, new_info: Any) -> dict[str, tuple]:
    """Compare metadata between two versions.

    Returns a dict of {field: (old_value, new_value)} for fields
    that differ.
    """
    changes: dict[str, tuple] = {}

    for field in ["source", "league", "season", "n_rows", "n_columns"]:
        old_val = getattr(old_info, field, None)
        new_val = getattr(new_info, field, None)
        if old_val != new_val:
            changes[field] = (old_val, new_val)

    return changes
