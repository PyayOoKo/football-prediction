"""
Tests for ``src.data_versioning.differ``.

Covers delta computation, version comparison, change type detection,
and edge cases (empty data, identical data, all-rows-changed).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_versioning.differ import compute_delta, compare_versions
from src.data_versioning.models import ChangeType, VersionInfo, VersionDiff


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["A", "B", "C", "D", "E"],
        "value": [10, 20, 30, 40, 50],
    })


@pytest.fixture
def new_df() -> pd.DataFrame:
    # id=3 deleted, id=6 inserted, id=2 updated (value 20→25), rest unchanged
    # Names for common IDs must match base_df:
    #   id=4 → name='D', id=5 → name='E' (unchanged)
    return pd.DataFrame({
        "id": [1, 2, 4, 5, 6],
        "name": ["A", "B", "D", "E", "F"],
        "value": [10, 25, 40, 50, 60],
    })


@pytest.fixture
def base_info() -> VersionInfo:
    return VersionInfo(
        version_id="v001",
        source="test",
        league="E0",
        season="2425",
        n_rows=5,
        n_columns=3,
    )


@pytest.fixture
def new_info() -> VersionInfo:
    return VersionInfo(
        version_id="v002",
        source="test",
        league="E0",
        season="2425",
        n_rows=5,
        n_columns=3,
    )


# ── compute_delta ──────────────────────────────────────


class TestComputeDelta:
    def test_inserted_detected(self, base_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
        delta = compute_delta(base_df, new_df, key_columns=["id"])
        inserted = delta[delta["_change_type"] == "inserted"]
        assert len(inserted) == 1
        assert inserted.iloc[0]["id"] == 6

    def test_deleted_detected(self, base_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
        delta = compute_delta(base_df, new_df, key_columns=["id"])
        deleted = delta[delta["_change_type"] == "deleted"]
        assert len(deleted) == 1
        assert deleted.iloc[0]["id"] == 3

    def test_updated_detected(self, base_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
        delta = compute_delta(base_df, new_df, key_columns=["id"])
        updated = delta[delta["_change_type"] == "updated"]
        assert len(updated) == 1
        assert updated.iloc[0]["id"] == 2
        assert updated.iloc[0]["value"] == 25  # new value

    def test_identical_data(self, base_df: pd.DataFrame) -> None:
        delta = compute_delta(base_df, base_df.copy(), key_columns=["id"])
        assert len(delta) == 0

    def test_empty_base(self, new_df: pd.DataFrame) -> None:
        empty = pd.DataFrame(columns=new_df.columns)
        delta = compute_delta(empty, new_df, key_columns=["id"])
        assert len(delta) == len(new_df)
        assert (delta["_change_type"] == "inserted").all()

    def test_empty_new(self, base_df: pd.DataFrame) -> None:
        empty = pd.DataFrame(columns=base_df.columns)
        delta = compute_delta(base_df, empty, key_columns=["id"])
        assert len(delta) == len(base_df)
        assert (delta["_change_type"] == "deleted").all()

    def test_all_updated(self) -> None:
        old = pd.DataFrame({"id": [1, 2], "val": [0, 0]})
        new = pd.DataFrame({"id": [1, 2], "val": [9, 9]})
        delta = compute_delta(old, new, key_columns=["id"])
        assert len(delta) == 2
        assert (delta["_change_type"] == "updated").all()

    def test_compound_key(self) -> None:
        """Test with multiple key columns."""
        old = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "home": ["A", "B"],
            "away": ["X", "Y"],
            "score": [1, 2],
        })
        new = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "home": ["A", "B"],
            "away": ["X", "Z"],  # changed away team
            "score": [1, 3],
        })
        delta = compute_delta(old, new, key_columns=["date", "home", "away"])
        assert len(delta) == 2  # 1 deleted (B vs Y) + 1 inserted (B vs Z)

    def test_no_key_columns_raises(self, base_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="required"):
            compute_delta(base_df, base_df.copy(), key_columns=[])

    def test_preserves_columns(self, base_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
        delta = compute_delta(base_df, new_df, key_columns=["id"])
        assert "_change_type" in delta.columns
        assert "_fingerprint" in delta.columns
        # All original columns should be present
        for col in base_df.columns:
            assert col in delta.columns


# ── compare_versions ───────────────────────────────────


class TestCompareVersions:
    def test_returns_diff(self, base_df: pd.DataFrame, new_df: pd.DataFrame,
                         base_info: VersionInfo, new_info: VersionInfo) -> None:
        diff = compare_versions(
            base_df, new_df, base_info, new_info,
            key_columns=["id"], include_samples=True,
        )
        assert isinstance(diff, VersionDiff)
        assert diff.base_version == "v001"
        assert diff.target_version == "v002"

    def test_counts(self, base_df: pd.DataFrame, new_df: pd.DataFrame,
                    base_info: VersionInfo, new_info: VersionInfo) -> None:
        diff = compare_versions(
            base_df, new_df, base_info, new_info,
            key_columns=["id"], include_samples=True,
        )
        assert diff.n_inserted == 1
        assert diff.n_updated == 1
        assert diff.n_deleted == 1
        assert diff.n_unchanged == 4  # 5 - 1 deleted

    def test_identical_versions(self, base_df: pd.DataFrame,
                                base_info: VersionInfo) -> None:
        diff = compare_versions(
            base_df, base_df.copy(), base_info, base_info,
            key_columns=["id"], include_samples=False,
        )
        assert diff.n_inserted == 0
        assert diff.n_updated == 0
        assert diff.n_deleted == 0
        assert diff.n_unchanged == 5

    def test_changed_columns_detected(self, base_df: pd.DataFrame, new_df: pd.DataFrame,
                                      base_info: VersionInfo, new_info: VersionInfo) -> None:
        diff = compare_versions(
            base_df, new_df, base_info, new_info,
            key_columns=["id"], include_samples=False,
        )
        # The id=2 row had value changed from 20 to 25
        assert "value" in diff.changed_columns

    def test_metadata_changes(self, base_df: pd.DataFrame,
                              base_info: VersionInfo, new_info: VersionInfo) -> None:
        # Both versions have identical metadata → no metadata changes
        diff = compare_versions(
            base_df, base_df.copy(), base_info, new_info,
            key_columns=["id"], include_samples=False,
        )
        assert len(diff.metadata_changes) == 0

    def test_sample_rows(self, base_df: pd.DataFrame, new_df: pd.DataFrame,
                         base_info: VersionInfo, new_info: VersionInfo) -> None:
        diff = compare_versions(
            base_df, new_df, base_info, new_info,
            key_columns=["id"], include_samples=True,
        )
        assert diff.inserted_rows is not None
        assert len(diff.inserted_rows) >= 1
        assert diff.deleted_rows is not None
        assert len(diff.deleted_rows) >= 1

    def test_no_samples(self, base_df: pd.DataFrame, new_df: pd.DataFrame,
                        base_info: VersionInfo, new_info: VersionInfo) -> None:
        diff = compare_versions(
            base_df, new_df, base_info, new_info,
            key_columns=["id"], include_samples=False,
        )
        assert diff.inserted_rows is None
        assert diff.deleted_rows is None
        assert diff.updated_rows is None

    def test_same_version_id(self, base_df: pd.DataFrame,
                             base_info: VersionInfo) -> None:
        diff = compare_versions(
            base_df, base_df.copy(), base_info, base_info,
            key_columns=["id"], include_samples=False,
        )
        assert diff.n_unchanged == 5
