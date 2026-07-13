"""
Tests for ``src.data_versioning.manager``.

Covers the full VersionManager lifecycle: create, list, compare,
rollback, verify, and load current data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_versioning import VersionManager
from src.data_versioning.models import VersionDiff, VersionSummary


@pytest.fixture
def mgr() -> VersionManager:
    tmp = tempfile.mkdtemp()
    return VersionManager(
        data_dir=Path(tmp) / "versions",
        fingerprint_columns=["id"],
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
        "value": [10, 20, 30],
    })


class TestVersionManager:
    def test_create_version(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        info = mgr.create_version(sample_df, source="test", league="E0", user="pytest")
        assert info.version_id == "v001"
        assert info.n_rows == 3
        assert info.source == "test"
        assert info.league == "E0"
        assert info.user == "pytest"

    def test_create_empty_raises(self, mgr: VersionManager) -> None:
        with pytest.raises(ValueError, match="empty"):
            mgr.create_version(pd.DataFrame())

    def test_create_multiple(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        v1 = mgr.create_version(sample_df, source="test")
        v2 = mgr.create_version(sample_df, source="test")
        assert v1.version_id == "v001"
        assert v2.version_id == "v002"

    def test_custom_version_id(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        info = mgr.create_version(sample_df, source="test", version_id="my_custom_v1")
        assert info.version_id == "my_custom_v1"

    def test_list_versions(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test", league="E0")
        mgr.create_version(sample_df, source="test", league="E1")

        versions = mgr.list_versions()
        assert len(versions) == 2
        assert all(isinstance(v, VersionSummary) for v in versions)
        assert versions[0].version_id == "v001"
        assert versions[1].version_id == "v002"

    def test_get_version(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test", league="E0")
        info = mgr.get_version("v001")
        assert info is not None
        assert info.league == "E0"

    def test_get_nonexistent_version(self, mgr: VersionManager) -> None:
        info = mgr.get_version("nonexistent")
        assert info is None

    def test_get_current_version(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test")
        current = mgr.get_current_version()
        assert current is not None
        assert current.version_id == "v001"

    def test_get_current_version_no_versions(self, mgr: VersionManager) -> None:
        current = mgr.get_current_version()
        assert current is None

    def test_compare(self, mgr: VersionManager) -> None:
        df1 = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        df2 = pd.DataFrame({"id": [1, 3, 4], "val": ["a", "c", "d"]})

        mgr.create_version(df1, source="test", version_id="v001")
        mgr.create_version(df2, source="test", version_id="v002")

        diff = mgr.compare("v001", "v002", include_samples=True)
        assert diff.n_deleted == 1  # id=2
        assert diff.n_inserted == 1  # id=4
        assert diff.n_updated == 0

    def test_compare_same_version(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test")
        diff = mgr.compare("v001", "v001")
        assert diff.n_unchanged == 3
        assert diff.n_inserted == 0
        assert diff.n_deleted == 0

    def test_compare_missing_base_raises(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test")
        with pytest.raises(ValueError, match="not found"):
            mgr.compare("nonexistent", "v001")

    def test_rollback(self, mgr: VersionManager) -> None:
        df1 = pd.DataFrame({"id": [1], "val": ["old"]})
        df2 = pd.DataFrame({"id": [1], "val": ["new"]})

        mgr.create_version(df1, source="test", version_id="v001")
        mgr.create_version(df2, source="test", version_id="v002")

        current_before = mgr.get_current_version()
        assert current_before.version_id == "v002"

        # Rollback to v001
        info = mgr.rollback("v001", create_backup=False)
        assert info.version_id == "v001"

        current_after = mgr.get_current_version()
        assert current_after.version_id == "v001"

    def test_rollback_with_backup(self, mgr: VersionManager) -> None:
        df1 = pd.DataFrame({"id": [1], "val": ["a"]})
        df2 = pd.DataFrame({"id": [2], "val": ["b"]})

        mgr.create_version(df1, source="test")
        mgr.create_version(df2, source="test")
        mgr.rollback("v001", create_backup=True)

        versions = mgr.list_versions()
        backup_ids = [v.version_id for v in versions if "backup" in v.version_id]
        assert len(backup_ids) >= 1

    def test_verify_all(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test")
        mgr.create_version(sample_df, source="test")

        results = mgr.verify()
        assert len(results) == 2
        assert all(r["valid"] for r in results.values())

    def test_verify_single(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test")
        mgr.create_version(sample_df, source="test")

        results = mgr.verify(version_id="v001")
        assert "v001" in results
        assert results["v001"]["valid"]

    def test_load_current_data(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        mgr.create_version(sample_df, source="test")
        loaded = mgr.load_current_data()
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_load_current_data_no_version_raises(self, mgr: VersionManager) -> None:
        with pytest.raises(ValueError, match="No current version"):
            mgr.load_current_data()

    def test_create_version_from_csv(self, mgr: VersionManager) -> None:
        tmp = tempfile.mktemp(suffix=".csv")
        df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        df.to_csv(tmp, index=False)

        info = mgr.create_version_from_csv(
            tmp, source="csv_test", user="pytest", parse_dates=[],
        )
        assert info.n_rows == 2
        assert info.source == "csv_test"

    def test_create_version_from_csv_nonexistent(self, mgr: VersionManager) -> None:
        with pytest.raises(FileNotFoundError):
            mgr.create_version_from_csv("/nonexistent/path.csv")

    def test_version_with_tags(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        info = mgr.create_version(
            sample_df, source="test",
            tags={"env": "test", "pipeline": "nightly"},
        )
        assert info.tags["env"] == "test"
        assert info.tags["pipeline"] == "nightly"

    def test_version_with_import_duration(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        info = mgr.create_version(
            sample_df, source="test", import_duration=12.345,
        )
        assert info.import_duration == 12.345

    def test_version_with_notes(self, mgr: VersionManager, sample_df: pd.DataFrame) -> None:
        info = mgr.create_version(
            sample_df, source="test",
            notes="Manual import of test data",
        )
        assert info.notes == "Manual import of test data"
