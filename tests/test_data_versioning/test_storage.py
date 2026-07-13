"""
Tests for ``src.data_versioning.storage``.

Covers hashing, fingerprinting, snapshot save/load, delta computation,
metadata management, version listing, rollback, and integrity verification.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_versioning.models import VersionInfo
from src.data_versioning.storage import (
    VersionStorage,
    compute_fingerprint,
    compute_hash,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "home_team": ["Arsenal", "Chelsea", "Liverpool"],
        "away_team": ["Man Utd", "Man City", "Tottenham"],
        "result": ["H", "A", "D"],
        "home_goals": [2, 1, 1],
        "away_goals": [0, 3, 1],
    })


@pytest.fixture
def storage() -> VersionStorage:
    tmp = tempfile.mkdtemp()
    return VersionStorage(base_dir=Path(tmp) / "versions")


# ── Hash ────────────────────────────────────────────────


class TestComputeHash:
    def test_deterministic(self, sample_df: pd.DataFrame) -> None:
        h1 = compute_hash(sample_df)
        h2 = compute_hash(sample_df.copy())
        assert h1 == h2

    def test_content_change(self, sample_df: pd.DataFrame) -> None:
        h1 = compute_hash(sample_df)
        df2 = sample_df.copy()
        df2.iloc[0, df2.columns.get_loc("home_goals")] = 99
        h2 = compute_hash(df2)
        assert h1 != h2

    def test_column_order_independent(self, sample_df: pd.DataFrame) -> None:
        h1 = compute_hash(sample_df)
        shuffled = sample_df[sample_df.columns[::-1]]
        h2 = compute_hash(shuffled)
        assert h1 == h2

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"a": []})
        h = compute_hash(df)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex digest length

    def test_large_dataframe_chunked(self) -> None:
        """Verify that hashing works correctly with chunking (>50k rows)."""
        n = 100_000
        df = pd.DataFrame({
            "id": range(n),
            "value": np.random.randn(n),
        })
        h = compute_hash(df)
        assert isinstance(h, str)
        assert len(h) == 64


class TestComputeFingerprint:
    def test_unique_fingerprints(self, sample_df: pd.DataFrame) -> None:
        fps = compute_fingerprint(sample_df, ["date", "home_team", "away_team"])
        assert fps.nunique() == len(sample_df)  # all different

    def test_deterministic(self, sample_df: pd.DataFrame) -> None:
        fps1 = compute_fingerprint(sample_df, ["date", "home_team", "away_team"])
        fps2 = compute_fingerprint(sample_df.copy(), ["date", "home_team", "away_team"])
        assert fps1.tolist() == fps2.tolist()

    def test_single_key_column(self, sample_df: pd.DataFrame) -> None:
        fps = compute_fingerprint(sample_df, ["date"])
        assert len(fps) == len(sample_df)

    def test_missing_key_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            compute_fingerprint(sample_df, ["nonexistent"])

    def test_empty_key_raises(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="required"):
            compute_fingerprint(sample_df, [])


# ── Save / Load ─────────────────────────────────────────


class TestVersionStorage:
    def test_save_and_load(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        info = storage.save_version(
            df=sample_df,
            version_id="v001",
            source="test",
            league="E0",
            season="2425",
            user="pytest",
            notes="Test version",
        )

        assert info.version_id == "v001"
        assert info.n_rows == 3
        assert info.n_columns == 6
        assert info.source == "test"
        assert info.league == "E0"
        assert info.season == "2425"
        assert info.user == "pytest"
        assert info.notes == "Test version"
        assert info.previous_version is None
        assert len(info.hash) == 64

        # Load back
        loaded = storage.load_snapshot("v001")
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_metadata_json(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        info = storage.save_version(sample_df, "v001", source="test")
        meta_path = storage._metadata_path("v001")
        assert meta_path.exists()

        with open(meta_path, "r") as f:
            data = json.load(f)
        assert data["version_id"] == "v001"
        assert data["n_rows"] == 3
        assert data["source"] == "test"

    def test_parquet_snapshot(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        storage.save_version(sample_df, "v001", source="test")
        parquet_path = storage._snapshot_path("v001")
        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"

        # Read back via parquet
        df_back = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(df_back, sample_df)

    def test_current_link(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        storage.save_version(sample_df, "v001", source="test")
        storage.save_version(sample_df, "v002", source="test")

        current_id = storage.get_current_version_id()
        assert current_id == "v002", f"Expected v002, got {current_id}"

    def test_list_versions(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        storage.save_version(sample_df, "v001", source="test")
        storage.save_version(sample_df, "v002", source="test")

        versions = storage.list_versions()
        assert len(versions) == 2
        assert versions[0].version_id == "v001"
        assert versions[1].version_id == "v002"

    def test_delete_version(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        storage.save_version(sample_df, "v001", source="test")
        assert storage._version_dir("v001").exists()

        storage.delete_version("v001")
        assert not storage._version_dir("v001").exists()

    def test_delta(self) -> None:
        """Test delta computation uses storage.fingerprint_columns."""
        custom_storage = VersionStorage(
            base_dir=Path(tempfile.mkdtemp()) / "versions",
            fingerprint_columns=["id"],
        )
        df1 = pd.DataFrame({
            "id": [1, 2, 3],
            "val": ["a", "b", "c"],
        })
        df2 = pd.DataFrame({
            "id": [1, 2, 4],
            "val": ["a", "b", "d"],
        })

        v1 = custom_storage.save_version(df1, "v001", source="test")
        v2 = custom_storage.save_version(df2, "v002", prev_version=v1, source="test")

        delta = custom_storage.load_delta("v002")
        assert len(delta) > 0
        change_types = delta["_change_type"].value_counts().to_dict()
        assert change_types.get("deleted", 0) == 1  # id=3 deleted
        assert change_types.get("inserted", 0) == 1  # id=4 inserted

    def test_empty_delta(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        """Same data = empty delta."""
        v1 = storage.save_version(sample_df, "v001", source="test")
        v2 = storage.save_version(sample_df, "v002", prev_version=v1, source="test")

        delta = storage.load_delta("v002")
        assert len(delta) == 0

    def test_load_nonexistent_version(self, storage: VersionStorage) -> None:
        with pytest.raises(FileNotFoundError):
            storage.load_snapshot("nonexistent")

    def test_metadata_nonexistent(self, storage: VersionStorage) -> None:
        info = storage.load_metadata("nonexistent")
        assert info is None

    def test_rollback(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        df2 = sample_df.copy()
        df2["home_goals"] = [9, 9, 9]

        storage.save_version(sample_df, "v001", source="test")
        storage.save_version(df2, "v002", source="test")

        # Rollback to v001
        rolled = storage.rollback("v001", create_backup=False)
        assert rolled.version_id == "v001"

        current_id = storage.get_current_version_id()
        assert current_id == "v001"

    def test_rollback_with_backup(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        storage.save_version(sample_df, "v001", source="test")
        storage.rollback("v001", create_backup=True)

        # Backup version should exist
        versions = storage.list_versions()
        backup_ids = [v.version_id for v in versions if "backup" in v.version_id]
        assert len(backup_ids) > 0

    def test_integrity_valid(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        storage.save_version(sample_df, "v001", source="test")
        assert storage.verify_integrity("v001") is True

    def test_integrity_tampered(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        info = storage.save_version(sample_df, "v001", source="test")

        # Tamper with the data file
        parquet_path = storage._snapshot_path("v001")
        df_tampered = pd.read_parquet(parquet_path)
        df_tampered.iloc[0, 0] = "TAMPERED"
        df_tampered.to_parquet(parquet_path)

        assert storage.verify_integrity("v001") is False

    def test_log_metadata(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        info = storage.save_version(
            df=sample_df,
            version_id="v001",
            source="test",
            league="E0",
            season="2425",
            user="pytest",
            import_duration=1.234,
            tags={"env": "test", "branch": "main"},
        )

        assert info.user == "pytest"
        assert info.import_duration == 1.234
        assert info.tags == {"env": "test", "branch": "main"}

    def test_previous_version(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        v1 = storage.save_version(sample_df, "v001", source="test")
        v2 = storage.save_version(sample_df, "v002", prev_version=v1, source="test")

        assert v1.previous_version is None
        assert v2.previous_version == "v001"

    def test_column_pruning(self, storage: VersionStorage, sample_df: pd.DataFrame) -> None:
        storage.save_version(sample_df, "v001", source="test")
        loaded = storage.load_snapshot("v001", columns=["date", "home_team"])
        assert list(loaded.columns) == ["date", "home_team"]

    def test_chunked_write(self, storage: VersionStorage) -> None:
        """Write > chunk_size rows to test chunked writing."""
        n = 10_000
        df = pd.DataFrame({
            "id": range(n),
            "value": list(range(n)),
        })
        info = storage.save_version(df, "v001", source="test")
        assert info.n_rows == n

        loaded = storage.load_snapshot("v001")
        assert len(loaded) == n

    def test_multiple_versions_metadata_preserved(
        self, storage: VersionStorage, sample_df: pd.DataFrame
    ) -> None:
        for i in range(3):
            storage.save_version(sample_df, f"v{i+1:03d}", source="test", league=f"L{i}")

        versions = storage.list_versions()
        assert len(versions) == 3
        for j, v in enumerate(versions):
            assert v.league == f"L{j}"
