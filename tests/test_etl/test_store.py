"""
Tests for the ETL storage stage — FileStore, DatabaseStore.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.etl.store import DataStore, DatabaseStore, FileStore
from src.etl.models import PipelineStage, StageStatus


# ═══════════════════════════════════════════════════════════
#  DataStore (abstract base)
# ═══════════════════════════════════════════════════════════

class TestDataStore:
    def test_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            DataStore()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════════════
#  FileStore
# ═══════════════════════════════════════════════════════════

class TestFileStore:
    def test_write_csv(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path, format="csv")
        data = [{"id": 1, "name": "Test"}, {"id": 2, "name": "Test2"}]
        result = store.write(data, source="test_source")

        assert result.status == StageStatus.SUCCESS
        assert result.records_out == 2
        assert "file_size_mb" in result.metrics

        # Check file was created
        files = list(tmp_path.glob("*.csv"))
        assert len(files) >= 1

    def test_write_parquet(self, tmp_path: Path) -> None:
        import importlib
        # Check if parquet engine is available
        parquet_available = importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")
        if not parquet_available:
            store = FileStore(output_dir=tmp_path, format="parquet")
            data = [{"id": 1, "name": "Test"}]
            result = store.write(data, source="test")
            assert result.status == StageStatus.FAILED
        else:
            store = FileStore(output_dir=tmp_path, format="parquet")
            data = [{"id": 1, "name": "Test"}]
            result = store.write(data, source="test")
            assert result.status == StageStatus.SUCCESS
            files = list(tmp_path.glob("*.parquet"))
            assert len(files) >= 1

    def test_write_empty_data_warning(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path)
        result = store.write([])

        assert result.status == StageStatus.WARNING
        assert "No data" in result.errors[0]

    def test_unsupported_format(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path, format="json")
        data = [{"id": 1}]
        result = store.write(data, source="test")

        assert result.status == StageStatus.FAILED
        assert "Unsupported format" in result.errors[0]

    def test_custom_filename(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path, format="csv", filename="custom.csv")
        data = [{"x": 1}]
        store.write(data)

        assert (tmp_path / "custom.csv").exists()

    def test_filename_from_kwargs(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path)
        data = [{"x": 1}]
        store.write(data, source="s1", filename="override.csv")
        assert (tmp_path / "override.csv").exists()

    def test_batch_size_parameter_accepted(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path)
        data = [{"x": 1} for _ in range(100)]
        result = store.write(data, batch_size=50, source="test")
        assert result.status == StageStatus.SUCCESS
        assert result.records_out == 100

    def test_output_dir_created(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "dir"
        assert not new_dir.exists()
        store = FileStore(output_dir=new_dir)
        assert new_dir.exists()

    def test_resolve_path_generates_filename(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path)
        path = store._resolve_path("test_source")
        assert str(path).endswith(".csv")
        assert "test_source" in str(path)

    def test_warning_on_no_data_no_crash(self, tmp_path: Path) -> None:
        store = FileStore(output_dir=tmp_path)
        result = store.write([], source="empty")
        assert result.status == StageStatus.WARNING


# ═══════════════════════════════════════════════════════════
#  DatabaseStore
# ═══════════════════════════════════════════════════════════

class TestDatabaseStore:
    def test_init_defaults(self) -> None:
        store = DatabaseStore(model_class=MagicMock())
        assert store.unique_columns == []
        assert store.batch_size == 1000

    def test_init_with_unique(self) -> None:
        store = DatabaseStore(
            model_class=MagicMock(),
            unique_columns=["match_id"],
            batch_size=500,
        )
        assert store.unique_columns == ["match_id"]
        assert store.batch_size == 500

    def test_write_empty_data(self) -> None:
        store = DatabaseStore(model_class=MagicMock())
        result = store.write([])
        assert result.status == StageStatus.WARNING
        assert "No data" in result.errors[0]

    @patch("src.etl.store.get_session")
    def test_write_success(self, mock_get_session) -> None:
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session.return_value = mock_session

        # Mock model_class with __table__ attribute
        mock_table = MagicMock()
        mock_table.insert.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.__table__ = mock_table

        store = DatabaseStore(model_class=mock_model)
        data = [{"id": 1, "name": "Test"}]
        result = store.write(data)

        assert result.status == StageStatus.SUCCESS
        assert result.records_out == 1

    @patch("src.etl.store.get_session")
    def test_write_batches(self, mock_get_session) -> None:
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session.return_value = mock_session

        mock_table = MagicMock()
        mock_table.insert.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.__table__ = mock_table

        store = DatabaseStore(model_class=mock_model, batch_size=2)
        data = [{"id": i} for i in range(5)]
        result = store.write(data)

        assert result.status == StageStatus.SUCCESS
        assert result.metrics["inserted"] == float(5)

    @patch("src.etl.store.get_session")
    def test_batch_error_handling(self, mock_get_session) -> None:
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_get_session.return_value = mock_session

        mock_model = MagicMock()
        mock_model.__table__ = MagicMock()

        store = DatabaseStore(model_class=mock_model, batch_size=2)
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = store.write(data, batch_size=2)

        assert result.status in (StageStatus.WARNING, StageStatus.FAILED)

    @patch("src.etl.store.get_session")
    def test_session_error_rollback(self, mock_get_session) -> None:
        mock_session = MagicMock()
        mock_session.__enter__.side_effect = RuntimeError("Connection lost")
        mock_get_session.return_value = mock_session

        store = DatabaseStore(model_class=MagicMock())
        data = [{"id": 1}]
        result = store.write(data)

        assert result.status == StageStatus.FAILED
