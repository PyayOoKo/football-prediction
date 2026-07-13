"""Pytest configuration and fixtures for monitoring tests.

On Windows, SQLite file handles are not released until the connection
is explicitly closed via wal_checkpoint + journal_mode=DELETE, which
causes PermissionError when pytest tries to clean up temporary
directories. This conftest ensures all connections are closed
before test teardown.
"""

from __future__ import annotations

import gc
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import pytest

from src.monitoring import Monitor
from src.monitoring.store import MonitoringStore


def force_close_connections() -> None:
    """Forcibly close all lingering SQLite connections.

    Call this before TemporaryDirectory cleanup to avoid
    PermissionError on Windows.
    """
    gc.collect()
    gc.collect()
    for obj in gc.get_objects():
        if isinstance(obj, sqlite3.Connection):
            try:
                obj.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            try:
                obj.close()
            except Exception:
                pass


@pytest.fixture(autouse=True)
def _close_all_connections() -> Iterator[None]:
    """Yield, then forcibly close all SQLite connections."""
    yield
    force_close_connections()


@pytest.fixture
def store() -> Iterator[MonitoringStore]:
    """Create a MonitoringStore in a temporary directory."""
    tmp = TemporaryDirectory()
    db_path = Path(tmp.name) / "monitor.db"
    s = MonitoringStore(db_path=db_path, retention_days=90)
    yield s
    s.close()
    force_close_connections()
    try:
        tmp.cleanup()
    except PermissionError:
        pass


@pytest.fixture
def monitor() -> Iterator[Monitor]:
    """Create a Monitor with temp paths."""
    tmp = TemporaryDirectory()
    tmp_path = Path(tmp.name)
    m = Monitor(
        db_path=str(tmp_path / "monitor.db"),
        output_dir=str(tmp_path / "reports"),
        data_dir=str(tmp_path / "data"),
        retention_days=90,
    )
    yield m
    m.close()
    m.store.close()
    force_close_connections()
    try:
        tmp.cleanup()
    except PermissionError:
        pass
