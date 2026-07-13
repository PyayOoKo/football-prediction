"""
Pytest configuration for data profiling tests.

Cleans the reports/profiling directory before each test to ensure
isolation. Reports are auto-generated and safe to delete.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.data_profiling.cli import REPORTS_DIR


@pytest.fixture(autouse=True)
def clean_reports_dir() -> None:
    """Clean the profiling reports directory before each test."""
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    yield
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
