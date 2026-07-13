"""Pytest configuration for cache tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the asyncio marker."""
    config.addinivalue_line("markers", "asyncio: mark test as async")
