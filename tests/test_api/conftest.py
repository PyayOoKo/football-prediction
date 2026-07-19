"""
Shared fixtures for API tests.

Clears PREDICTION_API_KEY before each test to prevent the .env file's
value from leaking into tests that use ``clear=False`` with
``patch.dict(os.environ, ...)``.

**Important**: The key is kept clear after each test (not restored).
Tests that need PREDICTION_API_KEY must set it explicitly via their own
``patch.dict()`` blocks or fixtures.  Restoring the original value would
re-infect later tests (pytest fixture teardown runs before the conftest's
tear-down, so a test's ``patch.dict()`` restore would revert to no-key,
and then the conftest restore would add the .env hash back — infecting
the next test).
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_prediction_api_key() -> None:
    """Remove PREDICTION_API_KEY from the environment before every test.

    The project's .env file sets this variable (loaded via load_dotenv()
    in api/main.py).  Many auth tests assume a clean environment and use
    ``patch.dict(os.environ, ..., clear=False)``, which means any
    pre-existing value leaks through.

    This fixture strips the variable before each test and ensures it
    stays stripped after, so every test starts with a clean slate.
    """
    os.environ.pop("PREDICTION_API_KEY", None)
    yield
    os.environ.pop("PREDICTION_API_KEY", None)
