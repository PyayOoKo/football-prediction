"""
Shared fixtures for service tests.

Bootstraps the DI container with a mock ConfigProvider so that
PredictionService and TrainingService (which call
``get_container().resolve(ConfigProvider)`` in __init__) can
be instantiated without a real configuration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.di_container import (
    configure_container,
    reset_container,
    set_container,
)


@pytest.fixture(autouse=True)
def _bootstrap_di_container() -> None:
    """Register a default mock ConfigProvider before every service test.

    ``configure_container()`` returns a configured container but does
    **not** set it globally — we must call ``set_container()`` so that
    ``get_container().resolve(ConfigProvider)`` in service constructors
    finds the mock config.

    The container is cleared after each test to prevent cross-test
    pollution.
    """
    mock_cfg = MagicMock()
    mock_cfg.paths.models = Path("/tmp/models")
    mock_cfg.paths.raw = Path("/tmp/raw")
    mock_cfg.paths.processed = Path("/tmp/processed")
    mock_cfg.paths.data = Path("/tmp/data")
    mock_cfg.data.results_file = "results_clean.csv"
    mock_cfg.preprocessing.normalise_teams = False
    mock_cfg.worldcup.data_path = "/tmp/worldcup.csv"
    mock_cfg.train.model_type = "logistic_regression"
    mock_cfg.train.cv_folds = 5
    mock_cfg.feature_selection.enabled = False
    mock_cfg.feature_selection.drop_redundant_first = False
    mock_cfg.feature_selection.correlation_threshold = 0.95
    mock_cfg.feature_selection.method = "mutual_info"
    mock_cfg.feature_selection.n_features = 30
    mock_cfg.feature_selection.importance_threshold = 0.01
    mock_cfg.data.split_ratios = [0.7, 0.15, 0.15]

    container = configure_container(mock_cfg)
    set_container(container)
    yield
    reset_container()
