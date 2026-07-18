"""
Services package — business logic orchestration layer.

Services sit between the API/web layer and the data access layer.
They encapsulate domain logic, coordinate multiple repositories
and external APIs, and are the primary unit for integration tests.

Modules
-------
prediction_service
    Coordinates model inference, feature building, and result storage.
training_service
    Manages model training lifecycle (scheduling, versioning, evaluation).
betting_service
    Value betting calculations, Kelly criterion, bankroll management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import config as _global_config

logger = logging.getLogger(__name__)

__all__ = [
    "PredictionService",
    "TrainingService",
    "resolve_data_path",
    "add_target_col",
    "load_and_prepare",
]

_RESULT_TO_TARGET = {"H": 2, "D": 1, "A": 0}


def add_target_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``target`` column from ``result`` (H=2, D=1, A=0, NaN=-1).

    Many service methods need this before calling ``build_features()``.
    This is a no-op if the column already exists.
    """
    if "target" not in df.columns:
        df["target"] = df["result"].map(_RESULT_TO_TARGET).fillna(-1).astype("int8")
    return df


def resolve_data_path(
    hint: str | Path | None = None, config: Any | None = None
) -> Path:
    """Resolve the most likely match-data CSV path.

    Tries, in order:
    1. The *hint* argument (if given).
    2. ``config.paths.raw / config.data.results_file``
    3. ``config.paths.raw / "worldcup_all.csv"``
    4. ``config.paths.raw / "results.csv"``
    5. ``config.worldcup.data_path`` (string, resolved relative to project root).

    Parameters
    ----------
    hint : str | Path, optional
        Explicit path to a CSV file.  If given, returned immediately.
    config : Config, optional
        Config instance for dependency injection.  Defaults to the
        global singleton ``config``.

    Returns
    -------
    Path
        The first existing path found, or a default candidate if none exist.
    """
    cfg = config or _global_config
    if hint is not None:
        return Path(hint)

    candidates = [
        cfg.paths.raw / cfg.data.results_file,
        cfg.paths.raw / "worldcup_all.csv",
        cfg.paths.raw / "results.csv",
        Path(cfg.worldcup.data_path),
    ]
    seen = set()
    for p in candidates:
        resolved = Path(p).resolve()
        if resolved not in seen:
            seen.add(resolved)
            if resolved.exists():
                return resolved

    return candidates[0]


def load_and_prepare(
    data_path: str | Path | None = None,
    add_temporal: bool = True,
    config: Any | None = None,
) -> pd.DataFrame:
    """Load match data through the full pipeline and return a clean DataFrame.

    Convenience wrapper that chains:
        ``DataLoader.load_results()`` → ``DataCleaner`` (auto-detected)
        → ``DataPreprocessor.fit_transform()`` → ``add_target_col()``

    Parameters
    ----------
    data_path : str | Path, optional
        Passed to :func:`resolve_data_path` then to
        :meth:`DataLoader.load_results`.
    add_temporal : bool
        Whether the ``DataPreprocessor`` adds year/month/day-of-week
        columns (default ``True``).
    config : Config, optional
        Config instance for DI. Defaults to the global singleton.

    Returns
    -------
    pd.DataFrame
        Cleaned, preprocessed DataFrame with ``target`` column present.
    """
    cfg = config or _global_config
    from src.data import DataCleaner, DataLoader, DataPreprocessor

    resolved = resolve_data_path(data_path, config=cfg)
    loader = DataLoader()
    df = loader.load_results(resolved)

    # Auto-detect source format from column signatures
    if "Div" in df.columns or "FTHG" in df.columns:
        df = DataCleaner.football_data_co_uk(df)
    elif "utcDate" in df.columns or "score.fullTime.home" in df.columns:
        df = DataCleaner.football_data_org(df)
    else:
        logger.info("Source not recognised — skipping source-specific cleaning")

    preprocessor = DataPreprocessor(
        normalise_teams=cfg.preprocessing.normalise_teams,
        add_temporal_features=add_temporal,
    )
    df = preprocessor.fit_transform(df)
    df = add_target_col(df)
    return df


# Late imports to avoid circular dependencies between the two services
from src.services.prediction_service import PredictionService  # noqa: E402
from src.services.training_service import TrainingService  # noqa: E402
