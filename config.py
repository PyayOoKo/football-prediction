"""
Project-wide configuration — re-exported from ``src.config.settings``.

Previously this file contained its own independent ``Config`` dataclass and
all sub-config dataclasses.  These have been unified into a single hierarchy
in ``src/config/settings.py`` to eliminate the duplicate config system.

All existing imports (``from config import config``) continue to work
transparently — this module simply re-exports the singleton from
``src.config.settings``.
"""

from __future__ import annotations

import warnings

from src.config.settings import (  # noqa: F401
    APIConfig,
    AppConfig,
    BacktestConfig,
    BlendConfig,
    ConfidenceConfig,
    Config,
    DataCollectionConfig,
    DataConfig,
    DatabaseConfig,
    DixonColesConfig,
    EloConfig,
    EnsembleConfig,
    EvalConfig,
    ExtendedFeaturesConfig,
    FeatureConfig,
    FeatureSelectionConfig,
    HyperTuneConfig,
    LoggingConfig,
    OddsAPIConfig,
    OddsConfig,
    Paths,
    PlayerFeaturesConfig,
    PlayerInfoConfig,
    PoissonConfig,
    PredictConfig,
    PreprocessingConfig,
    RefereeCollectorConfig,
    RefereeConfig,
    ScheduleConfig,
    StatsBombCollectorConfig,
    TransferCollectorConfig,
    TrainConfig,
    ValueBetConfig,
    WeatherCollectorConfig,
    WeatherConfig,
    WorldCupConfig,
    XgConfig,
    config,
)

from src.config import configure_logging  # noqa: F401

__all__ = [
    "APIConfig",
    "AppConfig",
    "BacktestConfig",
    "BlendConfig",
    "ConfidenceConfig",
    "Config",
    "DataCollectionConfig",
    "DataConfig",
    "DatabaseConfig",
    "DixonColesConfig",
    "EloConfig",
    "EnsembleConfig",
    "EvalConfig",
    "ExtendedFeaturesConfig",
    "FeatureConfig",
    "FeatureSelectionConfig",
    "HyperTuneConfig",
    "LoggingConfig",
    "OddsAPIConfig",
    "OddsConfig",
    "Paths",
    "PlayerFeaturesConfig",
    "PlayerInfoConfig",
    "PoissonConfig",
    "PredictConfig",
    "PreprocessingConfig",
    "RefereeCollectorConfig",
    "RefereeConfig",
    "ScheduleConfig",
    "StatsBombCollectorConfig",
    "TransferCollectorConfig",
    "TrainConfig",
    "ValueBetConfig",
    "WeatherCollectorConfig",
    "WeatherConfig",
    "WorldCupConfig",
    "XgConfig",
    "config",
]

warnings.warn(
    "config.py is deprecated. Import from 'src.config.settings' directly.",
    DeprecationWarning,
    stacklevel=2,
)
