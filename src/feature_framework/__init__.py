"""
Feature Engineering Framework — production-grade infrastructure for creating,
versioning, computing, and serving features for every ML model.

This framework becomes the **single source of truth** for all feature
engineering in the football prediction platform. Every future model
must define its features through this framework.

Architecture
------------
The framework wraps and extends ``src.feature_store`` with:

    ┌─────────────────────────────────────────────────────────────┐
    │                  FeaturePipeline (orchestrator)              │
    │  Resolves DAG → parallel computation → validate → store     │
    └────┬───────────┬────────────┬──────────┬────────────────────┘
         │           │            │          │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌───▼──────────────────┐
    │Feature  │ │Parallel │ │Feature  │ │FeatureStore          │
    │Transform│ │Computer │ │Config   │ │(from src.feature_     │
    │(ABC)    │ │(pool)   │ │(YAML)   │ │  store)              │
    └─────────┘ └─────────┘ └─────────┘ └──────────────────────┘

**Key design decisions:**
- **Plugin-based**: features auto-discover via ``FeaturePluginRegistry``
- **Declarative**: feature definitions in YAML, not code
- **Parallel by default**: computation uses configurable thread/process pools
- **Resumable**: interrupted batches continue from last checkpoint
- **Versioned**: every feature definition has full version history
- **Validated**: every computed value passes configurable validation rules

Quick Start
-----------
::

    from src.feature_framework import FeaturePipeline

    pipeline = FeaturePipeline(config_path="features.yaml")
    report = pipeline.run(
        entity_type="match",
        entity_ids=[1, 2, 3],
        trigger="manual",
    )
    print(report.to_dict())
"""

from __future__ import annotations

from src.feature_framework.base import (
    FeatureTransformer,
    FeaturePipelineABC,
)
from src.feature_framework.config import (
    FeatureConfig,
    FeatureDefinitionSchema,
    load_feature_config,
)
from src.feature_framework.decorators import timeit, log_call, retry
from src.feature_framework.exceptions import (
    FeatureComputationError,
    FeatureNotFoundError,
    FeatureValidationError,
    FeatureDependencyCycleError,
    FeatureConfigError,
)
from src.feature_framework.league_strength import (
    LeagueStrengthEngine,
    LeagueStrengthRecord,
    create_league_strength_engine,
)
from src.feature_framework.models import (
    ComputationResult,
    PipelineReport,
    FeatureMetadata,
    TransformContext,
    FeatureSet,
)
from src.feature_framework.parallel import (
    ParallelComputer,
    make_thread_pool,
    make_process_pool,
)
from src.feature_framework.pipeline import FeaturePipeline
from src.feature_framework.plugins import FeaturePluginRegistry
from src.feature_framework.orchestrator import (
    FeatureOrchestrator,
    OrchestratorReport,
    FeatureExecutionRecord,
    FeatureStatus,
    OrchestratorStage,
)

__all__ = [
    # Pipeline
    "FeaturePipeline",
    "FeaturePipelineABC",
    # Orchestrator
    "FeatureOrchestrator",
    "OrchestratorReport",
    "FeatureExecutionRecord",
    "FeatureStatus",
    "OrchestratorStage",
    # Base
    "FeatureTransformer",
    # League Strength
    "LeagueStrengthEngine",
    "LeagueStrengthRecord",
    "create_league_strength_engine",
    # Config
    "FeatureConfig",
    "FeatureDefinitionSchema",
    "load_feature_config",
    # Parallel
    "ParallelComputer",
    "make_thread_pool",
    "make_process_pool",
    # Models
    "ComputationResult",
    "PipelineReport",
    "FeatureMetadata",
    "TransformContext",
    "FeatureSet",
    # Plugins
    "FeaturePluginRegistry",
    # Decorators
    "timeit",
    "log_call",
    "retry",
    # Exceptions
    "FeatureComputationError",
    "FeatureNotFoundError",
    "FeatureValidationError",
    "FeatureDependencyCycleError",
    "FeatureConfigError",
]

__version__ = "0.1.0"
