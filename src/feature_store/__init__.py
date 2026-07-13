"""
Feature Store — production-grade feature storage infrastructure for ML models.

Provides centralised registry, versioned storage, validation, computation
orchestration, caching, lineage tracking, and CLI management for all
football prediction features.

Architecture
------------
FeatureRegistry
    Manages the feature catalog — definitions, versions, dependencies,
    lifecycle states (draft → active → deprecated → retired).
FeatureStore
    CRUD operations for feature values with batch/incremental support,
    computation batch tracking, and feature vector assembly.
FeatureValidator
    Validation rules engine (schema, bounds, cardinality, consistency).
FeatureComputer (abstract)
    Base class for implementing concrete feature calculators.
ComputerRegistry
    Decorator-based registry mapping feature types to computers.
FeatureCache
    Look-aside caching layer wrapping FeatureStore + CacheManager.
FeatureLineage
    Provenance tracking from source data → transforms → features → models.
FeatureComputationEngine
    Batch computation orchestration with dependency resolution, incremental
    updates, resume support, and progress tracking.
LazyFeature / LazyFeatureSet
    Deferred computation — features computed only on first access.
"""

from __future__ import annotations

from src.feature_store.cache import FeatureCache
from src.feature_store.computation import (
    FeatureComputationEngine,
    ComputationReport,
    LazyFeature,
    LazyFeatureSet,
)
from src.feature_store.cache import FeatureCache
from src.feature_store.computation import (
    FeatureComputationEngine,
    ComputationReport,
    LazyFeature,
    LazyFeatureSet,
)
from src.feature_store.computers import (
    FeatureComputer,
    ComputerRegistry,
)
from src.feature_store.lineage import (
    FeatureLineage,
    FeatureLineageEntry,
    LineageNode,
    LineageProvenance,
)
from src.feature_store.lineage import (
    FeatureLineage,
    FeatureLineageEntry,
    LineageNode,
    LineageProvenance,
)
from src.feature_store.models import (
    FeatureDefinition,
    FeatureDependency,
    FeatureValue,
    FeatureVersion,
    FeatureComputationBatch,
    FeatureStatus,
    FeatureCategory,
    EntityType,
)
from src.feature_store.registry import FeatureRegistry
from src.feature_store.store import FeatureStore
from src.feature_store.validation import (
    FeatureValidator,
    ValidationRule,
    RangeRule,
    NotNullRule,
    CardinalityRule,
    ConsistencyRule,
    ValidationResult,
)

__all__ = [
    # ORM Models
    "FeatureDefinition",
    "FeatureDependency",
    "FeatureValue",
    "FeatureVersion",
    "FeatureComputationBatch",
    "FeatureLineageEntry",
    "FeatureStatus",
    "FeatureCategory",
    "EntityType",
    # Core services
    "FeatureRegistry",
    "FeatureStore",
    "FeatureValidator",
    "FeatureCache",
    "FeatureLineage",
    "FeatureComputationEngine",
    # Validation
    "ValidationRule",
    "RangeRule",
    "NotNullRule",
    "CardinalityRule",
    "ConsistencyRule",
    "ValidationResult",
    # Computers
    "FeatureComputer",
    "ComputerRegistry",
    # Computation
    "ComputationReport",
    "LazyFeature",
    "LazyFeatureSet",
    # Lineage
    "LineageNode",
    "LineageProvenance",
]
