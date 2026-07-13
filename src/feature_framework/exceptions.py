"""
Custom Exceptions — typed exceptions for the feature engineering framework.
"""

from __future__ import annotations


class FeatureEngineError(Exception):
    """Base exception for all feature framework errors."""
    pass


class FeatureComputationError(FeatureEngineError):
    """Raised when feature computation fails for an entity."""
    def __init__(self, feature_name: str, entity_id: int, reason: str = "") -> None:
        self.feature_name = feature_name
        self.entity_id = entity_id
        self.reason = reason
        super().__init__(f"Failed computing {feature_name!r} for entity {entity_id}: {reason}")


class FeatureNotFoundError(FeatureEngineError):
    """Raised when a feature definition or value is not found."""
    def __init__(self, name: str, version: int | None = None) -> None:
        self.feature_name = name
        self.version = version
        version_str = f" v{version}" if version else ""
        super().__init__(f"Feature {name!r}{version_str} not found")


class FeatureValidationError(FeatureEngineError):
    """Raised when a computed feature value fails validation."""
    def __init__(self, feature_name: str, errors: list[str]) -> None:
        self.feature_name = feature_name
        self.errors = errors
        super().__init__(
            f"Validation failed for {feature_name!r}: {'; '.join(errors)}"
        )


class FeatureDependencyCycleError(FeatureEngineError):
    """Raised when a circular dependency is detected in the feature DAG."""
    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__(
            f"Circular dependency detected: {' → '.join(cycle)}"
        )


class FeatureConfigError(FeatureEngineError):
    """Raised when feature configuration is invalid."""
    def __init__(self, message: str, path: str = "") -> None:
        self.config_path = path
        super().__init__(f"Config error{' at ' + path if path else ''}: {message}")
