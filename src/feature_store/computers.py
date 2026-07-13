"""
FeatureComputer — abstract base class for feature computation.

This module defines the interface that concrete feature computers
must implement. No actual feature computation is performed here.

ComputerRegistry
    Maps feature types to their computer classes for lookup and dependency
    resolution. Implementations register themselves with the registry.

FeatureComputer (abstract)
    Base class with lifecycle hooks: ``init``, ``compute``, ``validate``.
    Subclasses implement ``compute_one`` and ``compute_batch``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class FeatureComputer(ABC):
    """Abstract base class for feature computation implementations.

    Subclasses must implement ``compute_one`` (single entity) and
    ``compute_batch`` (multiple entities). Optional hooks ``init``
    and ``validate`` can be overridden for custom setup/validation.

    Lifecycle
    ---------
    1. ``__init__`` — store parameters
    2. ``init()`` — load reference data, prepare state (optional)
    3. ``compute_one(entity)`` — compute for a single entity
    4. ``compute_batch(entities)`` — compute for multiple entities
    5. ``validate(result)`` — post-computation validation (optional)

    Parameters
    ----------
    name : str
        Computer identifier (e.g. ``elo_computer``).
    description : str
        Human-readable description.
    required_data : list[str]
        Data sources this computer needs (e.g. ``matches``, ``teams``).
    version : str
        Computer version string.
    **params : Any
        Additional parameters passed to subclasses.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        required_data: list[str] | None = None,
        version: str = "1.0.0",
        **params: Any,
    ) -> None:
        self.name = name
        self.description = description
        self.required_data = required_data or []
        self.version = version
        self.params = params

    # ── Lifecycle hooks ───────────────────────────────────

    def init(self) -> None:
        """Initialize the computer — load reference data, set up state.

        Called once before any ``compute_one`` or ``compute_batch`` calls.
        Override in subclasses if setup is required.
        """
        logger.info("Initialized computer: %s (v%s)", self.name, self.version)

    @abstractmethod
    def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
        """Compute features for a single entity.

        Parameters
        ----------
        entity_id : int
            Entity ID (match ID, team ID, etc.).
        **kwargs : Any
            Additional context needed for computation
            (e.g. match data, team data, date).

        Returns
        -------
        dict[str, Any]
            Feature values keyed by feature name.
        """
        ...

    def compute_batch(
        self,
        entity_ids: list[int],
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        """Compute features for multiple entities.

        The base implementation calls ``compute_one`` in a loop.
        Subclasses should override with optimized implementations.

        Parameters
        ----------
        entity_ids : list[int]
        **kwargs : Any

        Returns
        -------
        dict[int, dict[str, Any]]
            Mapping of ``{entity_id: {feature_name: value}}``.
        """
        return {
            eid: self.compute_one(eid, **kwargs) for eid in entity_ids
        }

    def validate(self, result: dict[str, Any]) -> bool:
        """Post-computation validation hook.

        Override to add custom validation logic after computation.

        Parameters
        ----------
        result : dict
            Computed feature values.

        Returns
        -------
        bool
            True if valid.
        """
        return True

    # ── Metadata ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize computer metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "required_data": self.required_data,
            "version": self.version,
            "params": self.params,
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.name!r} v{self.version}>"
        )


class ComputerRegistry:
    """Registry mapping feature types to their computer implementations.

    Computers register themselves by feature type string. The registry
    enables the feature store to look up the appropriate computer for
    a given feature definition.

    Usage
    -----
    ::

        registry = ComputerRegistry()

        @registry.register("elo")
        class EloComputer(FeatureComputer):
            def compute_one(self, ...): ...

        # Or manually:
        registry.add("rolling_stat", RollingStatComputer())
    """

    def __init__(self) -> None:
        self._computers: dict[str, type[FeatureComputer] | FeatureComputer] = {}
        self._instances: dict[str, FeatureComputer] = {}

    def register(
        self,
        feature_type: str,
    ) -> Any:
        """Decorator: register a computer class for a feature type.

        Usage
        -----
        ::

            @registry.register("elo_rating")
            class EloComputer(FeatureComputer):
                ...
        """
        def _wrapper(cls: type[FeatureComputer]) -> type[FeatureComputer]:
            self._computers[feature_type] = cls
            logger.debug(
                "Registered computer %s for feature type %r",
                cls.__name__, feature_type,
            )
            return cls
        return _wrapper

    def add(self, feature_type: str, computer: FeatureComputer) -> None:
        """Register an instantiated computer for a feature type.

        Parameters
        ----------
        feature_type : str
            Feature type string (matching ``FeatureDefinition.feature_type``).
        computer : FeatureComputer
            Instantiated computer instance.
        """
        self._instances[feature_type] = computer
        logger.debug(
            "Registered computer instance %s for feature type %r",
            computer.name, feature_type,
        )

    def get(self, feature_type: str) -> FeatureComputer | None:
        """Get a computer for the given feature type.

        Returns an instantiated computer. If a class was registered,
        instantiates it with default params first.

        Parameters
        ----------
        feature_type : str

        Returns
        -------
        FeatureComputer | None
        """
        # Check instances first
        if feature_type in self._instances:
            return self._instances[feature_type]

        # Check class registry
        cls = self._computers.get(feature_type)
        if cls is not None:
            instance = cls(name=feature_type)
            self._instances[feature_type] = instance
            instance.init()
            return instance

        return None

    def list_types(self) -> list[str]:
        """List all registered feature types."""
        return sorted(
            set(self._computers.keys()) | set(self._instances.keys()),
        )

    def has_type(self, feature_type: str) -> bool:
        """Check if a feature type has a registered computer."""
        return feature_type in self._computers or feature_type in self._instances

    def remove(self, feature_type: str) -> None:
        """Remove a computer registration."""
        self._computers.pop(feature_type, None)
        self._instances.pop(feature_type, None)
