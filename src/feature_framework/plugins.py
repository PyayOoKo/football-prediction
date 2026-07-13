"""
Plugin Discovery — auto-discovery of feature transformers.

Supports three discovery mechanisms:
1. Entry points (``feature_transformers`` group in pyproject.toml)
2. Convention (``src/feature_framework/transformers/*.py``)
3. Explicit registration

Each plugin module exports a ``register(registry)`` function that
registers feature transformers.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

from src.feature_framework.base import FeatureTransformer

if TYPE_CHECKING:
    from src.feature_framework.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class FeaturePluginRegistry:
    """Registry for discovering and managing feature transformer plugins.

    Parameters
    ----------
    auto_discover : bool
        Run discovery on init (default True).
    """

    def __init__(self, auto_discover: bool = True) -> None:
        self._transformers: dict[str, type[FeatureTransformer]] = {}
        self._instances: dict[str, FeatureTransformer] = {}
        self._plugin_modules: dict[str, Any] = {}

        if auto_discover:
            self.discover()

    # ── Registration ──────────────────────────────────

    def register(
        self,
        transformer_cls: type[FeatureTransformer],
        name: str | None = None,
    ) -> type[FeatureTransformer]:
        """Register a feature transformer class.

        Can be used as a decorator:

        .. code-block:: python

            registry = FeaturePluginRegistry()

            @registry.register
            class EloTransformer(FeatureTransformer):
                name = "elo_rating"
                ...

        Parameters
        ----------
        transformer_cls : type[FeatureTransformer]
            The transformer class to register.
        name : str, optional
            Override name. Defaults to ``transformer_cls.name``.

        Returns
        -------
        type[FeatureTransformer]
            The registered class (for decorator usage).
        """
        if not inspect.isclass(transformer_cls) or not issubclass(transformer_cls, FeatureTransformer):
            raise TypeError(f"{transformer_cls} must be a FeatureTransformer subclass")

        key = name or transformer_cls.name
        if not key:
            raise ValueError(f"Transformer {transformer_cls} has no name. Set class attribute 'name'.")

        if key in self._transformers:
            logger.warning("Overwriting registered transformer: %s", key)

        self._transformers[key] = transformer_cls
        logger.debug("Registered feature transformer: %s (%s)", key, transformer_cls.__name__)
        return transformer_cls

    def unregister(self, name: str) -> bool:
        """Remove a transformer registration."""
        if name in self._transformers:
            del self._transformers[name]
            self._instances.pop(name, None)
            logger.debug("Unregistered feature transformer: %s", name)
            return True
        return False

    # ── Discovery ─────────────────────────────────────

    def discover(self) -> int:
        """Auto-discover transformers from all available sources.

        Returns
        -------
        int
            Number of transformers discovered.
        """
        count = 0
        count += self._discover_entry_points()
        count += self._discover_package()
        logger.info("Discovered %d feature transformers", count)
        return count

    def _discover_entry_points(self) -> int:
        """Discover via setuptools entry points."""
        count = 0
        try:
            from pkg_resources import iter_entry_points
        except ImportError:
            return 0

        try:
            for ep in iter_entry_points(group="feature_transformers"):
                try:
                    cls = ep.load()
                    if inspect.isclass(cls) and issubclass(cls, FeatureTransformer):
                        self.register(cls)
                        count += 1
                except Exception as exc:
                    logger.debug("Entry point '%s' failed: %s", ep.name, exc)
        except Exception:
            pass
        return count

    def _discover_package(self) -> int:
        """Discover in ``src.feature_framework.transformers`` package."""
        count = 0
        try:
            import src.feature_framework.transformers as pkg
            if not hasattr(pkg, "__path__"):
                return 0
            import pkgutil
            import importlib
            for _importer, modname, _is_pkg in pkgutil.iter_modules(
                pkg.__path__, prefix="src.feature_framework.transformers.",
            ):
                try:
                    mod = importlib.import_module(modname)
                    # Auto-register all FeatureTransformer subclasses in the module
                    for name, obj in inspect.getmembers(mod):
                        if (inspect.isclass(obj) and issubclass(obj, FeatureTransformer)
                                and obj is not FeatureTransformer):
                            self.register(obj)
                            self._plugin_modules[modname] = mod
                            count += 1
                except Exception as exc:
                    logger.debug("Plugin '%s' discovery failed: %s", modname, exc)
        except ImportError:
            pass
        return count

    # ── Instance management ──────────────────────────

    def get(self, name: str) -> FeatureTransformer | None:
        """Get a transformer instance by name.

        Creates and caches the instance on first access.

        Parameters
        ----------
        name : str
            Transformer name.

        Returns
        -------
        FeatureTransformer | None
        """
        # Check cache
        instance = self._instances.get(name)
        if instance is not None:
            return instance

        # Create from class
        cls = self._transformers.get(name)
        if cls is None:
            return None

        instance = cls()
        self._instances[name] = instance
        return instance

    def get_or_create(self, name: str, **params: Any) -> FeatureTransformer | None:
        """Get a transformer, creating with params if not cached.

        Parameters
        ----------
        name : str
            Transformer name.
        **params
            Parameters passed to the transformer constructor.

        Returns
        -------
        FeatureTransformer | None
        """
        if name in self._instances:
            return self._instances[name]

        cls = self._transformers.get(name)
        if cls is None:
            return None

        instance = cls(**params)
        self._instances[name] = instance
        return instance

    # ── Querying ─────────────────────────────────────

    def list_types(self) -> list[str]:
        """List all registered transformer type names."""
        return sorted(self._transformers.keys())

    def list_instances(self) -> dict[str, FeatureTransformer]:
        """List all cached transformer instances."""
        return dict(self._instances)

    def __len__(self) -> int:
        return len(self._transformers)

    def __repr__(self) -> str:
        return f"<FeaturePluginRegistry: {len(self._transformers)} transformers, {len(self._instances)} instances>"
