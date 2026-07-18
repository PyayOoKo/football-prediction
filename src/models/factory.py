"""
ModelFactory — creates model instances using the plugin registry and
configuration, with automatic fallback chains and caching.

Supports:
- Config-driven creation from a type string
- Fallback to nearest available plugin
- Singleton-like caching (same config → same instance)
- Registration of transient model parameters
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.base import BaseModel
from src.models.plugins import Plugin, PluginRegistry
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances.

    The factory discovers available model types through the
    ``PluginRegistry`` and instantiates them via ``create()``.

    Parameters
    ----------
    plugin_registry : PluginRegistry, optional
        Shared plugin registry. Creates a new one if omitted.
    model_registry : ModelRegistry, optional
        Shared model registry. Creates a new one if omitted.
    cache_models : bool
        If True, reuse instances for the same key (default False).
    """

    def __init__(
        self,
        plugin_registry: PluginRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        cache_models: bool = False,
    ) -> None:
        self.plugins = plugin_registry or PluginRegistry()
        self.models = model_registry or ModelRegistry()
        self.cache_models = cache_models
        self._cache: dict[str, BaseModel] = {}

    # ── Core factory method ───────────────────────────

    def create(
        self,
        model_type: str,
        **kwargs: Any,
    ) -> BaseModel:
        """Create a model instance by type.

        Steps:
        1. Look up the plugin for the requested model type.
        2. If not available, try fallback chain.
        3. Instantiate the plugin with provided kwargs.
        4. Optionally cache and register the instance.

        Parameters
        ----------
        model_type : str
            Model type identifier (e.g. ``xgboost``, ``poisson``, ``ensemble``).
        **kwargs
            Passed to the model constructor. Common options:
            - ``model_name``: custom name
            - ``model_version``: version string
            - ``random_seed``: random seed
            - ``metadata``: dict of metadata

        Returns
        -------
        BaseModel

        Raises
        ------
        ValueError
            If no plugin is available for the requested type.
        """
        cache_key = self._cache_key(model_type, kwargs)

        # Check cache
        if self.cache_models and cache_key in self._cache:
            logger.debug("Factory cache hit: %s", cache_key)
            return self._cache[cache_key]

        # Find and resolve plugin
        plugin = self.plugins.resolve(model_type)
        if plugin is None:
            available = list(self.plugins.list_available().keys())
            raise ValueError(
                f"No available plugin for model type '{model_type}'. "
                f"Available types: {available}. "
                f"Install the required package or use a different type."
            )

        if plugin.model_type != model_type:
            logger.info(
                "Resolved '%s' → using available plugin '%s'",
                model_type, plugin.model_type,
            )

        # Extract known kwargs
        known_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"model_name", "model_version", "metadata", "random_seed"}
        }

        # Instantiate
        try:
            model = plugin.instantiate(**known_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate '{plugin.model_type}': {exc}"
            ) from exc

        # Cache
        if self.cache_models:
            self._cache[cache_key] = model

        # Auto-register
        if kwargs.get("auto_register", True):
            try:
                self.models.register(
                    model,
                    experiment_name=kwargs.get("experiment_name"),
                )
            except Exception as exc:
                logger.warning("Auto-registration failed (non-fatal): %s", exc)

        logger.info("Created model: %s v%s", model.model_name, model.model_version)
        return model

    # ── Convenience methods ──────────────────────────

    def create_ensemble(
        self,
        model_names: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Create an ensemble model with specified sub-models.

        Parameters
        ----------
        model_names : list[str], optional
            Model types to include in the ensemble (e.g.
            ``[\\\"xgboost\\\", \\\"logistic_regression\\\", \\\"poisson\\\"]``).
            Defaults to the project config.
        **kwargs
            Additional kwargs for the ensemble and sub-models.

        Returns
        -------
        BaseModel
        """
        from config import config as _global_config
        names = model_names or list(_global_config.ensemble.model_names)
        return self.create(
            "ensemble",
            model_names=names,
            **kwargs,
        )

    def create_default(self, **kwargs: Any) -> BaseModel:
        """Create the default prediction model.

        Uses the project config's ``model_type`` setting.
        Falls back to ``xgboost`` if the configured type is unavailable.

        Returns
        -------
        BaseModel
        """
        from config import config as _global_config
        model_type = _global_config.train.model_type
        try:
            return self.create(model_type, **kwargs)
        except ValueError:
            logger.warning(
                "Configured model '%s' not available, falling back to xgboost",
                model_type,
            )
            return self.create("xgboost", **kwargs)

    def create_all(self, **kwargs: Any) -> dict[str, BaseModel]:
        """Create instances of all available model types.

        Parameters
        ----------
        **kwargs
            Passed to each model constructor.

        Returns
        -------
        dict[str, BaseModel]
            ``{model_type: instance}``
        """
        models: dict[str, BaseModel] = {}
        for model_type in self.plugins.list_available():
            try:
                models[model_type] = self.create(model_type, **kwargs)
            except Exception as exc:
                logger.warning("Skipping '%s': %s", model_type, exc)
        return models

    # ── Plugin management ────────────────────────────

    def register_plugin(
        self,
        model_type: str,
        model_class: type[BaseModel],
        **plugin_kwargs: Any,
    ) -> Plugin:
        """Register a new model plugin at runtime.

        Parameters
        ----------
        model_type : str
            Unique type identifier.
        model_class : type[BaseModel]
            The model class.
        **plugin_kwargs
            Plugin metadata: priority, description, dependencies.

        Returns
        -------
        Plugin
        """
        return self.plugins.register(
            model_type,
            model_class=model_class,
            **plugin_kwargs,
        )

    def list_available_types(self) -> list[str]:
        """List all available model type identifiers."""
        return list(self.plugins.list_available().keys())

    # ── Internal ─────────────────────────────────────

    @staticmethod
    def _cache_key(model_type: str, kwargs: dict) -> str:
        """Generate a cache key from model type and kwargs."""
        # Only use hashable kwargs for the cache key
        key_parts = [model_type]
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def clear_cache(self) -> None:
        """Clear the instance cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        return f"<ModelFactory: {len(self.list_available_types())} types available>"
