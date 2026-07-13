"""
Plugin Architecture — automatic model discovery and registration.

Plugins are self-contained modules that export a ``register`` function.
The plugin system discovers them via:

1. Entry points (``package.model_plugins``)
2. Convention (``src/models/plugins/*.py``)
3. Explicit registration

Each plugin must define:
- ``MODEL_TYPE`` — string identifier (e.g. ``"xgboost"``)
- ``MODEL_NAME`` — human-readable name (e.g. ``"XGBoost Classifier"``)
- ``register(registry)`` — function that registers the model class
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class Plugin:
    """Descriptor for a model plugin.

    Attributes
    ----------
    model_type : str
        Unique type identifier (e.g. ``xgboost``).
    model_name : str
        Human-readable name.
    model_class : type[BaseModel]
        The model class (not instantiated).
    priority : int
        Priority for resolution order (higher = preferred).
    description : str
        Brief description of the model.
    dependencies : list[str]
        Required Python packages.
    """

    def __init__(
        self,
        model_type: str,
        model_name: str,
        model_class: type[BaseModel],
        priority: int = 0,
        description: str = "",
        dependencies: list[str] | None = None,
    ) -> None:
        self.model_type = model_type
        self.model_name = model_name
        self.model_class = model_class
        self.priority = priority
        self.description = description
        self.dependencies = dependencies or []
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Check if all dependencies are installed."""
        if self._available is not None:
            return self._available
        self._available = True
        for dep in self.dependencies:
            try:
                importlib.import_module(dep.replace("-", "_").split("[")[0])
            except ImportError:
                self._available = False
                break
        return self._available

    def instantiate(self, **kwargs: Any) -> BaseModel:
        """Create a new instance of this model.

        Parameters
        ----------
        **kwargs
            Passed to the model constructor.

        Returns
        -------
        BaseModel

        Raises
        ------
        ImportError
            If dependencies are not available.
        """
        if not self.available:
            missing = [
                d for d in self.dependencies
                if not _is_installed(d.replace("-", "_").split("[")[0])
            ]
            raise ImportError(
                f"Plugin '{self.model_type}' requires: {', '.join(missing)}. "
                "Install them with: pip install " + " ".join(missing)
            )
        return self.model_class(**kwargs)

    def __repr__(self) -> str:
        status = "✅" if self.available else "❌"
        return (
            f"<Plugin {self.model_type}: {self.model_name} "
            f"{status} prio={self.priority}>"
        )


def _is_installed(module_name: str) -> bool:
    """Check if a Python module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


class PluginRegistry:
    """Registry of all available model plugins.

    Supports auto-discovery via entry points, package scanning, and
    explicit registration.

    Parameters
    ----------
    auto_discover : bool
        Automatically discover plugins on init (default True).
    """

    def __init__(self, auto_discover: bool = True) -> None:
        self._plugins: dict[str, Plugin] = {}

        if auto_discover:
            self.discover()

    # ── Registration ──────────────────────────────────

    def register(
        self,
        model_type: str,
        model_name: str,
        model_class: type[BaseModel],
        priority: int = 0,
        description: str = "",
        dependencies: list[str] | None = None,
    ) -> Plugin:
        """Register a model plugin.

        Parameters
        ----------
        model_type : str
            Unique type identifier.
        model_name : str
            Human-readable name.
        model_class : type[BaseModel]
            The model class (must subclass ``BaseModel``).
        priority : int
            Resolution priority (higher = preferred).
        description : str
            Brief description.
        dependencies : list[str], optional
            Required pip package names.

        Returns
        -------
        Plugin

        Raises
        ------
        ValueError
            If model_type is already registered.
        TypeError
            If model_class does not subclass BaseModel.
        """
        if model_type in self._plugins:
            raise ValueError(
                f"Plugin '{model_type}' is already registered. "
                "Use unregister() first or choose a different type."
            )

        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"'{model_class.__name__}' must subclass BaseModel."
            )

        plugin = Plugin(
            model_type=model_type,
            model_name=model_name,
            model_class=model_class,
            priority=priority,
            description=description or model_class.__doc__ or "",
            dependencies=dependencies,
        )
        self._plugins[model_type] = plugin
        logger.info("Registered plugin: %s (%s)", model_type, model_name)
        return plugin

    def unregister(self, model_type: str) -> bool:
        """Remove a plugin from the registry."""
        if model_type in self._plugins:
            del self._plugins[model_type]
            logger.info("Unregistered plugin: %s", model_type)
            return True
        return False

    # ── Discovery ────────────────────────────────────

    def discover(self) -> int:
        """Auto-discover plugins from all available sources.

        Returns
        -------
        int
            Number of plugins discovered.
        """
        count = 0
        count += self._discover_entry_points()
        count += self._discover_package()
        logger.info("Discovered %d model plugins", count)
        return count

    def _discover_entry_points(self) -> int:
        """Discover plugins via setuptools entry points."""
        count = 0
        try:
            from pkg_resources import iter_entry_points
        except ImportError:
            return 0

        try:
            for ep in iter_entry_points(group="model_plugins"):
                try:
                    cls = ep.load()
                    if inspect.isclass(cls) and issubclass(cls, BaseModel):
                        self._register_from_class(cls)
                        count += 1
                except Exception as exc:
                    logger.debug("Entry point '%s' failed: %s", ep.name, exc)
        except Exception:
            pass
        return count

    def _discover_package(self) -> int:
        """Discover plugins in the ``src.models.plugins`` sub-package.

        Iterates ``.py`` files inside ``src/models/plugins/`` (a separate
        directory package), calling ``register(registry)`` on each module
        that exposes such a function.
        """
        count = 0
        try:
            import src.models.plugins as pkg
            # pkg.__path__ only exists when ``plugins`` is a *directory*
            # package (i.e. ``src/models/plugins/``, not a single-file
            # module ``src/models/plugins.py``).
            if not hasattr(pkg, "__path__"):
                return 0
            for _importer, modname, _is_pkg in pkgutil.iter_modules(
                pkg.__path__, prefix="src.models.plugins.",
            ):
                try:
                    mod = importlib.import_module(modname)
                    register_fn = getattr(mod, "register", None)
                    if callable(register_fn):
                        register_fn(self)
                        count += 1
                except Exception as exc:
                    logger.debug("Plugin '%s' discovery failed: %s", modname, exc)
        except ImportError:
            pass
        return count

    def _register_from_class(self, cls: type[BaseModel]) -> None:
        """Auto-register a model class by reading its class attributes."""
        model_type = getattr(cls, "model_type", None)
        if model_type and model_type not in self._plugins:
            self.register(
                model_type=model_type,
                model_name=getattr(cls, "model_name", model_type),
                model_class=cls,
                priority=getattr(cls, "plugin_priority", 0),
                description=cls.__doc__ or "",
                dependencies=getattr(cls, "plugin_dependencies", None),
            )

    # ── Querying ─────────────────────────────────────

    def list_plugins(self) -> dict[str, Plugin]:
        """Return all registered plugins keyed by model type."""
        return dict(self._plugins)

    def list_available(self) -> dict[str, Plugin]:
        """Return only plugins whose dependencies are satisfied."""
        return {
            t: p for t, p in self._plugins.items()
            if p.available
        }

    def get(self, model_type: str) -> Plugin | None:
        """Get a plugin by model type."""
        return self._plugins.get(model_type)

    def get_by_class(self, cls: type) -> Plugin | None:
        """Find the plugin that corresponds to a model class."""
        for plugin in self._plugins.values():
            if plugin.model_class is cls:
                return plugin
        return None

    def resolve(self, model_type: str) -> Plugin | None:
        """Resolve a model type to its best available plugin.

        Falls back through available plugins with the same type prefix.

        Parameters
        ----------
        model_type : str
            Desired model type.

        Returns
        -------
        Plugin | None
            The best matching available plugin, or None.
        """
        # Exact match
        plugin = self.get(model_type)
        if plugin and plugin.available:
            return plugin

        # Partial match — find all with this prefix
        candidates = [
            p for t, p in self._plugins.items()
            if model_type in t and p.available
        ]
        if not candidates:
            return None

        # Sort by priority (highest first), then by name
        candidates.sort(key=lambda p: (-p.priority, p.model_type))
        return candidates[0]

    def __repr__(self) -> str:
        total = len(self._plugins)
        available = len(self.list_available())
        return f"<PluginRegistry: {available}/{total} plugins available>"

    def to_dict(self) -> dict[str, Any]:
        """Export registry state."""
        return {
            "n_plugins": len(self._plugins),
            "n_available": len(self.list_available()),
            "plugins": {
                t: {
                    "model_type": p.model_type,
                    "model_name": p.model_name,
                    "model_class": p.model_class.__name__,
                    "priority": p.priority,
                    "available": p.available,
                }
                for t, p in self._plugins.items()
            },
        }
