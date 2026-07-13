"""
Plugin System — auto-discovery of betting modules.

Supports three discovery mechanisms:
1. Entry points (``betting_modules`` group in pyproject.toml)
2. Convention (``src/betting/plugins/*.py``)
3. Explicit registration

Each plugin module must export a ``register(registry)`` function that
registers its module(s) into the ``BettingRegistry``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any

from src.betting.registry import BettingRegistry

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Discovers and registers betting modules from installed plugins.

    Parameters
    ----------
    auto_discover : bool
        Run discovery on init (default True).
    """

    def __init__(self, auto_discover: bool = True) -> None:
        self._plugins: dict[str, Any] = {}
        if auto_discover:
            self.discover()

    # ── Discovery ─────────────────────────────────────

    def discover(self) -> int:
        """Discover all available betting plugins.

        Returns
        -------
        int
            Number of plugins discovered.
        """
        count = 0
        count += self._discover_entry_points()
        count += self._discover_package()
        logger.info("Discovered %d betting plugins", count)
        return count

    def _discover_entry_points(self) -> int:
        """Discover plugins via setuptools entry points."""
        count = 0
        try:
            from pkg_resources import iter_entry_points
        except ImportError:
            return 0

        try:
            for ep in iter_entry_points(group="betting_modules"):
                try:
                    mod = ep.load()
                    if callable(getattr(mod, "register", None)):
                        self._plugins[ep.name] = mod
                        count += 1
                except Exception as exc:
                    logger.debug("Entry point '%s' failed: %s", ep.name, exc)
        except Exception:
            pass
        return count

    def _discover_package(self) -> int:
        """Discover plugins in the ``src.betting.plugins`` sub-package."""
        count = 0
        try:
            import src.betting.plugins as pkg
            if not hasattr(pkg, "__path__"):
                return 0
            for _importer, modname, _is_pkg in pkgutil.iter_modules(
                pkg.__path__, prefix="src.betting.plugins.",
            ):
                try:
                    mod = importlib.import_module(modname)
                    register_fn = getattr(mod, "register", None)
                    if callable(register_fn):
                        self._plugins[modname] = mod
                        count += 1
                except Exception as exc:
                    logger.debug("Plugin '%s' discovery failed: %s", modname, exc)
        except ImportError:
            pass
        return count

    # ── Registration into BettingRegistry ─────────────

    def discover_into(self, registry: BettingRegistry) -> int:
        """Discover plugins and register them into a BettingRegistry.

        Parameters
        ----------
        registry : BettingRegistry
            Target registry.

        Returns
        -------
        int
            Number of modules registered.
        """
        count = 0
        self.discover()

        for mod in self._plugins.values():
            register_fn = getattr(mod, "register", None)
            if callable(register_fn):
                try:
                    register_fn(registry)
                    count += 1
                except Exception as exc:
                    logger.warning("Plugin registration failed: %s", exc)

        return count

    # ── Querying ─────────────────────────────────────

    def list_plugins(self) -> list[str]:
        """Return names of all discovered plugins."""
        return list(self._plugins.keys())

    def __repr__(self) -> str:
        return f"<BettingPluginRegistry: {len(self._plugins)} plugins>"
