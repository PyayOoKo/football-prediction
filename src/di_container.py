"""
Dependency Injection Container — provides centralized configuration and service management.

This module implements a lightweight dependency injection system to replace
the global config singleton pattern, making the codebase more testable and maintainable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigProvider(Protocol):
    """Protocol for configuration providers.
    
    This protocol defines the interface for configuration access,
    allowing different implementations (global, injected, mock).
    """
    
    @property
    def paths(self) -> Any:
        """Access path configurations."""
        ...
    
    @property
    def train(self) -> Any:
        """Access training configurations."""
        ...
    
    @property
    def predict(self) -> Any:
        """Access prediction configurations."""
        ...
    
    @property
    def data_collection(self) -> Any:
        """Access data collection configurations."""
        ...
    
    @property
    def preprocessing(self) -> Any:
        """Access preprocessing configurations."""
        ...
    
    @property
    def features(self) -> Any:
        """Access feature engineering configurations."""
        ...
    
    @property
    def ensemble(self) -> Any:
        """Access ensemble model configurations."""
        ...
    
    @property
    def odds_api(self) -> Any:
        """Access odds API configurations."""
        ...
    
    @property
    def value_bet(self) -> Any:
        """Access value betting configurations."""
        ...
    
    @property
    def backtest(self) -> Any:
        """Access backtesting configurations."""
        ...
    
    @property
    def worldcup(self) -> Any:
        """Access World Cup specific configurations."""
        ...
    
    @property
    def db(self) -> Any:
        """Access database configurations."""
        ...
    
    @property
    def monitoring(self) -> Any:
        """Access monitoring configurations."""
        ...


class Container:
    """Dependency Injection Container.
    
    Central registry for application dependencies. Supports both
    singleton and transient lifetime scopes.
    
    Examples
    --------
    >>> container = Container()
    >>> container.register(ConfigProvider, lambda: config)
    >>> provider = container.resolve(ConfigProvider)
    """
    
    def __init__(self) -> None:
        self._services: dict[type, Any] = {}
        self._factories: dict[type, callable] = {}
        self._instances: dict[type, Any] = {}
    
    def register(self, interface: type[T], factory: callable, singleton: bool = True) -> None:
        """Register a service factory.
        
        Parameters
        ----------
        interface : type
            The interface/protocol being registered.
        factory : callable
            Factory function that creates the implementation.
        singleton : bool
            If True, cache the instance (default True).
        """
        self._factories[interface] = factory
        self._services[interface] = {
            'factory': factory,
            'singleton': singleton,
            'instance': None
        }
        logger.debug(f"Registered {interface.__name__} with singleton={singleton}")
    
    def register_instance(self, interface: type[T], instance: T) -> None:
        """Register an existing instance.
        
        Parameters
        ----------
        interface : type
            The interface/protocol being registered.
        instance : T
            The instance to register.
        """
        self._instances[interface] = instance
        self._services[interface] = {
            'factory': None,
            'singleton': True,
            'instance': instance
        }
        logger.debug(f"Registered instance for {interface.__name__}")
    
    def resolve(self, interface: type[T]) -> T:
        """Resolve a dependency.
        
        Parameters
        ----------
        interface : type
            The interface/protocol to resolve.
            
        Returns
        -------
        T
            The resolved instance.
            
        Raises
        ------
        ValueError
            If the interface is not registered.
        """
        if interface in self._instances:
            return self._instances[interface]
        
        if interface not in self._services:
            raise ValueError(f"No registration found for {interface.__name__}")
        
        service_info = self._services[interface]
        
        if service_info['singleton'] and service_info['instance'] is not None:
            return service_info['instance']
        
        factory = service_info['factory']
        if factory is None:
            raise ValueError(f"No factory registered for {interface.__name__}")
        
        instance = factory()
        
        if service_info['singleton']:
            service_info['instance'] = instance
        
        return instance
    
    def clear(self) -> None:
        """Clear all registered services and instances.
        
        Useful for testing to ensure isolation between tests.
        """
        self._services.clear()
        self._factories.clear()
        self._instances.clear()
        logger.debug("Container cleared")


# Global container instance (used as last resort, prefer explicit injection)
_container: Container | None = None


def get_container() -> Container:
    """Get the global container instance.
    
    Returns
    -------
    Container
        The global container instance.
    """
    global _container
    if _container is None:
        _container = Container()
    return _container


def set_container(container: Container) -> None:
    """Set the global container instance.
    
    Parameters
    ----------
    container : Container
        The container to set as global.
    """
    global _container
    _container = container


def reset_container() -> None:
    """Reset the global container.
    
    Useful for testing to ensure clean state.
    """
    global _container
    if _container is not None:
        _container.clear()
    _container = None


def configure_container(config: Any) -> Container:
    """Configure the container with default services.
    
    Parameters
    ----------
    config : Any
        Configuration instance to register.
        
    Returns
    -------
    Container
        The configured container.
    """
    container = Container()
    
    # Register config as ConfigProvider
    container.register_instance(ConfigProvider, config)
    
    # Register config itself for backward compatibility
    container.register_instance(type(config), config)
    
    logger.info("Container configured with default services")
    return container
