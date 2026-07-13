"""
General-purpose helper functions.

Provides reusable utilities for common tasks:
timing, file I/O, date parsing, and data conversion.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F | None = None, *, label: str = "") -> Callable[..., Any]:
    """Decorator / context manager to time function execution.

    Usage as a decorator::

        @timer
        def train_model() -> None:
            ...

        @timer(label="Training")
        def train_model() -> None:
            ...

    Usage as a context manager::

        with timer(label="Batch predict"):
            predict_all()
    """
    if func is not None:
        # Used as a plain decorator: @timer
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = label or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"[{name}] completed in {elapsed:.3f}s")
            return result
        return wrapper
    else:
        # Used as a decorator with arguments: @timer(label="...")
        def decorator(f: F) -> Callable[..., Any]:
            @functools.wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                name = label or f.__name__
                start = time.perf_counter()
                result = f(*args, **kwargs)
                elapsed = time.perf_counter() - start
                print(f"[{name}] completed in {elapsed:.3f}s")
                return result
            return wrapper
        return decorator


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into chunks of at most ``size`` elements."""
    return [items[i:i + size] for i in range(0, len(items), size)]
