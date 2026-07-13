"""
Decorators — @timeit, @log_call, @retry for feature computation methods.

Composable decorators that wrap feature pipeline stages with
cross-cutting concerns without modifying the transformer logic.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════
#  @timeit — measure and log execution time
# ═══════════════════════════════════════════════════════════


def timeit(func: F) -> F:
    """Decorator that measures execution time and logs it.

    Adds ``duration_seconds`` to the return dict if the result is a dict.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if isinstance(result, dict) and "duration_seconds" not in result:
            result["duration_seconds"] = round(elapsed, 4)

        logger.debug("%s took %.3fs", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore


# ═══════════════════════════════════════════════════════════
#  @log_call — log method entry/exit with parameters
# ═══════════════════════════════════════════════════════════


def log_call(level: int = logging.DEBUG) -> Callable[[F], F]:
    """Decorator that logs method entry and exit.

    Parameters
    ----------
    level : int
        Logging level (default ``logging.DEBUG``).
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if logger.isEnabledFor(level):
                cls_name = args[0].__class__.__name__ if args else ""
                arg_preview = ", ".join(str(a)[:60] for a in args[1:])
                kwarg_preview = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                logger.log(
                    level, "%s.%s(%s%s)",
                    cls_name, func.__name__,
                    arg_preview,
                    f", {kwarg_preview}" if kwarg_preview else "",
                )
            result = func(*args, **kwargs)
            logger.log(level, "%s.%s → OK", cls_name, func.__name__)
            return result

        return wrapper  # type: ignore

    return decorator


# ═══════════════════════════════════════════════════════════
#  @retry — retry on transient failures with exponential backoff
# ═══════════════════════════════════════════════════════════


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function on failure.

    Parameters
    ----------
    max_attempts : int
        Maximum retry attempts (default 3).
    delay : float
        Initial delay in seconds (default 1.0).
    backoff : float
        Exponential backoff multiplier (default 2.0).
    exceptions : tuple
        Exception types to catch (default all).
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        logger.warning(
                            "%s attempt %d/%d failed: %s. Retrying in %.1fs...",
                            func.__qualname__,
                            attempt, max_attempts, exc, wait,
                        )
                        time.sleep(wait)
            raise last_exc  # type: ignore

        return wrapper  # type: ignore

    return decorator
