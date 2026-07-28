"""
Logging utilities — structured logging context and helpers.

Usage
-----
::

    from src.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("Training started", extra={"model": "xgboost", "n_features": 50})

    with logger.context(model="xgboost", fold=3):
        logger.info("Fold complete")                    # extra injected automatically
        raise ValueError("bad")
        # logger.exception("...") also includes context
"""

from __future__ import annotations

import logging
from typing import Any


class _ContextAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """A logger adapter that merges a static ``extra`` dict with ``extra``
    passed at the call site, then forwards to the underlying logger."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None) -> None:
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        kwargs.setdefault("extra", {})
        kwargs["extra"].update(self.extra)
        return msg, kwargs


def get_logger(name: str, **context: Any) -> _ContextAdapter:
    """Return a logger adapter with static context injected on every call.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    **context
        Key-value pairs added to ``extra`` on every log call.

    Returns
    -------
    _ContextAdapter
    """
    return _ContextAdapter(logging.getLogger(name), context)
