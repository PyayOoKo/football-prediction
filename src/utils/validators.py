"""
Input validation and data quality helpers.

Functions for validating user inputs, API parameters,
and ensuring data meets minimum quality standards.
"""

from __future__ import annotations

import re
from typing import Any

# ── Team name patterns ─────────────────────────────────
_TEAM_NAME_PATTERN = re.compile(r"^[A-Za-z0-9\s\-\.\'À-ÿ]+$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_team_name(name: str) -> bool:
    """Check if a team name contains only valid characters."""
    return bool(_TEAM_NAME_PATTERN.match(name.strip()))


def validate_date_string(date_str: str) -> bool:
    """Check if a string matches YYYY-MM-DD format."""
    return bool(_DATE_PATTERN.match(date_str.strip()))


def validate_probability(value: float) -> bool:
    """Check if a value is a valid probability between 0 and 1."""
    return 0.0 <= value <= 1.0


def validate_probabilities(probs: list[float]) -> bool:
    """Check if a list sums to approximately 1.0 and each is [0, 1]."""
    if not all(validate_probability(p) for p in probs):
        return False
    return abs(sum(probs) - 1.0) < 0.02


def validate_positive_int(value: Any, name: str = "value") -> int:
    """Cast to ``int`` and ensure it is positive.

    Raises
    ------
    ValueError
        If the value cannot be cast or is not positive.
    """
    try:
        val = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val}")
    return val
