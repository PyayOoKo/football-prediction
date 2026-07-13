"""
Tests for utility helper functions.
"""

from __future__ import annotations

from src.utils.helpers import chunks
from src.utils.validators import (
    validate_date_string,
    validate_probabilities,
    validate_team_name,
)


class TestHelpers:
    def test_chunks_even_split(self) -> None:
        result = chunks([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_chunks_uneven_split(self) -> None:
        result = chunks([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunks_empty(self) -> None:
        result = chunks([], 2)
        assert result == []


class TestValidators:
    def test_valid_team_name(self) -> None:
        assert validate_team_name("Arsenal") is True
        assert validate_team_name("FC Barcelona") is True

    def test_invalid_team_name(self) -> None:
        assert validate_team_name("") is False

    def test_valid_date(self) -> None:
        assert validate_date_string("2024-01-07") is True
        assert validate_date_string("invalid") is False

    def test_valid_probabilities(self) -> None:
        assert validate_probabilities([0.5, 0.3, 0.2]) is True

    def test_invalid_probabilities(self) -> None:
        assert validate_probabilities([0.5, 0.5, 0.5]) is False
