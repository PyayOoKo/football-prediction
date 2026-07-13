"""
Tests for AliasRegistry — alias dictionary, fuzzy matching, overrides.
"""

from __future__ import annotations

from src.team_normalizer import AliasRegistry


class TestAliasRegistry:
    def setup_method(self) -> None:
        self.registry = AliasRegistry()

    def test_has_thousands_of_aliases(self) -> None:
        """Should have at least 1500 aliases covering 300+ teams."""
        assert self.registry.count >= 1500, f"Only {self.registry.count} aliases"
        assert self.registry.canonical_count >= 300, f"Only {self.registry.canonical_count} teams"

    def test_exact_lookup_found(self) -> None:
        result = self.registry.exact_lookup("Man Utd")
        assert result == "Manchester United"

    def test_exact_lookup_not_found(self) -> None:
        result = self.registry.exact_lookup("XYZ Fake Club")
        assert result is None

    def test_exact_lookup_case_insensitive(self) -> None:
        result = self.registry.exact_lookup("MAN UTD")
        assert result == "Manchester United"

    def test_fuzzy_match(self) -> None:
        result = self.registry.fuzzy_match("Manchestir")
        assert result == "Manchester United"

    def test_fuzzy_match_too_short(self) -> None:
        """Very short strings should not fuzzy match."""
        result = self.registry.fuzzy_match("ab")
        assert result is None

    def test_fuzzy_match_cached(self) -> None:
        """Second call with same input should return quickly."""
        self.registry.fuzzy_match("Manchestir")
        result = self.registry.fuzzy_match("Manchestir")
        assert result == "Manchester United"

    def test_suffix_strip_fc(self) -> None:
        result = self.registry.suffix_strip("Arsenal FC")
        assert result == "Arsenal"

    def test_suffix_strip_no_match(self) -> None:
        result = self.registry.suffix_strip("Arsenal")
        assert result is None  # Already canonical

    def test_add_override(self) -> None:
        self.registry.add_override("MUFC", "Melbourne United")
        result = self.registry.check_override("MUFC")
        assert result == "Melbourne United"

    def test_add_override_case_insensitive(self) -> None:
        self.registry.add_override("MUFC", "Melbourne United")
        result = self.registry.check_override("mufc")
        assert result == "Melbourne United"

    def test_add_alias(self) -> None:
        self.registry.add_alias("Inter Miami CF", "Inter Miami", "Miami")
        result = self.registry.exact_lookup("Inter Miami")
        assert result == "Inter Miami CF"
        assert "Inter Miami CF" in self.registry.get_canonical_names()

    def test_get_all_aliases(self) -> None:
        aliases = self.registry.get_all_aliases("Arsenal")
        assert "gunners" in aliases
        assert "arsenal fc" in aliases

    def test_get_canonical_names_sorted(self) -> None:
        names = self.registry.get_canonical_names()
        assert names == sorted(names)

    def test_levenshtein_distance(self) -> None:
        """Internal Levenshtein function."""
        dist = self.registry._levenshtein("kitten", "sitting")
        assert dist == 3

        dist = self.registry._levenshtein("same", "same")
        assert dist == 0

        dist = self.registry._levenshtein("abc", "abcdef")
        assert dist == 3
