"""
Tests for TeamNormalizer — resolution pipeline, confidence scores, batch operations.
"""

from __future__ import annotations

from src.team_normalizer import TeamNormalizer


class TestTeamNormalizerSingleResolution:
    """Verify the 5-step resolution pipeline."""

    def setup_method(self) -> None:
        self.n = TeamNormalizer(log_resolutions=False, log_low_confidence=False)

    # ── Step 1: Manual override ───────────────────────
    def test_manual_override(self) -> None:
        """Override should take priority over everything."""
        self.n.add_override("RSL", "Real Salt Lake")
        result = self.n.resolve("RSL")
        assert result.canonical == "Real Salt Lake"
        assert result.confidence == 1.0
        assert result.method == "override"

    # ── Step 2: Exact alias lookup ─────────────────────
    def test_exact_canonical(self) -> None:
        """Canonical name should resolve to itself."""
        result = self.n.resolve("Arsenal")
        assert result.canonical == "Arsenal"
        assert result.confidence == 0.95
        assert result.method == "alias"

    def test_common_alias(self) -> None:
        """Common alias should resolve correctly."""
        result = self.n.resolve("Man Utd")
        assert result.canonical == "Manchester United"
        assert result.confidence == 0.95
        assert result.method == "alias"

    def test_abbreviation(self) -> None:
        """Abbreviation should resolve."""
        result = self.n.resolve("MUFC")
        assert result.canonical == "Manchester United"

    def test_case_insensitive(self) -> None:
        """Lookup should be case-insensitive."""
        r1 = self.n.resolve("liverpool")
        r2 = self.n.resolve("LIVERPOOL")
        r3 = self.n.resolve("Liverpool")
        assert r1.canonical == r2.canonical == r3.canonical == "Liverpool"

    def test_nickname(self) -> None:
        """Nickname should resolve."""
        result = self.n.resolve("Gunners")
        assert result.canonical == "Arsenal"

    def test_club_with_fc(self) -> None:
        """Club with 'FC' suffix should resolve."""
        result = self.n.resolve("Arsenal FC")
        assert result.canonical == "Arsenal"

    # ── Step 3: Fuzzy matching ────────────────────────
    def test_fuzzy_match_typo(self) -> None:
        """Small typo should fuzzy-match."""
        result = self.n.resolve("Manchestir United")
        assert result.canonical == "Manchester United"
        assert result.method == "fuzzy"
        assert result.confidence == 0.75

    def test_fuzzy_match_missing_letter(self) -> None:
        """Missing letter should fuzzy-match."""
        result = self.n.resolve("Chelsa")
        assert result.canonical == "Chelsea"
        assert result.method == "fuzzy"

    def test_fuzzy_too_different(self) -> None:
        """Very different name should NOT fuzzy-match."""
        result = self.n.resolve("Totally Fake Club FC")
        assert result.method == "fallback"
        assert result.confidence == 0.0

    # ── Step 4: Suffix stripping ──────────────────────
    def test_suffix_strip_fc(self) -> None:
        """'FC' suffix should be stripped before lookup."""
        result = self.n.resolve("Some Random FC")
        # Should strip 'Some Random' → lookup 'Some Random' → fallback
        assert result.method in ("suffix", "fallback")

    # ── Step 5: Fallback ──────────────────────────────
    def test_fallback_unknown(self) -> None:
        """Unknown name should return itself with 0.0 confidence."""
        result = self.n.resolve("Unknown Team X")
        assert result.canonical == "Unknown Team X"
        assert result.confidence == 0.0
        assert result.method == "fallback"

    # ── Edge cases ────────────────────────────────────
    def test_null_input(self) -> None:
        """None input should return gracefully."""
        result = self.n.resolve(None)
        assert result.canonical == ""
        assert result.confidence == 0.0

    def test_empty_string(self) -> None:
        """Empty string should return gracefully."""
        result = self.n.resolve("")
        assert result.confidence == 0.0

    def test_whitespace_only(self) -> None:
        """Whitespace-only should return gracefully."""
        result = self.n.resolve("   ")
        assert result.confidence == 0.0

    def test_whitespace_stripping(self) -> None:
        """Leading/trailing whitespace should be stripped."""
        result = self.n.resolve("  Man Utd  ")
        assert result.canonical == "Manchester United"

    # ── Specific known mappings ───────────────────────
    def test_top_english_clubs(self) -> None:
        """Verify common English club resolutions."""
        cases = [
            ("Man City", "Manchester City"),
            ("Man United", "Manchester United"),
            ("Spurs", "Tottenham Hotspur"),
            ("Wolves", "Wolverhampton Wanderers"),
            ("Liverpool FC", "Liverpool"),
            ("Chelsea FC", "Chelsea"),
            ("Nott'm Forest", "Nottingham Forest"),
        ]
        for raw, expected in cases:
            result = self.n.resolve(raw)
            assert result.canonical == expected, f"{raw!r} -> {result.canonical} != {expected}"

    def test_international_teams(self) -> None:
        """Verify international team resolutions."""
        cases = [
            ("USA", "United States"),
            ("Brazil", "Brazil"),
            ("Holland", "Netherlands"),
            ("Germany", "Germany"),
            ("Cote d'Ivoire", "Ivory Coast"),
        ]
        for raw, expected in cases:
            result = self.n.resolve(raw)
            assert result.canonical == expected, f"{raw!r} -> {result.canonical} != {expected}"


class TestTeamNormalizerBatch:
    """Verify batch operations."""

    def setup_method(self) -> None:
        self.n = TeamNormalizer(log_resolutions=False, log_low_confidence=False)

    def test_resolve_batch(self) -> None:
        names = ["Arsenal", "Man Utd", "Chelsea", "Unknown"]
        results = self.n.resolve_batch(names)
        assert len(results) == 4
        assert results[0].canonical == "Arsenal"
        assert results[1].canonical == "Manchester United"
        assert results[2].canonical == "Chelsea"
        assert results[3].canonical == "Unknown"
        assert results[3].confidence == 0.0

    def test_resolve_dataframe(self) -> None:
        data = [
            {"home": "Arsenal", "away": "Man Utd"},
            {"home": "Liverpool", "away": "Chelsea"},
        ]
        result = self.n.resolve_dataframe(data, columns=["home", "away"])
        assert len(result) == 2
        assert result[0]["home_normalized"] == "Arsenal"
        assert result[0]["away_normalized"] == "Manchester United"
        assert "home_confidence" in result[0]
        assert "away_method" in result[0]


class TestTeamNormalizerRegistryManagement:
    """Verify adding teams and overrides dynamically."""

    def setup_method(self) -> None:
        self.n = TeamNormalizer(log_resolutions=False, log_low_confidence=False)

    def test_add_new_team(self) -> None:
        self.n.add_team("Inter Miami CF", "Inter Miami", "Miami")
        result = self.n.resolve("Inter Miami")
        assert result.canonical == "Inter Miami CF"
        assert result.confidence == 0.95

    def test_add_alias_to_existing(self) -> None:
        self.n.add_team("Liverpool", "LFC Official")
        result = self.n.resolve("LFC Official")
        assert result.canonical == "Liverpool"

    def test_add_override(self) -> None:
        self.n.add_override("LFC", "Liverpool FC")
        result = self.n.resolve("LFC")
        assert result.canonical == "Liverpool FC"
        assert result.confidence == 1.0

    def test_override_beats_alias(self) -> None:
        """Override should always beat alias lookup."""
        self.n.add_override("MUFC", "Melbourne United FC")
        result = self.n.resolve("MUFC")
        assert result.canonical == "Melbourne United FC"
        assert result.confidence == 1.0
