"""
Universal Football Team Normalization System.

Resolves any team name variant to its canonical form with a
confidence score, using a multi-stage pipeline:

    input → manual override → exact alias → fuzzy match
            → suffix stripping → fallback

Features
--------
- Alias dictionary with 2000+ entries covering 500+ clubs
- Fuzzy matching (Levenshtein distance) for typos
- Manual override for complex cases
- Confidence score for every resolution
- Fast O(1) dictionary lookup
- Comprehensive logging

Usage
-----
::

    from src.team_normalizer import TeamNormalizer

    normalizer = TeamNormalizer()

    # Exact alias
    result = normalizer.resolve("Man Utd")
    # → NormalizationResult(canonical="Manchester United", confidence=0.95, method="alias")

    # Fuzzy match
    result = normalizer.resolve("Manchestir United")
    # → NormalizationResult(canonical="Manchester United", confidence=0.75, method="fuzzy")

    # Unknown
    result = normalizer.resolve("AC Some Random Team")
    # → NormalizationResult(canonical="AC Some Random Team", confidence=0.0, method="fallback")
"""

from src.team_normalizer.core import NormalizationResult, TeamNormalizer
from src.team_normalizer.registry import AliasRegistry

__all__ = [
    "AliasRegistry",
    "NormalizationResult",
    "TeamNormalizer",
]
