"""
TeamNormalizer — multi-stage team name resolution.

Resolution pipeline (in order):
1. Manual override → confidence 1.0
2. Exact alias lookup → confidence 0.95
3. Fuzzy match (Levenshtein) → confidence 0.65-0.85
4. Suffix stripping + re-lookup → confidence 0.60
5. Fallback (return input unchanged) → confidence 0.0

Each step logs its result for auditability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.team_normalizer.registry import AliasRegistry

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of resolving a team name.

    Attributes
    ----------
    original : str
        The input name as provided.
    canonical : str
        The resolved canonical name (or original if unresolved).
    confidence : float
        Confidence score 0.0-1.0.
    method : str
        How the name was resolved:
        - ``override`` — manual override
        - ``alias`` — exact match in alias dictionary
        - ``fuzzy`` — fuzzy/Levenshtein match
        - ``suffix`` — suffix stripping then alias match
        - ``fallback`` — no resolution found
    """

    original: str
    canonical: str
    confidence: float
    method: str = "fallback"

    @property
    def resolved(self) -> bool:
        """True if the name was successfully resolved (confidence > 0)."""
        return self.confidence > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "canonical": self.canonical,
            "confidence": self.confidence,
            "method": self.method,
        }


class TeamNormalizer:
    """Universal football team name normalizer.

    Resolves any team name variant to a canonical form with
    a confidence score.

    Parameters
    ----------
    registry : AliasRegistry, optional
        Custom alias registry. Defaults to built-in 2000+ entries.
    log_resolutions : bool
        Log each resolution at INFO level (default True).
    log_low_confidence : bool
        Log low-confidence resolutions (< 0.7) at WARNING level (default True).
    """

    def __init__(
        self,
        registry: AliasRegistry | None = None,
        log_resolutions: bool = True,
        log_low_confidence: bool = True,
    ) -> None:
        self.registry = registry or AliasRegistry()
        self.log_resolutions = log_resolutions
        self.log_low_confidence = log_low_confidence

        logger.info(
            "TeamNormalizer initialised with %d aliases covering %d teams",
            self.registry.count,
            self.registry.canonical_count,
        )

    # ── Single resolution ──────────────────────────────

    def resolve(self, name: str | None) -> NormalizationResult:
        """Resolve a single team name to its canonical form.

        Parameters
        ----------
        name : str or None
            Team name to resolve.

        Returns
        -------
        NormalizationResult
            Result with canonical name, confidence, and method.
        """
        if name is None or not name.strip():
            return NormalizationResult(
                original=str(name),
                canonical=str(name) if name else "",
                confidence=0.0,
                method="fallback",
            )

        original = name.strip()

        # Step 1: Manual override
        override = self.registry.check_override(original)
        if override is not None:
            return self._result(original, override, 1.0, "override")

        # Step 2: Exact alias lookup
        exact = self.registry.exact_lookup(original)
        if exact is not None:
            return self._result(original, exact, 0.95, "alias")

        # Step 3: Fuzzy match
        fuzzy = self.registry.fuzzy_match(original)
        if fuzzy is not None:
            return self._result(original, fuzzy, 0.75, "fuzzy")

        # Step 4: Suffix stripping + re-lookup
        suffix = self.registry.suffix_strip(original)
        if suffix is not None:
            return self._result(original, suffix, 0.60, "suffix")

        # Step 5: Fallback — return unchanged
        return self._result(original, original, 0.0, "fallback")

    # ── Batch resolution ───────────────────────────────

    def resolve_batch(
        self,
        names: list[str | None],
    ) -> list[NormalizationResult]:
        """Resolve multiple team names.

        Parameters
        ----------
        names : list of str or None
            Team names to resolve.

        Returns
        -------
        list of NormalizationResult
            One result per input name.
        """
        return [self.resolve(name) for name in names]

    def resolve_dataframe(
        self,
        data: list[dict[str, Any]],
        columns: list[str],
    ) -> list[dict[str, Any]]:
        """Resolve team names in a list-of-dicts dataset.

        Adds ``{col}_normalized`` and ``{col}_confidence`` columns
        for each specified column.

        Parameters
        ----------
        data : list[dict]
            Dataset as a list of row dicts.
        columns : list[str]
            Column names containing team names to resolve.

        Returns
        -------
        list[dict]
            Dataset with additional normalised columns.
        """
        results = []
        for row in data:
            new_row = dict(row)
            for col in columns:
                if col in row:
                    r = self.resolve(row[col])
                    new_row[f"{col}_normalized"] = r.canonical
                    new_row[f"{col}_confidence"] = r.confidence
                    new_row[f"{col}_method"] = r.method
            results.append(new_row)
        return results

    # ── Registry management ────────────────────────────

    def add_override(self, raw: str, canonical: str) -> None:
        """Add a manual override for a specific input.

        Parameters
        ----------
        raw : str
            The raw input to override.
        canonical : str
            The canonical name to resolve to.
        """
        self.registry.add_override(raw, canonical)

    def add_team(self, canonical: str, *aliases: str) -> None:
        """Register a new team with its aliases.

        Parameters
        ----------
        canonical : str
            Canonical team name.
        aliases : str
            Alias variants for this team.
        """
        self.registry.add_alias(canonical, *aliases)

    # ── Stats ──────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Return statistics about the normalizer."""
        return {
            "total_aliases": self.registry.count,
            "canonical_teams": self.registry.canonical_count,
            "overrides": 0,  # Could track internally
        }

    # ── Internal ───────────────────────────────────────

    def _result(
        self, original: str, canonical: str,
        confidence: float, method: str,
    ) -> NormalizationResult:
        """Create and log a NormalizationResult."""
        result = NormalizationResult(
            original=original,
            canonical=canonical,
            confidence=confidence,
            method=method,
        )

        if self.log_resolutions and method != "fallback":
            logger.info(
                "Resolved %r -> %s (conf=%.2f, method=%s)",
                original, canonical, confidence, method,
            )

        if self.log_low_confidence and confidence < 0.7:
            logger.warning(
                "Low confidence resolution: %r -> %s (conf=%.2f, method=%s)",
                original, canonical, confidence, method,
            )

        return result
