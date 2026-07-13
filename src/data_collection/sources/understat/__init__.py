"""
Understat — expected goals (xG) data importer for understat.com.

Collects xG, xGA, shot locations, match-level xG, and team xG
statistics from Understat's embedded JSON data.

Features
--------
- Extracts JSON from JavaScript variables embedded in HTML script tags
- Decodes Unicode-escaped strings
- Caches HTTP responses for speed
- Incremental sync with match_id checkpointing
- Duplicate detection via match fingerprinting
- Validation of xG ranges, shot counts, and team alignment

Usage
-----
::

    from src.data_collection.sources.understat import UnderstatImporter

    async with UnderstatImporter() as imp:
        # League-level xG data
        df = await imp.get_league_xg("EPL", 2024)

        # Match shot data with locations
        shots = await imp.get_match_shots(22260)

        # All xG for a league with incremental sync
        result = await imp.sync_league("EPL", 2024)
        print(result["matches_imported"], result["shots_imported"])

Optimizations
-------------
- Async HTTP via httpx with connection pooling
- Response caching with configurable TTL
- Batched match processing with semaphore concurrency
- Checkpoint-based incremental sync
"""

from __future__ import annotations

from src.data_collection.sources.understat.client import UnderstatClient
from src.data_collection.sources.understat.importer import UnderstatImporter
from src.data_collection.sources.understat.models import (
    LEAGUE_NAMES,
    MatchXG,
    ShotData,
    TeamXG,
)
from src.data_collection.sources.understat.parser import UnderstatParser

__all__ = [
    "UnderstatImporter",
    "UnderstatClient",
    "UnderstatParser",
    "MatchXG",
    "ShotData",
    "TeamXG",
    "LEAGUE_NAMES",
]
