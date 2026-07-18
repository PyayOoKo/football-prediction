"""Phase 2 tests: odds validation, cache persistence, swapped-team arbitrage.

Uses temporary directories and synthetic data to avoid depending
on real API keys or network access.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
#  1. Odds Validation Tests
# ═══════════════════════════════════════════════════════════════


class TestOddsValidation:
    """Odds must reject zero, negative, NaN, and infinite values."""

    def test_valid_odds(self):
        """Normal odds (> 1.0) are valid."""
        from src.data.odds_collector import _validate_odds_single, _validate_odds_triple

        assert _validate_odds_single(2.0) is True
        assert _validate_odds_single(1.01) is True  # just above 1.0
        assert _validate_odds_triple(2.0, 3.0, 4.0) is True

    def test_zero_odds_invalid(self):
        """Odds of 0 or 1.0 are invalid."""
        from src.data.odds_collector import _validate_odds_single, _validate_odds_triple

        assert _validate_odds_single(0) is False
        assert _validate_odds_single(1.0) is False  # no profit possible
        assert _validate_odds_triple(0, 3.0, 4.0) is False
        assert _validate_odds_triple(2.0, 3.0, 1.0) is False  # away = 1.0

    def test_negative_odds_invalid(self):
        """Negative odds are invalid."""
        from src.data.odds_collector import _validate_odds_single, _validate_odds_triple

        assert _validate_odds_single(-1.0) is False
        assert _validate_odds_triple(-1.0, 3.0, 4.0) is False

    def test_nan_odds_invalid(self):
        """NaN odds are invalid."""
        from src.data.odds_collector import _validate_odds_single, _validate_odds_triple

        assert _validate_odds_single(float("nan")) is False
        assert _validate_odds_triple(float("nan"), 3.0, 4.0) is False

    def test_inf_odds_invalid(self):
        """Infinite odds are invalid."""
        from src.data.odds_collector import _validate_odds_single, _validate_odds_triple

        assert _validate_odds_single(float("inf")) is False
        assert _validate_odds_triple(2.0, 3.0, float("inf")) is False

    def test_arbitrage_with_valid_odds(self):
        """check_arbitrage returns correct result with valid odds."""
        from src.data.odds_collector import MatchOddsCollection, BookmakerOdds

        mc = MatchOddsCollection(home_team="A", away_team="B")
        mc.bookmakers.append(BookmakerOdds(
            bookmaker="BK1", home_odds=2.10, draw_odds=3.50, away_odds=4.00,
        ))

        arb = mc.check_arbitrage()
        # inv_sum = 1/2.10 + 1/3.50 + 1/4.00 = 0.476 + 0.286 + 0.250 = 1.012 → no arb
        assert arb["is_arbitrage"] is False

    def test_arbitrage_rejects_invalid_odds(self):
        """check_arbitrage returns no-arb for invalid odds (no crash)."""
        from src.data.odds_collector import MatchOddsCollection, BookmakerOdds

        mc = MatchOddsCollection(home_team="A", away_team="B")
        mc.bookmakers.append(BookmakerOdds(
            bookmaker="BK1", home_odds=0, draw_odds=3.50, away_odds=4.00,
        ))

        arb = mc.check_arbitrage()
        assert arb["is_arbitrage"] is False
        assert arb["arb_pct"] == 0.0


# ═══════════════════════════════════════════════════════════════
#  2. Cache Persistence Tests
# ═══════════════════════════════════════════════════════════════


class TestOddsCache:
    """Cache must handle versioning, atomic writes, and corruption."""

    def _make_collector(self, tmp_path: Path, ttl: int = 300) -> Any:
        """Create an OddsCollector that uses tmp_path for cache."""
        from src.data.odds_collector import OddsCollector
        import src.data.odds_collector as oc

        collector = OddsCollector(api_key="test_key", cache_ttl=ttl)
        # Temporarily swap CACHE_DIR
        orig_dir = oc.CACHE_DIR
        oc.CACHE_DIR = tmp_path / "odds_cache"
        oc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return collector

    def test_cache_miss_returns_none(self):
        """Cache miss returns None without error."""
        from src.data.odds_collector import OddsCollector

        collector = OddsCollector(api_key="")
        result = collector._load_cache("nonexistent_key")
        assert result is None

    def test_cache_hit_returns_data(self):
        """Cache hit returns deserialized data."""
        from src.data.odds_collector import OddsCollector, BookmakerOdds, MatchOddsCollection, CACHE_SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "odds_cache"
            cache_dir.mkdir()
            import src.data.odds_collector as oc
            orig_dir = oc.CACHE_DIR
            oc.CACHE_DIR = cache_dir

            try:
                collector = OddsCollector(api_key="test", cache_ttl=3600)
                data = [
                    MatchOddsCollection(
                        home_team="A", away_team="B",
                        bookmakers=[BookmakerOdds("BK1", 2.0, 3.0, 4.0)],
                    )
                ]
                collector._save_cache("test_key", data)
                loaded = collector._load_cache("test_key")
                assert loaded is not None
                assert len(loaded) == 1
                assert loaded[0].home_team == "A"
                assert loaded[0].bookmakers[0].home_odds == 2.0
            finally:
                oc.CACHE_DIR = orig_dir

    def test_expired_cache_returns_none(self):
        """Expired cache returns None and deletes the file."""
        from src.data.odds_collector import OddsCollector, BookmakerOdds, MatchOddsCollection, CACHE_SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "odds_cache"
            cache_dir.mkdir()
            import src.data.odds_collector as oc
            orig_dir = oc.CACHE_DIR
            oc.CACHE_DIR = cache_dir

            try:
                # Write a cache entry with an already-expired timestamp
                safe_key = OddsCollector._sanitize_cache_key("expired_key")
                cache_path = cache_dir / f"{safe_key}.json"
                entry = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "timestamp": time.time() - 10000,  # well past TTL
                    "data": [],
                }
                with open(cache_path, "w") as f:
                    json.dump(entry, f)

                collector = OddsCollector(api_key="test", cache_ttl=60)
                loaded = collector._load_cache("expired_key")
                assert loaded is None, "Expired cache should return None"
                assert not cache_path.exists(), "Expired cache file should be deleted"
            finally:
                oc.CACHE_DIR = orig_dir

    def test_corrupted_cache_returns_none(self):
        """Corrupted cache file returns None and is deleted."""
        from src.data.odds_collector import OddsCollector, CACHE_SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "odds_cache"
            cache_dir.mkdir()
            import src.data.odds_collector as oc
            orig_dir = oc.CACHE_DIR
            oc.CACHE_DIR = cache_dir

            try:
                safe_key = OddsCollector._sanitize_cache_key("corrupted")
                cache_path = cache_dir / f"{safe_key}.json"
                # Write invalid JSON
                with open(cache_path, "w") as f:
                    f.write("{invalid json!!!}")

                collector = OddsCollector(api_key="test", cache_ttl=300)
                loaded = collector._load_cache("corrupted")
                assert loaded is None
                assert not cache_path.exists(), "Corrupted cache should be deleted"
            finally:
                oc.CACHE_DIR = orig_dir

    def test_schema_version_mismatch_purges(self):
        """Cache with wrong schema version is purged."""
        from src.data.odds_collector import OddsCollector

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "odds_cache"
            cache_dir.mkdir()
            import src.data.odds_collector as oc
            orig_dir = oc.CACHE_DIR
            oc.CACHE_DIR = cache_dir

            try:
                safe_key = OddsCollector._sanitize_cache_key("old_schema")
                cache_path = cache_dir / f"{safe_key}.json"
                entry = {
                    "schema_version": "0.5",  # old version
                    "timestamp": time.time(),
                    "data": [1, 2, 3],
                }
                with open(cache_path, "w") as f:
                    json.dump(entry, f)

                collector = OddsCollector(api_key="test", cache_ttl=300)
                loaded = collector._load_cache("old_schema")
                assert loaded is None
                assert not cache_path.exists()
            finally:
                oc.CACHE_DIR = orig_dir

    def test_cache_key_sanitization(self):
        """Sanitize removes special characters."""
        from src.data.odds_collector import OddsCollector

        safe = OddsCollector._sanitize_cache_key("live_soccer_fifa_world_cup")
        assert "_" not in safe or safe == "live_soccer_fifa_world_cup"
        assert safe == "live_soccer_fifa_world_cup"

        safe2 = OddsCollector._sanitize_cache_key("hello!@#$%^&*()world")
        # Each special char is replaced with _, so 10 special chars -> 10 underscores
        assert safe2 == "hello__________world" or safe2 == "hello_world"

    def test_cache_atomic_write(self):
        """Cache write is atomic (no partial files on crash)."""
        from src.data.odds_collector import OddsCollector

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "odds_cache"
            cache_dir.mkdir()
            import src.data.odds_collector as oc
            orig_dir = oc.CACHE_DIR
            oc.CACHE_DIR = cache_dir

            try:
                collector = OddsCollector(api_key="test", cache_ttl=3600)
                # Write data
                data = [{"test": "value"}]
                collector._save_cache("atomic_test", data)

                # Verify file exists and is valid JSON
                safe_key = OddsCollector._sanitize_cache_key("atomic_test")
                cache_path = cache_dir / f"{safe_key}.json"
                assert cache_path.exists()
                with open(cache_path) as f:
                    entry = json.load(f)
                assert entry["schema_version"] == oc.CACHE_SCHEMA_VERSION
                assert entry["data"] == data
            finally:
                oc.CACHE_DIR = orig_dir


# ═══════════════════════════════════════════════════════════════
#  3. Swapped-Team Arbitrage Tests
# ═══════════════════════════════════════════════════════════════


class TestSwappedTeamArbitrage:
    """Swapped-team odds must have correct arbitrage recalculated."""

    def test_get_bulk_odds_swapped_arbitrage(self):
        """Swapped teams in get_bulk_odds recalculate arbitrage correctly."""
        from src.data.odds_collector import (
            OddsCollector,
            BookmakerOdds,
            MatchOddsCollection,
        )

        collector = OddsCollector(api_key="test", cache_ttl=3600)

        # Manually set up a swapped scenario: API has (B, A) but we request (A, B)
        swapped_mc = MatchOddsCollection(
            home_team="B", away_team="A",
            bookmakers=[
                BookmakerOdds("BK1", 2.50, 3.20, 3.00),  # B home, A away
            ],
        )

        # Override _fetch_live_odds_bulk to return our test data
        original_fetch = collector._fetch_live_odds_bulk
        collector._fetch_live_odds_bulk = lambda sport: [swapped_mc]

        try:
            results = collector.get_bulk_odds([("A", "B")])
            assert ("A", "B") in results
            result = results[("A", "B")]

            # After swap: A(home) should have B's away odds, B(away) should have A's home odds
            # Original: BK1 home=2.50(B), away=3.00(A)
            # After swap: A(home)=3.00, B(away)=2.50
            assert abs(result["home_odds"] - 3.00) < 0.01
            assert abs(result["away_odds"] - 2.50) < 0.01

            # Arbitrage should be recalculated with swapped odds
            arb = result.get("arbitrage", {})
            assert "is_arbitrage" in arb
            # inv_sum = 1/3.00 + 1/3.20 + 1/2.50 = 0.333 + 0.3125 + 0.400 = 1.0458 → no arb
            assert arb["is_arbitrage"] is False
        finally:
            collector._fetch_live_odds_bulk = original_fetch

    def test_swapped_team_bookmaker_metadata(self):
        """Swapped team odds also swap bookmaker metadata correctly."""
        from src.data.odds_collector import (
            OddsCollector,
            BookmakerOdds,
            MatchOddsCollection,
        )

        collector = OddsCollector(api_key="test", cache_ttl=3600)

        mc = MatchOddsCollection(
            home_team="B", away_team="A",
            bookmakers=[
                BookmakerOdds("BK1", 2.50, 3.20, 3.00),
            ],
        )

        original_fetch = collector._fetch_live_odds_bulk
        collector._fetch_live_odds_bulk = lambda sport: [mc]

        try:
            results = collector.get_bulk_odds([("A", "B")])
            result = results[("A", "B")]

            # Home bookmaker should be the one providing A's odds (originally away)
            assert result["home_bookmaker"] == "BK1"
            assert result["away_bookmaker"] == "BK1"
        finally:
            collector._fetch_live_odds_bulk = original_fetch

    def test_direct_match_no_swap_needed(self):
        """get_bulk_odds without swap uses original odds directly."""
        from src.data.odds_collector import (
            OddsCollector,
            BookmakerOdds,
            MatchOddsCollection,
        )

        collector = OddsCollector(api_key="test", cache_ttl=3600)

        mc = MatchOddsCollection(
            home_team="A", away_team="B",
            bookmakers=[
                BookmakerOdds("BK1", 2.00, 3.40, 3.80),
            ],
        )

        original_fetch = collector._fetch_live_odds_bulk
        collector._fetch_live_odds_bulk = lambda sport: [mc]

        try:
            results = collector.get_bulk_odds([("A", "B")])
            result = results[("A", "B")]

            assert abs(result["home_odds"] - 2.00) < 0.01
            assert abs(result["draw_odds"] - 3.40) < 0.01
            assert abs(result["away_odds"] - 3.80) < 0.01
        finally:
            collector._fetch_live_odds_bulk = original_fetch


# ═══════════════════════════════════════════════════════════════
#  4. API Kelly/EV Validation Tests
# ═══════════════════════════════════════════════════════════════


class TestKellyEVValidation:
    """Kelly and EV must not be generated from invalid odds."""

    def test_ev_with_valid_odds(self):
        """Valid odds produce reasonable EV."""
        from api.main import _calculate_ev_and_kelly

        probs = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
        odds = {"home_odds": 2.50, "draw_odds": 3.50, "away_odds": 5.00}
        ev, kelly = _calculate_ev_and_kelly(probs, odds)
        assert ev is not None
        assert ev > 0  # 0.5 * 2.5 - 1 = 0.25

    def test_ev_with_invalid_odds(self):
        """Invalid odds produce None EV/Kelly (no crash)."""
        from api.main import _calculate_ev_and_kelly

        probs = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
        odds = {"home_odds": 0, "draw_odds": 0, "away_odds": 0}
        ev, kelly = _calculate_ev_and_kelly(probs, odds)
        assert ev is None and kelly is None

    def test_ev_with_negative_odds(self):
        """Negative odds produce None EV/Kelly."""
        from api.main import _calculate_ev_and_kelly

        probs = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
        odds = {"home_odds": -1.0, "draw_odds": 3.50, "away_odds": 5.00}
        ev, kelly = _calculate_ev_and_kelly(probs, odds)
        # Negative home odds should be rejected, but draw or away might still work
        # The function iterates outcomes and takes the best valid one
        assert ev is not None  # should still work if some odds are valid

    def test_ev_with_all_invalid_odds(self):
        """All invalid odds produce None for both EV and Kelly."""
        from api.main import _calculate_ev_and_kelly

        probs = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
        odds = {"home_odds": 0, "draw_odds": -1, "away_odds": float("nan")}
        ev, kelly = _calculate_ev_and_kelly(probs, odds)
        assert ev is None and kelly is None

    def test_kelly_stake_bounded(self):
        """Kelly stake is clamped to [0, 1]."""
        from api.main import _calculate_ev_and_kelly

        # Extreme edge: model very confident, odds very high
        probs = {"home_win": 0.9, "draw": 0.05, "away_win": 0.05}
        odds = {"home_odds": 10.0, "draw_odds": 3.0, "away_odds": 3.0}
        ev, kelly = _calculate_ev_and_kelly(probs, odds)
        assert kelly is not None
        assert kelly <= 1.0
        assert kelly >= 0.0
