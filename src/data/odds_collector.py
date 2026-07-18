"""
Odds Collector — multi-source odds aggregation, best-odds identification,
arbitrage detection, and historical odds persistence.

Sources supported
-----------------
1. **The Odds API** (the-odds-api.com) — Primary source for live odds
   across 50+ bookmakers. Requires ``THE_ODDS_API_KEY`` env var.
2. **Historical CSV** — Pre-collected odds from Football-Data.co.uk format
   stored in the preprocessed dataset.

Features
--------
- Multi-bookmaker best-odds identification (per outcome)
- Arbitrage / sure-bet detection across all bookmakers
- BTTS and Over/Under market odds (when available)
- Historical odds lookup from processed data
- Odds caching with configurable TTL
- CLV calculation (your odds vs closing)

Usage
-----
::

    from src.data.odds_collector import OddsCollector

    collector = OddsCollector()
    odds = collector.get_best_odds("France", "England", sport_key="soccer_fifa_world_cup")
    # Returns best odds across all bookmakers for each outcome

    arb = collector.find_arbitrage("France", "England")
    # Returns arbitrage opportunity if one exists
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

API_BASE_URL = "https://api.the-odds-api.com/v4"
CACHE_DIR = Path("data") / "external" / "odds_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_SCHEMA_VERSION = "2.0"

MARKETS_1X2 = "h2h"
MARKETS_BTTS = "btts"
MARKETS_OU = "totals"

DEFAULT_SPORT = "soccer_fifa_world_cup"
DEFAULT_REGIONS = "us,uk,eu"

_FOOTBALL_DATA_COLS: dict[str, tuple[str, str, str]] = {
    "Bet365": ("B365H", "B365D", "B365A"),
    "Bet&Win": ("BWH", "BWD", "BWA"),
    "Interwetten": ("IWH", "IWD", "IWA"),
    "Ladbrokes": ("LBH", "LBD", "LBA"),
    "Pinnacle": ("PSH", "PSD", "PSA"),
    "William Hill": ("WHH", "WHD", "WHA"),
    "VC Bet": ("VCH", "VCD", "VCA"),
    "Betbrain avg": ("BbAvH", "BbAvD", "BbAvA"),
    "Betbrain max": ("BbMxH", "BbMxD", "BbMxA"),
}

# ── Data structures ─────────────────────────────────────


@dataclass
class BookmakerOdds:
    """Odds from a single bookmaker for a single match."""

    bookmaker: str
    home_odds: float
    draw_odds: float
    away_odds: float
    timestamp: str = ""


@dataclass
class MatchOddsCollection:
    """All odds data gathered for a single match."""

    home_team: str
    away_team: str
    match_date: str = ""
    sport_key: str = ""
    bookmakers: list[BookmakerOdds] = field(default_factory=list)

    @property
    def best_home_odds(self) -> tuple[float, str]:
        """Best home odds across all bookmakers."""
        if not self.bookmakers:
            return (0.0, "")
        best = max(self.bookmakers, key=lambda b: b.home_odds)
        return (best.home_odds, best.bookmaker)

    @property
    def best_draw_odds(self) -> tuple[float, str]:
        """Best draw odds across all bookmakers."""
        if not self.bookmakers:
            return (0.0, "")
        best = max(self.bookmakers, key=lambda b: b.draw_odds)
        return (best.draw_odds, best.bookmaker)

    @property
    def best_away_odds(self) -> tuple[float, str]:
        """Best away odds across all bookmakers."""
        if not self.bookmakers:
            return (0.0, "")
        best = max(self.bookmakers, key=lambda b: b.away_odds)
        return (best.away_odds, best.bookmaker)

    @property
    def best_combined(self) -> dict[str, float]:
        """Best odds for each outcome, potentially from different bookmakers."""
        h_odds, h_bk = self.best_home_odds
        d_odds, d_bk = self.best_draw_odds
        a_odds, a_bk = self.best_away_odds
        return {
            "home_odds": h_odds,
            "home_bookmaker": h_bk,
            "draw_odds": d_odds,
            "draw_bookmaker": d_bk,
            "away_odds": a_odds,
            "away_bookmaker": a_bk,
        }

    def check_arbitrage(self) -> dict[str, Any]:
        """Check if an arbitrage (sure bet) opportunity exists.

        Arbitrage exists when the sum of inverse best odds < 1.0.
        Validates odds before calculation — rejects zero, negative,
        NaN, and infinite values.
        """
        best = self.best_combined
        h, d, a = best["home_odds"], best["draw_odds"], best["away_odds"]

        if not _validate_odds_triple(h, d, a):
            return {"is_arbitrage": False, "arb_pct": 0.0}

        inv_sum = 1.0 / h + 1.0 / d + 1.0 / a
        arb_pct = (1.0 - inv_sum) * 100

        return {
            "is_arbitrage": arb_pct > 0.0,
            "arb_pct": round(arb_pct, 3),
            "inv_sum": round(inv_sum, 4),
            "best_odds": best,
            "stake_allocation": {
                "home_pct": round((1.0 / h) / inv_sum * 100, 2) if inv_sum > 0 else 0,
                "draw_pct": round((1.0 / d) / inv_sum * 100, 2) if inv_sum > 0 else 0,
                "away_pct": round((1.0 / a) / inv_sum * 100, 2) if inv_sum > 0 else 0,
            },
            "guaranteed_return_pct": round(arb_pct, 2),
        }


# ═══════════════════════════════════════════════════════════
#  Main Collector
# ═══════════════════════════════════════════════════════════


class OddsCollector:
    """Multi-source odds collector with best-odds and arbitrage detection.

    Parameters
    ----------
    api_key : str, optional
        The Odds API key. Reads from ``THE_ODDS_API_KEY`` env var if not provided.
    regions : str
        Bookmaker regions (default ``"us,uk,eu"``).
    cache_ttl : int
        Cache TTL in seconds (default 1800 = 30 min).
    """

    def __init__(
        self,
        api_key: str | None = None,
        regions: str = DEFAULT_REGIONS,
        cache_ttl: int = 1800,
    ):
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY", "")
        self.regions = regions
        self.cache_ttl = cache_ttl
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._historical_odds_cache: dict[str, Any] | None = None
        self._historical_odds_loaded = False

        if not self.api_key:
            logger.warning(
                "THE_ODDS_API_KEY not set. Live odds unavailable. "
                "Set it in .env or as environment variable."
            )

    # ── Public API ──────────────────────────────────────

    def get_best_odds(
        self,
        home_team: str,
        away_team: str,
        sport_key: str = DEFAULT_SPORT,
        use_historical: bool = True,
    ) -> dict[str, Any] | None:
        """Get the best available odds across all bookmakers for a match.

        Tries live API first, falls back to historical CSV data.

        Parameters
        ----------
        home_team, away_team : str
            Team names.
        sport_key : str
            Sport key for The Odds API.
        use_historical : bool
            Whether to fall back to historical odds if live unavailable.

        Returns
        -------
        dict or None
            Best odds dict or None if no odds found.
        """
        # Try live API first
        collection = self._fetch_live_match_odds(home_team, away_team, sport_key)
        if collection and collection.bookmakers:
            result = collection.best_combined
            result["arbitrage"] = collection.check_arbitrage()
            result["source"] = "live"
            result["all_bookmakers"] = [
                {"name": b.bookmaker, "odds": [b.home_odds, b.draw_odds, b.away_odds]}
                for b in collection.bookmakers
            ]
            return result

        # Fall back to historical
        if use_historical:
            hist = self._get_historical_odds(home_team, away_team)
            if hist:
                hist["source"] = "historical"
                return hist

        return None

    def get_bulk_odds(
        self,
        team_pairs: list[tuple[str, str]],
        sport_key: str = DEFAULT_SPORT,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Get best odds for multiple matches in bulk.

        Parameters
        ----------
        team_pairs : list[tuple[str, str]]
            List of (home_team, away_team) tuples.
        sport_key : str
            Sport key.

        Returns
        -------
        dict[tuple[str, str], dict]
            Mapping of match -> best odds result.
        """
        results: dict[tuple[str, str], dict[str, Any]] = {}

        # Fetch all live matches once
        live_matches = self._fetch_live_odds_bulk(sport_key)
        # Build lookup
        lookup: dict[str, MatchOddsCollection] = {}
        for mc in live_matches:
            key = f"{mc.home_team.lower()}|{mc.away_team.lower()}"
            lookup[key] = mc

        for h, a in team_pairs:
            key = f"{h.lower()}|{a.lower()}"
            mc = lookup.get(key)
            if mc is None:
                # Try swapped
                swapped_key = f"{a.lower()}|{h.lower()}"
                mc = lookup.get(swapped_key)
                if mc:
                    # Swap teams and odds
                    result = {
                        "home_odds": mc.best_combined["away_odds"],
                        "home_bookmaker": mc.best_combined["away_bookmaker"],
                        "draw_odds": mc.best_combined["draw_odds"],
                        "draw_bookmaker": mc.best_combined["draw_bookmaker"],
                        "away_odds": mc.best_combined["home_odds"],
                        "away_bookmaker": mc.best_combined["home_bookmaker"],
                        "source": "live",
                    }
                    # Recalculate arbitrage with swapped odds
                    swapped_h, swapped_d, swapped_a = (
                        mc.best_combined["away_odds"],
                        mc.best_combined["draw_odds"],
                        mc.best_combined["home_odds"],
                    )
                    if _validate_odds_triple(swapped_h, swapped_d, swapped_a):
                        inv_sum = 1.0 / swapped_h + 1.0 / swapped_d + 1.0 / swapped_a
                        arb_pct = (1.0 - inv_sum) * 100
                        result["arbitrage"] = {
                            "is_arbitrage": arb_pct > 0.0,
                            "arb_pct": round(arb_pct, 3),
                            "inv_sum": round(inv_sum, 4),
                            "stake_allocation": {
                                "home_pct": round((1.0 / swapped_h) / inv_sum * 100, 2),
                                "draw_pct": round((1.0 / swapped_d) / inv_sum * 100, 2),
                                "away_pct": round((1.0 / swapped_a) / inv_sum * 100, 2),
                            },
                            "guaranteed_return_pct": round(arb_pct, 2),
                        }
                    else:
                        result["arbitrage"] = {"is_arbitrage": False, "arb_pct": 0.0}
                    results[(h, a)] = result
                    continue

                # Try historical
                hist = self._get_historical_odds(h, a)
                if hist:
                    hist["source"] = "historical"
                    results[(h, a)] = hist
                continue

            result = mc.best_combined
            result["source"] = "live"
            result["arbitrage"] = mc.check_arbitrage()
            results[(h, a)] = result

        return results

    def find_arbitrage(
        self,
        home_team: str,
        away_team: str,
        sport_key: str = DEFAULT_SPORT,
    ) -> dict[str, Any] | None:
        """Check if an arbitrage opportunity exists for a match.

        Returns arbitrage details if found, None otherwise.
        """
        odds = self.get_best_odds(home_team, away_team, sport_key)
        if odds is None:
            return None
        return odds.get("arbitrage")

    def find_all_arbitrage(
        self,
        sport_key: str = DEFAULT_SPORT,
    ) -> list[dict[str, Any]]:
        """Scan all available matches for arbitrage opportunities.

        Returns
        -------
        list[dict]
            List of arbitrage opportunities found.
        """
        live_matches = self._fetch_live_odds_bulk(sport_key)
        opportunities = []
        for mc in live_matches:
            arb = mc.check_arbitrage()
            if arb["is_arbitrage"]:
                opportunities.append(
                    {
                        "home_team": mc.home_team,
                        "away_team": mc.away_team,
                        "match_date": mc.match_date,
                        **arb,
                    }
                )
        return opportunities

    def get_historical_bookmaker_odds(
        self,
        home_team: str,
        away_team: str,
    ) -> list[dict[str, Any]]:
        """Get historical odds broken down by bookmaker for a match.

        Returns
        -------
        list[dict]
            List of {bookmaker, home_odds, draw_odds, away_odds} dicts.
        """
        self._load_historical_odds()
        if self._historical_odds_cache is None:
            return []

        df = self._historical_odds_cache
        match = df[
            (df["home_team"].str.lower() == home_team.lower())
            & (df["away_team"].str.lower() == away_team.lower())
        ]
        if match.empty:
            return []

        match = match.iloc[0]
        results = []
        for bk_name, (h_col, d_col, a_col) in _FOOTBALL_DATA_COLS.items():
            h = float(match.get(h_col, 0)) if h_col in match.index else 0
            d = float(match.get(d_col, 0)) if d_col in match.index else 0
            a = float(match.get(a_col, 0)) if a_col in match.index else 0
            if h > 0 and a > 0:
                results.append(
                    {
                        "bookmaker": bk_name,
                        "home_odds": h,
                        "draw_odds": d,
                        "away_odds": a,
                    }
                )
        return results

    def get_bookmaker_spreads(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any]:
        """Analyze odds spreads across bookmakers for a match.

        Wide spreads indicate market disagreement — possible value.

        Returns
        -------
        dict
            {home_spread, draw_spread, away_spread, max_disagreement}
        """
        bk_odds = self.get_historical_bookmaker_odds(home_team, away_team)
        bk_odds += self._get_live_bookmaker_odds(home_team, away_team)

        if not bk_odds:
            return {
                "home_spread": 0,
                "draw_spread": 0,
                "away_spread": 0,
                "n_bookmakers": 0,
            }

        home_vals = [b["home_odds"] for b in bk_odds]
        draw_vals = [b["draw_odds"] for b in bk_odds]
        away_vals = [b["away_odds"] for b in bk_odds]

        def spread(vals):
            return round(max(vals) - min(vals), 3) if vals else 0

        return {
            "home_spread": spread(home_vals),
            "draw_spread": spread(draw_vals),
            "away_spread": spread(away_vals),
            "n_bookmakers": len(bk_odds),
            "max_disagreement": max(
                spread(home_vals), spread(draw_vals), spread(away_vals)
            ),
        }

    def compute_clv(
        self,
        your_odds: float,
        closing_odds: float | None,
    ) -> dict[str, Any]:
        """Compute Closing Line Value from odds comparison.

        CLV = (your_odds - closing_odds) / closing_odds

        Positive CLV means you beat the closing line.
        """
        if closing_odds is None or closing_odds <= 1.0 or your_odds <= 1.0:
            return {"clv": 0.0, "clv_pct": 0.0, "positive": False}
        clv = (your_odds - closing_odds) / closing_odds
        return {
            "clv": round(clv, 6),
            "clv_pct": round(clv * 100, 4),
            "positive": clv > 0,
            "your_odds": your_odds,
            "closing_odds": closing_odds,
        }

    # ── Internal: Live odds fetching ────────────────────

    def _fetch_live_match_odds(
        self,
        home_team: str,
        away_team: str,
        sport_key: str,
    ) -> MatchOddsCollection | None:
        """Fetch odds from The Odds API for a single match."""
        if not self.api_key:
            return None

        matches = self._fetch_live_odds_bulk(sport_key)
        for mc in matches:
            if (
                mc.home_team.lower() == home_team.lower()
                and mc.away_team.lower() == away_team.lower()
            ):
                return mc
            # Try swapped
            if (
                mc.home_team.lower() == away_team.lower()
                and mc.away_team.lower() == home_team.lower()
            ):
                # Swap team names in collection
                mc.home_team, mc.away_team = mc.away_team, mc.home_team
                for bk in mc.bookmakers:
                    bk.home_odds, bk.away_odds = bk.away_odds, bk.home_odds
                return mc
        return None

    def _fetch_live_odds_bulk(self, sport_key: str) -> list[MatchOddsCollection]:
        """Fetch all upcoming matches for a sport."""
        if not self.api_key:
            return []

        cache_key = f"live_{sport_key}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        url = f"{API_BASE_URL}/sports/{sport_key}/odds"

        # Try markets from richest to most basic; fall back if 422
        market_sets = [
            f"{MARKETS_1X2},{MARKETS_BTTS},{MARKETS_OU}",
            f"{MARKETS_1X2},{MARKETS_OU}",
            MARKETS_1X2,
        ]

        data = None
        for markets in market_sets:
            params = {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": markets,
            }
            try:
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 422:
                    logger.warning(
                        "Markets %s not available for %s, trying fewer markets",
                        markets, sport_key,
                    )
                    continue
                logger.error("Failed to fetch odds: %s", e)
                return []
            except Exception as e:
                logger.error("Failed to fetch odds: %s", e)
                return []

        if data is None:
            logger.error("Failed to fetch odds for %s: all market sets failed", sport_key)
            return []

        collections = self._parse_response(data)
        self._save_cache(cache_key, collections)
        return collections

    def _parse_response(self, data: list[dict]) -> list[MatchOddsCollection]:
        """Parse API response into MatchOddsCollection objects."""
        collections = []
        for event in data:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence_time = event.get("commence_time", "")
            sport_key = event.get("sport_key", "")
            bookmakers_raw = event.get("bookmakers", [])

            if not bookmakers_raw:
                continue

            mc = MatchOddsCollection(
                home_team=home_team,
                away_team=away_team,
                match_date=commence_time,
                sport_key=sport_key,
            )

            for bk in bookmakers_raw:
                bk_title = bk.get("title", "unknown")
                for market in bk.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    odds_map = {o["name"]: o["price"] for o in outcomes}
                    mc.bookmakers.append(
                        BookmakerOdds(
                            bookmaker=bk_title,
                            home_odds=odds_map.get(home_team, 0.0),
                            draw_odds=odds_map.get("Draw", 0.0),
                            away_odds=odds_map.get(away_team, 0.0),
                            timestamp=commence_time,
                        )
                    )

            if mc.bookmakers:
                collections.append(mc)

        return collections

    def _get_live_bookmaker_odds(
        self,
        home_team: str,
        away_team: str,
    ) -> list[dict[str, Any]]:
        """Get per-bookmaker odds from live API."""
        mc = self._fetch_live_match_odds(home_team, away_team, DEFAULT_SPORT)
        if not mc:
            return []
        return [
            {
                "bookmaker": b.bookmaker,
                "home_odds": b.home_odds,
                "draw_odds": b.draw_odds,
                "away_odds": b.away_odds,
            }
            for b in mc.bookmakers
        ]

    # ── Internal: Historical odds ────────────────────────

    def _load_historical_odds(self) -> None:
        """Load historical odds from preprocessed data."""
        if self._historical_odds_loaded:
            return
        self._historical_odds_loaded = True

        csv_path = Path("data/processed/results_clean.csv")
        if not csv_path.exists():
            logger.debug("No historical odds CSV at %s", csv_path)
            return

        try:
            df = pd.read_csv(csv_path, low_memory=False, nrows=50000)
            self._historical_odds_cache = df
            logger.debug("Loaded %d historical odds rows", len(df))
        except Exception as e:
            logger.warning("Failed to load historical odds: %s", e)

    def _get_historical_odds(
        self,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any] | None:
        """Look up historical odds from preprocessed data."""
        self._load_historical_odds()
        if self._historical_odds_cache is None:
            return None

        df = self._historical_odds_cache
        match = df[
            (df["home_team"].str.lower() == home_team.lower())
            & (df["away_team"].str.lower() == away_team.lower())
        ]
        if match.empty:
            return None

        match = match.iloc[0]
        # Use best available odds columns
        h = float(match.get("BbAvH", match.get("B365H", 0)))
        d = float(match.get("BbAvD", match.get("B365D", 0)))
        a = float(match.get("BbAvA", match.get("B365A", 0)))
        if not _validate_odds_triple(h, d, a):
            return None

        return {
            "home_odds": h,
            "draw_odds": d,
            "away_odds": a,
            "source": "historical",
            "arbitrage": self._check_arb_simple(h, d, a),
        }

    @staticmethod
    def _check_arb_simple(h: float, d: float, a: float) -> dict[str, Any]:
        """Simple arbitrage check for three odds.

        Delegates to a MatchOddsCollection for consistent logic.
        """
        mc = MatchOddsCollection(home_team="", away_team="")
        mc.bookmakers.append(
            BookmakerOdds(
                bookmaker="combined",
                home_odds=h,
                draw_odds=d,
                away_odds=a,
            )
        )
        return mc.check_arbitrage()

    # ── Caching ─────────────────────────────────────────

    @staticmethod
    def _sanitize_cache_key(key: str) -> str:
        """Sanitize a cache key: lowercase, replace special chars, strip."""
        import re

        safe = re.sub(r"[^a-zA-Z0-9_]", "_", key).lower().strip("_")
        return safe or "_default_"

    def _cache_path(self, key: str) -> Path:
        safe_key = self._sanitize_cache_key(key)
        return CACHE_DIR / f"{safe_key}.json"

    def _load_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                entry = json.load(f)
            # Schema version check
            schema_ver = entry.get("schema_version", "")
            if schema_ver != CACHE_SCHEMA_VERSION:
                logger.debug(
                    "Cache schema mismatch: expected %s, got %s — purging",
                    CACHE_SCHEMA_VERSION,
                    schema_ver,
                )
                path.unlink(missing_ok=True)
                return None
            # TTL check
            age = time.time() - entry.get("timestamp", 0)
            if age > self.cache_ttl:
                path.unlink(missing_ok=True)
                return None
            data = entry.get("data")
            return _deserialize_cache_data(data)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            logger.debug("Cache read error for %s — purging", key)
            path.unlink(missing_ok=True)
            return None

    def _save_cache(self, key: str, data: Any) -> None:
        """Cache data to JSON file with atomic write and schema versioning."""
        import tempfile

        try:
            serializable = _make_serializable(data)
            entry = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "timestamp": time.time(),
                "data": serializable,
            }
            path = self._cache_path(key)
            # Atomic write: write to temp file, then rename
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=".json",
                dir=str(CACHE_DIR),
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(entry, f, indent=2)
                os.replace(tmp_path, str(path))
            except Exception:
                # Cleanup temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning("Failed to save cache: %s", e)


# ═══════════════════════════════════════════════════════════
#  Convenience function
# ═══════════════════════════════════════════════════════════


def _make_serializable(obj: Any) -> Any:
    """Convert non-serializable objects for JSON caching."""
    if isinstance(obj, (MatchOddsCollection, BookmakerOdds)):
        # Convert dataclass instances to dicts
        return {k: _make_serializable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def _deserialize_cache_data(data: Any) -> Any:
    """Reconstruct dataclass objects from cached dictionaries.

    Inverse of ``_make_serializable``. Converts dicts back into
    ``MatchOddsCollection`` and ``BookmakerOdds`` instances.
    """
    if isinstance(data, list):
        # Heuristic: if items look like MatchOddsCollection dicts, convert them
        result = []
        for item in data:
            if (
                isinstance(item, dict)
                and "home_team" in item
                and "away_team" in item
                and "bookmakers" in item
            ):
                # Rebuild MatchOddsCollection
                bookmakers = []
                for bk in item.get("bookmakers", []):
                    if isinstance(bk, dict) and "bookmaker" in bk:
                        bookmakers.append(
                            BookmakerOdds(
                                bookmaker=bk["bookmaker"],
                                home_odds=bk.get("home_odds", 0.0),
                                draw_odds=bk.get("draw_odds", 0.0),
                                away_odds=bk.get("away_odds", 0.0),
                                timestamp=bk.get("timestamp", ""),
                            )
                        )
                result.append(
                    MatchOddsCollection(
                        home_team=item["home_team"],
                        away_team=item["away_team"],
                        match_date=item.get("match_date", ""),
                        sport_key=item.get("sport_key", ""),
                        bookmakers=bookmakers,
                    )
                )
            else:
                result.append(_deserialize_cache_data(item))
        return result
    if isinstance(data, dict):
        return {k: _deserialize_cache_data(v) for k, v in data.items()}
    return data


def find_best_odds(
    home_team: str,
    away_team: str,
    sport_key: str = DEFAULT_SPORT,
) -> dict[str, Any] | None:
    """One-shot convenience: get the best odds for a match."""
    collector = OddsCollector()
    return collector.get_best_odds(home_team, away_team, sport_key)


def detect_arbitrage(
    home_team: str,
    away_team: str,
    sport_key: str = DEFAULT_SPORT,
) -> dict[str, Any] | None:
    """One-shot convenience: check for arbitrage."""
    collector = OddsCollector()
    return collector.find_arbitrage(home_team, away_team, sport_key)


# ═══════════════════════════════════════════════════════════
#  Odds validation utilities
# ═══════════════════════════════════════════════════════════


def _validate_odds_single(odds: float) -> bool:
    """Check that a single odds value is valid (> 1.0, finite, not NaN).

    Returns True if valid, False otherwise.
    """
    if odds <= 1.0:
        return False
    if not np.isfinite(odds):
        return False
    if np.isnan(odds):
        return False
    return True


def _validate_odds_triple(home_odds: float, draw_odds: float, away_odds: float) -> bool:
    """Validate all three odds for a 1X2 market.

    All three must be valid (no zero, negative, NaN, or infinite values).
    Draw odds may be 0.0 if the market doesn't support draws, but home
    and away must always be > 1.0.

    Returns True if all required odds are valid.
    """
    if not _validate_odds_single(home_odds):
        return False
    if not _validate_odds_single(away_odds):
        return False
    # Draw odds are optional (some markets don't have draws)
    if draw_odds > 0 and not _validate_odds_single(draw_odds):
        return False
    return True
