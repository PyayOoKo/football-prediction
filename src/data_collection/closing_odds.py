"""
Closing Odds Collection — collect and store closing odds from multiple sources.

Sources
-------
1. **Football-Data.co.uk** — already-downloaded CSV files contain closing odds
   columns (BbAvH/D/A for 1X2, BbAv>2.5/<2.5 for O/U, etc.). This collector
   extracts those columns and matches them to database ``Match`` records.

2. **OddsPortal.com** — historical closing odds via web scraping. Provides 1X2,
   BTTS, and Over/Under closing odds with bookmaker consensus.

3. **BetExplorer.com** — historical match odds via scraping. Provides 1X2,
   BTTS, and Over/Under lines with detailed bookmaker breakdowns.

Architecture
------------
Each source has a ``collector`` class that:
  1. Fetches or extracts raw odds data
  2. Normalises to a standard schema
  3. Matches odds to ``Match`` records by (date, home_team, away_team, league)
  4. Upserts into the ``closing_odds`` database table

Usage
-----
::

    from src.data_collection.closing_odds import ClosingOddsOrchestrator

    collector = ClosingOddsOrchestrator()
    collector.collect_all(sources=["football_data", "betexplorer"])
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.database.base import Base
from src.database.session import get_session
from src import database  # ensure models registered

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════


@dataclass
class ClosingOddsRecord:
    """Normalised closing odds for a single match from one source.

    After ``match_to_database()``, ``match_id`` is populated with the
    matched ``Match.id`` (or None if no match found).
    """

    match_date: date
    home_team: str
    away_team: str
    league: str
    match_id: int | None = None  # populated by match_to_database()

    # 1X2 closing odds (decimal)
    odds_home: float | None = None
    odds_draw: float | None = None
    odds_away: float | None = None

    # BTTS closing odds
    btts_yes: float | None = None
    btts_no: float | None = None

    # Over/Under 2.5 closing odds
    over25: float | None = None
    under25: float | None = None

    # Source identifier
    source: str = ""

    # Timestamp when the odds were recorded (close to kick-off)
    odds_timestamp: datetime | None = None


# ═══════════════════════════════════════════════════════════
#  Match matching
# ═══════════════════════════════════════════════════════════


def _normalise_team(name: str) -> str:
    """Normalise a team name for matching across sources."""
    name = name.strip().lower()
    # Remove common suffixes/prefixes
    name = re.sub(r"\s+(fc|cf|afc|utd|united|city|town|rangers|athletic|"
                  r"ac|acm|as|ssc|ss|fk|bk|if|il|jk)$", "", name)
    name = re.sub(r"^(fc|cf|afc|ac|as|ssc|ss|fk|bk|if|il|jk)\s+", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def _match_key(row: dict[str, Any]) -> str:
    """Create a hashable match key from a dict with date, home_team, away_team."""
    return (
        _normalise_team(str(row.get("home_team", ""))),
        _normalise_team(str(row.get("away_team", ""))),
    )


def match_to_database(
    records: list[ClosingOddsRecord],
    session: Any,
    date_tolerance_days: int = 3,
) -> list[ClosingOddsRecord]:
    """Match closing odds records to existing database matches.

    For each record, looks up the ``Match`` table by
    (match_date ± tolerance, home_team, away_team) and populates
    ``match_id`` if a unique match is found.

    Parameters
    ----------
    records : list[ClosingOddsRecord]
        Closing odds records to match.
    session : sqlalchemy.orm.Session
        Active database session.
    date_tolerance_days : int
        Maximum days away from the recorded date to consider a match.

    Returns
    -------
    list[ClosingOddsRecord]
        Records with ``match_id`` populated where matched.
    """
    from src.database.models.match import Match
    from sqlalchemy import or_

    unmatched: list[ClosingOddsRecord] = []
    matched_count = 0
    multiple_count = 0

    for rec in records:
        if rec.match_date is None:
            unmatched.append(rec)
            continue

        # Build date range
        start = pd.Timestamp(rec.match_date) - pd.Timedelta(
            days=date_tolerance_days
        )
        end = pd.Timestamp(rec.match_date) + pd.Timedelta(
            days=date_tolerance_days
        )

        # Query database for potential matches in date range
        matches = (
            session.query(Match)
            .filter(Match.match_date.between(start, end))
            .all()
        )

        # Filter by normalised team names
        rec_home_norm = _normalise_team(rec.home_team)
        rec_away_norm = _normalise_team(rec.away_team)

        candidates: list[Match] = []
        for m in matches:
            db_home = _normalise_team(m.home_team.name if m.home_team else "")
            db_away = _normalise_team(m.away_team.name if m.away_team else "")
            if (db_home == rec_home_norm and db_away == rec_away_norm):
                candidates.append(m)
            elif (db_home == rec_away_norm and db_away == rec_home_norm):
                # Inverted home/away — flag but still match
                logger.debug(
                    "Inverted teams for %s vs %s on %s — swapping",
                    rec.home_team, rec.away_team, rec.match_date,
                )
                candidates.append(m)

        if len(candidates) == 1:
            rec.match_id = candidates[0].id  # type: ignore[attr-defined]
            matched_count += 1
        elif len(candidates) > 1:
            multiple_count += 1
            unmatched.append(rec)
        else:
            unmatched.append(rec)

    if matched_count:
        logger.info(
            "Matched %d / %d closing odds records to database matches",
            matched_count, len(records),
        )
    if multiple_count:
        logger.warning(
            "%d records matched MULTIPLE potential matches — skipped",
            multiple_count,
        )

    return [r for r in records if hasattr(r, "match_id") and r.match_id is not None]


# ═══════════════════════════════════════════════════════════
#  Database persistence
# ═══════════════════════════════════════════════════════════


def upsert_closing_odds(
    records: list[ClosingOddsRecord],
    session: Any,
) -> int:
    """Upsert closing odds records into the database.

    Uses ``match_id + source`` as the unique key.
    Returns the number of rows inserted or updated.
    """
    from src.database.models.closing_odds import ClosingOdds

    count = 0
    for rec in records:
        match_id = getattr(rec, "match_id", None)
        if match_id is None:
            continue

        # Check for existing record
        existing = (
            session.query(ClosingOdds)
            .filter(
                ClosingOdds.match_id == match_id,
                ClosingOdds.source == rec.source,
            )
            .first()
        )

        if existing:
            # Update existing
            existing.odds_home = rec.odds_home
            existing.odds_draw = rec.odds_draw
            existing.odds_away = rec.odds_away
            existing.btts_yes = rec.btts_yes
            existing.btts_no = rec.btts_no
            existing.over25 = rec.over25
            existing.under25 = rec.under25
            existing.timestamp = rec.odds_timestamp or datetime.now(timezone.utc)
        else:
            # Insert new
            new = ClosingOdds(
                match_id=match_id,
                source=rec.source,
                timestamp=rec.odds_timestamp or datetime.now(timezone.utc),
                odds_home=rec.odds_home,
                odds_draw=rec.odds_draw,
                odds_away=rec.odds_away,
                btts_yes=rec.btts_yes,
                btts_no=rec.btts_no,
                over25=rec.over25,
                under25=rec.under25,
            )
            session.add(new)
        count += 1

    session.flush()
    logger.info("Upserted %d closing odds records", count)
    return count


# ═══════════════════════════════════════════════════════════
#  Base collector
# ═══════════════════════════════════════════════════════════


class BaseClosingOddsCollector(ABC):
    """Abstract base for closing odds collectors.

    Subclasses must implement ``collect()`` which returns a list of
    ``ClosingOddsRecord`` for matches within the given date range.
    """

    source_name: str = ""

    @abstractmethod
    def collect(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        leagues: list[str] | None = None,
    ) -> list[ClosingOddsRecord]:
        """Collect closing odds for the given date range and leagues."""
        ...

    def run(
        self,
        session: Any | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        leagues: list[str] | None = None,
    ) -> int:
        """Collect and persist closing odds.

        Returns the number of records upserted.
        """
        records = self.collect(start_date, end_date, leagues)
        logger.info(
            "%s: collected %d closing odds records",
            self.source_name, len(records),
        )

        if not records:
            return 0

        own_session = session is None
        if own_session:
            from src.database.session import get_session as _get_session
            _session = _get_session()
        else:
            _session = session

        try:
            matched = match_to_database(records, _session)
            count = upsert_closing_odds(matched, _session)
            if own_session:
                _session.commit()
            return count
        finally:
            if own_session and _session:
                _session.close()


# ═══════════════════════════════════════════════════════════
#  Source 1: Football-Data.co.uk (CSV-based)
# ═══════════════════════════════════════════════════════════

# Mapping of Football-Data.co.uk columns to ClosingOddsRecord fields
_COLUMN_MAP_1X2 = {
    "bbavh": "odds_home",
    "bbavd": "odds_draw",
    "bbava": "odds_away",
}
_COLUMN_MAP_BTTS = {
    "bbavbts_yes": "btts_yes",
    "bbavbts_no": "btts_no",
}
_COLUMN_MAP_OU = {
    "bbav>2.5": "over25",
    "bbav<2.5": "under25",
}
# Fallback column names (some CSVs use different naming)
_FALLBACK_1X2 = {
    "b365h": "odds_home",
    "b365d": "odds_draw",
    "b365a": "odds_away",
}


class FootballDataClosingOddsCollector(BaseClosingOddsCollector):
    """Extract closing odds from Football-Data.co.uk CSV files.

    The CSV files already contain consensus closing odds columns
    (BbAvH/D/A, BbAv>2.5, etc.). This collector loads the raw CSVs
    and extracts those columns into ``ClosingOddsRecord`` objects.
    """

    source_name = "football-data"

    def __init__(self, raw_dir: str = "data/raw/football-data") -> None:
        self.raw_dir = raw_dir

    def collect(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        leagues: list[str] | None = None,
    ) -> list[ClosingOddsRecord]:
        """Load CSV files and extract closing odds columns."""
        from pathlib import Path

        data_dir = Path(self.raw_dir)
        if not data_dir.is_dir():
            logger.warning("Football-Data directory not found: %s", data_dir)
            return []

        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", data_dir)
            return []

        logger.info("Scanning %d CSV files for closing odds...", len(csv_files))

        all_records: list[ClosingOddsRecord] = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as exc:
                logger.warning("Could not read %s: %s", csv_path.name, exc)
                continue

            # Detect league code from filename (e.g. E0_2425.csv → E0)
            league_code = csv_path.stem.split("_")[0] if "_" in csv_path.stem else ""
            if leagues and league_code not in leagues:
                continue

            records = self._extract_from_dataframe(df, league_code)
            all_records.extend(records)

        # Filter by date range
        if start_date or end_date:
            filtered: list[ClosingOddsRecord] = []
            for r in all_records:
                if start_date and r.match_date < start_date:
                    continue
                if end_date and r.match_date > end_date:
                    continue
                filtered.append(r)
            all_records = filtered

        logger.info(
            "Extracted %d closing odds records from %d CSV files",
            len(all_records), len(csv_files),
        )
        return all_records

    def _extract_from_dataframe(
        self,
        df: pd.DataFrame,
        league_code: str,
    ) -> list[ClosingOddsRecord]:
        """Extract closing odds from a single DataFrame."""
        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect available column sets
        cols_1x2 = self._resolve_cols(df, [_COLUMN_MAP_1X2, _FALLBACK_1X2])
        cols_btts = self._resolve_cols(df, [_COLUMN_MAP_BTTS])
        cols_ou = self._resolve_cols(df, [_COLUMN_MAP_OU])

        if not cols_1x2:
            logger.debug("No 1X2 closing odds columns in this CSV")
            return []

        # Parse date column
        date_col = "date" if "date" in df.columns else None
        if date_col is None:
            return []

        df[date_col] = pd.to_datetime(
            df[date_col], dayfirst=True, errors="coerce",
        )
        df.dropna(subset=[date_col], inplace=True)

        # Extract home/away team columns
        home_col = "hometeam" if "hometeam" in df.columns else (
            "home_team" if "home_team" in df.columns else None
        )
        away_col = "awayteam" if "awayteam" in df.columns else (
            "away_team" if "away_team" in df.columns else None
        )
        if home_col is None or away_col is None:
            return []

        records: list[ClosingOddsRecord] = []
        for _, row in df.iterrows():
            try:
                match_date = row[date_col].date()
            except AttributeError:
                continue

            home_team = str(row.get(home_col, "")).strip()
            away_team = str(row.get(away_col, "")).strip()
            if not home_team or not away_team:
                continue

            rec = ClosingOddsRecord(
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                league=league_code,
                source=self.source_name,
                odds_timestamp=datetime.combine(
                    match_date, datetime.min.time(), tzinfo=timezone.utc,
                ),
            )

            # 1X2 odds
            for csv_col, rec_field in cols_1x2.items():
                val = row.get(csv_col)
                if val is not None and not pd.isna(val):
                    try:
                        setattr(rec, rec_field, float(val))
                    except (ValueError, TypeError):
                        pass

            # BTTS odds
            if cols_btts:
                for csv_col, rec_field in cols_btts.items():
                    val = row.get(csv_col)
                    if val is not None and not pd.isna(val):
                        try:
                            setattr(rec, rec_field, float(val))
                        except (ValueError, TypeError):
                            pass

            # Over/Under odds
            if cols_ou:
                for csv_col, rec_field in cols_ou.items():
                    val = row.get(csv_col)
                    if val is not None and not pd.isna(val):
                        try:
                            setattr(rec, rec_field, float(val))
                        except (ValueError, TypeError):
                            pass

            records.append(rec)

        return records

    @staticmethod
    def _resolve_cols(
        df: pd.DataFrame,
        column_maps: list[dict[str, str]],
    ) -> dict[str, str]:
        """Resolve which column mapping works for this DataFrame.

        Returns the first mapping where all keys exist in the DataFrame columns.
        """
        cols_lower = set(c.lower() for c in df.columns)
        for mapping in column_maps:
            if all(k in cols_lower for k in mapping):
                return mapping
        return {}


# ═══════════════════════════════════════════════════════════
#  Source 2: OddsPortal.com (scraping)
# ═══════════════════════════════════════════════════════════


class OddsPortalClosingOddsCollector(BaseClosingOddsCollector):
    """Scrape closing odds from OddsPortal.com.

    OddsPortal provides historical odds for most European leagues
    including 1X2, BTTS, and Over/Under markets with consensus prices.
    """

    source_name = "oddsportal"

    BASE_URL = "https://www.oddsportal.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        sess = requests.Session()
        retries = Retry(
            total=3, backoff_factor=2.0,
            status_forcelist=[429, 502, 503, 504],
        )
        sess.mount("https://", HTTPAdapter(max_retries=retries))
        sess.headers.update(self.HEADERS)
        return sess

    def collect(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        leagues: list[str] | None = None,
    ) -> list[ClosingOddsRecord]:
        """Scrape OddsPortal for historical closing odds.

        Note: OddsPortal requires a subscription for full historical
        data access. This implementation provides the scraping framework
        and handles the free-tier access pattern.
        """
        logger.warning(
            "OddsPortal collector: historical odds access may require "
            "a premium subscription for leagues beyond the most recent results."
        )

        # Build date range
        today = date.today()
        start = start_date or date(today.year - 1, 1, 1)
        end = end_date or today

        records: list[ClosingOddsRecord] = []
        for league in leagues or ["england/premier-league"]:
            league_records = self._scrape_league(
                league, start, end,
            )
            records.extend(league_records)

        logger.info(
            "Collected %d closing odds records from OddsPortal",
            len(records),
        )
        return records

    def _scrape_league(
        self,
        league_slug: str,
        start_date: date,
        end_date: date,
    ) -> list[ClosingOddsRecord]:
        """Scrape historical results page for a league."""
        records: list[ClosingOddsRecord] = []
        # OddsPortal URL pattern: /soccer/{league}/results/
        url = f"{self.BASE_URL}/soccer/{league_slug}/results/"

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "Failed to fetch OddsPortal page for %s: %s",
                league_slug, exc,
            )
            return records

        html = resp.text

        # Parse match rows from HTML table
        # OddsPortal tables have a consistent structure with
        # date, teams, and odds columns
        matches = self._parse_html(html, league_slug)
        for m in matches:
            if m.match_date < start_date or m.match_date > end_date:
                continue
            records.append(m)

        # Follow pagination until we hit the start_date boundary
        next_page = self._find_next_page(html)
        while next_page:
            try:
                next_resp = self._session.get(
                    f"{self.BASE_URL}{next_page}", timeout=self.timeout,
                )
                next_resp.raise_for_status()
                next_matches = self._parse_html(
                    next_resp.text, league_slug,
                )
                # Stop if we've gone past the start_date
                if next_matches:
                    oldest = min(m.match_date for m in next_matches)
                    if oldest < start_date:
                        # Add matches within range and stop
                        for m in next_matches:
                            if start_date <= m.match_date <= end_date:
                                records.append(m)
                        break
                for m in next_matches:
                    if m.match_date > end_date:
                        continue
                    records.append(m)
                next_page = self._find_next_page(next_resp.text)
            except requests.RequestException:
                break

        return records

    def _parse_html(
        self,
        html: str,
        league_slug: str,
    ) -> list[ClosingOddsRecord]:
        """Parse match rows from an OddsPortal HTML page.

        Uses regex-based extraction which is resilient to minor HTML changes.
        """
        records: list[ClosingOddsRecord] = []

        # Find table rows containing match data
        # Pattern: date, home team, score, away team, odds (1, X, 2)
        row_pattern = re.compile(
            r'<tr[^>]*class="[^"]*deactivate[^"]*"[^>]*>'
            r'.*?<td[^>]*class="[^"]*date[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*name[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*score[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*name[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*odds[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*odds[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*odds[^"]*"[^>]*>(.*?)</td>',
            re.DOTALL,
        )

        for match in row_pattern.finditer(html):
            date_text = match.group(1).strip()
            home_text = match.group(2).strip()
            score_text = match.group(3).strip()
            away_text = match.group(4).strip()
            odds_1 = match.group(5).strip()
            odds_x = match.group(6).strip()
            odds_2 = match.group(7).strip()

            # Parse date
            try:
                match_date = datetime.strptime(date_text, "%d %b %Y").date()
            except ValueError:
                try:
                    match_date = datetime.strptime(date_text, "%Y-%m-%d").date()
                except ValueError:
                    continue

            # Parse odds (could be "-" meaning no odds available)
            try:
                oh = float(odds_1) if odds_1 not in ("-", "") else None
            except ValueError:
                oh = None
            try:
                od = float(odds_x) if odds_x not in ("-", "") else None
            except ValueError:
                od = None
            try:
                oa = float(odds_2) if odds_2 not in ("-", "") else None
            except ValueError:
                oa = None

            records.append(ClosingOddsRecord(
                match_date=match_date,
                home_team=home_text,
                away_team=away_text,
                league=league_slug,
                odds_home=oh,
                odds_draw=od,
                odds_away=oa,
                source=self.source_name,
                odds_timestamp=datetime.combine(
                    match_date, datetime.min.time(), tzinfo=timezone.utc,
                ),
            ))

        return records

    def _find_next_page(self, html: str) -> str | None:
        """Find the URL for the next page of results."""
        match = re.search(
            r'<a[^>]*class="[^"]*next[^"]*"[^>]*href="([^"]+)"',
            html,
        )
        if match:
            return match.group(1)
        return None


# ═══════════════════════════════════════════════════════════
#  Source 3: BetExplorer.com (scraping)
# ═══════════════════════════════════════════════════════════


class BetExplorerClosingOddsCollector(BaseClosingOddsCollector):
    """Scrape closing odds from BetExplorer.com.

    BetExplorer provides detailed historical odds including 1X2, BTTS,
    Over/Under, and Double Chance markets for a wide range of leagues.
    """

    source_name = "betexplorer"

    BASE_URL = "https://www.betexplorer.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        sess = requests.Session()
        retries = Retry(
            total=3, backoff_factor=2.0,
            status_forcelist=[429, 502, 503, 504],
        )
        sess.mount("https://", HTTPAdapter(max_retries=retries))
        sess.headers.update(self.HEADERS)
        return sess

    def collect(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        leagues: list[str] | None = None,
    ) -> list[ClosingOddsRecord]:
        """Scrape BetExplorer for historical closing odds.

        Fetches results pages for each league and extracts 1X2 odds
        from the odds table.
        """
        today = date.today()
        start = start_date or date(today.year - 2, 1, 1)
        end = end_date or today

        records: list[ClosingOddsRecord] = []
        for league in leagues or ["soccer/england/premier-league"]:
            league_records = self._scrape_league(league, start, end)
            records.extend(league_records)

        logger.info(
            "Collected %d closing odds records from BetExplorer",
            len(records),
        )
        return records

    def _scrape_league(
        self,
        league_slug: str,
        start_date: date,
        end_date: date,
    ) -> list[ClosingOddsRecord]:
        """Scrape results page for a league and extract closing odds.

        URL pattern: /{league_slug}/{season}/results/
        """
        from src.data_collection.sources.football_data_co_uk import (
            _guess_current_season,
        )

        season = _guess_current_season()
        url = f"{self.BASE_URL}/{league_slug}/{season}/results/"

        records: list[ClosingOddsRecord] = []

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "Failed to fetch BetExplorer page for %s: %s",
                league_slug, exc,
            )
            return records

        html = resp.text

        # Parse the results table
        results = self._parse_results_table(html, league_slug)
        for r in results:
            if r.match_date < start_date or r.match_date > end_date:
                continue
            records.append(r)

        # Try previous seasons
        for offset in range(1, 3):
            prev_season = self._prev_season(season, offset)
            prev_url = f"{self.BASE_URL}/{league_slug}/{prev_season}/results/"
            try:
                prev_resp = self._session.get(prev_url, timeout=self.timeout)
                prev_resp.raise_for_status()
                prev_results = self._parse_results_table(
                    prev_resp.text, league_slug,
                )
                for r in prev_results:
                    if r.match_date < start_date:
                        continue
                    if r.match_date > end_date:
                        records.append(r)
            except requests.RequestException:
                pass

        return records

    def _parse_results_table(
        self,
        html: str,
        league_slug: str,
    ) -> list[ClosingOddsRecord]:
        """Parse the results table from BetExplorer HTML.

        Uses regex to extract match rows with dates, teams, and odds.
        BetExplorer tables use a consistent structure:
        ``<tr>`` with ``<td>`` for date, home, score, away, odds(1,X,2).
        """
        records: list[ClosingOddsRecord] = []

        # Pattern for standard BetExplorer results table rows
        row_pattern = re.compile(
            r'<tr[^>]*>'
            r'.*?<td[^>]*class="[^"]*date[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*team[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*class="[^"]*score[^"]*"[^>]*>\s*(\d+)\s*:\s*(\d+)\s*</td>'
            r'.*?<td[^>]*class="[^"]*team[^"]*"[^>]*>(.*?)</td>'
            r'.*?<td[^>]*>\s*([\d.]+)\s*</td>'
            r'.*?<td[^>]*>\s*([\d.]+)\s*</td>'
            r'.*?<td[^>]*>\s*([\d.]+)\s*</td>',
            re.DOTALL,
        )

        for match in row_pattern.finditer(html):
            date_text = match.group(1).strip()
            home_text = match.group(2).strip()
            score_h = match.group(3).strip()
            score_a = match.group(4).strip()
            away_text = match.group(5).strip()
            odds_1 = match.group(6).strip()
            odds_x = match.group(7).strip()
            odds_2 = match.group(8).strip()

            # Parse date
            try:
                match_date = datetime.strptime(date_text, "%d.%m.%Y").date()
            except ValueError:
                try:
                    match_date = datetime.strptime(date_text, "%Y-%m-%d").date()
                except ValueError:
                    continue

            # Parse odds
            try:
                oh = float(odds_1)
            except ValueError:
                oh = None
            try:
                od = float(odds_x)
            except ValueError:
                od = None
            try:
                oa = float(odds_2)
            except ValueError:
                oa = None

            records.append(ClosingOddsRecord(
                match_date=match_date,
                home_team=home_text,
                away_team=away_text,
                league=league_slug,
                odds_home=oh,
                odds_draw=od,
                odds_away=oa,
                source=self.source_name,
                odds_timestamp=datetime.combine(
                    match_date, datetime.min.time(), tzinfo=timezone.utc,
                ),
            ))

        return records

    @staticmethod
    def _prev_season(current_season: str, offset: int = 1) -> str:
        """Compute previous season code (e.g. '2425' → '2324')."""
        start = int(current_season[:2]) - offset
        end = int(current_season[2:]) - offset
        return f"{start:02d}{end:02d}"


# ═══════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════


class ClosingOddsOrchestrator:
    """Orchestrate closing odds collection from all sources.

    Usage
    -----
    ::

        collector = ClosingOddsOrchestrator()
        total = collector.collect_all()
        print(f"{total} closing odds records stored")
    """

    def __init__(self) -> None:
        self.collectors: dict[str, BaseClosingOddsCollector] = {
            "football_data": FootballDataClosingOddsCollector(),
            "oddsportal": OddsPortalClosingOddsCollector(),
            "betexplorer": BetExplorerClosingOddsCollector(),
        }

    def collect_all(
        self,
        sources: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        leagues: list[str] | None = None,
    ) -> dict[str, int]:
        """Run selected or all closing odds collectors.

        Parameters
        ----------
        sources : list[str], optional
            List of source names to run. Default: all sources.
        start_date, end_date : date, optional
            Date range filter.
        leagues : list[str], optional
            League codes / slugs to collect. If None, collects all available.

        Returns
        -------
        dict[str, int]
            Mapping from source name to number of records stored.
        """
        if sources is None:
            sources = list(self.collectors.keys())

        results: dict[str, int] = {}

        from src.database.session import get_session as _get_session

        with _get_session() as session:
            for source_name in sources:
                collector = self.collectors.get(source_name)
                if collector is None:
                    logger.warning("Unknown source: %s", source_name)
                    continue

                logger.info(
                    "Running collector: %s (start=%s, end=%s, leagues=%s)",
                    source_name, start_date, end_date, leagues,
                )
                count = collector.run(
                    session=session,
                    start_date=start_date,
                    end_date=end_date,
                    leagues=leagues,
                )
                results[source_name] = count
                logger.info("%s: stored %d records", source_name, count)

        return results

    def summary(self, results: dict[str, int]) -> str:
        """Return a formatted summary string from collector results."""
        total = sum(results.values())
        parts = [f"  {k}: {v} records" for k, v in sorted(results.items())]
        return (
            f"Closing Odds Collection Summary:\n"
            f"{chr(10).join(parts)}\n"
            f"  TOTAL: {total} records"
        )
