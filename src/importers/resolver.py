"""
EntityResolver — maps CSV row data to database foreign-key IDs.

Resolves names/identifiers found in CSV rows to their corresponding
primary keys in the database::

    "Manchester Utd"  ─→  team_id=42
    "E0"              ─→  competition_id=1
    "2024/2025"       ─→  season_id=7

Resolution is backed by the TeamNormalizer for team names and by
direct database queries for competitions and seasons.  Results are
cached to minimise round-trips during large imports.

Two modes
---------
- **Lookup-only** — raises if entity not found (default, strict).
- **Auto-create** — creates missing teams/competitions/seasons on
  the fly (optional, for fully automated imports).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.competition import Competition
from src.database.models.season import Season
from src.database.models.team import Team
from src.database.session import get_session
from src.importers.parser import _four_digit_to_season_name
from src.team_normalizer import TeamNormalizer

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolves CSV identifiers to database primary keys.

    Parameters
    ----------
    normalizer : TeamNormalizer, optional
        Team name normalizer.  Defaults to a new instance with
        2000+ built-in aliases.
    auto_create_teams : bool
        If True, create unknown teams on the fly (default False).
    auto_create_competitions : bool
        If True, create unknown competitions on the fly (default False).
    auto_create_seasons : bool
        If True, create unknown seasons on the fly (default False).
    log_resolutions : bool
        Log each resolution at INFO level (default True).
    """

    def __init__(
        self,
        normalizer: TeamNormalizer | None = None,
        auto_create_teams: bool = False,
        auto_create_competitions: bool = False,
        auto_create_seasons: bool = False,
        log_resolutions: bool = True,
    ) -> None:
        self.normalizer = normalizer or TeamNormalizer(log_low_confidence=True)
        self.auto_create_teams = auto_create_teams
        self.auto_create_competitions = auto_create_competitions
        self.auto_create_seasons = auto_create_seasons
        self.log_resolutions = log_resolutions

        # Caches: { lookup_key → db_id }
        self._team_cache: dict[str, int] = {}
        self._competition_cache: dict[str, int] = {}
        self._season_cache: dict[str, int] = {}

    # ── Team resolution ────────────────────────────────

    def resolve_team(self, name: str) -> int | None:
        """Resolve a team name to its database ID.

        Uses TeamNormalizer for fuzzy matching, then queries the
        database by canonical name.  Results are cached.

        Parameters
        ----------
        name : str
            Team name or alias.

        Returns
        -------
        int or None
            Team database ID, or None if not found.
        """
        if not name or not name.strip():
            return None

        key = name.strip().lower()

        # Check cache first
        if key in self._team_cache:
            return self._team_cache[key]

        # Normalize the name
        result = self.normalizer.resolve(name)
        canonical = result.canonical

        if self.log_resolutions and result.confidence < 1.0:
            logger.info(
                "Team resolved: %r -> %s (conf=%.2f, method=%s)",
                name, canonical, result.confidence, result.method,
            )

        # Query database
        team_id = self._query_team(canonical)

        if team_id is not None:
            self._team_cache[key] = team_id
            self._team_cache[canonical.lower()] = team_id
        elif self.auto_create_teams:
            team_id = self._create_team(canonical)
            if team_id is not None:
                self._team_cache[key] = team_id
                self._team_cache[canonical.lower()] = team_id

        if team_id is None:
            logger.warning("Team not found in database: %r (canonical=%r)", name, canonical)

        return team_id

    def resolve_teams_batch(
        self, names: list[str],
    ) -> dict[str, int | None]:
        """Resolve multiple team names at once.

        Parameters
        ----------
        names : list[str]
            Team names to resolve.

        Returns
        -------
        dict[str, int | None]
            Mapping of input name → database ID (or None).
        """
        return {name: self.resolve_team(name) for name in names}

    # ── Competition resolution ─────────────────────────

    def resolve_competition(
        self,
        code: str | None,
        name: str | None = None,
    ) -> int | None:
        """Resolve a league code/name to a Competition database ID.

        Parameters
        ----------
        code : str or None
            League code (e.g. ``\"E0\"`` for Premier League).
        name : str or None
            Competition name fallback (e.g. ``\"Premier League\"``).

        Returns
        -------
        int or None
            Competition ID, or None if not found.
        """
        # Try code first
        if code and code.strip():
            key = f"code:{code.strip().upper()}"
            if key in self._competition_cache:
                return self._competition_cache[key]

            comp_id = self._query_competition_by_code(code.strip().upper())
            if comp_id is not None:
                self._competition_cache[key] = comp_id
                return comp_id

        # Try name
        if name and name.strip():
            key = f"name:{name.strip().lower()}"
            if key in self._competition_cache:
                return self._competition_cache[key]

            comp_id = self._query_competition_by_name(name.strip())
            if comp_id is not None:
                self._competition_cache[key] = comp_id
                return comp_id

        # Auto-create
        if self.auto_create_competitions and (code or name):
            display = name or code or "Unknown"
            comp_id = self._create_competition(
                code=code.strip().upper() if code else None,
                name=display,
            )
            if comp_id is not None:
                if code:
                    self._competition_cache[f"code:{code.strip().upper()}"] = comp_id
                if name:
                    self._competition_cache[f"name:{name.strip().lower()}"] = comp_id
                return comp_id

        logger.warning("Competition not found: code=%r, name=%r", code, name)
        return None

    # ── Season resolution ──────────────────────────────

    def resolve_season(
        self,
        season_name: str | None,
        competition_id: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int | None:
        """Resolve a season name (e.g. ``\"2024/2025\"``) to a Season database ID.

        Parameters
        ----------
        season_name : str or None
            Season display name (e.g. ``\"2024/2025\"``).
        competition_id : int or None
            Competition ID filter (required for uniqueness).
        start_date : date, optional
            Season start date (for auto-creation).
        end_date : date, optional
            Season end date (for auto-creation).

        Returns
        -------
        int or None
            Season ID, or None if not found.
        """
        if not season_name or not season_name.strip():
            return None

        key = season_name.strip().lower()
        if competition_id is not None:
            key = f"{competition_id}:{key}"

        # Check cache
        if key in self._season_cache:
            return self._season_cache[key]

        # Query database
        season_id = self._query_season(season_name.strip(), competition_id)

        if season_id is not None:
            self._season_cache[key] = season_id
        elif self.auto_create_seasons:
            season_id = self._create_season(
                name=season_name.strip(),
                competition_id=competition_id,
                start_date=start_date,
                end_date=end_date,
            )
            if season_id is not None:
                self._season_cache[key] = season_id

        if season_id is None:
            logger.warning(
                "Season not found: %r (competition_id=%s)",
                season_name, competition_id,
            )

        return season_id

    # ── Bulk row resolver ──────────────────────────────

    def resolve_row(
        self,
        row: dict[str, Any],
        league_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Resolve all entities in a single parsed row.

        Adds the following keys to the row dict:
        - ``home_team_id``
        - ``away_team_id``
        - ``competition_id``
        - ``season_id``

        Parameters
        ----------
        row : dict
            A parsed and standardised row from ``CSVParser``.
        league_map : dict[str, str], optional
            Mapping of league codes → competition names
            (e.g. ``{\"E0\": \"Premier League\"}``).

        Returns
        -------
        dict
            Row with resolved foreign-key IDs added.
        """
        result = dict(row)

        # Resolve teams
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        result["home_team_id"] = self.resolve_team(home_team)
        result["away_team_id"] = self.resolve_team(away_team)

        # Resolve competition
        league_code = row.get("league", "")
        league_name = (
            league_map.get(league_code.upper(), league_code)
            if league_map and league_code
            else league_code
        )
        result["competition_id"] = self.resolve_competition(
            code=league_code if league_code else None,
            name=league_name if league_name else None,
        )

        # Resolve season
        season_name = row.get("season", "")
        if season_name:
            # Convert 4-digit code to readable name if needed
            if season_name.isdigit() and len(season_name) == 4:
                season_name = _four_digit_to_season_name(season_name)
        result["season_id"] = self.resolve_season(
            season_name=season_name if season_name else None,
            competition_id=result.get("competition_id"),
            start_date=row.get("match_date"),
        )

        return result

    def resolve_rows(
        self,
        rows: list[dict[str, Any]],
        league_map: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve entities for all parsed rows.

        Parameters
        ----------
        rows : list[dict]
            Parsed rows from ``CSVParser``.
        league_map : dict[str, str], optional
            League code → name map.

        Returns
        -------
        list[dict]
            Rows with resolved FK IDs added.
        """
        resolved = []
        for row in rows:
            resolved.append(self.resolve_row(row, league_map))
        return resolved

    # ── Cache management ───────────────────────────────

    def clear_cache(self) -> None:
        """Clear all internal caches (useful between imports)."""
        self._team_cache.clear()
        self._competition_cache.clear()
        self._season_cache.clear()

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache sizes for diagnostics."""
        return {
            "teams": len(self._team_cache),
            "competitions": len(self._competition_cache),
            "seasons": len(self._season_cache),
        }

    # ── Database queries ───────────────────────────────

    @staticmethod
    def _query_team(canonical: str) -> int | None:
        """Look up a team by exact canonical name."""
        with get_session() as session:
            stmt = select(Team.id).where(Team.name == canonical)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    @staticmethod
    def _query_competition_by_code(code: str) -> int | None:
        """Look up a competition by its short code (e.g. 'E0')."""
        with get_session() as session:
            stmt = select(Competition.id).where(Competition.code == code)
            return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def _query_competition_by_name(name: str) -> int | None:
        """Look up a competition by its full name."""
        with get_session() as session:
            stmt = select(Competition.id).where(Competition.name == name)
            return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def _query_season(name: str, competition_id: int | None) -> int | None:
        """Look up a season by name and optional competition ID."""
        with get_session() as session:
            stmt = select(Season.id).where(Season.name == name)
            if competition_id is not None:
                stmt = stmt.where(Season.competition_id == competition_id)
            return session.execute(stmt).scalar_one_or_none()

    # ── Auto-creation ──────────────────────────────────

    def _create_team(self, name: str) -> int | None:
        """Create a new team record in the database."""
        try:
            with get_session() as session:
                team = Team(name=name)
                session.add(team)
                session.flush()
                team_id = team.id
                logger.info("Auto-created team: %s (id=%d)", name, team_id)
                return team_id
        except Exception as exc:
            logger.error("Failed to auto-create team %r: %s", name, exc)
            return None

    def _create_competition(
        self,
        code: str | None,
        name: str,
        type: str = "league",
    ) -> int | None:
        """Create a new competition record in the database."""
        try:
            with get_session() as session:
                comp = Competition(
                    code=code,
                    name=name,
                    type=type,
                )
                session.add(comp)
                session.flush()
                comp_id = comp.id
                logger.info(
                    "Auto-created competition: %s (code=%s, id=%d)",
                    name, code, comp_id,
                )
                return comp_id
        except Exception as exc:
            logger.error(
                "Failed to auto-create competition %r (code=%s): %s",
                name, code, exc,
            )
            return None

    def _create_season(
        self,
        name: str,
        competition_id: int | None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int | None:
        """Create a new season record in the database."""
        if competition_id is None:
            logger.error("Cannot auto-create season %r without a competition_id", name)
            return None

        # Default date range from season name if not provided
        if start_date is None or end_date is None:
            try:
                parts = name.split("/")
                if len(parts) == 2:
                    start_year = int(parts[0])
                    end_year = int(parts[1])
                    start_date = date(start_year, 8, 1)
                    end_date = date(end_year, 7, 31)
                else:
                    year = int(parts[0]) if parts else 2024
                    start_date = date(year, 1, 1)
                    end_date = date(year, 12, 31)
            except (ValueError, IndexError):
                start_date = start_date or date(2024, 8, 1)
                end_date = end_date or date(2025, 7, 31)

        try:
            with get_session() as session:
                season = Season(
                    name=name,
                    competition_id=competition_id,
                    start_date=start_date,
                    end_date=end_date,
                )
                session.add(season)
                session.flush()
                season_id = season.id
                logger.info(
                    "Auto-created season: %s (competition_id=%d, id=%d)",
                    name, competition_id, season_id,
                )
                return season_id
        except Exception as exc:
            logger.error(
                "Failed to auto-create season %r: %s", name, exc,
            )
            return None
