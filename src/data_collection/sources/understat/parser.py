"""
UnderstatParser — converts raw Understat JSON into structured data models.

Handles the nested JSON structure from Understat's JavaScript variables:
- teamsData: { team_id: { "id": ..., "title": ..., "history": [...] } }
- shotsData: { match_id: { "h": {...}, "a": {...} } }
- datesData: array of match fixtures with results
"""

from __future__ import annotations

import logging
from typing import Any

from src.data_collection.sources.understat.models import (
    LEAGUE_NAMES,
    MatchXG,
    ShotData,
    TeamXG,
    validate_match_xg,
    validate_xg_shot,
)

logger = logging.getLogger(__name__)


class UnderstatParser:
    """Parses Understat JSON into structured data models.

    Parameters
    ----------
    validate : bool
        Whether to validate parsed data (default True).
    strict : bool
        Whether to raise on validation errors (default False).
        If False, invalid records are logged as warnings and skipped.
    """

    def __init__(self, validate: bool = True, strict: bool = False) -> None:
        self.validate = validate
        self.strict = strict

    # ══════════════════════════════════════════════════════
    #  League data → TeamXG records
    # ══════════════════════════════════════════════════════

    def parse_league_teams(
        self,
        teams_data: dict[str, Any],
        league_code: str,
        year: int,
    ) -> list[TeamXG]:
        """Parse teamsData from a league page into TeamXG records.

        Understat teamsData structure::
            {
                "team_id_1": {
                    "id": 123,
                    "title": "Manchester United",
                    "history": [
                        {"season": "2024", "xg": ..., "scored": ..., ...},
                        ...
                    ]
                },
                ...
            }

        Parameters
        ----------
        teams_data : dict
            Raw teamsData from Understat.
        league_code : str
            League code (e.g. ``EPL``).
        year : int
            Season starting year.

        Returns
        -------
        list[TeamXG]
            Parsed team xG records.
        """
        teams: list[TeamXG] = []
        season_str = str(year)

        for team_id, team_info in teams_data.items():
            if not isinstance(team_info, dict):
                continue

            title = team_info.get("title", "")
            history = team_info.get("history", [])

            # Find the entry for our season
            season_data = None
            for entry in history:
                if str(entry.get("season", "")) == season_str:
                    season_data = entry
                    break

            if season_data is None and history:
                # Use the most recent season
                season_data = history[-1]
                season_str = str(season_data.get("season", season_str))

            if season_data is None:
                continue

            team = TeamXG(
                team_name=title,
                season=season_str,
                matches_played=int(season_data.get("played", 0)),
                xg=float(season_data.get("xG", 0)),
                xga=float(season_data.get("xGA", 0)),
                scored=int(season_data.get("scored", 0)),
                conceded=int(season_data.get("missed", 0)),
                wins=int(season_data.get("wins", 0)),
                draws=int(season_data.get("draws", 0)),
                losses=int(season_data.get("loses", 0)),
                pts=int(season_data.get("pts", 0)),
                npxg=float(season_data.get("npxG", 0)),
                npxga=float(season_data.get("npxGA", 0)),
            )

            # Compute per-match averages
            if team.matches_played > 0:
                team.xg_per_match = round(team.xg / team.matches_played, 2)
                team.xga_per_match = round(team.xga / team.matches_played, 2)

            if self.validate and team.matches_played > 0:
                # Basic sanity checks
                if team.xg < 0:
                    logger.warning("Negative xG for %s: %.2f", title, team.xg)

            teams.append(team)

        logger.info(
            "Parsed %d teams from %s %s", len(teams), league_code, year,
        )
        return teams

    # ══════════════════════════════════════════════════════
    #  Match shots → ShotData records
    # ══════════════════════════════════════════════════════

    def parse_match_shots(
        self,
        shots_data: dict[str, Any],
        match_id: int,
    ) -> list[ShotData]:
        """Parse shotsData from a match page into ShotData records.

        Understat shotsData structure::
            {
                "h": [  # home team shots
                    {"id": ..., "minute": ..., "result": ..., "X": ..., "Y": ...,
                     "xG": ..., "player": ..., "situation": ..., ...}
                ],
                "a": [  # away team shots
                    ...
                ]
            }

        Parameters
        ----------
        shots_data : dict
            Raw shotsData from Understat.
        match_id : int
            Understat match identifier.

        Returns
        -------
        list[ShotData]
            Parsed shot records.
        """
        shots: list[ShotData] = []

        # Extract home/away team info from match context
        home_team = shots_data.get("h_team", "")
        away_team = shots_data.get("a_team", "")
        date = shots_data.get("date", "")

        for side, team_name in [("h", home_team), ("a", away_team)]:
            side_shots = shots_data.get(side, [])
            if not isinstance(side_shots, list):
                continue

            for raw in side_shots:
                if not isinstance(raw, dict):
                    continue

                shot = self._parse_single_shot(
                    raw, match_id, team_name, side,
                )
                if shot is not None:
                    shot.date = date
                    shot.home_team = home_team
                    shot.away_team = away_team
                    shots.append(shot)

        logger.info(
            "Parsed %d shots for match %d", len(shots), match_id,
        )
        return shots

    def _parse_single_shot(
        self,
        raw: dict[str, Any],
        match_id: int,
        team_name: str,
        side: str,
    ) -> ShotData | None:
        """Parse a single shot dict into ShotData, with validation."""
        try:
            shot = ShotData(
                match_id=match_id,
                shooter=str(raw.get("player", "")),
                team=team_name,
                minute=self._safe_int(raw.get("minute", 0)),
                x=self._safe_float(raw.get("X", 0)),
                y=self._safe_float(raw.get("Y", 0)),
                xg=self._safe_float(raw.get("xG", 0)),
                result=str(raw.get("result", "")).upper(),
                situation=str(raw.get("situation", "")),
                last_action=str(raw.get("lastAction", "") or None),
            )

            if self.validate:
                issues = validate_xg_shot(shot)
                if issues:
                    msg = f"Shot validation issues: {issues} (match {match_id})"
                    if self.strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                    return None

            return shot

        except Exception as exc:
            logger.debug("Failed to parse shot: %s", exc)
            return None

    # ══════════════════════════════════════════════════════
    #  Match data → MatchXG records
    # ══════════════════════════════════════════════════════

    def parse_league_matches(
        self,
        dates_data: dict[str, Any],
        teams_data: dict[str, Any],
        league_code: str,
        year: int,
    ) -> list[MatchXG]:
        """Parse datesData from a league page into MatchXG records.

        Understat datesData maps team IDs to their match history::
            {
                "team_id_1": [
                    {"id": ..., "isResult": true, "goals": {...}, "xG": {...},
                     "h_team": "...", "a_team": "..."},
                    ...
                ],
                ...
            }

        We deduplicate by match_id since each match appears in both
        home and away teams' date lists.

        Parameters
        ----------
        dates_data : dict
            Raw datesData from Understat.
        teams_data : dict
            Raw teamsData (for team name lookups).
        league_code : str
            League code.
        year : int
            Season starting year.

        Returns
        -------
        list[MatchXG]
            Parsed match xG records.
        """
        # Build team_id → team_name lookup
        team_names: dict[str, str] = {}
        for tid, info in teams_data.items():
            if isinstance(info, dict):
                team_names[str(tid)] = info.get("title", str(tid))

        league_name = LEAGUE_NAMES.get(league_code, league_code)
        season_str = str(year)
        seen_match_ids: set[int] = set()
        matches: list[MatchXG] = []

        for team_id, match_list in dates_data.items():
            if not isinstance(match_list, list):
                continue

            for raw in match_list:
                if not isinstance(raw, dict):
                    continue

                match_id = int(raw.get("id", 0))
                if match_id in seen_match_ids:
                    continue
                seen_match_ids.add(match_id)

                is_result = raw.get("isResult", False)

                # Extract teams
                h_team = raw.get("h_team", {})
                a_team = raw.get("a_team", {})

                home_team_id = str(h_team.get("id", "")) if isinstance(h_team, dict) else ""
                away_team_id = str(a_team.get("id", "")) if isinstance(a_team, dict) else ""

                home_team = team_names.get(home_team_id, str(h_team.get("title", "")))
                away_team = team_names.get(away_team_id, str(a_team.get("title", "")))

                # Extract goals
                goals = raw.get("goals", {})
                home_goals = self._safe_int(goals.get("h", 0)) if isinstance(goals, dict) else 0
                away_goals = self._safe_int(goals.get("a", 0)) if isinstance(goals, dict) else 0

                # Extract xG
                xg_data = raw.get("xG", {})
                home_xg = self._safe_float(xg_data.get("h", 0)) if isinstance(xg_data, dict) else 0
                away_xg = self._safe_float(xg_data.get("a", 0)) if isinstance(xg_data, dict) else 0

                # Extract shots
                home_shots = raw.get("h_shots", raw.get("shots_h", 0))
                away_shots = raw.get("a_shots", raw.get("shots_a", 0))
                home_sot = raw.get("h_sot", raw.get("shots_on_target_h", 0))
                away_sot = raw.get("a_sot", raw.get("shots_on_target_a", 0))

                match = MatchXG(
                    match_id=match_id,
                    league=league_name,
                    season=season_str,
                    date=raw.get("date", ""),
                    home_team=home_team,
                    away_team=away_team,
                    home_xg=home_xg,
                    away_xg=away_xg,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_shots=self._safe_int(home_shots),
                    away_shots=self._safe_int(away_shots),
                    home_shots_on_target=self._safe_int(home_sot),
                    away_shots_on_target=self._safe_int(away_sot),
                    is_result=is_result,
                )

                if self.validate:
                    issues = validate_match_xg(match)
                    if issues and self.strict:
                        raise ValueError(
                            f"Match {match_id} validation: {issues}"
                        )
                    elif issues:
                        logger.debug(
                            "Match %d issues: %s", match_id, issues,
                        )

                matches.append(match)

        logger.info(
            "Parsed %d matches from %s %s", len(matches), league_code, year,
        )
        return matches

    # ══════════════════════════════════════════════════════
    #  Convenience: parse from raw HTML
    # ══════════════════════════════════════════════════════

    def parse_league_from_html(
        self,
        html: str,
        league_code: str,
        year: int,
    ) -> tuple[list[TeamXG], list[MatchXG]]:
        """Parse both teams and matches from a league page HTML.

        Parameters
        ----------
        html : str
            Raw HTML of the league page.
        league_code : str
            League code.
        year : int
            Season starting year.

        Returns
        -------
        tuple[list[TeamXG], list[MatchXG]]
            Parsed teams and matches.
        """
        from src.data_collection.sources.understat.client import UnderstatClient

        teams_data = UnderstatClient._extract_json(html, "teamsData")
        dates_data = UnderstatClient._extract_json(html, "datesData")

        teams = self.parse_league_teams(teams_data, league_code, year)
        matches = self.parse_league_matches(
            dates_data, teams_data, league_code, year,
        )
        return teams, matches

    # ── Helpers ─────────────────────────────────────────

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
