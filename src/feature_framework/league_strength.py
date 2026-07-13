"""
League Strength Module — per-season, per-league analytics for football competitions.

Estimates offensive/defensive strength, competitive balance, home advantage, and
cross-league normalisation factors.  Tracks promoted/relegated teams and European
competition participation.

Features
--------
+---------------------------+------------------------------------------------+---------+
| Metric                    | Description                                    | Type    |
+===========================+================================================+=========+
| offensive_strength        | Avg goals scored per match (league-wide)       | float   |
| defensive_strength        | Avg goals conceded per match (league-wide)     | float   |
| avg_goals                | Total goals / total matches                    | float   |
| avg_xg                   | Avg xG per match (when data available)         | float   |
| competitive_balance      | Std of goal difference across all matches      | float   |
| home_adv                 | Avg home goals - avg away goals (per match)    | float   |
| home_win_rate            | Proportion of home wins in the league          | float   |
| draw_rate                | Proportion of draws in the league              | float   |
| away_win_rate            | Proportion of away wins in the league          | float   |
| btts_rate                | Proportion of matches where both teams scored  | float   |
| over_2_5_rate            | Proportion of matches with > 2.5 total goals   | float   |
| attack_factor            | Normalised attack strength (ref league = 1.0)  | float   |
| defence_factor           | Normalised defence strength (ref league = 1.0) | float   |
+---------------------------+------------------------------------------------+---------+

Cross-league normalisation
--------------------------
All metrics can be normalised so that a reference league (default: Premier League)
has an attack/defence factor of 1.0.  Other leagues are expressed relative to this
baseline, allowing meaningful comparison across competitions.

Example
-------
::

    from src.feature_framework.league_strength import LeagueStrengthEngine

    engine = LeagueStrengthEngine()
    result = engine.compute(df, season_col=\"season\", league_col=\"league\")
    print(result.summary())

    # Historical storage
    engine.store_season(2024, \"PL\", result)
    stored = engine.get_season(2024, \"PL\")

    # Normalised comparison
    normalised = engine.normalise_across_leagues(
        seasons=[2024], leagues=[\"PL\", \"L1\", \"SA\", \"BL\"],
        reference_league=\"PL\"
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Metric keys ─────────────────────────────────────────
OFFENSIVE_STRENGTH = "offensive_strength"
DEFENSIVE_STRENGTH = "defensive_strength"
AVG_GOALS = "avg_goals"
AVG_XG = "avg_xg"
COMPETITIVE_BALANCE = "competitive_balance"
HOME_ADV = "home_adv"
HOME_WIN_RATE = "home_win_rate"
DRAW_RATE = "draw_rate"
AWAY_WIN_RATE = "away_win_rate"
BTTS_RATE = "btts_rate"
OVER_2_5_RATE = "over_2_5_rate"
ATTACK_FACTOR = "attack_factor"
DEFENCE_FACTOR = "defence_factor"
TOTAL_MATCHES = "total_matches"
AVG_HOME_GOALS = "avg_home_goals"
AVG_AWAY_GOALS = "avg_away_goals"
STD_GOAL_DIFF = "std_goal_diff"

# Default reference league code
_DEFAULT_REFERENCE = "E0"  # Premier League code on football-data.co.uk


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════


@dataclass
class LeagueStrengthRecord:
    """Per-season, per-league strength metrics.

    Parameters
    ----------
    season : str
        Season identifier (e.g. ``\"2024\"``, ``\"2024/2025\"``).
    league : str
        League code or name (e.g. ``\"E0\"``, ``\"PL\"``, ``\"La Liga\"``).
    league_name : str, optional
        Human-readable league name.
    offensive_strength : float
        Average goals scored per match in this league this season.
    defensive_strength : float
        Average goals conceded per match.
    avg_goals : float
        Total goals / total matches.
    avg_xg : float, optional
        Average xG per match (None if no xG data).
    competitive_balance : float
        Standard deviation of goal difference across all matches.
    home_adv : float
        Average home goals minus average away goals per match.
    home_win_rate : float
        Proportion of matches ending in home wins.
    draw_rate : float
        Proportion of draws.
    away_win_rate : float
        Proportion of away wins.
    btts_rate : float
        Proportion of matches where both teams scored.
    over_2_5_rate : float
        Proportion of matches with total goals > 2.5.
    attack_factor : float
        Normalised attack strength (1.0 = reference league).
    defence_factor : float
        Normalised defence strength (1.0 = reference league).
    total_matches : int
        Number of matches in the sample.
    avg_home_goals : float
        Average home goals per match.
    avg_away_goals : float
        Average away goals per match.
    std_goal_diff : float
        Same as competitive_balance.
    n_promoted_teams : int, optional
        Teams promoted into this league for this season.
    n_relegated_teams : int, optional
        Teams relegated out of this league.
    n_european_teams : int, optional
        Teams participating in European competitions.
    metadata : dict
        Additional metadata.
    """

    season: str = ""
    league: str = ""
    league_name: str = ""

    offensive_strength: float = 0.0
    defensive_strength: float = 0.0
    avg_goals: float = 0.0
    avg_xg: float | None = None
    competitive_balance: float = 0.0
    home_adv: float = 0.0

    home_win_rate: float = 0.0
    draw_rate: float = 0.0
    away_win_rate: float = 0.0
    btts_rate: float = 0.0
    over_2_5_rate: float = 0.0

    attack_factor: float = 1.0
    defence_factor: float = 1.0

    total_matches: int = 0
    avg_home_goals: float = 0.0
    avg_away_goals: float = 0.0
    std_goal_diff: float = 0.0

    n_promoted_teams: int = 0
    n_relegated_teams: int = 0
    n_european_teams: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeagueStrengthRecord:
        """Deserialize from a dict."""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @property
    def goal_diff_std(self) -> float:
        """Alias for competitive_balance."""
        return self.competitive_balance

    def __repr__(self) -> str:
        return (
            f"<LeagueStrengthRecord {self.season} {self.league}: "
            f"GS={self.offensive_strength:.2f} GC={self.defensive_strength:.2f} "
            f"HA={self.home_adv:.2f} CB={self.competitive_balance:.2f}>"
        )


# ═══════════════════════════════════════════════════════════════
#  LeagueStrengthEngine
# ═══════════════════════════════════════════════════════════════


class LeagueStrengthEngine:
    """Compute league-level strength metrics from match data.

    Supports per-season, per-league estimates with cross-league
    normalisation, promoted/relegated team tracking, and European
    competition adjustment.

    Parameters
    ----------
    reference_league : str
        League code used as baseline (attack/defence factor = 1.0).
        Default: ``\"E0\"`` (Premier League).
    min_matches : int
        Minimum matches required per season-league before metrics are
        considered reliable.  Leagues with fewer matches return None.
    auto_normalise : bool
        Automatically compute attack/defence factors against the reference
        league when ``compute()`` is called.
    store_history : bool
        Keep computed records in memory (accessible via ``get_season()``).
    """

    def __init__(
        self,
        reference_league: str = _DEFAULT_REFERENCE,
        min_matches: int = 10,
        auto_normalise: bool = True,
        store_history: bool = True,
    ) -> None:
        self.reference_league = reference_league
        self.min_matches = min_matches
        self.auto_normalise = auto_normalise
        self.store_history = store_history

        # In-memory history store: {(season, league): LeagueStrengthRecord}
        self._history: dict[tuple[str, str], LeagueStrengthRecord] = {}

        # Track promoted/relegated teams per season per league
        self._promotions: dict[tuple[str, str], set[str]] = {}
        self._relegations: dict[tuple[str, str], set[str]] = {}

        # European competition participation per season per league
        self._european: dict[tuple[str, str], set[str]] = {}

    # ══════════════════════════════════════════════════════
    #  Core computation
    # ══════════════════════════════════════════════════════

    def compute(
        self,
        df: pd.DataFrame,
        season_col: str = "season",
        league_col: str = "league",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        result_col: str = "result",
        home_xg_col: str | None = "home_xg",
        away_xg_col: str | None = "away_xg",
    ) -> dict[str, LeagueStrengthRecord]:
        """Compute league strength metrics for all (season, league) pairs.

        Parameters
        ----------
        df : pd.DataFrame
            Match data. Must contain: ``season``, ``league``,
            ``home_goals``, ``away_goals``, ``result``.
        season_col : str
            Column name for season.
        league_col : str
            Column name for league/competition.
        home_goals_col : str
            Column name for home goals.
        away_goals_col : str
            Column name for away goals.
        result_col : str
            Column name for match result (H/D/A).
        home_xg_col : str, optional
            Column name for home xG.
        away_xg_col : str, optional
            Column name for away xG.

        Returns
        -------
        dict[str, LeagueStrengthRecord]
            Mapping from ``\"{season}/{league}\"`` to record.
        """
        results: dict[str, LeagueStrengthRecord] = {}

        required = {season_col, league_col, home_goals_col, away_goals_col, result_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        has_xg = (
            home_xg_col is not None
            and away_xg_col is not None
            and home_xg_col in df.columns
            and away_xg_col in df.columns
        )

        # Group by (season, league)
        pairs = df.groupby([season_col, league_col], sort=False)

        for (season, league), grp in pairs:
            if len(grp) < self.min_matches:
                logger.debug(
                    "Skipping %s/%s: only %d matches (min=%d)",
                    season, league, len(grp), self.min_matches,
                )
                continue

            record = self._compute_single(
                grp, season, league,
                home_goals_col, away_goals_col, result_col,
                home_xg_col, away_xg_col, has_xg,
            )
            key = f"{season}/{league}"
            results[key] = record

            if self.store_history:
                self._history[(str(season), str(league))] = record

        # Cross-league normalisation
        if self.auto_normalise and results:
            self._normalise_all(results)

        return results

    def _compute_single(
        self,
        df: pd.DataFrame,
        season: Any,
        league: Any,
        home_goals_col: str,
        away_goals_col: str,
        result_col: str,
        home_xg_col: str | None,
        away_xg_col: str | None,
        has_xg: bool,
    ) -> LeagueStrengthRecord:
        """Compute league strength for one (season, league) group."""
        hg = pd.to_numeric(df[home_goals_col], errors="coerce")
        ag = pd.to_numeric(df[away_goals_col], errors="coerce")
        hg_valid = hg.dropna()
        ag_valid = ag.dropna()

        n = len(hg_valid)
        if n < self.min_matches:
            # This shouldn't happen because the caller checks, but be safe
            return LeagueStrengthRecord(season=str(season), league=str(league))

        total_goals = hg_valid + ag_valid

        # Core metrics
        avg_home = float(hg_valid.mean())
        avg_away = float(ag_valid.mean())
        # offensive/defensive = average goals scored/conceded per team per match
        # At league aggregate level, total scored == total conceded, so these
        # values are mathematically identical. The distinction becomes useful
        # only in cross-league comparison via attack_factor/defence_factor.
        all_scored = pd.concat([hg_valid, ag_valid])
        all_conceded = pd.concat([ag_valid, hg_valid])
        offensive_overall = float(all_scored.mean())
        defensive_overall = float(all_conceded.mean())
        avg_goals = float(total_goals.mean())
        goal_diff = hg_valid - ag_valid
        cb = float(goal_diff.std())

        # Result rates
        result_upper = df[result_col].astype(str).str.upper()
        home_win = float((result_upper == "H").sum()) / n
        draw = float((result_upper == "D").sum()) / n
        away_win = float((result_upper == "A").sum()) / n

        # BTTS and over 2.5 (only on non-null goals)
        both_valid = hg.notna() & ag.notna()
        hg_both = hg[both_valid]
        ag_both = ag[both_valid]
        btts = float(((hg_both > 0) & (ag_both > 0)).sum()) / max(len(hg_both), 1)
        over_2_5 = float(((hg_both + ag_both) > 2.5).sum()) / max(len(hg_both), 1)

        # xG
        avg_xg_val: float | None = None
        if has_xg and home_xg_col and away_xg_col:
            hxg = pd.to_numeric(df[home_xg_col], errors="coerce")
            axg = pd.to_numeric(df[away_xg_col], errors="coerce")
            all_xg = pd.concat([hxg.dropna(), axg.dropna()])
            if len(all_xg) > 0:
                avg_xg_val = float(all_xg.mean())

        # Fetch stored promoted/relegated/european counts
        seas_str = str(season)
        league_str = str(league)
        n_prom = len(self._promotions.get((seas_str, league_str), set()))
        n_rel = len(self._relegations.get((seas_str, league_str), set()))
        n_euro = len(self._european.get((seas_str, league_str), set()))

        return LeagueStrengthRecord(
            season=seas_str,
            league=league_str,
            offensive_strength=offensive_overall,
            defensive_strength=defensive_overall,
            avg_goals=avg_goals,
            avg_xg=avg_xg_val,
            competitive_balance=cb,
            home_adv=avg_home - avg_away,
            home_win_rate=home_win,
            draw_rate=draw,
            away_win_rate=away_win,
            btts_rate=btts,
            over_2_5_rate=over_2_5,
            total_matches=n,
            avg_home_goals=avg_home,
            avg_away_goals=avg_away,
            std_goal_diff=cb,
            n_promoted_teams=n_prom,
            n_relegated_teams=n_rel,
            n_european_teams=n_euro,
        )

    # ══════════════════════════════════════════════════════
    #  Cross-league normalisation
    # ══════════════════════════════════════════════════════

    def _normalise_all(
        self,
        results: dict[str, LeagueStrengthRecord],
    ) -> None:
        """Compute attack/defence factors relative to the reference league.

        Reference league gets factor = 1.0.  Other leagues get factors
        expressing their relative strength: higher attack_factor means
        more goals scored (weaker defence means more conceded).
        """
        ref_record: LeagueStrengthRecord | None = None

        # Find reference league record
        for rec in results.values():
            if rec.league == self.reference_league:
                ref_record = rec
                break

        if ref_record is None or ref_record.avg_goals == 0:
            # No reference league found or zero goals — set all to 1.0
            for rec in results.values():
                rec.attack_factor = 1.0
                rec.defence_factor = 1.0
            return

        ref_attack = ref_record.offensive_strength
        ref_defence = ref_record.defensive_strength

        for rec in results.values():
            # Attack factor: how many goals this league scores vs reference
            # > 1.0 = more attacking than reference league
            rec.attack_factor = rec.offensive_strength / max(ref_attack, 0.01)

            # Defence factor: how many goals this league concedes vs reference
            # > 1.0 = concedes more = weaker defence
            rec.defence_factor = rec.defensive_strength / max(ref_defence, 0.01)

    def normalise_across_leagues(
        self,
        seasons: list[str] | None = None,
        leagues: list[str] | None = None,
        reference_league: str | None = None,
    ) -> pd.DataFrame:
        """Return a normalised comparison DataFrame across leagues.

        Parameters
        ----------
        seasons : list[str], optional
            Filter to these seasons (all if None).
        leagues : list[str], optional
            Filter to these leagues (all if None).
        reference_league : str, optional
            Override reference league (uses engine default if None).

        Returns
        -------
        pd.DataFrame
            Rows are (season, league); columns are strength metrics with
            normalised attack/defence factors.
        """
        records = list(self._history.values())

        if seasons:
            records = [r for r in records if r.season in seasons]
        if leagues:
            records = [r for r in records if r.league in leagues]

        if not records:
            return pd.DataFrame()

        ref = reference_league or self.reference_league

        # Find reference record to compute factors
        ref_records = [r for r in records if r.league == ref]
        if not ref_records:
            return pd.DataFrame([r.to_dict() for r in records])

        ref_off = float(np.mean([r.offensive_strength for r in ref_records]))
        ref_def = float(np.mean([r.defensive_strength for r in ref_records]))

        for rec in records:
            rec.attack_factor = rec.offensive_strength / max(ref_off, 0.01)
            rec.defence_factor = rec.defensive_strength / max(ref_def, 0.01)

        return pd.DataFrame([r.to_dict() for r in records])

    def summary(self) -> str:
        """Return a human-readable summary of all stored records."""
        if not self._history:
            return "No league strength data available."

        lines = [
            "LEAGUE STRENGTH REPORT",
            "=" * 70,
            f"{'Season':<10} {'League':<10} {'Off':>7} {'Def':>7} {'AvgG':>7} "
            f"{'HA':>7} {'CB':>7} {'Att':>7} {'Def':>7} {'n':>5}",
            "-" * 70,
        ]

        for key in sorted(self._history.keys(), key=lambda x: (x[0], x[1])):
            r = self._history[key]
            lines.append(
                f"{r.season:<10} {r.league:<10} "
                f"{r.offensive_strength:>7.3f} {r.defensive_strength:>7.3f} "
                f"{r.avg_goals:>7.3f} {r.home_adv:>7.3f} "
                f"{r.competitive_balance:>7.3f} {r.attack_factor:>7.3f} "
                f"{r.defence_factor:>7.3f} {r.total_matches:>5d}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════
    #  Promoted / Relegated team tracking
    # ══════════════════════════════════════════════════════

    def set_promoted(
        self,
        season: str,
        league: str,
        teams: set[str],
    ) -> None:
        """Record which teams were promoted into a league for a season.

        Parameters
        ----------
        season : str
            Season identifier.
        league : str
            League code.
        teams : set[str]
            Set of promoted team names.
        """
        self._promotions[(season, league)] = set(teams)

    def set_relegated(
        self,
        season: str,
        league: str,
        teams: set[str],
    ) -> None:
        """Record which teams were relegated from a league for a season.

        Parameters
        ----------
        season : str
            Season identifier.
        league : str
            League code.
        teams : set[str]
            Set of relegated team names.
        """
        self._relegations[(season, league)] = set(teams)

    def set_european(
        self,
        season: str,
        league: str,
        teams: set[str],
    ) -> None:
        """Record which teams participated in European competitions.

        Parameters
        ----------
        season : str
            Season identifier.
        league : str
            League code.
        teams : set[str]
            Set of team names in European competitions.
        """
        self._european[(season, league)] = set(teams)

    def get_promoted(self, season: str, league: str) -> set[str]:
        """Get promoted teams for a (season, league)."""
        return self._promotions.get((season, league), set())

    def get_relegated(self, season: str, league: str) -> set[str]:
        """Get relegated teams for a (season, league)."""
        return self._relegations.get((season, league), set())

    def get_european(self, season: str, league: str) -> set[str]:
        """Get European competition teams for a (season, league)."""
        return self._european.get((season, league), set())

    def auto_detect_promoted_relegated(
        self,
        df: pd.DataFrame,
        team_col: str = "team",
        season_col: str = "season",
        league_col: str = "league",
        league_groups: dict[str, list[str]] | None = None,
    ) -> None:
        """Auto-detect promoted/relegated teams by tracking league changes.

        Compares which teams appear in which leagues across consecutive
        seasons.  Detection is only performed within the same
        ``league_groups`` — each group represents a country's division
        hierarchy (e.g. ``{"England": ["E0", "E1", "E2"]}``).
        Teams that switch leagues within a group are flagged as promoted
        or relegated.

        .. note::

            Without ``league_groups``, the method only detects teams that
            *join* or *leave* a specific league between seasons, but cannot
            distinguish promotion/relegation from cross-country moves.
            Always pass ``league_groups`` for accurate detection.

        Parameters
        ----------
        df : pd.DataFrame
            Team-season-league mapping (one row per team per season).
            Must contain ``team``, ``season``, ``league`` columns.
        team_col : str
            Column name for team identifiers.
        season_col : str
            Column name for season identifiers.
        league_col : str
            Column name for league identifiers.
        league_groups : dict[str, list[str]], optional
            Group leagues into division hierarchies.  Each key is a
            country or system name, each value is the list of league
            codes in that system (Tier 1 first).
            Example: ``{"England": ["E0", "E1", "E2"], "Scotland": ["S1"]}``
        """
        required = {team_col, season_col, league_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Group teams by season and league
        teams_by_season_league: dict[tuple, set[str]] = {}
        for _, row in df.iterrows():
            key = (str(row[season_col]), str(row[league_col]))
            if key not in teams_by_season_league:
                teams_by_season_league[key] = set()
            teams_by_season_league[key].add(str(row[team_col]))

        # For each season, compare with previous to detect movement
        seasons = sorted(set(k[0] for k in teams_by_season_league))
        if len(seasons) < 2:
            return

        if league_groups:
            # Detect promotions/relegations within each group
            for _group_name, group_leagues in league_groups.items():
                self._detect_within_group(
                    teams_by_season_league, seasons, group_leagues,
                )
        else:
            # Without groups, just detect same-league changes (teams that
            # join or leave). This will miss cross-tier moves but avoids
            # false positives from unrelated leagues.
            for i, season in enumerate(seasons):
                if i == 0:
                    continue
                prev_season = seasons[i - 1]

                # Collect all leagues seen
                all_leagues = sorted(set(
                    k[1] for k in teams_by_season_league
                ))

                for league in all_leagues:
                    curr_teams = teams_by_season_league.get((season, league), set())
                    prev_teams = teams_by_season_league.get((prev_season, league), set())

                    new_teams = curr_teams - prev_teams
                    departed = prev_teams - curr_teams

                    if new_teams:
                        self.set_promoted(season, league, new_teams)
                    if departed:
                        self.set_relegated(prev_season, league, departed)

    def _detect_within_group(
        self,
        teams_by_season_league: dict[tuple, set[str]],
        seasons: list[str],
        group_leagues: list[str],
    ) -> None:
        """Detect promotions/relegations within a league group.

        Compares teams across tiers within the group. A team that moves
        from a higher tier to a lower tier is relegated; the reverse is
        promoted.
        """
        tiers = {league: idx for idx, league in enumerate(group_leagues)}

        for i, season in enumerate(seasons):
            if i == 0:
                continue
            prev_season = seasons[i - 1]

            # Build mapping of team -> tier for current and previous season
            curr_team_tiers: dict[str, int] = {}
            prev_team_tiers: dict[str, int] = {}

            for league in group_leagues:
                tier = tiers[league]
                for team in teams_by_season_league.get((season, league), set()):
                    curr_team_tiers[team] = tier
                for team in teams_by_season_league.get((prev_season, league), set()):
                    prev_team_tiers[team] = tier

            # Detect promotions (team moved to a lower tier number = higher division)
            for team, curr_tier in curr_team_tiers.items():
                if team in prev_team_tiers and prev_team_tiers[team] != curr_tier:
                    prev_tier = prev_team_tiers[team]
                    curr_league = group_leagues[curr_tier]
                    prev_league = group_leagues[prev_tier]
                    if curr_tier < prev_tier:
                        # Moved to higher division → promoted
                        self.set_promoted(season, curr_league, {team})
                        self.set_relegated(prev_season, prev_league, {team})
                    else:
                        # Moved to lower division → relegated
                        self.set_relegated(prev_season, prev_league, {team})
                        self.set_promoted(season, curr_league, {team})

            # Detect new teams in the group (not seen in previous season)
            for league in group_leagues:
                curr_teams = teams_by_season_league.get((season, league), set())
                prev_teams = teams_by_season_league.get((prev_season, league), set())

                # Teams in current but not in previous (for this league)
                new_teams = curr_teams - prev_teams
                # Filter: only flag if team wasn't in the group at all last season
                truly_new = {
                    t for t in new_teams
                    if t not in prev_team_tiers
                }

                # Teams in previous but not in current (for this league)
                departed = prev_teams - curr_teams
                truly_departed = {
                    t for t in departed
                    if t not in curr_team_tiers
                }

                if truly_new:
                    # Team wasn't in any group league last season → promoted in
                    self.set_promoted(season, league, truly_new)
                if truly_departed:
                    # Team not in any group league this season → relegated out
                    self.set_relegated(prev_season, league, truly_departed)

    # ══════════════════════════════════════════════════════
    #  History management
    # ══════════════════════════════════════════════════════

    def store_season(
        self,
        season: str,
        league: str,
        record: LeagueStrengthRecord,
    ) -> None:
        """Manually store a league strength record."""
        self._history[(season, league)] = record

    def get_season(
        self,
        season: str,
        league: str,
    ) -> LeagueStrengthRecord | None:
        """Retrieve a stored league strength record."""
        return self._history.get((season, league))

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return all stored records as a DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._history.values()])

    def clear_history(self) -> None:
        """Clear all stored records and tracked promotions/relegations."""
        self._history.clear()
        self._promotions.clear()
        self._relegations.clear()
        self._european.clear()

    # ══════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════

    def save_json(self, path: str | Path) -> None:
        """Save all stored records to a JSON file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        """
        data = {
            f"{k[0]}/{k[1]}": v.to_dict() for k, v in self._history.items()
        }
        meta = {
            "reference_league": self.reference_league,
            "min_matches": self.min_matches,
        }
        payload = {"metadata": meta, "records": data}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Saved %d league strength records to %s", len(data), path)

    def load_json(self, path: str | Path) -> int:
        """Load stored records from a JSON file.

        Returns
        -------
        int
            Number of records loaded.
        """
        with open(path) as f:
            payload = json.load(f)

        meta = payload.get("metadata", {})
        self.reference_league = meta.get("reference_league", self.reference_league)
        self.min_matches = meta.get("min_matches", self.min_matches)

        count = 0
        for key_str, record_dict in payload.get("records", {}).items():
            # key_str is "season/league"
            parts = key_str.split("/", 1)
            if len(parts) != 2:
                continue
            season, league = parts
            record = LeagueStrengthRecord.from_dict(record_dict)
            self._history[(season, league)] = record
            count += 1

        logger.info("Loaded %d league strength records from %s", count, path)
        return count

    # ══════════════════════════════════════════════════════
    #  European competition adjustment
    # ══════════════════════════════════════════════════════

    def european_adjustment(
        self,
        df: pd.DataFrame,
        season_col: str = "season",
        league_col: str = "league",
        home_team_col: str = "home_team",
        away_team_col: str = "away_team",
        home_goals_col: str = "home_goals",
        away_goals_col: str = "away_goals",
        result_col: str = "result",
        home_xg_col: str | None = "home_xg",
        away_xg_col: str | None = "away_xg",
    ) -> pd.DataFrame:
        """Compute league strength adjusted for European competition drain.

        Matches involving teams that participate in European competitions
        are flagged, and the league metrics are recomputed both with and
        without those matches, so the user can see the effect.

        Parameters
        ----------
        df : pd.DataFrame
            Match data.
        season_col, league_col : str
            Season and league columns.
        home_team_col, away_team_col : str
            Team columns.
        home_goals_col, away_goals_col : str
            Goals columns.
        result_col : str
            Column name for match result (default ``\"result\"``).
        home_xg_col, away_xg_col : str, optional
            xG column names.

        Returns
        -------
        pd.DataFrame
            Comparison of league metrics with and without European matches.
        """
        if not self._history:
            logger.warning("No historical data for European adjustment.")
            return pd.DataFrame()

        has_xg = (
            home_xg_col is not None
            and away_xg_col is not None
            and home_xg_col in df.columns
            and away_xg_col in df.columns
        )

        # Add a flag for matches involving European teams
        df = df.copy()
        df["_has_european"] = False

        for (season, league), euro_teams in self._european.items():
            mask = (
                (df[season_col].astype(str) == season)
                & (df[league_col].astype(str) == league)
                & (
                    df[home_team_col].isin(euro_teams)
                    | df[away_team_col].isin(euro_teams)
                )
            )
            df.loc[mask, "_has_european"] = True

        # Compute metrics with and without European matches
        rows: list[dict[str, Any]] = []
        for (season, league), record in self._history.items():
            subset = df[
                (df[season_col].astype(str) == season)
                & (df[league_col].astype(str) == league)
            ]
            euro_mask = subset["_has_european"]

            # With European
            with_euro = record.to_dict()
            with_euro["type"] = "with_european"

            # Without European
            no_euro = subset[~euro_mask]
            if len(no_euro) >= self.min_matches:
                no_euro_record = self._compute_single(
                    no_euro, season, league,
                    home_goals_col, away_goals_col, result_col,
                    home_xg_col, away_xg_col, has_xg,
                )
                without = no_euro_record.to_dict()
                without["type"] = "without_european"
                rows.append(with_euro)
                rows.append(without)
            else:
                rows.append(with_euro)

        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════


def create_league_strength_engine(
    reference_league: str = _DEFAULT_REFERENCE,
    min_matches: int = 10,
    auto_normalise: bool = True,
) -> LeagueStrengthEngine:
    """Create a configured LeagueStrengthEngine.

    Parameters
    ----------
    reference_league : str
        Reference league code for normalisation (default ``\"E0\"``).
    min_matches : int
        Minimum matches per season-league to compute metrics (default 10).
    auto_normalise : bool
        Auto-compute attack/defence factors (default True).

    Returns
    -------
    LeagueStrengthEngine
    """
    return LeagueStrengthEngine(
        reference_league=reference_league,
        min_matches=min_matches,
        auto_normalise=auto_normalise,
    )
