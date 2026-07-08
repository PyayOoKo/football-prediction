"""Transfermarkt — squad player data scraper.

Scrapes per-player information (age, market value, position, injury/suspension
status) for national team squads from Transfermarkt, and outputs DataFrames that
slot directly into ``src.player_info.add_player_features()``.

Data source: https://www.transfermarkt.com/
Licence: Freely accessible data — respect robots.txt and rate limits.

Two output formats
------------------
1. **players_df** — Per-team squad roster (one row per player).
   Columns: team, player_name, position, age, market_value, is_starter,
            injured, suspended, goals_scored
2. **lineups_df** — This scraper does NOT provide starting XI data for
   individual matches (FBref is a better source for that). The lineups
   DataFrame is returned as ``None``.

Usage
-----
    from src.data_collection.sources.transfermarkt import scrape_squads

    players = scrape_squads(team_mapping)
    # players: pd.DataFrame with squad info for all teams
    # lineups: None
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

TRANSFERMARKT_BASE = "https://www.transfermarkt.com"
"""Base URL of Transfermarkt."""

SQUAD_PATH = "kader/verein"
"""URL path for squad pages (German: Kader = squad)."""

REQUEST_TIMEOUT = 20
"""HTTP request timeout in seconds."""

CACHE_DIR_NAME = "external"
"""Subdirectory within data/ where player CSVs are stored."""

# ── National team Transfermarkt IDs ─────────────────────
# Mapping from openfootball team name → Transfermarkt national team ID.
# Includes all teams from World Cup data (2002-2026).
# Source: Transfermarkt national team pages.
TEAM_TO_TM_ID: dict[str, int] = {
    # 2026 World Cup + historical teams
    # IDs verified against Transfermarkt national team pages
    "Algeria":            3478,
    "Angola":             3481,
    "Argentina":          3437,
    "Australia":          3433,
    "Austria":            3383,
    "Belgium":            3382,
    "Bosnia-Herzegovina": 3581,
    "Bosnia & Herzegovina": 3581,
    "Brazil":             3439,
    "Cameroon":           3434,
    "Canada":             3510,
    "Cape Verde":         3488,
    "Chile":              3700,
    "China":              3445,
    "Colombia":           3816,
    "Costa Rica":         3447,
    "Croatia":            3556,
    "Curaçao":            3583,
    "Czech Republic":     3586,
    "Côte d'Ivoire":      3451,
    "DR Congo":           3584,
    "Denmark":            3436,
    "Ecuador":            5750,
    "Egypt":              3453,
    "England":            3299,
    "France":             3377,
    "Germany":            3262,
    "Ghana":              3441,
    "Greece":             3458,
    "Haiti":              3477,
    "Honduras":           3472,
    "Iceland":            3621,
    "Iran":               3582,
    "Iraq":               3474,
    "Ireland":            3509,
    "Italy":              3376,
    "Ivory Coast":        3451,  # alias for Côte d'Ivoire
    "Japan":              3435,
    "Jordan":             3480,
    "Mexico":             6303,
    "Morocco":            3465,
    "Netherlands":        3379,
    "New Zealand":        3482,
    "Nigeria":            3444,
    "North Korea":        3468,
    "Norway":             3440,
    "Portugal":           3300,
    "Saudi Arabia":       3807,
    "Scotland":           3380,
    "Senegal":            3499,
    "Serbia":             3438,
    "Serbia and Montenegro": 3585,
    "Slovakia":           3615,
    "Slovenia":           3614,
    "South Africa":       3473,
    "South Korea":        3589,
    "Spain":              3375,
    "Sweden":             3557,
    "Switzerland":        3384,
    "Togo":               3492,
    "Trinidad and Tobago": 3491,
    "Tunisia":            3489,
    "Turkey":             3381,
    "USA":                3505,
    "Ukraine":            3699,
    "Uruguay":            3449,
    "Uzbekistan":         3624,
}

# Known URL slugs for Transfermarkt team pages (lowercase, dashed).
# Used to verify team URLs. If a team isn't here, the slug is derived
# from the team name automatically.
TEAM_SLUG_OVERRIDES: dict[str, str] = {
    "Bosnia & Herzegovina": "bosnien-herzegowina",
    "Bosnia-Herzegovina": "bosnien-herzegowina",
    "Cape Verde": "kap-verde",
    "Côte d'Ivoire": "elfenbeinkueste",
    "Curaçao": "curaçao",
    "Czech Republic": "tschechien",
    "DR Congo": "kongo-demokratische-republik",
    "Ivory Coast": "elfenbeinkueste",
    "North Korea": "nordkorea",
    "Saudi Arabia": "saudi-arabien",
    "South Korea": "suedkorea",
    "Trinidad and Tobago": "trinidad-und-tobago",
    "Australia": "australien",
    "Austria": "oesterreich",
    "Belgium": "belgien",
    "Brazil": "brasilien",
    "Croatia": "kroatien",
    "Czech Republic": "tschechien",
    "Denmark": "daenemark",
    "Greece": "griechenland",
    "Iceland": "island",
    "Italy": "italien",
    "Japan": "japan",
    "Netherlands": "niederlande",
    "Norway": "norwegen",
    "Poland": "polen",
    "Saudi Arabia": "saudi-arabien",
    "Scotland": "schottland",
    "South Korea": "suedkorea",
    "Spain": "spanien",
    "Sweden": "schweden",
    "Switzerland": "schweiz",
    "Turkey": "tuerkei",
    "Ukraine": "ukraine",
    "USA": "usa",
}

# ── Session ─────────────────────────────────────────────


def _session() -> requests.Session:
    """Create a requests session with retry logic and polite headers."""
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=2.0, status_forcelist=[502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return sess


# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class PlayerRecord:
    """Scraped data for a single player."""

    team: str
    player_name: str
    position: str
    age: float
    market_value: float  # in millions of Euros (€m)
    is_starter: bool = False
    injured: bool = False
    suspended: bool = False
    goals_scored: int = 0
    shirt_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict matching the ``players_df`` schema."""
        return {
            "team": self.team,
            "player_name": self.player_name,
            "position": self.position,
            "age": self.age,
            "market_value": self.market_value,
            "is_starter": self.is_starter,
            "injured": self.injured,
            "suspended": self.suspended,
            "goals_scored": self.goals_scored,
            "shirt_number": self.shirt_number,
        }


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════


def scrape_squads(
    team_names: list[str],
    team_id_map: dict[str, int] | None = None,
    delay: float = 1.0,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Scrape squad data for a list of national teams.

    Parameters
    ----------
    team_names : list[str]
        List of team names (must match keys in ``team_id_map``).
    team_id_map : dict[str, int], optional
        Mapping from team name → Transfermarkt team ID.
        Defaults to ``TEAM_TO_TM_ID``.
    delay : float
        Seconds to wait between requests (default 1.0 — be polite).
    save_path : str, optional
        If provided, save the resulting DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Players DataFrame with columns matching the ``player_info`` schema:
        ``team``, ``player_name``, ``position``, ``age``, ``market_value``,
        ``is_starter``, ``injured``, ``suspended``, ``goals_scored``.
    """
    if team_id_map is None:
        team_id_map = TEAM_TO_TM_ID

    all_players: list[PlayerRecord] = []
    sess = _session()
    missed: list[str] = []

    for i, team in enumerate(team_names):
        tm_id = team_id_map.get(team)
        if tm_id is None:
            logger.warning("  [W] No Transfermarkt ID for '%s' — skipping", team)
            missed.append(team)
            continue

        slug = _team_slug(team)
        url = f"{TRANSFERMARKT_BASE}/{slug}/{SQUAD_PATH}/{tm_id}"

        logger.info("  [%d/%d] %s ...", i + 1, len(team_names), team)

        try:
            players = _scrape_squad_page(url, team, sess)
            all_players.extend(players)
            logger.info("    -> %d players", len(players))
        except Exception as exc:
            logger.warning("    [W] Failed: %s", exc)
            missed.append(team)

        # Be polite: delay between requests
        if i < len(team_names) - 1:
            time.sleep(delay)

    df = _build_dataframe(all_players)

    if missed:
        logger.warning("Teams with no data: %s", ", ".join(missed))

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved %d player records to %s", len(df), save_path)

    return df


def scrape_single_team(team_name: str) -> pd.DataFrame:
    """Scrape squad data for a single team.

    Parameters
    ----------
    team_name : str
        Team name (must be in ``TEAM_TO_TM_ID``).

    Returns
    -------
    pd.DataFrame
        Players DataFrame for the team.
    """
    df = scrape_squads([team_name], delay=0)
    return df


# ═══════════════════════════════════════════════════════════
#  Internal helpers — scraping
# ═══════════════════════════════════════════════════════════


def _team_slug(team: str) -> str:
    """Convert an openfootball team name to a Transfermarkt URL slug.

    Uses overrides where needed, otherwise lowercases and hyphenates.
    """
    if team in TEAM_SLUG_OVERRIDES:
        return TEAM_SLUG_OVERRIDES[team]

    # Default: lowercase, spaces → hyphens, remove special chars
    slug = team.lower()
    slug = slug.replace(" & ", "-")
    slug = slug.replace(" and ", "-")
    slug = slug.replace(" ", "-")
    slug = slug.replace("'", "")
    slug = slug.replace(".", "")
    slug = slug.replace("é", "e")
    slug = slug.replace("ô", "o")
    slug = slug.replace("è", "e")
    slug = slug.replace("ü", "ue")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


def _scrape_squad_page(
    url: str,
    team_name: str,
    sess: requests.Session,
) -> list[PlayerRecord]:
    """Scrape a single Transfermarkt squad page and return player records.

    Parameters
    ----------
    url : str
        Full URL to the squad page.
    team_name : str
        Canonical team name (from the openfootball convention).
    sess : requests.Session
        Reusable HTTP session.

    Returns
    -------
    list[PlayerRecord]
        Parsed player records from the page.
    """
    resp = sess.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the squad table — Transfermarkt uses a responsive table
    # with rows containing player data
    table = soup.find("table", class_="items")
    if table is None:
        # Try alternative: find the main content table
        table = soup.find("div", class_="responsive-table")
        if table is not None:
            table = table.find("table")

    if table is None:
        raise ValueError(f"No squad table found on page: {url}")

    tbody = table.find("tbody")
    if tbody is None:
        tbody = table

    rows = tbody.find_all("tr", recursive=False)
    players: list[PlayerRecord] = []

    for tr in rows:
        # Skip header rows, separator rows, or rows without data cells
        if tr.get("class") and "no-data" in (tr.get("class") or []):
            continue

        player = _parse_player_row(tr, team_name)
        if player is not None:
            players.append(player)

    return players


def _parse_player_row(tr: Tag, team_name: str) -> PlayerRecord | None:
    """Parse a single <tr> element from the squad table.

    Handles both layouts:
    - **National team** (5 columns): #, Player+Position, Age, Club, Value
    - **Club** (8+ columns):       #, Player+Position, Age, ..., Value

    Returns ``None`` if the row is not a valid player row.
    """
    cells = tr.find_all("td")
    n_cells = len(cells)
    if n_cells < 3:
        return None

    # Shirt number check: skip rows without a digit in col 0
    num_text = cells[0].get_text(strip=True)
    if not num_text.isdigit():
        return None

    try:
        shirt_number = int(num_text)

        # Player name & position (always col 1)
        name_cell = cells[1]
        name_link = name_cell.find("a")
        player_name = name_link.get_text(strip=True) if name_link else ""
        if not player_name:
            return None

        # Position: try <span class="pos"> first
        pos_span = name_cell.find("span", class_="pos")
        if pos_span:
            position = pos_span.get_text(strip=True)
        else:
            # National team layout: position is inline text after name
            # e.g. "Jude Bellingham\nAttacking Midfield"
            full_text = name_cell.get_text(separator="\n", strip=True)
            parts = [p.strip() for p in full_text.split("\n") if p.strip()]
            position = parts[-1] if len(parts) >= 2 else "Unknown"

        position = _normalise_position(position)

        # Age (col 2 in both layouts)
        age_cell = cells[2] if n_cells > 2 else None
        age = 25.0
        if age_cell:
            try:
                age = float(age_cell.get_text(strip=True))
            except (ValueError, TypeError):
                pass

        # Injury / Suspension icons
        injured = _check_injury_icon(name_cell)
        suspended = _check_suspension_icon(name_cell)

        # Market value: last column in both layouts
        value_cell = cells[-1]
        market_value = _parse_market_value(value_cell.get_text(strip=True))

        # Starter status: not reliable from shirt numbers
        is_starter = False

        return PlayerRecord(
            team=team_name,
            player_name=player_name,
            position=position,
            age=age,
            market_value=market_value,
            is_starter=is_starter,
            injured=injured,
            suspended=suspended,
            goals_scored=0,
            shirt_number=shirt_number,
        )

    except Exception as exc:
        logger.debug("Failed to parse row: %s", exc)
        return None


def _check_injury_icon(cell: Tag) -> bool:
    """Check a name cell for injury indicator icons."""
    # Transfermarkt uses icons with specific classes for injuries
    icons = cell.find_all("img", class_=lambda c: c and "injury" in str(c).lower())
    if icons:
        return True
    # Also check for injury-related font-awesome icons or span classes
    spans = cell.find_all(
        "span", class_=lambda c: c and "verletzt" in str(c).lower()
    )
    if spans:
        return True
    # Check for medical cross icon (✚ or ⚕)
    for sup in cell.find_all("sup"):
        text = sup.get_text(strip=True)
        if text in ("✚", "✛", "+", "†"):
            return True
    return False


def _check_suspension_icon(cell: Tag) -> bool:
    """Check a name cell for suspension indicator icons."""
    # Yellow/red card icons for suspended players
    icons = cell.find_all("img", class_=lambda c: c and "card" in str(c).lower())
    if icons:
        return True
    spans = cell.find_all(
        "span", class_=lambda c: c and "gesperrt" in str(c).lower()
    )
    if spans:
        return True
    return False


def _parse_market_value(text: str) -> float:
    """Parse a Transfermarkt market value string into millions of Euros.

    Examples
    --------
    >>> _parse_market_value("€75.00m")
    75.0
    >>> _parse_market_value("€1.2bn")
    1200.0
    >>> _parse_market_value("€500k")
    0.5
    >>> _parse_market_value("-")
    0.0
    """
    text = text.strip()
    if not text or text in ("-", "—", ""):
        return 0.0

    text = text.replace("€", "").replace(",", "").strip()

    try:
        if text.endswith("bn"):
            return float(text[:-2]) * 1000.0
        elif text.endswith("m"):
            return float(text[:-1])
        elif text.endswith("k"):
            return float(text[:-1]) / 1000.0
        elif text.endswith("Tsd."):
            return float(text[:-4]) / 1000.0
        else:
            return float(text)
    except (ValueError, TypeError):
        return 0.0


def _normalise_position(pos: str) -> str:
    """Normalise a position string to one of GK, DEF, MID, FWD."""
    pos = pos.strip().upper()
    if pos in ("GK", "GOALKEEPER", "GOAL", "T"):
        return "GK"
    if pos in ("DEF", "DEFENDER", "CB", "LB", "RB", "LWB", "RWB", "AB", "IV"):
        return "DEF"
    if pos in ("MID", "MIDFIELDER", "CM", "CDM", "CAM", "LM", "RM", "DM", "MF", "ZM", "OM"):
        return "MID"
    if pos in ("FWD", "FORWARD", "ST", "CF", "LW", "RW", "SS", "LF", "RF", "ANG", "MS"):
        return "FWD"
    # Some German abbreviations
    if pos in ("TW",):  # Torwart
        return "GK"
    if pos in ("LV", "RV"):  # Linker/Rücker Verteidiger
        return "DEF"
    if pos in ("DM", "ZM", "OM"):  # Defensives/Zentrales/Offensives Mittelfeld
        return "MID"
    if pos in ("MS", "HS"):  # Mittel-/Halbstürmer
        return "FWD"
    # Unknown — default to MID
    return "MID"


# ═══════════════════════════════════════════════════════════
#  Data assembly
# ═══════════════════════════════════════════════════════════


def _build_dataframe(players: list[PlayerRecord]) -> pd.DataFrame:
    """Convert a list of PlayerRecords to a DataFrame.

    Ensures all expected columns exist, even if the list is empty.
    """
    if not players:
        return pd.DataFrame(columns=[
            "team", "player_name", "position", "age", "market_value",
            "is_starter", "injured", "suspended", "goals_scored",
        ])

    records = [p.to_dict() for p in players]
    df = pd.DataFrame(records)

    # Ensure correct dtypes
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(25.0)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)
    df["is_starter"] = df["is_starter"].astype(bool)
    df["injured"] = df["injured"].astype(bool)
    df["suspended"] = df["suspended"].astype(bool)
    df["goals_scored"] = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0).astype(int)

    return df


# ═══════════════════════════════════════════════════════════
#  Team listing helper
# ═══════════════════════════════════════════════════════════


def get_supported_teams() -> list[str]:
    """Return the list of teams with Transfermarkt IDs, alphabetically sorted.

    Returns
    -------
    list[str]
        Unique team names from ``TEAM_TO_TM_ID`` (duplicate aliases removed).
    """
    seen: set[str] = set()
    teams: list[str] = []
    for name in sorted(TEAM_TO_TM_ID):
        if name not in seen:
            seen.add(name)
            teams.append(name)
    return teams


# ═══════════════════════════════════════════════════════════
#  CLI for testing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Test: scrape a single team
    test_team = sys.argv[1] if len(sys.argv) > 1 else "Brazil"
    print(f"\\n  Testing scrape of {test_team} squad...")
    print(f"  Team ID: {TEAM_TO_TM_ID.get(test_team, 'N/A')}")

    df = scrape_single_team(test_team)
    print(f"\\n  Scraped {len(df)} players for {test_team}:")
    print(f"  {'Name':<25} {'Pos':<6} {'Age':<5} {'Value (€m)':<12} {'Inj':<5} {'Start':<6}")
    print(f"  {'-' * 60}")
    for _, r in df.iterrows():
        print(f"  {r['player_name']:<25} {r['position']:<6} "
              f"{r['age']:<5.0f} {r['market_value']:<12.1f} "
              f"{'Y' if r['injured'] else 'N':<5} "
              f"{'Y' if r['is_starter'] else 'N':<6}")
    print(f"\\n  Squad value: €{df['market_value'].sum():.1f}m")
    print(f"  Avg age: {df['age'].mean():.1f} years")
