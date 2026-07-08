"""transfermarkt_lineups.py — Starting XI lineup scraper for Transfermarkt match reports.

Scrapes actual starting XI lineups from Transfermarkt match report pages and
outputs a lineups DataFrame that slots directly into
``src.player_info.add_player_features()``.

Data source: https://www.transfermarkt.com/
Licence: Freely accessible data — respect robots.txt and rate limits.

Output schema (lineups_df)
---------------------------
team         str       Team name (matches openfootball convention)
date         str/date  Match date
player_name  str       Player in the starting XI

Match ID ranges (per tournament, estimated)
--------------------------------------------
World Cup 2022:  3788838–3975879
World Cup 2018:  3100000–3300000 (estimated)
World Cup 2014:  2500000–2700000 (estimated)
World Cup 2010:  1900000–2100000 (estimated)
World Cup 2006:  1500000–1700000 (estimated)
World Cup 2002:  1100000–1300000 (estimated)
World Cup 2026:  Will be discovered as matches are played

Usage
-----
    from src.data_collection.sources.transfermarkt_lineups import (
        scrape_match_lineup, find_match_ids_for_tournament
    )

    # Scrape a single match
    lineup = scrape_match_lineup("https://.../spielbericht/3788846")

    # Find match IDs for a tournament
    match_ids = find_match_ids_for_tournament(2022, 3788838, 3800000)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data_collection.sources.transfermarkt import (
    TEAM_TO_TM_ID,
    _team_slug,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

TRANSFERMARKT_BASE = "https://www.transfermarkt.com"
REQUEST_TIMEOUT = 20
MATCH_REPORT_PATH = "index/spielbericht"
TEAM_MATCHES_PATH = "spielplan/verein"

# ── Session ─────────────────────────────────────────────

def session() -> requests.Session:
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
#  Public API
# ═══════════════════════════════════════════════════════════


def scrape_match_lineup(
    match_url: str,
    team_name_home: str | None = None,
    team_name_away: str | None = None,
    sess: requests.Session | None = None,
    html: str | None = None,
) -> pd.DataFrame:
    """Scrape starting XI lineups from a Transfermarkt match report page.

    Parameters
    ----------
    match_url : str
        Full URL to the Transfermarkt match report page (used when ``html`` is not supplied).
    team_name_home : str, optional
        Expected home team name. If provided, used to verify the page.
    team_name_away : str, optional
        Expected away team name.
    sess : requests.Session, optional
        Reusable HTTP session.
    html : str, optional
        Pre-fetched HTML content.  When supplied, skips the HTTP request
        (avoids a double-fetch when the caller already retrieved the page).

    Returns
    -------
    pd.DataFrame
        Lineups DataFrame with columns: ``team``, ``date``, ``player_name``.
        One row per player in the starting XI (11 per team, 22 total).
        Returns empty DataFrame if the page can't be parsed.
    """
    if html is not None:
        return _parse_match_page(html, team_name_home, team_name_away)

    if sess is None:
        sess = session()

    try:
        resp = sess.get(match_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Failed to fetch %s: %s", match_url, exc)
        return pd.DataFrame(columns=["team", "date", "player_name"])

    return _parse_match_page(resp.text, team_name_home, team_name_away)


def scrape_team_matches(
    team_name: str,
    season: int,
    sess: requests.Session | None = None,
) -> pd.DataFrame:
    """Scrape all matches and their match report links for a team's season.

    Parameters
    ----------
    team_name : str
        Team name (must be in TEAM_TO_TM_ID mapping).
    season : int
        Season year (e.g., 2022 for 2022/23 season).
    sess : requests.Session, optional

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``date``, ``home_team``, ``away_team``,
        ``result``, ``match_url``, ``match_id``.
    """
    tm_id = TEAM_TO_TM_ID.get(team_name)
    if tm_id is None:
        logger.warning("No Transfermarkt ID for '%s'", team_name)
        return pd.DataFrame()

    close_session = sess is None
    if sess is None:
        sess = session()

    slug = _team_slug(team_name)
    url = (
        f"{TRANSFERMARKT_BASE}/{slug}/{TEAM_MATCHES_PATH}/{tm_id}"
        f"/saison_id/{season}/plus/1"
    )

    try:
        resp = sess.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Failed to fetch %s: %s", url, exc)
        return pd.DataFrame()

    return _parse_team_matches_page(resp.text, team_name)


def scrape_all_tournament_lineups(
    year: int,
    team_names: list[str],
    id_start: int,
    id_end: int,
    delay: float = 1.5,
) -> pd.DataFrame:
    """Scrape lineups for all World Cup matches.

    Discovers match IDs by checking each team's match history page,
    then scrapes lineups for matches between known teams.

    Parameters
    ----------
    year : int
        Tournament year.
    team_names : list[str]
        List of team names participating in the tournament.
    id_start : int
        Lower bound for Transfermarkt match IDs.
    id_end : int
        Upper bound for Transfermarkt match IDs.
    delay : float
        Seconds between requests.

    Returns
    -------
    pd.DataFrame
        Combined lineups DataFrame with columns: ``team``, ``date``, ``player_name``.
    """
    sess = session()
    all_lineups: list[pd.DataFrame] = []
    found_match_ids: set[int] = set()

    logger.info("Discovering match IDs for %d World Cup ...", year)

    # Step 1: Find match IDs from team match history pages
    for team in team_names:
        tm_id = TEAM_TO_TM_ID.get(team)
        if tm_id is None:
            continue

        slug = _team_slug(team)
        url = (
            f"{TRANSFERMARKT_BASE}/{slug}/{TEAM_MATCHES_PATH}/{tm_id}"
            f"/saison_id/{year}/plus/1"
        )

        try:
            resp = sess.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception:
            continue

        matches_df = _parse_team_matches_page(resp.text, team)
        if matches_df.empty:
            continue

        for _, row in matches_df.iterrows():
            match_id = row.get("match_id")
            if match_id and match_id not in found_match_ids:
                found_match_ids.add(match_id)

        time.sleep(delay)

    logger.info("Found %d unique match IDs", len(found_match_ids))

    # Step 2: Scrape lineups for each match
    for i, match_id in enumerate(sorted(found_match_ids)):
        url = f"{TRANSFERMARKT_BASE}/index/spielbericht/{match_id}"
        logger.info("  [%d/%d] Match %d ...", i + 1, len(found_match_ids), match_id)

        try:
            lineup_df = scrape_match_lineup(url, sess=sess)
            if not lineup_df.empty:
                all_lineups.append(lineup_df)
                logger.info("    -> %d players", len(lineup_df))
        except Exception as exc:
            logger.debug("  [W] Failed match %d: %s", match_id, exc)

        if i < len(found_match_ids) - 1:
            time.sleep(delay)

    if all_lineups:
        result = pd.concat(all_lineups, ignore_index=True)
        logger.info(
            "Scraped %d lineup records for %d matches",
            len(result), len(all_lineups),
        )
        return result

    return pd.DataFrame(columns=["team", "date", "player_name"])


# ═══════════════════════════════════════════════════════════
#  Internal helpers — parsing
# ═══════════════════════════════════════════════════════════


def _parse_match_page(
    html: str,
    team_name_home: str | None = None,
    team_name_away: str | None = None,
) -> pd.DataFrame:
    """Parse a Transfermarkt match report page for starting XIs.

    Strategy:
    1. Extract team names from page title or from "Line-Ups" section
    2. Collect ALL player profile links (both starters and subs)
    3. Detect which section each player is in by checking parent containers
       - Players under "aufstellung-vereinsseite" div = starters
       - Players under "aufstellung-ersatzbank" div = substitutes

    Returns
    -------
    pd.DataFrame
        Columns: team, date, player_name
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    date = _extract_date_from_title(title)

    # Extract team names from title: "Brazil - Serbia, 24/11/2022 - World Cup - Match sheet"
    team_names = _extract_team_names_from_title(title, team_name_home, team_name_away)

    records: list[dict[str, str]] = []

    # Find all player profile links
    all_links = soup.find_all(
        "a", href=lambda h: h and "/profil/spieler/" in (h or "")
    )

    for a_tag in all_links:
        player_name = a_tag.get_text(strip=True)
        if not player_name or len(player_name) < 2:
            continue

        # Check if this player link is in the starting XI or substitutes section
        # by walking up the parent tree
        parent = a_tag.parent
        is_starter = False
        team_name = None

        while parent and parent != soup:
            parent_classes = parent.get("class", [])
            parent_class_str = " ".join(parent_classes) if parent_classes else ""

            # Check if we're in the starting XI section
            if "aufstellung-vereinsseite" in parent_class_str:
                is_starter = True

            # Check if we're in the substitutes section
            if "aufstellung-ersatzbank" in parent_class_str:
                is_starter = False
                break

            # Check if we're in a box with team name
            if "aufstellung-box" in parent_class_str:
                # Determine which team this box belongs to
                if len(records) == 0:
                    # First box = first team
                    team_name = team_names[0] if team_names else "Home"
                else:
                    # Check if we've already assigned team0's players
                    team0_count = sum(1 for r in records if r["team"] == (team_names[0] if team_names else "Home"))
                    if team0_count < 14:  # 11 starters + few subs
                        team_name = team_names[0] if team_names else "Home"
                    else:
                        team_name = team_names[1] if len(team_names) > 1 else "Away"

            parent = parent.parent

        if is_starter and team_name:
            records.append({
                "team": team_name,
                "date": date,
                "player_name": player_name,
            })

    # Direct approach: find ALL aufstellung-vereinsseite sections with player links
    # This is the most reliable approach - these divs contain starting XI players
    xi_sections = soup.find_all("div", class_=lambda c: c and "aufstellung-vereinsseite" in (" ".join(c) if isinstance(c, list) else str(c)))
    xi_sections_with_players = []
    for section in xi_sections:
        links = section.find_all("a", href=lambda h: h and "/profil/spieler/" in (h or ""))
        # A starting XI section has 5-15 player links
        # A substitutes section might have more or fewer
        # Exclude sections that are clearly substitutes (contain "ersatz" in class)
        section_classes = " ".join(section.get("class", []))
        if "ersatz" not in section_classes.lower() and 5 <= len(links) <= 15:
            xi_sections_with_players.append((section, links))

    # Take XI sections and deduplicate: no more than 15 per team
    seen: set[tuple[str, str]] = set()  # (team, player_name)
    for team_idx, (section, links) in enumerate(xi_sections_with_players):
        box_team = team_names[team_idx] if team_idx < len(team_names) else f"Team{team_idx + 1}"
        for a_tag in links:
            pname = a_tag.get_text(strip=True)
            if pname and len(pname) > 1:
                key = (box_team, pname)
                if key not in seen:
                    seen.add(key)
                    records.append({"team": box_team, "date": date, "player_name": pname})
        # Stop after we have enough for both teams
        team_counts = {}
        for r in records:
            team_counts[r["team"]] = team_counts.get(r["team"], 0) + 1
        if len(team_counts) >= 2 and all(c >= 11 for c in team_counts.values()):
            break

    if records:
        # Success - limit to 11 per team
        final: list[dict[str, str]] = []
        team_seen: dict[str, int] = {}
        for r in records:
            t = r["team"]
            team_seen[t] = team_seen.get(t, 0) + 1
            if team_seen[t] <= 11:
                final.append(r)
        records = final

    # Last resort: text-based fallback
    if not records:
        records = _parse_lineup_text_based(soup, date, team_names)

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["team", "date", "player_name"]
    )
    return df


def _extract_team_names_from_title(
    title: str,
    team_name_home: str | None = None,
    team_name_away: str | None = None,
) -> tuple[str, str]:
    """Extract team names from the page title or provided names.

    Title format: "Brazil - Serbia, 24/11/2022 - World Cup - Match sheet"
    """
    # Extract from title
    parts = title.split(" - ")
    if len(parts) >= 2:
        teams_part = parts[0]
        # Remove date/score suffix
        team_parts = teams_part.split(",")[0].strip()
        # Split by " vs " or " - "
        for sep in [" vs ", " v ", " - ", "–", "—"]:
            if sep in team_parts:
                names = team_parts.split(sep, 1)
                return (names[0].strip(), names[1].strip())
        # Fallback: just return the whole first part
        return (team_parts, team_name_away or "Away")

    return (team_name_home or "Home", team_name_away or "Away")



def _parse_lineup_text_based(soup: BeautifulSoup, date: str, team_names: tuple[str, str] | None = None) -> list[dict[str, str]]:
    """Fallback: parse starting XI from plain text sections."""
    text = soup.get_text(separator="\n")
    lines = text.split("\n")
    records: list[dict[str, str]] = []

    # Find "Line-Ups" or "Aufstellung" header
    lineup_start = -1
    for i, line in enumerate(lines):
        ls = line.strip()
        if ls in ("Line-Ups", "Aufstellung", "LINE-UPS"):
            lineup_start = i
            break

    if lineup_start < 0:
        return records

    current_team = None
    in_starting_xi = False

    for i in range(lineup_start + 1, len(lines)):
        line = lines[i].strip()

        if line in ("Goals", "Tore"):
            break

        if "Starting Line-up" in line or "Startaufstellung" in line:
            in_starting_xi = True
            continue

        if "Substitutes" in line or "Auswechselspieler" in line:
            in_starting_xi = False
            current_team = None
            continue

        if line and in_starting_xi and current_team:
            name = _extract_player_name(line)
            if name:
                records.append({"team": current_team, "date": date, "player_name": name})

        # Team name detection
        if not in_starting_xi and line and len(line) < 30 and not line[0].isdigit():
            for j in range(1, 6):
                if i + j < len(lines):
                    ahead = lines[i + j].strip()
                    if "Starting Line-up" in ahead:
                        current_team = line
                        break
                    if ahead in ("Substitutes", "Goals", "Cards") or len(ahead) > 50:
                        break

    return records


def _extract_date_from_title(title: str) -> str:
    """Extract date from Transfermarkt match page title.

    Handles formats like:
    - "Brazil - Serbia, 24/11/2022 - World Cup - Match sheet"
    """
    # Try DD/MM/YYYY pattern
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", title)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # Try YYYY-MM-DD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", title)
    if m:
        return m.group(1)
    # Try "Nov 24, 2022" pattern
    m = re.search(
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\.?\s*(\d{4})",
        title, re.IGNORECASE,
    )
    if m:
        months = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        month = months.get(m.group(2).lower()[:3], "01")
        day = m.group(1).zfill(2)
        return f"{m.group(3)}-{month}-{day}"
    return ""


def _extract_player_name(text: str) -> str | None:
    """Extract a player name from a line of text.

    Cleans up formatting, removes shirt numbers, and returns just the name.
    """
    if not text or len(text) < 2:
        return None

    # Remove leading number and dash (e.g., "23 - Alisson" or "10. Neymar")
    text = re.sub(r"^\d+\s*[.-]\s*", "", text).strip()

    # Remove trailing position descriptions in parentheses
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()

    # Remove common suffixes that aren't part of the name
    text = re.sub(r"\s*,\s*(Tactical|Injury|Yellow card|Red card|Goal).*$", "", text, re.IGNORECASE).strip()

    # Filter out non-player lines
    skip_words = {"goals", "assist", "cards", "substitutions", "manager", "timeline",
                  "world cup", "referee", "attendance", "stadium", "line-ups"}
    text_lower = text.lower().strip()
    if not text or text_lower in skip_words or any(text_lower.startswith(w) for w in skip_words):
        return None

    # Minimum name length (a real name has at least 3 chars and isn't all caps formation like "4-2-3-1")
    if len(text) < 3 or re.match(r"^\d+[-]\d+[-]\d+", text):
        return None

    return text.strip()


def _parse_lineup_from_html(
    soup: BeautifulSoup,
    date: str,
    team_home: str | None,
    team_away: str | None,
) -> list[dict[str, str]]:
    """Fallback: parse starting XI from HTML structure.

    Looks for table or div elements containing lineup information.
    """
    records: list[dict[str, str]] = []
    teams_found: list[str] = []

    # Look for divs that contain "Starting Line-up"
    for div in soup.find_all("div"):
        div_text = div.get_text(separator="\n", strip=True)
        if "Starting Line-up" not in div_text:
            continue

        # Try to identify which team this is
        parent = div.parent
        if parent:
            parent_text = parent.get_text(separator="\n", strip=True)
            # Find team name: look before "Starting Line-up"
            lines = parent_text.split("\n")
            for i, line in enumerate(lines):
                if "Starting Line-up" in line and i > 0:
                    potential_team = lines[i - 1].strip()
                    if potential_team and len(potential_team) < 30:
                        teams_found.append(potential_team)
                        break

        # Find player names in the same container area
        container = div.find_parent(["div", "td"])
        if container:
            for p_tag in container.find_all(["p", "span", "a"]):
                name = p_tag.get_text(strip=True)
                if name and 2 < len(name) < 40:
                    player_name = _extract_player_name(name)
                    if player_name and teams_found:
                        records.append({
                            "team": teams_found[-1],
                            "date": date,
                            "player_name": player_name,
                        })

    return records


def _parse_team_matches_page(html: str, team_name: str) -> pd.DataFrame:
    """Parse a team's match history page to find match report links.

    Transfermarkt now renders the schedule with plain <table> elements
    (not class="items").  Each match-data table contains a "Date" column
    header and rows with ``spielbericht`` links.

    Returns
    -------
    pd.DataFrame
        Columns: date, home_team, away_team, result, match_url, match_id
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []

    # Find ALL tables that contain match data (rows with spielbericht links)
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        # Check if this table has a header row with "Date"
        header_cells = rows[0].find_all(["th", "td"])
        header_texts = [c.get_text(strip=True).lower() for c in header_cells]
        if "date" not in header_texts:
            continue

        # Parse match rows (skip header row)
        for tr in rows[1:]:
            cells = tr.find_all("td")
            if len(cells) < 6:
                continue

            # Extract match date from the first cell
            date_text = cells[0].get_text(strip=True) if len(cells) > 0 else ""

            # Find match report link
            match_link = None
            match_id = None
            for cell in cells:
                link = cell.find("a", href=lambda h: h and "spielbericht" in h)
                if link:
                    match_link = link.get("href")
                    m = re.search(r"/spielbericht/(\d+)", str(link.get("href", "")))
                    if m:
                        match_id = int(m.group(1))
                    break

            if match_id is None:
                continue

            # Extract home/away teams from cells
            # Transfermarkt schedule table layout:
            #   [Date, Time, (crest), Home, (crest), Away, Formation, Coach, Attendance, Result]
            # The exact column index varies by table, so find text cells that aren't empty
            text_cells = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]

            # Skip cells that are date, time, or metadata
            # Find team names by looking for non-empty text cells after the time column
            home_team, away_team = "", ""

            # Cells with team crest images have no text but contain <img> tags.
            # The adjacent cell has the team name.
            # Walk through cells looking for team name patterns (> 2 chars, not date/time/numbers)
            team_candidates = []
            for cell in cells[2:]:  # skip date + time
                txt = cell.get_text(strip=True)
                if txt and len(txt) > 2 and not re.match(r"^[\d:.-]+$", txt):
                    team_candidates.append(txt)

            if len(team_candidates) >= 2:
                home_team = team_candidates[0]
                away_team = team_candidates[1]
            elif team_candidates:
                home_team = team_candidates[0]

            # Extract result from text cells if present (e.g. "1:5" or "2-0")
            result = ""
            for txt in text_cells:
                if re.match(r"^\d+[ :-]\d+$", txt):
                    result = txt
                    break

            records.append({
                "date": date_text,
                "home_team": home_team,
                "away_team": away_team,
                "result": result,
                "match_url": str(match_link) if match_link else "",
                "match_id": match_id,
            })

    if not records:
        return pd.DataFrame(
            columns=["date", "home_team", "away_team", "result", "match_url", "match_id"]
        )

    df = pd.DataFrame(records)
    return df
