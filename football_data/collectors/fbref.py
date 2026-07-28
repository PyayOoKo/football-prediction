"""
FBref — Match results parser for second-tier leagues.

FBref (https://fbref.com) covers our target second-tier leagues (Superettan,
OBOS-ligaen, Ykkösliiga, etc.) but blocks automated scraping with Cloudflare.

This module provides a **manual save mode** (the only reliable approach):

1. User opens FBref Scores & Fixtures pages in their browser
2. Saves as "Webpage, HTML only" into the fbref_pages directory
3. This parser extracts all match results from the saved HTML
4. Data is stored in football_data.db with source='fbref'

The parsed data includes match dates, teams, scores, attendance, and referee
but **no betting odds** (FBref doesn't provide them). These leagues are for
match prediction only, not value betting.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from football_data.config import config

logger = logging.getLogger(__name__)

# FBref Scores & Fixtures tables have IDs matching ``results_{season}_{comp}_1``
# or ``sched_{season}_{comp}_1``. We match with a broad regex.

class FBrefParser:
    """Parser for locally saved FBref HTML pages.

    The user visits FBref.com, navigates to a competition's match
    schedule/results page, saves it as "Webpage, HTML only", then
    drops the file into the fbref_pages directory.

    Usage
    -----
    >>> parser = FBrefParser()
    >>> matches = parser.parse_saved_page("data/fbref_pages/superettan_2025.html")
    """

    def __init__(self) -> None:
        self.saved_dir = config.fbref.saved_pages_dir
        self.saved_dir.mkdir(parents=True, exist_ok=True)

    def parse_saved_page(self, html_path: str | Path) -> list[dict[str, Any]]:
        """Parse a saved FBref HTML page and extract match results.

        Uses ``:meth:`parse_scores_fixtures_html`` under the hood,
        with league auto-detection from the filename.

        Parameters
        ----------
        html_path : str | Path
            Path to the saved HTML file.

        Returns
        -------
        list[dict]
            List of match dicts with keys matching the DB schema.
        """
        path = Path(html_path)
        if not path.exists():
            raise FileNotFoundError(f"FBref HTML file not found: {path}")

        html = path.read_text(encoding="utf-8", errors="replace")
        league = self._guess_league(path.name)

        matches = self.parse_scores_fixtures_html(html, league)
        logger.info("Parsed %d matches from %s (league: %s)", len(matches), path.name, league)
        return matches

    def parse_all_saved_pages(self) -> dict[str, list[dict[str, Any]]]:
        """Parse all saved FBref HTML files in the configured directory.

        Returns
        -------
        dict[str, list[dict]]
            Maps league code to list of match dicts.
        """
        results: dict[str, list[dict[str, Any]]] = {}
        for fpath in sorted(self.saved_dir.glob("*.html")):
            try:
                matches = self.parse_saved_page(fpath)
                if matches:
                    league = matches[0].get("league", "unknown")
                    results.setdefault(league, []).extend(matches)
            except Exception as exc:
                logger.error("Failed to parse %s: %s", fpath.name, exc)

        return results

    @staticmethod
    def print_instructions() -> str:
        """Print step-by-step instructions for saving FBref pages."""
        return """
    How to collect FBref match data
    ------------------------------------

    1. Open Chrome and go to the Scores & Fixtures page for your league:
       Superettan:  https://fbref.com/en/comps/440/Superettan-Scores-and-Fixtures
       OBOS-ligaen: https://fbref.com/en/comps/438/OBOS-ligaen-Scores-and-Fixtures
       Ykkösliiga:  https://fbref.com/en/comps/441/Ykk%C3%B6sliiga-Scores-and-Fixtures
       1st Div:     https://fbref.com/en/comps/442/1st-Division-Scores-and-Fixtures
       I Liga:      https://fbref.com/en/comps/436/I-Liga-Scores-and-Fixtures

    2. Scroll down to see the full match results table.
       Make sure ALL seasons you want are visible (scroll down).

    3. Right-click anywhere on the page -> "Save As..."
       Format: "Webpage, HTML only"
       Save to: football_data/data/fbref_pages/
       File name: e.g. "superettan_2025.html"

    4. Run the parser:
       python -c "from football_data.collectors import FBrefParser;
       p = FBrefParser(); print(\\n.join(f'{k}: {len(v)} matches' for k,v in p.parse_all_saved_pages().items()))"

    5. Import into database:
       python -m football_data.scheduler.update_daily --source fbref
    """

    # ── Internal helpers ─────────────────────────────────

    @staticmethod
    def parse_scores_fixtures_html(
        html: str, league: str, season: str = ""
    ) -> list[dict[str, Any]]:
        """Parse match results from FBref Scores & Fixtures HTML.

        This parses raw HTML (from a saved page or string) and extracts
        all match rows from the schedule tables.

        Parameters
        ----------
        html : str
            Raw HTML content of the saved FBref page.
        league : str
            League code (e.g. ``"SE1"``, ``"NO2"``).
        season : str
            Season identifier (e.g. ``"2025"``).

        Returns
        -------
        list[dict]
            Match dicts ready for database insertion.
        """
        soup = BeautifulSoup(html, "html.parser")
        matches: list[dict[str, Any]] = []

        # Find all score tables (one per season in multi-season pages)
        tables = soup.find_all("table", id=re.compile(r"^(results|sched)_"))

        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                try:
                    match = FBrefParser._parse_data_stat_row(row, league, season)
                    if match:
                        matches.append(match)
                except Exception as exc:
                    logger.debug("Skipping unparseable row: %s", exc)
                    continue

        # Deduplicate
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, Any]] = []
        for m in matches:
            key = (m.get("date", ""), m.get("home_team", ""), m.get("away_team", ""))
            if key not in seen:
                seen.add(key)
                unique.append(m)

        return unique

    @staticmethod
    def _parse_data_stat_row(
        row: Any, league: str, season: str
    ) -> dict[str, Any] | None:
        """Parse a single <tr> using data-stat attributes (Scores & Fixtures format).

        Returns a match dict matching the DB schema, or None if invalid.
        """
        cells = row.find_all(["th", "td"])

        # Extract data-stat attributes from all cells
        row_data: dict[str, str] = {}
        for cell in cells:
            stat = cell.get("data-stat")
            if stat:
                text = cell.get_text(strip=True)
                text = text.replace("\xa0", " ").strip()  # &nbsp;
                if text:
                    row_data[stat] = text

        # Minimum fields to identify a match
        date_str = row_data.get("date", "")
        home = row_data.get("home_team", row_data.get("team", ""))
        away = row_data.get("away_team", "")
        score_str = row_data.get("score", "")

        if not date_str or not home or not away:
            return None

        # Parse score: "2–1" → home_goals=2, away_goals=1
        home_goals: int | None = None
        away_goals: int | None = None
        result: str | None = None

        if score_str:
            # Handle en-dash (–), em-dash (—), and hyphen (-)
            score_clean = score_str.replace("\u2013", "-").replace("\u2014", "-")
            if "-" in score_clean:
                parts = score_clean.split("-")
                if len(parts) == 2:
                    try:
                        hg = int(parts[0].strip())
                        ag = int(parts[1].strip())
                        home_goals = hg
                        away_goals = ag
                        if hg > ag:
                            result = "H"
                        elif hg < ag:
                            result = "A"
                        else:
                            result = "D"
                    except (ValueError, IndexError):
                        pass

        # Parse date
        parsed_date = FBrefParser._parse_date(date_str)

        # If no result (postponed/cancelled), mark goals as None
        if result is None:
            home_goals = None
            away_goals = None

        return {
            "source": "fbref",
            "league": league,
            "season": season or row_data.get("season", ""),
            "date": parsed_date,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "round": row_data.get("round", ""),
            "venue": row_data.get("venue", ""),
            "attendance": FBrefParser._safe_int(row_data.get("attendance")),
            "referee": row_data.get("referee", ""),
            "notes": row_data.get("notes", ""),
            "home_xg": FBrefParser._safe_float(row_data.get("xg_home")),
            "away_xg": FBrefParser._safe_float(row_data.get("xg_away")),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _parse_date(date_str: str) -> str | None:
        """Parse FBref date strings into YYYY-MM-DD format.

        Handles: ISO (2025-04-21), DD-MM-YYYY, DD/MM/YYYY, ISO+time.
        """
        if not date_str:
            return None
        date_str = date_str.strip().replace("\xa0", " ")
        for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Fallback: extract YYYY-MM-DD
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_str)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return date_str[:10] if len(date_str) >= 10 else None

    @staticmethod
    def _guess_league(filename: str) -> str:
        """Guess the league code from the filename."""
        fname = filename.lower()
        league_map = {
            "superettan": "SE1",
            "obos": "NO2",
            "ykkösliiga": "FI2",
            "ykkonen": "FI3",
            "premier": "IRL",
            "1st-division": "D2",
            "1-division": "D2",
            "i-liga": "P1",
            "poland": "P1",
        }
        for keyword, code in league_map.items():
            if keyword in fname:
                return code
        return "unknown"

    @staticmethod
    def _safe_float(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value.replace(",", ".").replace("\xa0", "").strip())
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _safe_int(value: str | None) -> int | None:
        """Safely parse an integer from a string."""
        if value is None:
            return None
        try:
            # Remove commas and non-numeric chars
            cleaned = "".join(c for c in value if c.isdigit())
            return int(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None
