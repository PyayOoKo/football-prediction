"""
Referee statistics collector — scrapes referee data from FBref.

Collects per-referee performance statistics (cards, fouls, penalties)
for a given competition/season from FBref's referee summary pages.

Data source: https://fbref.com/en/comps/referees/

Output columns
--------------
- ``referee_name``           Referee's full name
- ``competition``            Competition name
- ``season``                 Season identifier
- ``matches_officiated``     Total matches officiated
- ``avg_home_cards_p90``     Avg home team cards per 90 min
- ``avg_away_cards_p90``     Avg away team cards per 90 min
- ``avg_total_cards_p90``    Avg total cards per 90 min
- ``avg_fouls_p90``          Avg fouls per 90 min
- ``avg_penalties_p90``      Avg penalties awarded per 90 min
- ``home_win_rate``          Home win rate in matches officiated

Usage
-----
    from src.data_collection.sources.referee_stats import scrape_referees

    df = scrape_referees(competition_id="9", season="2024-2025")
    # df: pd.DataFrame with per-referee statistics
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

FBREF_BASE = "https://fbref.com"
"""Base URL of FBref."""

REFEREES_PATH = "/en/comps/{competition_id}/referees/{competition_name}-Referee-Stats"
"""URL path pattern for referee stats pages."""

REQUEST_TIMEOUT = 20
"""HTTP request timeout in seconds."""

# Known competition name slugs for FBref URLs
COMPETITION_NAMES: dict[str, str] = {
    "9": "Premier-League",
    "12": "La-Liga",
    "11": "Serie-A",
    "20": "Bundesliga",
    "13": "Ligue-1",
    "8": "World-Cup",
    "19": "Eredivisie",
    "32": "Primeira-Liga",
    "26": "Champions-League",
}


# ── Session ─────────────────────────────────────────────


def _session() -> requests.Session:
    """Create a requests session with retry logic and polite headers."""
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=2.0, status_forcelist=[429, 502, 503, 504])
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


# ── Public API ──────────────────────────────────────────


def scrape_referees(
    competition_id: str = "9",
    season: str = "2024-2025",
    delay: float = 2.0,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Scrape referee statistics for a competition and season.

    Parameters
    ----------
    competition_id : str
        FBref competition ID (default ``9`` = Premier League).
    season : str
        Season slug (default ``2024-2025``).
    delay : float
        Seconds to wait (default 2.0 — be very polite to FBref).
    save_path : str, optional
        If provided, save the resulting DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Per-referee statistics for the requested competition/season.
    """
    comp_name = COMPETITION_NAMES.get(competition_id, "Premier-League")
    # Build season-specific URL: /en/comps/9/2024-2025/referees/Premier-League-Referee-Stats
    url = f"{FBREF_BASE}/en/comps/{competition_id}/{season}/referees/{comp_name}-Referee-Stats"

    logger.info("Fetching referee stats from %s", url)

    sess = _session()

    try:
        resp = sess.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        time.sleep(delay)  # Be polite

        df = _parse_referee_table(resp.text, competition_id, season)

    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning(
                "Referee stats not found for %s/%s (404) — "
                "trying without season in URL",
                competition_id, season,
            )
            # Try the generic referee page
            url = f"{FBREF_BASE}/en/comps/{competition_id}/referees/{comp_name}-Referee-Stats"
            resp = sess.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            df = _parse_referee_table(resp.text, competition_id, season)
        else:
            raise

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved %d referee rows to %s", len(df), save_path)

    return df


def scrape_match_referee(
    match_url: str,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Scrape referee info from a single match report page.

    Parameters
    ----------
    match_url : str
        Full FBref match URL.
    delay : float
        Seconds to wait (default 1.0).

    Returns
    -------
    dict
        Referee info: name, cards issued, fouls, etc.
    """
    logger.info("Fetching match referee from %s", match_url)

    sess = _session()
    resp = sess.get(match_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    time.sleep(delay)

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find referee name — typically in the scorebox area
    referee_info: dict[str, Any] = {
        "referee_name": "",
        "home_yellow": None,
        "away_yellow": None,
        "home_red": None,
        "away_red": None,
    }

    # Look for referee name in the match info section
    scorebox = soup.find("div", class_="scorebox")
    if scorebox:
        # Sometimes referee info is in a div with class "referee"
        ref_elem = scorebox.find(string=re.compile(r"Referee", re.I))
        if ref_elem:
            parent = ref_elem.find_parent()
            if parent:
                ref_text = parent.get_text(strip=True)
                # Remove "Referee:" prefix
                ref_text = re.sub(r"^Referee[:\s]*", "", ref_text, flags=re.I)
                referee_info["referee_name"] = ref_text.strip()

    # Look for referee in the match stats table
    if not referee_info["referee_name"]:
        for th in soup.find_all("th", string=re.compile(r"Referee", re.I)):
            td = th.find_next("td")
            if td:
                referee_info["referee_name"] = td.get_text(strip=True)
                break

    return referee_info


# ═══════════════════════════════════════════════════════════
#  Internal helpers — parsing
# ═══════════════════════════════════════════════════════════


def _parse_referee_table(
    html: str,
    competition_id: str,
    season: str,
) -> pd.DataFrame:
    """Parse the referee stats table from an FBref page.

    Parameters
    ----------
    html : str
        Raw HTML of the FBref page.
    competition_id : str
        FBref competition ID (for metadata).
    season : str
        Season slug (for metadata).

    Returns
    -------
    pd.DataFrame
        Parsed referee statistics.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the main stats table — usually has id "stats_referee"
    table = soup.find("table", id="stats_referee")
    if table is None:
        # Fallback: find any table with "referee" in the id or class
        table = soup.find("table", id=re.compile(r"referee", re.I))
    if table is None:
        # Try to find the first stats table on the page
        table = soup.find("table", class_="stats_table")
    if table is None:
        logger.warning("No referee stats table found on page")
        return pd.DataFrame()

    # Extract headers
    thead = table.find("thead")
    headers: list[str] = []
    if thead:
        header_rows = thead.find_all("tr")
        # FBref often has multi-row headers; take the last full row
        for tr in header_rows:
            cells = tr.find_all(["th", "td"])
            row_headers = []
            for cell in cells:
                # Skip partial headers that span multiple levels
                txt = cell.get_text(strip=True)
                if txt:
                    row_headers.append(txt)
            if row_headers:
                headers = row_headers  # Last non-empty row wins

    # Extract body rows
    tbody = table.find("tbody")
    if tbody is None:
        tbody = table

    rows = tbody.find_all("tr")
    data_rows: list[dict[str, Any]] = []

    for tr in rows:
        if "class" in tr.attrs and any(c in ["thead", "over_header"] for c in tr.get("class", [])):
            continue

        cells = tr.find_all(["td", "th"])
        row_data: dict[str, Any] = {}

        for i, cell in enumerate(cells):
            col_name = _normalise_column_name(
                headers[i] if i < len(headers) else f"col_{i}"
            )
            value = _clean_cell_value(cell)
            row_data[col_name] = value

        # Skip empty rows
        if any(v not in (None, "", "-") for v in row_data.values()):
            row_data["competition_id"] = competition_id
            row_data["season"] = season
            data_rows.append(row_data)

    if not data_rows:
        logger.warning("No referee data rows parsed")
        return pd.DataFrame()

    df = pd.DataFrame(data_rows)

    # Normalise key columns
    if "referee" in df.columns:
        df.rename(columns={"referee": "referee_name"}, inplace=True)
    if "matches" in df.columns:
        df.rename(columns={"matches": "matches_officiated"}, inplace=True)

    # Convert numeric columns
    num_cols = df.columns.difference(["referee_name", "competition", "season", "competition_id"])
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _normalise_column_name(raw: str) -> str:
    """Convert an FBref column header to a snake_case feature name."""
    import unicodedata
    name = unicodedata.normalize("NFKD", raw.strip().lower())
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.replace(" ", "_").replace("/", "_per_").replace("-", "_")
    name = name.replace("%", "pct")
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _clean_cell_value(cell: Any) -> Any:
    """Extract and clean a single table cell value."""
    # Check for embedded links (common in FBref tables)
    link = cell.find("a")
    if link and link.get_text(strip=True):
        return link.get_text(strip=True)

    text = cell.get_text(strip=True)
    if not text or text == "-":
        return 0.0

    # Try to convert to float
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return text


# ═══════════════════════════════════════════════════════════
#  CLI for testing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    comp = sys.argv[1] if len(sys.argv) > 1 else "9"
    season = sys.argv[2] if len(sys.argv) > 2 else "2024-2025"

    print(f"\n  Testing referee scrape for competition {comp}, season {season}...\n")
    df = scrape_referees(competition_id=comp, season=season, delay=0)

    if df.empty:
        print("  No referee data found.")
    else:
        print(f"  Found {len(df)} referees:\n")
        cols = ["referee_name", "matches_officiated"] if "matches_officiated" in df.columns else df.columns[:5].tolist()
        print(df[cols].to_string(index=False))
