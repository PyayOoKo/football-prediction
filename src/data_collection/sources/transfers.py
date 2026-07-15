"""
Transfermarkt — transfer history scraper.

Scrapes per-season transfer data (signings, departures, net spend) for
national teams and clubs from Transfermarkt, and outputs DataFrames that
slot directly into the advanced feature pipeline.

Data source: https://www.transfermarkt.com/
Licence: Freely accessible data — respect robots.txt and rate limits.

Output columns
--------------
- ``team``               Team name (matches project convention)
- ``season``             Season code (e.g. "23/24")
- ``signings_count``     Number of players signed that window
- ``departures_count``   Number of players sold/released
- ``net_spend_meur``     Net spend in millions of Euros (positive = spent more)
- ``squad_churn_pct``    % of squad changed (in+out) / estimated squad size

Usage
-----
    from src.data_collection.sources.transfers import scrape_transfers

    df = scrape_transfers(team_names=["Brazil", "England"], max_windows=3)
    # df: pd.DataFrame with transfer data per team per window
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

TRANSFERS_PATH = "transfers/verein"
"""URL path for transfer history pages."""

REQUEST_TIMEOUT = 20
"""HTTP request timeout in seconds."""

CACHE_DIR_NAME = "external"
"""Subdirectory within data/ where transfer CSVs are stored."""


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


# ── Data structures ─────────────────────────────────────


@dataclass
class TransferWindow:
    """Aggregated transfer data for a single window (summer or winter)."""

    season: str                  # e.g. "23/24"
    window: str                  # "Summer" or "Winter"
    signings_count: int = 0
    departures_count: int = 0
    total_incoming_fees_meur: float = 0.0
    total_outgoing_fees_meur: float = 0.0

    @property
    def net_spend_meur(self) -> float:
        """Net spend (positive = club spent more than received)."""
        return self.total_incoming_fees_meur - self.total_outgoing_fees_meur

    @property
    def squad_churn_pct(self) -> float:
        """% of a nominal 25-player squad changed."""
        total = self.signings_count + self.departures_count
        return round((total / 25.0) * 100.0, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "window": self.window,
            "signings_count": self.signings_count,
            "departures_count": self.departures_count,
            "net_spend_meur": round(self.net_spend_meur, 2),
            "squad_churn_pct": self.squad_churn_pct,
        }


# ── Public API ──────────────────────────────────────────


def scrape_transfers(
    team_names: list[str],
    team_id_map: dict[str, int] | None = None,
    max_windows: int = 5,
    delay: float = 1.5,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Scrape transfer history for a list of teams.

    Parameters
    ----------
    team_names : list[str]
        List of team names (must match keys in ``team_id_map``).
    team_id_map : dict[str, int], optional
        Mapping from team name → Transfermarkt team ID.
        Defaults to ``TEAM_TO_TM_ID`` (same mapping as squad scraper).
    max_windows : int
        Maximum number of transfer windows to scrape per team (default 5).
        5 = ~2.5 years of transfer history.
    delay : float
        Seconds to wait between requests (default 1.5 — be polite).
    save_path : str, optional
        If provided, save the resulting DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Transfer data with columns ``team``, ``season``, ``window``,
        ``signings_count``, ``departures_count``, ``net_spend_meur``,
        ``squad_churn_pct``.
    """
    if team_id_map is None:
        from src.data_collection.sources.transfermarkt import TEAM_TO_TM_ID
        team_id_map = TEAM_TO_TM_ID

    all_records: list[dict[str, Any]] = []
    sess = _session()
    missed: list[str] = []

    for i, team in enumerate(team_names):
        tm_id = team_id_map.get(team)
        if tm_id is None:
            logger.warning("  [W] No Transfermarkt ID for '%s' — skipping", team)
            missed.append(team)
            continue

        slug = _team_slug(team)
        url = f"{TRANSFERMARKT_BASE}/{slug}/{TRANSFERS_PATH}/{tm_id}"

        logger.info("  [%d/%d] %s ...", i + 1, len(team_names), team)

        try:
            windows = _scrape_transfer_history(url, team, sess, max_windows)
            for w in windows:
                rec = w.to_dict()
                rec["team"] = team
                all_records.append(rec)
            logger.info("    -> %d windows", len(windows))
        except Exception as exc:
            logger.warning("    [W] Failed: %s", exc)
            missed.append(team)

        if i < len(team_names) - 1:
            time.sleep(delay)

    df = _build_dataframe(all_records)

    if missed:
        logger.warning("Teams with no transfer data: %s", ", ".join(missed))

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved %d transfer rows to %s", len(df), save_path)

    return df


def scrape_single_team(team_name: str, max_windows: int = 5) -> pd.DataFrame:
    """Scrape transfer history for a single team.

    Parameters
    ----------
    team_name : str
        Team name (must be in ``TEAM_TO_TM_ID``).
    max_windows : int
        Number of transfer windows to scrape (default 5).

    Returns
    -------
    pd.DataFrame
        Transfer data for the team.
    """
    return scrape_transfers(
        [team_name],
        max_windows=max_windows,
        delay=0,
    )


# ═══════════════════════════════════════════════════════════
#  Internal helpers — scraping
# ═══════════════════════════════════════════════════════════


def _team_slug(team: str) -> str:
    """Convert an openfootball team name to a Transfermarkt URL slug.

    Uses the same algorithm as the squad scraper (inlined to avoid
    importing a private function from another module).
    """
    import unicodedata
    slug = unicodedata.normalize("NFKD", team).encode("ascii", "ignore").decode()
    slug = slug.lower()
    slug = slug.replace(" & ", "-")
    slug = slug.replace(" and ", "-")
    slug = slug.replace(" ", "-")
    slug = slug.replace("'", "")
    slug = slug.replace(".", "")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _scrape_transfer_history(
    url: str,
    team_name: str,
    sess: requests.Session,
    max_windows: int,
) -> list[TransferWindow]:
    """Scrape a Transfermarkt transfer history page.

    Parses the transfer tables (arrivals and departures) for each
    displayed window and aggregates to ``TransferWindow`` objects.

    Parameters
    ----------
    url : str
        Full URL to the transfer history page.
    team_name : str
        Canonical team name.
    sess : requests.Session
        Reusable HTTP session.
    max_windows : int
        Maximum number of windows to parse (page shows ~5 by default).

    Returns
    -------
    list[TransferWindow]
        Aggregated transfer data for each window found.
    """
    resp = sess.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Transfermarkt organises transfers by window box.
    # Each window box contains two tables: arrivals and departures.
    window_boxes = soup.find_all("div", class_="tm-transfer-history__table")

    windows: list[TransferWindow] = []
    for box_div in window_boxes[:max_windows * 2]:  # *2 = arrivals + departures
        # Determine if this is arrivals or departures
        heading = box_div.find_previous(
            "h2", class_="tm-transfer-history__heading"
        )
        if heading:
            heading_text = heading.get_text(strip=True).lower()
        else:
            # Try the table header
            h3 = box_div.find_previous("h3")
            heading_text = h3.get_text(strip=True).lower() if h3 else ""

        # Determine season from window heading text
        # e.g. "23/24" or "2023"
        season_match = re.search(r"(\d{2}/\d{2}|\d{4})", heading_text)
        season = season_match.group(1) if season_match else "unknown"

        # Determine window type
        is_arrival = any(kw in heading_text for kw in ["arrivals", "incoming", "signings"])
        is_departure = any(kw in heading_text for kw in ["departures", "outgoing", "sold"])

        if not is_arrival and not is_departure:
            # Check for a sibling header
            prev_box = box_div.find_previous("div", class_="tm-transfer-history__table")
            if prev_box:
                prev_heading = prev_box.find_previous("h2", class_="tm-transfer-history__heading")
                if prev_heading:
                    heading_text = prev_heading.get_text(strip=True).lower()

        # Parse table rows
        table = box_div.find("table")
        if not table:
            continue

        rows = _parse_transfer_rows(table)
        n_transfers = len(rows)
        total_fees = sum(r["fee_meur"] for r in rows if r.get("fee_meur") and r["fee_meur"] > 0)

        # Map to window (use season modulo to match arrivals/departures)
        window_label = "Summer" if "summer" in heading_text or not "winter" in heading_text else "Winter"

        if is_arrival:
            windows.append(TransferWindow(
                season=season,
                window=window_label,
                signings_count=n_transfers,
                total_incoming_fees_meur=total_fees,
            ))
        elif is_departure:
            windows.append(TransferWindow(
                season=season,
                window=window_label,
                departures_count=n_transfers,
                total_outgoing_fees_meur=total_fees,
            ))

    # Merge arrivals + departures from same window
    merged = _merge_windows(windows)

    return merged[:max_windows]


def _parse_transfer_rows(table: Tag) -> list[dict[str, Any]]:
    """Parse a transfer table and return a list of transfer dicts.

    Each dict contains: player_name, from_team, to_team, fee_str, fee_meur.
    """
    tbody = table.find("tbody")
    if tbody is None:
        tbody = table

    rows = tbody.find_all("tr", recursive=False)
    transfers: list[dict[str, Any]] = []

    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue

        try:
            # Player name
            name_cell = cells[1] if len(cells) > 1 else cells[0]
            name_link = name_cell.find("a")
            player_name = name_link.get_text(strip=True) if name_link else ""

            # Fee
            fee_cell = cells[-1]
            fee_text = fee_cell.get_text(strip=True)
            fee_meur = _parse_transfer_fee(fee_text)

            transfers.append({
                "player_name": player_name,
                "fee_str": fee_text,
                "fee_meur": fee_meur,
            })
        except Exception:
            continue

    return transfers


def _parse_transfer_fee(text: str) -> float:
    """Parse a Transfermarkt transfer fee into millions of Euros.

    Examples
    --------
    >>> _parse_transfer_fee("€75.00m")
    75.0
    >>> _parse_transfer_fee("Free Transfer")
    0.0
    >>> _parse_transfer_fee("€1.2bn")
    1200.0
    >>> _parse_transfer_fee("Loan fee: €2m")
    2.0
    >>> _parse_transfer_fee("End of loan")
    0.0
    """
    text = text.strip()
    if not text or text in ("-", "—", ""):
        return 0.0

    # Check for free transfer indicators
    if any(kw in text.lower() for kw in ["free", "end of loan", "loan fee"]):
        # Still try to extract a loan fee if present
        fee_match = re.search(r"€?(\d+[.,]?\d*)\s*[mM]", text)
        if fee_match:
            return float(fee_match.group(1).replace(",", "."))
        return 0.0

    text = text.replace("€", "").replace(",", "").strip()

    try:
        if text.endswith("bn"):
            return float(text[:-2]) * 1000.0
        elif text.endswith("m"):
            return float(text[:-1])
        elif text.endswith("k"):
            return float(text[:-1]) / 1000.0
        else:
            return float(text)
    except (ValueError, TypeError):
        return 0.0


def _merge_windows(windows: list[TransferWindow]) -> list[TransferWindow]:
    """Merge arrival and departure TransferWindow entries for the same window.

    Transfermarkt presents arrivals and departures as separate boxes;
    this function combines them by (season, window) key.
    """
    merged_map: dict[tuple[str, str], TransferWindow] = {}

    for w in windows:
        key = (w.season, w.window)
        if key in merged_map:
            existing = merged_map[key]
            existing.signings_count += w.signings_count
            existing.departures_count += w.departures_count
            existing.total_incoming_fees_meur += w.total_incoming_fees_meur
            existing.total_outgoing_fees_meur += w.total_outgoing_fees_meur
        else:
            merged_map[key] = w

    # Sort by season descending (most recent first)
    # Parse "23/24" → start year 2023 for proper numeric sort
    def _season_sort_key(w: TransferWindow) -> tuple[int, str]:
        parts = w.season.split("/")
        start_year = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else 0
        return (start_year, w.window)

    return sorted(merged_map.values(), key=_season_sort_key, reverse=True)


# ═══════════════════════════════════════════════════════════
#  Data assembly helpers
# ═══════════════════════════════════════════════════════════


def _build_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of transfer records to a DataFrame.

    Ensures all expected columns exist, even if the list is empty.
    """
    if not records:
        return pd.DataFrame(columns=[
            "team", "season", "window",
            "signings_count", "departures_count",
            "net_spend_meur", "squad_churn_pct",
        ])

    df = pd.DataFrame(records)

    # Ensure numeric types
    for col in ["signings_count", "departures_count", "net_spend_meur", "squad_churn_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ═══════════════════════════════════════════════════════════
#  CLI for testing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    test_team = sys.argv[1] if len(sys.argv) > 1 else "Brazil"
    print(f"\n  Testing transfer scrape of {test_team}...\n")

    df = scrape_single_team(test_team, max_windows=3)
    print(f"  Scraped {len(df)} transfer windows for {test_team}:\n")
    if not df.empty:
        print(f"  {'Season':<10} {'Window':<10} {'Signings':<10} {'Departures':<12} {'Net (€m)':<12} {'Churn%':<8}")
        print(f"  {'-' * 62}")
        for _, r in df.iterrows():
            print(f"  {r['season']:<10} {r['window']:<10} "
                  f"{r['signings_count']:<10} {r['departures_count']:<12} "
                  f"{r['net_spend_meur']:<12.1f} {r['squad_churn_pct']:<8.1f}")
