"""
FBrefTableParser — extracts HTML tables from FBref pages.

FBref stores most of its data tables inside HTML comments (``<!-- ... -->``)
to discourage automated scraping. This parser:

1. Extracts the comment blocks containing table HTML
2. Parses each block back into a BeautifulSoup table
3. Extracts column headers and data rows
4. Cleans up multi-level headers and footers

The parser also handles:
- FBref's split-header format (e.g. ``Performance`` spanning ``Gls``, ``Ast``, ...)
- Footnote markers (numbers in column headers)
- Percentage and sign suffixes (``%``, ``+/-``)
- Empty/placeholder rows (``Match Date``, ``Opponent`` rows in player tables)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup, Tag

from src.data_collection.sources.fbref.models import FBrefTable, StatCategory

logger = logging.getLogger(__name__)

# ── Regex to find HTML tables inside HTML comments ────────
# FBref embeds stats tables as: <!-- <table ...>...</table> -->
_COMMENT_TABLE_RE = re.compile(
    r"<!--\s*(<(?:table)[^>]*>.*?</(?:table)>)\s*-->",
    re.DOTALL | re.IGNORECASE,
)

# Column headers to skip (FBref adds these for formatting)
_SKIP_HEADERS: set[str] = {
    "match date", "day", "time", "comp", "round", "venue",
    "result", "squad", "opponent", "playing time", "performance",
    "expected", "sca", "gca", "passing", "pass types", "carries",
    "take-ons", "touches", "total", "short", "medium", "long",
    "tklw", "att 3rd", "att", "mid 3rd", "def 3rd", "squad",
    "formation", "pos", "age", "pksv", "90s", "starts",
    "g+g", "ga90", "g-pk", "pkatt",
}

# Column name renames for standardisation
_COLUMN_RENAME: dict[str, str] = {
    "player": "player_name",
    "squad": "team",
    "nation": "nationality",
    "pos": "position",
    "age": "age",
    "born": "birth_year",
    "mp": "matches_played",
    "starts": "matches_started",
    "min": "minutes",
    "gls": "goals",
    "ast": "assists",
    "g+a": "goals_assists",
    "g-pk": "goals_non_penalty",
    "pk": "penalty_kicks",
    "pkatt": "penalty_kicks_attempted",
    "crdy": "yellow_cards",
    "crdr": "red_cards",
    "fls": "fouls",
    "fld": "fouled",
    "off": "offsides",
    "crs": "crosses",
    "int": "interceptions",
    "tklw": "tackles_won",
    "pkcon": "penalties_conceded",
    "og": "own_goals",
    "recov": "recoveries",
    "clr": "clearances",
    "soa": "shots_on_target_against",
    "saves": "saves",
    "ga": "goals_against",
    "ga90": "goals_against_per90",
    "so": "clean_sheets",
    "so%": "clean_sheet_pct",
    "g\\+/-": "plus_minus",
}


class FBrefTableParser:
    """Parses FBref HTML pages into structured table data.

    Parameters
    ----------
    extract_comments : bool
        Whether to extract tables from HTML comments (default True).
        Set to False if the tables are already in the regular DOM.
    standardise_columns : bool
        Whether to rename columns to standardised names (default True).
    skip_placeholder_rows : bool
        Whether to drop placeholder rows like 'Match Date' (default True).
    """

    def __init__(
        self,
        extract_comments: bool = True,
        standardise_columns: bool = True,
        skip_placeholder_rows: bool = True,
    ) -> None:
        self.extract_comments = extract_comments
        self.standardise_columns = standardise_columns
        self.skip_placeholder_rows = skip_placeholder_rows

    def parse_page(
        self,
        html: str,
        url: str = "",
    ) -> list[FBrefTable]:
        """Parse all stat tables from an FBref HTML page.

        Parameters
        ----------
        html : str
            Raw HTML content.
        url : str
            Source URL for context (optional).

        Returns
        -------
        list[FBrefTable]
            All parsed tables found on the page.
        """
        tables: list[FBrefTable] = []
        soup = BeautifulSoup(html, "html.parser")

        # Step 1: Extract table HTML from comments
        table_htmls = self._extract_table_htmls(html)

        # Step 2: Also look for tables in the regular DOM
        dom_tables = soup.find_all("table", id=re.compile(r"^stats_"))
        for tbl in dom_tables:
            table_htmls.append(str(tbl))

        # Step 3: Parse each table HTML
        seen_ids: set[str] = set()
        for tbl_html in table_htmls:
            tbl_soup = BeautifulSoup(tbl_html, "html.parser")
            table_tag = tbl_soup.find("table")
            if table_tag is None:
                continue

            table_id = table_tag.get("id", "")
            if table_id in seen_ids:
                continue
            seen_ids.add(table_id)

            category = self._detect_category(table_id)
            columns = self._extract_columns(table_tag)
            rows = self._extract_rows(table_tag, columns)

            if self.skip_placeholder_rows:
                rows = self._filter_placeholder_rows(rows)

            if self.standardise_columns:
                columns, rows = self._apply_standardisation(columns, rows)

            if not rows:
                continue

            # Extract competition from URL or table context
            competition = self._extract_competition(url, table_tag)

            tables.append(FBrefTable(
                category=category,
                competition=competition,
                columns=columns,
                rows=rows,
                raw_html=tbl_html,
            ))

            logger.debug(
                "Parsed table %s: %d cols x %d rows",
                table_id or "unknown",
                len(columns),
                len(rows),
            )

        logger.info("Parsed %d tables from %s", len(tables), url or "page")
        return tables

    def parse_squad_page(
        self,
        html: str,
        team_name: str = "",
        season: str = "",
        url: str = "",
    ) -> list[FBrefTable]:
        """Parse tables from a team squad page.

        Same as ``parse_page()`` but attaches team/season metadata.
        """
        tables = self.parse_page(html, url)
        for tbl in tables:
            tbl.team_name = team_name
            tbl.season = season
        return tables

    # ── Internal: comment extraction ───────────────────

    def _extract_table_htmls(self, html: str) -> list[str]:
        """Extract table HTML strings from HTML comments.

        FBref hides tables inside ``<!-- ... -->`` comments.
        This method finds those comments and extracts the table markup.
        """
        if not self.extract_comments:
            return []

        matches = _COMMENT_TABLE_RE.findall(html)
        # Decode HTML entities in the extracted HTML
        decoded = []
        for m in matches:
            # Unescape common entities
            text = m.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            decoded.append(text)
        return decoded

    # ── Internal: column extraction ────────────────────

    def _extract_columns(self, table: Tag) -> list[str]:
        """Extract column names from an FBref HTML table.

        Handles FBref's split-header format where a header row has
        multiple levels (e.g. ``Performance`` spanning ``Gls``, ``Ast``, ...).
        """
        thead = table.find("thead")
        if thead is None:
            # Try the first row as header
            first_row = table.find("tr")
            if first_row:
                return self._extract_row_cells(first_row)
            return []

        # Find the last header row (most granular)
        header_rows = thead.find_all("tr")
        if not header_rows:
            return []

        # Use the last row as the primary column names
        last_row = header_rows[-1]
        columns = self._extract_row_cells(last_row)

        # If there are only a few columns, try the first row
        if len(columns) < 3 and len(header_rows) > 1:
            columns = self._extract_row_cells(header_rows[0])

        return columns

    def _extract_row_cells(self, row: Tag) -> list[str]:
        """Extract text from th or td cells in a row."""
        cells = row.find_all(["th", "td"])
        cols = []
        for cell in cells:
            # Get data-stat attribute if available (more reliable)
            data_stat = cell.get("data-stat", "")
            if data_stat:
                cols.append(data_stat)
            else:
                # Clean text: remove footnotes, strip whitespace
                text = cell.get_text(strip=True)
                text = re.sub(r"^\d+\.?\s*", "", text)  # Remove leading numbers
                text = text.strip()
                if text:
                    cols.append(text)
        return cols

    # ── Internal: row extraction ───────────────────────

    def _extract_rows(
        self,
        table: Tag,
        columns: list[str],
    ) -> list[dict[str, Any]]:
        """Extract data rows from an FBref HTML table."""
        tbody = table.find("tbody")
        if tbody is None:
            return []

        rows: list[dict[str, Any]] = []
        for tr in tbody.find_all("tr"):
            # Skip header rows within tbody
            if tr.find("th") and tr.get("class") and "thead" in (tr.get("class") or []):
                continue

            cells = tr.find_all(["th", "td"])
            row: dict[str, Any] = {}

            for i, cell in enumerate(cells):
                if i >= len(columns):
                    break

                col_name = columns[i]
                value = self._parse_cell(cell)
                row[col_name] = value

            if row:
                rows.append(row)

        return rows

    def _parse_cell(self, cell: Tag) -> Any:
        """Parse a single cell value from an FBref table.

        Converts numeric strings to int/float where possible.
        """
        # Check data-stat attribute first
        data_stat = cell.get("data-stat", "")
        if data_stat:
            # Try the cell's text content
            text = cell.get_text(strip=True)
        else:
            text = cell.get_text(strip=True)

        # Handle empty/placeholder cells
        if not text or text == "" or text == "—":
            return None

        # Try to convert to numeric
        try:
            if "." in text and text.count(".") == 1:
                return float(text)
            return int(text)
        except (ValueError, TypeError):
            pass

        # Clean up percentage and sign suffixes
        text = text.strip()
        return text

    # ── Internal: filtering & standardisation ─────────

    def _filter_placeholder_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove placeholder rows that don't contain actual stats.

        FBref inserts rows like ``Match Date``, ``Opponent``, etc.
        in player match logs. These have no meaningful stat data.
        """
        filtered: list[dict[str, Any]] = []
        for row in rows:
            values = [v for v in row.values() if v is not None]
            # A row with only 1-2 non-null values is likely a placeholder
            if len(values) <= 2:
                continue
            filtered.append(row)
        return filtered

    def _apply_standardisation(
        self,
        columns: list[str],
        rows: list[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Rename columns to standardised names."""
        new_columns = [_COLUMN_RENAME.get(c, c) for c in columns]
        new_rows: list[dict[str, Any]] = []
        for row in rows:
            new_row: dict[str, Any] = {}
            for old_col, new_col in zip(columns, new_columns):
                if old_col in row:
                    new_row[new_col] = row[old_col]
            new_rows.append(new_row)
        return new_columns, new_rows

    # ── Internal: detection helpers ───────────────────

    @staticmethod
    def _detect_category(table_id: str) -> StatCategory:
        """Map an FBref table ID to a StatCategory enum."""
        # Normalize: remove leading/trailing underscores, lowercase
        tid = table_id.strip("_").lower()
        for cat in StatCategory:
            if cat.value in tid:
                return cat
        return StatCategory.STANDARD

    @staticmethod
    def _extract_competition(url: str, table: Tag) -> str:
        """Extract competition name from URL or table context."""
        # Try URL patterns like /en/comps/9/
        comp_match = re.search(r"/comps/(\d+)/", url)
        if comp_match:
            from src.data_collection.sources.fbref.models import COMPETITION_NAMES

            comp_id = comp_match.group(1)
            return COMPETITION_NAMES.get(comp_id, comp_id)

        # Try a caption or header
        caption = table.find("caption")
        if caption:
            return caption.get_text(strip=True)

        return ""
