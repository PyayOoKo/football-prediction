"""
Unit tests for FBrefTableParser — HTML comment extraction and table parsing.
"""

from __future__ import annotations

import pytest

from src.data_collection.sources.fbref.parser import FBrefTableParser
from src.data_collection.sources.fbref.models import StatCategory


# ── Sample HTML with a comment-embedded table ─────────────

_STANDARD_TABLE_HTML = """\
<html>
<body>
<!--
<table id="stats_standard">
<thead><tr><th data-stat="player">Player</th><th data-stat="goals">Goals</th><th data-stat="assists">Assists</th></tr></thead>
<tbody>
<tr><td>Erling Haaland</td><td>27</td><td>5</td></tr>
<tr><td>Bukayo Saka</td><td>16</td><td>9</td></tr>
</tbody>
</table>
-->
</body>
</html>
"""

_SHOOTING_TABLE_HTML = """\
<html>
<body>
<!--
<table id="stats_shooting">
<thead><tr><th data-stat="player">Player</th><th data-stat="sh_total">Shots Total</th><th data-stat="shots_on_target">Shots on Target</th><th data-stat="expected">xG</th></tr></thead>
<tbody>
<tr><td>Erling Haaland</td><td>89</td><td>42</td><td>28.4</td></tr>
<tr><td>Mohamed Salah</td><td>76</td><td>35</td><td>21.2</td></tr>
</tbody>
</table>
-->
</html>
"""

_EMPTY_TABLE_HTML = """\
<html>
<body>
<!--
<table id="stats_standard">
<thead><tr><th data-stat="player">Player</th><th data-stat="goals">Goals</th></tr></thead>
<tbody>
</tbody>
</table>
-->
</body>
</html>
"""

_MULTI_HEADER_TABLE = """\
<html>
<body>
<!--
<table id="stats_standard">
<thead>
<tr><th colspan="2">Performance</th><th>Expected</th></tr>
<tr><th data-stat="player">Player</th><th data-stat="goals">Goals</th><th data-stat="assists">Assists</th><th data-stat="expected">xG</th></tr>
</thead>
<tbody>
<tr><td>Player A</td><td>10</td><td>5</td><td>8.5</td></tr>
</tbody>
</table>
-->
</body>
</html>
"""


class TestFBrefTableParser:
    def test_parse_standard_table(self) -> None:
        """Parse a standard stats table with goals and assists."""
        parser = FBrefTableParser()
        tables = parser.parse_page(_STANDARD_TABLE_HTML)

        assert len(tables) == 1
        table = tables[0]
        assert table.category == StatCategory.STANDARD
        assert len(table.columns) == 3
        assert len(table.rows) == 2

        # Check data
        assert table.rows[0]["player_name"] == "Erling Haaland"
        assert table.rows[0]["goals"] == 27
        assert table.rows[0]["assists"] == 5

        assert table.rows[1]["player_name"] == "Bukayo Saka"

    def test_parse_shooting_table(self) -> None:
        """Parse a shooting stats table with numeric and float values."""
        parser = FBrefTableParser()
        tables = parser.parse_page(_SHOOTING_TABLE_HTML)

        assert len(tables) == 1
        table = tables[0]
        assert table.category == StatCategory.SHOOTING

        # Check xG is parsed as float
        assert table.rows[0]["xG"] == 28.4
        assert table.rows[0]["shots_total"] == 89

    def test_parse_empty_table(self) -> None:
        """Empty tables return no rows."""
        parser = FBrefTableParser()
        tables = parser.parse_page(_EMPTY_TABLE_HTML)

        assert len(tables) == 1
        table = tables[0]
        assert len(table.rows) == 0

    def test_parse_multi_header(self) -> None:
        """Multi-level headers use the last (most granular) row."""
        parser = FBrefTableParser()
        tables = parser.parse_page(_MULTI_HEADER_TABLE)

        assert len(tables) == 1
        table = tables[0]
        # Should use the last header row: player, goals, assists, xG
        assert "player_name" in table.columns or "player" in table.columns
        assert len(table.rows) == 1

    def test_extract_comments_disabled(self) -> None:
        """When comment extraction is disabled, no tables found."""
        parser = FBrefTableParser(extract_comments=False)
        tables = parser.parse_page(_STANDARD_TABLE_HTML)
        assert len(tables) == 0

    def test_detect_category(self) -> None:
        """Table IDs map to correct StatCategory."""
        assert FBrefTableParser._detect_category("stats_standard") == StatCategory.STANDARD
        assert FBrefTableParser._detect_category("stats_shooting") == StatCategory.SHOOTING
        assert FBrefTableParser._detect_category("stats_defense") == StatCategory.DEFENSE
        assert FBrefTableParser._detect_category("stats_keeper") == StatCategory.KEEPING
        assert FBrefTableParser._detect_category("unknown") == StatCategory.STANDARD

    def test_column_standardisation(self) -> None:
        """Column renames are applied correctly."""
        parser = FBrefTableParser()
        columns, rows = parser._apply_standardisation(
            ["player", "gls", "ast", "crdy"],
            [
                {"player": "P1", "gls": 10, "ast": 5, "crdy": 2},
            ],
        )
        assert columns == ["player_name", "goals", "assists", "yellow_cards"]
        assert rows[0]["player_name"] == "P1"
        assert rows[0]["goals"] == 10

    def test_filter_placeholders(self) -> None:
        """Rows with only 1-2 values are filtered out."""
        parser = FBrefTableParser()
        rows = [
            {"player_name": "Real Player", "goals": 10, "assists": 5},
            {"player_name": "Match Date", "goals": None, "assists": None},
            {"player_name": "", "goals": None},
        ]
        filtered = parser._filter_placeholder_rows(rows)
        assert len(filtered) == 1
        assert filtered[0]["player_name"] == "Real Player"
