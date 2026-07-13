"""
Unit tests for the CSVParser — football-data.co.uk CSV parsing.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.importers.parser import CSVParser, _four_digit_to_season_name


# ── Fixtures ──────────────────────────────────────────────

SIMPLE_CSV = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
E0,07/01/24,Arsenal,Chelsea,2,1,H
E0,07/01/24,Liverpool,Manchester City,0,0,D
E0,08/01/24,Manchester United,Tottenham,1,2,A
"""

CSV_WITH_STATS = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR
E0,14/01/24,Arsenal,Chelsea,3,1,H,15,8,6,3,7,4,10,12,2,3,0,0
"""

CSV_WITH_ODDS = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,BbAvH,BbAvD,BbAvA
E0,21/01/24,Arsenal,Chelsea,2,0,H,1.50,4.00,6.50,1.53,4.10,6.20
"""

CSV_INVALID_DATE = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
E0,not-a-date,Arsenal,Chelsea,2,1,H
"""

CSV_MISSING_TEAM = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
E0,07/01/24,,Chelsea,2,1,H
E0,07/01/24,Liverpool,,0,0,D
"""

CSV_EMPTY = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
"""

CSV_WITH_XG = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,hxG,axG,hs,as
E0,14/01/24,Arsenal,Chelsea,3,1,H,2.1,0.8,15,8
"""


# ── _four_digit_to_season_name ───────────────────────────


class TestSeasonNameConversion:
    def test_standard_season(self) -> None:
        assert _four_digit_to_season_name("2425") == "2024/2025"

    def test_old_season(self) -> None:
        assert _four_digit_to_season_name("9394") == "1993/1994"

    def test_current_season(self) -> None:
        assert _four_digit_to_season_name("2526") == "2025/2026"

    def test_unknown_code(self) -> None:
        assert _four_digit_to_season_name("abc") == "abc"


# ── CSVParser ─────────────────────────────────────────────


class TestCSVParser:
    def test_parse_simple(self) -> None:
        """Basic parsing of a minimal CSV with 3 matches."""
        parser = CSVParser()
        rows = parser.parse_to_dicts(SIMPLE_CSV)

        assert len(rows) == 3

        # First row
        row0 = rows[0]
        assert row0["home_team"] == "Arsenal"
        assert row0["away_team"] == "Chelsea"
        assert row0["home_goals"] == 2
        assert row0["away_goals"] == 1
        assert row0["result"] == "H"
        assert row0["match_date"] == date(2024, 1, 7)
        assert row0["league"] == "E0"

        # Second row - draw
        row1 = rows[1]
        assert row1["home_team"] == "Liverpool"
        assert row1["away_team"] == "Manchester City"
        assert row1["home_goals"] == 0
        assert row1["away_goals"] == 0
        assert row1["result"] == "D"

        # Third row - away win
        row2 = rows[2]
        assert row2["result"] == "A"

    def test_parse_with_stats(self) -> None:
        """Parsing CSV with match statistics columns."""
        parser = CSVParser()
        rows = parser.parse_to_dicts(CSV_WITH_STATS)

        assert len(rows) == 1
        row = rows[0]
        assert row["home_shots"] == 15
        assert row["away_shots"] == 8
        assert row["home_shots_target"] == 6
        assert row["away_shots_target"] == 3
        assert row["home_corners"] == 7
        assert row["away_corners"] == 4
        assert row["home_fouls"] == 10
        assert row["away_fouls"] == 12
        assert row["home_yellow"] == 2
        assert row["away_yellow"] == 3
        assert row["home_red"] == 0
        assert row["away_red"] == 0

    def test_parse_with_odds(self) -> None:
        """Parsing CSV with odds columns preserved."""
        parser = CSVParser()
        rows = parser.parse_to_dicts(CSV_WITH_ODDS)

        assert len(rows) == 1
        row = rows[0]

        # Odds columns should be preserved as float values
        assert "B365H" in row or "b365h" in row
        # Check one of them
        odds_keys = [k for k in row if k.lower().startswith("b365")]
        assert len(odds_keys) == 3
        # The odds values should be floats
        home_odds = row.get("B365H") or row.get("b365h")
        assert home_odds == 1.50
        assert row.get("b365d") == 4.00
        assert row.get("b365a") == 6.50

    def test_parse_invalid_date(self) -> None:
        """Row with invalid date should have None match_date and be excluded."""
        parser = CSVParser()
        rows = parser.parse_to_dicts(CSV_INVALID_DATE)

        assert len(rows) == 0  # Invalid date → validation fails → excluded

    def test_parse_missing_team(self) -> None:
        """Rows with missing team names should be excluded."""
        parser = CSVParser()
        rows = parser.parse_to_dicts(CSV_MISSING_TEAM)

        assert len(rows) == 0  # Both rows have missing teams

    def test_parse_empty_csv(self) -> None:
        """Empty CSV (header-only) should return empty list."""
        parser = CSVParser()
        rows = parser.parse_to_dicts(CSV_EMPTY)
        assert len(rows) == 0

    def test_parse_season_override(self) -> None:
        """Season override should be applied to all rows."""
        parser = CSVParser(season_override="2024/2025")
        rows = parser.parse_to_dicts(SIMPLE_CSV)
        assert len(rows) == 3
        for row in rows:
            assert row.get("season") == "2024/2025"

    def test_parse_league_override(self) -> None:
        """League override should be applied to all rows."""
        parser = CSVParser(league_override="E0")
        rows = parser.parse_to_dicts(SIMPLE_CSV)
        assert len(rows) == 3
        for row in rows:
            assert row.get("league") == "E0"

    def test_parse_with_xg(self) -> None:
        """Parsing CSV with expected goals columns."""
        parser = CSVParser()
        rows = parser.parse_to_dicts(CSV_WITH_XG)

        assert len(rows) == 1
        row = rows[0]
        assert row["home_xg"] == 2.1
        assert row["away_xg"] == 0.8
        assert row["home_shots"] == 15
        assert row["away_shots"] == 8

    def test_parse_strict_mode(self) -> None:
        """Strict mode raises on missing required columns."""
        bad_csv = "SomeCol,AnotherCol\n1,2\n"
        parser = CSVParser(strict=True)
        with pytest.raises(ValueError, match="Missing required columns"):
            parser.parse_to_dicts(bad_csv)

    def test_parse_non_strict_mode(self) -> None:
        """Non-strict mode logs warning but continues."""
        bad_csv = "SomeCol,AnotherCol\n1,2\n"
        parser = CSVParser(strict=False)
        rows = parser.parse_to_dicts(bad_csv)
        assert len(rows) == 0  # No valid data without required cols

    def test_raw_parsed_rows(self) -> None:
        """Test that ParsedRow objects have correct properties."""
        parser = CSVParser()
        parsed_rows = parser.parse_raw(SIMPLE_CSV)
        assert len(parsed_rows) == 3
        for pr in parsed_rows:
            assert hasattr(pr, "raw")
            assert hasattr(pr, "standardised")
            assert hasattr(pr, "errors")
            assert hasattr(pr, "warnings")
            assert pr.valid is True  # All rows valid
            to_dict = pr.to_dict()
            assert "_row" in to_dict
            assert "_errors" in to_dict

    def test_parse_date_formats(self) -> None:
        """Test different date formats are parsed correctly."""
        parser = CSVParser()

        # DD/MM/YY
        d1 = parser._parse_date("07/01/24")
        assert d1 == date(2024, 1, 7)

        # DD/MM/YYYY
        d2 = parser._parse_date("07/01/2024")
        assert d2 == date(2024, 1, 7)

        # YYYY-MM-DD
        d3 = parser._parse_date("2024-01-07")
        assert d3 == date(2024, 1, 7)

        # None/empty
        assert parser._parse_date("") is None
        assert parser._parse_date("None") is None

        # Invalid
        assert parser._parse_date("not-a-date") is None
