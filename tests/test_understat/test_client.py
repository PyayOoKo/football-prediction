"""
Unit tests for UnderstatClient — JSON extraction from HTML script tags.
"""

from __future__ import annotations

import pytest

from src.data_collection.sources.understat.client import UnderstatClient


# ── Sample HTML with embedded JSON in script tags ─────────

_HTML_WITH_TEAMS_DATA = """\
<html>
<body>
<script>
var teamsData = JSON.parse('[{"id":1,"title":"Arsenal"}]');
</script>
</body>
</html>
"""

_HTML_WITH_SHOTS_DATA = """\
<html>
<body>
<script>
var shotsData = JSON.parse('{"h":[],"a":[]}');
</script>
</body>
</html>
"""

_HTML_NO_DATA = """\
<html><body><p>No data here</p></body></html>
"""


class TestUnderstatClientExtraction:
    def test_extract_teams_data(self) -> None:
        """Extract teamsData from a simple HTML page."""
        data = UnderstatClient._extract_json(_HTML_WITH_TEAMS_DATA, "teamsData")
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["title"] == "Arsenal"

    def test_extract_shots_data(self) -> None:
        """Extract shotsData from a simple HTML page."""
        data = UnderstatClient._extract_json(_HTML_WITH_SHOTS_DATA, "shotsData")
        assert isinstance(data, dict)
        assert "h" in data
        assert "a" in data

    def test_extract_no_data(self) -> None:
        """Extracting from page without data returns empty dict."""
        data = UnderstatClient._extract_json(_HTML_NO_DATA, "teamsData")
        assert data == {}

    def test_extract_unknown_var(self) -> None:
        """Unknown variable name returns empty dict with warning."""
        data = UnderstatClient._extract_json(_HTML_WITH_TEAMS_DATA, "unknownVar")
        assert data == {}

    def test_extract_dates_data(self) -> None:
        """Extract datesData from HTML with embedded JSON."""
        html = """\
<html><body><script>
var datesData = JSON.parse('{"42":[]}');
</script></body></html>"""
        data = UnderstatClient._extract_json(html, "datesData")
        assert isinstance(data, dict)
        assert "42" in data

    def test_extract_with_unicode_escapes(self) -> None:
        """Extract JSON with unicode-escaped characters."""
        html = """\
<html><body><script>
var teamsData = JSON.parse('{"team":"Manchester\\\\u0020United"}');
</script></body></html>"""
        data = UnderstatClient._extract_json(html, "teamsData")
        # The extraction should handle unicode escapes
        assert "team" in data

    def test_extract_league_names(self) -> None:
        """Known league URLs work with the extraction pattern."""
        # Simulate EPL page structure
        html = """\
<html><body><script>
var teamsData = JSON.parse('{"1":{"title":"Arsenal","history":[{"season":"2024","xG":20.5}]}}');
</script></body></html>"""
        data = UnderstatClient._extract_json(html, "teamsData")
        assert isinstance(data, dict)
        assert "1" in data
        assert data["1"]["title"] == "Arsenal"
        assert data["1"]["history"][0]["xG"] == 20.5
