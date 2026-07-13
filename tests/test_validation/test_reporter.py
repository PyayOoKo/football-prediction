"""
Tests for report generation — HTML, CSV, and JSON output.
"""

from __future__ import annotations

import json
import os
import tempfile

from src.validation.checks import check_duplicate_matches, check_missing_teams
from src.validation.engine import ValidationEngine
from src.validation.models import ValidationResult


class TestJSONReporter:
    def test_to_dict(self) -> None:
        engine = ValidationEngine(verbose=False)
        result = engine.run(
            [{"id": 1, "home_team": "A", "away_team": "B", "date": "2024-01-01"}],
            source_name="test",
        )
        d = result.to_dict()
        assert d["source_name"] == "test"
        assert "checks" in d
        assert "total_checks" in d

    def test_to_json_file(self) -> None:
        engine = ValidationEngine(verbose=False)
        result = engine.run(
            [{"id": 1, "home_team": "A", "away_team": "B", "date": "2024-01-01"}],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            result.to_json(f.name)
            f.flush()
            with open(f.name) as f_read:
                loaded = json.load(f_read)
            assert "checks" in loaded
            assert "total_checks" in loaded
        os.unlink(f.name)

    def test_to_json_string(self) -> None:
        engine = ValidationEngine(verbose=False)
        result = engine.run(
            [{"id": 1, "home_team": "A", "away_team": "B", "date": "2024-01-01"}],
        )
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        loaded = json.loads(json_str)
        assert "source_name" in loaded


class TestCSVReporter:
    def test_csv_export(self) -> None:
        engine = ValidationEngine(verbose=False)
        result = engine.run(
            [{"id": 1, "home_team": None, "away_team": "B", "date": "2024-01-01"}],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            result.to_csv(f.name)
            with open(f.name) as f_read:
                content = f_read.read()
            assert "check_name" in content
            assert "severity" in content
        os.unlink(f.name)


class TestHTMLReporter:
    def test_html_export(self) -> None:
        engine = ValidationEngine(verbose=False)
        result = engine.run(
            [{"id": 1, "home_team": "A", "away_team": "B", "date": "2024-01-01"}],
            source_name="test_source",
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            result.to_html(f.name)
            with open(f.name, encoding="utf-8") as f_read:
                content = f_read.read()
            assert "<!DOCTYPE html>" in content
            assert "Football Data Validation" in content
            assert "test_source" in content
        os.unlink(f.name)

    def test_html_shows_failures(self) -> None:
        """HTML report should show failure badges when checks fail."""
        dirty = [
            {"id": 1, "home_team": None, "away_team": "B", "date": "2024-01-01"},
        ]
        engine = ValidationEngine(verbose=False)
        result = engine.run(dirty)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            result.to_html(f.name)
            with open(f.name, encoding="utf-8") as f_read:
                content = f_read.read()
            assert "FAIL" in content or "fail" in content.lower()
        os.unlink(f.name)
