"""Tests for the DataValidator processor."""

from football_data.processors.validation import DataValidator


class TestDataValidator:
    """Test the DataValidator class."""

    def setup_method(self):
        self.validator = DataValidator()

    def test_valid_record_passes(self):
        records = [{
            "source": "football-data.co.uk",
            "league": "SE1",
            "date": "2025-04-15",
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
            "home_goals": 2,
            "away_goals": 1,
            "result": "H",
            "home_odds": 2.5,
        }]
        report = self.validator.validate_matches(records)
        assert report["fatal"] == 0
        assert report["passed"] == 1

    def test_missing_required_field_fails(self):
        records = [{
            "source": "football-data.co.uk",
            # Missing league and date
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
        }]
        report = self.validator.validate_matches(records)
        assert report["fatal"] > 0
        assert report["passed"] == 0

    def test_invalid_result_fails(self):
        records = [{
            "source": "test",
            "league": "SE1",
            "date": "2025-04-15",
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
            "result": "X",
        }]
        report = self.validator.validate_matches(records)
        assert report["fatal"] > 0

    def test_negative_goals_fails(self):
        records = [{
            "source": "test",
            "league": "SE1",
            "date": "2025-04-15",
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
            "home_goals": -1,
            "away_goals": 0,
        }]
        report = self.validator.validate_matches(records)
        assert report["fatal"] > 0

    def test_impossible_odds_warning(self):
        records = [{
            "source": "test",
            "league": "SE1",
            "date": "2025-04-15",
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
            "home_odds": 0.5,
        }]
        report = self.validator.validate_matches(records)
        assert report["warnings"] > 0

    def test_result_score_mismatch_warning(self):
        records = [{
            "source": "test",
            "league": "SE1",
            "date": "2025-04-15",
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
            "home_goals": 1,
            "away_goals": 2,
            "result": "H",  # Should be A
        }]
        report = self.validator.validate_matches(records)
        assert report["warnings"] > 0

    def test_empty_team_name_fails(self):
        records = [{
            "source": "test",
            "league": "SE1",
            "date": "2025-04-15",
            "home_team": "IK Brage",
            "away_team": "",  # Empty
        }]
        report = self.validator.validate_matches(records)
        assert report["fatal"] > 0
