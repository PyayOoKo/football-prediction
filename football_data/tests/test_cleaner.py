"""Tests for the DataCleaner processor."""

from football_data.processors.clean import DataCleaner, normalise_team_name


class TestTeamNormalisation:
    """Test team name normalisation."""

    def test_swedish_teams(self):
        assert normalise_team_name("Brage") == "IK Brage"
        assert normalise_team_name("degerfors") == "Degerfors IF"
        assert normalise_team_name("OREBRO") == "Örebro SK"

    def test_norwegian_teams(self):
        assert normalise_team_name("Aalesund") == "Aalesunds FK"
        assert normalise_team_name("bryne") == "Bryne FK"
        assert normalise_team_name("Start") == "IK Start"

    def test_finnish_teams(self):
        assert normalise_team_name("FF Jaro") == "FF Jaro"
        assert normalise_team_name("KTP") == "KTP"
        assert normalise_team_name("jippo") == "JIPPO"

    def test_irish_teams(self):
        assert normalise_team_name("Shamrock Rovers") == "Shamrock Rovers"
        assert normalise_team_name("shamrock") == "Shamrock Rovers"
        assert normalise_team_name("St Patricks") == "St Patrick's Athletic"

    def test_polish_teams(self):
        assert normalise_team_name("Arka Gdynia") == "Arka Gdynia"
        assert normalise_team_name("wisla krakow") == "Wisła Kraków"
        assert normalise_team_name("LKS Lodz") == "ŁKS Łódź"

    def test_danish_teams(self):
        assert normalise_team_name("Aalborg") == "Aalborg BK"
        assert normalise_team_name("copenhagen") == "FC København"
        assert normalise_team_name("Viborg") == "Viborg FF"

    def test_unknown_team_preserved(self):
        """Unknown teams should be title-cased."""
        assert normalise_team_name("real madrid") == "Real Madrid"
        assert normalise_team_name("my team") == "My Team"


class TestDataCleaner:
    """Test the DataCleaner class."""

    def setup_method(self):
        self.cleaner = DataCleaner()

    def test_clean_basic_record(self):
        records = [{
            "home_team": "brage",
            "away_team": "degerfors",
            "league": "SE1",
            "date": "01/04/2025",
            "source": "football-data.co.uk",
            "home_goals": "2",
            "away_goals": "1",
        }]
        cleaned = self.cleaner.clean_matches(records)
        assert len(cleaned) == 1
        assert cleaned[0]["home_team"] == "IK Brage"
        assert cleaned[0]["away_team"] == "Degerfors IF"
        assert cleaned[0]["home_goals"] == 2
        assert cleaned[0]["away_goals"] == 1
        assert cleaned[0]["result"] == "H"

    def test_draw_result_computed(self):
        records = [{
            "home_team": "FF Jaro",
            "away_team": "KTP",
            "league": "FI2",
            "date": "2025-04-15",
            "source": "football-data.co.uk",
            "home_goals": 1,
            "away_goals": 1,
        }]
        cleaned = self.cleaner.clean_matches(records)
        assert cleaned[0]["result"] == "D"

    def test_duplicate_removal(self):
        records = [
            {
                "home_team": "IK Brage",
                "away_team": "Degerfors IF",
                "league": "SE1",
                "date": "2025-04-15",
                "source": "football-data.co.uk",
            },
            {
                "home_team": "IK Brage",
                "away_team": "Degerfors IF",
                "league": "SE1",
                "date": "2025-04-15",
                "source": "football-data.co.uk",
            },
        ]
        cleaned = self.cleaner.clean_matches(records)
        assert len(cleaned) == 1

    def test_empty_record_skipped(self):
        records = [
            {"home_team": "IK Brage", "away_team": "Degerfors IF", "league": "SE1", "date": "2025-04-15", "source": "test"},
            {},
            {"home_team": "", "away_team": "Test", "league": "SE1", "date": "2025-04-15", "source": "test"},
        ]
        cleaned = self.cleaner.clean_matches(records)
        assert len(cleaned) == 1

    def test_date_normalisation(self):
        records = [{
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
            "league": "SE1",
            "date": "01/04/2025",
            "source": "test",
        }]
        cleaned = self.cleaner.clean_matches(records)
        assert cleaned[0]["date"] == "2025-04-01"
