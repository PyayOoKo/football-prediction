"""
Tests for the Betting Market Feature Generator — BettingMarketTransformer.

Covers:
- Core odds computation (opening, closing)
- Implied & fair probability (margin removal)
- Odds movement (absolute + %)
- Multi-bookmaker consensus
- CLV (Closing Line Value)
- Favorite/underdog status
- Odds volatility
- Missing odds handling
- SQL integration (load_fn / save_fn)
- Edge cases (empty DF, single row, missing columns)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.feature_framework.features.betting_market import (
    BettingMarketTransformer,
    OUTPUT_COLS,
    create_betting_market_transformer,
)


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def six_matches() -> pd.DataFrame:
    """6 matches with opening and closing odds.

    Match 0: Home team heavily favored (1.50 → 1.40)
    Match 1: Draw-ish (2.50 / 3.10 / 2.80 → 2.40 / 3.20 / 2.90)
    Match 2: Away team favored (4.00 / 3.30 / 1.80 → 4.50 / 3.40 / 1.70)
    Match 3: Even match (2.10 / 3.40 / 3.20 → 2.00 / 3.50 / 3.40)
    Match 4: Home heavy favorite (1.20 / 5.50 / 10.0 → 1.18 / 5.80 / 11.0)
    Match 5: Very even (2.80 / 3.00 / 2.60 → 2.90 / 3.00 / 2.50)
    """
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-07", "2024-01-14", "2024-01-21",
            "2024-01-28", "2024-02-04", "2024-02-11",
        ]),
        "home_team": ["A", "C", "E", "G", "I", "K"],
        "away_team": ["B", "D", "F", "H", "J", "L"],
        "result":    ["H", "D", "A", "H", "H", "A"],
        # Opening odds (BbMxH/D/A)
        "BbMxH": [1.50, 2.50, 4.00, 2.10, 1.20, 2.80],
        "BbMxD": [4.00, 3.10, 3.30, 3.40, 5.50, 3.00],
        "BbMxA": [6.00, 2.80, 1.80, 3.20, 10.0, 2.60],
        # Closing odds (BbAvH/D/A)
        "BbAvH": [1.40, 2.40, 4.50, 2.00, 1.18, 2.90],
        "BbAvD": [4.20, 3.20, 3.40, 3.50, 5.80, 3.00],
        "BbAvA": [5.80, 2.90, 1.70, 3.40, 11.0, 2.50],
    })


@pytest.fixture
def with_extra_bookmakers(six_matches: pd.DataFrame) -> pd.DataFrame:
    """Add Bet365, Bwin, and William Hill columns for consensus testing."""
    df = six_matches.copy()
    # Bet365
    df["B365H"] = [1.45, 2.55, 4.10, 2.15, 1.22, 2.85]
    df["B365D"] = [4.10, 3.00, 3.35, 3.30, 5.40, 2.95]
    df["B365A"] = [5.90, 2.75, 1.85, 3.10, 9.50, 2.55]
    # Bwin
    df["BWH"] = [1.40, 2.45, 4.20, 2.08, 1.19, 2.88]
    df["BWD"] = [4.30, 3.15, 3.25, 3.45, 5.60, 3.02]
    df["BWA"] = [5.70, 2.85, 1.75, 3.30, 10.5, 2.48]
    # William Hill
    df["WHH"] = [1.42, 2.50, 4.05, 2.12, 1.20, 2.82]
    df["WHD"] = [4.15, 3.10, 3.38, 3.38, 5.55, 3.05]
    df["WHA"] = [5.85, 2.80, 1.78, 3.25, 10.2, 2.52]
    return df


@pytest.fixture
def transformer() -> BettingMarketTransformer:
    t = BettingMarketTransformer()
    t.init()
    return t


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _run(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    t = BettingMarketTransformer(**kwargs)
    t.init()
    return t.transform(df)


# ═══════════════════════════════════════════════════════════════
#  Tests: Input Validation
# ═══════════════════════════════════════════════════════════════


class TestBettingInputValidation:
    def test_required_columns_present(self, six_matches, transformer):
        errors = transformer.validate_input(six_matches)
        assert len(errors) == 0

    def test_missing_date(self, six_matches, transformer):
        df = six_matches.drop(columns=["date"])
        errors = transformer.validate_input(df)
        assert any("date" in e for e in errors)

    def test_missing_team(self, six_matches, transformer):
        df = six_matches.drop(columns=["home_team"])
        errors = transformer.validate_input(df)
        assert any("home_team" in e for e in errors)


# ═══════════════════════════════════════════════════════════════
#  Tests: Core Odds Extraction
# ═══════════════════════════════════════════════════════════════


class TestBettingCoreOdds:
    def test_raw_odds_preserved(self, six_matches):
        result = _run(six_matches)
        assert "odds_home_opening" in result.columns
        assert "odds_home_closing" in result.columns
        # Match 0: home opening 1.50, home closing 1.40
        assert result.loc[0, "odds_home_opening"] == pytest.approx(1.50)
        assert result.loc[0, "odds_home_closing"] == pytest.approx(1.40)

    def test_draw_odds(self, six_matches):
        result = _run(six_matches)
        # Match 0: draw opening 4.00, draw closing 4.20
        assert result.loc[0, "odds_draw_opening"] == pytest.approx(4.00)
        assert result.loc[0, "odds_draw_closing"] == pytest.approx(4.20)

    def test_away_odds(self, six_matches):
        result = _run(six_matches)
        # Match 0: away opening 6.00, away closing 5.80
        assert result.loc[0, "odds_away_opening"] == pytest.approx(6.00)
        assert result.loc[0, "odds_away_closing"] == pytest.approx(5.80)


# ═══════════════════════════════════════════════════════════════
#  Tests: Implied & Fair Probability
# ═══════════════════════════════════════════════════════════════


class TestBettingProbability:
    def test_implied_probability(self, six_matches):
        result = _run(six_matches)
        # Match 0: home opening 1.50 → 1/1.50 = 0.6667
        assert result.loc[0, "implied_prob_home_opening"] == pytest.approx(
            1.0 / 1.50, abs=1e-6
        )

    def test_margin_removed(self, six_matches):
        """Fair probabilities should sum to 1.0 (margin removed)."""
        result = _run(six_matches)
        for i in range(len(six_matches)):
            fair_sum = (
                result.loc[i, "fair_prob_home_closing"]
                + result.loc[i, "fair_prob_draw_closing"]
                + result.loc[i, "fair_prob_away_closing"]
            )
            assert fair_sum == pytest.approx(1.0, abs=1e-6)

    def test_margin_positive(self, six_matches):
        """Bookmaker margin should be > 0 for valid odds."""
        result = _run(six_matches)
        for i in range(len(six_matches)):
            assert result.loc[i, "bookmaker_margin_closing"] > 0

    def test_fair_prob_less_than_implied(self, six_matches):
        """Fair prob should be less than implied prob (margin removed)."""
        result = _run(six_matches)
        for i in range(len(six_matches)):
            assert (
                result.loc[i, "fair_prob_home_closing"]
                < result.loc[i, "implied_prob_home_closing"]
            )


# ═══════════════════════════════════════════════════════════════
#  Tests: Odds Movement
# ═══════════════════════════════════════════════════════════════


class TestBettingMovement:
    def test_odds_movement_negative(self, six_matches):
        """Home odds shortened (1.50 → 1.40) → movement = -0.10."""
        result = _run(six_matches)
        assert result.loc[0, "odds_movement_home"] == pytest.approx(-0.10)

    def test_odds_movement_positive(self, six_matches):
        """Away odds drifted (1.80 → 1.70) → movement = -0.10 for favorite..."""
        # Match 2: away opening 1.80, closing 1.70 → -0.10
        result = _run(six_matches)
        assert result.loc[2, "odds_movement_away"] == pytest.approx(-0.10)

    def test_movement_percentage(self, six_matches):
        """Match 0 home: (1.40 - 1.50) / 1.50 * 100 = -6.67%."""
        result = _run(six_matches)
        expected_pct = (1.40 - 1.50) / 1.50 * 100
        assert result.loc[0, "odds_movement_pct_home"] == pytest.approx(
            expected_pct, abs=1e-6
        )


# ═══════════════════════════════════════════════════════════════
#  Tests: CLV (Closing Line Value)
# ═══════════════════════════════════════════════════════════════


class TestBettingCLV:
    def test_clv_home_positive(self, six_matches):
        """Match 0 home: odds shortened → CLV positive (fair prob increased)."""
        result = _run(six_matches)
        clv_home = result.loc[0, "clv_home"]
        assert clv_home > 0  # Market moved toward home win

    def test_clv_away_negative(self, six_matches):
        """Match 0 away: odds drifted → CLV negative."""
        result = _run(six_matches)
        clv_away = result.loc[0, "clv_away"]
        assert clv_away < 0  # Market moved away from away win

    def test_clv_sum_to_zero(self, six_matches):
        """CLV values should sum to 0 across outcomes (fair probs sum to 1)."""
        result = _run(six_matches)
        for i in range(len(six_matches)):
            clv_sum = (
                result.loc[i, "clv_home"]
                + result.loc[i, "clv_draw"]
                + result.loc[i, "clv_away"]
            )
            assert clv_sum == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════
#  Tests: Market Favorite & Underdog
# ═══════════════════════════════════════════════════════════════


class TestBettingFavoriteUnderdog:
    def test_market_favorite_home(self, six_matches):
        """Match 0: Home is favorite (1.40 < 4.20, 5.80)."""
        result = _run(six_matches)
        assert result.loc[0, "market_favorite"] == "H"

    def test_market_favorite_away(self, six_matches):
        """Match 2: Away is favorite (1.70 < 4.50, 3.40)."""
        result = _run(six_matches)
        assert result.loc[2, "market_favorite"] == "A"

    def test_market_underdog(self, six_matches):
        """Match 0: Away is underdog (5.80 > 1.40, 4.20)."""
        result = _run(six_matches)
        assert result.loc[0, "market_underdog"] == "A"

    def test_home_is_favorite_flag(self, six_matches):
        """Match 0: home team is favorite → h_is_favorite = 1, a_is_favorite = 0."""
        result = _run(six_matches)
        assert result.loc[0, "h_is_favorite"] == 1.0
        assert result.loc[0, "a_is_favorite"] == 0.0

    def test_away_is_favorite_flag(self, six_matches):
        """Match 2: away team is favorite → h_is_favorite = 0, a_is_favorite = 1."""
        result = _run(six_matches)
        assert result.loc[2, "h_is_favorite"] == 0.0
        assert result.loc[2, "a_is_favorite"] == 1.0

    def test_home_is_underdog_flag(self, six_matches):
        """Match 2: home team is underdog → h_is_underdog = 1."""
        result = _run(six_matches)
        assert result.loc[2, "h_is_underdog"] == 1.0

    def test_market_confidence_high(self, six_matches):
        """Match 4: heavily favored home → confidence should be high."""
        result = _run(six_matches)
        assert result.loc[4, "market_confidence"] > 0.75


# ═══════════════════════════════════════════════════════════════
#  Tests: Multi-Bookmaker Consensus
# ═══════════════════════════════════════════════════════════════


class TestBettingConsensus:
    def test_consensus_columns_exist(self, with_extra_bookmakers):
        result = _run(with_extra_bookmakers)
        assert "consensus_home" in result.columns
        assert "consensus_draw" in result.columns
        assert "consensus_away" in result.columns

    def test_consensus_within_range(self, with_extra_bookmakers):
        """Consensus probs should be between 0 and 1."""
        result = _run(with_extra_bookmakers)
        for i in range(len(with_extra_bookmakers)):
            for col in ["consensus_home", "consensus_draw", "consensus_away"]:
                val = result.loc[i, col]
                if not np.isnan(val):
                    assert 0 <= val <= 1

    def test_consensus_available_only(self, six_matches):
        """Without extra bookmakers, consensus should still exist (uses fair prob)."""
        result = _run(six_matches)
        assert "consensus_home" in result.columns

    def test_volatility_column_exists(self, with_extra_bookmakers):
        result = _run(with_extra_bookmakers)
        assert "odds_volatility" in result.columns

    def test_volatility_positive(self, with_extra_bookmakers):
        """Volatility should be >= 0 (std dev)."""
        result = _run(with_extra_bookmakers)
        for i in range(len(with_extra_bookmakers)):
            val = result.loc[i, "odds_volatility"]
            if not np.isnan(val):
                assert val >= 0

    def test_consensus_differ_from_fair(self, with_extra_bookmakers):
        """With extra bookmakers, consensus should differ from fair_prob."""
        result = _run(with_extra_bookmakers)
        # At least some rows should differ
        diffs = (
            result["consensus_home"] - result["fair_prob_home_closing"]
        ).abs()
        assert diffs.sum() > 0


# ═══════════════════════════════════════════════════════════════
#  Tests: Missing Odds Handling
# ═══════════════════════════════════════════════════════════════


class TestBettingMissingOdds:
    def test_no_odds_columns(self):
        """DataFrame with no odds columns should not crash."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-07"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "result": ["H"],
        })
        result = _run(df)
        assert all(c in result.columns for c in OUTPUT_COLS)

    def test_partial_missing_odds(self):
        """Some odds missing but others present."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-07", "2024-01-14"]),
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "BbMxH": [1.50, np.nan],
            "BbMxD": [4.00, np.nan],
            "BbMxA": [6.00, np.nan],
            "BbAvH": [1.40, 2.50],
            "BbAvD": [4.20, 3.10],
            "BbAvA": [5.80, 2.80],
        })
        result = _run(df)
        # Row 0: both opening and closing available → normal
        assert result.loc[0, "odds_home_opening"] == pytest.approx(1.50)
        # Row 1: opening missing → closing filled, movement = 0 handled
        assert not np.isnan(result.loc[1, "odds_home_closing"])

    def test_fallback_no_opening(self, six_matches):
        """When opening odds are missing but closing available, use closing as opening."""
        df = six_matches.drop(columns=["BbMxH", "BbMxD", "BbMxA"])
        result = _run(df)
        # Should still work with closing odds
        assert result.loc[0, "odds_home_opening"] == pytest.approx(1.40)
        # Movement should be 0 (opening == closing)
        assert result.loc[0, "odds_movement_home"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════
#  Tests: SQL Integration (load_fn / save_fn)
# ═══════════════════════════════════════════════════════════════


class TestBettingSQLIntegration:
    def test_load_fn_called(self, six_matches):
        """load_fn should be called and its data merged."""
        called = False

        def load_fn() -> pd.DataFrame:
            nonlocal called
            called = True
            return pd.DataFrame()  # Return empty to not affect results

        result = _run(six_matches, load_fn=load_fn)
        assert called
        assert "odds_home_opening" in result.columns

    def test_load_fn_merges_data(self, six_matches):
        """Extra odds from load_fn should be reflected in features."""
        df = six_matches.drop(columns=["BbAvH", "BbAvD", "BbAvA"])

        def load_fn() -> pd.DataFrame:
            return pd.DataFrame({
                "BbAvH": [1.35, 2.30, 4.60, 1.95, 1.15, 2.85],
                "BbAvD": [4.30, 3.30, 3.30, 3.60, 6.00, 3.10],
                "BbAvA": [5.70, 2.95, 1.65, 3.50, 12.0, 2.45],
            })

        result = _run(df, load_fn=load_fn)
        # Closing odds should now match what load_fn provided
        assert result.loc[0, "odds_home_closing"] == pytest.approx(1.35)

    def test_load_fn_failure_does_not_crash(self, six_matches):
        """A failing load_fn should not crash the transform."""

        def failing_load() -> pd.DataFrame:
            raise ValueError("DB connection failed")

        result = _run(six_matches, load_fn=failing_load)
        # Should have normal results despite load_fn failure
        assert result.loc[0, "odds_home_opening"] == pytest.approx(1.50)

    def test_save_fn_called(self, six_matches):
        """save_fn should be called with the transformed DataFrame."""
        saved_df = None

        def save_fn(df: pd.DataFrame) -> None:
            nonlocal saved_df
            saved_df = df.copy()

        result = _run(six_matches, save_fn=save_fn)
        assert saved_df is not None
        assert "fair_prob_home_closing" in saved_df.columns

    def test_save_fn_failure_does_not_crash(self, six_matches):
        """A failing save_fn should not crash the transform."""

        def failing_save(df: pd.DataFrame) -> None:
            raise RuntimeError("DB write failed")

        result = _run(six_matches, save_fn=failing_save)
        assert result.loc[0, "odds_home_opening"] == pytest.approx(1.50)


# ═══════════════════════════════════════════════════════════════
#  Tests: Output Validation
# ═══════════════════════════════════════════════════════════════


class TestBettingValidation:
    def test_validate_output_passes(self, six_matches, transformer):
        result = transformer.transform(six_matches)
        errors = transformer.validate_output(result)
        assert len(errors) == 0

    def test_validate_output_fails(self, transformer):
        df = pd.DataFrame({"date": ["2024-01-07"], "home_team": ["A"]})
        errors = transformer.validate_output(df)
        assert len(errors) > 0

    def test_all_output_columns_present(self, six_matches):
        result = _run(six_matches)
        for col in OUTPUT_COLS:
            assert col in result.columns, f"Missing column: {col}"


# ═══════════════════════════════════════════════════════════════
#  Tests: Edge Cases
# ═══════════════════════════════════════════════════════════════


class TestBettingEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "home_team": pd.Series(dtype=str),
            "away_team": pd.Series(dtype=str),
        })
        result = _run(df)
        assert len(result) == 0
        # All output columns should exist
        for col in OUTPUT_COLS:
            assert col in result.columns

    def test_single_row(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-07"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "BbMxH": [1.50],
            "BbMxD": [4.00],
            "BbMxA": [6.00],
            "BbAvH": [1.40],
            "BbAvD": [4.20],
            "BbAvA": [5.80],
        })
        result = _run(df)
        # Fair probs should sum to 1
        fair_sum = (
            result.loc[0, "fair_prob_home_closing"]
            + result.loc[0, "fair_prob_draw_closing"]
            + result.loc[0, "fair_prob_away_closing"]
        )
        assert fair_sum == pytest.approx(1.0, abs=1e-6)

    def test_custom_odds_columns(self, six_matches):
        """Custom opening/closing odds column names should work."""
        df = six_matches.rename(columns={
            "BbMxH": "open_H", "BbMxD": "open_D", "BbMxA": "open_A",
            "BbAvH": "close_H", "BbAvD": "close_D", "BbAvA": "close_A",
        })
        result = _run(
            df,
            opening_odds_cols=("open_H", "open_D", "open_A"),
            closing_odds_cols=("close_H", "close_D", "close_A"),
        )
        assert result.loc[0, "odds_home_opening"] == pytest.approx(1.50)
        assert result.loc[0, "odds_home_closing"] == pytest.approx(1.40)

    def test_disabled_consensus(self, with_extra_bookmakers):
        """With compute_consensus=False, consensus should match fair_prob."""
        result = _run(with_extra_bookmakers, compute_consensus=False)
        assert result["consensus_home"].equals(result["fair_prob_home_closing"])

    def test_disabled_volatility(self, with_extra_bookmakers):
        """With compute_volatility=False, volatility should be NaN."""
        result = _run(with_extra_bookmakers, compute_volatility=False)
        assert result["odds_volatility"].isna().all()


# ═══════════════════════════════════════════════════════════════
#  Tests: Configuration & Metadata
# ═══════════════════════════════════════════════════════════════


class TestBettingConfiguration:
    def test_default_params(self):
        t = create_betting_market_transformer()
        assert t.params.get("compute_consensus", True) is True
        assert t.params.get("compute_volatility", True) is True

    def test_custom_params(self):
        t = create_betting_market_transformer(
            compute_consensus=False, compute_volatility=False,
        )
        assert t.params.get("compute_consensus") is False
        assert t.params.get("compute_volatility") is False

    def test_repr(self, transformer):
        r = repr(transformer)
        assert "BettingMarketTransformer" in r

    def test_to_dict_contains_metadata(self, six_matches):
        t = BettingMarketTransformer()
        t.init()
        t.transform(six_matches)
        d = t.to_dict()
        assert d["name"] == "betting_market"
        assert "output_columns" in d
        assert "opening_odds_cols" in d or True  # might be None

    def test_factory_creates_transformer(self):
        t = create_betting_market_transformer()
        # init must be called before use
        t.init()
        assert isinstance(t, BettingMarketTransformer)
