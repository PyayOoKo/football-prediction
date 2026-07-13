"""
Betting Market Feature Generator — opening odds, closing odds, implied
probability, odds movement, consensus, CLV, favorite/underdog, volatility.

Transforms raw bookmaker decimal odds (from football-data.co.uk or any
source) into a rich set of market-derived features with overround removal,
multi-bookmaker consensus, and missing-odds resilience.

Features
--------
+---------------------------+----------------------------------------------------+
| Feature                   | Description                                        |
+===========================+====================================================+
| Opening odds (H/D/A)      | Raw decimal odds at market open                    |
| Closing odds (H/D/A)      | Raw decimal odds at market close (kick-off)        |
| Implied probability       | 1 / decimal_odds (includes bookmaker margin)       |
| Fair probability          | Implied prob with bookmaker margin removed         |
| Odds movement             | closing_odds - opening_odds (absolute + %)         |
| Market consensus          | Mean fair probability across all available books   |
| CLV reference             | fair_prob_closing - fair_prob_opening              |
| Favorite status           | Which team is the favorite at closing odds         |
| Underdog status           | Which team is the underdog at closing odds         |
| Odds volatility           | Std dev of fair probabilities across bookmakers    |
| Bookmaker margin          | Overround = sum(implied) - 1 (opening & closing)   |
+---------------------------+----------------------------------------------------+

Requirements coverage
---------------------
- **Remove margin**: multiplicative method (fair = implied / (1 + margin))
- **Time-aware**: DataFrames sorted chronologically before computation
- **Multiple bookmakers**: Consensus + volatility across all available sets
- **Missing odds**: Graceful NaN propagation, no crashes on sparse data
- **SQL storage**: Optional ``load_fn`` / ``save_fn`` for DB integration
- **Validation**: Standard ``validate_input()`` / ``validate_output()``
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.feature_framework.base import FeatureTransformer
from src.feature_framework.models import TransformContext

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

# Default column sets for football-data.co.uk format
_DEFAULT_OPENING: tuple[str, str, str] = ("BbMxH", "BbMxD", "BbMxA")
_DEFAULT_CLOSING: tuple[str, str, str] = ("BbAvH", "BbAvD", "BbAvA")

# Fallback: Bet365 (very commonly available)
_FALLBACK_CLOSING: tuple[str, str, str] = ("B365H", "B365D", "B365A")

# Known bookmaker column triplets for consensus computation
_BOOKMAKER_SETS: list[tuple[str, str, str]] = [
    ("BbAvH", "BbAvD", "BbAvA"),  # BetBrain average (closing consensus)
    ("B365H", "B365D", "B365A"),  # Bet365
    ("BWH", "BWD", "BWA"),        # Bet&Win
    ("IWH", "IWD", "IWA"),        # Interwetten
    ("LBH", "LBD", "LBA"),        # Ladbrokes
    ("SBH", "SBD", "SBA"),        # Sportingbet
    ("WHH", "WHD", "WHA"),        # William Hill
    ("SJH", "SJD", "SJA"),        # Stan James
    ("VCH", "VCD", "VCA"),        # VC Bet
]

_OUTCOME_LABELS = ["H", "D", "A"]

# ── Output column names ─────────────────────────────────

OUTPUT_COLS = [
    # Raw odds
    "odds_home_opening", "odds_draw_opening", "odds_away_opening",
    "odds_home_closing", "odds_draw_closing", "odds_away_closing",
    # Implied probability (raw, includes margin)
    "implied_prob_home_opening", "implied_prob_draw_opening",
    "implied_prob_away_opening",
    "implied_prob_home_closing", "implied_prob_draw_closing",
    "implied_prob_away_closing",
    # Fair probability (margin removed)
    "fair_prob_home_opening", "fair_prob_draw_opening",
    "fair_prob_away_opening",
    "fair_prob_home_closing", "fair_prob_draw_closing",
    "fair_prob_away_closing",
    # Odds movement
    "odds_movement_home", "odds_movement_draw", "odds_movement_away",
    "odds_movement_pct_home", "odds_movement_pct_draw",
    "odds_movement_pct_away",
    # CLV (change in fair probability)
    "clv_home", "clv_draw", "clv_away",
    # Market structure
    "market_favorite", "market_underdog",
    "market_confidence",
    # Consensus (across bookmakers)
    "consensus_home", "consensus_draw", "consensus_away",
    # Volatility
    "odds_volatility",
    # Margin
    "bookmaker_margin_opening", "bookmaker_margin_closing",
    # Team-level indicators
    "h_is_favorite", "a_is_favorite",
    "h_is_underdog", "a_is_underdog",
]


# ═══════════════════════════════════════════════════════════════
#  BettingMarketTransformer
# ═══════════════════════════════════════════════════════════════


class BettingMarketTransformer(FeatureTransformer):
    """Generate betting market features from raw decimal odds.

    Processes opening and closing odds from one or more bookmakers,
    removes overround to compute fair probabilities, and derives
    movement, CLV, consensus, volatility, and favorite/underdog status.

    Parameters
    ----------
    opening_odds_cols : tuple[str, str, str], optional
        Column names for opening odds ``(home, draw, away)``.
        Default: ``(\"BbMxH\", \"BbMxD\", \"BbMxA\")``.
    closing_odds_cols : tuple[str, str, str], optional
        Column names for closing odds ``(home, draw, away)``.
        Default: ``(\"BbAvH\", \"BbAvD\", \"BbAvA\")``.
    bookmaker_sets : list[tuple[str, str, str]], optional
        Additional bookmaker column triplets for consensus computation.
        Will be auto-detected from available columns if not provided.
    compute_consensus : bool
        If True, compute consensus fair probabilities across bookmakers
        (default True).
    compute_volatility : bool
        If True, compute odds volatility (std across bookmakers).
        Default True.
    load_fn : Callable | None
        Optional function signature ``(season, league) -> pd.DataFrame``
        that returns historical odds data from SQL.  Called once during
        ``transform()`` and merged before computation.
    save_fn : Callable | None
        Optional function signature ``(df) -> None`` that stores computed
        features back to SQL after transformation.
    sort_by_date : bool
        Sort DataFrame chronologically before computation (default True).
    fill_missing : bool
        If True, forward-fill missing closing odds from opening odds
        (default True).
    """

    name: str = "betting_market"
    version: int = 1
    description: str = (
        "Betting market features: odds, implied/fair probability, "
        "movement, CLV, consensus, volatility, and favorite/underdog status."
    )
    dependencies: list[str] = []
    data_type: str = "float"
    computation_time: str = "fast"
    category: str = "betting"
    author: str = "system"
    tags: list[str] = ["betting", "odds", "market", "clv", "consensus"]
    source: str = "derived"

    output_columns: list[str] = list(OUTPUT_COLS)

    _REQUIRED_COLS: frozenset[str] = frozenset({
        "date", "home_team", "away_team",
    })

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self._opening_cols: tuple[str, str, str] | None = None
        self._closing_cols: tuple[str, str, str] | None = None
        self._detected_bookmaker_sets: list[tuple[str, str, str]] = []
        self._resolved_outputs: list[str] = []

    def init(self, context: TransformContext | None = None) -> None:
        """Resolve output columns and configure odds column detection."""
        self._resolved_outputs = list(OUTPUT_COLS)
        self.output_columns = list(self._resolved_outputs)
        self._initialized = True
        logger.debug(
            "BettingMarketTransformer initialized: %d output columns",
            len(self.output_columns),
        )

    # ── Input validation ──────────────────────────────────

    def validate_input(self, df: pd.DataFrame) -> list[str]:
        """Check required columns exist."""
        errors: list[str] = []
        for col in self._REQUIRED_COLS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        return errors

    # ── Core transform ────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        context: TransformContext | None = None,
    ) -> pd.DataFrame:
        """Compute betting market features and add them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ``date``, ``home_team``, ``away_team``,
            and raw odds columns (detected automatically).
        context : TransformContext, optional
            Pipeline context (ignored in this implementation).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with betting market feature columns added.
        """
        df = df.copy()

        # ── 1. Sort chronologically ───────────────────────
        if self.params.get("sort_by_date", True) and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(["date"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        logger.debug("BettingMarket: transforming %d rows", len(df))

        # ── 2. Load extra data from SQL if provided ───────
        load_fn: Callable | None = self.params.get("load_fn")
        if load_fn is not None:
            try:
                extra_odds = load_fn()
                if extra_odds is not None and not extra_odds.empty:
                    # Merge extra odds columns into df
                    for col in extra_odds.columns:
                        if col not in df.columns and col != "match_id":
                            df[col] = extra_odds[col].values
                    logger.debug(
                        "Loaded %d rows of extra odds data", len(extra_odds),
                    )
            except Exception as exc:
                logger.warning("Failed to load extra odds data: %s", exc)

        # ── 3. Detect available odds columns ──────────────
        opening_cols = self._resolve_odds_cols(
            df, self.params.get("opening_odds_cols"), _DEFAULT_OPENING, "opening",
        )
        closing_cols = self._resolve_odds_cols(
            df, self.params.get("closing_odds_cols"), _DEFAULT_CLOSING, "closing",
        )

        # Fallback: if no opening odds found, use closing as opening
        if opening_cols is None and closing_cols is not None:
            opening_cols = closing_cols
            logger.warning(
                "No opening odds found — using closing odds as opening. "
                "Movement/CLV features will be zero."
            )

        # Fallback: if no closing odds found either, try Bet365
        if closing_cols is None:
            closing_cols = self._resolve_odds_cols(
                df, None, _FALLBACK_CLOSING, "closing (fallback)",
            )
            if closing_cols is not None and opening_cols is None:
                opening_cols = closing_cols

        self._opening_cols = opening_cols
        self._closing_cols = closing_cols

        # Detect additional bookmaker sets for consensus
        self._detected_bookmaker_sets = self._detect_bookmaker_sets(df)

        # ── 4. Extract and clean raw odds ─────────────────
        if opening_cols is not None:
            odds_open = df[list(opening_cols)].values.astype(float)
            odds_open = np.where(np.isfinite(odds_open), odds_open, np.nan)
        else:
            odds_open = np.full((len(df), 3), np.nan)

        if closing_cols is not None:
            odds_close = df[list(closing_cols)].values.astype(float)
            odds_close = np.where(np.isfinite(odds_close), odds_close, np.nan)
        else:
            odds_close = np.full((len(df), 3), np.nan)

        # ── 5. Fill missing closing odds from opening ─────
        fill_missing = self.params.get("fill_missing", True)
        if fill_missing:
            for j in range(3):
                missing_close = np.isnan(odds_close[:, j])
                has_open = np.isfinite(odds_open[:, j])
                both = missing_close & has_open
                if both.any():
                    odds_close[both, j] = odds_open[both, j]
                    logger.debug(
                        "Filled %d missing closing odds (index %d) from opening",
                        int(both.sum()), j,
                    )

        # ── 6. Compute derived features ───────────────────
        features = self._compute_all_features(
            df, odds_open, odds_close,
        )

        # ── 7. Assign columns ─────────────────────────────
        for col_name, values in features.items():
            df[col_name] = values

        # ── 8. SQL storage if provided ────────────────────
        save_fn: Callable | None = self.params.get("save_fn")
        if save_fn is not None:
            try:
                save_fn(df)
                logger.debug("Saved betting market features via save_fn")
            except Exception as exc:
                logger.warning("Failed to save betting market features: %s", exc)

        # ── 9. Ensure all output columns exist ────────────
        for col in self._resolved_outputs:
            if col not in df.columns:
                df[col] = np.nan

        logger.debug(
            "BettingMarket: added %d / %d columns",
            len([c for c in self._resolved_outputs if c in df.columns]),
            len(self._resolved_outputs),
        )
        return df

    # ══════════════════════════════════════════════════════
    #  Internal: odds column resolution
    # ══════════════════════════════════════════════════════

    def _resolve_odds_cols(
        self,
        df: pd.DataFrame,
        user_cols: tuple[str, str, str] | None,
        default_cols: tuple[str, str, str],
        label: str,
    ) -> tuple[str, str, str] | None:
        """Resolve which odds column triplet to use.

        Checks user-provided columns first, then defaults.
        Returns None if none of the columns exist.
        """
        candidates = [user_cols, default_cols] if user_cols else [default_cols]

        for cols in candidates:
            if cols is not None and all(c in df.columns for c in cols):
                logger.debug("Using %s odds columns: %s", label, cols)
                return cols

        return None

    def _detect_bookmaker_sets(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[str, str, str]]:
        """Detect which additional bookmaker column sets exist in the DataFrame."""
        detected: list[tuple[str, str, str]] = []
        for h_col, d_col, a_col in _BOOKMAKER_SETS:
            if (
                h_col in df.columns
                and d_col in df.columns
                and a_col in df.columns
            ):
                # Skip if these are the same as opening/closing columns
                if (
                    self._closing_cols is not None
                    and (h_col, d_col, a_col) == self._closing_cols
                ):
                    continue
                if (
                    self._opening_cols is not None
                    and (h_col, d_col, a_col) == self._opening_cols
                ):
                    continue
                detected.append((h_col, d_col, a_col))

        logger.debug("Detected %d extra bookmaker sets", len(detected))
        return detected

    # ══════════════════════════════════════════════════════
    #  Internal: feature computation
    # ══════════════════════════════════════════════════════

    def _compute_all_features(
        self,
        df: pd.DataFrame,
        odds_open: np.ndarray,
        odds_close: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute all betting market features from raw odds arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Original match DataFrame (used for team names).
        odds_open : np.ndarray
            Shape ``(n, 3)`` — opening odds ``(home, draw, away)``.
        odds_close : np.ndarray
            Shape ``(n, 3)`` — closing odds ``(home, draw, away)``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from column name to values.
        """
        n = len(df)
        features: dict[str, np.ndarray] = {}

        # ── Implied probabilities ────────────────────────
        ip_open = 1.0 / odds_open
        ip_close = 1.0 / odds_close

        features["odds_home_opening"] = odds_open[:, 0]
        features["odds_draw_opening"] = odds_open[:, 1]
        features["odds_away_opening"] = odds_open[:, 2]
        features["odds_home_closing"] = odds_close[:, 0]
        features["odds_draw_closing"] = odds_close[:, 1]
        features["odds_away_closing"] = odds_close[:, 2]

        features["implied_prob_home_opening"] = ip_open[:, 0]
        features["implied_prob_draw_opening"] = ip_open[:, 1]
        features["implied_prob_away_opening"] = ip_open[:, 2]
        features["implied_prob_home_closing"] = ip_close[:, 0]
        features["implied_prob_draw_closing"] = ip_close[:, 1]
        features["implied_prob_away_closing"] = ip_close[:, 2]

        # ── Bookmaker margin ─────────────────────────────
        margin_open = np.sum(ip_open, axis=1) - 1.0
        margin_close = np.sum(ip_close, axis=1) - 1.0
        margin_open = np.where(margin_open > 0, margin_open, np.nan)
        margin_close = np.where(margin_close > 0, margin_close, np.nan)

        features["bookmaker_margin_opening"] = margin_open
        features["bookmaker_margin_closing"] = margin_close

        # ── Fair probabilities (margin removed) ──────────
        fair_open = np.where(
            np.isfinite(margin_open)[:, None],
            ip_open / (1.0 + margin_open[:, None]),
            np.nan,
        )
        fair_close = np.where(
            np.isfinite(margin_close)[:, None],
            ip_close / (1.0 + margin_close[:, None]),
            np.nan,
        )

        features["fair_prob_home_opening"] = fair_open[:, 0]
        features["fair_prob_draw_opening"] = fair_open[:, 1]
        features["fair_prob_away_opening"] = fair_open[:, 2]
        features["fair_prob_home_closing"] = fair_close[:, 0]
        features["fair_prob_draw_closing"] = fair_close[:, 1]
        features["fair_prob_away_closing"] = fair_close[:, 2]

        # ── Odds movement ────────────────────────────────
        mov = odds_close - odds_open
        mov_pct = np.where(
            odds_open > 0,
            mov / odds_open * 100,
            np.nan,
        )

        features["odds_movement_home"] = mov[:, 0]
        features["odds_movement_draw"] = mov[:, 1]
        features["odds_movement_away"] = mov[:, 2]
        features["odds_movement_pct_home"] = mov_pct[:, 0]
        features["odds_movement_pct_draw"] = mov_pct[:, 1]
        features["odds_movement_pct_away"] = mov_pct[:, 2]

        # ── CLV (Closing Line Value) ─────────────────────
        clv = fair_close - fair_open

        features["clv_home"] = clv[:, 0]
        features["clv_draw"] = clv[:, 1]
        features["clv_away"] = clv[:, 2]

        # ── Market favorite and underdog ─────────────────
        fav_probs = fair_close.copy()
        # Fallback: use implied if fair is all NaN
        all_nan = ~np.any(np.isfinite(fav_probs), axis=1)
        for i in range(n):
            if all_nan[i]:
                fav_probs[i] = ip_close[i]

        fav_indices = np.full(n, -1, dtype=int)
        fav_probs_max = np.full(n, np.nan)
        und_indices = np.full(n, -1, dtype=int)
        und_probs_min = np.full(n, np.nan)

        for i in range(n):
            row = fav_probs[i]
            valid = np.isfinite(row)
            if valid.any():
                fav_indices[i] = int(np.argmax(row))
                fav_probs_max[i] = float(np.max(row))
                if valid.sum() >= 3:
                    valid_indices = np.where(valid)[0]
                    min_idx_in_valid = int(np.argmin(row[valid]))
                    und_indices[i] = valid_indices[min_idx_in_valid]
                    und_probs_min[i] = float(row[valid][min_idx_in_valid])

        market_fav = np.array([
            _OUTCOME_LABELS[idx] if idx >= 0 else np.nan
            for idx in fav_indices
        ])
        market_und = np.array([
            _OUTCOME_LABELS[idx] if idx >= 0 else np.nan
            for idx in und_indices
        ])

        features["market_favorite"] = market_fav
        features["market_underdog"] = market_und
        features["market_confidence"] = fav_probs_max

        # ── Team-level favorite/underdog indicators ───────
        home_team = df["home_team"].values
        away_team = df["away_team"].values

        h_is_fav = np.full(n, np.nan)
        a_is_fav = np.full(n, np.nan)
        h_is_und = np.full(n, np.nan)
        a_is_und = np.full(n, np.nan)

        for i in range(n):
            fav_label = market_fav[i]
            und_label = market_und[i]
            if isinstance(fav_label, str):
                if fav_label == "H":
                    h_is_fav[i] = 1.0
                    a_is_fav[i] = 0.0
                elif fav_label == "A":
                    h_is_fav[i] = 0.0
                    a_is_fav[i] = 1.0
                else:
                    h_is_fav[i] = 0.0
                    a_is_fav[i] = 0.0  # Draw is favorite — neither team
            if isinstance(und_label, str):
                if und_label == "H":
                    h_is_und[i] = 1.0
                    a_is_und[i] = 0.0
                elif und_label == "A":
                    h_is_und[i] = 0.0
                    a_is_und[i] = 1.0
                else:
                    h_is_und[i] = 0.0
                    a_is_und[i] = 0.0

        features["h_is_favorite"] = h_is_fav
        features["a_is_favorite"] = a_is_fav
        features["h_is_underdog"] = h_is_und
        features["a_is_underdog"] = a_is_und

        # ── Multi-bookmaker consensus ────────────────────
        compute_consensus = self.params.get("compute_consensus", True)
        if compute_consensus and self._detected_bookmaker_sets:
            consensus = self._compute_consensus(
                df, self._detected_bookmaker_sets,
            )
            features["consensus_home"] = consensus[:, 0]
            features["consensus_draw"] = consensus[:, 1]
            features["consensus_away"] = consensus[:, 2]
        else:
            features["consensus_home"] = fair_close[:, 0]
            features["consensus_draw"] = fair_close[:, 1]
            features["consensus_away"] = fair_close[:, 2]

        # ── Odds volatility ──────────────────────────────
        compute_volatility = self.params.get("compute_volatility", True)
        if compute_volatility and self._detected_bookmaker_sets:
            features["odds_volatility"] = self._compute_volatility(
                df, self._detected_bookmaker_sets,
            )
        else:
            features["odds_volatility"] = np.full(n, np.nan)

        return features

    # ══════════════════════════════════════════════════════
    #  Internal: multi-bookmaker consensus
    # ══════════════════════════════════════════════════════

    def _compute_consensus(
        self,
        df: pd.DataFrame,
        bookmaker_sets: list[tuple[str, str, str]],
    ) -> np.ndarray:
        """Compute mean fair probability across multiple bookmakers.

        Returns
        -------
        np.ndarray
            Shape ``(n, 3)`` — consensus fair probabilities ``(H, D, A)``.
        """
        all_fair: list[np.ndarray] = []

        for h_col, d_col, a_col in bookmaker_sets:
            try:
                odds = df[[h_col, d_col, a_col]].values.astype(float)
                odds = np.where(np.isfinite(odds), odds, np.nan)
                ip = 1.0 / odds
                margin = np.sum(ip, axis=1) - 1.0
                margin = np.where(margin > 0, margin, np.nan)
                fair = np.where(
                    np.isfinite(margin)[:, None],
                    ip / (1.0 + margin[:, None]),
                    np.nan,
                )
                all_fair.append(fair)
            except Exception:
                continue

        if not all_fair:
            return np.full((len(df), 3), np.nan)

        # Mean across bookmakers
        stacked = np.stack(all_fair, axis=-1)
        consensus = np.nanmean(stacked, axis=-1)
        return consensus

    def _compute_volatility(
        self,
        df: pd.DataFrame,
        bookmaker_sets: list[tuple[str, str, str]],
    ) -> np.ndarray:
        """Compute std dev of fair probabilities across bookmakers.

        Higher volatility = less agreement among bookmakers (obscure/
        unpredictable market).  Lower volatility = tight consensus.

        Returns
        -------
        np.ndarray
            Shape ``(n,)`` — average std across the three outcomes.
        """
        all_fair: list[np.ndarray] = []

        for h_col, d_col, a_col in bookmaker_sets:
            try:
                odds = df[[h_col, d_col, a_col]].values.astype(float)
                odds = np.where(np.isfinite(odds), odds, np.nan)
                ip = 1.0 / odds
                margin = np.sum(ip, axis=1) - 1.0
                margin = np.where(margin > 0, margin, np.nan)
                fair = np.where(
                    np.isfinite(margin)[:, None],
                    ip / (1.0 + margin[:, None]),
                    np.nan,
                )
                all_fair.append(fair)
            except Exception:
                continue

        if not all_fair:
            return np.full(len(df), np.nan)

        stacked = np.stack(all_fair, axis=-1)
        # Std across bookmakers for each outcome, then mean across outcomes
        volatility = np.nanmean(np.nanstd(stacked, axis=-1, ddof=1), axis=1)
        return volatility

    # ── Output validation ───────────────────────────────

    def validate_output(self, df: pd.DataFrame) -> list[str]:
        """Check that all betting market feature columns exist."""
        errors: list[str] = []
        if not self.output_columns:
            return errors

        for col in self.output_columns:
            if col not in df.columns:
                errors.append(f"Missing output column: {col}")
        return errors

    # ── Metadata ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "opening_odds_cols": self._opening_cols,
            "closing_odds_cols": self._closing_cols,
            "detected_bookmaker_sets": len(self._detected_bookmaker_sets),
        })
        return base

    def __repr__(self) -> str:
        has_open = self._opening_cols is not None
        has_close = self._closing_cols is not None
        n_bks = len(self._detected_bookmaker_sets)
        return (
            f"<BettingMarketTransformer v{self.version}: "
            f"opening={'yes' if has_open else 'no'}, "
            f"closing={'yes' if has_close else 'no'}, "
            f"bookmakers={n_bks}>"
        )


# ═══════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════


def create_betting_market_transformer(
    opening_odds_cols: tuple[str, str, str] | None = None,
    closing_odds_cols: tuple[str, str, str] | None = None,
    compute_consensus: bool = True,
    compute_volatility: bool = True,
    **kwargs: Any,
) -> BettingMarketTransformer:
    """Create a configured BettingMarketTransformer.

    Parameters
    ----------
    opening_odds_cols : tuple[str, str, str], optional
        Opening odds column names ``(home, draw, away)``.
    closing_odds_cols : tuple[str, str, str], optional
        Closing odds column names ``(home, draw, away)``.
    compute_consensus : bool
        Enable multi-bookmaker consensus (default True).
    compute_volatility : bool
        Enable odds volatility (default True).
    **kwargs
        Additional parameters passed to the transformer.

    Returns
    -------
    BettingMarketTransformer
    """
    return BettingMarketTransformer(
        opening_odds_cols=opening_odds_cols,
        closing_odds_cols=closing_odds_cols,
        compute_consensus=compute_consensus,
        compute_volatility=compute_volatility,
        **kwargs,
    )
