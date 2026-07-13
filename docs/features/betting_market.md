# Betting Market Feature Generator

Transforms raw bookmaker decimal odds into a rich set of market-derived features with overround removal, multi-bookmaker consensus, and missing-odds resilience.

## Quick Start

```python
from src.feature_framework.features.betting_market import BettingMarketTransformer

transformer = BettingMarketTransformer()
result = transformer.transform(df)

# Access features
print(result["odds_home_opening"])      # Raw opening odds
print(result["fair_prob_home_closing"])  # Margin-removed probability
print(result["clv_home"])               # Closing Line Value
```

## Features

| Feature | Description | Formula |
|---------|-------------|---------|
| **Opening odds** (H/D/A) | Raw decimal odds at market open | — |
| **Closing odds** (H/D/A) | Raw decimal odds at market close | — |
| **Implied probability** | Raw probability including margin | `1 / odds` |
| **Fair probability** | Margin-removed probability | `IP / (1 + margin)` |
| **Odds movement** | Closing − opening (abs + %) | `∆odds`, `∆odds / open × 100` |
| **Market consensus** | Mean fair prob across bookmakers | `mean(fair₁, ..., fairₙ)` |
| **CLV reference** | Change in fair prob from open to close | `fair_close − fair_open` |
| **Favorite status** | Which team has shortest odds | `argmin(odds)` |
| **Underdog status** | Which team has longest odds | `argmax(odds)` |
| **Odds volatility** | Std dev of fair probs across books | `mean(std(H), std(D), std(A))` |
| **Bookmaker margin** | Overround percentage | `sum(IP) − 1` |

## Requirements Coverage

### Remove Bookmaker Margin

Uses the **multiplicative method** — the standard approach used by professional bettors:

```
fair_prob = implied_prob / (1 + margin)
```

This ensures all three fair probabilities sum to exactly 1.0, removing the bookmaker's built-in commission.

### Time-Aware

DataFrames are sorted chronologically by `date` before computation. The `sort_by_date` parameter (default `True`) controls this behavior.

### Multiple Bookmakers

Auto-detects available bookmaker column sets from these known sets:
- `BbAvH/D/A` — BetBrain average
- `B365H/D/A` — Bet365
- `BWH/D/A` — Bet&Win
- `IWH/D/A` — Interwetten
- `LBH/D/A` — Ladbrokes
- `SBH/D/A` — Sportingbet
- `WHH/D/A` — William Hill
- `SJH/D/A` — Stan James
- `VCH/D/A` — VC Bet

When multiple bookmakers are available, `consensus_home/draw/away` provides the mean fair probability and `odds_volatility` measures market disagreement (higher = less consensus = more unpredictable).

### Missing Odds Handling

| Scenario | Behavior |
|----------|----------|
| No odds columns | All features filled with `NaN` |
| Opening odds missing | Uses closing odds as opening (movement = 0) |
| Closing odds missing | Uses opening odds as closing |
| Partially missing | Propagates `NaN` through computations |
| Single bookmaker | Consensus falls back to fair probability |

### SQL Storage

Two callback parameters for database integration:

```python
def load_from_db() -> pd.DataFrame:
    """Load historical odds from SQL."""
    ...

def save_to_db(df: pd.DataFrame) -> None:
    """Save computed features to SQL."""
    ...

transformer = BettingMarketTransformer(
    load_fn=load_from_db,
    save_fn=save_to_db,
)
result = transformer.transform(df)
```

## Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `odds_home_opening` | float | Opening decimal odds for home win |
| `odds_draw_opening` | float | Opening decimal odds for draw |
| `odds_away_opening` | float | Opening decimal odds for away win |
| `odds_home_closing` | float | Closing decimal odds for home win |
| `odds_draw_closing` | float | Closing decimal odds for draw |
| `odds_away_closing` | float | Closing decimal odds for away win |
| `implied_prob_home_opening` | float | 1 / opening odds (H) |
| `implied_prob_draw_opening` | float | 1 / opening odds (D) |
| `implied_prob_away_opening` | float | 1 / opening odds (A) |
| `implied_prob_home_closing` | float | 1 / closing odds (H) |
| `implied_prob_draw_closing` | float | 1 / closing odds (D) |
| `implied_prob_away_closing` | float | 1 / closing odds (A) |
| `fair_prob_home_opening` | float | Margin-removed prob (H, opening) |
| `fair_prob_draw_opening` | float | Margin-removed prob (D, opening) |
| `fair_prob_away_opening` | float | Margin-removed prob (A, opening) |
| `fair_prob_home_closing` | float | Margin-removed prob (H, closing) |
| `fair_prob_draw_closing` | float | Margin-removed prob (D, closing) |
| `fair_prob_away_closing` | float | Margin-removed prob (A, closing) |
| `odds_movement_home` | float | Closing − opening odds (H) |
| `odds_movement_draw` | float | Closing − opening odds (D) |
| `odds_movement_away` | float | Closing − opening odds (A) |
| `odds_movement_pct_home` | float | % change in odds (H) |
| `odds_movement_pct_draw` | float | % change in odds (D) |
| `odds_movement_pct_away` | float | % change in odds (A) |
| `clv_home` | float | fair_close − fair_open (H) |
| `clv_draw` | float | fair_close − fair_open (D) |
| `clv_away` | float | fair_close − fair_open (A) |
| `market_favorite` | str | Shortest-odds outcome (H/D/A) |
| `market_underdog` | str | Longest-odds outcome (H/D/A) |
| `market_confidence` | float | Fair prob of favorite (0-1) |
| `consensus_home` | float | Mean fair prob across books (H) |
| `consensus_draw` | float | Mean fair prob across books (D) |
| `consensus_away` | float | Mean fair prob across books (A) |
| `odds_volatility` | float | Std dev of fair probs across books |
| `bookmaker_margin_opening` | float | Overround in opening odds |
| `bookmaker_margin_closing` | float | Overround in closing odds |
| `h_is_favorite` | float | 1.0 if home is favorite |
| `a_is_favorite` | float | 1.0 if away is favorite |
| `h_is_underdog` | float | 1.0 if home is underdog |
| `a_is_underdog` | float | 1.0 if away is underdog |

## API Reference

### `BettingMarketTransformer`

| Method | Description |
|--------|-------------|
| `transform(df, context)` | Compute all betting market features |
| `validate_input(df)` | Check required cols (date, home_team, away_team) |
| `validate_output(df)` | Check all 33 output columns exist |

### Constructor Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `opening_odds_cols` | `("BbMxH", "BbMxD", "BbMxA")` | Opening odds column names |
| `closing_odds_cols` | `("BbAvH", "BbAvD", "BbAvA")` | Closing odds column names |
| `compute_consensus` | `True` | Enable multi-bookmaker consensus |
| `compute_volatility` | `True` | Enable odds volatility |
| `load_fn` | `None` | SQL load callback |
| `save_fn` | `None` | SQL save callback |
| `sort_by_date` | `True` | Sort chronologically |
| `fill_missing` | `True` | Forward-fill missing closing odds |

### Factory

```python
from src.feature_framework.features.betting_market import create_betting_market_transformer

transformer = create_betting_market_transformer(
    opening_odds_cols=("MaxH", "MaxD", "MaxA"),
    closing_odds_cols=("AvgH", "AvgD", "AvgA"),
)
```

## Test Coverage

| Class | Tests | Key Coverage |
|-------|:-----:|--------------|
| `TestBettingInputValidation` | 3 | Required cols, missing cols |
| `TestBettingCoreOdds` | 3 | Raw odds preserved, draw, away |
| `TestBettingProbability` | 4 | Implied prob, margin removal, fair sums to 1 |
| `TestBettingMovement` | 3 | Negative, positive, percentage |
| `TestBettingCLV` | 3 | Positive CLV, negative CLV, sums to zero |
| `TestBettingFavoriteUnderdog` | 8 | Favorite, underdog, flags, confidence |
| `TestBettingConsensus` | 6 | Columns, range, available, volatility |
| `TestBettingMissingOdds` | 3 | No odds, partial, fallback |
| `TestBettingSQLIntegration` | 6 | load_fn, merge data, failure, save_fn |
| `TestBettingValidation` | 3 | Pass, fail, all columns |
| `TestBettingEdgeCases` | 6 | Empty, single row, custom cols, disabled |
| `TestBettingConfiguration` | 5 | Defaults, custom, repr, to_dict, factory |
