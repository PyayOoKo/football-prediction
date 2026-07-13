# League Strength Module

Per-season, per-league analytics for football competitions. Estimates offensive/defensive strength, competitive balance, home advantage, and cross-league normalisation factors. Tracks promoted/relegated teams and European competition participation.

## Quick Start

```python
from src.feature_framework.league_strength import LeagueStrengthEngine

engine = LeagueStrengthEngine(reference_league="E0")
results = engine.compute(df)

for key, record in results.items():
    print(f"{key}: GS={record.offensive_strength:.2f} HA={record.home_adv:.2f}")
```

## Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `offensive_strength` | Avg goals scored per team per match | float |
| `defensive_strength` | Avg goals conceded per team per match | float |
| `avg_goals` | Total goals / total matches | float |
| `avg_xg` | Average xG per match (if available) | float |
| `competitive_balance` | Std of goal difference | float |
| `home_adv` | Avg home goals − avg away goals | float |
| `home_win_rate` | Proportion of home wins | float |
| `draw_rate` | Proportion of draws | float |
| `away_win_rate` | Proportion of away wins | float |
| `btts_rate` | Proportion of matches both teams scored | float |
| `over_2_5_rate` | Proportion of matches with >2.5 total goals | float |
| `attack_factor` | Normalised attack strength (ref=1.0) | float |
| `defence_factor` | Normalised defence strength (ref=1.0) | float |

> **Note:** At league aggregate level, `offensive_strength == defensive_strength` — every goal scored is also conceded. The cross-league `attack_factor` / `defence_factor` comparison is what distinguishes leagues.

## Features

### Cross-league Normalisation

```python
# Reference league (default "E0") gets factor=1.0
# Other leagues expressed relative to it
normalised = engine.normalise_across_leagues(
    seasons=["2024"],
    leagues=["E0", "E1", "SA"],
    reference_league="E0",
)
# E0 attack_factor = 1.0, SA > 1.0 means more attacking
```

### Promoted / Relegated Teams

Manual:
```python
engine.set_promoted("2025", "E0", {"Team_X", "Team_Y"})
engine.set_relegated("2024", "E0", {"Team_A", "Team_B"})
```

Auto-detection (requires league groups):
```python
engine.auto_detect_promoted_relegated(
    team_seasons_df,
    league_groups={
        "England": ["E0", "E1", "E2"],
        "Scotland": ["S1", "S2"],
    },
)
```

### European Competition Adjustment

```python
engine.set_european("2024", "E0", {"Arsenal", "Man City", "Liverpool"})
comparison = engine.european_adjustment(match_df)
# Returns metrics "with_european" and "without_european"
```

### Persistence

```python
engine.save_json("data/league_strength.json")
# ... later ...
engine.load_json("data/league_strength.json")
engine.summary()
```

## API Reference

### `LeagueStrengthEngine`

| Method | Description |
|--------|-------------|
| `compute(df, ...)` | Compute metrics for all (season, league) pairs |
| `normalise_across_leagues(seasons, leagues, reference_league)` | Return normalised comparison DataFrame |
| `summary()` | Human-readable report |
| `set_promoted(season, league, teams)` | Record promoted teams |
| `set_relegated(season, league, teams)` | Record relegated teams |
| `set_european(season, league, teams)` | Record European teams |
| `get_promoted(season, league)` | Retrieve promoted teams |
| `get_relegated(season, league)` | Retrieve relegated teams |
| `get_european(season, league)` | Retrieve European teams |
| `auto_detect_promoted_relegated(df, league_groups)` | Auto-detect from team-season mapping |
| `european_adjustment(df, ...)` | Compare metrics with/without Euro matches |
| `store_season(season, league, record)` | Manually store a record |
| `get_season(season, league)` | Retrieve a stored record |
| `get_history_dataframe()` | All records as DataFrame |
| `clear_history()` | Reset all stored data |
| `save_json(path)` | Persist to JSON |
| `load_json(path)` | Load from JSON |

### Constructor Parameters

| Param | Default | Description |
|-------|---------|-------------|
| `reference_league` | `"E0"` | Baseline league for normalisation |
| `min_matches` | `10` | Minimum matches for reliable metrics |
| `auto_normalise` | `True` | Auto-compute attack/defence factors |
| `store_history` | `True` | Keep records in memory |

## Test Coverage

| Class | Tests | Key Coverage |
|-------|:-----:|--------------|
| `TestLeagueCoreMetrics` | 9 | Basic computation, home adv, BTTS, over 2.5, xG, min matches |
| `TestLeagueNormalisation` | 5 | Factor=1 for ref, stronger league, disable, across leagues, empty |
| `TestLeaguePromotionRelegation` | 4 | Manual set, auto-detect, compute reflect, missing cols |
| `TestLeagueEuropean` | 3 | Set, adjustment, no data |
| `TestLeagueHistory` | 6 | Store/retrieve, missing, compute stores, clear, dataframe |
| `TestLeaguePersistence` | 1 | JSON save/load round-trip |
| `TestLeagueRecord` | 4 | to_dict, from_dict, alias, repr |
| `TestLeagueSummary` | 2 | With data, empty |
| `TestLeagueFactory` | 2 | Custom params, defaults |
