# Team Form Feature Generator

## Overview

The **TeamFormTransformer** is the first concrete football feature built on the feature engineering framework. It computes rolling averages of team performance metrics across configurable time windows and venue contexts.

It replaces and extends the legacy `_add_rolling_features()` function from `src.feature_engineering.py` with a modern, framework-native implementation.

## Quick Start

```python
from src.feature_framework import FeaturePipeline
from src.feature_framework.features import TeamFormTransformer

# Option 1: Register directly
pipeline = FeaturePipeline(show_progress=False)
pipeline.plugins.register(TeamFormTransformer)

report = pipeline.run(entity_type="dataframe", df=matches_df, trigger="manual")

# Option 2: Via YAML config (features.yaml)
pipeline = FeaturePipeline(config_path="features.yaml")
report = pipeline.run(entity_type="dataframe", df=matches_df)

# Option 3: Standalone
from src.feature_framework.features import TeamFormTransformer
t = TeamFormTransformer(windows=[3, 5, 10], contexts=["overall", "home"])
t.init()
result = t.transform(matches_df.copy())
```

## Required Input Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Match date (used for chronological ordering) |
| `home_team` | str | Home team name |
| `away_team` | str | Away team name |
| `home_goals` | int/float | Home team goals scored |
| `away_goals` | int/float | Away team goals scored |
| `result` | str | Match result (`H`, `D`, or `A`) |

## Optional Input Columns (Auto-Detected)

The transformer detects these columns via case-insensitive pattern matching. Only metrics with available source columns are computed.

| Metric | Home Column | Away Column | Aliases Detected |
|--------|-------------|-------------|------------------|
| xG | `home_xg` | `away_xg` | `h_xg`, `xg_home`, `hxg` |
| Shots | `home_shots` | `away_shots` | `h_shots`, `hs`, `shots_home` |
| Shots on Target | `home_shots_on_target` | `away_shots_on_target` | `home_shots_target`, `h_sot`, `hst` |
| Possession | `home_possession` | `away_possession` | `h_possession`, `possession_home` |
| Corners | `home_corners` | `away_corners` | `h_corners`, `hc` |
| Yellow Cards | `home_yellow_cards` | `away_yellow_cards` | `home_yellow`, `h_yellow`, `hy` |
| Red Cards | `home_red_cards` | `away_red_cards` | `home_red`, `h_red`, `hr` |

## Output Columns

### Naming Convention

```
{h|a}_{context}_{metric}_avg{window}
```

| Part | Values | Description |
|------|--------|-------------|
| `{h\|a}` | `h_` or `a_` | Home team's features vs away team's features |
| `{context}` | `overall`, `home`, `away` | Venue context |
| `{metric}` | (see below) | Performance metric |
| `{window}` | `3`, `5`, `10`, `20` | Rolling window size |

### Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `points` | float | result (3/1/0) | Average points per match |
| `wins` | binary → float | result | Win rate (proportion) |
| `draws` | binary → float | result | Draw rate (proportion) |
| `losses` | binary → float | result | Loss rate (proportion) |
| `goals_scored` | float | home/away_goals | Average goals scored |
| `goals_conceded` | float | home/away_goals | Average goals conceded |
| `goal_diff` | float | goals_scored - conceded | Average goal difference |
| `clean_sheets` | binary → float | goals_conceded == 0 | Clean sheet rate |
| `btts` | binary → float | both teams scored > 0 | Both Teams Scored rate |
| `over_2.5` | binary → float | total goals > 2.5 | Over 2.5 goals rate |
| `under_2.5` | binary → float | total goals <= 2.5 | Under 2.5 goals rate |
| `xg` | float | home/away_xg | Average expected goals |
| `xga` | float | opponent_xg | Average xG conceded |
| `xgd` | float | xg - xga | Average xG difference |
| `shots` | float | home/away_shots | Average shots per match |
| `shots_on_target` | float | home/away_sot | Average shots on target |
| `possession` | float | home/away_possession | Average possession % |
| `corners` | float | home/away_corners | Average corners per match |
| `yellow_cards` | float | home/away_yc | Average yellow cards |
| `red_cards` | float | home/away_rc | Average red cards |

### Example Output Columns (default config)

```
h_overall_points_avg3       # Home team's avg points, last 3 matches
h_overall_points_avg5       # Home team's avg points, last 5 matches
h_overall_points_avg10      # Home team's avg points, last 10 matches
h_overall_points_avg20      # Home team's avg points, last 20 matches
h_home_points_avg3          # Home team's avg points in HOME matches, last 3 home matches
h_away_points_avg3          # Home team's avg points in AWAY matches, last 3 away matches
a_overall_goals_scored_avg5 # Away team's avg goals scored, last 5 matches
a_away_clean_sheets_avg10   # Away team's clean sheet rate in away matches, last 10 away
```

## Leakage Prevention

All rolling statistics use `pandas.Series.rolling().mean().shift(1)`, which ensures:

1. **Current match excluded**: The `.shift(1)` moves all values down by one row, so match N's data is never used to compute match N's features.
2. **Only historical data**: Features for a match only include data from matches that occurred *before* it.

```
team_stats for Team_X (sorted by date):
┌──────────┬────────┬─────────┬───────────┐
│ match_id │ points │ rolling │ shift(1)  │ ← Feature value used for
│          │        │ mean(3) │           │   this match's predictions
├──────────┼────────┼─────────┼───────────┤
│ 0        │ 3      │ 3.0     │ NaN       │ ← No history before match 0
│ 4        │ 1      │ 2.0     │ 3.0       │ ← Uses match 0 only
│ 8        │ 3      │ 2.3     │ 2.0       │ ← Uses matches 0 & 4 (not 8)
└──────────┴────────┴─────────┴───────────┘
```

## Configuration

### YAML Config

```yaml
# features.yaml
version: "1.0"
pipeline:
  default_entity_type: dataframe
  show_progress: true
  parallel: true

features:
  - name: team_form
    type: team_form
    category: form
    version: 1
    data_type: float
    computation_time: medium
    enabled: true
    params:
      windows: [3, 5, 10, 20]    # Default windows
      contexts: [overall, home, away]  # Default contexts
      league_specific: true       # Reset per league
      sort_by_date: true          # Sort input chronologically
      include_xg: true            # Enable xG metrics
      include_shots: true         # Enable shot metrics
      include_possession: false   # Enable possession metrics
      include_cards: false        # Enable card metrics
```

### Python Config

```python
from src.feature_framework.features import create_team_form_transformer

# Pre-configured instance
t = create_team_form_transformer(
    windows=[3, 5, 10],
    contexts=["overall", "home"],
    league_specific=True,
    include_xg=True,
    include_shots=True,
    include_possession=False,
    include_cards=False,
)
```

## Performance

| Configuration | Rows | Time |
|---------------|------|------|
| Default (4 windows, 3 contexts) | 1,000 | ~50ms |
| Default (4 windows, 3 contexts) | 10,000 | ~400ms |
| Default (4 windows, 3 contexts) | 50,000 | ~2s |
| All metrics + optional | 10,000 | ~800ms |

Computation is vectorized via `pandas.groupby().apply()` with NumPy arrays for derived columns. The transformer is **thread-safe** and works with the framework's `ParallelComputer`.

## Differences from Legacy `_add_rolling_features()`

| Aspect | Legacy (`feature_engineering.py`) | New (`TeamFormTransformer`) |
|--------|-----------------------------------|-----------------------------|
| Framework | Standalone function | `FeatureTransformer` ABC |
| Contexts | Overall only | Overall + Home + Away |
| Windows | Fixed (config) | Configurable per instance |
| Metrics | Goals, points, GD | 20 metrics (with optional) |
| Column detection | Manual | Auto-detection (pattern-based) |
| Optional stats | No | xG, shots, cards, etc. |
| Validation | None | `validate_input/output` |
| Config | `config.py` global | Per-instance params |
| League-specific | Via groupby | Configurable |
| Integration | `build_features()` | `FeaturePipeline` |
| Reusability | Single pipeline | Plug-and-play with registry |
| Tests | None | 40+ unit tests |

## Extending

To add a new metric:

1. Add its source column patterns to `_COLUMN_PATTERNS` dict
2. Add its definition to `_OPTIONAL_METRICS` dict with:
   - `is_binary`: whether it's a 0/1 indicator
   - `source_home`/`source_away`: which team_stats columns to use
   - `depends_on`: (optional) list of other metrics it depends on

To add a new context:

1. Add the context name to `_DEFAULT_CONTEXTS` tuple
2. The `_compute_rolling()` method automatically handles filtering by `is_home`
3. Column names auto-generate with the context prefix
