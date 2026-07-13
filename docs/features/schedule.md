# Schedule Feature Generator — `ScheduleTransformer`

Computes **fixture congestion and schedule features** that capture how much rest a team has had, how frequently they've been playing, whether they're on a home/away streak, and how far they've travelled.

---

## Features (8 total)

| Feature | Description | Type | Leakage-free |
|---------|-------------|------|:---:|
| `rest_days` | Days since the team's last match (any venue) | `float` | ✓ |
| `days_since_last_match` | Alias for `rest_days` | `float` | ✓ |
| `matches_last_7_days` | Number of matches played in the last 7 days | `int` | ✓ |
| `matches_last_14_days` | Number of matches played in the last 14 days | `int` | ✓ |
| `consec_home` | Consecutive home matches before this one | `int` | ✓ |
| `consec_away` | Consecutive away matches before this one | `int` | ✓ |
| `is_back_to_back` | 1 if facing the same opponent as last match | `binary` | ✓ |
| `travel_distance` | Great-circle distance from previous venue (km) | `float` | ✓ |
| `days_since_competition` | Days since last match in the same competition | `float` | ✓ |

### Column naming

Pattern: `{h|a}_{feature_name}`

| Column | Meaning |
|--------|---------|
| `h_rest_days` | Home team's rest days before this match |
| `a_rest_days` | Away team's rest days before this match |
| `h_matches_last_7_days` | Home team's matches in the last 7 days |
| `a_consec_away` | Consecutive away matches for the away team |
| `h_is_back_to_back` | Home team facing same opponent as last match |
| `a_travel_distance` | Distance the away team travelled (km) |

---

## Leakage Prevention

All features are **non-leaking** — the current match's data never contributes to its own feature values.

| Mechanism | Applied to |
|-----------|-----------|
| `np.diff()` then shift by 1 | `rest_days`, `days_since_last_match` |
| `dates[:i]` loop (excludes current) | `matches_last_7_days`, `matches_last_14_days` |
| `cumcount()` within venue-streak | `consec_home`, `consec_away` |
| `shift(1)` on opponent column | `is_back_to_back` |
| Previous venue coordinates | `travel_distance` |
| Per-competition last-match tracking | `days_since_competition` |

---

## Usage

### Standalone

```python
import pandas as pd
from src.feature_framework.features.schedule import ScheduleTransformer

df = pd.DataFrame({
    "date":      ["2024-01-01", "2024-01-08", "2024-01-15"],
    "home_team": ["Team_A",     "Team_B",     "Team_A"    ],
    "away_team": ["Team_B",     "Team_A",     "Team_B"    ],
    "result":    ["H",          "A",          "H"         ],
})

t = ScheduleTransformer()
t.init()
result = t.transform(df)

# Inspect rest days for the home team
print(result[["h_rest_days", "a_rest_days", "h_matches_last_7_days"]])
```

### With Travel Distance

```python
df = pd.DataFrame({
    "date":     ["2024-01-01", "2024-01-08"],
    "home_team":["Team_A",     "Team_B"    ],
    "away_team":["Team_B",     "Team_A"    ],
    "home_lat": [51.5,         53.5        ],  # London
    "home_lon": [-0.1,         -2.2        ],  # Manchester
    "away_lat": [53.5,         51.5        ],
    "away_lon": [-2.2,         -0.1        ],
})

result = ScheduleTransformer().init().transform(df)
print(result[["h_travel_distance", "a_travel_distance"]])
# ~262 km for both teams (London ↔ Manchester)
```

### With FeaturePipeline

```python
from src.feature_framework import FeaturePipeline

pipeline = FeaturePipeline(config_dict={
    "features": [{
        "name": "schedule",
        "type": "schedule",
        "category": "schedule",
        "params": {"league_specific": True},
    }],
}, show_progress=False)
pipeline.plugins.register(ScheduleTransformer)

report = pipeline.run(entity_type="dataframe", df=matches_df)
```

### Factory Function

```python
from src.feature_framework.features.schedule import create_schedule_transformer

t = create_schedule_transformer(
    league_specific=True,
    include_travel_distance=False,  # No lat/lon data available
)
t.init()
result = t.transform(df)
```

---

## Auto-detection of Travel Columns

The transformer scans for lat/lon columns using case-insensitive pattern matching:

| Canonical | Detected from |
|-----------|--------------|
| `home_lat` | `home_lat`, `home_latitude`, `h_lat`, `venue_lat_home` |
| `home_lon` | `home_lon`, `home_longitude`, `home_lng`, `h_lon`, `venue_lon_home`, `venue_lng_home` |
| `away_lat` | `away_lat`, `away_latitude`, `a_lat`, `venue_lat_away` |
| `away_lon` | `away_lon`, `away_longitude`, `away_lng`, `a_lon`, `venue_lon_away`, `venue_lng_away` |

Distance uses the **Haversine formula** (great-circle distance) with Earth radius = 6371 km. Distances ≥ 20,000 km are capped to NaN.

---

## Requirements Coverage

| Requirement | How it's handled |
|-------------|------------------|
| **League aware** | `league_specific=True` groups by `(team, league)` for per-competition tracking |
| **Cup competitions** | `days_since_competition` tracks gaps per league name |
| **International breaks** | Long gaps (≥ 14 days) naturally captured by `rest_days` and `days_since_competition` |
| **Incremental computation** | Engine state persists across calls; use `append_mode` via `EloTransformer`-style pattern |
| **Batch updates** | Full DataFrame processed in one `transform()` call |
| **Feature validation** | Standard `validate_input()` / `validate_output()` hooks via `FeatureTransformer` ABC |

---

## Configuration Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `league_specific` | `bool` | `True` | Reset per-team features per league |
| `include_travel_distance` | `bool` | `True` | Include `h_travel_distance` / `a_travel_distance` |
| `sort_by_date` | `bool` | `True` | Sort input DataFrame chronologically |

---

## Internal Architecture

```
transform(df)
    │
    ├── Sort by date
    ├── Detect travel columns
    ├── _build_team_schedule(df)
    │     └── 2 rows per match (home + away team)
    │
    ├── _compute_schedule_features(team_schedule)
    │     ├── Group by (team) or (team, league)
    │     └── _compute_per_team_features(grp)
    │           ├── rest_days = np.diff(dates)
    │           ├── matches_last_7/14 = _count_in_window(dates, N)
    │           ├── consec_home/away = cumcount per venue-streak
    │           ├── is_back_to_back = opponent == shift(opponent)
    │           ├── travel_distance = haversine(prev_lat/lon, curr_lat/lon)
    │           └── days_since_competition = per-league date tracking
    │
    └── _merge_features(df, schedule_features)
          └── Lookup by (team, match_id) for home & away teams
```

---

## Test Coverage (48 tests)

| Test Class | Tests | Coverage |
|-----------|:-----:|----------|
| `TestScheduleInputValidation` | 3 | Missing columns, all present, empty DF |
| `TestScheduleRestDays` | 5 | Columns exist, first match NaN, computation, tight schedule, alias |
| `TestScheduleMatchDensity` | 5 | Columns exist, first match 0, 7-day, 14-day, difference |
| `TestScheduleConsecutiveStreaks` | 4 | Columns, home streaks, away streaks, no leakage |
| `TestScheduleBackToBack` | 4 | Columns, detection, first match, no false positives |
| `TestScheduleTravelDistance` | 5 | Columns, first NaN, no data, excluded, distance value |
| `TestScheduleDaysSinceCompetition` | 5 | Columns, first NaN, same comp, different comp, fallback |
| `TestScheduleLeagueSpecific` | 2 | Feature computation, league_specific=False |
| `TestScheduleEdgeCases` | 5 | Empty DF with columns, single row, two matches, preserve originals, no duplicates |
| `TestScheduleValidation` | 4 | Output passes, missing columns, all present |
| `TestScheduleMetadata` | 4 | Metadata, repr, to_dict, factory, defaults, custom params |
