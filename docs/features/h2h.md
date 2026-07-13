# Head-to-Head Feature Generator — `H2HTransformer`

Computes **historical matchup statistics** between team pairs — rolling win/draw/loss rates, goals, xG, BTTS, and clean sheets across the last N meetings, with venue context separation.

---

## Features (12 metrics × 3 contexts × 3 windows)

### Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `wins` | Team won the H2H meeting | binary |
| `draws` | H2H meeting ended in a draw | binary |
| `losses` | Team lost the H2H meeting | binary |
| `goals_scored` | Goals the team scored in the H2H meeting | int |
| `goals_conceded` | Goals the team conceded in the H2H meeting | int |
| `goal_diff` | goals_scored - goals_conceded | int |
| `btts` | Both teams scored in the H2H meeting | binary |
| `over_2.5` | Total goals > 2.5 in the H2H meeting | binary |
| `clean_sheets` | Team kept a clean sheet in H2H meeting | binary |
| `xg` | Team's xG in the H2H meeting | float |
| `xga` | Opponent's xG in the H2H meeting | float |
| `xgd` | xg - xga | float |

### Windows

Default: `[3, 5, 10]` — configurable via `params["windows"]`.

### Contexts

| Context | Description |
|---------|-------------|
| `overall` | All H2H meetings regardless of venue |
| `home` | Only H2H meetings where the team was at home |
| `away` | Only H2H meetings where the team was away |

### Column naming

Pattern: `{h|a}_h2h_{context}_{metric}_last{window}`

| Column | Meaning |
|--------|---------|
| `h_h2h_overall_wins_last3` | Home team's win rate vs away team in last 3 meetings |
| `a_h2h_away_goals_scored_last5` | Away team's avg goals when away vs this opponent |
| `h_h2h_home_clean_sheets_last10` | Home team's clean sheet rate at home vs this opponent |

---

## Leakage Prevention

All features use `_rolling_last_n()` which computes the **mean of the previous N values** (excluding the current match). The first meeting between any pair yields NaN for all H2H features.

| Mechanism | Applied to |
|-----------|-----------|
| `_rolling_last_n(values[i-n:i])` — mean of prior N | All metrics |
| Position 0 always NaN | First meeting of any pair |
| Home/away context mapping | Only same-venue meetings counted |

---

## Usage

### Standalone

```python
import pandas as pd
from src.feature_framework.features.h2h import H2HTransformer

df = pd.DataFrame({
    "date":       ["2024-01-01", "2024-01-08", "2024-01-15"],
    "home_team":  ["Team_A",     "Team_B",     "Team_A"    ],
    "away_team":  ["Team_B",     "Team_A",     "Team_B"    ],
    "home_goals": [2,            1,            1           ],
    "away_goals": [0,            0,            1           ],
    "result":     ["H",          "H",          "D"         ],
})

t = H2HTransformer()
t.init()
result = t.transform(df)

print(result[["h_h2h_overall_wins_last3", "a_h2h_overall_goals_scored_last3"]])
```

### With Custom Windows and Contexts

```python
t = H2HTransformer(
    windows=[5, 10],
    contexts=["overall"],
    include_xg=True,
)
t.init()
result = t.transform(df)
```

### With SQL Integration (via `load_fn`)

```python
from src.database import get_session

def load_historical_matches():
    """Load extra H2H data from the database for more accurate rolling stats."""
    with get_session() as session:
        return pd.read_sql(
            "SELECT * FROM historical_h2h",
            session.bind,
        )

t = H2HTransformer(load_fn=load_historical_matches)
t.init()
result = t.transform(df)
```

### With FeaturePipeline

```python
from src.feature_framework import FeaturePipeline

pipeline = FeaturePipeline(config_dict={
    "features": [{
        "name": "head_to_head",
        "type": "head_to_head",
        "category": "h2h",
        "params": {"windows": [3, 5, 10]},
    }],
}, show_progress=False)
pipeline.plugins.register(H2HTransformer)

report = pipeline.run(entity_type="dataframe", df=matches_df)
```

---

## Internal Architecture

```
transform(df)
    │
    ├── Sort by date
    ├── Detect optional xG columns
    ├── _build_pair_stats(df)
    │     └── 2 rows per match (home + away perspective)
    │         - Team_A vs Team_B from A's view
    │         - Team_B vs Team_A from B's view
    │         └── Optional: load_fn() merges extra historical data
    │
    ├── _compute_rolling_h2h(pair_stats)
    │     └── Group by (team, opponent)
    │           └── For each pair:
    │                 ├── For each context (overall / home / away):
    │                 │     ├── Filter rows to context
    │                 │     ├── _rolling_last_n(values, N) for each metric
    │                 │     └── Map back to full group by match_id
    │                 └── Result has h2h_{ctx}_{metric}_last{N} columns
    │
    └── _merge_features(df, pair_rolling)
          └── Lookup by (team, match_id) for h_ and a_ prefixes
```

---

## Configuration Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `windows` | `list[int]` | `[3, 5, 10]` | Number of meetings to look back |
| `contexts` | `list[str]` | `["overall","home","away"]` | Venue contexts |
| `include_xg` | `bool` | `True` | Include xG/xGA/xGD when data available |
| `load_fn` | `callable` | `None` | Function returning historical DataFrame for SQL integration |
| `sort_by_date` | `bool` | `True` | Sort input chronologically |

---

## Test Coverage (38 tests)

| Test Class | Tests | Coverage |
|-----------|:-----:|----------|
| `TestH2HInputValidation` | 3 | Missing columns, all present, empty DF |
| `TestH2HOutputColumns` | 7 | Default, windows, contexts, xG, custom windows |
| `TestH2HCoreMetrics` | 8 | First match, accumulation, goals, clean sheets, home/away filtering |
| `TestH2HGoalMetrics` | 2 | BTTS, over 2.5 |
| `TestH2HXGMetrics` | 2 | xG rolling, xGA rolling |
| `TestH2HLeakage` | 3 | No future data, goals not leaked, all draws |
| `TestH2HEdgeCases` | 5 | Single row, no overlap, 12 meetings, preserve columns, no duplicates |
| `TestH2HSQLIntegration` | 2 | load_fn called + merged, load_fn failure handled |
| `TestH2HConfiguration` | 5 | Empty windows, to_dict, metadata, repr, factory |
| `TestH2HValidation` | 2 | Output passes, missing |
