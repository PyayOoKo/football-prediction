# Elo Rating Engine

## Overview

The **Elo Rating Engine** is a production-grade team strength rating system for football. It wraps the math of the classic Elo chess rating system with football-specific adjustments: home advantage, goal-margin scaling, match importance, league strength, season regression, host nation bonuses, and new team handling.

The engine has two entry points:
- **`EloEngine`** — standalone rating engine (use for batch processing, history, visualization)
- **`EloTransformer`** — `FeatureTransformer` wrapper for the framework (use with `FeaturePipeline`)

## Quick Start

### Standalone Engine

```python
from src.feature_framework.features.elo_rating import EloEngine

engine = EloEngine(k=20, home_advantage=100)
df = engine.process_matches(matches_df)

# df now has columns: h_elo, a_elo, elo_diff, elo_k
print(engine.ratings)  # Current ratings for all teams
engine.print_standings()  # Ranked table

# Team trajectory
traj = engine.team_trajectory("Arsenal")
print(traj.head())

# Historical reconstruction
hist = engine.get_history_df()
```

### With FeaturePipeline

```python
from src.feature_framework import FeaturePipeline
from src.feature_framework.features import EloTransformer

pipeline = FeaturePipeline(show_progress=False)
pipeline.plugins.register(EloTransformer)

report = pipeline.run(entity_type="dataframe", df=matches_df, trigger="manual")
# report contains h_elo, a_elo, elo_diff
```

### Via YAML Config

```yaml
# features.yaml
features:
  - name: elo_rating
    type: elo_rating
    category: elo_rating
    version: 1
    data_type: float
    computation_time: medium
    params:
      k: 20
      home_advantage: 100
      use_goal_margin: true
      use_importance: true
      use_league_strength: true
```

## Formulas

### Expected Score (Home Team)

```
E_home = 1 / (1 + 10 ^ ((R_away − R_home − H) / 400))

Where:
- R_home, R_away = pre-match Elo ratings
- H = home advantage (default 100 Elo points)
- 400 = Elo scaling factor
```

### Rating Update

```
R_new = R_old + K_eff × (S − E)

Where:
- K_eff = K × ln(1 + min(GD, max_margin)) × I × L
- K = base K-factor (default 20, Club Elo standard)
- GD = goal margin (xG margin preferred)
- I = match importance multiplier (0.5–1.5)
- L = league strength multiplier (0.8–1.4)
- S = actual score: 1.0 (win), 0.5 (draw), 0.0 (loss)
- E = expected score (from formula above)
```

### Season Regression

```
R_new = μ + (R_old − μ) × (1 − r)

Where:
- μ = mean rating across all active teams
- r = regression factor (default 1/3, Club Elo standard)
```

## Features

| Feature | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| **Home advantage** | `home_advantage` | 100 | Elo points added for home team |
| **Dynamic K-factor** | `k` | 20 | Base K scaled by margin, importance, league |
| **Goal margin** | `use_goal_margin` | True | `ln(GD+1)` multiplier (min 0.5x) |
| **New teams** | `new_team_rating` | 1300 | Lower starting rating for new/promoted teams |
| **Season regression** | `regress_to_mean` | True | Pull ratings toward the mean between seasons |
| **Host bonus** | `host_bonus` | 50 | Temporary host nation boost to expected score |
| **Match importance** | `use_importance` | True | K-factor multiplier by competition type |
| **League strength** | `use_league_strength` | True | K-factor multiplier by league tier |

## Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `h_elo` | float | Home team's pre-match Elo rating |
| `a_elo` | float | Away team's pre-match Elo rating |
| `elo_diff` | float | `h_elo - a_elo` (home advantage already in expected score) |
| `elo_k` | float | K-factor used for the match (informational) |

## Match Importance Multipliers

| Competition | Multiplier |
|-------------|-----------|
| World Cup | 1.5 |
| Continental Championship (Euros, Copa America) | 1.3 |
| World Cup Qualifier | 1.2 |
| Continental Qualifier | 1.1 |
| Domestic League | 1.0 |
| Domestic Cup | 0.9 |
| International Friendly | 0.6 |
| Club Friendly | 0.5 |

## League Strength by Tier

| Tier | Example | Multiplier |
|------|---------|-----------|
| 1 (Top division) | Premier League, La Liga | 1.0 |
| 2 | Championship, Serie B | 1.1 |
| 3 | League One, 3. Liga | 1.2 |
| 4 | League Two | 1.3 |
| 5+ | Lower divisions | 1.4 |

## New Team / Promoted Teams

Newly encountered teams start at `new_team_rating` (default 1300), which is 200 points below the standard `initial_rating` (1500). This models the common pattern where newly promoted teams are weaker than established top-flight teams. The rating rises or falls based on actual match results.

To use the same rating for all teams (no promotion penalty), set `new_team_rating=initial_rating`.

## Host Nation Bonus

Host nations get a temporary bonus to their expected score (not baked into their actual rating). After the match, the rating update uses the unboosted rating, so the bonus is **temporary** — it only affects the expected score calculation for that specific match.

```python
engine = EloEngine(host_bonus=50)
engine.process_matches(df, host_nations={"2026": "USA"})
```

## Engine State Management

### Append Mode

By default, each call to `process_matches()` resets the engine. To continue from previous state:

```python
result1 = engine.process_matches(df_part1, append=True)   # Processes & keeps state
result2 = engine.process_matches(df_part2, append=True)   # Continues from result1
```

### Incremental Updates

```python
record = engine.update("Arsenal", "Chelsea", "H",
                       home_goals=2, away_goals=0,
                       season="2024", league="Premier League")
```

### Reset

```python
engine.reset()
```

## History & Reconstruction

```python
# Full match history as DataFrame
hist = engine.get_history_df()

# Single team trajectory
traj = engine.team_trajectory("Liverpool")

# Current snapshot
snap = engine.current_snapshot()
print(snap.ratings)
```

## Visualization

Requires `matplotlib` (optional dependency).

```python
# Plot a single team's Elo trajectory
fig = engine.plot_team_trajectory("Arsenal")
fig.savefig("arsenal_elo.png")

# Plot the current rating distribution
fig = engine.plot_rating_distribution()
fig.savefig("elo_distribution.png")

# Text-based standings table
engine.print_standings(top_n=20)
```

## Club Elo Alignment

The default parameters align with the Club Elo methodology (FiveThirtyEight-style):

| Parameter | Club Elo Standard | Default |
|-----------|-------------------|---------|
| K-factor | 20 | 20 |
| Home advantage | 100 | 100 |
| Initial rating | 1500 | 1500 |
| New team rating | 1300 | 1300 |
| Goal margin | ln(GD + 1) | ln(GD + 1) |
| Season regression | 1/3 toward mean | 1/3 toward mean |

```python
engine = EloEngine()
report = engine.benchmark_report()
print(report["club_elo_aligned"])  # True if defaults unchanged
```

## Parameter Reference

### EloEngine Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 20 | Base K-factor |
| `home_advantage` | int | 100 | Home advantage Elo points |
| `initial_rating` | float | 1500.0 | Default rating for established teams |
| `new_team_rating` | float | 1300.0 | Rating for newly seen/promoted teams |
| `regression_factor` | float | 1/3 | Season regression toward mean |
| `use_goal_margin` | bool | True | Scale K by goal margin |
| `max_margin` | int | 5 | Cap on goal margin multiplier |
| `use_importance` | bool | True | Scale K by match importance |
| `use_league_strength` | bool | True | Scale K by league tier |
| `host_bonus` | float | 50.0 | Host nation bonus points |
| `regress_to_mean` | bool | True | Regress ratings between seasons |

### EloTransformer Additional Params

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `append_mode` | bool | False | Don't reset engine on transform |
| `host_nations` | dict | None | Season->host mapping |
| `host_nations_file` | str | None | JSON file path for hosts |

## Differences from Legacy `EloSystem`

| Aspect | Legacy (`src/elo.py`) | New (`EloEngine`) |
|--------|---------------------|-------------------|
| Framework | Standalone class | Framework-native `FeatureTransformer` |
| K-factor | Fixed or goal-margin only | Dynamic: margin × importance × league |
| New teams | `initial_rating` only | Separate `new_team_rating` (lower) |
| Host bonus | Manual check in caller | Built-in via `host_nations` param |
| History | None | Full `EloMatchRecord` history |
| Season regression | Separate method | Auto-detected on season change |
| League strength | None | Tier-based multiplier |
| Visualization | None | `plot_trajectory`, `plot_distribution`, `print_standings` |
| Club Elo alignment | None | `benchmark_report()` |
| Tests | Minimal | 56 unit tests |
