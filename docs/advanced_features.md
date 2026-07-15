# Advanced Predictive Features

> **Last updated:** 2026-07-15  
> **Status:** ✅ Implemented and integrated

This document describes the advanced feature sets added to the prediction
pipeline. These features are **opt-in** — they are disabled by default and
must be explicitly enabled in `config.py` to avoid breaking existing
workflows.

---

## Table of Contents

1. [Quick Comparison](#1-quick-comparison)
2. [Weather Features](#2-weather-features)
3. [Referee Statistics](#3-referee-statistics)
4. [Schedule & Congestion Features](#4-schedule--congestion-features)
5. [Extended Head-to-Head (H2H)](#5-extended-head-to-head-h2h)
6. [Extended Team Form](#6-extended-team-form)
7. [Feature Store Registration](#7-feature-store-registration)
8. [Data Sources](#8-data-sources)
9. [Performance Impact](#9-performance-impact)

---

## 1. Quick Comparison

| Feature Set | Config Flag | Default | New Columns | Data Source |
|-------------|------------|---------|-------------|-------------|
| **Weather** | `weather.enabled` | `False` | 12 | `weather.csv` |
| **Referee** | `referee.enabled` | `False` | 4 | `referees.csv` |
| **Schedule** | `schedule.enabled` | `True` | 16 | Derived |
| **Extended H2H** | `extended_features.enabled` | `False` | ~180 | Derived |
| **Extended Form** | `extended_features.enabled` | `False` | ~300 | Derived |

**Total advanced columns available:** ~512 (when all enabled)

---

## 2. Weather Features

### Config (`config.py`)

```python
config.weather.enabled = True          # Enable weather features
config.weather.default_temp = 15.0     # Fallback temperature (Celsius)
config.weather.placeholder_value = 0.0 # Fallback for missing data
```

### Feature Columns

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `h_temperature_celsius` | float | -10–45 | Match-day temperature for home team (Celsius) |
| `a_temperature_celsius` | float | -10–45 | Match-day temperature for away team |
| `h_humidity_pct` | float | 0–100 | Humidity percentage |
| `a_humidity_pct` | float | 0–100 | Humidity percentage |
| `h_wind_speed_kmh` | float | 0–150 | Wind speed in km/h |
| `a_wind_speed_kmh` | float | 0–150 | Wind speed in km/h |
| `h_precipitation_mm` | float | 0–100 | Precipitation in mm |
| `a_precipitation_mm` | float | 0–100 | Precipitation in mm |
| `h_pitch_condition_encoded` | int | 0–3 | 0=dry, 1=wet, 2=waterlogged, 3=frozen |
| `a_pitch_condition_encoded` | int | 0–3 | Same for away team |
| `h_weather_severity` | float | 0–1 | Composite score: 0.4*precip + 0.3*wind + 0.3*temp |
| `a_weather_severity` | float | 0–1 | Composite score for away team |

### Known Effects

| Condition | Impact |
|-----------|--------|
| **Heavy rain** (+precip, +wind) | Reduces scoring (~15%), favours defensive teams |
| **High wind** (>30 km/h) | Disrupts long passes, benefits direct-play teams |
| **High heat** (>30°C) | Increases fatigue, benefits fitter/acclimated teams |
| **Wet pitch** | Slows ball movement, reduces through-ball effectiveness |

---

## 3. Referee Statistics

### Config (`config.py`)

```python
config.referee.enabled = True       # Enable referee features
config.referee.window = 20          # Rolling window for referee stats
```

### Feature Columns

| Column | Type | Description |
|--------|------|-------------|
| `referee_home_yellow_rate` | float | Rolling avg home yellows under this ref (last 20 matches) |
| `referee_away_yellow_rate` | float | Rolling avg away yellows under this ref |
| `referee_home_win_rate` | float | Rolling home win rate under this ref |
| `referee_card_total_avg` | float | Rolling avg total cards (YC+RC) per match |

### Leakage Protection

All referee statistics use `.shift(1)` — the current match's data is
never used to compute its own referee features.

### Usage Notes

- Features are computed per-referee (group by referee name)
- Falls back to neutral values (0.0, 0.5) when no referee data available
- Referee bias is real: some referees give 2x more home yellows than others

---

## 4. Schedule & Congestion Features

### Config (`config.py`)

```python
config.schedule.enabled = True
config.schedule.include_travel_distance = True
config.schedule.include_fatigue = True
```

### Feature Columns

| Column | Type | Description |
|--------|------|-------------|
| `h_rest_days` | float | Days since home team's last match |
| `a_rest_days` | float | Days since away team's last match |
| `h_days_since_last_match` | float | Alias for rest_days |
| `a_days_since_last_match` | float | Alias for rest_days |
| `h_matches_last_7_days` | int | Home team matches in last 7 days |
| `a_matches_last_7_days` | int | Away team matches in last 7 days |
| `h_matches_last_14_days` | int | Home team matches in last 14 days |
| `a_matches_last_14_days` | int | Away team matches in last 14 days |
| `h_consec_home` | int | Consecutive home matches (streak length) |
| `a_consec_away` | int | Consecutive away matches (streak length) |
| `h_is_back_to_back` | bool | Facing same opponent as last match |
| `a_is_back_to_back` | bool | Facing same opponent as last match |
| `h_travel_distance` | float | Distance from previous venue (km) |
| `a_travel_distance` | float | Distance from previous venue (km) |
| `h_days_since_competition` | float | Days since last match in same competition |
| `a_days_since_competition` | float | Days since last match in same competition |

### Fatigue Thresholds

| Metric | Normal | Elevated | Critical |
|--------|--------|----------|----------|
| `rest_days` | 4–7 days | 2–3 days | 0–1 day (fixture pile-up) |
| `matches_last_7_days` | 0–1 | 2 | 3+ (extreme congestion) |
| `travel_distance` | <500 km | 500–3000 km | >3000 km (jet lag risk) |

---

## 5. Extended Head-to-Head (H2H)

### Config (`config.py`)

```python
config.extended_features.enabled = True
config.extended_features.include_extended_h2h = True
config.extended_features.h2h_windows = (3, 5, 10)  # Number of meetings
```

### Compared to Basic H2H

| Aspect | Basic (`_add_h2h_features`) | Extended (`H2HTransformer`) |
|--------|----------------------------|---------------------------|
| Windows | 1 (expanding) | 3 (3, 5, 10 meetings) |
| Contexts | None | 3 (overall, home, away) |
| Metrics | 7 (points, goals, win rate) | 11+ (includes BTTS, O/U 2.5, clean sheets, xG) |

### Column Naming

Pattern: `{h|a}_h2h_{context}_{metric}_last{window}`

Examples:
- `h_h2h_overall_wins_last3` — home win rate vs opponent, last 3 meetings
- `a_h2h_away_goals_scored_last10` — away goals when away vs opponent, last 10

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `wins` | binary | Team won the H2H meeting |
| `draws` | binary | H2H meeting was a draw |
| `losses` | binary | Team lost the H2H meeting |
| `goals_scored` | numeric | Goals scored in H2H meetings |
| `goals_conceded` | numeric | Goals conceded in H2H meetings |
| `goal_diff` | numeric | Goal difference in H2H |
| `btts` | binary | Both Teams Scored in H2H |
| `over_2.5` | binary | Over 2.5 goals in H2H |
| `clean_sheets` | binary | Clean sheet in H2H |
| `xg` | numeric | xG in H2H (if available) |
| `xga` | numeric | xGA conceded in H2H (if available) |
| `xgd` | numeric | xG difference in H2H (if available) |

---

## 6. Extended Team Form

### Config (`config.py`)

```python
config.extended_features.enabled = True
config.extended_features.include_extended_form = True
config.extended_features.form_windows = (3, 5, 10, 20)
```

### Compared to Basic Rolling Features

| Aspect | Basic (`_add_rolling_features`) | Extended (`TeamFormTransformer`) |
|--------|--------------------------------|--------------------------------|
| Windows | 2 (5, N) | 4 (3, 5, 10, 20) |
| Contexts | 1 (overall) | 3 (overall, home, away) |
| Core metrics | 3 (points, goals, GD) | 11 (points, W/D/L, goals, GD, CS, BTTS, O/U) |
| Optional metrics | None | 9 (xG, xGA, xGD, shots, SOT, possession, corners, cards) |

### Column Naming

Pattern: `{h|a}_{context}_{metric}_avg{window}`

Examples:
- `h_overall_clean_sheets_avg5` — home clean sheet rate, last 5 matches
- `a_away_goals_scored_avg10` — away goals when away, last 10 away matches

---

## 7. Feature Store Registration

All advanced features are registered in the Feature Store. To register:

```bash
# Register in the database Feature Store
python scripts/register_advanced_features.py

# Preview without registering
python scripts/register_advanced_features.py --dry-run

# Export to CSV
python scripts/register_advanced_features.py --csv reports/advanced_features.csv
```

This registers **32 representative definitions** covering all advanced
feature types. Individual column-level features (e.g., every H2H metric
× window × context combination) are computed dynamically at pipeline time.

---

## 8. Data Sources

### Required CSV files (place in `data/external/`)

| File | Required? | Columns |
|------|-----------|---------|
| `weather.csv` | For weather features | `match_id`, `temperature`, `humidity`, `wind`, `precipitation`, `pitch`, `condition` |
| `referees.csv` | For referee features | `referee_name`, `home_yellow_cards`, `away_yellow_cards`, `home_red_cards`, `away_red_cards`, `result`, `date` |

Both files are **optional** — if missing, the feature functions gracefully
fall back to neutral placeholder values.

### Derived features (no external data needed)

Schedule, extended H2H, and extended form features are computed entirely
from the match DataFrame's existing columns (`date`, `home_team`,
`away_team`, `home_goals`, `away_goals`, `result`).

---

## 9. Performance Impact

| Feature Set | Extra Time (10k rows) | Extra Columns | Impact |
|-------------|----------------------|---------------|--------|
| Weather | <10 ms | 12 | Negligible |
| Referee | <20 ms | 4 | Negligible |
| Schedule | ~100 ms | 16 | Very low |
| Extended H2H | ~500 ms | ~180 | Low |
| Extended Form | ~800 ms | ~300 | Low |

**Total:** ~1.5 seconds added to the pipeline for 10,000 rows.

### Memory Impact

Each additional column is ~80 KB per 10,000 rows (float64).
Total: ~42 MB for all 512 advanced columns at 10k rows.
Compared to the base feature set (~100 columns), this is a ~5× increase
in memory.

### Disabling Features

To disable all advanced features:
```python
config.weather.enabled = False
config.referee.enabled = False
config.schedule.enabled = False
config.extended_features.enabled = False
```
