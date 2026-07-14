# Current Database Schema

> Auto-generated from ORM models (22 tables, 4 domains).
> Generated: 2026-07-14

## Domain Overview

| Domain | Tables | Purpose |
|--------|--------|---------|
| **Core** | 16 | Countries, competitions, seasons, teams, matches, match detail |
| **Team Analytics** | 3 | Elo history, form streaks, xG time-series |
| **Betting** | 4 | Odds, predictions, EV bets, CLV, betting results |
| **ML Ops** | 0 | Experiment tracking (not yet migrated from filesystem) |

---

## Table: `countries`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| name | VARCHAR(128) | NO | UNIQUE | |
| code | VARCHAR(8) | YES | | ISO 2-letter |
| fifa_code | VARCHAR(4) | YES | | FIFA code |
| flag_url | VARCHAR(512) | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK on `id`, UNIQUE on `name`
**Est. rows:** ~250

---

## Table: `competitions`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| name | VARCHAR(128) | NO | | |
| country_id | INTEGER | YES | FK→countries.id | |
| type | VARCHAR(16) | NO | 'league' | league/cup |
| level | INTEGER | YES | | Tier 1-10 |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK on `id`, FK on `country_id`
**Constraints:** `type IN ('league', 'cup', 'international')`
**Est. rows:** ~500

---

## Table: `seasons`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| competition_id | INTEGER | NO | FK→competitions.id | |
| name | VARCHAR(64) | NO | | e.g. '2024/2025' |
| start_date | DATE | NO | | |
| end_date | DATE | NO | | |
| created_at | TIMESTAMPTZ | NO | now() | |
| updated_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK on `id`, FK on `competition_id`, UNIQUE on `(competition_id, name)`
**Constraints:** `start_date <= end_date`
**Est. rows:** ~5,000

---

## Table: `teams`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| name | VARCHAR(128) | NO | UNIQUE | |
| short_name | VARCHAR(8) | YES | | |
| country_id | INTEGER | YES | FK→countries.id | |
| stadium_id | INTEGER | YES | FK→stadiums.id | |
| year_founded | INTEGER | YES | | |
| logo_url | VARCHAR(512) | YES | | |
| website | VARCHAR(256) | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |
| updated_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK on `id`, UNIQUE on `name`, FK on `country_id`, FK on `stadium_id`
**Est. rows:** ~10,000

---

## Table: `stadiums`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| name | VARCHAR(128) | NO | | |
| city | VARCHAR(128) | YES | | |
| country_id | INTEGER | YES | FK→countries.id | |
| capacity | INTEGER | YES | | |
| surface | VARCHAR(32) | YES | | |
| latitude | FLOAT | YES | | |
| longitude | FLOAT | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK on `id`, FK on `country_id`
**Est. rows:** ~2,000

---

## Table: `referees`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| name | VARCHAR(128) | NO | | |
| nationality | VARCHAR(64) | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK on `id`
**Est. rows:** ~2,000

---

## Table: `matches` (partitioned)

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | BIGINT | NO | PK auto | PK, seq cache 1000 |
| competition_id | INTEGER | YES | FK→competitions.id | |
| season_id | INTEGER | YES | FK→seasons.id | |
| home_team_id | INTEGER | NO | FK→teams.id | |
| away_team_id | INTEGER | NO | FK→teams.id | |
| stadium_id | INTEGER | YES | FK→stadiums.id | |
| referee_id | INTEGER | YES | FK→referees.id | |
| match_date | DATE | NO | | |
| round | VARCHAR(32) | YES | | |
| is_neutral_venue | BOOLEAN | YES | false | |
| attendance | INTEGER | YES | | |
| home_goals | INTEGER | YES | | |
| away_goals | INTEGER | YES | | |
| result | VARCHAR(4) | YES | | H/D/A |
| duration | VARCHAR(8) | YES | 'regular' | regular/extra_time/penalties |
| status | VARCHAR(16) | NO | 'scheduled' | scheduled/live/finished/... |
| created_at | TIMESTAMPTZ | NO | now() | |
| updated_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK, BRIN(match_date), 5 B-tree composites, 3 partial
**Constraints:** home≠away, result IN('H','D','A'), duration IN(3), status IN(6)
**Partitioning:** RANGE by match_date (yearly)
**Fillfactor:** 70
**Est. rows:** 50M+

---

## Table: `match_statistics`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| match_id | INTEGER | NO | FK→matches.id, UNIQUE | 1:1 |
| home_shots..away_shots | INTEGER | YES | | |
| home_possession..away_possession | FLOAT | YES | | 0-100 |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK on `id`, UNIQUE on `match_id`
**Est. rows:** 50M (1:1 with matches)

---

## Table: `odds`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | BIGINT | NO | PK auto | PK, seq cache 1000 |
| match_id | INTEGER | NO | FK→matches.id | |
| source | VARCHAR(32) | NO | | Bookmaker |
| timestamp | TIMESTAMPTZ | NO | | |
| odds_home | FLOAT | YES | | |
| odds_draw | FLOAT | YES | | |
| odds_away | FLOAT | YES | | |
| implied_prob_home | FLOAT | YES | | 1/odds |
| implied_prob_draw | FLOAT | YES | | |
| implied_prob_away | FLOAT | YES | | |
| margin | FLOAT | YES | | sum(implied) - 1 |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK, FK on `match_id`, UNIQUE on `(match_id, source, timestamp)`, covering composite
**Constraints:** all odds > 1.0
**Fillfactor:** 90
**Est. rows:** 100M+

---

## Table: `predictions`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| match_id | INTEGER | NO | FK→matches.id | |
| model_name | VARCHAR(64) | NO | | |
| model_version | VARCHAR(32) | YES | | |
| prob_home | FLOAT | NO | | |
| prob_draw | FLOAT | NO | | |
| prob_away | FLOAT | NO | | |
| predicted_result | VARCHAR(4) | YES | | |
| confidence | FLOAT | YES | | |
| expected_value | FLOAT | YES | | |
| kelly_stake | FLOAT | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK, FK on `match_id`, covering composite `(model_name, match_id)`, partial by model
**Est. rows:** 50M+

---

## Table: `team_elo_history`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| team_id | INTEGER | NO | FK→teams.id | |
| match_id | INTEGER | NO | FK→matches.id | |
| side | VARCHAR(4) | NO | | home/away |
| elo_before | FLOAT | NO | | |
| elo_after | FLOAT | NO | | |
| elo_change | FLOAT | YES | | |
| k_factor | FLOAT | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK, FK on team_id + match_id, covering chrono, partial by side
**Constraints:** `side IN ('home', 'away')`
**Fillfactor:** 70
**Est. rows:** 50M+

---

## Table: `team_form`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| team_id | INTEGER | NO | FK→teams.id | |
| match_id | INTEGER | NO | FK→matches.id | |
| side | VARCHAR(4) | NO | | home/away |
| last_5_ppg | FLOAT | YES | | |
| last_5_goals_scored | FLOAT | YES | | |
| last_5_goals_conceded | FLOAT | YES | | |
| season_matches_played | INTEGER | YES | | |
| last_5_wins | INTEGER | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK, FK, covering composite
**Fillfactor:** 70
**Est. rows:** 50M+

---

## Table: `team_xg_history`

| Column | Type | Nullable | Default | Notes |
|--------|------|----------|---------|-------|
| id | INTEGER | NO | PK auto | PK |
| team_id | INTEGER | NO | FK→teams.id | |
| match_id | INTEGER | NO | FK→matches.id | |
| source | VARCHAR(16) | NO | | |
| side | VARCHAR(4) | NO | | home/away |
| xg | FLOAT | YES | | |
| xa | FLOAT | YES | | |
| shots | INTEGER | YES | | |
| created_at | TIMESTAMPTZ | NO | now() | |

**Indexes:** PK, FK, UNIQUE on `(team_id, match_id, source)`, covering composite
**Fillfactor:** 70
**Est. rows:** 50M+

---

## Remaining Tables (10)

See `docs/database_performance.md` for full details on:
- `players` — ~25K rows, player registry
- `player_match_stats` — 200M+ rows, per-match performance
- `weather` — 50M rows, match conditions
- `lineups` — 50M rows, formations + starting XI
- `injuries` — 500K rows, injury tracking
- `transfers` — 100K rows, transfer fees
- `betting_results` — 5M rows, bet outcomes
- `closing_line_values` — 5M rows, CLV tracking
- `expected_value_bets` — 1M rows, EV calculations
- `materialized_views` — 3 views (mv_league_standings, mv_model_performance, mv_team_dashboard)

---

## Materialized Views

| View | Refresh | Purpose |
|------|---------|---------|
| `mv_league_standings` | After match days | Pre-computed points, goals, W/D/L per team |
| `mv_model_performance` | Hourly | Accuracy, Brier, log-loss by model version |
| `mv_team_dashboard` | After match days | Joined view of elo + form + xG + match result |

---

## Monitoring Views

| View | Purpose |
|------|---------|
| `v_slow_queries` | Queries running >5s |
| `v_table_bloat` | Tables with >10K dead tuples |
| `v_index_usage` | Index sorted by scan count |
| `v_seq_scans` | Tables with heavy seq scans |
| `v_table_sizes` | Table + index size ranking |
| `v_missing_indexes` | Tables with >100 seq scans |
| `v_cache_hit_ratio` | Buffer cache hit ratio |
| `v_query_stats` | Top 50 queries by total time |
