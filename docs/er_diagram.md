# Entity-Relationship Diagram

## Schema Overview

This document describes the fully normalised PostgreSQL schema for football analytics.
The schema contains **21 tables** organised into 5 logical groups.

---

## ER Diagram (Mermaid)

```mermaid
erDiagram
    %% ── Core entities ──────────────────────────────────
    COUNTRY {
        int id PK
        string name UK
        string iso_alpha2 UK
        string iso_alpha3 UK
        string fifa_code UK "nullable"
        string continent "nullable"
    }

    COMPETITION {
        int id PK
        string name
        string code UK "nullable"
        string type "league|cup|playoff|friendly"
        int country_id FK "nullable"
        int level "nullable"
    }

    SEASON {
        int id PK
        string name
        int competition_id FK
        date start_date
        date end_date
    }

    STADIUM {
        int id PK
        string name
        string city "nullable"
        int country_id FK "nullable"
        int capacity "nullable"
        string surface "nullable"
    }

    TEAM {
        int id PK
        string name UK
        string short_name "nullable"
        int country_id FK "nullable"
        int stadium_id FK "nullable"
        int year_founded "nullable"
    }

    REFEREE {
        int id PK
        string full_name
        int country_id FK "nullable"
    }

    %% ── Match ──────────────────────────────────────────
    MATCH {
        int id PK
        int competition_id FK "nullable"
        int season_id FK "nullable"
        int home_team_id FK
        int away_team_id FK
        int stadium_id FK "nullable"
        int referee_id FK "nullable"
        date match_date
        string round "nullable"
        bool is_neutral_venue
        int attendance "nullable"
        int home_goals "nullable"
        int away_goals "nullable"
        string result "nullable"
        string duration "regular|extra_time|penalties"
        string status "scheduled|live|finished|postponed|cancelled|abandoned"
    }

    MATCH_STATISTICS {
        int id PK
        int match_id FK UK "1:1"
        int home_shots "nullable"
        int home_shots_on_target "nullable"
        float home_possession "nullable"
        int home_corners "nullable"
        int home_yellow_cards "nullable"
        int home_red_cards "nullable"
        int away_shots "nullable"
        int away_shots_on_target "nullable"
        float away_possession "nullable"
    }

    ODDS {
        int id PK
        int match_id FK
        string source "bookmaker name"
        datetime timestamp
        float odds_home "nullable"
        float odds_draw "nullable"
        float odds_away "nullable"
        float implied_prob_home "nullable"
        float implied_prob_draw "nullable"
        float implied_prob_away "nullable"
        float margin "nullable"
    }

    WEATHER {
        int id PK
        int match_id FK UK "1:1"
        float temperature_celsius "nullable"
        int humidity_pct "nullable"
        float wind_speed_kmh "nullable"
        string condition "nullable"
        string pitch_condition "nullable"
    }

    LINEUP {
        int id PK
        int match_id FK
        int team_id FK
        string formation "nullable"
        jsonb starting_xi "nullable"
        jsonb substitutes "nullable"
        string coach "nullable"
    }

    %% ── Player ─────────────────────────────────────────
    PLAYER {
        int id PK
        string full_name
        date date_of_birth "nullable"
        int country_id FK "nullable"
        string position "nullable"
        string preferred_foot "nullable"
        int height_cm "nullable"
        int weight_kg "nullable"
        int current_team_id FK "nullable"
        float market_value_eur "nullable"
    }

    PLAYER_MATCH_STATS {
        int id PK
        int match_id FK
        int player_id FK
        int team_id FK
        int minutes_played "nullable"
        bool is_starter
        int goals
        int assists
        int shots
        int shots_on_target
        float rating "nullable"
        float xg "nullable"
        float xa "nullable"
    }

    INJURY {
        int id PK
        int player_id FK
        string injury_type "nullable"
        string severity "nullable"
        date injury_date
        date expected_return "nullable"
        date actual_return "nullable"
        int missed_matches "nullable"
    }

    TRANSFER {
        int id PK
        int player_id FK
        int from_team_id FK "nullable"
        int to_team_id FK
        date transfer_date
        float transfer_fee_eur "nullable"
        bool is_loan
        date loan_end_date "nullable"
    }

    %% ── Team analytics ─────────────────────────────────
    TEAM_ELO_HISTORY {
        int id PK
        int team_id FK
        int match_id FK
        string side "home|away"
        float elo_before
        float elo_after
        float elo_change
        float k_factor "nullable"
    }

    TEAM_FORM {
        int id PK
        int team_id FK
        int match_id FK
        string side "home|away"
        float last_5_ppg "nullable"
        float last_5_goals_scored "nullable"
        float last_5_goals_conceded "nullable"
        float last_10_ppg "nullable"
        float last_20_ppg "nullable"
        string current_streak "nullable"
    }

    TEAM_XG_HISTORY {
        int id PK
        int team_id FK
        int match_id FK
        string side "home|away"
        string source "opta|understat|statsbomb"
        float xg
        float xa "nullable"
        int shots "nullable"
        int shots_on_target "nullable"
    }

    %% ── Betting ───────────────────────────────────────
    PREDICTION {
        int id PK
        int match_id FK
        string model_name
        string model_version "nullable"
        float prob_home "nullable"
        float prob_draw "nullable"
        float prob_away "nullable"
        float confidence "nullable"
    }

    EXPECTED_VALUE_BET {
        int id PK
        int match_id FK
        string bookmaker
        float model_prob_home
        float model_prob_draw
        float model_prob_away
        float book_prob_home
        float book_prob_draw
        float book_prob_away
        float ev_home "Expected Value"
        float ev_draw
        float ev_away
        float kelly_stake_home "nullable"
        float kelly_stake_draw "nullable"
        float kelly_stake_away "nullable"
        string recommended_bet "nullable"
    }

    CLOSING_LINE_VALUE {
        int id PK
        int match_id FK
        string bookmaker
        string outcome "H|D|A"
        float opening_price
        float closing_price
        float clv
    }

    BETTING_RESULT {
        int id PK
        int match_id FK
        string strategy
        string bookmaker "nullable"
        string bet_outcome "H|D|A"
        float decimal_odds
        float stake
        bool won "nullable"
        float profit "nullable"
        float roi_pct "nullable"
    }

    %% ── Relationships ─────────────────────────────────
    COUNTRY ||--o{ COMPETITION : "has"
    COUNTRY ||--o{ TEAM : "has"
    COUNTRY ||--o{ STADIUM : "located_in"
    COUNTRY ||--o{ REFEREE : "has"
    COUNTRY ||--o{ PLAYER : "citizen_of"

    COMPETITION ||--o{ SEASON : "has"
    COMPETITION ||--o{ MATCH : "contains"

    SEASON ||--o{ MATCH : "groups"

    TEAM ||--o{ MATCH : "home_team"
    TEAM ||--o{ MATCH : "away_team"
    TEAM ||--o{ PLAYER : "employs"
    TEAM ||--o{ TRANSFER : "sells_to"
    TEAM ||--o{ TRANSFER : "buys_from"
    TEAM ||--o{ LINEUP : "fielded"
    TEAM ||--o{ TEAM_ELO_HISTORY : "elo_tracked"
    TEAM ||--o{ TEAM_FORM : "form_tracked"
    TEAM ||--o{ TEAM_XG_HISTORY : "xg_tracked"

    STADIUM ||--o{ MATCH : "hosts"
    STADIUM ||--o{ TEAM : "home_ground"

    REFEREE ||--o{ MATCH : "officiates"

    MATCH ||--|| MATCH_STATISTICS : "has"
    MATCH ||--o{ ODDS : "has"
    MATCH ||--|| WEATHER : "has"
    MATCH ||--o{ LINEUP : "has"
    MATCH ||--o{ PREDICTION : "predicted_by"
    MATCH ||--o{ EXPECTED_VALUE_BET : "value_analysis"
    MATCH ||--o{ CLOSING_LINE_VALUE : "clv_tracked"
    MATCH ||--o{ BETTING_RESULT : "bet_outcome"

    PLAYER ||--o{ PLAYER_MATCH_STATS : "performs"
    PLAYER ||--o{ INJURY : "sustains"
    PLAYER ||--o{ TRANSFER : "transferred"

    MATCH ||--o{ PLAYER_MATCH_STATS : "records"
```

---

## Table Groups

### 1. Core Entities (7 tables)
Foundational reference data that changes slowly.

| Table | Rows (est.) | Primary Key | Notes |
|-------|-------------|-------------|-------|
| `countries` | ~250 | `id` | ISO codes for every nation |
| `competitions` | ~500 | `id` | Replaces old `leagues` table |
| `seasons` | ~5,000 | `id` | Each comp has ~10-100 seasons |
| `teams` | ~10,000 | `id` | Clubs + national teams |
| `stadiums` | ~5,000 | `id` | Venue info |
| `referees` | ~5,000 | `id` | Match officials |
| `players` | ~500,000 | `id` | Individual footballers |

### 2. Matches (1 central table + 4 detail tables)
The heart of the schema.

| Table | Rows (est.) | Link | Notes |
|-------|-------------|------|-------|
| `matches` | 10M+ | — | Central fact table, 6 FKs |
| `match_statistics` | = matches | 1:1 | Shots, possession, cards |
| `weather` | = matches | 1:1 | Optional weather data |
| `odds` | 50M+ | 1:N | Multi-bookmaker, multi-timestamp |
| `lineups` | 2× matches | 1:N | Home + away lineups |

### 3. Player Analytics (4 tables)

| Table | Rows (est.) | Link | Notes |
|-------|-------------|------|-------|
| `player_match_stats` | 100M+ | 1:N | Per-player per-match stats |
| `injuries` | 500K | 1:N | Injury tracking |
| `transfers` | 200K | 1:N | Transfer fees and loans |

### 4. Team Analytics (3 tables)
Pre-computed features for ML pipelines.

| Table | Rows (est.) | Link | Notes |
|-------|-------------|------|-------|
| `team_elo_history` | 2× matches | 1:N | Elo before/after each match |
| `team_form` | 2× matches | 1:N | Rolling form at match time |
| `team_xg_history` | 2× matches | 1:N | xG by source (opta, understat) |

### 5. Betting & Predictions (5 tables)

| Table | Rows (est.) | Link | Notes |
|-------|-------------|------|-------|
| `predictions` | = matches | 1:N | Model output per match |
| `expected_value_bets` | = matches | 1:N | EV per bookmaker |
| `closing_line_values` | = matches | 1:N | CLV tracking |
| `betting_results` | = bets | 1:N | Actual P&L tracking |

---

## Foreign Key Map

```
countries   ──→ competitions.country_id
countries   ──→ teams.country_id
countries   ──→ stadiums.country_id
countries   ──→ referees.country_id
countries   ──→ players.country_id

competitions ──→ seasons.competition_id
competitions ──→ matches.competition_id

seasons      ──→ matches.season_id

stadiums     ──→ matches.stadium_id
stadiums     ──→ teams.stadium_id

referees     ──→ matches.referee_id

teams        ──→ matches.home_team_id
teams        ──→ matches.away_team_id
teams        ──→ players.current_team_id
teams        ──→ transfers.from_team_id
teams        ──→ transfers.to_team_id
teams        ──→ lineups.team_id
teams        ──→ team_elo_history.team_id
teams        ──→ team_form.team_id
teams        ──→ team_xg_history.team_id
teams        ──→ player_match_stats.team_id

matches      ──→ match_statistics.match_id
matches      ──→ odds.match_id
matches      ──→ weather.match_id
matches      ──→ lineups.match_id
matches      ──→ predictions.match_id
matches      ──→ expected_value_bets.match_id
matches      ──→ closing_line_values.match_id
matches      ──→ betting_results.match_id
matches      ──→ player_match_stats.match_id
matches      ──→ team_elo_history.match_id
matches      ──→ team_form.match_id
matches      ──→ team_xg_history.match_id

players      ──→ player_match_stats.player_id
players      ──→ injuries.player_id
players      ──→ transfers.player_id
```

---

## Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Tables | `snake_case` plural | `match_statistics`, `team_elo_history` |
| Columns | `snake_case` | `home_team_id`, `market_value_eur` |
| PKs | `id` | `matches.id` |
| FKs | `<referenced_table_singular>_id` | `match_id`, `player_id` |
| Unique constraints | `uq_<table>_<columns>` | `uq_odds_match_source_time` |
| Foreign keys | `fk_<table>_<column>` | `fk_matches_home_team_id` |
| Check constraints | `ck_<table>_<description>` | `ck_odds_home_positive` |
| Indexes | `ix_<table>_<columns>` | `ix_matches_comp_season_date` |
