---
tags:
  - football-prediction
  - database
  - orm
  - schema
created: 2026-07-12
---

# 🗄 Database Schema

> Fully normalised PostgreSQL schema with 21 tables for football analytics.

See also: [[Architecture Overview]], [[Config System]], [[Auxiliary Modules]]

---

## ER Diagram (Core Tables)

```mermaid
erDiagram
    MATCH {
        int id PK
        int competition_id FK
        int season_id FK
        int home_team_id FK
        int away_team_id FK
        int stadium_id FK
        int referee_id FK
        date match_date
        string round
        int home_goals
        int away_goals
        string result
        string status
    }

    MATCH_STATISTICS {
        int id PK
        int match_id FK "1:1"
        int home_shots
        int away_shots
        float home_possession
        int home_corners
    }

    ODDS {
        int id PK
        int match_id FK
        string source
        datetime timestamp
        float odds_home
        float odds_draw
        float odds_away
    }

    TEAM {
        int id PK
        string name
        int country_id FK
    }
    
    COMPETITION {
        int id PK
        string name
        string type
    }
    
    PLAYER {
        int id PK
        string full_name
        int current_team_id FK
        float market_value_eur
    }

    PREDICTION {
        int id PK
        int match_id FK
        string model_name
        float prob_home
        float prob_draw
        float prob_away
    }

    MATCH ||--o| MATCH_STATISTICS : has
    MATCH ||--o{ ODDS : has
    MATCH ||--o{ PREDICTION : predicted_by
    MATCH ||--o{ TEAM : home_team
    MATCH ||--o{ TEAM : away_team
    COMPETITION ||--o{ MATCH : contains
    TEAM ||--o{ PLAYER : employs
```

---

## Table Groups

### 1. Core Entities (7 tables)

| Table | Est. Rows | Description |
|-------|-----------|-------------|
| `countries` | ~250 | ISO-coded country reference |
| `competitions` | ~500 | League/cup/tournament |
| `seasons` | ~5,000 | Time-bound grouping within competition |
| `teams` | ~10,000 | Club or national team |
| `stadiums` | ~5,000 | Venue info (city, capacity, surface) |
| `referees` | ~5,000 | Match officials |
| `players` | ~500,000 | Individual footballers |

### 2. Match Detail (5 tables)

| Table | Est. Rows | Link | Description |
|-------|-----------|------|-------------|
| `matches` | 10M+ | — | Central fact table (7 FKs) |
| `match_statistics` | = matches | 1:1 | Shots, possession, cards, corners |
| `weather` | = matches | 1:1 | Temperature, humidity, wind |
| `odds` | 50M+ | 1:N | Multi-bookmaker, multi-timestamp |
| `lineups` | 2× matches | 1:N | Formation, starting XI, substitutes |

### 3. Player Analytics (4 tables)

| Table | Est. Rows | Description |
|-------|-----------|-------------|
| `player_match_stats` | 100M+ | Per-match performance (goals, xG, rating) |
| `injuries` | 500K | Injury tracking (type, severity, return) |
| `transfers` | 200K | Transfer fees, loans, dates |

### 4. Team Analytics (3 tables)

| Table | Est. Rows | Description |
|-------|-----------|-------------|
| `team_elo_history` | 2× matches | Elo rating before/after each match |
| `team_form` | 2× matches | Rolling form at match time |
| `team_xg_history` | 2× matches | xG by source (opta, understat) |

### 5. Betting & Predictions (5 tables)

| Table | Est. Rows | Description |
|-------|-----------|-------------|
| `predictions` | = matches | Model probabilities, confidence |
| `expected_value_bets` | = matches | EV per match+bookmaker |
| `closing_line_values` | = matches | Opening-to-closing line movement |
| `betting_results` | = bets | Actual P&L tracking |

---

## Performance Optimizations

For large-scale queries, the schema uses:

- **Composite indexes** — `ix_matches_comp_season_date`, `ix_matches_home_date`, etc.
- **Partial indexes** — only index finished matches, positive EV bets
- **Table partitioning** — `matches`, `odds`, `player_match_stats` partitioned by month

---

## Full ER Diagram

See [[er_diagram]] for the complete 21-table ERD with all foreign keys and constraints.
