# Database Indexes

> All indexes across 22 tables, organized by table and index type.

---

## Index Strategy

| Type | Count | Use Case |
|------|-------|----------|
| B-tree (PK) | 22 | Primary key lookups |
| B-tree (FK) | 18 | Foreign key joins |
| B-tree (unique) | 6 | Uniqueness enforcement |
| B-tree (composite) | 12 | Multi-column query patterns |
| B-tree (covering) | 10 | Index-only scans |
| B-tree (partial) | 8 | Filtered queries |
| BRIN | 1 | Large time-range scans |

---

## 1. `matches` — 9 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_matches` | B-tree PK | `id` | — | — | Primary key |
| `ix_matches_competition_id` | B-tree FK | `competition_id` | — | — | Competition joins |
| `ix_matches_season_id` | B-tree FK | `season_id` | — | — | Season joins |
| `ix_matches_home_team_id` | B-tree FK | `home_team_id` | — | — | Home team joins |
| `ix_matches_away_team_id` | B-tree FK | `away_team_id` | — | — | Away team joins |
| `ix_matches_stadium_id` | B-tree partial | `stadium_id` | — | `stadium_id IS NOT NULL` | Stadium joins |
| `ix_matches_referee_id` | B-tree partial | `referee_id` | — | `referee_id IS NOT NULL` | Referee joins |
| `ix_matches_match_date` | B-tree | `match_date` | — | — | Date-range queries |
| `ix_matches_match_date_brin` | BRIN | `match_date` | — | — | Large date-range scans |
| `ix_matches_comp_season_date` | B-tree composite | `competition_id, season_id, match_date` | — | — | League standings |
| `ix_matches_home_date` | B-tree composite | `home_team_id, match_date` | `home_goals, away_goals, result` | — | Team home history |
| `ix_matches_away_date` | B-tree composite | `away_team_id, match_date` | `home_goals, away_goals, result` | — | Team away history |
| `ix_matches_upcoming` | B-tree partial | `match_date` | — | `status = 'scheduled'` | Dashboard |
| `ix_matches_completed` | B-tree partial | `result` | — | `result IS NOT NULL` | Completed matches |
| `ix_matches_live` | B-tree partial | `match_date` | — | `status = 'live'` | Live matches |

---

## 2. `odds` — 4 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_odds` | B-tree PK | `id` | — | — | Primary key |
| `ix_odds_match_id` | B-tree FK | `match_id` | — | — | Match joins |
| `uq_odds_match_source_time` | B-tree unique | `match_id, source, timestamp` | — | — | Dedup |
| `ix_odds_match_source_covering` | B-tree covering | `match_id, source, timestamp` | `odds_home, odds_draw, odds_away, implied_prob_home, margin` | — | Odds analysis |
| `ix_odds_pinnacle` | B-tree partial | `match_id` | `odds_home, odds_draw, odds_away` | `source = 'Pinnacle'` | Value betting |

---

## 3. `predictions` — 3 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_predictions` | B-tree PK | `id` | — | — | Primary key |
| `ix_predictions_match_id` | B-tree FK | `match_id` | — | — | Match joins |
| `ix_predictions_match_model` | B-tree covering | `match_id, model_name, model_version` | `prob_home, prob_draw, confidence` | — | Model analysis |
| `ix_predictions_by_model` | B-tree partial | `model_name, match_id DESC` | `prob_home, prob_draw, prob_away, confidence` | — | Model-filtered queries |

---

## 4. `player_match_stats` — 3 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_pms` | B-tree PK | `id` | — | — | Primary key |
| `ix_pms_match_id` | B-tree FK | `match_id` | — | — | Match joins |
| `uq_pms_match_player` | B-tree unique | `match_id, player_id` | — | — | Dedup |
| `ix_pms_player_history` | B-tree covering | `player_id, match_id` | `goals, assists, shots, xg, xa, rating, minutes_played` | — | Player history |

---

## 5. `team_elo_history` — 4 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_team_elo_history` | B-tree PK | `id` | — | — | Primary key |
| `ix_team_elo_team_id` | B-tree FK | `team_id` | — | — | Team joins |
| `ix_team_elo_chrono` | B-tree covering | `team_id, match_id` | `elo_before, elo_after, elo_change, side` | — | Elo timeline |
| `ix_team_elo_home` | B-tree partial | `team_id, match_id DESC` | `elo_before, elo_after` | `side = 'home'` | Home Elo history |
| `ix_team_elo_away` | B-tree partial | `team_id, match_id DESC` | `elo_before, elo_after` | `side = 'away'` | Away Elo history |

---

## 6. `team_form` — 3 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_team_form` | B-tree PK | `id` | — | — | Primary key |
| `ix_team_form_team_id` | B-tree FK | `team_id` | — | — | Team joins |
| `ix_team_form_chrono` | B-tree covering | `team_id, match_id` | `last_5_ppg, last_5_goals_scored, last_5_goals_conceded, side` | — | Form timeline |

---

## 7. `team_xg_history` — 3 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_team_xg_history` | B-tree PK | `id` | — | — | Primary key |
| `ix_team_xg_team_id` | B-tree FK | `team_id` | — | — | Team joins |
| `ix_team_xg_chrono` | B-tree covering | `team_id, match_id, source` | `xg, xa, shots, side` | — | xG timeline |

---

## 8. `betting_results` — 3 indexes

| Index Name | Type | Columns | INCLUDE | WHERE | Purpose |
|------------|------|---------|---------|-------|---------|
| `pk_betting_results` | B-tree PK | `id` | — | — | Primary key |
| `ix_betting_results_match_id` | B-tree FK | `match_id` | — | — | Match joins |
| `ix_betting_results_strategy_date` | B-tree covering | `match_id, strategy, created_at` | `stake, profit, roi_pct, won` | — | Strategy analysis |
| `ix_betting_results_completed` | B-tree partial | `created_at` | `profit, roi_pct` | `won IS NOT NULL` | Completed bets |

---

## 9. Reference Tables — 1 index each

| Table | Index | Column |
|-------|-------|--------|
| `countries` | PK | `id` |
| `competitions` | PK, FK | `id`, `country_id` |
| `seasons` | PK, FK, UNIQUE | `id`, `competition_id`, `(competition_id, name)` |
| `teams` | PK, FK, FK, UNIQUE | `id`, `country_id`, `stadium_id`, `name` |
| `stadiums` | PK, FK | `id`, `country_id` |
| `referees` | PK | `id` |
| `players` | PK, FK, UNIQUE | `id`, `current_team_id`, `name` |

---

## Size Estimates at 100M Matches

| Table | Index Count | Est. Index Size | Most Expensive Index |
|-------|-------------|-----------------|---------------------|
| matches | 9 | ~2.5 GB | `ix_matches_home_date` (covering, ~800 MB) |
| odds | 4 | ~6 GB | `ix_odds_match_source_covering` (~3 GB) |
| predictions | 3 | ~1.5 GB | `ix_predictions_match_model` (~700 MB) |
| player_match_stats | 3 | ~8 GB | `ix_pms_player_history` (~5 GB) |
| team_elo_history | 4 | ~1.5 GB | `ix_team_elo_home` (~500 MB) |
| team_form | 3 | ~1.5 GB | `ix_team_form_chrono` (~500 MB) |
| team_xg_history | 3 | ~1.5 GB | `ix_team_xg_chrono` (~500 MB) |
| betting_results | 3 | ~3 GB | `ix_betting_results_strategy_date` (~2 GB) |
| Reference (7) | 12 | <100 MB | — |
| **Total** | **~50** | **~25 GB** | |
