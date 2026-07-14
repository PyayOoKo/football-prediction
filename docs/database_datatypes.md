# Database Data Type Optimization

> Audit of current types vs. optimal types for 100M+ row scale.

---

## Recommendations Summary

| Change | Count | Space Saving | Risk |
|--------|-------|-------------|------|
| `Float` → `Numeric` (monetary/odds) | 8 tables | N/A (correctness) | Low |
| `Integer` → `BigInteger` (PKs) | 3 tables | N/A (overflow safety) | Medium |
| `Integer` → `SmallInteger` (counts) | 6 tables | ~40% per column | Low |
| `String(32)` → `String(64)` | 1 column | N/A (correctness) | Low |

---

## 1. PK Overflow Protection

`Integer` PKs overflow at 2.1 billion rows. At 100M+ matches, large tables (player_match_stats, odds) are at risk.

| Table | Current | Recommended | Rationale |
|-------|---------|-------------|-----------|
| `matches` | `BigInteger` ✅ | — | Already correct |
| `odds` | `BigInteger` ✅ | — | Already correct |
| `player_match_stats` | `BigInteger` ✅ | — | Already correct (m006) |
| `match_statistics` | `Integer` | `BigInteger` | 1:1 with matches, 50M+ rows |
| `predictions` | `Integer` | `BigInteger` | ~5 predictions per match = 500M rows |
| `betting_results` | `Integer` | `BigInteger` | ~10 bets per match = 1B rows |
| `expected_value_bets` | `Integer` | `BigInteger` | ~10 EV bets per match = 1B rows |
| `closing_line_values` | `Integer` | `BigInteger` | ~5 per match = 500M rows |
| `team_elo_history` | `Integer` | `BigInteger` | 2 per match = 200M rows |
| `team_form` | `Integer` | `BigInteger` | 2 per match = 200M rows |
| `team_xg_history` | `Integer` | `BigInteger` | 2 per match = 200M rows |
| `lineups` | `Integer` | `BigInteger` | ~2 per match = 200M rows |
| `weather` | `Integer` | `BigInteger` | 1:1 with matches = 100M rows |

**Note:** These are optional until row counts approach 1B. The current `Integer` PKs will last for years at current data volumes.

---

## 2. Float → Numeric (Monetary/Odds)

`Float` stores approximate values. Monetary amounts and decimal odds need exact precision.

### High Priority (money/odds)

| Table | Columns | Current | Recommended |
|-------|---------|---------|-------------|
| `odds` | `odds_home, odds_draw, odds_away` | `Float` | `Numeric(6,2)` |
| `odds` | `implied_prob_home, implied_prob_draw, implied_prob_away` | `Float` | `Numeric(5,4)` |
| `odds` | `margin` | `Float` | `Numeric(5,4)` |
| `betting_results` | `decimal_odds` | `Float` | `Numeric(6,2)` |
| `betting_results` | `stake, profit` | `Float` | `Numeric(10,2)` |
| `betting_results` | `roi_pct` | `Float` | `Numeric(6,2)` |
| `expected_value_bets` | `model_prob_*, book_prob_*` | `Float` | `Numeric(5,4)` |
| `expected_value_bets` | `ev_*, kelly_stake_*` | `Float` | `Numeric(10,4)` |
| `closing_line_values` | `opening_price, closing_price` | `Float` | `Numeric(6,2)` |
| `closing_line_values` | `clv` | `Float` | `Numeric(6,4)` |
| `predictions` | `prob_home, prob_draw, prob_away` | `Float` | `Numeric(5,4)` |
| `predictions` | `expected_value` | `Float` | `Numeric(10,4)` |
| `predictions` | `kelly_stake` | `Float` | `Numeric(10,2)` |
| `predictions` | `confidence` | `Float` | `Numeric(5,4)` |
| `players` | `market_value_eur` | `Float` | `Numeric(12,2)` |
| `transfers` | `transfer_fee_eur` | `Float` | `Numeric(14,2)` |

### Medium Priority (percentages/ratios)

| Table | Columns | Current | Recommended |
|-------|---------|---------|-------------|
| `player_match_stats` | `pass_accuracy` | `Float` | `Numeric(5,2)` |
| `match_statistics` | `home_possession, away_possession` | `Float` | `Numeric(5,2)` |

---

## 3. Integer → SmallInteger (Counts)

`SmallInteger` (2 bytes, ±32K) is sufficient for count fields. `Integer` (4 bytes) wastes space.

| Table | Columns | Max Expected | Space Saved |
|-------|---------|-------------|-------------|
| `matches` | `home_goals, away_goals` | 30 | 50% per value |
| `matches` | `attendance` | 200,000 | Keep Integer |
| `match_statistics` | `*_shots, *_corners, *_fouls, *_cards, *_offsides, *_shots_inside_box, *_shots_outside_box` | 50 | 50% per value |
| `player_match_stats` | `minutes_played` | 120 | 50% |
| `player_match_stats` | `goals, assists, shots, shots_on_target, passes, tackles, interceptions, fouls_committed, fouls_drawn, saves` | 50 | 50% |
| `player_match_stats` | `position` | — | Keep String(8) |
| `competitions` | `level` | 20 | 50% |
| `players` | `height_cm` | 250 | 50% |
| `players` | `weight_kg` | 150 | 50% |
| `players` | `shirt_number` | 99 | 50% |
| `injuries` | `missed_matches` | 100 | 50% |
| `lineups` | `substitutions_made` | 5 | 50% |
| `team_form` | `last_5_wins, draws, losses, clean_sheets, btts` | 5 | 50% |
| `team_form` | `season_matches_played` | 60 | 50% |
| `team_xg_history` | `shots, shots_on_target, deep_completions` | 50 | 50% |
| `weather` | `humidity_pct` | 100 | 50% |

---

## 4. Date vs. Timestamp Audit

All date-only columns correctly use `Date`:

| Table | Column | Type | Correct? |
|-------|--------|------|----------|
| `matches` | `match_date` | `Date` | ✅ |
| `seasons` | `start_date` | `Date` | ✅ |
| `seasons` | `end_date` | `Date` | ✅ |
| `players` | `date_of_birth` | `Date` | ✅ |
| `injuries` | `injury_date` | `Date` | ✅ |
| `injuries` | `expected_return` | `Date` | ✅ |
| `injuries` | `actual_return` | `Date` | ✅ |
| `transfers` | `transfer_date` | `Date` | ✅ |
| `transfers` | `loan_end_date` | `Date` | ✅ |
| All | `created_at` | `Timestamptz` | ✅ (needs time) |
| All | `updated_at` | `Timestamptz` | ✅ (needs time) |

---

## 5. String Length Audit

| Table | Column | Current | Recommended | Reason |
|-------|--------|---------|-------------|--------|
| `seasons` | `name` | `VARCHAR(64)` ✅ | — | Already corrected |
| `matches` | `result` | `VARCHAR(4)` | `CHAR(1)` | Only 'H', 'D', 'A' |
| `matches` | `duration` | `VARCHAR(8)` | `VARCHAR(12)` | Extra values possible |
| `matches` | `status` | `VARCHAR(16)` | ✅ | 6 values, well-sized |
| `matches` | `round` | `VARCHAR(32)` | `VARCHAR(64)` | "Semi-final" needs >32 |
| `odds` | `source` | `VARCHAR(32)` | ✅ | Bookmaker names ≤32 |
| `teams` | `name` | `VARCHAR(128)` | ✅ | Club names can be long |
| `teams` | `short_name` | `VARCHAR(8)` | ✅ | "ARS", "MCI" |

---

## 6. Boolean Audit

All binary flags correctly use `Boolean`:

| Table | Column | Type | Correct? |
|-------|--------|------|----------|
| `matches` | `is_neutral_venue` | `Boolean` | ✅ |
| `player_match_stats` | `is_starter` | `Boolean` | ✅ |
| `player_match_stats` | `yellow_card` | `Boolean` | ✅ |
| `player_match_stats` | `red_card` | `Boolean` | ✅ |

---

## Estimated Space Savings

| Optimization | Tables Affected | Est. Saving (100M matches) |
|-------------|-----------------|---------------------------|
| `Float` → `Numeric` | 8 | Minimal (correctness only) |
| `Integer` → `SmallInteger` | 6 | ~200 MB |
| `Integer` → `BigInteger` (PKs) | 10 | ~500 MB (safety) |
| Total | 24 | ~700 MB + correctness |

**Current total database size estimate at 100M matches:** ~1.5 TB (data + indexes).
**Post-optimization estimate:** ~1.49 TB (marginal savings, but correctness guaranteed).
