# Data Leakage Audit Checklist

## Usage
Run the automated audit first: `python scripts/audit_leakage.py --fix`

Then use this checklist for manual verification of the critical leakage-prone areas.

---

## 1. Rolling Team Features (`src/feature_engineering.py:_merge_team_stats`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 1.1 | All rolling averages use `.shift(1)` | **HIGH** | ✅ | Verified at lines 449, 457, 459 |
| 1.2 | Win rates use `.expanding().mean().shift(1)` | **HIGH** | ✅ | Lines 464, 466, 470 |
| 1.3 | `matches_this_season` uses `.shift(1)` | **HIGH** | ✅ | Lines 480, 484 |
| 1.4 | `days_since_last_match` uses `.diff().dt.days.shift(1)` | **HIGH** | ✅ | Line 488 |
| 1.5 | `home_matches`/`away_matches` use `.expanding().sum().shift(1)` | **HIGH** | ✅ | Lines 492-493 |
| 1.6 | Data is sorted chronologically before processing | **HIGH** | ✅ | Line 121 (`sort_values(["date", "home_team"])`) |

## 2. Head-to-Head Statistics (`src/feature_engineering.py:_compute_h2h_stats`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 2.1 | H2H stats use `.expanding().mean().shift(1)` | **HIGH** | ✅ | Lines 596-602 — all six H2H features |
| 2.2 | `h2h_matches_played` uses `.expanding().count().shift(1)` | **HIGH** | ✅ | Line 602 |
| 2.3 | Only past matches are used (group sorted by date) | **HIGH** | ✅ | Line 588 (`group.sort_values("date")`) |

## 3. League Position (`src/feature_engineering.py:_compute_league_positions`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 3.1 | Points computed before match (not including current) | **HIGH** | ✅ | Points retrieved via `_get_pts(home, 0)` before update |
| 3.2 | Position recorded before updating stats | **HIGH** | ✅ | Recorded at line 730, updated at line 745+ |
| 3.3 | Goal difference used for tie-breaking | **LOW** | ✅ | Used in standings sort key |

## 4. Target Encoding (`src/feature_engineering.py:_target_encode`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 4.1 | Expanding mean uses `.shift(1)` | **HIGH** | ✅ | Line 845 |
| 4.2 | First occurrence filled with global mean | **MEDIUM** | ✅ | Line 850 — uses pre-shift global mean |

## 5. Attack/Defence Ratios (`src/feature_engineering.py:_add_attack_defence_ratios`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 5.1 | League average is computed from past data only | **CRITICAL** | ✅ **FIXED** | Uses `_add_running_league_avg` (expanding window with `.shift(1)`) |
| 5.2 | Ratios use already-shifted rolling averages | **HIGH** | ✅ | Built on `_add_rolling_features` outputs |

## 6. Elo Ratings (`src/elo.py:process_matches`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 6.1 | Pre-match rating recorded before update | **HIGH** | ✅ | `R_home = _get_rating(home)` at line 431, update at line 449 |
| 6.2 | Post-match rating not used as feature | **HIGH** | ✅ | `update_ratings` returns pre-match values (docstring verified) |
| 6.3 | Host-nation bonus is temporary (not stored) | **LOW** | ✅ | Applied only to expected score, not stored rating |

## 7. Poisson Model (`src/poisson_model.py:add_poisson_features`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 7.1 | Expanding window: only past matches used | **HIGH** | ✅ | `team_stats` dict starts empty, grows per iteration |
| 7.2 | League averages computed from past matches only | **HIGH** | ✅ | `total_home_goals / total_matches` before current match is added |
| 7.3 | Team strengths computed from past matches only | **HIGH** | ✅ | `_strength()` reads from `team_stats` which has no current match data |

## 8. Dixon-Coles Model (`src/dixon_coles.py:add_features`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 8.1 | Refit cutoff excludes current match from training | **CRITICAL** | ✅ **FIXED** | `range(first_cutoff_pos)` and `range(last_filled_pos + 1, cutoff_pos)` exclude cutoff |
| 8.2 | Future matches not used in training | **HIGH** | ✅ | `_get_train_df(up_to_pos)` uses only matches up to positional index |
| 8.3 | Recency weighting uses only past data | **MEDIUM** | ✅ | Reference date is max(df[date_col]) + 1 day |

## 9. xG Features (`src/xg_features.py:_compute_rolling_xg`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 9.1 | Rolling xG uses `.shift(1)` | **HIGH** | ✅ | Lines 366, 372, 379 |
| 9.2 | xG Difference uses `.shift(1)` | **HIGH** | ✅ | Line 379 |
| 9.3 | Placeholder xG does not leak future data | **LOW** | ✅ | Zero-filled (no temporal info) |

## 10. Odds Processing (`src/odds_processing.py`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 10.1 | Odds data is pre-match only | **MEDIUM** | ✅ | Opening/closing odds available at kick-off |
| 10.2 | No rolling computations on odds | **LOW** | ✅ | All features per-match, not temporal |

## 11. Preprocessing (`src/preprocessing.py`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 11.1 | `target` column is derived from `result` (not used as feature) | **HIGH** | ✅ | `_get_target_columns` drops `result`, `target`, `home_goals`, `away_goals` |
| 11.2 | Duplicates removed chronologically | **LOW** | ✅ | Most recent keep, no future bias |
| 11.3 | Team names normalised (consistency) | **LOW** | ✅ | Name mapping applied |

## 12. Train/Test Split (`src/time_series_cv.py`)

| # | Check | Risk | Status | Notes |
|---|-------|------|--------|-------|
| 12.1 | `TimeSeriesSplit` used for CV | **HIGH** | ✅ | Expanding window (sklearn's implementation) |
| 12.2 | No shuffling in CV folds | **HIGH** | ✅ | `TimeSeriesSplit` has no shuffle param |
| 12.3 | Chronological train/val/test split available | **HIGH** | ✅ | `time_series_train_val_test_split` at line 385 |
| 12.4 | `train_val_test_split` in feature_engineering uses `shuffle=False` | **MEDIUM** | ✅ | Line 1039, 1043 |

---

## Summary

| Area | Critical | High | Medium | Low | Pass |
|------|----------|------|--------|-----|------|
| Rolling features | 0 | 6 | 0 | 0 | ✅ |
| H2H stats | 0 | 3 | 0 | 0 | ✅ |
| League position | 0 | 2 | 0 | 1 | ✅ |
| Target encoding | 0 | 1 | 1 | 0 | ✅ |
| Attack/defence ratios | 1→0 | 1 | 0 | 0 | ✅ (fixed) |
| Elo ratings | 0 | 2 | 0 | 1 | ✅ |
| Poisson model | 0 | 3 | 0 | 0 | ✅ |
| Dixon-Coles | 1→0 | 1 | 1 | 0 | ✅ (fixed) |
| xG features | 0 | 3 | 0 | 0 | ✅ |
| Odds processing | 0 | 0 | 1 | 0 | ✅ |
| Preprocessing | 0 | 1 | 0 | 2 | ✅ |
| Train/test split | 0 | 3 | 1 | 0 | ✅ |
| **Total** | **0** | **26** | **4** | **4** | **PASS** |

---

## Verification Steps

1. [ ] Run `python scripts/audit_leakage.py` – should report PASS
2. [ ] Run `python scripts/audit_leakage.py --fix` to apply any outstanding fixes
3. [ ] Verify feature matrix columns: no `result`, `home_goals`, `away_goals`
4. [ ] Verify Feature Store re-run after fixes: `python -m src.feature_store.cli compute-all`
5. [ ] Re-run model training to validate no regression: `python src/train.py`

_Last updated: 2026-07-14_
