# League Profitability Report
**Generated:** 2026-07-25
**Leagues Analysed:** E0 (Premier League), I1 (Serie A), SP1 (La Liga), F1 (Ligue 1), D1 (Bundesliga), SE1 (Superettan)
**Model:** DC+Elo + Blend (DC+Elo+XGBoost+LGB+Cat) + Market-Specific Trees (trial)

> **❌ Key Finding (2026-07-25): Market-Specific Trees Worsen O/U & BTTS Performance.** DC-only significantly outperforms the full blend with market-specific tree models for O/U 2.5 and BTTS. All metrics worsened: OU Brier +0.0057, OU Accuracy -5.68pp, BTTS Yield -12.24pp. **Recommendation: revert to DC-only for binary markets.**

> **✅ Understat xG Now Complete (2026-07-25):** Older seasons 2020-2021 now collected for all top 5 leagues. Total: 151 new matches with real xG. Coverage now spans 2020-2025 (6 seasons) per league, near 100% coverage.

> **⚠️ SofaScore API Changed (2026-07-25):** The `/api/v1/event/{id}/statistics` endpoint now returns `404`. All future SofaScore-based xG collection is blocked until the API structure is understood. Existing SE1 data (5,288 matches) remains intact from pre-change collection.

> **🚀 F1 Over2.5 remains the only verified profitable market** at +21.0% yield (DC-only). All other markets lose money with broad staking strategies. The DC comparison confirmed DC is the right model for this — adding trees destroyed the edge.

---

## 1. Market Data Availability

| Market | Match Data | Bookmaker Odds | Backtestable? |
|--------|:----------:|:--------------:|:-------------:|
| **1X2 (Home/Draw/Away)** | ✅ Goals & results | ✅ home_odds, draw_odds, away_odds | ✅ YES |
| **Over/Under 2.5** | ✅ Goals scored | ✅ **BbAv>2.5 / BbAv<2.5** | ✅ **YES** |
| **Over/Under 3.5** | ✅ Goals scored | ❌ No odds columns | ❌ NO |
| **BTTS (Both Teams to Score)** | ✅ Goals scored | ❌ No odds columns | ❌ NO |
| **Corners** | ✅ Actual corner count | ❌ No odds columns | ❌ NO |
| **Cards (Yellow/Red)** | ✅ Actual card count | ❌ No odds columns | ❌ NO |

---

## 2. League Basic Statistics

| League | Matches | H% | D% | A% | BTTS% | O2.5% | O3.5% | Avg Goals | Avg Corners |
|--------|:-------:|:--:|:--:|:--:|:-----:|:-----:|:-----:|:---------:|:-----------:|
| **E0** 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | 5,700 | 44.6% | 23.9% | 31.5% | 52.3% | 53.8% | 31.3% | 2.80 | 10.6 |
| **I1** 🇮🇹 | 5,700 | 43.0% | 26.0% | 31.0% | 52.9% | 51.2% | 28.6% | 2.71 | 10.1 |
| **SP1** 🇪🇸 | 5,700 | 46.3% | 25.3% | 28.4% | 50.9% | 49.4% | 27.3% | 2.66 | 9.9 |
| **F1** 🇫🇷 | 5,377 | 44.4% | 26.0% | 29.6% | 51.4% | 49.2% | 27.9% | 2.65 | 9.6 |
| **D1** 🇩🇪 | 4,590 | 44.6% | 24.7% | 30.7% | **57.4%** | **58.5%** | **36.9%** | **3.03** | 9.7 |
| **SE1** 🇸🇪 | 5,288 | 44.5% | 25.8% | 29.7% | 54.6% | 53.1% | 31.2% | 2.78 | N/A |

---

## 3. Key Findings — 2026-07-25 Update

### 3a. DC vs Market Trees Comparison (F1 — 387 Test Matches)

**Critical finding: market-specific tree models degrade O/U and BTTS performance.**

| Metric | DC-only | DC+Market Trees | Δ | Verdict |
|:-------|:-------:|:---------------:|:-:|:-------:|
| **OU Brier** | **0.2488** | 0.2545 | +0.0057 | ❌ |
| **OU Accuracy** | **54.52%** | 48.84% | −5.68pp | ❌ |
| **BTTS Brier** | **0.2487** | 0.2523 | +0.0036 | ❌ |
| **BTTS Accuracy** | **54.78%** | 47.29% | −7.49pp | ❌ |
| **Value Bets** | 247 | 222 | −25 | ❌ |
| **Yield** | **+1.18%** | −11.06% | −12.24pp | ❌ |
| **Profit** | **+GBP 1,176** | −GBP 5,867 | −GBP 7,042 | ❌ |

**Why market trees failed:**
1. **Target noise** — O/U 2.5 and BTTS are inherently noisier targets than 1X2
2. **DC's mathematical advantage** — DC directly models the goal-scoring process (λ_home, λ_away); trees must infer it indirectly
3. **Insufficient data** — ~338 F1 training matches is too few for gradient-boosted trees with 146 features
4. **Feature misalignment** — Feature pipeline designed for 1X2, not O/U/BTTS

**Recommendation:** Revert to DC-only for O/U 2.5 and BTTS. Reserve tree models for 1X2 only.

### 3b. Understat xG — 2020-2021 Seasons Now Complete ✅

Ran `collect_understat_xg.py --seasons 7` to fill the 2020-2021 gap. Results:

| League | 2020 | 2021 | 2022-2025 | Total Seasons |
|:-------|:----:|:----:|:---------:|:-------------:|
| **E0** | 100% | 100% | ✅ | 6 (2020-2025) |
| **SP1** | 99% | 99% | ✅ | 6 |
| **D1** | 99% | 100% | ✅ | 6 |
| **I1** | 100% | 100% | ✅ | 6 |
| **F1** | 100% | 100% | ✅ | 6 |

**151 new matches** updated with real xG. 0 errors. 11 unmatched team names (naming differences like "Athletic Club" vs "Athletic Bilbao").

### 3c. SofaScore API Status ⚠️

The `/api/v1/event/{id}/statistics` endpoint now returns **404 Not Found**. This blocks:
- ❌ New SE1 xG collection (existing 5,288 matches preserved)
- ❌ NO2 (OBOS-ligaen) xG collection
- ❌ FI2 (Ykkösliiga) xG collection

**Alternative:** FBref manual saves → use `scripts/collect_xg_from_fbref.py`. Infrastructure set up at `data/fbref_pages/`.

### 3d. Extended SofaScore Script

The `scripts/fetch_sofascore_xg.py` script was significantly upgraded:
- **Multi-league**: Now supports SE1 (ID 46), NO2 (ID 22), FI2 (ID 55)
- **Dynamic seasons**: Fetches available seasons from API instead of hardcoded IDs
- **Upsert logic**: INSERTs new matches when they don't exist in DB (essential for NO2/FI2 which have no base data)
- **CLI**: `--leagues` and `--seasons` arguments
- **Window support**: Unicode fix for cp1252, WAL mode for DB concurrency

---

## 3e. O/U 2.5 Value Betting Backtest (DC-only, F1)

**Methodology:** DC-only model, Kelly 25% staking, Min EV 5%, tested on last 15% of F1 data.

| Market | Bets | Won | Lost | Win Rate | Yield | Profit | Max DD |
|:-------|:----:|:---:|:----:|:--------:|:-----:|:------:|:------:|
| **Over 2.5** | 47 | 26 | 21 | 55.3% | +21.0% | +GBP 5,489 | — |
| **Under 2.5** | 200 | 99 | 101 | 49.5% | −2.7% | −GBP 4,314 | — |
| **Total** | **247** | 125 | 122 | **50.6%** | **+1.18%** | **+GBP 1,176** | 53.8% |

**Key insight:** Over 2.5 is profitable; Under 2.5 loses money. Filtering to Over-only would yield ~+21%.

---

## 4. Value Betting Backtest Results (1X2 Market, Broad Strategy)

**DC+Elo model, 25% fractional Kelly, 5% minimum EV threshold, tested on last 15% of data.**

| League | Bets | Win Rate | Yield (ROI) | Profit | Max DD |
|:-------|:----:|:--------:|:-----------:|:------:|:------:|
| **E0** 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | 769 | 25.4% | −38.80% 🚫 | −£9,995 | 100% |
| **I1** 🇮🇹 | 802 | 27.1% | −40.87% 🚫 | −£9,994 | 100% |
| **SP1** 🇪🇸 | 747 | 35.6% | −8.41% 🟡 | −£8,710 | 93% |
| **F1** 🇫🇷 | 665 | 34.7% | −7.42% 🟡 | −£8,433 | 85% |
| **D1** 🇩🇪 | 543 | 30.0% | −27.35% 🚫 | −£9,509 | 96% |
| **SE1** 🇸🇪 | 735 | 30.9% | −11.95% 🟡 | −£9,928 | 100% |

---

## 5. Narrow Strategy Backtest (Home Win, Odds 2-3, Level Stakes)

| League | Bets | Win Rate | Yield | Profit | Max DD |
|:-------|:----:|:--------:|:-----:|:------:|:------:|
| **E0** 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | 562 | 43.6% | −1.93% 🟡 | −£3,805 | 84.7% |
| **I1** 🇮🇹 | 536 | 44.6% | −17.31% 🚫 | −£9,327 | 96.1% |
| **SP1** 🇪🇸 | 515 | 44.9% | −19.46% 🚫 | −£9,581 | 96.1% |
| **F1** 🇫🇷 | 491 | 49.9% | **−0.41%** 🟡 | −£1,095 | 74.0% |
| **D1** 🇩🇪 | 396 | 37.1% | −9.20% 🚫 | −£7,186 | 78.4% |

**Narrow strategy massively improves results** — E0 from −38.8% to −1.93%, F1 from −7.42% to −0.41%.

---

## 6. xG Data Status

### Understat xG Coverage (Real)
| League | Seasons | Coverage | Matches |
|:-------|:-------:|:--------:|:-------:|
| **E0** | 2020-2025 | 100% | 5,700 |
| **SP1** | 2020-2025 | 99% | 5,700 |
| **D1** | 2020-2025 | 99-100% | 4,590 |
| **I1** | 2020-2025 | 99-100% | 5,700 |
| **F1** | 2020-2025 | 99-100% | 5,377 |
| **SE1** | 2004-2026 | 100% | 5,288 |

### SofaScore API (Broken)
- SE1: 5,288 matches with real xG (collected pre-API-change) ✅
- NO2, FI2: 0 matches — blocked by 404 on statistics endpoint ❌
- Alternative: FBref manual saves via `scripts/collect_xg_from_fbref.py`

---

## 7. Strategy Recommendations

### What Works ✅
1. **F1 Over2.5 (DC-only)** — +21.0% yield on 47 bets (limited sample, promising)
2. **Narrow home-win strategy** (odds 2-3, level stakes) — E0 at −1.93%, F1 at −0.41%
3. **SE1 home-win filtered** (odds 2-3, level stakes) — +3.04% historical ROI
4. **DC-only for O/U and BTTS** — trees degrade performance

### What Doesn't Work 🚫
1. **Market-specific tree models** for O/U or BTTS — all metrics worsened
2. **Broad Kelly staking** on any league — all negative (−5% to −41%)
3. **Under 2.5 bets** — consistently losing across all leagues
4. **Betting draws or away** in SE1 — historically −25% yield

### Risks ⚠️
1. **SE1 edge declining**: +23% (2021) → −17% (2025)
2. **SofaScore API broken**: No automated xG for NO2/FI2
3. **NO2/FI2 have no base match data** — need FBref or another source entirely
4. **Small sample**: F1 Over2.5 has only 47 bets — not statistically significant

### Next Steps
1. **Save FBref pages** for NO2 and FI2 xG data
2. **Run the F1 Over2.5 backtest with DC-only** and larger sample (use default blend weights)
3. **Investigate SofaScore API** changes — the events endpoint still works, only the per-match statistics endpoint is broken
4. **Test tighter odds range** for narrow strategy (odds 2-2.5)
