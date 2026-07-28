"""
compare_se1_calibration.py — Compare raw vs calibrated predictions on the last 100 SE1 matches.

Loads the 4-model blend, gets predictions for the most recent 100 completed matches,
applies all three calibrators, and reports which method is most accurate.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_calibration")

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODEL_DIR = PROJECT_ROOT / "models" / "per_league" / "SE1"

EQUAL_WEIGHTS = {
    "1X2": {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
}


# Load data
SEP = "=" * 70
SUB = "-" * 40

print(SEP)
print("  SE1 CALIBRATION COMPARISON -- Last 100 Matches")
print(SEP)

# 1. Load data
conn = sqlite3.connect(str(DB_PATH))
df = pd.read_sql_query(
    """SELECT * FROM matches
       WHERE league = 'SE1'
         AND home_goals IS NOT NULL AND away_goals IS NOT NULL
         AND result IN ('H', 'D', 'A')
       ORDER BY date ASC""",
    conn,
)
conn.close()
print(f"  Total SE1 matches: {len(df)}")
print(f"  Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

# 2. Take last 100 matches for evaluation
test_df = df.tail(100).copy()
y_true = test_df["result"].map({"A": 0, "D": 1, "H": 2}).values
print(f"  Last 100 matches: {test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]}")

# 3. Train data = everything before last 100
train_df = df.iloc[:-100].copy()

# 4. Load blend + calibrators
from src.dixon_coles import DixonColesModel
from src.elo import EloSystem
from src.models.three_model_blend import ThreeModelBlend, ConditionalRates
from src.form_adjuster import RecentFormAdjuster

dc = joblib.load(MODEL_DIR / "dixon_coles.joblib")
elo = joblib.load(MODEL_DIR / "elo.joblib")
xgb = joblib.load(MODEL_DIR / "xgboost.joblib")
lgb = joblib.load(MODEL_DIR / "lightgbm.joblib")

# Retrain Elo on the full training set (so ratings are current for last 100 matches)
elo_full = EloSystem(k=32, home_advantage=100, initial_rating=1500)
elo_full.process_matches(train_df)

# Form adjuster
adjuster = RecentFormAdjuster(n_matches=6, form_weight=50.0)
adjuster.fit(train_df)

# Blends
cond_rates = ConditionalRates.from_data(train_df)
blend_raw = ThreeModelBlend(
    dc_model=dc, elo_model=elo_full, xgb_model=xgb, lgb_model=lgb,
    weights=EQUAL_WEIGHTS, conditional_rates=cond_rates, historical_df=train_df,
)
blend_form = ThreeModelBlend(
    dc_model=dc, elo_model=elo_full, xgb_model=xgb, lgb_model=lgb,
    weights=EQUAL_WEIGHTS, conditional_rates=cond_rates, historical_df=train_df,
)
blend_form.form_adjuster = adjuster

# Load calibrators
calibrators = {}
for method in ["platt", "isotonic", "hybrid"]:
    cal_path = MODEL_DIR / f"blend_calibrator_{method}.joblib"
    if cal_path.exists():
        calibrators[method] = joblib.load(cal_path)

print(f"  Calibrators loaded: {list(calibrators.keys())}")
print()

# 5. Evaluate: simulate match-by-match prediction
from sklearn.metrics import log_loss, brier_score_loss

def evaluate_predictions(probs_list, y_true, label):
    """Evaluate a list of (n,3) probability arrays."""
    acc = float(np.mean(np.argmax(probs_list, axis=1) == y_true))
    ll = float(log_loss(y_true, probs_list))
    y_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs_list - y_onehot) ** 2, axis=1)))
    return {"accuracy": acc, "log_loss": ll, "brier": brier}

# Collect predictions for each method
methods = {"Raw": [], "Form-Adjusted": []}
for m in calibrators:
    methods[f"Calibrated ({m})"] = []

all_home_teams = test_df["home_team"].tolist()
all_away_teams = test_df["away_team"].tolist()

for i in range(len(test_df)):
    ht = all_home_teams[i]
    at = all_away_teams[i]
    actual = y_true[i]

    # Raw blend
    raw_1x2 = blend_raw.predict_1x2(ht, at)
    raw_arr = np.array([raw_1x2["A"], raw_1x2["D"], raw_1x2["H"]])
    methods["Raw"].append(raw_arr)

    # Form-adjusted
    form_1x2 = blend_form.predict_1x2(ht, at)
    form_arr = np.array([form_1x2["A"], form_1x2["D"], form_1x2["H"]])
    methods["Form-Adjusted"].append(form_arr)

    # Calibrated
    for cal_name, calibrator in calibrators.items():
        cal_arr = calibrator.transform(raw_arr.reshape(1, -1))[0]
        methods[f"Calibrated ({cal_name})"].append(cal_arr)

# Compute metrics
results = {}
for name, probs_list in methods.items():
    if probs_list:
        probs_arr = np.array(probs_list)
        results[name] = evaluate_predictions(probs_arr, y_true, name)

# Print comparison table
print(f"  {'Method':30s} {'Accuracy':>10s} {'LogLoss':>10s} {'Brier':>10s}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

best_acc = ("", 0)
best_ll = ("", float("inf"))
best_brier = ("", float("inf"))

for name, r in sorted(results.items(), key=lambda x: x[1]["brier"]):
    marker = " "
    print(f"  {marker} {name:28s} {r['accuracy']*100:>8.2f}%  {r['log_loss']:>8.4f}  {r['brier']:>8.4f}")
    if r["accuracy"] > best_acc[1]:
        best_acc = (name, r["accuracy"])
    if r["log_loss"] < best_ll[1]:
        best_ll = (name, r["log_loss"])
    if r["brier"] < best_brier[1]:
        best_brier = (name, r["brier"])

print()
print(f"  >> Best Accuracy:  {best_acc[0]} ({best_acc[1]*100:.2f}%)")
print(f"  >> Best LogLoss:   {best_ll[0]} ({best_ll[1]:.4f})")
print(f"  >> Best Brier:     {best_brier[0]} ({best_brier[1]:.4f})")

# Value betting simulation
print(f"\n{SUB}")
print("  VALUE BETTING SIMULATION (last 100 matches)")
print(SUB)
print(f"  Using actual SE1 odds from database (simulating GBP 10,000 bankroll)")
print(f"  Kelly fraction: 25%  |  Min EV: 2%")

# For value betting, we need odds - check how many have odds
with_odds = test_df[test_df["home_odds"].notna() & test_df["draw_odds"].notna() & test_df["away_odds"].notna()]
print(f"  Matches with odds data: {len(with_odds)}/{len(test_df)}")

if len(with_odds) >= 10:
    for name, probs_list in methods.items():
        if not probs_list:
            continue
        probs_arr = np.array(probs_list)
        bankroll = 10000.0
        bets = 0
        wins = 0
        total_profit = 0.0
        
        for i, (idx, row) in enumerate(with_odds.iterrows()):
            match_idx = test_df.index.get_loc(idx)
            if match_idx >= len(probs_arr):
                continue
            
            probs = probs_arr[match_idx]
            result = row["result"]
            actual_idx = {"A": 0, "D": 1, "H": 2}[result]
            
            outcomes = [
                ("H", 2, float(row["home_odds"]), probs[2]),
                ("D", 1, float(row["draw_odds"]), probs[1]),
                ("A", 0, float(row["away_odds"]), probs[0]),
            ]
            
            for label, idx_outcome, odds, prob in outcomes:
                if odds <= 1 or prob <= 0:
                    continue
                implied = 1.0 / odds
                ev = prob / implied - 1.0
                if ev > 0.02:
                    kelly = (prob * odds - 1.0) / (odds - 1.0)
                    kelly = max(0.0, min(kelly, 0.25)) * bankroll
                    if kelly > 0 and bankroll > 1:
                        won = idx_outcome == actual_idx
                        profit = kelly * (odds - 1.0) if won else -kelly
                        bankroll += profit
                        bets += 1
                        if won:
                            wins += 1
                        total_profit += profit
        
        if bets > 0:
            roi = (total_profit / (bets * (bankroll / len(with_odds)))) if bets < len(with_odds) else 0
            print(f"  {name:28s}: {bets:3d} bets, {wins:2d}W/{bets-wins:2d}L, "
                  f"profit=GBP {total_profit:+.0f}, bankroll=GBP {bankroll:.0f}, "
                  f"WR={wins/bets*100:.0f}%")
else:
    # Simulate with fair odds from raw model + 5% overround
    print("  (No real odds in DB -- simulating with +5% overround)")
    for name, probs_list in methods.items():
        if not probs_list:
            continue
        probs_arr = np.array(probs_list)
        bankroll = 10000.0
        bets = 0
        wins = 0
        
        for i in range(len(test_df)):
            probs = probs_arr[i]
            result = test_df.iloc[i]["result"]
            actual_idx = {"A": 0, "D": 1, "H": 2}[result]
            
            # Simulate odds: fair + 5% overround
            total = probs.sum()
            fair = probs / total
            implied = fair * 1.05
            odds_arr = np.clip(1.0 / implied, 1.01, 50.0)
            
            for idx_outcome in range(3):
                decimal_odds = odds_arr[idx_outcome]
                if not np.isfinite(decimal_odds) or decimal_odds <= 1.0:
                    continue
                prob = probs[idx_outcome]
                implied_prob = 1.0 / decimal_odds
                ev = prob / implied_prob - 1.0
                if ev > 0.02:
                    kelly = (prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
                    kelly = max(0.0, min(kelly, 0.25)) * bankroll
                    if kelly > 0 and bankroll > 1:
                        won = idx_outcome == actual_idx
                        profit = kelly * (decimal_odds - 1.0) if won else -kelly
                        bankroll += profit
                        bets += 1
                        if won:
                            wins += 1
        
        if bets > 0:
            print(f"  {name:28s}: {bets:3d} bets, {wins:2d}W/{bets-wins:2d}L, "
                  f"bankroll=GBP {bankroll:.0f}, WR={wins/bets*100:.0f}%")
        else:
            print(f"  {name:28s}: No value bets found")

print(f"\n{SUB}")
print("  VERDICT")
print(SUB)
print(f"  Best method for last 100 matches: {best_brier[0]}")
print()
print(f"  Calibrators saved to: {MODEL_DIR}/blend_calibrator_*.joblib")
print(SEP)
