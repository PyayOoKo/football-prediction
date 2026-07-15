"""Explore available data for backtesting all models."""
import sys
sys.path.insert(0, '.')
import json
import pandas as pd
import numpy as np

print('=== MATCH DATA ===')
df = pd.read_csv('data/raw/worldcup_all.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'Seasons: {sorted(df["season"].unique())}')
print(f'Unique teams: home={df["home_team"].nunique()}, away={df["away_team"].nunique()}')

# Show result distribution
result_dist = df['result'].value_counts()
print(f'Result distribution: {result_dist.to_dict()}')

# Chronological split (same as calibration)
df_sorted = df.copy()
df_sorted['date'] = pd.to_datetime(df_sorted['date'])
df_sorted.sort_values(['date', 'home_team'], inplace=True)
df_sorted.reset_index(drop=True, inplace=True)

n = len(df_sorted)
n_train = int(n * 0.6)
n_val = int(n * 0.2)
train = df_sorted.iloc[:n_train]
val = df_sorted.iloc[n_train:n_train+n_val]
test = df_sorted.iloc[n_train+n_val:]
print(f'Split: train={len(train)}, val={len(val)}, test={len(test)}')
print(f'Test date range: {test["date"].min()} to {test["date"].max()}')
print(f'Test seasons: {sorted(test["season"].unique())}')

print()
print('=== CALIBRATION DATA ===')
cal = json.load(open('reports/calibration_results_20260714_233850.json'))
print(f'Split sizes: {cal.get("split_sizes", {})}')
print(f'Best calibration per model:')
for model_name, info in cal.get('best_calibration_per_model', {}).items():
    print(f'  {model_name}: method={info.get("best_method")}, brier_test={info.get("calibrated_brier", info.get("brier_test", "?"))}')

print()
print('=== PHASE COMPARISON ===')
comp = json.load(open('reports/phase3_vs_phase4_20260714_195024.json'))
phase4 = comp.get('phase4', [])
phase3 = comp.get('phase3', [])
print(f'Phase4: {len(phase4)} entries')
for m in phase4:
    if isinstance(m, dict):
        print(f'  {m.get("model")}: brier={m.get("brier", m.get("brier_test", "?"))}, accuracy={m.get("accuracy", "?")}')
print(f'Phase3: {len(phase3)} entries')
for m in phase3:
    if isinstance(m, dict):
        print(f'  {m.get("model")}: brier={m.get("brier", m.get("brier_test", "?"))}, accuracy={m.get("accuracy", "?")}')

print()
print('=== ODDS CHECK ===')
odds_cols = [c for c in df.columns if 'odds' in c.lower() or 'BbAv' in c or 'PSH' in c or 'PSD' in c or 'PSA' in c]
print(f'Odds columns found: {odds_cols}')
if not odds_cols:
    # Check results.csv too
    df2 = pd.read_csv('data/raw/results.csv', nrows=5)
    odds2 = [c for c in df2.columns if 'odds' in c.lower() or 'BbAv' in c]
    print(f'Odds columns in results.csv: {odds2}')
    print(f'results.csv columns: {list(df2.columns)}')

print()
print('=== FOOTBALL-DATA DIR ===')
import os
if os.path.isdir('data/raw/football-data'):
    files = os.listdir('data/raw/football-data')
    print(f'Files: {files[:10]}')
