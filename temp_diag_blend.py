"""Diagnostic: reproduce what fit_per_league_dc_models does during pipeline."""
import pandas as pd
from pathlib import Path

# Load data the same way step_retrain_blend does
data_path = Path("data/processed/results_clean.csv")
df = pd.read_csv(data_path, low_memory=False)

# Apply batch-size 3000 (last 3000 rows)
batch_size = 3000
if batch_size > 0 and len(df) > batch_size:
    df = df.iloc[-batch_size:].copy()

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"Has 'league' column: {'league' in df.columns}")
print(f"'league' dtype: {df['league'].dtype if 'league' in df.columns else 'N/A'}")
print(f"'league' unique values: {df['league'].dropna().unique() if 'league' in df.columns else 'N/A'}")
print(f"'league' null count: {df['league'].isna().sum() if 'league' in df.columns else 'N/A'}")

if "league" in df.columns:
    counts = df["league"].value_counts()
    print("\nLeague distribution in batch:")
    for league, cnt in counts.items():
        print(f"  {league} ({type(league).__name__}): {cnt}")
    
    print("\nLeagues meeting min_matches=100:")
    for league, cnt in counts.items():
        if cnt >= 100:
            print(f"  {league}: {cnt}")
        else:
            print(f"  {league}: {cnt} — BELOW 100")
    
    # Now simulate what fit_per_league_dc_models does
    print("\n--- Simulating fit_per_league_dc_models ---")
    for league_code in sorted(df["league"].dropna().unique()):
        print(f"\nProcessing league: {league_code} (type={type(league_code).__name__})")
        league_code_str = str(league_code)
        print(f"  str version: {league_code_str}")
        
        # Try filtering various ways
        way1 = len(df[df["league"] == league_code])
        way2 = len(df[df["league"] == league_code_str])
        way3 = len(df[df["league"].astype(str) == league_code_str])
        
        print(f"  df[league] == original:     {way1} matches")
        print(f"  df[league] == str:          {way2} matches")
        print(f"  df[league].astype(str) == str: {way3} matches")
