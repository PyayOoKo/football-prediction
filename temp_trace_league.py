"""Trace the 'league' column through the data pipeline."""
import pandas as pd
from pathlib import Path

# 1. Raw data (original source)
raw_paths = list(Path("data/raw").glob("*.csv"))
print("=== RAW DATA FILES ===")
for p in raw_paths:
    try:
        df = pd.read_csv(p, low_memory=False, nrows=5)
        has_league = "league" in df.columns or "Div" in df.columns or "div" in df.columns
        print(f"  {p.name}: {len(pd.read_csv(p, low_memory=False))} rows, cols={list(df.columns)[:15]}")
        print(f"    Has league/Div: {has_league}")
    except Exception as e:
        print(f"  {p.name}: ERROR - {e}")

# 2. Check the input to the preprocessing pipeline
print("\n=== DATA COLLECTION OUTPUT ===")
raw_file = Path("data/raw/results.csv")
if raw_file.exists():
    df = pd.read_csv(raw_file, low_memory=False, nrows=5)
    print(f"  results.csv columns: {list(df.columns)[:20]}")
    print(f"  Has 'Div': {'Div' in df.columns}")
    print(f"  Has 'league': {'league' in df.columns}")
    print(f"  Div values: {df['Div'].unique() if 'Div' in df.columns else 'N/A'}")

# 3. Check the processed data
print("\n=== PROCESSED DATA ===")
clean = Path("data/processed/results_clean.csv")
if clean.exists():
    df = pd.read_csv(clean, low_memory=False, nrows=10)
    print(f"  results_clean.csv columns: {list(df.columns)[:25]}")
    print(f"  Has 'league': {'league' in df.columns}")

# 4. Check what the preprocessing.py does with the league column
print("\n=== PREPROCESSING SOURCE ===")
import inspect
try:
    from src import preprocessing
    # Find the run_preprocessing function
    src_path = Path(inspect.getfile(preprocessing))
    print(f"  Source: {src_path}")
except Exception as e:
    print(f"  Cannot inspect: {e}")
