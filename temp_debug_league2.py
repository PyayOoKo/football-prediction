"""Deep dive into the league column issue."""
import pandas as pd
from pathlib import Path

# 1. Check results_clean.csv more carefully
clean = Path("data/processed/results_clean.csv")
df = pd.read_csv(clean, low_memory=False)

print("=== results_clean.csv ===")
print(f"Rows: {len(df)}")

# Check league column in first 20 rows
if "league" in df.columns:
    print(f"League dtype: {df['league'].dtype}")
    print(f"First 20 league values:")
    for i, val in enumerate(df["league"].head(20)):
        print(f"  row {i}: repr={repr(val)}, type={type(val).__name__}")
    print(f"Null count: {df['league'].isna().sum()}")
    print(f"Empty string count: {(df['league'] == '').sum() if df['league'].dtype == object else 'N/A'}")
    # Unique non-null values
    non_null = df["league"].dropna()
    print(f"Non-null unique: {non_null.unique()}")
    print(f"Non-null count: {len(non_null)}")
    print(f"Sample non-null values (first 10): {non_null.head(10).tolist()}")
else:
    print("NO 'league' column found!")
    print(f"Columns: {list(df.columns)[:30]}")

# 2. Check raw data
raw = Path("data/raw/results.csv")
if raw.exists():
    df_raw = pd.read_csv(raw, low_memory=False)
    print(f"\n=== results.csv ===")
    print(f"Rows: {len(df_raw)}")
    if "league" in df_raw.columns:
        print(f"League dtype: {df_raw['league'].dtype}")
        non_null = df_raw["league"].dropna()
        print(f"Non-null values: {len(non_null)}/{len(df_raw)}")
        print(f"Unique values: {non_null.unique()[:15]}")
        print(f"Sample (first 10): {df_raw['league'].head(10).tolist()}")
    if "Div" in df_raw.columns:
        print(f"Div values (first 10): {df_raw['Div'].head(10).tolist()}")
