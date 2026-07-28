import pandas as pd
df = pd.read_csv("data/processed/results_clean.csv", low_memory=False)
print("Rows:", len(df))
print("Columns:", list(df.columns)[:20], "...")
print("Has 'league' column:", "league" in df.columns)
if "league" in df.columns:
    print("League codes:", sorted(df["league"].dropna().unique()))
    counts = df["league"].value_counts()
    print("Per league (top 20):")
    for league, cnt in counts.head(20).items():
        print(f"  {league}: {cnt}")
else:
    # Check for similar columns
    for c in df.columns:
        if "league" in c.lower() or "div" in c.lower():
            print(f"Similar column found: {c}")
