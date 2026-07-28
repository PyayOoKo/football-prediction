import pandas as pd
df = pd.read_csv("data/processed/results_clean.csv", low_memory=False)
# Last 500 rows (what batch-size uses)
tail = df.iloc[-500:]
if "league" in tail.columns:
    counts = tail["league"].value_counts()
    print("League distribution in last 500 rows:")
    for league, cnt in counts.items():
        print(f"  {league}: {cnt}")
    print()
    # Check if any league meets min_matches=100
    meets = counts[counts >= 100]
    print(f"Leagues with >= 100 matches: {len(meets)}")
    for league, cnt in meets.items():
        print(f"  {league}: {cnt}")
