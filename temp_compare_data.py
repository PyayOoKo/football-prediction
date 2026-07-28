import pandas as pd
import numpy as np
import os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

m = pd.read_csv("data/matches.csv", low_memory=False)
l = pd.read_csv("data/raw/league_all.csv", low_memory=False)

m_size = os.path.getsize("data/matches.csv") / 1024 / 1024
l_size = os.path.getsize("data/raw/league_all.csv") / 1024 / 1024

print("=" * 72)
print("  DATA VOLUME COMPARISON: matches.csv vs league_all.csv")
print("=" * 72)

print("\n--- BASICS ---")
print(f"  File size:        {m_size:>8.1f} MB  vs  {l_size:>8.1f} MB  ({l_size/m_size:.1f}x larger)")
print(f"  Rows:             {len(m):>8,}  vs  {len(l):>8,}  ({len(l)-len(m):+>8,} diff)")
print(f"  Columns:          {len(m.columns):>8}  vs  {len(l.columns):>8}  ({len(l.columns)-len(m.columns):+>8} diff)")

m["date"] = pd.to_datetime(m["date"])
l["date"] = pd.to_datetime(l["date"])

m_min = m["date"].min().strftime("%Y-%m-%d")
m_max = m["date"].max().strftime("%Y-%m-%d")
l_min = l["date"].min().strftime("%Y-%m-%d")
l_max = l["date"].max().strftime("%Y-%m-%d")
m_years = m["date"].dt.year.max() - m["date"].dt.year.min()
l_years = l["date"].dt.year.max() - l["date"].dt.year.min()

print(f"  Date range start: {m_min:>15}  vs  {l_min:>15}")
print(f"  Date range end:   {m_max:>15}  vs  {l_max:>15}")
print(f"  Years covered:    {m_years:>8}  vs  {l_years:>8}")

# Leagues
m_leagues = sorted(set(str(x) for x in m["league"].dropna().unique()))
l_leagues = sorted(set(str(x) for x in l["league"].dropna().unique()))
print(f"  Unique leagues:   {len(m_leagues):>8}  vs  {len(l_leagues):>8}")
only_m = set(m_leagues) - set(l_leagues)
only_l = set(l_leagues) - set(m_leagues)
if only_m: print(f"  Leagues only in matches.csv:     {sorted(only_m)}")
if only_l: print(f"  Leagues only in league_all.csv:  {sorted(only_l)}")

print("\n--- COLUMN COVERAGE ---")
key_cols = [
    ("home_goals", "Goals"), ("away_goals", "Goals"), ("total_goals", "Goals"),
    ("result", "Result"), ("btts", "BTTS"), ("over_2_5", "O/U 2.5"),
    ("home_xg", "xG"), ("away_xg", "xG"),
    ("home_shots", "Shots"), ("away_shots", "Shots"),
    ("home_shots_target", "Shots on target"), ("away_shots_target", "Shots on target"),
    ("home_corners", "Corners"), ("away_corners", "Corners"),
    ("home_fouls", "Fouls"), ("away_fouls", "Fouls"),
    ("home_odds", "1X2 odds"), ("draw_odds", "1X2 odds"), ("away_odds", "1X2 odds"),
    ("over25_odds", "O/U odds"), ("under25_odds", "O/U odds"),
    ("b365h", "Bet365 odds"), ("bbavh", "Avg odds"), ("bbav>2.5", "Avg O/U odds"),
]
print(f"  {'Column':<20} {'Type':<20} {'matches.csv':>12} {'league_all.csv':>14}")
print(f"  {'-'*20} {'-'*20} {'-'*12} {'-'*14}")
for col, ctype in key_cols:
    in_m = col in m.columns
    in_l = col in l.columns
    m_cov = f"{m[col].notna().mean()*100:.0f}%" if in_m else "MISSING"
    l_cov = f"{l[col].notna().mean()*100:.0f}%" if in_l else "MISSING"
    print(f"  {col:<20} {ctype:<20} {m_cov:>12} {l_cov:>14}")

print("\n--- ROWS PER LEAGUE ---")
m["league_str"] = m["league"].astype(str)
l["league_str"] = l["league"].astype(str)
m_lc = m["league_str"].value_counts()
l_lc = l["league_str"].value_counts()
all_ll = sorted(set(m["league_str"].unique()) | set(l["league_str"].unique()))
print(f"  {'League':<8} {'matches.csv':>10} {'league_all.csv':>14} {'Diff':>10} {'Gain':>8}")
print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*10} {'-'*8}")
for league in all_ll:
    if league in ("nan", ""): continue
    mc = m_lc.get(league, 0)
    lc = l_lc.get(league, 0)
    diff = lc - mc
    gp = f"{diff/mc*100:+.0f}%" if mc > 0 else "NEW"
    print(f"  {league:<8} {mc:>10,} {lc:>14,} {diff:>+10,} {gp:>8}")

print("\n--- TARGET DISTRIBUTIONS ---")
for nm, df in [("matches.csv", m), ("league_all.csv", l)]:
    if "over_2_5" in df.columns:
        print(f"  {nm}: Over 2.5={df['over_2_5'].mean()*100:.1f}%  BTTS Yes={df['btts'].mean()*100:.1f}%")
    avg_h = df["home_goals"].mean()
    avg_a = df["away_goals"].mean()
    print(f"    Avg goals: Home={avg_h:.2f} Away={avg_a:.2f} Total={avg_h+avg_a:.2f}")

print("\n--- DATA BONUS WITH league_all.csv ---")
extra = len(l) - len(m)
print(f"  Extra rows in league_all.csv: {extra:,}")
print(f"  BUT matches.csv has {len(m):,} rows vs {len(l):,} rows")
print(f"  So league_all.csv has FEWER rows than matches.csv!")
print(f"  However league_all.csv has 184 columns vs 25 = 159 extra columns")
print(f"  The VALUE is in the richer betting odds data (full market coverage)")
print()

# Check what odds columns league_all has that matches.csv doesn't
odds_only_in_l = [c for c in l.columns if any(x in c.lower() for x in ["b365","bbav","ps","wh","iw","lb","bw","vc","max","av"])]
odds_in_m = [c for c in m.columns if any(x in c.lower() for x in ["odds"])]
print(f"  Betting odds columns in league_all.csv: {len(odds_only_in_l)}")
print(f"  Betting odds columns in matches.csv:    {len(odds_in_m)}")
print(f"  Extra odds columns: {len(odds_only_in_l) - len(odds_in_m)}")

print("\nDone!")
