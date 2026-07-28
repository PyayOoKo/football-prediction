"""Validate ThreeModelBlend with market-specific models."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from src.models.three_model_blend import ThreeModelBlend
from src.dixon_coles import DixonColesModel
from src.elo import EloSystem

print("=" * 60)
print("  VALIDATION: ThreeModelBlend + Market-Specific Models")
print("=" * 60)

print("\n1. Loading F1 data...")
conn = sqlite3.connect("data/football_data.db")
df = pd.read_sql_query(
    """SELECT date, home_team, away_team, home_goals, away_goals, result, season,
              home_xg, away_xg, league
       FROM matches
       WHERE league = 'F1' AND date >= '2023-01-01'
       ORDER BY date ASC""",
    conn,
)
conn.close()

split_idx = int(len(df) * 0.85)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]
print(f"   Train: {len(df_train)}, Test: {len(df_test)}")

print("\n2. Fitting DC and Elo...")
dc = DixonColesModel(decay_halflife_days=1460).fit(df_train, verbose=False)
elo = EloSystem()
elo.process_matches(df_train)
print(f"   DC fitted: {dc.fitted}, Elo teams: {len(elo._ratings)}")

print("\n3. Creating blend with market models...")
blend = ThreeModelBlend(dc_model=dc, elo_model=elo, historical_df=df_train)
loaded = blend.load_market_models(league="F1")
print(f"   Loaded: {loaded}")
print(f"   ou_models: {list(blend.ou_models.keys())}")
print(f"   btts_models: {list(blend.btts_models.keys())}")
assert loaded["ou"] > 0, "O/U models not loaded!"
assert loaded["btts"] > 0, "BTTS models not loaded!"

print("\n4. Testing single predict()...")
row = df_test.iloc[0]
pred = blend.predict(row["home_team"], row["away_team"])
print(f"   {row['home_team']} vs {row['away_team']}")
print(f"   1X2: {pred['1x2']}")
print(f"   O/U: {pred['over_under']}")
print(f"   BTTS: {pred['btts']}")

print("\n5. Batch predict vs no-market-models version...")
df_fixtures = df_test.head(20)[["home_team", "away_team"]].copy()
df_preds = blend.predict_matches(df_fixtures)

blend_no_mm = ThreeModelBlend(dc_model=dc, elo_model=elo, historical_df=df_train)
df_preds_no_mm = blend_no_mm.predict_matches(df_fixtures)

print(f"   With MM - over_2_5_probs: {[round(p, 4) for p in df_preds['over_2_5_prob'].head(5).tolist()]}")
print(f"   No  MM - over_2_5_probs: {[round(p, 4) for p in df_preds_no_mm['over_2_5_prob'].head(5).tolist()]}")
print(f"   With MM - btts_probs:     {[round(p, 4) for p in df_preds['btts_prob'].head(5).tolist()]}")
print(f"   No  MM - btts_probs:     {[round(p, 4) for p in df_preds_no_mm['btts_prob'].head(5).tolist()]}")

diff_ou = (df_preds["over_2_5_prob"] != df_preds_no_mm["over_2_5_prob"]).sum()
diff_btts = (df_preds["btts_prob"] != df_preds_no_mm["btts_prob"]).sum()
print(f"\n   Differences: O/U={diff_ou}/{len(df_preds)}, BTTS={diff_btts}/{len(df_preds)}")

assert diff_ou > 0, "Market models NOT affecting O/U predictions!"
assert diff_btts > 0, "Market models NOT affecting BTTS predictions!"
print("   ✅ Market models ARE actively influencing predictions!")

print("\n6. Over 3.5 still uses derived approach...")
diff_ou35 = (df_preds["over_3_5_prob"] != df_preds_no_mm["over_3_5_prob"]).sum()
print(f"   Over 3.5 differences: {diff_ou35}/{len(df_preds)} (should be 0)")
assert diff_ou35 == 0, "Over 3.5 should NOT use market models!"

print("\n" + "=" * 60)
print("  ✅ ALL VALIDATIONS PASSED")
print("=" * 60)
