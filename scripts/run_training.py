"""
One-shot training + evaluation script.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import config
from src.feature_engineering import build_features, train_val_test_split
from src.train import train_model, save_model, load_model
from src.evaluate import evaluate_model

df = pd.read_csv("data/processed/results_clean.csv", low_memory=False)
print(f"Data loaded: {df.shape}")

X, y = build_features(df, is_training=True)
print(f"Features built: {X.shape}")

splits = train_val_test_split(X, y)
print(f"Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | Test: {len(splits['X_test'])}")

model, history = train_model(
    splits["X_train"], splits["y_train"],
    splits["X_val"], splits["y_val"],
)
save_model(model, "xgboost_model.joblib")

if history:
    print("\nTraining History:")
    for k, v in history.items():
        vals = [x for x in (v if isinstance(v, list) else [v]) if x is not None]
        if vals:
            print(f"  {k}: {vals[-1]:.4f}")

metrics = evaluate_model(model, splits["X_test"], splits["y_test"])
print(f"\n{'='*50}")
print("  EVALUATION METRICS (Test Set)")
print(f"{'='*50}")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k:25s}: {v:.4f}")
    else:
        print(f"  {k:25s}: {v}")

print(f"\n{'='*50}")
print("  RECENT PREDICTIONS (Last 15 Test Matches)")
print(f"{'='*50}")

probs = model.predict_proba(splits["X_test"])
preds = model.predict(splits["X_test"])
label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
idx_map = list(splits["X_test"].index)

recent_df = df.loc[splits["X_test"].index].tail(15)
print(f"  {'Date':<12} {'Home':<18} {'Away':<18} {'Score':<7} {'Prediction':<10} {'H%':<6} {'D%':<6} {'A%':<6}")
print(f"  {'-'*77}")
for i in range(len(recent_df)):
    r = recent_df.iloc[i]
    pos = idx_map.index(r.name)
    hp, dp, ap = probs[pos][2], probs[pos][1], probs[pos][0]
    score = f"{int(r['home_goals'])}-{int(r['away_goals'])}"
    print(f"  {pd.to_datetime(r['date']).strftime('%Y-%m-%d'):<12} "
          f"{str(r['home_team']):<18} {str(r['away_team']):<18} {score:<7} "
          f"{label_map[int(preds[pos])]:<10} {hp:.0%}  {dp:.0%}  {ap:.0%}")

print(f"\n{'='*50}")
print("  MODEL INFO")
print(f"{'='*50}")
print(f"  Type: {config.train.model_type}")
print(f"  Estimators: {config.train.n_estimators}")
print(f"  Max Depth: {config.train.max_depth}")
print(f"  Learning Rate: {config.train.learning_rate}")
print(f"  Features in model: {X.shape[1]}")
print(f"  Total matches: {len(df)}")
print(f"  Teams: {df['home_team'].nunique()}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
