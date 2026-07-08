"""
Evaluate the existing trained model and generate report.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from config import config
from src.feature_engineering import build_features, train_val_test_split
from src.train import load_model
from src.evaluate import evaluate_model

print("Loading data...")
df = pd.read_csv("data/processed/results_clean.csv", low_memory=False)
print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

print("Building features...")
t0 = time.time()
X, y = build_features(df, is_training=True)
print(f"  Shape: {X.shape}, Time: {time.time()-t0:.1f}s")

splits = train_val_test_split(X, y)
print(f"  Train: {len(splits['X_train'])} | Val: {len(splits['X_val'])} | Test: {len(splits['X_test'])}")

print("Loading model...")
model = load_model("xgboost_model.joblib")

print("\nEvaluating on test set...")
metrics = evaluate_model(model, splits["X_test"], splits["y_test"])
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k:30s}: {v:.4f}")
    else:
        print(f"  {k:30s}: {v}")

probs = model.predict_proba(splits["X_test"])
preds = model.predict(splits["X_test"])
label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
idx_map = list(splits["X_test"].index)

print("\nRecent 10 test predictions:")
print(f"{'Date':<12} {'Home':<18} {'Away':<18} {'Score':<7} {'Prediction':<10} {'H%':<6} {'D%':<6} {'A%':<6}")
print("-" * 77)
for i, idx in enumerate(splits["X_test"].index[-10:]):
    r = df.loc[idx]
    pos = idx_map.index(idx)
    hp, dp, ap = probs[pos][2], probs[pos][1], probs[pos][0]
    score = f"{int(r['home_goals'])}-{int(r['away_goals'])}"
    print(f"{pd.to_datetime(r['date']).strftime('%Y-%m-%d'):<12} "
          f"{str(r['home_team']):<18} {str(r['away_team']):<18} {score:<7} "
          f"{label_map[int(preds[pos])]:<10} {hp:.0%}  {dp:.0%}  {ap:.0%}")

print("\n--- MODEL REPORT ---")
print(f"Model Type: {config.train.model_type}")
print(f"Estimators: {config.train.n_estimators}")
print(f"Max Depth: {config.train.max_depth}")
print(f"Learning Rate: {config.train.learning_rate}")
print(f"Total Matches: {len(df):,}")
print(f"Teams: {df['home_team'].nunique()}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"Leagues: {df['league'].nunique() if 'league' in df.columns else 'N/A'}")
print(f"Features Used: {X.shape[1]}")
print(f"Training Samples: {len(splits['X_train'])}")
print(f"Test Samples: {len(splits['X_test'])}")

# Model accuracy
from sklearn.metrics import accuracy_score, classification_report
test_acc = accuracy_score(splits['y_test'], preds)
print(f"\nTest Accuracy: {test_acc:.2%}")
print("\nClassification Report:")
print(classification_report(splits['y_test'], preds, target_names=['Away Win', 'Draw', 'Home Win']))

# Feature importance
if hasattr(model, 'feature_importances_'):
    imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print("\nTop 15 Features:")
    for _, row in imp.head(15).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
