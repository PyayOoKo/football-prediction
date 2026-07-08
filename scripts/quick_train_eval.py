"""
Quick training + evaluation using basic features.
Skips slow feature modules (Dixon-Coles, Elo, Poisson, H2H, league pos).
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
import xgboost as xgb

df = pd.read_csv("data/processed/results_clean.csv", low_memory=False)
print(f"Data: {len(df)} rows x {len(df.columns)} cols")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Teams: {df['home_team'].nunique()}")

# Encode target: H=2, D=1, A=0
df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})
df = df.dropna(subset=["target"])

# Basic feature columns to use
basic_cols = [
    "home_goals_ht", "away_goals_ht", "home_shots", "away_shots",
    "home_shots_target", "away_shots_target", "home_fouls", "away_fouls",
    "home_corners", "away_corners", "home_yellow", "away_yellow",
    "home_red", "away_red",
]

# Add any existing odds columns
odds_prefixes = ["B365", "BW", "IW", "PS", "WH", "VC", "Max", "Avg"]
for c in df.columns:
    for p in odds_prefixes:
        if c.startswith(p) and c[len(p):] in ("H", "D", "A"):
            if c not in basic_cols:
                basic_cols.append(c)

# Only keep existing columns
feature_cols = [c for c in basic_cols if c in df.columns]
print(f"Using {len(feature_cols)} feature columns")

X = df[feature_cols].fillna(0)
y = df["target"]

# Chronological split
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Train XGBoost
print("\nTraining XGBoost...")
t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    objective="multi:softprob", num_class=3,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.1,
    random_state=42, eval_metric="mlogloss", n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time()-t0:.1f}s")

# Evaluate
for name, X_set, y_set in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    preds = model.predict(X_set)
    probs = model.predict_proba(X_set)
    acc = accuracy_score(y_set, preds)
    ll = log_loss(y_set, probs)
    print(f"\n{name}: Accuracy={acc:.2%}, LogLoss={ll:.4f}")

print(f"\nTest Classification Report:")
print(classification_report(y_test, model.predict(X_test), target_names=["Away Win", "Draw", "Home Win"]))

# Recent predictions
print(f"\n{'='*60}")
print("  RECENT PREDICTIONS (Last 15 Test Matches)")
print(f"{'='*60}")
label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
probs = model.predict_proba(X_test)
preds = model.predict(X_test)
recent = df.iloc[val_end:].tail(15)
idx_map = list(range(len(X_test)))
print(f"  {'Date':<12} {'Home':<20} {'Away':<20} {'Score':<6} {'Prediction':<10} {'H%':<5} {'D%':<5} {'A%':<5}")
for i in range(len(recent) - 1, -1, -1):
    r = recent.iloc[i]
    pos = idx_map.index(i)
    hp, dp, ap = probs[pos][2], probs[pos][1], probs[pos][0]
    score = f"{int(r['home_goals'])}-{int(r['away_goals'])}"
    print(f"  {pd.to_datetime(r['date']).strftime('%Y-%m-%d'):<12} "
          f"{str(r['home_team']):<20} {str(r['away_team']):<20} {score:<6} "
          f"{label_map[int(preds[pos])]:<10} {hp:.0%} {dp:.0%} {ap:.0%}")

# Feature importance
imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
imp = imp.sort_values("importance", ascending=False)
print(f"\nTop 10 Features:")
for _, r in imp.head(10).iterrows():
    print(f"  {r['feature']:30s}: {r['importance']:.4f}")

# Save model
from src.train import save_model
save_model(model, "xgboost_model.joblib")
print("\nModel saved to models/xgboost_model.joblib")
