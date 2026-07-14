import sys, warnings, os
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '42'
import numpy as np
np.random.seed(42)

from config import config
config.features.include_h2h = False
config.features.include_league_position = False
config.odds.compute_consensus = False
config.odds.warn_missing = False
config.player_info.enabled = False
config.xg.warn_missing = False
config.xg.compute_xpts = True
config.elo.regress_to_mean = True
config.elo.home_advantage = 50
config.train.model_type = "xgboost"
config.train.n_estimators = 300
config.train.max_depth = 5
config.train.learning_rate = 0.05

import pandas as pd
import numpy as np
from pathlib import Path

print("Loading data...")
df = pd.read_csv('data/raw/worldcup_all.csv', low_memory=False, parse_dates=['date'])
RESULT_TO_TARGET = {'H': 2, 'D': 1, 'A': 0}
df['target'] = df['result'].map(RESULT_TO_TARGET).fillna(-1).astype('int8')
completed_mask = df['result'].notna()
df_completed = df[completed_mask].copy()
df_upcoming = df[~completed_mask].copy()

# Predictable: non-placeholder upcoming matches
from src.data_collection.sources.worldcup import is_placeholder_team
predict_mask = df_upcoming.apply(
    lambda r: not is_placeholder_team(r['home_team']) and not is_placeholder_team(r['away_team']),
    axis=1
)
df_predictable = df_upcoming[predict_mask].copy()
print(f"Predictable matches: {len(df_predictable)}")

from src.feature_engineering import build_features, train_val_test_split
from src.train import train_model, save_model
from sklearn.metrics import accuracy_score

print("Building features on completed data...")
X, y = build_features(df_completed, is_training=True)
print(f"Feature matrix: {X.shape}")
splits = train_val_test_split(X, y)

print("Training XGBoost...")
model, history = train_model(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'])
print(f"Train loss: {history.get('train_loss',[None])[0]:.4f} | Val loss: {history.get('val_loss',[None])[0]:.4f} | Val acc: {history.get('val_accuracy',[None])[0]:.4f}")
y_pred = model.predict(splits['X_test'])
acc = accuracy_score(splits['y_test'], y_pred)
print(f"Test accuracy: {acc:.4f}")
save_model(model, 'worldcup_xgboost.joblib')

# Predict original order
df_predictable['_pred_id'] = np.arange(len(df_predictable))
X_orig, _ = build_features(pd.concat([df_completed, df_predictable], ignore_index=True), is_training=True)
X_orig_pred = X_orig.iloc[len(df_completed):].copy()
pred_order_orig = X_orig_pred.pop('_pred_id').values
probs_orig = model.predict_proba(X_orig_pred)

# Predict swapped order
df_swapped = df_predictable.copy()
df_swapped['home_team'] = df_predictable['away_team'].values
df_swapped['away_team'] = df_predictable['home_team'].values
df_swapped['home_goals'] = np.nan
df_swapped['away_goals'] = np.nan
df_swapped['result'] = np.nan
df_swapped['target'] = -1

X_swp, _ = build_features(pd.concat([df_completed, df_swapped], ignore_index=True), is_training=True)
X_swp_pred = X_swp.iloc[len(df_completed):].copy()
pred_order_swp = X_swp_pred.pop('_pred_id').values
probs_swp = model.predict_proba(X_swp_pred)

# Swap-average
orig_sort = np.argsort(pred_order_orig)
swp_sort = np.argsort(pred_order_swp)
probs_orig = probs_orig[orig_sort]
probs_swp = probs_swp[swp_sort]

avg_home = (probs_orig[:, 2] + probs_swp[:, 0]) / 2
avg_draw = (probs_orig[:, 1] + probs_swp[:, 1]) / 2
avg_away = (probs_orig[:, 0] + probs_swp[:, 2]) / 2
probs = np.column_stack([avg_away, avg_draw, avg_home])
preds = np.argmax(probs, axis=1)
confidences = probs.max(axis=1)
LABEL_MAP = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

cols = ['date', 'home_team', 'away_team', 'round', 'ground']
output = df_predictable[[c for c in cols if c in df_predictable.columns]].copy()
output['prediction'] = [LABEL_MAP[p] for p in preds]
output['home_win_prob'] = avg_home
output['draw_prob'] = avg_draw
output['away_win_prob'] = avg_away
output['confidence'] = confidences

print()
print('=' * 105)
print('  KNOCKOUT ROUND PREDICTIONS (swap-averaged, neutral venue)')
print('=' * 105)
print(f"  {'Date':<14} {'Match':<48} {'Prediction':<18} {'Home%':<8} {'Draw%':<8} {'Away%':<8}")
print(f"  {'-' * 105}")
for _, r in output.iterrows():
    match = f"{r['home_team']} vs {r['away_team']}"
    print(f"  {str(r['date'])[:10]:<14} {match:<48} {r['prediction']:<18} {r['home_win_prob']*100:>5.1f}%   {r['draw_prob']*100:>5.1f}%   {r['away_win_prob']*100:>5.1f}%")

out_path = Path('reports/predictions_worldcup/worldcup_predictions.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(out_path, index=False)
print(f'\nSaved predictions to {out_path}')
