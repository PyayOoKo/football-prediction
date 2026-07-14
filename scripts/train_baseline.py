"""
Train a logistic regression baseline model and establish reference performance metrics.

Usage::

    python scripts/train_baseline.py

Outputs:
    - reports/baseline_performance_{timestamp}.json   — full metrics report
    - reports/baseline_feature_importance.png         — top-20 coefficient plot
    - reports/baseline_reliability_diagram.png        — calibration curve
    - models/baseline_logistic_regression             — serialised model
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ── Project setup ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import config
from src.time_series_cv import time_series_train_val_test_split
from src.feature_engineering import build_features
from src.train import save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("baseline")

for name in ("matplotlib", "PIL", "urllib3", "sklearn", "xgboost"):
    logging.getLogger(name).setLevel(logging.WARNING)

# ── Paths ─────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "processed" / "results_clean.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
REPORT_PATH = REPORTS_DIR / f"baseline_performance_{TIMESTAMP}.json"

# ═══════════════════════════════════════════════════════════
#  Step 1 — Load & build features
# ═══════════════════════════════════════════════════════════

print("=" * 72)
print("  BASELINE — Logistic Regression Performance Benchmark")
print("=" * 72)

print("\n[1/6] Loading data and building features ...")
t0 = time.time()

df = pd.read_csv(DATA_PATH, low_memory=False)
log.info("Loaded %d rows x %d cols", len(df), len(df.columns))

X, y = build_features(df, is_training=True)
log.info("Feature matrix: %s, target: %s", X.shape, y.shape)
t_build = time.time() - t0
print(f"      {X.shape[0]} samples, {X.shape[1]} features  ({t_build:.1f}s)")

# ═══════════════════════════════════════════════════════════
#  Step 2 — Chronological split (60 / 20 / 20)
# ═══════════════════════════════════════════════════════════

print("\n[2/6] Chronological split (60/20/20) ...")
t0 = time.time()

splits = time_series_train_val_test_split(X, y, ratios=(0.6, 0.2, 0.2))

train_start = df["date"].iloc[0] if "date" in df.columns else "?"
train_end = df["date"].iloc[splits["X_train"].shape[0] - 1] if "date" in df.columns else "?"
val_start = df["date"].iloc[splits["X_train"].shape[0]] if "date" in df.columns else "?"
val_end = df["date"].iloc[splits["X_train"].shape[0] + splits["X_val"].shape[0] - 1] if "date" in df.columns else "?"
test_start = df["date"].iloc[splits["X_train"].shape[0] + splits["X_val"].shape[0]] if "date" in df.columns else "?"
test_end = df["date"].iloc[-1] if "date" in df.columns else "?"

print(f"      Train: {splits['X_train'].shape[0]} samples  ({train_start} → {train_end})")
print(f"      Val:   {splits['X_val'].shape[0]} samples  ({val_start} → {val_end})")
print(f"      Test:  {splits['X_test'].shape[0]} samples  ({test_start} → {test_end})")
t_split = time.time() - t0
print(f"      Done  ({t_split:.2f}s)")

# Impute NaN with training column means
col_means = splits["X_train"].mean().fillna(0)
X_train_c = splits["X_train"].fillna(col_means)
X_val_c = splits["X_val"].fillna(col_means)
X_test_c = splits["X_test"].fillna(col_means)

# ═══════════════════════════════════════════════════════════
#  Step 3 — Train logistic regression
# ═══════════════════════════════════════════════════════════

print("\n[3/6] Training LogisticRegression ...")
t0 = time.time()

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
    class_weight="balanced",
    C=1.0,
    n_jobs=-1,
)
model.fit(X_train_c, splits["y_train"])

train_probs = model.predict_proba(X_train_c)
val_probs = model.predict_proba(X_val_c)
train_loss = log_loss(splits["y_train"], train_probs)
val_loss = log_loss(splits["y_val"], val_probs)
val_acc = accuracy_score(splits["y_val"], model.predict(X_val_c))

t_train = time.time() - t0
print(f"      Train log-loss: {train_loss:.4f}")
print(f"      Val   log-loss: {val_loss:.4f}")
print(f"      Val   accuracy: {val_acc:.2%}")
print(f"      Done  ({t_train:.1f}s)")

# ═══════════════════════════════════════════════════════════
#  Step 4 — Classification & probabilistic metrics
# ═══════════════════════════════════════════════════════════

print("\n[4/6] Computing metrics ...")
t0 = time.time()

y_test_pred = model.predict(X_test_c)
y_test_proba = model.predict_proba(X_test_c)

# ── Classification ────────────────────────────────────────
test_acc = accuracy_score(splits["y_test"], y_test_pred)
cm = confusion_matrix(splits["y_test"], y_test_pred).tolist()

class_labels = {0: "away", 1: "draw", 2: "home"}
per_class = {}
for i, label in class_labels.items():
    per_class[label] = {
        "precision": float(precision_score(splits["y_test"], y_test_pred, labels=[i], average="macro")),
        "recall": float(recall_score(splits["y_test"], y_test_pred, labels=[i], average="macro")),
        "f1": float(f1_score(splits["y_test"], y_test_pred, labels=[i], average="macro")),
    }

# ROC-AUC (OvR macro)
try:
    roc_auc = float(roc_auc_score(splits["y_test"], y_test_proba, multi_class="ovr", average="macro"))
except Exception:
    roc_auc = None

# ── Probabilistic ─────────────────────────────────────────
y_onehot = np.eye(3)[splits["y_test"].values if hasattr(splits["y_test"], "values") else splits["y_test"]]
brier = float(np.mean(np.sum((y_test_proba - y_onehot) ** 2, axis=1)))
test_ll = float(log_loss(splits["y_test"], y_test_proba))

# Brier score per class
brier_per_class = {}
for i, label in class_labels.items():
    brier_per_class[label] = float(brier_score_loss((splits["y_test"] == i).astype(int), y_test_proba[:, i]))

# ── Calibration curve (reliability diagram data) ─────────
n_bins = 10
pred_class = np.argmax(y_test_proba, axis=1)
pred_conf = np.max(y_test_proba, axis=1)
correct = (pred_class == splits["y_test"]).astype(float)

bins = np.linspace(0, 1, n_bins + 1)
reliability = []
for i in range(n_bins):
    in_bin = (pred_conf >= bins[i]) & (pred_conf < bins[i + 1])
    count = int(in_bin.sum())
    if count > 0:
        acc = float(correct[in_bin].mean())
        conf = float(pred_conf[in_bin].mean())
    else:
        acc = conf = 0.0
    reliability.append({
        "bin_lower": float(round(bins[i], 3)),
        "bin_upper": float(round(bins[i + 1], 3)),
        "count": count,
        "accuracy": float(round(acc, 4)),
        "confidence": float(round(conf, 4)),
        "gap": float(round(abs(acc - conf), 4)),
    })

# ECE
total = float(sum(r["count"] for r in reliability))
ece = float(
    sum(r["count"] / total * r["gap"] for r in reliability if total > 0)
) if total > 0 else 0.0

# Brier score decomposition (Murphy, 1973): BS = reliability + resolution - uncertainty
# uncertainty = mean(y_true) * (1 - mean(y_true)) per class → not well-defined for multi-class
# Simplified: report overall Brier and ECE

t_metrics = time.time() - t0
print(f"      Test accuracy:   {test_acc:.2%}")
print(f"      Test log-loss:   {test_ll:.4f}")
print(f"      Test Brier:      {brier:.4f}")
print(f"      ECE:             {ece:.4f}")
print(f"      ROC-AUC:         {roc_auc:.4f}" if roc_auc else "      ROC-AUC:         N/A")
print(f"      Done  ({t_metrics:.1f}s)")

# ═══════════════════════════════════════════════════════════
#  Step 5 — Feature importance
# ═══════════════════════════════════════════════════════════

print("\n[5/6] Extracting feature importance ...")
t0 = time.time()

coef = model.coef_  # shape (3, n_features)
# Average absolute coefficient across classes
avg_abs_coef = np.mean(np.abs(coef), axis=0)
feature_names = X.columns.tolist()

feature_imp = pd.DataFrame({
    "feature": feature_names,
    "coef_home": coef[2],
    "coef_draw": coef[1],
    "coef_away": coef[0],
    "abs_coef": avg_abs_coef,
}).sort_values("abs_coef", ascending=False)

top_20 = feature_imp.head(20)
top_20_records = []
for _, row in top_20.iterrows():
    top_20_records.append({
        "feature": row["feature"],
        "coefficient_home": float(round(row["coef_home"], 4)),
        "coefficient_draw": float(round(row["coef_draw"], 4)),
        "coefficient_away": float(round(row["coef_away"], 4)),
        "abs_coefficient": float(round(row["abs_coef"], 4)),
    })

t_importance = time.time() - t0
print(f"      Top feature: {top_20.iloc[0]['feature']}  (|coef| = {top_20.iloc[0]['abs_coef']:.4f})")
print(f"      Done  ({t_importance:.1f}s)")

# ── Feature importance plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
top20_sorted = top_20.sort_values("abs_coef")
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top20_sorted["coef_home"]]
ax.barh(range(len(top20_sorted)), top20_sorted["abs_coef"], color="#3498db", alpha=0.85)
ax.set_yticks(range(len(top20_sorted)))
ax.set_yticklabels(top20_sorted["feature"], fontsize=9)
ax.set_xlabel("Mean |Coefficient|")
ax.set_title("Top 20 Features — Logistic Regression Baseline", fontweight="bold")
plt.tight_layout()
imp_plot_path = str(REPORTS_DIR / "baseline_feature_importance.png")
fig.savefig(imp_plot_path, dpi=150)
plt.close(fig)

# ── Reliability diagram ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
valid_bins = [r for r in reliability if r["count"] > 0]
if valid_bins:
    ax.plot(
        [r["confidence"] for r in valid_bins],
        [r["accuracy"] for r in valid_bins],
        "o-", color="#3498db", linewidth=2, markersize=8,
    )
    ax.fill_between(
        [r["confidence"] for r in valid_bins],
        [r["accuracy"] for r in valid_bins],
        [r["confidence"] for r in valid_bins],
        alpha=0.15, color="#3498db",
    )
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability (confidence)")
ax.set_ylabel("Observed fraction positive (accuracy)")
ax.set_title(f"Reliability Diagram — ECE = {ece:.4f}", fontweight="bold")
ax.legend(loc="lower right")
ax.set_aspect("equal")
plt.tight_layout()
rel_plot_path = str(REPORTS_DIR / "baseline_reliability_diagram.png")
fig.savefig(rel_plot_path, dpi=150)
plt.close(fig)

# ═══════════════════════════════════════════════════════════
#  Step 5b — Backtest / betting metrics
# ═══════════════════════════════════════════════════════════

print("\n[5b/6] Backtest (value-betting simulation) ...")
t0 = time.time()

betting_metrics: dict = {
    "roi": 0.0,
    "yield": 0.0,
    "clv": None,
    "n_bets": 0,
    "win_rate": 0.0,
    "max_drawdown": 0.0,
    "sharpe_ratio": None,
}

try:
    from src.backtesting import run_backtest

    bt_result = run_backtest(
        model,
        X_test_c,
        splits["y_test"],
        odds_df=None,
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
        min_ev=0.0,
        output_dir=str(REPORTS_DIR / "backtest"),
        print_report=False,
        show_charts=False,
    )
    bt = bt_result["metrics"]
    betting_metrics = {
        "roi": round(bt.roi_pct, 4),
        "yield": round(bt.yield_pct, 4),
        "clv": None,
        "n_bets": bt.total_bets,
        "win_rate": round(bt.win_rate_pct, 2),
        "max_drawdown": round(bt.max_drawdown_pct, 4),
        "sharpe_ratio": None,
    }
    log.info(
        "Backtest: %d bets, ROI=%.2f%%, yield=%.2f%%, maxDD=%.2f%%",
        bt.total_bets, bt.roi_pct, bt.yield_pct, bt.max_drawdown_pct,
    )
except Exception as exc:
    log.warning("Backtest skipped: %s", exc)

t_backtest = time.time() - t0
print(f"      Bets: {betting_metrics['n_bets']}, ROI: {betting_metrics['roi']}%, "
      f"Yield: {betting_metrics['yield']}%")
print(f"      Done  ({t_backtest:.1f}s)")

# ═══════════════════════════════════════════════════════════
#  Step 6 — Save report
# ═══════════════════════════════════════════════════════════

print("\n[6/6] Saving report ...")
t0 = time.time()

# Get git commit
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT), text=True,
    ).strip()
except Exception:
    git_commit = "unknown"

# Build date-range strings
def fmt_date(s):
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return str(s)

report = {
    "model": {
        "type": "LogisticRegression",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
        "C": 1.0,

    },
    "data": {
        "train_size": int(splits["X_train"].shape[0]),
        "val_size": int(splits["X_val"].shape[0]),
        "test_size": int(splits["X_test"].shape[0]),
        "total_samples": int(X.shape[0]),
        "date_range": {
            "train": {
                "start": fmt_date(train_start),
                "end": fmt_date(train_end),
            },
            "val": {
                "start": fmt_date(val_start),
                "end": fmt_date(val_end),
            },
            "test": {
                "start": fmt_date(test_start),
                "end": fmt_date(test_end),
            },
        },
        "features": {
            "total": int(X.shape[1]),
            "numeric": int(X.select_dtypes(include=[np.number]).shape[1]),
            "categorical": int(X.select_dtypes(exclude=[np.number]).shape[1]),
        },
    },
    "metrics": {
        "accuracy": round(test_acc, 4),
        "per_class": per_class,
        "brier_score": round(brier, 4),
        "brier_score_per_class": {k: round(v, 4) for k, v in brier_per_class.items()},
        "log_loss": round(test_ll, 4),
        "train_log_loss": round(train_loss, 4),
        "val_log_loss": round(val_loss, 4),
        "val_accuracy": round(val_acc, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "ece": round(ece, 4),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            splits["y_test"], y_test_pred,
            target_names=["away", "draw", "home"],
            output_dict=True,
        ),
        "reliability_diagram": reliability,
    },
    "backtest": betting_metrics,
    "features": {
        "top_20": top_20_records,
        "feature_importance_plot": imp_plot_path,
        "reliability_diagram_plot": rel_plot_path,
    },
    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_commit": git_commit,
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

t_save = time.time() - t0
print(f"      Report saved to {REPORT_PATH}  ({t_save:.1f}s)")

# Save model
model_path = save_model(model, "baseline_logistic_regression")

# ═══════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════

total_time = sum([t_build, t_split, t_train, t_metrics, t_importance, t_backtest, t_save])

print()
print("=" * 72)
print("  BASELINE COMPLETE")
print("=" * 72)
print(f"  Model:       {report['model']['type']} ({report['model']['solver']})")
print(f"  Test acc:    {test_acc:.2%}")
print(f"  Test logloss:{test_ll:.4f}")
print(f"  Test Brier:  {brier:.4f}")
print(f"  ECE:         {ece:.4f}")
print(f"  ROC-AUC:     {roc_auc:.4f}" if roc_auc else "  ROC-AUC:     N/A")
print(f"  Top feature: {top_20.iloc[0]['feature']} (|coef|={top_20.iloc[0]['abs_coef']:.4f})")
print(f"  Report:      {REPORT_PATH}")
print(f"  Model:       {model_path}")
print(f"  Duration:    {total_time:.1f}s total")
print("=" * 72)
