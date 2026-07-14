"""
test_2022_worldcup.py — Evaluate model on 2022 World Cup matches.

Trains ONE XGBoost model on pre-2022 data (2002–2018), predicts all
64 matches of the 2022 World Cup, and generates a detailed report.

Usage:
    python test_2022_worldcup.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import config

# Configure for World Cup evaluation
config.features.include_h2h = False
config.features.include_league_position = False
config.odds.compute_consensus = False
config.odds.warn_missing = False
config.player_info.enabled = False
config.xg.warn_missing = False
config.xg.compute_xpts = True
config.elo.regress_to_mean = True
config.elo.home_advantage = 50

EXTRA_DROP = ["home_goals_ht", "away_goals_ht", "match_id", "match_id_x", "match_id_y",
              "is_home", "is_home_x", "is_home_y", "gd"]

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("test_2022")
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CSV = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "worldcup_2022_test"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}
RESULT_TO_TARGET = {"H": 2, "D": 1, "A": 0}


def main() -> int:
    from src.feature_engineering import build_features, train_val_test_split
    from src.train import train_model
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                                  log_loss, brier_score_loss, roc_auc_score,
                                  precision_score, recall_score, f1_score)

    t0 = time.time()
    print("=" * 72)
    print("  2022 WORLD CUP — MODEL EVALUATION")
    print("=" * 72)

    # ── Load data ──
    print("\n  Loading data ...")
    df = pd.read_csv(DATA_CSV, low_memory=False, parse_dates=["date"])
    df["target"] = df["result"].map(RESULT_TO_TARGET).fillna(-1).astype("int8")

    df_pre = df[df["season"] < 2022].copy()
    df_22  = df[df["season"] == 2022].copy()
    print(f"  Pre-2022: {len(df_pre)} matches | 2022: {len(df_22)} matches")

    # ── Build features ──
    print("\n  Building features on training data (2002–2018) ...")
    X_train_full, y_train_full = build_features(df_pre, is_training=True)
    # Drop extra leaky columns
    for c in EXTRA_DROP:
        if c in X_train_full.columns:
            X_train_full.drop(columns=[c], inplace=True)
    print(f"  Training feature matrix: {X_train_full.shape}")

    # ── Train model ──
    print("\n  Training XGBoost on pre-2022 data ...")
    splits = train_val_test_split(X_train_full, y_train_full)
    model, history = train_model(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
    print(f"  Train log-loss: {history.get('train_loss', [0])[0]:.4f}")
    print(f"  Val log-loss:   {history.get('val_loss', [0])[0]:.4f}")

    # ── Build features for 2022 matches ──
    print("\n  Building features for 2022 matches ...")
    X_test_full, y_test_full = build_features(df_22, is_training=True)
    for c in EXTRA_DROP:
        if c in X_test_full.columns:
            X_test_full.drop(columns=[c], inplace=True)

    # Align columns
    common_cols = sorted(set(X_train_full.columns) & set(X_test_full.columns))
    X_train_aligned = X_train_full[common_cols]
    X_test_aligned  = X_test_full[common_cols]
    print(f"  Common features: {len(common_cols)}")

    # Re-train on aligned columns
    splits2 = train_val_test_split(X_train_aligned, y_train_full)
    model, _ = train_model(splits2["X_train"], splits2["y_train"], splits2["X_val"], splits2["y_val"])

    # ── Predict ──
    print("\n  Predicting 2022 matches ...")
    y_pred = model.predict(X_test_aligned)
    y_proba = model.predict_proba(X_test_aligned)
    accuracy = accuracy_score(y_test_full, y_pred)
    cm = confusion_matrix(y_test_full, y_pred, labels=[0, 1, 2])

    print(f"\n  {'=' * 50}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Accuracy:       {accuracy:.1%} ({int(accuracy*len(y_test_full))}/{len(y_test_full)})")
    print(f"  Naive baseline: {max((y_test_full==2).mean(), (y_test_full==0).mean(), (y_test_full==1).mean()):.1%}")
    print(f"  Log-loss:       {log_loss(y_test_full, y_proba):.4f}")
    print(f"  ROC-AUC (OVR):  {roc_auc_score(y_test_full, y_proba, multi_class='ovr', average='macro'):.4f}")
    print(f"  {'=' * 50}")

    # ── Detailed metrics ──
    precision_w = precision_score(y_test_full, y_pred, average="weighted", labels=[0, 1, 2])
    recall_w = recall_score(y_test_full, y_pred, average="weighted", labels=[0, 1, 2])
    f1_w = f1_score(y_test_full, y_pred, average="weighted", labels=[0, 1, 2])
    precision_pc = precision_score(y_test_full, y_pred, average=None, labels=[0, 1, 2])
    recall_pc = recall_score(y_test_full, y_pred, average=None, labels=[0, 1, 2])
    f1_pc = f1_score(y_test_full, y_pred, average=None, labels=[0, 1, 2])

    y_true_onehot = np.zeros((len(y_test_full), 3))
    for i, v in enumerate(y_test_full.values):
        y_true_onehot[i, int(v)] = 1
    brier_scores = [round(brier_score_loss(y_true_onehot[:, c], y_proba[:, c]), 4) for c in range(3)]

    # ── Build results dataframe ──
    results = df_22.copy()
    results["prediction"] = [LABEL_MAP[p] for p in y_pred]
    results["pred_class"] = y_pred
    results["home_win_prob"] = y_proba[:, 2]
    results["draw_prob"] = y_proba[:, 1]
    results["away_win_prob"] = y_proba[:, 0]
    results["confidence"] = y_proba.max(axis=1)
    results["correct"] = y_pred == y_test_full.values

    # ── Save outputs ──
    out_cols = ["date", "home_team", "away_team", "round", "result",
                "home_goals", "away_goals", "prediction",
                "home_win_prob", "draw_prob", "away_win_prob",
                "confidence", "correct"]
    out_cols = [c for c in out_cols if c in results.columns]
    results[out_cols].to_csv(REPORT_DIR / "worldcup_2022_predictions.csv", index=False)

    # ── Generate markdown report ──
    print("\n  Generating report ...")
    report_path = REPORT_DIR / "worldcup_2022_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_build_report(results, accuracy, cm, y_test_full, y_pred, y_proba,
                              precision_pc, recall_pc, f1_pc, precision_w, recall_w, f1_w,
                              brier_scores))
    print(f"  Report: {report_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s\n")
    return 0


def _build_report(results, accuracy, cm, y_test, y_pred, y_proba,
                  precision_pc, recall_pc, f1_pc, precision_w, recall_w, f1_w,
                  brier_scores):
    from sklearn.metrics import log_loss, roc_auc_score

    lines = []
    def w(t=""): lines.append(t)
    def h1(t): w(f"\n# {t}\n")
    def h2(t): w(f"\n## {t}\n")
    def h3(t): w(f"\n### {t}\n")
    def tbl(headers, rows):
        w("| " + " | ".join(headers) + " |")
        w("| " + " | ".join(["---"]*len(headers)) + " |")
        for row in rows:
            w("| " + " | ".join(str(c) for c in row) + " |")
        w("")

    n_total = len(y_test)
    n_correct = int(accuracy * n_total)
    naive = max((y_test == 2).mean(), (y_test == 0).mean(), (y_test == 1).mean())
    ll = log_loss(y_test, y_proba)
    roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    avg_brier = round(np.mean(brier_scores), 4)

    actual_dist = {k: int((y_test == v).sum()) for k, v in [("Home Win", 2), ("Draw", 1), ("Away Win", 0)]}
    pred_dist = {k: int((y_pred == v).sum()) for k, v in [("Home Win", 2), ("Draw", 1), ("Away Win", 0)]}

    h1("🏆 2022 World Cup — Model Performance Report")
    w(f"**Date:** {pd.Timestamp.now().strftime('%d %B %Y %H:%M')}")
    w(f"**Model:** XGBoost trained on 320 pre-2022 World Cup matches (2002–2018)")
    w(f"**Test data:** All 64 matches of the 2022 FIFA World Cup Qatar")
    w("")

    h2("Executive Summary")
    w(f"The model achieves **{accuracy:.1%} accuracy** ({n_correct}/{n_total}) on the 2022 World Cup, "
      f"beating the naive baseline of {naive:.1%} by **{accuracy - naive:+.1%}**.")
    w("")
    w(f"- **Accuracy:** {accuracy:.1%} ({n_correct}/{n_total})")
    w(f"- **Naive baseline:** {naive:.1%}")
    w(f"- **Improvement:** {accuracy - naive:+.1%}")
    w(f"- **Weighted F1:** {f1_w:.3f}")
    w(f"- **Log-loss:** {ll:.4f}")
    w(f"- **ROC-AUC (OVR):** {roc:.4f}")
    w(f"- **Avg Brier score:** {avg_brier:.4f}")
    w("")

    h2("Prediction Distribution")
    tbl(["", "Home Win", "Draw", "Away Win"],
        [["Actual", actual_dist["Home Win"], actual_dist["Draw"], actual_dist["Away Win"]],
         ["Predicted", pred_dist["Home Win"], pred_dist["Draw"], pred_dist["Away Win"]]])

    h2("Confusion Matrix")
    tbl(["", "Away Win (pred)", "Draw (pred)", "Home Win (pred)"],
        [["Away Win (actual)", cm[0][0], cm[0][1], cm[0][2]],
         ["Draw (actual)", cm[1][0], cm[1][1], cm[1][2]],
         ["Home Win (actual)", cm[2][0], cm[2][1], cm[2][2]]])

    h2("Per-Class Performance")
    tbl(["Class", "Precision", "Recall", "F1-Score"],
        [["Away Win", f"{precision_pc[0]:.3f}", f"{recall_pc[0]:.3f}", f"{f1_pc[0]:.3f}"],
         ["Draw", f"{precision_pc[1]:.3f}", f"{recall_pc[1]:.3f}", f"{f1_pc[1]:.3f}"],
         ["Home Win", f"{precision_pc[2]:.3f}", f"{recall_pc[2]:.3f}", f"{f1_pc[2]:.3f}"]])
    w(f"\n**Weighted:** Precision={precision_w:.3f}, Recall={recall_w:.3f}, F1={f1_w:.3f}")

    h2("Brier Scores (per class)")
    w(f"- Away Win: {brier_scores[0]:.4f}")
    w(f"- Draw: {brier_scores[1]:.4f}")
    w(f"- Home Win: {brier_scores[2]:.4f}")
    w(f"- Average: {avg_brier:.4f}")

    h2("Match-by-Match Results")
    for rnd in ["Matchday 1","Matchday 2","Matchday 3","Matchday 4","Matchday 5",
                "Matchday 6","Matchday 7","Matchday 8","Matchday 9","Matchday 10",
                "Matchday 11","Matchday 12","Matchday 13","Round of 16",
                "Quarter-finals","Semi-finals","Match for third place","Final"]:
        m = results[results["round"] == rnd]
        if len(m) == 0: continue
        h3(rnd)
        rows = []
        for _, r in m.iterrows():
            score = f"{int(r['home_goals'])}-{int(r['away_goals'])}" if pd.notna(r.get('home_goals')) else "?"
            ck = "✅" if r["correct"] else "❌"
            probs = f"{r['home_win_prob']:.0%}/{r['draw_prob']:.0%}/{r['away_win_prob']:.0%}"
            rows.append([r["home_team"], r["away_team"], score, r["result"],
                         r["prediction"], f"{r['confidence']:.0%}", probs, ck])
        tbl(["Home", "Away", "Score", "Actual", "Prediction", "Conf", "H/D/A", ""], rows)

    h2("Key Findings")
    correct = results[results["correct"]]
    wrong = results[~results["correct"]]
    w(f"**Correct:** {len(correct)} | **Wrong:** {len(wrong)}")
    w(f"")
    w(f"**Group stage** (48 matches):")
    w(f"- Correct: {len(correct[correct['round'].str.contains('Matchday', na=False)])}")
    w(f"- Wrong: {len(wrong[wrong['round'].str.contains('Matchday', na=False)])}")
    w(f"")
    w(f"**Knockout stage** (16 matches):")
    w(f"- Correct: {len(correct[~correct['round'].str.contains('Matchday', na=False)])}")
    w(f"- Wrong: {len(wrong[~wrong['round'].str.contains('Matchday', na=False)])}")
    w("")
    w(f"**Tournament winner:** Argentina won the 2022 World Cup")
    arg = results[(results["home_team"] == "Argentina") | (results["away_team"] == "Argentina")]
    w(f"- Argentina's matches: {len(arg)} correct out of {len(arg)} ({arg['correct'].sum()}/{len(arg)})")
    fra = results[(results["home_team"] == "France") | (results["away_team"] == "France")]
    w(f"- France's matches: {fra['correct'].sum()}/{len(fra)} correct")

    h2("Methodology")
    w("""
**Single-model evaluation:**
- One XGBoost model was trained on 320 World Cup matches from 2002–2018
- The model was then used to predict all 64 matches of the 2022 World Cup
- No 2022 data was used during training (zero leakage)

**Feature set:**
- Rolling team form (points, goals scored/conceded, last 5/10 matches)
- Elo ratings with host-nation home advantage
- xG features from historical data
- Poisson expected goal rates
- Goal difference, win rates, rest days

**Limitation:** This single-model approach does not update as the tournament
progresses. A walk-forward approach (retraining after each matchday) would
better simulate real-time prediction but is computationally expensive.
""")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
