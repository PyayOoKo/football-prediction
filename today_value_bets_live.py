"""
today_value_bets_live.py — Today's value bets using Dixon-Coles features, LIVE odds,
and probability calibration (Platt scaling).

Usage:
    python today_value_bets_live.py
    python today_value_bets_live.py --force-hardcoded
    python today_value_bets_live.py --days 3
    python today_value_bets_live.py --dixon-coles      # Enable Dixon-Coles (slower but may improve quality)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("today_value_bets")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CSV = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "value_bets"

# Fallback odds: (home, away) -> (away_odds, draw_odds, home_odds)
FALLBACK_ODDS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("Portugal", "Spain"):         (2.80, 3.10, 2.50),
    ("USA", "Belgium"):            (3.75, 3.40, 1.95),
    ("Argentina", "Egypt"):        (7.50, 4.40, 1.40),
    ("Switzerland", "Colombia"):   (3.25, 3.00, 2.35),
    ("France", "Morocco"):         (3.25, 3.10, 2.30),
    ("Norway", "England"):         (4.20, 3.60, 1.85),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Value bets with live odds + calibration")
    p.add_argument("--force-hardcoded", action="store_true", help="Skip API")
    p.add_argument("--live-only", action="store_true", help="Fail if API unavailable")
    p.add_argument("--sport", default="soccer_fifa_world_cup")
    p.add_argument("--bookmaker", default=None)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly", type=float, default=0.25)
    p.add_argument("--max-odds", type=float, default=30.0,
                   help="Maximum decimal odds to accept (default: 30.0). "
                   "Bets above this are rejected to manage variance.")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--dixon-coles", action="store_true",
                   help="Enable Dixon-Coles features (slower but may improve accuracy)")
    p.add_argument("--calibrate", type=str, default="hybrid",
                   choices=["platt", "isotonic", "hybrid", "none"],
                   help=("Calibration method (default: hybrid)\n"
                         "platt=logistic regression, "
                         "isotonic=non-parametric, "
                         "hybrid=isotonic-tails+platt-mid (best for extreme odds), "
                         "none=skip calibration"))
    p.add_argument("--quiet", action="store_true",
                   help="Minimal output (for scheduled runs)")
    p.add_argument("--log-file", type=str, default=None,
                   help="Path to log file (appends output)")
    p.add_argument("--xgboost", action="store_true",
                   help="Force training XGBoost from scratch instead of "
                        "using the pre-trained 3-model blend")
    return p.parse_args(argv)


def get_odds(match_key, live_odds, fallback):
    if match_key in live_odds:
        od = live_odds[match_key]
        return ([od["away_odds"], od["draw_odds"], od["home_odds"]],
                f"LIVE ({od.get('bookmaker','API')})")
    if match_key in fallback:
        return (list(fallback[match_key]), "FALLBACK")
    rev = (match_key[1], match_key[0])
    if rev in live_odds:
        od = live_odds[rev]
        return ([od["home_odds"], od["draw_odds"], od["away_odds"]], "LIVE (swapped)")
    if rev in fallback:
        a, d, h = fallback[rev]
        return ([h, d, a], "FALLBACK (swapped)")
    return None


def main(argv=None):
    args = parse_args(argv)

    # Set up logging to file if requested
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)

    # In --quiet mode, suppress all console output from logging
    if args.quiet:
        root = logging.getLogger()
        for h in root.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root.removeHandler(h)

    def log(msg: str = "") -> None:
        """Log to file; print to stdout unless --quiet."""
        logging.getLogger("today_value_bets").info(msg)
        if not args.quiet:
            print(msg)

    def log_separator(title: str = "") -> None:
        log("=" * 100)
        if title:
            log(f"  {title}")
            log("=" * 100)

    dc_tag = " + Dixon-Coles" if args.dixon_coles else ""
    log_separator(f"WORLD CUP 2026 - VALUE BETS{dc_tag}  (Live Odds + Platt Calibration)")

    # -- 1. Load data --
    if not DATA_CSV.exists():
        log(f"  [X] Data not found at {DATA_CSV}")
        return 1

    df = pd.read_csv(DATA_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log(f"\n  Loaded {len(df)} matches")

    # -- 2. Prepare data --
    df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    completed = df[df["result"].notna()].copy()

    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=args.days)
    upcoming = df[df["result"].isna() & (df["date"] >= today) & (df["date"] <= cutoff)].copy()
    upcoming = upcoming[~upcoming["home_team"].str.contains(r"^[WL]\d+|TBD", na=False)].copy()

    log(f"  Training: {len(completed)} matches  |  Predict: {len(upcoming)} matches")
    if len(upcoming) == 0:
        log("  No upcoming matches in the selected window.")
        return 0

    # -- 3. Build features ONCE on combined data --
    log("\n  Building features (this may take a moment)...")
    from config import config
    config.dixon_coles.enabled = args.dixon_coles
    if args.dixon_coles:
        config.dixon_coles.decay_halflife_days = 1460.0
    config.features.include_h2h = True
    config.elo.regress_to_mean = True
    config.player_info.enabled = True

    from src.feature_engineering import build_features
    from src.train import train_model

    combined = pd.concat([completed, upcoming], ignore_index=True)
    X_all, y_all = build_features(combined)

    # Track upcoming rows by unique key
    upcoming_keys = set(
        f"{r['date']}|{r['home_team']}|{r['away_team']}"
        for _, r in upcoming.iterrows()
    )
    combined_keys = [
        f"{combined.iloc[i]['date']}|{combined.iloc[i]['home_team']}|{combined.iloc[i]['away_team']}"
        for i in range(len(combined))
    ]
    upcoming_idx = [i for i, k in enumerate(combined_keys) if k in upcoming_keys]
    train_idx = [i for i in range(len(combined)) if i not in upcoming_idx]

    X_train = X_all.iloc[train_idx].copy()
    y_train = y_all.iloc[train_idx].copy() if len(y_all) > 0 else pd.Series(dtype=float)
    X_pred = X_all.iloc[upcoming_idx].copy()
    pred_meta = combined.iloc[upcoming_idx].copy()

    log(f"  Features: {X_all.shape[1]} columns  |  Train: {len(X_train)} rows")

    # -- 4. Train model with optional calibration --
    log("  Training XGBoost model...")
    model, history = train_model(X_train, y_train)
    tl = history.get("train_loss", [0])[-1]
    log(f"  Train log-loss: {tl:.4f}")

    # -- 4b. Use 3-model blend by default (skip if --xgboost) --
    use_blend = not args.xgboost
    if use_blend:
        blend_path = PROJECT_ROOT / "models" / "three_model_blend.joblib"
        if blend_path.exists():
            log("  Loading 3-model blend ...")
            from src.models.three_model_blend import ThreeModelBlend

            blend = ThreeModelBlend.load(str(blend_path), historical_df=combined)

            # Predict using blend for all matchups
            blend_preds = blend.predict_matches(pred_meta)

            # Build probs array from blend results
            probs = np.column_stack([
                blend_preds["away_win_prob"].values,
                blend_preds["draw_prob"].values,
                blend_preds["home_win_prob"].values,
            ])

            predictions = pd.DataFrame({
                "date": pred_meta["date"].values,
                "round": pred_meta["round"].values,
                "home_team": pred_meta["home_team"].values,
                "away_team": pred_meta["away_team"].values,
                "home_win_prob": probs[:, 2],
                "draw_prob": probs[:, 1],
                "away_win_prob": probs[:, 0],
                "over_2_5_prob": blend_preds["over_2_5_prob"].values,
                "under_2_5_prob": blend_preds["under_2_5_prob"].values,
                "btts_prob": blend_preds["btts_prob"].values,
                "btts_no_prob": blend_preds["btts_no_prob"].values,
            })
            predictions["pick"] = np.argmax(probs, axis=1).astype(int)
            predictions["pick_label"] = predictions["pick"].map({0: "Away", 1: "Draw", 2: "Home"})
            log(f"\n  Predicted {len(predictions)} matches (3-model blend)")

            # Show calibrated probabilities with O/U and BTTS
            log(f"\n  {'Match':<30} {'Home':<10} {'Draw':<10} {'Away':<10} {'Pick':<10}  {'O/U':<10} {'BTTS':<10}")
            log(f"  {'-' * 80}")
            for _, r in predictions.iterrows():
                ms = f"{r['home_team'][:12]} vs {r['away_team'][:12]}"
                ou = f"O{r['over_2_5_prob']:.0%}" if 'over_2_5_prob' in predictions else ""
                bt = f"B{r['btts_prob']:.0%}" if 'btts_prob' in predictions else ""
                log(f"  {ms:<30} {r['home_win_prob']:<9.1%} {r['draw_prob']:<9.1%} {r['away_win_prob']:<9.1%} {r['pick_label']:<10}  {ou:<10} {bt:<10}")

            # Skip calibration section for blend (go directly to odds)
            predict_model = None  # signal that probs are already computed
        else:
            log("  [i] 3-model blend not found — training XGBoost instead")
            log("  Run 'run_pipeline.py' first to train and save the blend.")
            use_blend = False
            predict_model = model
    else:
        predict_model = model

    # -- 5. Apply calibration --
    use_calibration = args.calibrate != "none" if not use_blend else False
    if use_calibration:
        log(f"\n  -- CALIBRATION: {args.calibrate.upper()} --")
        from sklearn.metrics import log_loss
        from src.calibration import CalibratedModel, _fit_calibrators, calibration_report

        # Use last 20% of training data for calibration (chronological holdout)
        split = int(len(X_train) * 0.8)
        X_cal = X_train.iloc[split:].copy()
        y_cal = y_train.iloc[split:].copy()
        X_train_fit = X_train.iloc[:split].copy()
        y_train_fit = y_train.iloc[:split].copy()

        # Retrain base model on reduced training set
        cal_model, _ = train_model(X_train_fit, y_train_fit)

        # Wrap with calibration
        calibrated = CalibratedModel(
            base_model=cal_model,
            method=args.calibrate,
            n_classes=3,
        )
        calibrated._calibrators = _fit_calibrators(
            cal_model.predict_proba(X_cal),
            y_cal.values,
            3,
            args.calibrate,
        )
        calibrated._fitted = True
        log(f"  Calibrators fitted on {len(y_cal)} validation samples")

        # Show calibration report before vs after on calibration set
        raw_probs_cal = cal_model.predict_proba(X_cal)
        cal_probs_cal = calibrated.predict_proba(X_cal)
        raw_ll = log_loss(y_cal, raw_probs_cal)
        cal_ll = log_loss(y_cal, cal_probs_cal)
        log(f"  Raw log-loss:       {raw_ll:.4f}")
        log(f"  Calibrated log-loss:{cal_ll:.4f}")
        log(f"  Improvement:        {raw_ll - cal_ll:+.4f}")

        # Use calibrated model for predictions
        predict_model = calibrated
    else:
        log("  No calibration applied")
        predict_model = model

    # -- 6. Predict (skip if blend already computed probs) --
    if not use_blend:
        probs = predict_model.predict_proba(X_pred)  # [prob_away, prob_draw, prob_home]
        predictions = pd.DataFrame({
            "date": pred_meta["date"].values,
            "round": pred_meta["round"].values,
            "home_team": pred_meta["home_team"].values,
            "away_team": pred_meta["away_team"].values,
            "home_win_prob": probs[:, 2],
            "draw_prob": probs[:, 1],
            "away_win_prob": probs[:, 0],
        })
        predictions["pick"] = np.argmax(probs, axis=1).astype(int)
        predictions["pick_label"] = predictions["pick"].map({0: "Away", 1: "Draw", 2: "Home"})
        log(f"\n  Predicted {len(predictions)} matches (calibrated={'yes' if use_calibration else 'no'})")

        # Show calibrated probabilities
        log(f"\n  {'Match':<30} {'Home':<10} {'Draw':<10} {'Away':<10} {'Pick':<10}")
        log(f"  {'-' * 70}")
        for _, r in predictions.iterrows():
            ms = f"{r['home_team'][:12]} vs {r['away_team'][:12]}"
            log(f"  {ms:<30} {r['home_win_prob']:<9.1%} {r['draw_prob']:<9.1%} {r['away_win_prob']:<9.1%} {r['pick_label']:<10}")

    # -- 7. Fetch LIVE odds --
    log("\n  -- ODDS --")
    live_odds = {}
    if not args.force_hardcoded:
        from src.odds_api import OddsAPIClient
        client = OddsAPIClient(regions=config.odds_api.regions)
        if not client.api_key:
            if args.live_only:
                log("  [X] No API key, --live-only set.")
                return 1
            log("  No API key - using fallback odds.")
        else:
            pairs = list(zip(predictions["home_team"], predictions["away_team"]))
            log(f"  Fetching odds for {len(pairs)} matches...")
            live_odds = client.get_value_bet_odds(pairs, sport_key=args.sport,
                                                  bookmaker=args.bookmaker)
            if live_odds:
                log(f"  Got {len(live_odds)}/{len(pairs)} matches from API")
            elif args.live_only:
                log("  [X] No odds returned, --live-only set.")
                return 1
            else:
                log("  API returned no matches - using fallback.")

    src_label = "LIVE" if live_odds else "FALLBACK"
    log(f"  Odds source: {src_label}")

    # -- 8. Compute value bets --
    from src.value_betting import compute_value_bets

    odds_list, probs_list, names_list, extra = [], [], [], []
    for _, r in predictions.iterrows():
        res = get_odds((r["home_team"], r["away_team"]), live_odds, FALLBACK_ODDS)
        if res is None:
            log(f"  WARNING: No odds for {r['home_team']} vs {r['away_team']}")
            continue
        od, src = res
        odds_list.append(od)
        probs_list.append([r["away_win_prob"], r["draw_prob"], r["home_win_prob"]])
        names_list.append((r["home_team"], r["away_team"]))
        extra.append({"date": r["date"], "round": r["round"], "odds_source": src})

    if not odds_list:
        log("  No odds available.")
        return 1

    bets = compute_value_bets(
        odds=np.array(odds_list),
        model_probs=np.array(probs_list),
        team_matches=names_list,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        max_odds=args.max_odds,
    )
    info_map = {f"{h} vs {a}": e for (h, a), e in zip(names_list, extra)}
    bets["date"] = bets["match"].map(lambda m: info_map.get(m, {}).get("date", ""))
    bets["round"] = bets["match"].map(lambda m: info_map.get(m, {}).get("round", ""))
    bets["odds_source"] = bets["match"].map(lambda m: info_map.get(m, {}).get("odds_source", ""))

    # -- 9. Display --
    val = bets[bets["positive_ev"]]

    log_separator()
    cal_tag = f"  Calibration: {args.calibrate.upper()}" if use_calibration else "  Calibration: NONE"
    log(f"  VALUE BET RESULTS  |  {cal_tag}  |  Kelly: {args.kelly:.0%}")
    log_separator()

    if len(val) > 0:
        log(f"\n  [VALUE BETS]  {len(val)} opportunities\n")
        cols = ["match", "round", "outcome_label", "decimal_odds",
                "model_prob", "fair_prob", "prob_edge", "ev", "kelly_stake", "odds_source"]
        log(val[cols].to_string(index=False))

        best = val.iloc[0]
        log(f"\n  >> BEST BET: {best['match']} - {best['outcome_label']} @ {best['decimal_odds']:.2f}")
        log(f"     Prob: {best['model_prob']:.1%}  |  Fair: {best['fair_prob']:.1%}  |  Edge: {best['prob_edge']:+1%}  |  EV: {best['ev']:+1%}")
        if best["kelly_stake"] > 0:
            log(f"     Stake: {best['kelly_stake']:.2f} ({best['kelly_pct']:.1%} of bankroll)")
    else:
        log("\n  No value bets found.\n")

    # -- 10. Save --
    if not args.no_save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        p = REPORTS_DIR / f"value_bets_{ts}.csv"
        bets.to_csv(p, index=False)
        log(f"\n  Report saved: {p}")
        # Save a "latest" copy for the Streamlit dashboard
        latest_p = REPORTS_DIR / "latest.csv"
        bets.to_csv(latest_p, index=False)
        log(f"  Dashboard copy: {latest_p}")
        # Also save predictions + metadata for dashboard display
        meta = predictions.copy()
        meta["calibration_method"] = args.calibrate if use_calibration else "none"
        meta["n_matches"] = len(predictions)
        meta["n_value_bets"] = len(val)
        meta["odds_source"] = src_label
        meta_p = REPORTS_DIR / "latest_meta.csv"
        meta.to_csv(meta_p, index=False)
        log(f"  Metadata: {meta_p}")

    log_separator()
    log(f"  Matches: {len(predictions)}  |  Value bets: {len(val)}  |  Odds: {src_label}")
    log_separator()
    return 0


if __name__ == "__main__":
    sys.exit(main())
