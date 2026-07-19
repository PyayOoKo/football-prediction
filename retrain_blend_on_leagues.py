"""
retrain_blend_on_leagues.py — Retrain 3-model blend on EPL + La Liga + World Cup data.

Downloads 5 seasons of EPL and La Liga data from football-data.co.uk,
combines with existing World Cup data (downloaded from openfootball),
preprocesses, retrains XGBoost, fits Poisson + Elo, and re-optimises
blend weights on league-specific validation data.

Usage:
    python retrain_blend_on_leagues.py                         # Full pipeline
    python retrain_blend_on_leagues.py --skip-collection       # Use existing data
    python retrain_blend_on_leagues.py --skip-xgboost          # Use existing XGBoost model
    python retrain_blend_on_leagues.py --dry-run               # Data collection only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain_blend")

PROJECT_ROOT = Path(__file__).resolve().parent


def collect_league_data(
    league_codes: list[str],
    n_seasons: int = 5,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Download league data from football-data.co.uk for given league codes."""
    from src.data_collection.sources.football_data_co_uk import download_bulk
    logger.info("Downloading %d seasons for: %s", n_seasons, ", ".join(league_codes))
    df = download_bulk(leagues=league_codes, max_seasons=n_seasons, include_current=True)
    if df.empty:
        logger.warning("No league data downloaded")
        return df
    league_names = {"E0": "EPL", "SP1": "La_Liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}
    if "league" in df.columns:
        df["league"] = df["league"].map(lambda c: league_names.get(c, c))
    logger.info("League data: %d rows", len(df))
    if output_dir:
        p = output_dir / "league_raw.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return df


def collect_worldcup_data(output_dir: Path | None = None) -> pd.DataFrame:
    """Download World Cup data from openfootball JSON sources (2002-2026)."""
    import requests
    all_dfs = []
    for y in [2002, 2006, 2010, 2014, 2018, 2022, 2026]:
        url = f"https://raw.githubusercontent.com/openfootball/worldcup.json/master/{y}/worldcup.json"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            raw = resp.json().get("matches", [])
            rows = []
            for m in raw:
                ft = (m.get("score") or {}).get("ft") if isinstance(m.get("score"), dict) else None
                result, hg, ag = None, None, None
                if isinstance(ft, (list, tuple)) and len(ft) >= 2:
                    try:
                        hg, ag = int(ft[0]), int(ft[1])
                        result = "H" if hg > ag else ("A" if hg < ag else "D")
                    except (TypeError, ValueError):
                        pass
                rows.append({
                    "season": y, "date": m.get("date"), "league": "WC",
                    "round": m.get("round"), "home_team": m.get("team1", ""),
                    "away_team": m.get("team2", ""), "result": result,
                    "home_goals": hg, "away_goals": ag,
                })
            df = pd.DataFrame(rows)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values(["date", "home_team"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            all_dfs.append(df)
            logger.info("  %d: %d matches", y, len(df))
        except Exception as exc:
            logger.warning("  %d: failed (%s)", y, exc)
    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("WC data: %d rows", len(combined))
    if output_dir:
        p = output_dir / "worldcup_raw.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(p, index=False)
    return combined


def collect_all_data(n_seasons: int = 5, data_dir: Path | None = None) -> pd.DataFrame:
    """Collect EPL + La Liga + World Cup data and combine."""
    data_dir = data_dir or (PROJECT_ROOT / "data" / "raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    df_lg = collect_league_data(["E0", "SP1", "D1", "I1", "F1"], n_seasons=n_seasons, output_dir=data_dir)
    df_wc = collect_worldcup_data(output_dir=data_dir)
    parts = [df for df in [df_lg, df_wc] if not df.empty]
    if not parts:
        raise RuntimeError("No data collected from any source")
    combined = pd.concat(parts, ignore_index=True)
    for c in ["date", "home_team", "away_team", "result", "home_goals", "away_goals"]:
        if c not in combined.columns:
            combined[c] = None
    if "date" in combined.columns:
        combined = combined.sort_values("date").reset_index(drop=True)
    raw_path = data_dir / "combined_raw.csv"
    combined.to_csv(raw_path, index=False)
    logger.info("Combined: %d rows (%d EPL, %d LaLiga, %d WC)",
                len(combined),
                len(combined[combined["league"] == "EPL"]) if "league" in combined.columns else 0,
                len(combined[combined["league"] == "La_Liga"]) if "league" in combined.columns else 0,
                len(combined[combined["league"] == "WC"]) if "league" in combined.columns else 0)
    return combined


def run_preprocessing(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Run preprocessing pipeline on raw match data.

    Handles date parsing, team name normalisation, dedup, missing values,
    structured columns, and temporal features.
    """
    df = df_raw.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    from src.preprocessing import _TEAM_NAME_MAP
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            orig = df[col].astype(str).str.strip()
            df[col] = orig.str.lower().map(_TEAM_NAME_MAP).fillna(orig)
    key_cols = ["date", "home_team", "away_team"]
    if all(c in df.columns for c in key_cols):
        before = len(df)
        df = df.sort_values("date").drop_duplicates(subset=key_cols, keep="last")
        if before - len(df):
            logger.info("  Removed %d duplicates", before - len(df))
    for c in ["result", "home_goals", "away_goals"]:
        if c in df.columns:
            df = df.dropna(subset=[c])
    for c in ["home_goals", "away_goals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "result" in df.columns:
        df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0}).fillna(-1).astype("int8")
    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["total_goals"] = df["home_goals"] + df["away_goals"]
        df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["year"] = df["date"].dt.year.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["day_of_week"] = df["date"].dt.dayofweek.astype("Int64")
    df["day_of_year"] = df["date"].dt.dayofyear.astype("Int64")
    df["_aug"] = pd.to_datetime(
        df["year"].where(df["month"] >= 8, df["year"] - 1).astype(str) + "-08-01"
    )
    df["week_of_season"] = ((df["date"] - df["_aug"]).dt.days // 7 + 1).astype("Int64")
    df.drop(columns=["_aug"], inplace=True)
    df["is_midweek"] = df["day_of_week"].isin([1, 2, 3]).astype("int8")
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Preprocessing: %d rows, %d cols", len(df), len(df.columns))
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "results_clean.csv"
    df.to_csv(out, index=False)
    logger.info("Saved to %s", out)
    return df


def train_xgboost(df: pd.DataFrame) -> Any | None:
    """Train an XGBoost classifier on preprocessed data.

    Returns None if data not found or training fails.
    """
    from src.feature_engineering import build_features, train_val_test_split
    from src.train import tune_hyperparameters, train_model, save_model

    data_path = PROJECT_ROOT / "data" / "processed" / "results_clean.csv"
    if not data_path.exists():
        logger.error("Preprocessed data not found at %s", data_path)
        return None

    logger.info("Building features...")
    X, y = build_features(df, is_training=True)
    if X is None or len(X) == 0:
        logger.error("Feature engineering failed")
        return None
    logger.info("  Features: %d x %d", X.shape[0], X.shape[1])

    splits = train_val_test_split(X, y)
    logger.info("  Split: %d/%d/%d", len(splits["X_train"]), len(splits["X_val"]), len(splits["X_test"]))

    logger.info("Tuning hyper-parameters...")
    best_params = tune_hyperparameters(splits["X_train"], splits["y_train"], n_folds=5, n_iter=50)
    logger.info("  Best params: %s", best_params)

    from config import config
    for k, v in best_params.items():
        if hasattr(config.train, k):
            setattr(config.train, k, v)

    model, history = train_model(splits["X_train"], splits["y_train"],
                                 splits["X_val"], splits["y_val"])
    logger.info("  Train loss: %.4f | Val loss: %.4f",
                history.get("train_loss", [0])[0],
                history.get("val_loss", [0])[0])

    save_path = save_model(model, "xgboost_model.joblib")
    logger.info("Model saved to %s", save_path)
    return model


def build_and_optimise_blend(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    xgb_model: Any, data_label: str = "league",
) -> dict[str, Any]:
    """Build ThreeModelBlend, optimise weights, evaluate, save."""
    from src.poisson_model import PoissonModel
    from src.elo import EloSystem
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    result = {}

    poisson = PoissonModel(min_matches=0)
    poisson.fit(train_df)
    logger.info("Poisson fitted on %d matches", len(train_df))

    elo = EloSystem()
    elo.process_matches(train_df)
    logger.info("Elo processed on %d matches", len(train_df))

    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    cond_rates = ConditionalRates.from_data(fit_df)

    blend = ThreeModelBlend(
        poisson_model=poisson, elo_model=elo, xgb_model=xgb_model,
        conditional_rates=cond_rates, historical_df=fit_df,
    )

    logger.info("Optimising weights on %d validation matches...", len(val_df))
    optimised = blend.optimise_weights(val_df, markets=["1X2", "Over2.5", "BTTS", "Over3.5"],
                                       n_grid=6, metric="brier_score", verbose=True)
    result["weights"] = optimised

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_dir = PROJECT_ROOT / "config"
    weights_dir.mkdir(parents=True, exist_ok=True)
    wpath = weights_dir / f"three_model_weights_{data_label}_{ts}.json"
    with open(wpath, "w") as f:
        json.dump({"weights": optimised, "timestamp": ts,
                   "training_config": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}},
                  f, indent=2)
    result["weights_path"] = str(wpath)
    logger.info("Weights saved to %s", wpath)

    logger.info("Evaluating on %d test matches...", len(test_df))
    evaluation = blend.evaluate(test_df)
    result["evaluation"] = evaluation

    for mkt in ["1X2", "Over2.5", "BTTS", "Over3.5"]:
        md = evaluation.get("markets", {}).get(mkt, {})
        bm = md.get("models", {}).get("3-Model Blend", {})
        if bm:
            logger.info("  %s: Brier=%.4f LogLoss=%.4f Acc=%.2f%%",
                        mkt, bm.get("brier_score", 0), bm.get("log_loss", 0),
                        bm.get("accuracy", 0) * 100)

    rp = blend.generate_report(evaluation, output_dir=str(PROJECT_ROOT / "reports"), timestamp=ts)
    result["report_paths"] = rp
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain 3-model blend on EPL+LaLiga+WC")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seasons", type=int, default=5)
    parser.add_argument("--load-processed", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "=" * 72)
    print("  3-MODEL BLEND RETRAINING — EPL + La Liga + World Cup")
    print("=" * 72)

    # Phase 1: Data
    if args.load_processed:
        print(f"\n-- Loading processed data: {args.load_processed} --")
        df = pd.read_csv(args.load_processed, low_memory=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        print(f"  Loaded {len(df):,} rows")
    elif args.skip_collection:
        pp = PROJECT_ROOT / "data" / "processed" / "results_clean.csv"
        if pp.exists():
            print(f"\n-- Loading existing: {pp} --")
            df = pd.read_csv(pp, low_memory=False)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            print(f"  Loaded {len(df):,} rows")
        else:
            print("\n-- No existing data — running collection --")
            df = run_preprocessing(collect_all_data(n_seasons=args.seasons))
    else:
        print(f"\n-- Phase 1: Data Collection --")
        df = run_preprocessing(collect_all_data(n_seasons=args.seasons))

    if args.dry_run:
        print(f"\n  [DRY RUN] Complete. {len(df):,} rows processed.")
        print(f"  Time: {time.time() - t0:.1f}s")
        return 0

    # Phase 2: XGBoost
    if args.skip_xgboost:
        print("\n-- Phase 2: XGBoost (SKIPPED) --")
        import joblib
        xgb = None
        for c in [PROJECT_ROOT / "models" / "xgboost_model.joblib",
                  PROJECT_ROOT / "models" / "worldcup_lightgbm.joblib"]:
            if c.exists():
                xgb = joblib.load(c)
                logger.info("Loaded: %s", c.name)
                break
    else:
        print("\n-- Phase 2: Retraining XGBoost --")
        xgb = train_xgboost(df)

    # Phase 3: Split
    print("\n-- Phase 3: Chronological split (70/15/15) --")
    n = len(df)
    vs, ts = int(n * 0.70), int(n * 0.85)
    train_df, val_df, test_df = df.iloc[:vs].copy(), df.iloc[vs:ts].copy(), df.iloc[ts:].copy()
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Phase 4: Build blend
    print("\n-- Phase 4: 3-Model Blend with optimised weights --")
    br = build_and_optimise_blend(train_df, val_df, test_df, xgb, "league_wc")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Weights: {br.get('weights_path', 'N/A')}")
    print(f"  Report:  {br.get('report_paths', {}).get('report_md', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
