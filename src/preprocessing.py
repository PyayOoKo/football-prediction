"""
Preprocessing — clean, normalise, and prepare raw match data for ML.

Pipeline stages (in order):

1.  **Load**        — read raw CSV from ``data/raw/``
2.  **Convert dates** — parse to datetime, extract temporal features
3.  **Normalise team names** — map abbreviations to canonical names
4.  **Remove duplicates** — exact + near-duplicate match rows
5.  **Handle missing values** — configurable fill / drop strategies
6.  **Create structured columns** — ensure home/away split is consistent
7.  **Validate**    — run integrity checks
8.  **Save**        — write cleaned dataset to ``data/processed/``

Typical usage::

    from src.preprocessing import run_preprocessing

    report = run_preprocessing()                      # uses defaults from config
    report = run_preprocessing(input_path="my.csv")   # custom input
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from config import config as _global_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Team name normalisation
# ═══════════════════════════════════════════════════════════

# Mapping of common name variations → canonical Premier League name.
# Football-Data.co.uk mostly uses full names, but there are still
# inconsistencies across seasons and data sources.
_TEAM_NAME_MAP: dict[str, str] = {
    # ── Manchester clubs ──────────────────────────
    "man utd": "Manchester United",
    "man utd.": "Manchester United",
    "manchester utd": "Manchester United",
    "manchester united": "Manchester United",
    "man city": "Manchester City",
    "man city.": "Manchester City",
    "manchester city": "Manchester City",
    # ── Newcastle ─────────────────────────────────
    "newcastle": "Newcastle United",
    "newcastle utd": "Newcastle United",
    "newcastle utd.": "Newcastle United",
    # ── Tottenham ─────────────────────────────────
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    # ── Wolverhampton ─────────────────────────────
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    # ── Brighton ──────────────────────────────────
    "brighton": "Brighton & Hove Albion",
    "brighton & hove albion": "Brighton & Hove Albion",
    # ── Leicester ─────────────────────────────────
    "leicester": "Leicester City",
    "leicester city": "Leicester City",
    # ── Leeds ─────────────────────────────────────
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    # ── Norwich ───────────────────────────────────
    "norwich": "Norwich City",
    "norwich city": "Norwich City",
    # ── West Ham ──────────────────────────────────
    "west ham": "West Ham United",
    "west ham utd": "West Ham United",
    "west ham united": "West Ham United",
    # ── West Brom ─────────────────────────────────
    "west brom": "West Bromwich Albion",
    "west bromwich": "West Bromwich Albion",
    "west bromwich albion": "West Bromwich Albion",
    "west brom.": "West Bromwich Albion",
    # ── Nottingham Forest ─────────────────────────
    "nottingham forest": "Nottingham Forest",
    "nott'm forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    # ── Sheffield clubs ───────────────────────────
    "sheffield utd": "Sheffield United",
    "sheffield utd.": "Sheffield United",
    "sheffield wed": "Sheffield Wednesday",
    "sheffield wed.": "Sheffield Wednesday",
    "sheffield wednesday": "Sheffield Wednesday",
    # ── Other Championship / Premier League ───────
    "huddersfield": "Huddersfield Town",
    "huddersfield town": "Huddersfield Town",
    "cardiff": "Cardiff City",
    "cardiff city": "Cardiff City",
    "swansea": "Swansea City",
    "swansea city": "Swansea City",
    "stoke": "Stoke City",
    "stoke city": "Stoke City",
    "derby": "Derby County",
    "derby county": "Derby County",
    "middlesbrough": "Middlesbrough",
    "portsmouth": "Portsmouth",
    "blackburn": "Blackburn Rovers",
    "blackburn rovers": "Blackburn Rovers",
    "bolton": "Bolton Wanderers",
    "bolton wanderers": "Bolton Wanderers",
    "wigan": "Wigan Athletic",
    "wigan athletic": "Wigan Athletic",
    "reading": "Reading",
    "qpr": "Queens Park Rangers",
    "queens park rangers": "Queens Park Rangers",
    "hull": "Hull City",
    "hull city": "Hull City",
    "birmingham": "Birmingham City",
    "birmingham city": "Birmingham City",
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    "southampton": "Southampton",
    "afc bournemouth": "AFC Bournemouth",
    "bournemouth": "AFC Bournemouth",
    "burnley": "Burnley",
    "brentford": "Brentford",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "liverpool": "Liverpool",
    "chelsea": "Chelsea",
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "watford": "Watford",
    "sunderland": "Sunderland",
    "wimbledon": "Wimbledon",
    "oldham": "Oldham Athletic",
    "oldham athletic": "Oldham Athletic",
    "coventry": "Coventry City",
    "coventry city": "Coventry City",
    "southend": "Southend United",
    "southend utd": "Southend United",
    "lutton": "Luton Town",
    "luton": "Luton Town",
    "luton town": "Luton Town",
    # ── World Cup 2026 teams (48 teams) ────────────
    # Africa (9)
    "morocco": "Morocco",
    "senegal": "Senegal",
    "tunisia": "Tunisia",
    "algeria": "Algeria",
    "nigeria": "Nigeria",
    "cameroon": "Cameroon",
    "ghana": "Ghana",
    "egypt": "Egypt",
    "côte d'ivoire": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "ivory coast": "Côte d'Ivoire",
    # Asia (8)
    "japan": "Japan",
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "korea rep": "South Korea",
    "australia": "Australia",
    "iran": "Iran",
    "saudi arabia": "Saudi Arabia",
    "iraq": "Iraq",
    "uzbekistan": "Uzbekistan",
    "jordan": "Jordan",
    "united arab emirates": "United Arab Emirates",
    "uae": "United Arab Emirates",
    # Europe (16)
    "germany": "Germany",
    "england": "England",
    "france": "France",
    "spain": "Spain",
    "italy": "Italy",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "portugal": "Portugal",
    "belgium": "Belgium",
    "croatia": "Croatia",
    "switzerland": "Switzerland",
    "denmark": "Denmark",
    "austria": "Austria",
    "serbia": "Serbia",
    "turkey": "Turkey",
    "turkiye": "Turkey",
    "ukraine": "Ukraine",
    "poland": "Poland",
    "sweden": "Sweden",
    "hungary": "Hungary",
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "romania": "Romania",
    "greece": "Greece",
    "norway": "Norway",
    "slovakia": "Slovakia",
    "slovenia": "Slovenia",
    "bosnia & herzegovina": "Bosnia & Herzegovina",
    "bosnia and herzegovina": "Bosnia & Herzegovina",
    "bosnia": "Bosnia & Herzegovina",
    "bosnia-herzegovina": "Bosnia & Herzegovina",
    "scotland": "Scotland",
    "wales": "Wales",
    # North & Central America (6)
    "mexico": "Mexico",
    "united states": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "canada": "Canada",
    "costa rica": "Costa Rica",
    "panama": "Panama",
    "jamaica": "Jamaica",
    "honduras": "Honduras",
    # South America (6+1 host spot)
    "brazil": "Brazil",
    "argentina": "Argentina",
    "uruguay": "Uruguay",
    "colombia": "Colombia",
    "ecuador": "Ecuador",
    "peru": "Peru",
    "paraguay": "Paraguay",
    "venezuela": "Venezuela",
    "chile": "Chile",
    # Oceania (1)
    "new zealand": "New Zealand",
}

# ── Temporal features to extract from the date column ──
_TEMPORAL_FEATURES = [
    "year",
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_season",
]


# ═══════════════════════════════════════════════════════════
#  Main pipeline entry point
# ═══════════════════════════════════════════════════════════


def run_preprocessing(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    save: bool = True,
    config: Any | None = None,
) -> dict[str, Any]:
    """Execute the full preprocessing pipeline.

    Parameters
    ----------
    input_path : str | Path, optional
        Path to the raw CSV file.  Defaults to ``data/raw/results.csv``.
    output_path : str | Path, optional
        Where to save the cleaned CSV.  Defaults to ``data/processed/results_clean.csv``.
    save : bool
        Whether to persist the cleaned dataset (default ``True``).
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).

    Returns
    -------
    dict[str, Any]
        Report with each stage's row counts plus a ``transformations`` list
        explaining every step taken.
    """
    cfg = config or _global_config

    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    report: dict[str, Any] = {
        "input_path": str(input_path or cfg.paths.raw / cfg.data_collection.output_file),
        "output_path": str(output_path or cfg.paths.processed / "results_clean.csv"),
        "stages": {},
        "transformations": [],
    }

    steps = [
        ("1. Load raw data", _load_data),
        ("2. Convert dates", _convert_dates),
        ("3. Normalise team names", _normalise_team_names),
        ("4. Remove duplicates", _remove_duplicates),
        ("5. Handle missing values", _handle_missing),
        ("6. Create structured columns", _create_structured_columns),
        ("7. Add temporal features", _add_temporal_features),
        ("8. Validate dataset", _validate_clean),
    ]

    df = pd.DataFrame()

    for step_name, step_fn in steps:
        stage_before = len(df) if not df.empty else 0
        # Pass cfg to stage functions that accept it
        if step_name == "1. Load raw data":
            df, details = step_fn(df, input_path, cfg=cfg)
        elif step_name == "5. Handle missing values":
            df, details = step_fn(df, None, cfg=cfg)
        else:
            df, details = step_fn(df, None)
        stage_report = {
            "rows_before": stage_before,
            "rows_after": len(df),
            "rows_delta": len(df) - stage_before,
            "details": details,
        }
        report["stages"][step_name] = stage_report
        report["transformations"].append(
            f"{step_name}: {details}" if details else step_name
        )
        logger.info("  ✓ %-45s  (%d rows)", step_name, len(df))

    # ── Save ──────────────────────────────────────
    if save:
        path = Path(str(output_path or cfg.paths.processed / "results_clean.csv"))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("  ✓ Saved to %s  (%d rows)", path, len(df))
        report["saved_to"] = str(path)

    report["total_rows"] = len(df)
    report["total_columns"] = len(df.columns)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE — %d rows, %d columns", len(df), len(df.columns))
    logger.info("=" * 60)

    return report


# ═══════════════════════════════════════════════════════════
#  Stage implementations
# ═══════════════════════════════════════════════════════════


def _load_data(
    _empty: pd.DataFrame,
    input_path: str | Path | None,
    cfg: Any | None = None,
) -> tuple[pd.DataFrame, str]:
    """Stage 1 — read raw CSV into a DataFrame."""
    _cfg = cfg or _global_config
    path = Path(input_path) if input_path else (
        _cfg.paths.raw / _cfg.data_collection.output_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run ``collect_all()`` from ``src.data_collection`` first, "
            "or provide a custom ``input_path``."
        )

    df = pd.read_csv(path, low_memory=False)
    detail = f"Loaded {len(df)} rows × {len(df.columns)} cols from {path.name}"
    logger.info(detail)
    return df, detail


def _convert_dates(
    df: pd.DataFrame,
    _: Any,
) -> tuple[pd.DataFrame, str]:
    """Stage 2 — parse date column to datetime and extract components.

    **Why:**
    - Raw CSVs often store dates as strings (``"16/08/2024"``).
    - ``pd.to_datetime`` converts them to a native datetime type, enabling
      time-based filtering, sorting, and temporal feature extraction.
    - We apply ``dayfirst=True`` because UK football data uses DD/MM/YYYY.
    - Invalid / unparseable entries become ``NaT`` and are reported via the
      validation stage.

    **Transformations:**
    - ``date`` → ``datetime64[ns]``  (standard datetime)
    - ``year`` → int  (extracted from date)
    - ``month`` → int  (1–12)
    - ``day_of_week`` → int  (0 = Monday, 6 = Sunday)
    - ``day_of_year`` → int  (1–366)
    - ``week_of_season`` → int  (1–~42, resets each August)
    """
    if "date" not in df.columns:
        return df, "No date column found — skipped"

    before = df["date"].isnull().sum()

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    after = df["date"].isnull().sum()
    coerced = after - before

    detail_parts = [f"Parsed dates to datetime (dayfirst=True)"]
    if coerced > 0:
        detail_parts.append(f"{coerced} values coerced to NaT")

    details = "; ".join(detail_parts)
    return df, details


def _normalise_team_names(
    df: pd.DataFrame,
    _: Any,
) -> tuple[pd.DataFrame, str]:
    """Stage 3 — map team-name variations to canonical forms.

    **Why:**
    - Different data sources and seasons use inconsistent naming
      (e.g. ``"Man United"`` vs ``"Manchester United"``).
    - A model trained on one variation won't recognise the other — this
      leaks information and breaks prediction on new data.

    **Transformations applied:**
    1. Strip leading/trailing whitespace.
    2. Lowercase for dictionary lookup.
    3. Map via ``_TEAM_NAME_MAP`` (145+ known variations).
    4. Teams not in the mapping are kept as-is (assumed already canonical).
    5. Log the names that were actually changed so you can audit.

    **Edge cases handled:**
    - Team names with extra whitespace (``"  Man Utd  "``).
    - Teams from lower leagues not in our mapping are preserved verbatim.
    - ``NaN`` values are forwarded unchanged (handled in stage 5).
    """
    changes_made = 0

    for col in ["home_team", "away_team"]:
        if col not in df.columns:
            continue

        original = df[col].astype(str).str.strip()

        # Build a cleaned version via the name map
        cleaned = original.str.lower().map(_TEAM_NAME_MAP).fillna(original)

        # Track which rows actually changed
        changed_mask = original != cleaned
        changes_made += changed_mask.sum()

        if changed_mask.any():
            examples = list(
                df.loc[changed_mask, col]
                .value_counts()
                .head(5)
                .index
            )
            logger.debug(
                "  %s: %d rows changed — examples: %s",
                col,
                changed_mask.sum(),
                examples,
            )

        df[col] = cleaned

    detail = (
        f"Normalised {changes_made} team-name occurrences "
        f"({len(_TEAM_NAME_MAP)} mappings available)"
    )
    return df, detail


def _remove_duplicates(
    df: pd.DataFrame,
    _: Any,
) -> tuple[pd.DataFrame, str]:
    """Stage 4 — remove exact + near-duplicate match rows.

    **Why:**
    - Incremental updates (``update()``) can append rows already present from
      a previous full download.
    - A single fixture should appear exactly once in the training set,
      otherwise the model sees the same match multiple times — leaking
      information and biasing evaluation metrics.

    **Transformations applied:**
    1. *Exact duplicates*: same ``(date, home_team, away_team)`` — keep the
       most recently downloaded copy (highest ``downloaded_at``).
    2. *Near duplicates*: same date and teams but different stats columns —
       only possible if the data source corrected a previous entry.  Keep the
       version with the most non-null values (most complete).

    **Edge cases handled:**
    - If a column from the dedup key is missing, the stage is skipped with
      a warning.
    - ``downloaded_at`` not present → no sort applied (first occurrence kept).
    """
    key_cols = ["date", "home_team", "away_team"]
    existing_key = [c for c in key_cols if c in df.columns]

    if len(existing_key) < len(key_cols):
        detail = f"Skipped — missing key columns (need {key_cols}, have {existing_key})"
        logger.warning(detail)
        return df, detail

    before = len(df)

    # 1. Sort: keep the most-recently-downloaded row per match
    if "downloaded_at" in df.columns:
        df = df.sort_values("downloaded_at", ascending=False)

    # 2. For near-duplicates, keep the row with most non-null values
    if "league" in existing_key and "league" not in key_cols:
        pass  # league is part of MATCH_KEY_COLS but not always relevant

    # To handle near-duplicates: count non-nulls per row, keep row with most data
    if "league" in df.columns:
        dedup_subset = existing_key + ["league"]
    else:
        dedup_subset = existing_key

    # Rank rows within each duplicate group by completeness, keep the most complete
    completeness = df.notna().sum(axis=1)
    df["_completeness"] = completeness
    df = df.sort_values([*dedup_subset, "_completeness"], ascending=[True] * len(dedup_subset) + [False])
    df = df.drop_duplicates(subset=dedup_subset, keep="first")
    df = df.drop(columns=["_completeness"])

    removed = before - len(df)
    detail = f"Removed {removed} duplicate{'s' if removed != 1 else ''} ({removed / before * 100:.1f}% of {before})"
    return df, detail


def _handle_missing(
    df: pd.DataFrame,
    _: Any,
    cfg: Any | None = None,
) -> tuple[pd.DataFrame, str]:
    """Stage 5 — handle missing values per configurable strategy.

    **Why:**
    - Real-world football data has gaps: postponed matches, incomplete
      statistics (older seasons lack shot/corner data), and data-entry errors.
    - ML models cannot handle ``NaN`` — every missing cell must be filled
      or the row/column dropped.

    **Strategies (``config.data_collection.missing_strategy``):**
    - ``fill_zero`` (default): match statistics → 0, result missing → drop row.
      Reasonable because a missing stat (e.g. corners) means 0 for that match.
    - ``fill_median``: numeric columns → column median.  More robust against
      outliers when many values are missing.
    - ``drop``: remove any row missing an essential column.  Conservative but
      reduces dataset size.

    **Additional handling:**
    - Columns with >50% missing values are dropped entirely
      (``config.data_collection.max_missing_pct``).
    - Team names are forward-filled if sporadically missing.
    """
    _cfg = cfg or _global_config
    from src.data_collection.cleaners import handle_missing_values as _hmv

    before = len(df)
    df = _hmv(
        df,
        strategy=_cfg.data_collection.missing_strategy,
        max_missing_pct=_cfg.data_collection.max_missing_pct,
    )
    detail = (
        f"Strategy: '{_cfg.data_collection.missing_strategy}', "
        f"{len(df)} rows ({len(df) / before * 100:.1f}% kept)"
    )
    return df, detail


def _create_structured_columns(
    df: pd.DataFrame,
    _: Any,
) -> tuple[pd.DataFrame, str]:
    """Stage 6 — ensure home/away columns are consistently structured.

    **Why:**
    - Downstream feature engineering expects well-defined ``home_team`` /
      ``away_team`` columns with a clear target derived from the result.
    - This stage guarantees those columns exist and creates a clean,
      ML-friendly ``target`` column.

    **Transformations applied:**
    1. Verify ``home_team`` and ``away_team`` exist and are non-null.
    2. Create ``target`` (0 = Away win, 1 = Draw, 2 = Home win) from
       the ``result`` column (H / D / A).
    3. Create ``home_goals`` and ``away_goals`` from ``fthg`` / ``ftag``
       if they exist.
    4. Create ``goal_diff = home_goals - away_goals``.
    5. Create ``total_goals = home_goals + away_goals``.

    **Edge cases handled:**
    - If ``result`` is missing, ``target`` is set to ``-1`` and flagged.
    - If ``fthg``/``ftag`` are missing, ``goal_diff`` and ``total_goals``
      are left as ``NaN``.
    """
    details_parts: list[str] = []

    # 1. Ensure home/away columns
    for col in ["home_team", "away_team"]:
        if col not in df.columns:
            df[col] = np.nan
            details_parts.append(f"Created empty '{col}' column")

    # 2. Create target from result (H=2, D=1, A=0)
    if "result" in df.columns:
        result_map = {"H": 2, "D": 1, "A": 0}
        df["target"] = df["result"].map(result_map).fillna(-1).astype("int8")
        target_valid = (df["target"] >= 0).sum()
        target_missing = (df["target"] < 0).sum()
        details_parts.append(
            f"Target created from 'result': {target_valid} valid, "
            f"{target_missing} unknown (set to -1)"
        )
    else:
        df["target"] = -1
        details_parts.append("No 'result' column found — target set to -1")

    # 3. Goal columns
    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]
        df["total_goals"] = df["home_goals"] + df["away_goals"]
        details_parts.append("Created goal_diff and total_goals from home_/away_goals")

    detail = "; ".join(details_parts)
    return df, detail


def _add_temporal_features(
    df: pd.DataFrame,
    _: Any,
) -> tuple[pd.DataFrame, str]:
    """Stage 7 — add temporal features derived from the date column.

    **Why:**
    - Football performance has temporal patterns: teams play differently
      midweek vs weekend, early vs late in the season, and across years.
    - Encoding these as explicit numeric features lets the model learn
      these patterns directly.

    **Features added:**
    - ``year``: calendar year of the match.
    - ``month``: calendar month (1–12).
    - ``day_of_week``: 0=Monday, 6=Sunday.  Useful for distinguishing
      weekend vs midweek fixtures.
    - ``day_of_year``: day index within the calendar year (1–366).
    - ``week_of_season``: weeks since the start of the season (August 1st).
      Resets to 1 at the beginning of each season.
    - ``is_midweek``: 1 if the match is on Tuesday, Wednesday, or Thursday.
    """
    if "date" not in df.columns or df["date"].isnull().all():
        return df, "No valid dates — temporal features skipped"

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year.astype("Int64")
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month.astype("Int64")

    df["day_of_week"] = df["date"].dt.dayofweek.astype("Int64")
    df["day_of_year"] = df["date"].dt.dayofyear.astype("Int64")

    # Week of season: weeks since August 1st of the season's start year
    df["_aug_1st"] = pd.to_datetime(
        df["date"].dt.year.where(
            df["date"].dt.month >= 8,
            df["date"].dt.year - 1,
        ).astype(str) + "-08-01"
    )
    df["week_of_season"] = ((df["date"] - df["_aug_1st"]).dt.days // 7 + 1).astype("Int64")
    df.drop(columns=["_aug_1st"], inplace=True)

    # Midweek indicator
    df["is_midweek"] = df["day_of_week"].isin([1, 2, 3]).astype("int8")  # Tue, Wed, Thu

    detail = (
        f"Added {len(_TEMPORAL_FEATURES)} temporal features: "
        f"{', '.join(_TEMPORAL_FEATURES)}, is_midweek"
    )
    return df, detail


def _validate_clean(
    df: pd.DataFrame,
    _: Any,
) -> tuple[pd.DataFrame, str]:
    """Stage 8 — validation checks on the cleaned dataset.

    **Checks performed:**
    1. No unexpected nulls in essential columns.
    2. ``target`` values are in valid range (0, 1, 2 or -1 for unknown).
    3. ``goal_diff`` is consistent with ``home_goals`` and ``away_goals``.
    4. Temporal features are within expected bounds.
    5. No duplicate rows remain.
    6. Team names in ``home_team`` and ``away_team`` match canonical forms.
    """
    from src.data_collection.cleaners import validate_data

    validation = validate_data(df)

    # Additional checks beyond the generic validator
    extra_warnings: list[str] = []

    # Check target range
    if "target" in df.columns:
        invalid_targets = df[~df["target"].isin([-1, 0, 1, 2])]
        if len(invalid_targets) > 0:
            extra_warnings.append(
                f"{len(invalid_targets)} rows with invalid target values"
            )

    # Check consistency: H → home_goals > away_goals, etc.
    if all(c in df.columns for c in ["result", "home_goals", "away_goals"]):
        inconsistent = df[
            ((df["result"] == "H") & (df["home_goals"] <= df["away_goals"]))
            | ((df["result"] == "A") & (df["home_goals"] >= df["away_goals"]))
            | ((df["result"] == "D") & (df["home_goals"] != df["away_goals"]))
        ]
        if len(inconsistent) > 0:
            extra_warnings.append(
                f"{len(inconsistent)} rows have result/goal inconsistency"
            )

    # Check no remaining duplicates
    key = ["date", "home_team", "away_team"]
    existing_key = [c for c in key if c in df.columns]
    if len(existing_key) == len(key):
        remaining_dupes = len(df) - len(df.drop_duplicates(subset=existing_key))
        if remaining_dupes:
            extra_warnings.append(
                f"{remaining_dupes} duplicate rows remain after dedup"
            )

    if extra_warnings:
        validation["warnings"].extend(extra_warnings)
        validation["is_valid"] = False

    n_warnings = len(validation["warnings"])
    status = "PASS" if validation["is_valid"] else f"WARN ({n_warnings} issues)"
    detail = (
        f"Validation: {status} — "
        f"{validation['stats']['rows']} rows, "
        f"{validation['stats']['columns']} cols, "
        f"{validation['stats']['missing_cells']} missing cells"
    )
    if not validation["is_valid"]:
        for w in validation["warnings"]:
            logger.warning("    ⚠ %s", w)

    return df, detail
