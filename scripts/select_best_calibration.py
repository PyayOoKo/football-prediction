#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Select Best Calibration Method for Each Model                              ║
║                                                                             ║
║  Reads the latest calibration_results_{timestamp}.json and selects the      ║
║  best calibration method (Platt, Isotonic, Temperature, or None) for each   ║
║  model based on the lowest Brier Score on the test set (or validation set   ║
║  if test set is unavailable).                                               ║
║                                                                             ║
║  If no calibration method improves the Brier Score, the original            ║
║  uncalibrated model is selected.                                            ║
║                                                                             ║
║  Output: reports/calibration_selection_{timestamp}.json                     ║
║                                                                             ║
║  Usage:                                                                     ║
║      python scripts/select_best_calibration.py                               ║
║      python scripts/select_best_calibration.py --input reports/...json      ║
║      python scripts/select_best_calibration.py --min-improvement 0.001      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def find_latest_results() -> Path | None:
    """Find the most recent calibration_results JSON file."""
    files = sorted(REPORT_DIR.glob("calibration_results_*.json"))
    return files[-1] if files else None


def load_results(path: Path) -> dict[str, Any]:
    """Load calibration results from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_best_method(
    raw_brier: float,
    calibration_results: dict[str, dict[str, Any]],
    raw_log_loss: float | None = None,
    raw_brier_val: float | None = None,
    min_improvement: float = 0.0,
) -> dict[str, Any]:
    """Select the best calibration method based on lowest Brier Score.

    Parameters
    ----------
    raw_brier : float
        Raw (uncalibrated) Brier Score on test set.
    calibration_results : dict
        Nested dict with keys like "platt_scaling", "isotonic_scaling",
        "temperature_scaling", each containing brier_test, brier_val, etc.
    raw_log_loss : float, optional
        Raw log-loss on test set.
    raw_brier_val : float, optional
        Raw Brier Score on validation set.
    min_improvement : float
        Minimum Brier Score improvement required to select a calibration
        method. Default 0.0 (any improvement counts).

    Returns
    -------
    dict with best_method, original_brier, calibrated_brier,
    original_log_loss, calibrated_log_loss, improvement, method_details.
    """
    selection: dict[str, Any] = {
        "best_method": "none",
        "original_brier": round(raw_brier, 6),
        "calibrated_brier": round(raw_brier, 6),
        "original_log_loss": round(raw_log_loss, 6) if raw_log_loss is not None else None,
        "calibrated_log_loss": raw_log_loss,
        "improvement": 0.0,
        "method_details": {},
    }

    # Method display name mapping
    method_names = {
        "platt_scaling": "Platt",
        "isotonic_scaling": "Isotonic",
        "temperature_scaling": "Temperature",
    }

    best_brier = raw_brier
    best_method = "none"
    best_log_loss = raw_log_loss

    for cal_tag, cres in calibration_results.items():
        if not cres.get("fitted") or cres.get("error"):
            continue

        # Use test Brier if available, otherwise validation Brier
        cal_brier = cres.get("brier_test") or cres.get("brier_val")
        cal_log_loss = cres.get("log_loss_test")

        if cal_brier is None:
            continue

        improvement = raw_brier - cal_brier
        display_name = method_names.get(cal_tag, cal_tag)

        selection["method_details"][display_name] = {
            "brier_score": round(cal_brier, 6),
            "improvement": round(improvement, 6),
            "log_loss": round(cal_log_loss, 6) if cal_log_loss is not None else None,
        }

        if improvement > min_improvement and cal_brier < best_brier:
            best_brier = cal_brier
            best_method = display_name
            best_log_loss = cal_log_loss

    selection["best_method"] = best_method
    selection["calibrated_brier"] = round(best_brier, 6)
    selection["improvement"] = round(raw_brier - best_brier, 6)

    if best_log_loss is not None:
        selection["calibrated_log_loss"] = round(best_log_loss, 6)
    else:
        selection["calibrated_log_loss"] = selection["original_log_loss"]

    # Add temperature value if temperature scaling was selected
    if best_method == "Temperature" and "temperature_scaling" in calibration_results:
        temp_res = calibration_results["temperature_scaling"]
        if temp_res.get("fitted") and "temperature" in temp_res:
            selection["temperature"] = temp_res["temperature"]

    return selection


def run_selection(
    input_path: Path | None = None,
    min_improvement: float = 0.0,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the best calibration method selection pipeline.

    Parameters
    ----------
    input_path : Path, optional
        Path to calibration results JSON. If None, finds the latest.
    min_improvement : float
        Minimum Brier improvement to select a calibration method.
    quiet : bool
        Suppress detailed output.

    Returns
    -------
    dict with selection results and metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load input ──────────────────────────────────
    if input_path is None:
        input_path = find_latest_results()
        if input_path is None:
            print("[FAIL] No calibration_results_*.json found in reports/")
            print("  Run scripts/calibrate_all_models.py first.")
            sys.exit(1)

    print(f"\n  Reading: {input_path}")
    data = load_results(input_path)

    # ── Process each model ──────────────────────────
    selection: dict[str, Any] = {}
    all_results = data.get("phase4", []) + data.get("phase3", [])

    print(f"\n  Selecting best calibration method for {len(all_results)} models...")
    print(f"  Minimum improvement threshold: {min_improvement}")
    print()

    for result in all_results:
        model_name = result["model"]

        # Get raw Brier (prefer test, fall back to val)
        raw_brier = result.get("raw_brier_test") or result.get("raw_brier_val")
        raw_log_loss = result.get("raw_log_loss_test") or result.get("raw_log_loss_val")
        raw_brier_val = result.get("raw_brier_val")

        if raw_brier is None:
            print(f"  [SKIP] {model_name}: no Brier score available")
            continue

        sel = select_best_method(
            raw_brier=raw_brier,
            calibration_results=result.get("calibration_results", {}),
            raw_log_loss=raw_log_loss,
            raw_brier_val=raw_brier_val,
            min_improvement=min_improvement,
        )
        selection[model_name] = sel

        # Print result
        method = sel["best_method"]
        orig = sel["original_brier"]
        cal = sel["calibrated_brier"]
        imp = sel["improvement"]
        if method == "none":
            print(f"  {model_name:<22s} → none   (Brier: {orig:.4f})")
        else:
            print(f"  {model_name:<22s} → {method:<12s} (Brier: {orig:.4f} → {cal:.4f}, Δ={imp:+.4f})")

    # ── Build output ────────────────────────────────
    output = {
        "timestamp": timestamp,
        "min_improvement_threshold": min_improvement,
        "source_file": str(input_path.name),
        "source_timestamp": data.get("timestamp", "?"),
        "models": selection,
        "summary": {
            "total_models": len(selection),
            "calibration_recommended": sum(
                1 for v in selection.values() if v["best_method"] != "none"
            ),
            "no_improvement": sum(
                1 for v in selection.values() if v["best_method"] == "none"
            ),
        },
    }

    # Best overall calibration result
    best_improvement = None
    best_model = None
    for model_name, sel in selection.items():
        if sel["improvement"] > 0:
            if best_improvement is None or sel["improvement"] > best_improvement:
                best_improvement = sel["improvement"]
                best_model = model_name
    if best_model:
        output["summary"]["best_calibration_improvement"] = {
            "model": best_model,
            "method": selection[best_model]["best_method"],
            "improvement": best_improvement,
        }

    # ── Save output ────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"calibration_selection_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {output_path}")
    print(f"\n  Total models: {output['summary']['total_models']}")
    print(f"  Calibration recommended: {output['summary']['calibration_recommended']}")
    print(f"  No improvement (use raw): {output['summary']['no_improvement']}")
    if "best_calibration_improvement" in output["summary"]:
        bi = output["summary"]["best_calibration_improvement"]
        print(f"  Best improvement: {bi['model']} + {bi['method']} (Δ={bi['improvement']:.4f})")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select best calibration method for each model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to calibration_results JSON (default: latest in reports/)",
    )
    parser.add_argument(
        "--min-improvement", "-m",
        type=float,
        default=0.0,
        help="Minimum Brier Score improvement to select calibration (default: 0.0)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-model output",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None

    try:
        run_selection(
            input_path=input_path,
            min_improvement=args.min_improvement,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        print(f"\n[FAIL] Selection failed: {e}")
        return 1


if __name__ == "__main__":
    # Wrap stdout to handle Unicode on Windows terminals
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
