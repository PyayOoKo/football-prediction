#!/usr/bin/env python3
"""
Compute initial ensemble weights using inverse of Brier scores.

Method: weight_i = (1 / brier_i) / sum(1 / brier_j)

Lower Brier = better calibration = higher weight.
All weights are normalized to sum to 1.0.

Usage:
    python scripts/compute_ensemble_weights.py
    python scripts/compute_ensemble_weights.py --method equal
    python scripts/compute_ensemble_weights.py --method softmax --temperature 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"


def load_latest_selection() -> dict[str, Any]:
    """Load the latest ensemble_selection_*.json."""
    files = sorted(REPORT_DIR.glob("ensemble_selection_*.json"))
    if not files:
        print("[FAIL] No ensemble_selection_*.json found. Run select_ensemble_models.py first.")
        sys.exit(1)
    return json.loads(files[-1].read_text(encoding="utf-8"))


def compute_inverse_brier_weights(
    brier_scores: dict[str, float],
) -> dict[str, float]:
    """Compute weights as inverse of Brier scores, normalized to sum to 1.0."""
    inv = {name: 1.0 / max(brier, 1e-10) for name, brier in brier_scores.items()}
    total = sum(inv.values())
    return {name: w / total for name, w in inv.items()}


def compute_equal_weights(
    model_names: list[str],
) -> dict[str, float]:
    """Equal weights for all models."""
    w = 1.0 / max(len(model_names), 1)
    return {name: w for name in model_names}


def compute_softmax_weights(
    brier_scores: dict[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Compute weights using softmax of negative Brier scores.

    Higher temperature = more uniform weights.
    Lower temperature = more concentrated on best model.
    """
    scores = np.array([-brier / temperature for brier in brier_scores.values()])
    scores = scores - scores.max()  # numerical stability
    exp_s = np.exp(scores)
    probs = exp_s / exp_s.sum()
    return {name: float(probs[i]) for i, name in enumerate(brier_scores.keys())}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute initial ensemble weights from Brier scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method", "-m",
        choices=["inverse", "equal", "softmax"],
        default="inverse",
        help="Weight computation method (default: inverse)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float, default=1.0,
        help="Temperature for softmax method (default: 1.0)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    # ── Load selection ──
    selection = load_latest_selection()
    models = selection.get("selected_models", [])

    if not models:
        print("[FAIL] No selected_models in ensemble_selection")
        return 1

    # ── Compute weights ──
    brier_scores = {m["name"]: m["brier_score"] for m in models}

    if args.method == "inverse":
        raw_weights = compute_inverse_brier_weights(brier_scores)
    elif args.method == "equal":
        raw_weights = compute_equal_weights(list(brier_scores.keys()))
    elif args.method == "softmax":
        raw_weights = compute_softmax_weights(brier_scores, args.temperature)
    else:
        raw_weights = compute_inverse_brier_weights(brier_scores)

    # ── Round to 4 decimal places ──
    total = sum(raw_weights.values())
    weights = {name: round(w / total, 4) for name, w in raw_weights.items()}

    # Adjust last weight so sum is exactly 1.0
    diff = round(1.0 - sum(weights.values()), 4)
    if diff != 0:
        last_key = list(weights.keys())[-1]
        weights[last_key] = round(weights[last_key] + diff, 4)

    # ── Save raw weights dict (spec format) + metadata sidecar ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Primary output: flat dict {model_name: weight} matching the spec
    save_path = REPORT_DIR / f"ensemble_weights_{timestamp}.json"
    save_path.write_text(
        json.dumps(weights, indent=2),
        encoding="utf-8",
    )

    # Sidecar: metadata (for reproducibility)
    meta = {
        "timestamp": timestamp,
        "method": args.method,
        "temperature": args.temperature if args.method == "softmax" else None,
        "source_file": Path(selection.get("source_file", "?")).name,
        "weighted_average_brier": round(
            sum(weights[n] * brier_scores[n] for n in weights), 4
        ),
        "per_model": [
            {
                "name": m["name"],
                "brier_score": m["brier_score"],
                "weight": weights[m["name"]],
                "type": m["type"],
            }
            for m in models
        ],
    }
    meta_path = REPORT_DIR / f"ensemble_weights_{timestamp}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ── Print ──
    if not args.quiet:
        print(f"\n  {'=' * 50}")
        print(f"  ENSEMBLE WEIGHTS  [{args.method}]")
        print(f"  {'=' * 50}")
        print(f"\n  {'Model':<20s} {'Brier':<10s} {'Weight':<10s} {'InvBrier':<10s}")
        print(f"  {'-' * 50}")
        inv_total = sum(1.0 / max(brier_scores[n], 1e-10) for n in weights)
        for m in models:
            name = m["name"]
            brier = m["brier_score"]
            w = weights[name]
            inv = (1.0 / max(brier, 1e-10)) / inv_total if args.method == "inverse" else 0
            inv_str = f"{inv:.4f}" if args.method == "inverse" else "—"
            print(f"  {name:<20s} {brier:<10.4f} {w:<10.4f} {inv_str:<10s}")
        print(f"  {'-' * 50}")
        print(f"  {'Total':<20s} {'':<10s} {sum(weights.values()):<10.4f}")
        avg_brier = round(sum(weights[n] * brier_scores[n] for n in weights), 4)
        print(f"\n  Weighted avg Brier: {avg_brier:.4f}")
        print(f"\n  Saved: {save_path.name}")
        print(f"  Metadata: {meta_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
