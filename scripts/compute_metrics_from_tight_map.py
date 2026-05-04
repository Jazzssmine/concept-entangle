#!/usr/bin/env python3
"""
Recompute UA, IRR, IRA, NP, CP, DamageGap from CLIP per-image predictions using a tight
benchmark_maps.*.json (semantic_neighbors + non_neighbor_controls final_top_k).

IRR/IRA/NP/CP use the same definitions as src.benchmark.metrics.compute_per_target_metrics.
NP/CP are restricted to concepts listed in the map for that target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.evaluators import load_concept_map
from src.benchmark.io_utils import ensure_dir, save_csv
from src.benchmark.metrics import aggregate_metrics, compute_per_target_metrics, save_metrics_outputs


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if s.dtype == object:

        def _to_bool(x):
            if x is True or x is False:
                return x
            if pd.isna(x):
                return pd.NA
            return str(x).strip().lower() in {"1", "true", "yes"}

        return s.map(_to_bool)

    return s.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map_json",
        type=str,
        required=True,
        help="e.g. configs/benchmark_maps.horse_tight.json (must set target to eval target, e.g. horse)",
    )
    parser.add_argument(
        "--per_image_csv",
        type=str,
        required=True,
        help="CLIP per-image predictions, e.g. outputs/benchmark/eval/horse_per_image_clip_predictions.csv",
    )
    parser.add_argument(
        "--target_concept",
        type=str,
        default=None,
        help="Keep only this target_concept (default: all targets present in CSV)",
    )
    parser.add_argument("--base_model_name", type=str, default="base")
    parser.add_argument("--min_base_acc_for_normalization", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--prefix",
        type=str,
        default="tight_map",
        help="Output file prefix, e.g. horse_tight_map -> horse_tight_map_per_target_metrics.csv",
    )
    args = parser.parse_args()

    map_path = Path(args.map_json)
    if not map_path.is_absolute():
        map_path = REPO_ROOT / map_path
    pred_path = Path(args.per_image_csv)
    if not pred_path.is_absolute():
        pred_path = REPO_ROOT / pred_path
    out_dir = ensure_dir(Path(args.output_dir) if Path(args.output_dir).is_absolute() else REPO_ROOT / args.output_dir)

    neighbor_map = load_concept_map(map_path, kind="neighbor")
    control_map = load_concept_map(map_path, kind="control")

    pred_df = pd.read_csv(pred_path)
    for col in ("is_correct", "is_target_pred"):
        if col in pred_df.columns:
            pred_df[col] = _coerce_bool_series(pred_df[col])

    if args.target_concept is not None:
        t = str(args.target_concept).strip().lower()
        pred_df = pred_df[pred_df["target_concept"].astype(str).str.strip().str.lower() == t].copy()

    if len(pred_df) == 0:
        raise SystemExit("No rows left after filtering; check --target_concept and CSV path.")

    per_target_df, detailed_df, debug_df = compute_per_target_metrics(
        pred_df=pred_df,
        base_model_name=args.base_model_name,
        compute_cp=True,
        neighbor_map=neighbor_map,
        control_map=control_map,
        min_base_acc_for_normalization=args.min_base_acc_for_normalization,
    )
    aggregated_df = aggregate_metrics(per_target_df)
    prefix = args.prefix.strip("_")
    save_metrics_outputs(
        per_target_df=per_target_df,
        aggregated_df=aggregated_df,
        detailed_df=detailed_df,
        eval_dir=str(out_dir),
        prefix=prefix,
        debug_df=debug_df,
    )

    # Compact summary for quick reading
    summary_cols = ["target_concept", "model_name", "UA", "IRR", "IRA", "CRA", "NP", "CP", "DamageGap"]
    have = [c for c in summary_cols if c in per_target_df.columns]
    summary_path = out_dir / f"{prefix}_summary_metrics.csv"
    save_csv(summary_path, per_target_df[have], na_rep="NA")

    print(f"Map: {map_path}")
    print(f"Neighbors ({list(neighbor_map.keys())[0]}): {neighbor_map}")
    print(f"Controls ({list(control_map.keys())[0]}): {control_map}")
    print(f"Wrote under {out_dir} with prefix {prefix}_*")
    print(summary_path)
    print(per_target_df[have].to_string(index=False))


if __name__ == "__main__":
    main()
