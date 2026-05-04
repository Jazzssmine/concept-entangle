#!/usr/bin/env python3
"""Extract NP/CP per-concept rows from a *\_clip_detailed_breakdowns.csv slice for selected models."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="e.g. outputs/benchmark/eval/horse_clip_detailed_breakdowns.csv",
    )
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Filter target_concept (default: keep all rows in file)",
    )
    parser.add_argument("--models", nargs="+", default=["base", "stereo"])
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.target is not None:
        t = str(args.target).strip().lower()
        df = df[df["target_concept"].astype(str).str.strip().str.lower() == t]
    sub = df[
        df["model_name"].isin(args.models) & df["metric_component"].isin(["NP", "CP"])
    ].copy()
    sub = sub.sort_values(["metric_component", "concept", "model_name"]).reset_index(drop=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, na_rep="NA")
    print(f"Wrote {len(sub)} rows -> {out}")


if __name__ == "__main__":
    main()
