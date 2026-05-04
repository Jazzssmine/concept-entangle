import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split benchmark all_prompts.csv into per-family CSVs for generate-example-img.py."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to all_prompts.csv (expects prompt_family and prompt columns).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where control/direct/indirect/neighbor CSV files are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="evaluation_seed value assigned to every row (default: 0).",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["control", "direct", "indirect", "neighbor"],
        help="Prompt families to export (default: control direct indirect neighbor).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required_cols = {"prompt_family", "prompt"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    for family in args.families:
        subset = df[df["prompt_family"] == family].copy().reset_index(drop=True)
        subset["case_number"] = subset.index
        subset["evaluation_seed"] = args.seed
        subset = subset[["case_number", "prompt", "evaluation_seed"]]

        out_path = output_dir / f"{family}.csv"
        subset.to_csv(out_path, index=False)
        print(f"[ok] {family}: {len(subset)} rows -> {out_path}")


if __name__ == "__main__":
    main()
