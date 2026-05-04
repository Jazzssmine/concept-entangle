import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.concept_sets.caption_sources import iter_captions
from src.concept_sets.caption_subset import CaptionSubsetConfig, build_caption_subsets
from src.concept_sets.io_utils import load_synonyms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build low-storage caption subsets (per-target positives + shared background)."
    )
    parser.add_argument("--targets", nargs="+", required=True, help="Target concepts, e.g. horse cat jellyfish")
    parser.add_argument(
        "--source_type",
        type=str,
        required=True,
        choices=["hf_streaming", "parquet", "csv", "jsonl", "txt"],
        help="Caption source backend",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    # Schema / text selection.
    parser.add_argument("--text_field", type=str, default="text", help="Primary caption text field")
    parser.add_argument(
        "--fallback_text_fields",
        type=str,
        default="caption,prompt",
        help="Comma-separated fallback fields for caption text extraction",
    )

    # HF / parquet source args.
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset id")
    parser.add_argument("--hf_config", type=str, default=None, help="Optional HF config")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="Comma-separated parquet files/URLs for source_type=parquet",
    )
    parser.add_argument("--input_path", type=str, default=None, help="Path for csv/jsonl/txt source")
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable datasets streaming for hf/parquet sources",
    )

    # Subset construction controls.
    parser.add_argument("--positive_per_target", type=int, default=10_000)
    parser.add_argument("--background_size", type=int, default=50_000)
    parser.add_argument(
        "--lexical_mode",
        type=str,
        choices=["strict", "broad"],
        default="strict",
        help="Strict: target+plural; Broad: includes synonyms when provided",
    )
    parser.add_argument(
        "--synonyms_path",
        type=str,
        default=None,
        help="Optional JSON map used only in broad lexical mode",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint_every_rows",
        type=int,
        default=1_0000,
        help="Write partial outputs every N scanned rows (set 0 to disable)",
    )
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on scanned rows")
    parser.add_argument("--min_caption_chars", type=int, default=3)
    parser.add_argument(
        "--allow_positive_in_background",
        action="store_true",
        help="If set, do not exclude positive-caption lines from background sampling",
    )
    parser.add_argument("--example_count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fallback_fields = [x.strip() for x in args.fallback_text_fields.split(",") if x.strip()]
    synonyms_map = load_synonyms(args.synonyms_path)

    caption_iter = iter_captions(
        source_type=args.source_type,
        text_field=args.text_field,
        fallback_text_fields=fallback_fields,
        dataset_name=args.dataset_name,
        split=args.split,
        hf_config=args.hf_config,
        data_files=args.data_files,
        input_path=args.input_path,
        streaming=not args.no_streaming,
    )

    cfg = CaptionSubsetConfig(
        lexical_mode=args.lexical_mode,
        positive_per_target=args.positive_per_target,
        background_size=args.background_size,
        random_seed=args.random_seed,
        progress_every=args.progress_every,
        checkpoint_every_rows=args.checkpoint_every_rows,
        max_rows=args.max_rows,
        min_caption_chars=args.min_caption_chars,
        exclude_positive_from_background=not args.allow_positive_in_background,
        example_count=args.example_count,
    )

    result = build_caption_subsets(
        targets=args.targets,
        caption_iter=caption_iter,
        output_dir=args.output_dir,
        cfg=cfg,
        synonyms_map=synonyms_map,
    )

    print("Caption subset construction completed.")
    print(f"- Background captions: {result['background_path']}")
    print(f"- Summary CSV: {result['summary_csv']}")
    print(f"- Summary JSON: {result['summary_json']}")
    print("- Per-target positives:")
    for target, path in result["positive_paths"].items():
        print(f"  - {target}: {path}")


if __name__ == "__main__":
    main()

