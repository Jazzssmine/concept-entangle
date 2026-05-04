import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.concept_sets.prompt_generation import PromptGenerationConfig, build_prompts_from_concept_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prompt families from per-target concept-set JSON files.")
    parser.add_argument("--concept_sets_dir", type=str, required=True, help="Directory with per-target concept-set JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated prompt files")

    parser.add_argument("--direct_per_target", type=int, default=50)
    parser.add_argument("--indirect_per_target", type=int, default=50)
    parser.add_argument("--neighbor_per_concept", type=int, default=30)
    parser.add_argument("--control_per_concept", type=int, default=30)

    parser.add_argument(
        "--lexical_mode",
        type=str,
        choices=["strict", "broad"],
        default="strict",
        help="Lexical exclusion mode when not generating both indirect modes",
    )
    parser.add_argument(
        "--generate_both_indirect_modes",
        action="store_true",
        help="Generate both indirect_strict and indirect_broad prompts in one run",
    )
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--use_llm_indirect", action="store_true", help="Enable LLM-assisted indirect mode hook")
    parser.add_argument(
        "--allow_text_heavy_auxiliary",
        action="store_true",
        help="Allow prompts with text-heavy cues (poster/sign/logo/etc.)",
    )
    parser.add_argument("--min_prompt_len", type=int, default=10)
    parser.add_argument("--max_prompt_len", type=int, default=220)
    parser.add_argument("--near_duplicate_jaccard_threshold", type=float, default=0.90)
    parser.add_argument("--preview_count", type=int, default=0, help="Print N sample prompts per family/target")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PromptGenerationConfig(
        direct_per_target=args.direct_per_target,
        indirect_per_target=args.indirect_per_target,
        neighbor_per_concept=args.neighbor_per_concept,
        control_per_concept=args.control_per_concept,
        lexical_mode=args.lexical_mode,
        generate_both_indirect_modes=args.generate_both_indirect_modes,
        random_seed=args.random_seed,
        min_prompt_len=args.min_prompt_len,
        max_prompt_len=args.max_prompt_len,
        allow_text_heavy_auxiliary=args.allow_text_heavy_auxiliary,
        use_llm_indirect=args.use_llm_indirect,
        near_duplicate_jaccard_threshold=args.near_duplicate_jaccard_threshold,
    )

    summary = build_prompts_from_concept_sets(
        concept_sets_dir=args.concept_sets_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        preview_count=args.preview_count,
    )
    print(f"Saved prompt outputs to: {args.output_dir}")
    print(f"Targets processed: {len(summary.get('targets', {}))}")
    print(f"Summary JSON: {args.output_dir}/prompt_generation_summary.json")


if __name__ == "__main__":
    main()

