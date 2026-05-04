import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

from src.concept_sets.embedding_utils import EmbeddingConfig
from src.concept_sets.io_utils import load_captions, load_concept_vocab, load_group_metadata, load_synonyms
from src.concept_sets.pipeline import run_concept_set_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build concept evaluation sets for unlearning analysis.")
    parser.add_argument("--targets", nargs="+", required=True, help="Target concepts, e.g. horse cat jellyfish")
    parser.add_argument("--captions_path", type=str, default=None, help="Path to captions/prompts file")
    parser.add_argument(
        "--caption_subsets_dir",
        type=str,
        default=None,
        help="Directory from build_caption_subsets.py with background and per-target positive captions",
    )
    parser.add_argument(
        "--concept_vocab_path",
        type=str,
        required=True,
        help="Path to candidate concept vocabulary (txt/csv/json)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for JSON/CSV outputs")

    parser.add_argument(
        "--synonyms_path",
        type=str,
        default=None,
        help="Optional JSON map for broad lexical variants: {target: [syn1, syn2]}",
    )
    parser.add_argument(
        "--group_metadata_path",
        type=str,
        default=None,
        help="Optional concept->group metadata (json/csv) for taxonomy-aware ranking",
    )
    parser.add_argument(
        "--context_lexical_mode",
        type=str,
        choices=["strict", "broad"],
        default="broad",
        help="Use strict or broad lexical set when selecting target-matched captions",
    )

    parser.add_argument("--top_k_context", type=int, default=30)
    parser.add_argument("--top_k_neighbors", type=int, default=20)
    parser.add_argument("--top_k_controls", type=int, default=20)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--min_token_len", type=int, default=3)
    parser.add_argument("--no_bigrams", action="store_true", help="Disable short phrase extraction")
    parser.add_argument(
        "--include_corpus_cooccurrence",
        action="store_true",
        help="Include concept-level corpus co-occurrence as ranking signal",
    )
    parser.add_argument("--min_neighbor_similarity", type=float, default=-1.0)
    parser.add_argument("--max_control_similarity", type=float, default=0.2)
    parser.add_argument(
        "--max_control_cooccurrence_df",
        type=int,
        default=1,
        help="Maximum concept-level co-occurrence doc frequency allowed for controls",
    )
    parser.add_argument(
        "--max_control_pmi",
        type=float,
        default=-5.0,
        help="Maximum PMI allowed for control PMI-fallback selection (lower is stricter)",
    )
    parser.add_argument(
        "--control_similarity_weight",
        type=float,
        default=2.0,
        help="Weight for embedding dissimilarity in control scoring",
    )
    parser.add_argument(
        "--control_negative_pmi_weight",
        type=float,
        default=0.15,
        help="Reward weight for negative PMI in control scoring",
    )
    parser.add_argument(
        "--control_positive_pmi_penalty",
        type=float,
        default=0.10,
        help="Penalty weight for positive PMI in control scoring",
    )
    parser.add_argument(
        "--control_cooccurrence_penalty",
        type=float,
        default=0.01,
        help="Penalty weight for control co-occurrence doc frequency",
    )

    parser.add_argument(
        "--embedding_provider",
        type=str,
        choices=["clip", "sentence"],
        default="clip",
        help="Embedding backend for semantic similarity",
    )
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--sentence_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--prompt_template", type=str, default="a photo of a {}")
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--embedding_device", type=str, default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--embedding_cache_path", type=str, default=None, help="Optional embedding cache file path")
    return parser.parse_args()


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        key = x.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def main() -> None:
    args = parse_args()
    if bool(args.captions_path) == bool(args.caption_subsets_dir):
        raise ValueError("Provide exactly one of --captions_path or --caption_subsets_dir")

    if args.caption_subsets_dir:
        subset_dir = Path(args.caption_subsets_dir)
        if not subset_dir.exists():
            raise FileNotFoundError(f"caption_subsets_dir not found: {subset_dir}")

        files = [subset_dir / "background_captions.txt"]
        files.extend(subset_dir / f"{target}_positive_captions.txt" for target in args.targets)
        existing_files = [p for p in files if p.exists()]
        if not existing_files:
            raise FileNotFoundError(
                f"No expected caption subset files found in {subset_dir}. "
                "Expected background_captions.txt and/or <target>_positive_captions.txt"
            )
        captions = []
        for fp in existing_files:
            captions.extend(load_captions(str(fp)))
        captions = _dedup_keep_order(captions)
    else:
        captions = load_captions(args.captions_path)

    concept_vocab = load_concept_vocab(args.concept_vocab_path)
    synonyms_map = load_synonyms(args.synonyms_path)
    group_map = load_group_metadata(args.group_metadata_path)

    emb_kwargs = {
        "provider": args.embedding_provider,
        "clip_model_name": args.clip_model_name,
        "sentence_model_name": args.sentence_model_name,
        "prompt_template": args.prompt_template,
        "batch_size": args.embedding_batch_size,
        "cache_path": args.embedding_cache_path,
    }
    if args.embedding_device is not None:
        emb_kwargs["device"] = args.embedding_device
    embedding_cfg = EmbeddingConfig(**emb_kwargs)

    run_concept_set_pipeline(
        targets=args.targets,
        captions=captions,
        concept_vocab=concept_vocab,
        output_dir=args.output_dir,
        synonyms_map=synonyms_map,
        group_map=group_map,
        top_k_context=args.top_k_context,
        top_k_neighbors=args.top_k_neighbors,
        top_k_controls=args.top_k_controls,
        min_frequency=args.min_frequency,
        min_token_len=args.min_token_len,
        include_bigrams=not args.no_bigrams,
        include_corpus_cooccurrence=args.include_corpus_cooccurrence,
        context_lexical_mode=args.context_lexical_mode,
        min_neighbor_similarity=args.min_neighbor_similarity,
        max_control_similarity=args.max_control_similarity,
        max_control_cooccurrence_df=args.max_control_cooccurrence_df,
        max_control_pmi=args.max_control_pmi,
        control_similarity_weight=args.control_similarity_weight,
        control_negative_pmi_weight=args.control_negative_pmi_weight,
        control_positive_pmi_penalty=args.control_positive_pmi_penalty,
        control_cooccurrence_penalty=args.control_cooccurrence_penalty,
        embedding_cfg=embedding_cfg,
    )

    print(f"Saved outputs to: {args.output_dir}")
    print("Inspect:")
    print(f"- {args.output_dir}/concept_sets_summary.csv")
    print(f"- {args.output_dir}/per_target/*.json")
    print(f"- {args.output_dir}/intermediate/*")


if __name__ == "__main__":
    main()

