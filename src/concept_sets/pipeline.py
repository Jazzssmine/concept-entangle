from __future__ import annotations

import csv
from pathlib import Path

from .context_words import mine_context_words
from .embedding_utils import EmbeddingConfig, embed_concepts
from .io_utils import ensure_dir, write_json
from .lexical import build_lexical_sets
from .neighbors import NeighborConfig, compute_target_cooccurrence_scores, rank_neighbors_and_controls
from .text_utils import DEFAULT_STOPWORDS, normalize_phrase


def _save_ranked_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_concept_set_pipeline(
    targets: list[str],
    captions: list[str],
    concept_vocab: list[str],
    output_dir: str,
    synonyms_map: dict[str, list[str]] | None = None,
    group_map: dict[str, str] | None = None,
    top_k_context: int = 30,
    top_k_neighbors: int = 20,
    top_k_controls: int = 20,
    min_frequency: int = 2,
    min_token_len: int = 3,
    include_bigrams: bool = True,
    include_corpus_cooccurrence: bool = False,
    context_lexical_mode: str = "broad",
    min_neighbor_similarity: float = -1.0,
    max_control_similarity: float = 0.2,
    max_control_cooccurrence_df: int = 1,
    max_control_pmi: float = -5.0,
    control_similarity_weight: float = 2.0,
    control_negative_pmi_weight: float = 0.15,
    control_positive_pmi_penalty: float = 0.10,
    control_cooccurrence_penalty: float = 0.01,
    embedding_cfg: EmbeddingConfig | None = None,
) -> None:
    out_dir = ensure_dir(output_dir)
    per_target_dir = ensure_dir(out_dir / "per_target")
    intermediate_dir = ensure_dir(out_dir / "intermediate")
    summary_rows = []

    targets_norm = [normalize_phrase(t) for t in targets]
    vocab_norm = sorted(set(normalize_phrase(v) for v in concept_vocab if normalize_phrase(v)))
    stopwords = set(DEFAULT_STOPWORDS)

    # Make sure target concepts exist in embedding universe.
    embedding_concepts = sorted(set(vocab_norm + targets_norm))
    if embedding_cfg is None:
        embedding_cfg = EmbeddingConfig(cache_path=str(out_dir / "cache" / "concept_embeddings.pkl"))
    else:
        if embedding_cfg.cache_path is None:
            embedding_cfg.cache_path = str(out_dir / "cache" / "concept_embeddings.pkl")
    embedding_map = embed_concepts(embedding_concepts, embedding_cfg)

    concept_doc_freq = {}
    if include_corpus_cooccurrence:
        # Reused across all targets for target-candidate PMI.
        from .neighbors import compute_concept_doc_freq

        concept_doc_freq, _ = compute_concept_doc_freq(captions, embedding_concepts)
        write_json(intermediate_dir / "concept_doc_freq.json", concept_doc_freq)

    for target in targets_norm:
        lexical = build_lexical_sets(target=target, synonyms_map=synonyms_map, include_morph_variants=True)
        lexical_strict = lexical.strict
        lexical_broad = lexical.broad
        context_lexical = lexical_broad if context_lexical_mode == "broad" else lexical_strict

        context = mine_context_words(
            target=target,
            lexical_variants=context_lexical,
            captions=captions,
            min_token_len=min_token_len,
            min_frequency=min_frequency,
            top_k=top_k_context,
            include_bigrams=include_bigrams,
            stopwords=stopwords,
            excluded_phrases=set(vocab_norm),
        )

        cooc_stats = {}
        if include_corpus_cooccurrence:
            cooc_stats = compute_target_cooccurrence_scores(
                target=target,
                candidate_vocab=vocab_norm,
                concept_doc_freq=concept_doc_freq,
                captions=captions,
            )
            write_json(intermediate_dir / f"{target}_concept_cooccurrence.json", cooc_stats)

        neighbor_cfg = NeighborConfig(
            top_k_neighbors=top_k_neighbors,
            top_k_controls=top_k_controls,
            min_neighbor_similarity=min_neighbor_similarity,
            max_control_similarity=max_control_similarity,
            max_control_cooccurrence_df=max_control_cooccurrence_df,
            max_control_pmi=max_control_pmi,
            control_similarity_weight=control_similarity_weight,
            control_negative_pmi_weight=control_negative_pmi_weight,
            control_positive_pmi_penalty=control_positive_pmi_penalty,
            control_cooccurrence_penalty=control_cooccurrence_penalty,
        )
        ranked = rank_neighbors_and_controls(
            target=target,
            candidate_vocab=vocab_norm,
            embedding_map=embedding_map,
            cfg=neighbor_cfg,
            group_map=group_map,
            cooc_stats=cooc_stats,
        )

        concept_result = {
            "target": target,
            "lexical_set": {
                "strict": lexical_strict,
                "broad": lexical_broad,
            },
            "context_words": {
                "raw_candidates": context["raw_candidates"],
                "final_top_k": context["final_top_k"],
                "scoring_method": context["scoring_method"],
                "n_captions_target_match": context["n_captions_target_match"],
                "n_captions_total": context["n_captions_total"],
            },
            "semantic_neighbors": ranked["semantic_neighbors"],
            "non_neighbor_controls": ranked["non_neighbor_controls"],
            "metadata": {
                "embedding_provider": embedding_cfg.provider,
                "prompt_template": embedding_cfg.prompt_template,
                "include_corpus_cooccurrence": include_corpus_cooccurrence,
                "context_lexical_mode": context_lexical_mode,
                "min_frequency": min_frequency,
                "min_neighbor_similarity": min_neighbor_similarity,
                "max_control_similarity": max_control_similarity,
                "max_control_cooccurrence_df": max_control_cooccurrence_df,
                "max_control_pmi": max_control_pmi,
                "control_similarity_weight": control_similarity_weight,
                "control_negative_pmi_weight": control_negative_pmi_weight,
                "control_positive_pmi_penalty": control_positive_pmi_penalty,
                "control_cooccurrence_penalty": control_cooccurrence_penalty,
            },
        }

        write_json(per_target_dir / f"{target}.json", concept_result)
        _save_ranked_csv(intermediate_dir / f"{target}_context_raw.csv", context["raw_candidates"])
        _save_ranked_csv(
            intermediate_dir / f"{target}_neighbors_ranked.csv",
            ranked["semantic_neighbors"]["candidates_ranked"],
        )
        _save_ranked_csv(
            intermediate_dir / f"{target}_controls_ranked.csv",
            ranked["non_neighbor_controls"]["candidates_ranked"],
        )

        summary_rows.append(
            {
                "target": target,
                "lexical_strict_size": len(lexical_strict),
                "lexical_broad_size": len(lexical_broad),
                "context_top_k": "|".join(context["final_top_k"]),
                "neighbors_top_k": "|".join(ranked["semantic_neighbors"]["final_top_k"]),
                "controls_top_k": "|".join(ranked["non_neighbor_controls"]["final_top_k"]),
            }
        )

    _save_ranked_csv(out_dir / "concept_sets_summary.csv", summary_rows)

