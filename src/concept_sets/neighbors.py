from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .embedding_utils import cosine_similarity
from .text_utils import normalize_phrase, normalize_text


def _tokenize_for_phrase_matching(text: str) -> list[str]:
    return [t for t in normalize_text(text).split(" ") if t]


def _ngram_tuples(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _build_phrase_index(concepts: list[str]) -> tuple[dict[tuple[str, ...], list[str]], int]:
    index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    max_n = 1
    for concept in concepts:
        toks = tuple(_tokenize_for_phrase_matching(concept))
        if not toks:
            continue
        index[toks].append(concept)
        max_n = max(max_n, len(toks))
    return index, max_n


def compute_concept_doc_freq(
    captions: list[str],
    concepts: list[str],
) -> tuple[dict[str, int], int]:
    """Compute concept-level document frequency using exact phrase matching."""
    phrase_index, max_n = _build_phrase_index(concepts)
    doc_freq = defaultdict(int)

    for cap in captions:
        tokens = _tokenize_for_phrase_matching(cap)
        seen_in_doc = set()
        for n in range(1, max_n + 1):
            for ng in _ngram_tuples(tokens, n):
                if ng in phrase_index:
                    seen_in_doc.update(phrase_index[ng])
        for c in seen_in_doc:
            doc_freq[c] += 1
    return dict(doc_freq), len(captions)


def compute_target_cooccurrence_scores(
    target: str,
    candidate_vocab: list[str],
    concept_doc_freq: dict[str, int],
    captions: list[str],
) -> dict[str, dict]:
    """Compute target-candidate co-occurrence stats and PMI."""
    concepts = sorted(set(candidate_vocab + [target]))
    phrase_index, max_n = _build_phrase_index(concepts)

    co_df = defaultdict(int)
    target_df = 0
    n_docs = len(captions)
    target_norm = normalize_phrase(target)

    for cap in captions:
        tokens = _tokenize_for_phrase_matching(cap)
        seen_in_doc = set()
        for n in range(1, max_n + 1):
            for ng in _ngram_tuples(tokens, n):
                if ng in phrase_index:
                    seen_in_doc.update(phrase_index[ng])
        if target_norm not in seen_in_doc:
            continue
        target_df += 1
        for c in seen_in_doc:
            if c != target_norm:
                co_df[c] += 1

    out = {}
    for c in candidate_vocab:
        c_df = concept_doc_freq.get(c, 0)
        co = co_df.get(c, 0)
        p_xy = co / max(1, n_docs)
        p_x = target_df / max(1, n_docs)
        p_y = c_df / max(1, n_docs)
        pmi = math.log((p_xy + 1e-12) / max(p_x * p_y, 1e-12))
        out[c] = {
            "co_doc_freq": int(co),
            "target_doc_freq": int(target_df),
            "candidate_doc_freq": int(c_df),
            "pmi": float(pmi),
        }
    return out


@dataclass
class NeighborConfig:
    top_k_neighbors: int = 20
    top_k_controls: int = 20
    min_neighbor_similarity: float = -1.0
    max_control_similarity: float = 0.2
    weight_embed: float = 0.7
    weight_group: float = 0.2
    weight_cooc: float = 0.1
    max_control_cooccurrence_df: int = 1
    max_control_pmi: float = -5.0
    enforce_same_group_for_neighbors: bool = False
    prefer_same_group_for_controls: bool = False
    control_similarity_weight: float = 2.0
    control_negative_pmi_weight: float = 0.15
    control_positive_pmi_penalty: float = 0.10
    control_cooccurrence_penalty: float = 0.01


def rank_neighbors_and_controls(
    target: str,
    candidate_vocab: list[str],
    embedding_map: dict,
    cfg: NeighborConfig,
    group_map: dict[str, str] | None = None,
    cooc_stats: dict[str, dict] | None = None,
) -> dict:
    target_norm = normalize_phrase(target)
    group_map = group_map or {}
    cooc_stats = cooc_stats or {}
    target_group = group_map.get(target_norm)

    if target_norm not in embedding_map:
        raise ValueError(f"Missing embedding for target '{target_norm}'")

    candidates_ranked = []
    for cand in candidate_vocab:
        if cand == target_norm:
            continue
        if cand not in embedding_map:
            continue
        emb_sim = cosine_similarity(embedding_map[target_norm], embedding_map[cand])
        same_group = int(bool(target_group and group_map.get(cand) == target_group))
        cooc_pmi = cooc_stats.get(cand, {}).get("pmi", 0.0)
        cooc_df = cooc_stats.get(cand, {}).get("co_doc_freq", 0)

        if cfg.enforce_same_group_for_neighbors and target_group and not same_group:
            continue

        score = cfg.weight_embed * emb_sim + cfg.weight_group * same_group + cfg.weight_cooc * cooc_pmi
        candidates_ranked.append(
            {
                "concept": cand,
                "score": float(score),
                "embedding_similarity": float(emb_sim),
                "same_group": bool(same_group),
                "cooccurrence_pmi": float(cooc_pmi),
                "cooccurrence_doc_freq": int(cooc_df),
            }
        )

    candidates_ranked.sort(key=lambda x: x["score"], reverse=True)

    neighbor_ranked = [c for c in candidates_ranked if c["embedding_similarity"] >= cfg.min_neighbor_similarity]
    final_neighbors = neighbor_ranked[: cfg.top_k_neighbors]
    neighbor_names = {x["concept"] for x in final_neighbors}

    non_neighbor_pool = [c for c in candidates_ranked if c["concept"] not in neighbor_names]

    strict_pool = [
        c
        for c in non_neighbor_pool
        if c["embedding_similarity"] <= cfg.max_control_similarity
        and c["cooccurrence_doc_freq"] <= cfg.max_control_cooccurrence_df
    ]
    pmi_pool = [
        c
        for c in non_neighbor_pool
        if c["cooccurrence_pmi"] <= cfg.max_control_pmi
        and c["cooccurrence_doc_freq"] <= max(5, cfg.max_control_cooccurrence_df)
    ]

    # Progressive fallback strategy:
    # 1) strict thresholding (similarity + co-occurrence)
    # 2) PMI-based anti-association candidates
    # 3) lowest-similarity non-neighbors to guarantee non-empty controls
    selected_map: dict[str, dict] = {}
    for cand in strict_pool:
        selected_map[cand["concept"]] = {**cand, "_mode": "strict"}
    if len(selected_map) < cfg.top_k_controls:
        for cand in pmi_pool:
            if cand["concept"] not in selected_map:
                selected_map[cand["concept"]] = {**cand, "_mode": "pmi_fallback"}
            if len(selected_map) >= cfg.top_k_controls:
                break
    if len(selected_map) < cfg.top_k_controls:
        lowest_sim_pool = sorted(non_neighbor_pool, key=lambda x: x["embedding_similarity"])
        for cand in lowest_sim_pool:
            if cand["concept"] not in selected_map:
                selected_map[cand["concept"]] = {**cand, "_mode": "similarity_fallback"}
            if len(selected_map) >= cfg.top_k_controls:
                break

    control_candidates = []
    for cand in selected_map.values():
        # PMI-aware scoring: penalize positive PMI, reward negative PMI.
        # Keep embedding dissimilarity as the dominant signal for controls.
        ctrl_score = -cfg.control_similarity_weight * cand["embedding_similarity"]
        ctrl_score += cfg.control_negative_pmi_weight * max(0.0, -cand["cooccurrence_pmi"])
        ctrl_score -= cfg.control_positive_pmi_penalty * max(0.0, cand["cooccurrence_pmi"])
        ctrl_score -= cfg.control_cooccurrence_penalty * cand["cooccurrence_doc_freq"]
        if cfg.prefer_same_group_for_controls and target_group:
            ctrl_score += 0.05 if cand["same_group"] else 0.0

        reason_map = {
            "strict": "passes strict low-similarity and low-cooccurrence thresholds",
            "pmi_fallback": "selected by low/negative PMI (anti-association) fallback",
            "similarity_fallback": "selected by lowest similarity fallback after strict/PMI filters",
        }
        control_candidates.append(
            {
                **cand,
                "control_score": float(ctrl_score),
                "selection_reason": reason_map.get(cand["_mode"], "selected"),
            }
        )

    # Prioritize low embedding similarity first, then composite score.
    control_candidates.sort(key=lambda x: (x["embedding_similarity"], -x["control_score"]))
    final_controls = control_candidates[: cfg.top_k_controls]

    return {
        "semantic_neighbors": {
            "candidates_ranked": candidates_ranked,
            "final_top_k": [x["concept"] for x in final_neighbors],
            "final_top_k_scored": final_neighbors,
        },
        "non_neighbor_controls": {
            "candidates_ranked": control_candidates,
            "final_top_k": [x["concept"] for x in final_controls],
            "final_top_k_scored": final_controls,
            "selection_metadata": {
                "strict_pool_size": len(strict_pool),
                "pmi_pool_size": len(pmi_pool),
                "non_neighbor_pool_size": len(non_neighbor_pool),
                "max_control_pmi": cfg.max_control_pmi,
                "control_similarity_weight": cfg.control_similarity_weight,
                "control_negative_pmi_weight": cfg.control_negative_pmi_weight,
                "control_positive_pmi_penalty": cfg.control_positive_pmi_penalty,
                "control_cooccurrence_penalty": cfg.control_cooccurrence_penalty,
            },
        },
    }

