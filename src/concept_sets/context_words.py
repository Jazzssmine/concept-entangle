from __future__ import annotations

import math
from collections import Counter

from .text_utils import contains_any_phrase, normalize_phrase, tokenize_caption


def _build_doc_level_counts(
    captions: list[str],
    min_token_len: int,
    include_bigrams: bool,
    stopwords: set[str] | None,
) -> tuple[Counter[str], int]:
    """Build document-frequency counts for candidate context terms."""
    df_counter: Counter[str] = Counter()
    total_docs = len(captions)
    for cap in captions:
        toks = tokenize_caption(cap, min_token_len=min_token_len, stopwords=stopwords)
        terms = set(toks)
        if include_bigrams:
            terms.update(f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1))
        df_counter.update(terms)
    return df_counter, total_docs


def mine_context_words(
    target: str,
    lexical_variants: list[str],
    captions: list[str],
    min_token_len: int = 3,
    min_frequency: int = 2,
    top_k: int = 30,
    include_bigrams: bool = True,
    stopwords: set[str] | None = None,
    excluded_phrases: set[str] | None = None,
    exclude_terms_containing_target_lexical: bool = True,
) -> dict:
    """Mine correlated context words from target-matched captions."""
    target_norm = normalize_phrase(target)
    lexical_set = {normalize_phrase(x) for x in lexical_variants if normalize_phrase(x)}
    lexical_set.add(target_norm)
    excluded = set(lexical_set)
    if excluded_phrases is not None:
        excluded.update(normalize_phrase(x) for x in excluded_phrases if normalize_phrase(x))

    corpus_df, num_docs = _build_doc_level_counts(
        captions=captions,
        min_token_len=min_token_len,
        include_bigrams=include_bigrams,
        stopwords=stopwords,
    )

    matched_captions = [cap for cap in captions if contains_any_phrase(cap, lexical_set)]
    target_docs = len(matched_captions)

    target_term_df: Counter[str] = Counter()
    for cap in matched_captions:
        toks = tokenize_caption(cap, min_token_len=min_token_len, stopwords=stopwords)
        terms = set(toks)
        if include_bigrams:
            terms.update(f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1))
        target_term_df.update(terms)

    raw_candidates = []
    for term, co_df in target_term_df.items():
        term_norm = normalize_phrase(term)
        if not term_norm or term_norm in excluded:
            continue
        # Remove context terms that still explicitly mention the target (e.g., "horse racing").
        # Unigrams like "racing" remain because they do not contain the lexical phrase.
        if exclude_terms_containing_target_lexical and contains_any_phrase(term_norm, lexical_set):
            continue
        if any(piece.isdigit() for piece in term_norm.split()):
            continue
        if co_df < min_frequency:
            continue

        p_xy = co_df / max(1, num_docs)
        p_x = target_docs / max(1, num_docs)
        p_y = corpus_df.get(term_norm, 0) / max(1, num_docs)
        pmi = math.log((p_xy + 1e-12) / (max(p_x * p_y, 1e-12)))
        # Balanced score keeps common but specific terms near the top.
        score = 0.6 * math.log1p(co_df) + 0.4 * pmi
        raw_candidates.append(
            {
                "term": term_norm,
                "co_doc_freq": int(co_df),
                "corpus_doc_freq": int(corpus_df.get(term_norm, 0)),
                "pmi": float(pmi),
                "score": float(score),
            }
        )

    raw_candidates.sort(key=lambda x: (x["score"], x["co_doc_freq"]), reverse=True)
    final_top_k = [x["term"] for x in raw_candidates[:top_k]]

    return {
        "target": target_norm,
        "n_captions_total": num_docs,
        "n_captions_target_match": target_docs,
        "raw_candidates": raw_candidates,
        "final_top_k": final_top_k,
        "scoring_method": "freq+pmi",
        "exclude_terms_containing_target_lexical": exclude_terms_containing_target_lexical,
    }

