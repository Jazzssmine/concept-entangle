from __future__ import annotations

from dataclasses import dataclass

from .text_utils import normalize_phrase, simple_morph_variants


@dataclass
class LexicalSets:
    strict: list[str]
    broad: list[str]


def build_lexical_sets(
    target: str,
    synonyms_map: dict[str, list[str]] | None = None,
    include_morph_variants: bool = True,
) -> LexicalSets:
    target_norm = normalize_phrase(target)
    strict = {target_norm}
    if include_morph_variants:
        strict.update(simple_morph_variants(target_norm))

    broad = set(strict)
    if synonyms_map is not None:
        for syn in synonyms_map.get(target_norm, []):
            syn_norm = normalize_phrase(syn)
            if syn_norm:
                broad.add(syn_norm)
                if include_morph_variants:
                    broad.update(simple_morph_variants(syn_norm))

    return LexicalSets(strict=sorted(strict), broad=sorted(broad))

