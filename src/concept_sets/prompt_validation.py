from __future__ import annotations

from .text_utils import contains_any_phrase, normalize_phrase


def _normalize_prompt(prompt: str) -> str:
    return normalize_phrase(prompt)


def validate_prompt(
    prompt: str,
    family: str,
    target: str,
    concept_label: str | None,
    forbidden_lexical_items: set[str],
    allow_text_heavy_auxiliary: bool,
    text_heavy_cues: set[str],
    min_prompt_len: int | None = None,
    max_prompt_len: int | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate prompt according to prompt family and lexical constraints.
    """
    errors: list[str] = []
    prompt_norm = _normalize_prompt(prompt)

    if not prompt_norm:
        errors.append("empty_prompt")
        return False, errors

    if min_prompt_len is not None and len(prompt.strip()) < min_prompt_len:
        errors.append("too_short")
    if max_prompt_len is not None and len(prompt.strip()) > max_prompt_len:
        errors.append("too_long")

    if not allow_text_heavy_auxiliary and contains_any_phrase(prompt_norm, text_heavy_cues):
        errors.append("text_heavy_confounds")

    target_phrase = normalize_phrase(target)
    target_present = contains_any_phrase(prompt_norm, {target_phrase})
    forbidden_present = contains_any_phrase(prompt_norm, forbidden_lexical_items)

    if family == "direct":
        if not target_present:
            errors.append("target_missing_in_direct")
    elif family in {"indirect", "neighbor", "control"}:
        if forbidden_present:
            errors.append("forbidden_target_lexical_present")
    else:
        errors.append("unknown_family")

    if family in {"neighbor", "control"} and concept_label:
        concept_phrase = normalize_phrase(concept_label)
        if concept_phrase and not contains_any_phrase(prompt_norm, {concept_phrase}):
            errors.append("concept_label_missing")

    return len(errors) == 0, errors

