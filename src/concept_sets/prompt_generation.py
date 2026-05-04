from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from src.benchmark.prompt_loader import stable_prompt_id_from_fields

from .io_utils import ensure_dir, write_json
from .prompt_templates import CONCEPT_PROMPT_TEMPLATES, INDIRECT_PROMPT_TEMPLATES, TEXT_HEAVY_CUES
from .prompt_validation import validate_prompt
from .text_utils import normalize_phrase, normalize_text


@dataclass
class PromptGenerationConfig:
    direct_per_target: int = 50
    indirect_per_target: int = 50
    neighbor_per_concept: int = 30
    control_per_concept: int = 30
    lexical_mode: str = "strict"  # strict | broad
    generate_both_indirect_modes: bool = True
    random_seed: int = 42
    min_prompt_len: int = 10
    max_prompt_len: int = 220
    allow_text_heavy_auxiliary: bool = False
    use_llm_indirect: bool = False
    near_duplicate_jaccard_threshold: float = 0.90
    max_attempt_multiplier: int = 40


def _clean_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def _token_set(prompt: str) -> set[str]:
    return set(t for t in normalize_text(prompt).split(" ") if t)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _dedup_prompts(
    records: list[dict],
    near_duplicate_threshold: float,
) -> tuple[list[dict], int, int]:
    exact_seen = set()
    kept: list[dict] = []
    exact_removed = 0
    near_removed = 0

    for rec in records:
        p = _clean_prompt(rec["prompt"])
        if not p:
            exact_removed += 1
            continue
        p_norm = normalize_phrase(p)
        if p_norm in exact_seen:
            exact_removed += 1
            continue

        candidate_tokens = _token_set(p)
        near_dup = False
        for kept_rec in kept:
            s = _jaccard(candidate_tokens, _token_set(kept_rec["prompt"]))
            if s >= near_duplicate_threshold:
                near_dup = True
                break
        if near_dup:
            near_removed += 1
            continue

        exact_seen.add(p_norm)
        rec["prompt"] = p
        kept.append(rec)
    return kept, exact_removed, near_removed


def _forbidden_lexical_set(concept_obj: dict, lexical_mode: str) -> set[str]:
    key = "broad" if lexical_mode == "broad" else "strict"
    return {normalize_phrase(x) for x in concept_obj["lexical_set"].get(key, []) if normalize_phrase(x)}


def _make_record(
    target: str,
    family: str,
    concept_label: str | None,
    prompt: str,
    source_context_words: list[str] | None,
    lexical_mode: str | None,
) -> dict:
    return {
        "target": target,
        "prompt_family": family,
        "concept_label": concept_label,
        "prompt": prompt,
        "source_context_words": source_context_words or [],
        "lexical_mode": lexical_mode,
        "is_valid": False,
        "validation_errors": [],
    }


def _generate_concept_family_prompts(
    rng: random.Random,
    target: str,
    family: str,
    concepts: list[str],
    prompts_per_concept: int,
) -> list[dict]:
    out = []
    for concept in concepts:
        if not concept:
            continue
        concept_norm = normalize_phrase(concept)
        for _ in range(prompts_per_concept):
            template = rng.choice(CONCEPT_PROMPT_TEMPLATES)
            out.append(
                _make_record(
                    target=target,
                    family=family,
                    concept_label=concept_norm,
                    prompt=template.format(concept=concept_norm),
                    source_context_words=None,
                    lexical_mode=None,
                )
            )
    return out


def _generate_indirect_prompt_template_mode(
    rng: random.Random,
    target: str,
    context_words: list[str],
    lexical_mode: str,
    n_prompts: int,
) -> list[dict]:
    out = []
    usable = [normalize_phrase(w) for w in context_words if normalize_phrase(w)]
    if not usable:
        return out

    max_attempts = max(50, n_prompts * 20)
    attempts = 0
    while len(out) < n_prompts and attempts < max_attempts:
        attempts += 1
        template = rng.choice(INDIRECT_PROMPT_TEMPLATES)
        if len(usable) == 1:
            w1 = usable[0]
            w2 = usable[0]
            w3 = usable[0]
            src = [w1]
        elif len(usable) == 2:
            w1, w2 = rng.sample(usable, 2)
            w3 = w2
            src = [w1, w2]
        else:
            w1, w2, w3 = rng.sample(usable, 3)
            src = [w1, w2, w3]

        prompt = template.format(w1=w1, w2=w2, w3=w3)
        out.append(
            _make_record(
                target=target,
                family="indirect",
                concept_label=target,
                prompt=prompt,
                source_context_words=src,
                lexical_mode=lexical_mode,
            )
        )
    return out


def _generate_indirect_prompt_llm_mode(
    rng: random.Random,
    target: str,
    context_words: list[str],
    lexical_mode: str,
    n_prompts: int,
) -> list[dict]:
    """
    Placeholder for future LLM integration.
    Currently falls back to template-mode generation to keep pipeline runnable.
    """
    return _generate_indirect_prompt_template_mode(
        rng=rng,
        target=target,
        context_words=context_words,
        lexical_mode=lexical_mode,
        n_prompts=n_prompts,
    )


def _validate_records(
    records: list[dict],
    concept_obj: dict,
    lexical_mode_for_indirect: str,
    cfg: PromptGenerationConfig,
) -> tuple[list[dict], int]:
    forbidden = _forbidden_lexical_set(concept_obj, lexical_mode_for_indirect)
    target = concept_obj["target"]
    valid_records = []
    invalid_count = 0
    for rec in records:
        family = rec["prompt_family"]
        concept_label = rec["concept_label"]
        mode = rec["lexical_mode"] if rec["lexical_mode"] else lexical_mode_for_indirect
        valid, errors = validate_prompt(
            prompt=rec["prompt"],
            family=family,
            target=target,
            concept_label=concept_label,
            forbidden_lexical_items=forbidden,
            allow_text_heavy_auxiliary=cfg.allow_text_heavy_auxiliary,
            text_heavy_cues=TEXT_HEAVY_CUES,
            min_prompt_len=cfg.min_prompt_len,
            max_prompt_len=cfg.max_prompt_len,
        )
        rec["lexical_mode"] = mode
        rec["is_valid"] = valid
        rec["validation_errors"] = errors
        if valid:
            valid_records.append(rec)
        else:
            invalid_count += 1
    return valid_records, invalid_count


def _save_prompt_json(path: Path, records: list[dict]) -> None:
    write_json(path, {"count": len(records), "prompts": records})


def _save_prompt_csv(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "prompt_id",
        "target",
        "prompt_family",
        "concept_label",
        "prompt",
        "source_context_words",
        "lexical_mode",
        "is_valid",
        "validation_errors",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["prompt_id"] = stable_prompt_id_from_fields(
                row["target"],
                row["prompt_family"],
                row["concept_label"],
                row["prompt"],
                row.get("lexical_mode") or "strict",
            )
            row["source_context_words"] = "|".join(rec.get("source_context_words", []))
            row["validation_errors"] = "|".join(rec.get("validation_errors", []))
            writer.writerow(row)


def _load_concept_set_files(concept_sets_dir: str | Path) -> list[Path]:
    p = Path(concept_sets_dir)
    if p.is_file() and p.suffix.lower() == ".json":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if files:
            return files
    raise FileNotFoundError(f"Could not find concept-set JSON files in: {concept_sets_dir}")


def _read_json(path: Path) -> dict:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompts_from_concept_sets(
    concept_sets_dir: str | Path,
    output_dir: str | Path,
    cfg: PromptGenerationConfig,
    preview_count: int = 0,
) -> dict:
    out_dir = ensure_dir(output_dir)
    concept_files = _load_concept_set_files(concept_sets_dir)
    rng = random.Random(cfg.random_seed)

    global_summary = {
        "config": cfg.__dict__,
        "targets": {},
    }

    for concept_fp in concept_files:
        obj = _read_json(concept_fp)
        target = normalize_phrase(obj["target"])
        target_dir = ensure_dir(out_dir / target)

        direct_raw = _generate_concept_family_prompts(
            rng=rng,
            target=target,
            family="direct",
            concepts=[target],
            prompts_per_concept=cfg.direct_per_target,
        )

        neighbor_raw = _generate_concept_family_prompts(
            rng=rng,
            target=target,
            family="neighbor",
            concepts=[normalize_phrase(c) for c in obj["semantic_neighbors"].get("final_top_k", [])],
            prompts_per_concept=cfg.neighbor_per_concept,
        )

        control_raw = _generate_concept_family_prompts(
            rng=rng,
            target=target,
            family="control",
            concepts=[normalize_phrase(c) for c in obj["non_neighbor_controls"].get("final_top_k", [])],
            prompts_per_concept=cfg.control_per_concept,
        )

        indirect_modes = ["strict", "broad"] if cfg.generate_both_indirect_modes else [cfg.lexical_mode]
        indirect_by_mode: dict[str, list[dict]] = {}
        context_words = obj.get("context_words", {}).get("final_top_k", [])
        for mode in indirect_modes:
            if cfg.use_llm_indirect:
                indirect_raw = _generate_indirect_prompt_llm_mode(
                    rng=rng,
                    target=target,
                    context_words=context_words,
                    lexical_mode=mode,
                    n_prompts=cfg.indirect_per_target,
                )
            else:
                indirect_raw = _generate_indirect_prompt_template_mode(
                    rng=rng,
                    target=target,
                    context_words=context_words,
                    lexical_mode=mode,
                    n_prompts=cfg.indirect_per_target,
                )
            indirect_by_mode[mode] = indirect_raw

        # Validate and deduplicate.
        direct_valid, direct_invalid = _validate_records(
            records=direct_raw,
            concept_obj=obj,
            lexical_mode_for_indirect=cfg.lexical_mode,
            cfg=cfg,
        )
        neighbor_valid, neighbor_invalid = _validate_records(
            records=neighbor_raw,
            concept_obj=obj,
            lexical_mode_for_indirect=cfg.lexical_mode,
            cfg=cfg,
        )
        control_valid, control_invalid = _validate_records(
            records=control_raw,
            concept_obj=obj,
            lexical_mode_for_indirect=cfg.lexical_mode,
            cfg=cfg,
        )

        indirect_valid_by_mode = {}
        indirect_invalid_by_mode = {}
        for mode, records in indirect_by_mode.items():
            v, inv = _validate_records(
                records=records,
                concept_obj=obj,
                lexical_mode_for_indirect=mode,
                cfg=cfg,
            )
            indirect_valid_by_mode[mode] = v
            indirect_invalid_by_mode[mode] = inv

        direct_kept, d_exact, d_near = _dedup_prompts(direct_valid, cfg.near_duplicate_jaccard_threshold)
        neighbor_kept, n_exact, n_near = _dedup_prompts(neighbor_valid, cfg.near_duplicate_jaccard_threshold)
        control_kept, c_exact, c_near = _dedup_prompts(control_valid, cfg.near_duplicate_jaccard_threshold)
        indirect_kept_by_mode = {}
        indirect_dedup_stats = {}
        for mode, records in indirect_valid_by_mode.items():
            kept, x_removed, near_removed = _dedup_prompts(records, cfg.near_duplicate_jaccard_threshold)
            indirect_kept_by_mode[mode] = kept
            indirect_dedup_stats[mode] = {
                "exact_removed": x_removed,
                "near_removed": near_removed,
            }

        # Save per-family files.
        _save_prompt_json(target_dir / "direct_target_prompts.json", direct_kept)
        _save_prompt_json(target_dir / "neighbor_prompts.json", neighbor_kept)
        _save_prompt_json(target_dir / "control_prompts.json", control_kept)
        for mode, records in indirect_kept_by_mode.items():
            _save_prompt_json(target_dir / f"indirect_{mode}_prompts.json", records)

        all_records = []
        all_records.extend(direct_kept)
        all_records.extend(neighbor_kept)
        all_records.extend(control_kept)
        for records in indirect_kept_by_mode.values():
            all_records.extend(records)
        _save_prompt_csv(target_dir / "all_prompts.csv", all_records)

        target_summary = {
            "direct_count": len(direct_kept),
            "neighbor_count": len(neighbor_kept),
            "control_count": len(control_kept),
            "indirect_counts": {k: len(v) for k, v in indirect_kept_by_mode.items()},
            "invalid_counts": {
                "direct": direct_invalid,
                "neighbor": neighbor_invalid,
                "control": control_invalid,
                "indirect": indirect_invalid_by_mode,
            },
            "dedup_removed": {
                "direct": {"exact": d_exact, "near": d_near},
                "neighbor": {"exact": n_exact, "near": n_near},
                "control": {"exact": c_exact, "near": c_near},
                "indirect": indirect_dedup_stats,
            },
            "output_dir": str(target_dir),
        }
        global_summary["targets"][target] = target_summary

        if preview_count > 0:
            print(f"\n[target={target}]")
            for fam, rows in [
                ("direct", direct_kept),
                ("neighbor", neighbor_kept),
                ("control", control_kept),
            ]:
                print(f"  {fam} sample:")
                for rec in rows[:preview_count]:
                    print(f"    - {rec['prompt']}")
            for mode, rows in indirect_kept_by_mode.items():
                print(f"  indirect ({mode}) sample:")
                for rec in rows[:preview_count]:
                    print(f"    - {rec['prompt']}")

    write_json(out_dir / "prompt_generation_summary.json", global_summary)
    return global_summary

