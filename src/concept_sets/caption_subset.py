from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Iterable

from .io_utils import ensure_dir, write_json
from .lexical import build_lexical_sets
from .text_utils import contains_any_phrase, normalize_text


@dataclass
class CaptionSubsetConfig:
    lexical_mode: str = "strict"  # strict | broad
    positive_per_target: int = 10_000
    background_size: int = 50_000
    random_seed: int = 42
    progress_every: int = 100_000
    max_rows: int | None = None
    min_caption_chars: int = 3
    exclude_positive_from_background: bool = True
    example_count: int = 5
    checkpoint_every_rows: int = 1_000


def _caption_dedup_key(text: str) -> str:
    """
    Dedup key for captions.
    Uses lowercase + normalized whitespace but keeps punctuation semantics.
    """
    return normalize_text(text)


def _all_targets_full(saved: dict[str, list[str]], positive_per_target: int) -> bool:
    return all(len(v) >= positive_per_target for v in saved.values())


def _reservoir_insert(
    rng: Random,
    sample: list[tuple[str, str]],
    sample_keys: set[str],
    seen_unique_count: int,
    item_key: str,
    item_text: str,
    sample_size: int,
) -> int:
    """
    Reservoir insertion for a stream of unique items.
    Returns updated seen_unique_count.
    """
    seen_unique_count += 1
    if sample_size <= 0:
        return seen_unique_count

    if len(sample) < sample_size:
        sample.append((item_key, item_text))
        sample_keys.add(item_key)
        return seen_unique_count

    j = rng.randint(0, seen_unique_count - 1)
    if j < sample_size:
        old_key, _ = sample[j]
        if old_key in sample_keys:
            sample_keys.remove(old_key)
        sample[j] = (item_key, item_text)
        sample_keys.add(item_key)
    return seen_unique_count


def build_caption_subsets(
    targets: list[str],
    caption_iter: Iterable[str],
    output_dir: str | Path,
    cfg: CaptionSubsetConfig,
    synonyms_map: dict[str, list[str]] | None = None,
    progress_logger: Callable[[str], None] | None = None,
) -> dict:
    """
    Build per-target positive caption subsets and a shared background subset.
    """
    out_dir = ensure_dir(output_dir)
    rng = Random(cfg.random_seed)
    synonyms_map = synonyms_map or {}
    progress_logger = progress_logger or print

    lexical_by_target: dict[str, set[str]] = {}
    for t in targets:
        lex = build_lexical_sets(target=t, synonyms_map=synonyms_map, include_morph_variants=True)
        selected = lex.broad if cfg.lexical_mode == "broad" else lex.strict
        lexical_by_target[t] = set(selected)

    positives: dict[str, list[str]] = {t: [] for t in targets}
    positives_seen: dict[str, set[str]] = {t: set() for t in targets}
    target_duplicates_removed: dict[str, int] = {t: 0 for t in targets}
    target_match_rows: dict[str, int] = {t: 0 for t in targets}

    all_positive_keys: set[str] = set()

    background_sample: list[tuple[str, str]] = []
    background_sample_keys: set[str] = set()
    background_seen_keys: set[str] = set()
    background_seen_unique_count = 0
    background_duplicates_removed = 0
    background_excluded_positive = 0

    rows_scanned = 0
    empty_or_short_filtered = 0

    def _write_checkpoint() -> None:
        """Persist current partial outputs to reduce failure risk."""
        for target in targets:
            pos_path = out_dir / f"{target}_positive_captions.txt"
            with open(pos_path, "w", encoding="utf-8") as f:
                for cap in positives[target]:
                    f.write(cap + "\n")

            stat_obj = {
                "target": target,
                "lexical_mode": cfg.lexical_mode,
                "lexical_set_used": sorted(lexical_by_target[target]),
                "positive_output_path": str(pos_path),
                "matched_captions_saved": len(positives[target]),
                "source_rows_scanned": rows_scanned,
                "target_match_rows_before_dedup": target_match_rows[target],
                "duplicates_removed": target_duplicates_removed[target],
                "empty_or_short_filtered": empty_or_short_filtered,
                "example_captions": positives[target][: cfg.example_count],
                "filters": {
                    "min_caption_chars": cfg.min_caption_chars,
                    "word_boundary_aware_matching": True,
                    "dedup_key": "lowercase + whitespace normalized",
                },
                "is_final": False,
            }
            write_json(out_dir / f"{target}_stats.json", stat_obj)

        background_path = out_dir / "background_captions.txt"
        with open(background_path, "w", encoding="utf-8") as f:
            for _, cap in background_sample:
                f.write(cap + "\n")

        interim_summary_rows = []
        for target in targets:
            interim_summary_rows.append(
                {
                    "target": target,
                    "lexical_mode": cfg.lexical_mode,
                    "matched_saved": len(positives[target]),
                    "target_match_rows_before_dedup": target_match_rows[target],
                    "duplicates_removed": target_duplicates_removed[target],
                    "source_rows_scanned": rows_scanned,
                }
            )
        write_json(
            out_dir / "caption_subset_summary.json",
            {
                "per_target": interim_summary_rows,
                "global": {
                    "targets": targets,
                    "source_rows_scanned": rows_scanned,
                    "empty_or_short_filtered": empty_or_short_filtered,
                    "background_output_path": str(background_path),
                    "background_size_requested": cfg.background_size,
                    "background_size_saved": len(background_sample),
                    "background_duplicates_removed": background_duplicates_removed,
                    "background_excluded_positive": background_excluded_positive,
                    "exclude_positive_from_background": cfg.exclude_positive_from_background,
                    "random_seed": cfg.random_seed,
                    "stopped_early": False,
                    "is_final": False,
                },
            },
        )

    for raw_caption in caption_iter:
        rows_scanned += 1
        if cfg.max_rows is not None and rows_scanned > cfg.max_rows:
            break

        caption = str(raw_caption).strip()
        if len(caption) < cfg.min_caption_chars:
            empty_or_short_filtered += 1
            continue
        key = _caption_dedup_key(caption)
        if not key:
            empty_or_short_filtered += 1
            continue

        # Positive subset extraction for all targets in one pass.
        for target in targets:
            if len(positives[target]) >= cfg.positive_per_target:
                continue
            if contains_any_phrase(caption, lexical_by_target[target]):
                target_match_rows[target] += 1
                if key in positives_seen[target]:
                    target_duplicates_removed[target] += 1
                    continue
                positives_seen[target].add(key)
                positives[target].append(caption)
                all_positive_keys.add(key)

        # Background reservoir sampling from deduplicated stream.
        if cfg.background_size > 0:
            if cfg.exclude_positive_from_background and key in all_positive_keys:
                background_excluded_positive += 1
            else:
                if key in background_seen_keys:
                    background_duplicates_removed += 1
                else:
                    background_seen_keys.add(key)
                    background_seen_unique_count = _reservoir_insert(
                        rng=rng,
                        sample=background_sample,
                        sample_keys=background_sample_keys,
                        seen_unique_count=background_seen_unique_count,
                        item_key=key,
                        item_text=caption,
                        sample_size=cfg.background_size,
                    )

        if cfg.progress_every > 0 and rows_scanned % cfg.progress_every == 0:
            progress_logger(
                f"[caption_subsets] scanned={rows_scanned:,} "
                f"bg={len(background_sample):,}/{cfg.background_size:,} "
                + " ".join(
                    f"{t}={len(positives[t]):,}/{cfg.positive_per_target:,}"
                    for t in targets
                )
            )

        if cfg.checkpoint_every_rows > 0 and rows_scanned % cfg.checkpoint_every_rows == 0:
            _write_checkpoint()

        done_pos = _all_targets_full(positives, cfg.positive_per_target)
        done_bg = len(background_sample) >= cfg.background_size
        if done_pos and done_bg:
            break

    # Write positive files and per-target stats.
    summary_rows = []
    for target in targets:
        pos_path = out_dir / f"{target}_positive_captions.txt"
        with open(pos_path, "w", encoding="utf-8") as f:
            for cap in positives[target]:
                f.write(cap + "\n")

        stat_obj = {
            "target": target,
            "lexical_mode": cfg.lexical_mode,
            "lexical_set_used": sorted(lexical_by_target[target]),
            "positive_output_path": str(pos_path),
            "matched_captions_saved": len(positives[target]),
            "source_rows_scanned": rows_scanned,
            "target_match_rows_before_dedup": target_match_rows[target],
            "duplicates_removed": target_duplicates_removed[target],
            "empty_or_short_filtered": empty_or_short_filtered,
            "example_captions": positives[target][: cfg.example_count],
            "filters": {
                "min_caption_chars": cfg.min_caption_chars,
                "word_boundary_aware_matching": True,
                "dedup_key": "lowercase + whitespace normalized",
            },
            "is_final": True,
        }
        write_json(out_dir / f"{target}_stats.json", stat_obj)

        summary_rows.append(
            {
                "target": target,
                "lexical_mode": cfg.lexical_mode,
                "matched_saved": len(positives[target]),
                "target_match_rows_before_dedup": target_match_rows[target],
                "duplicates_removed": target_duplicates_removed[target],
                "source_rows_scanned": rows_scanned,
            }
        )

    # Write background file and global stats.
    background_path = out_dir / "background_captions.txt"
    with open(background_path, "w", encoding="utf-8") as f:
        for _, cap in background_sample:
            f.write(cap + "\n")

    global_stats = {
        "targets": targets,
        "source_rows_scanned": rows_scanned,
        "empty_or_short_filtered": empty_or_short_filtered,
        "background_output_path": str(background_path),
        "background_size_requested": cfg.background_size,
        "background_size_saved": len(background_sample),
        "background_duplicates_removed": background_duplicates_removed,
        "background_excluded_positive": background_excluded_positive,
        "exclude_positive_from_background": cfg.exclude_positive_from_background,
        "random_seed": cfg.random_seed,
        "stopped_early": _all_targets_full(positives, cfg.positive_per_target)
        and len(background_sample) >= cfg.background_size,
        "is_final": True,
    }
    write_json(out_dir / "caption_subset_summary.json", {"per_target": summary_rows, "global": global_stats})

    summary_csv_path = out_dir / "caption_subset_summary.csv"
    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "target",
            "lexical_mode",
            "matched_saved",
            "target_match_rows_before_dedup",
            "duplicates_removed",
            "source_rows_scanned",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "positive_paths": {t: str(out_dir / f"{t}_positive_captions.txt") for t in targets},
        "background_path": str(background_path),
        "summary_json": str(out_dir / "caption_subset_summary.json"),
        "summary_csv": str(summary_csv_path),
    }

