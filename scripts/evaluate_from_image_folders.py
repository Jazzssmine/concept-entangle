from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.evaluators import (  # noqa: E402
    CLIPConceptEvaluator,
    CLIPConceptEvaluatorConfig,
    load_clip_templates,
    load_concept_map,
    load_concept_vocabulary,
)
from src.benchmark.io_utils import ensure_dir, load_json, save_csv, save_json  # noqa: E402
from src.benchmark.metrics import aggregate_metrics, compute_per_target_metrics, save_metrics_outputs  # noqa: E402
from src.benchmark.model_registry import load_model_registry  # noqa: E402
from src.benchmark.prediction import run_evaluator_predictions  # noqa: E402
from src.benchmark.prompt_loader import load_prompts  # noqa: E402


FILENAME_RE = re.compile(r"^(?P<prompt_id>.+)__s(?P<seed>\d+)__i(?P<image_index>\d+)__([0-9a-f]{10})\.[A-Za-z0-9]+$")
INDEXED_NAME_RE = re.compile(r"^(?P<case_number>\d+)_(?P<image_index>\d+)\.[A-Za-z0-9]+$")
HEX12_RE = re.compile(r"_[0-9a-f]{12}$")
TRAILING_NUMERIC_ID_RE = re.compile(r"_[0-9]{6,}$")
VALID_FAMILIES = {"direct", "indirect", "neighbor", "control"}


def _resolve_repo_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _norm_text(value: object) -> str:
    return str(value).strip().lower()


def _infer_intended_from_prompt_id(prompt_id: str, target: str, family: str) -> str | None:
    prefix = f"{target}_{family}_"
    if not prompt_id.startswith(prefix):
        return None
    tail = prompt_id[len(prefix) :]
    tail = HEX12_RE.sub("", tail)
    # Some prompt_id variants append a long decimal identifier; strip it.
    tail = TRAILING_NUMERIC_ID_RE.sub("", tail)
    if not tail:
        return None
    return tail.replace("_", " ")


def _infer_target_from_prompt_id(prompt_id: str, family: str) -> str | None:
    marker = f"_{family}_"
    idx = prompt_id.find(marker)
    if idx <= 0:
        return None
    target = prompt_id[:idx].strip().lower()
    return target if target else None


def _build_prompt_lookup(prompts_path: str | None) -> dict[str, dict[str, str]]:
    if not prompts_path:
        return {}
    prompts_df = load_prompts(prompts_path)
    lookup: dict[str, dict[str, str]] = {}
    for _, row in prompts_df.iterrows():
        pid = str(row["prompt_id"])
        lookup[pid] = {
            "target_concept": _norm_text(row["target_concept"]),
            "prompt_family": _norm_text(row["prompt_family"]),
            "intended_label": _norm_text(row["intended_label"]),
            "domain": _norm_text(row["domain"]),
            "prompt": str(row["prompt"]),
            "lexical_mode": _norm_text(row["lexical_mode"]),
        }
    return lookup


def _build_prompt_text_lookup(prompts_path: str | None) -> dict[tuple[str, str, str], dict[str, str]]:
    if not prompts_path:
        return {}
    prompts_df = load_prompts(prompts_path)
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for _, row in prompts_df.iterrows():
        key = (
            _norm_text(row["target_concept"]),
            _norm_text(row["prompt_family"]),
            str(row["prompt"]).strip(),
        )
        lookup[key] = {
            "intended_label": _norm_text(row["intended_label"]),
            "domain": _norm_text(row["domain"]),
            "lexical_mode": _norm_text(row["lexical_mode"]),
            "prompt_id": str(row["prompt_id"]),
        }
    return lookup


def _infer_single_target_from_prompts(prompts_path: str | None) -> str | None:
    if not prompts_path:
        return None
    prompts_df = load_prompts(prompts_path)
    if "target_concept" not in prompts_df.columns or len(prompts_df) == 0:
        return None
    targets = sorted(set(prompts_df["target_concept"].astype(str).str.strip().str.lower().tolist()))
    if len(targets) == 1:
        return targets[0]
    return None


def _load_indexed_family_prompts(indexed_prompts_root: str | None, target: str, family: str) -> pd.DataFrame | None:
    if not indexed_prompts_root:
        return None
    root = Path(indexed_prompts_root)
    candidates = [
        root / target / "for_generate_example" / f"{family}.csv",
        root / "for_generate_example" / f"{family}.csv",
        root / f"{family}.csv",
    ]
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "case_number" in df.columns and "prompt" in df.columns:
            return df
    return None


def collect_images_metadata(
    images_root: str | Path,
    prompts_path: str | None = None,
    indexed_prompts_root: str | None = None,
    default_target_concept: str | None = None,
) -> pd.DataFrame:
    root = Path(images_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Images root not found: {root}")

    prompt_lookup = _build_prompt_lookup(prompts_path)
    prompt_text_lookup = _build_prompt_text_lookup(prompts_path)
    inferred_target = _infer_single_target_from_prompts(prompts_path)
    fallback_target = _norm_text(default_target_concept) if default_target_concept else inferred_target
    family_prompt_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    rows: list[dict[str, object]] = []
    parse_stats = {
        "total_images": 0,
        "with_prompt_id": 0,
        "matched_prompt_lookup": 0,
        "fallback_intended_used": 0,
        "missing_intended_for_non_direct": 0,
    }

    for img_path in sorted(root.rglob("*")):
        if not img_path.is_file():
            continue
        rel_parts = img_path.relative_to(root).parts
        if len(rel_parts) < 3:
            continue
        parse_stats["total_images"] += 1

        # Supported layouts:
        # 1) <images_root>/<model>/<target>/<family>/<file>
        # 2) <images_root>/<model>/<family>/<file>
        model_name = _norm_text(rel_parts[0])
        target_concept = "unknown"
        prompt_family = "unknown"

        if len(rel_parts) >= 4 and _norm_text(rel_parts[2]) in VALID_FAMILIES:
            target_concept = _norm_text(rel_parts[1])
            prompt_family = _norm_text(rel_parts[2])
        elif len(rel_parts) >= 3 and _norm_text(rel_parts[1]) in VALID_FAMILIES:
            prompt_family = _norm_text(rel_parts[1])
        if prompt_family not in VALID_FAMILIES:
            continue
        if target_concept == "unknown" and fallback_target:
            target_concept = fallback_target

        prompt_id = f"{target_concept}_{prompt_family}_{img_path.stem}"
        seed = 0
        image_index = 0
        prompt = ""
        lexical_mode = "strict"
        domain = "object"
        intended_label: str | None = target_concept if prompt_family == "direct" else None

        m = FILENAME_RE.match(img_path.name)
        if m:
            parse_stats["with_prompt_id"] += 1
            prompt_id = m.group("prompt_id")
            seed = int(m.group("seed"))
            image_index = int(m.group("image_index"))
            if target_concept == "unknown":
                inferred_target = _infer_target_from_prompt_id(prompt_id, prompt_family)
                if inferred_target:
                    target_concept = inferred_target

            if prompt_id in prompt_lookup:
                rec = prompt_lookup[prompt_id]
                target_concept = rec["target_concept"]
                prompt_family = rec["prompt_family"]
                intended_label = rec["intended_label"]
                domain = rec["domain"]
                prompt = rec["prompt"]
                lexical_mode = rec["lexical_mode"]
                parse_stats["matched_prompt_lookup"] += 1
            else:
                inferred = _infer_intended_from_prompt_id(prompt_id, target_concept, prompt_family)
                if inferred:
                    intended_label = _norm_text(inferred)
                    parse_stats["fallback_intended_used"] += 1
        else:
            # Support indexed naming format like "0_0.png" used by generate-example-img.py.
            m2 = INDEXED_NAME_RE.match(img_path.name)
            if m2:
                case_number = int(m2.group("case_number"))
                image_index = int(m2.group("image_index"))
                prompt_id = f"{target_concept}_{prompt_family}_case_{case_number}"
                cache_key = (target_concept, prompt_family)
                if cache_key not in family_prompt_cache:
                    family_prompt_cache[cache_key] = _load_indexed_family_prompts(
                        indexed_prompts_root=indexed_prompts_root,
                        target=target_concept,
                        family=prompt_family,
                    )
                family_df = family_prompt_cache.get(cache_key)
                if family_df is not None and len(family_df) > case_number:
                    prompt_row = family_df.iloc[case_number]
                    prompt = str(prompt_row["prompt"]).strip()
                    joined = prompt_text_lookup.get((target_concept, prompt_family, prompt))
                    if joined:
                        intended_label = joined["intended_label"]
                        domain = joined["domain"]
                        lexical_mode = joined["lexical_mode"]
                        prompt_id = joined["prompt_id"]
                        parse_stats["matched_prompt_lookup"] += 1

        if intended_label is None:
            parse_stats["missing_intended_for_non_direct"] += 1
            intended_label = "unknown"

        rows.append(
            {
                "image_path": str(img_path),
                "model_name": model_name,
                "target_concept": target_concept,
                "prompt_family": prompt_family,
                "intended_label": _norm_text(intended_label),
                "domain": domain,
                "prompt": prompt,
                "prompt_id": prompt_id,
                "seed": seed,
                "image_index": image_index,
                "guidance_scale": float("nan"),
                "num_inference_steps": float("nan"),
                "lexical_mode": lexical_mode,
                "status": "from_image_folder",
            }
        )

    if not rows:
        raise ValueError(f"No images found under: {root}")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return df, parse_stats


def _save_per_target_slices(
    output_dir: Path,
    per_image_df: pd.DataFrame,
    per_target_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    debug_df: pd.DataFrame,
) -> None:
    if len(per_image_df) == 0 or "target_concept" not in per_image_df.columns:
        return
    for target in sorted(per_image_df["target_concept"].dropna().astype(str).unique().tolist()):
        target_slug = str(target).strip().lower().replace(" ", "_")
        save_csv(
            output_dir / f"{target_slug}_per_image_clip_predictions.csv",
            per_image_df[per_image_df["target_concept"] == target].copy(),
        )
        if len(per_target_df) > 0 and "target_concept" in per_target_df.columns:
            save_csv(
                output_dir / f"{target_slug}_clip_per_target_metrics.csv",
                per_target_df[per_target_df["target_concept"] == target].copy(),
                na_rep="NA",
            )
        if len(detailed_df) > 0 and "target_concept" in detailed_df.columns:
            save_csv(
                output_dir / f"{target_slug}_clip_detailed_breakdowns.csv",
                detailed_df[detailed_df["target_concept"] == target].copy(),
                na_rep="NA",
            )
        if len(debug_df) > 0 and "target_concept" in debug_df.columns:
            save_csv(
                output_dir / f"{target_slug}_clip_debug_metric_inputs.csv",
                debug_df[debug_df["target_concept"] == target].copy(),
                na_rep="NA",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing benchmark image folders with CLIP (no generation required)."
    )
    parser.add_argument("--images_root", type=str, required=True, help="Root like outputs/benchmark/images")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write eval CSVs")
    parser.add_argument(
        "--model_registry_path",
        type=str,
        default=None,
        help="Optional model registry JSON; used to validate --model_key.",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help=(
            "Optional model key to evaluate (e.g. stereo_horse). "
            "When set, the evaluator still keeps --base_model_name rows (if available) for normalization."
        ),
    )
    parser.add_argument("--concept_vocab_path", type=str, required=True)
    parser.add_argument(
        "--target_synonyms_path",
        type=str,
        default="data/target_synonyms.json",
        help="Optional JSON map of target label synonyms (e.g., pony -> horse).",
    )
    parser.add_argument("--base_model_name", type=str, default="base")
    parser.add_argument("--neighbor_map_path", type=str, default=None)
    parser.add_argument("--control_map_path", type=str, default=None)
    parser.add_argument("--prompts_path", type=str, default=None, help="Optional prompts file/dir for exact prompt_id metadata joins")
    parser.add_argument(
        "--target_concept",
        type=str,
        default=None,
        help=(
            "Optional fallback target concept for layouts like <model>/<family>/0_0.png. "
            "If omitted, inferred when prompts_path contains a single target."
        ),
    )
    parser.add_argument(
        "--indexed_prompts_root",
        type=str,
        default=None,
        help=(
            "Optional root for index-style filenames like 0_0.png. Supports "
            "<root>/<target>/for_generate_example/<family>.csv, "
            "<root>/for_generate_example/<family>.csv, or <root>/<family>.csv."
        ),
    )
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_templates_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_topk", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--skip_existing_predictions", action="store_true")
    parser.add_argument("--min_base_acc_for_normalization", type=float, default=0.05)
    parser.add_argument("--compute_cp", action="store_true")
    parser.add_argument(
        "--save_extra_outputs",
        action="store_true",
        help=(
            "Write full debug/duplicate artifacts (per-image predictions, metadata snapshots, "
            "clip_* duplicates, per-target slices, run summary JSON). Default is minimal metrics-only output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.images_root = _resolve_repo_path(args.images_root)
    args.output_dir = _resolve_repo_path(args.output_dir)
    args.model_registry_path = _resolve_repo_path(args.model_registry_path)
    args.concept_vocab_path = _resolve_repo_path(args.concept_vocab_path)
    args.target_synonyms_path = _resolve_repo_path(args.target_synonyms_path)
    args.neighbor_map_path = _resolve_repo_path(args.neighbor_map_path)
    args.control_map_path = _resolve_repo_path(args.control_map_path)
    args.prompts_path = _resolve_repo_path(args.prompts_path)
    if args.target_concept is not None:
        args.target_concept = _norm_text(args.target_concept)
    args.indexed_prompts_root = _resolve_repo_path(args.indexed_prompts_root)
    args.clip_templates_path = _resolve_repo_path(args.clip_templates_path)

    selected_model: str | None = None
    if args.model_registry_path:
        registry = load_model_registry(args.model_registry_path)
        if args.model_key is not None:
            if args.model_key not in registry:
                available = ", ".join(sorted(registry.keys()))
                raise ValueError(
                    f"Unknown --model_key '{args.model_key}'. Available keys: {available}"
                )
            selected_model = args.model_key
    elif args.model_key is not None:
        selected_model = args.model_key

    output_root = Path(args.output_dir)
    if selected_model is not None and output_root.name != selected_model:
        output_root = output_root / selected_model
    output_dir = ensure_dir(output_root)

    metadata_df, parse_stats = collect_images_metadata(
        images_root=args.images_root,
        prompts_path=args.prompts_path,
        indexed_prompts_root=args.indexed_prompts_root,
        default_target_concept=args.target_concept,
    )
    selected_model_norm = _norm_text(selected_model) if selected_model is not None else None
    base_model_norm = _norm_text(args.base_model_name)
    discovered_models = sorted(
        set(metadata_df["model_name"].astype(str).str.strip().str.lower().tolist())
    )
    models_used_for_eval: list[str] = discovered_models

    if selected_model_norm is not None:
        keep_models = {selected_model_norm}
        if base_model_norm != selected_model_norm:
            # Keep base model rows for normalization (NP/CP/DamageGap), even when evaluating one model key.
            keep_models.add(base_model_norm)
        metadata_df = metadata_df[
            metadata_df["model_name"].astype(str).str.strip().str.lower().isin(keep_models)
        ].copy()
        models_used_for_eval = sorted(keep_models)
        if len(metadata_df) == 0:
            discovered = ", ".join(discovered_models)
            raise ValueError(
                f"No images found for model '{selected_model}' under {args.images_root}. "
                f"Discovered model folders: {discovered if discovered else '(none)'}"
            )
        if selected_model_norm not in discovered_models:
            discovered = ", ".join(discovered_models)
            raise ValueError(
                f"Model folder '{selected_model}' not found under {args.images_root}. "
                f"Discovered model folders: {discovered if discovered else '(none)'}"
            )
        if base_model_norm not in discovered_models:
            print(
                f"[warn] Base model '{args.base_model_name}' not found under images_root; "
                "NP/CP/DamageGap may be NA due to missing base rows."
            )

    if args.save_extra_outputs:
        save_csv(output_dir / "generated_images_from_folders.csv", metadata_df)

    concept_vocabulary = load_concept_vocabulary(args.concept_vocab_path)
    target_synonyms_map: dict[str, list[str]] = {}
    if args.target_synonyms_path:
        synonyms_path = Path(args.target_synonyms_path)
        if synonyms_path.exists():
            loaded = load_json(synonyms_path)
            if isinstance(loaded, dict):
                target_synonyms_map = {
                    str(k).strip().lower(): [str(v).strip().lower() for v in (vals or [])]
                    for k, vals in loaded.items()
                }
            else:
                raise ValueError(f"--target_synonyms_path must point to a JSON object map: {synonyms_path}")
        else:
            print(f"[warn] target synonyms file not found: {synonyms_path}; proceeding without synonym matching.")
    templates = load_clip_templates(args.clip_templates_path)
    neighbor_map = load_concept_map(args.neighbor_map_path, kind="neighbor")
    control_map = load_concept_map(args.control_map_path, kind="control")

    evaluator = CLIPConceptEvaluator(
        concept_vocabulary=concept_vocabulary,
        cfg=CLIPConceptEvaluatorConfig(
            batch_size=args.batch_size,
            device=args.device,
            top_k=args.top_k,
            save_topk=args.save_topk,
            clip_model_name=args.clip_model_name,
            clip_templates=templates,
            save_embeddings=args.save_embeddings,
        ),
    )

    per_image_path = output_dir / "per_image_clip_predictions.csv"
    pred_df, debug_payload = run_evaluator_predictions(
        generated_df=metadata_df,
        evaluator=evaluator,
        output_predictions_path=per_image_path,
        metadata_root=args.images_root,
        skip_existing_predictions=args.skip_existing_predictions,
        target_synonyms_map=target_synonyms_map,
    )
    per_target_df, detailed_df, debug_df = compute_per_target_metrics(
        pred_df=pred_df,
        base_model_name=args.base_model_name,
        compute_cp=args.compute_cp,
        neighbor_map=neighbor_map,
        control_map=control_map,
        min_base_acc_for_normalization=args.min_base_acc_for_normalization,
    )
    aggregated_df = aggregate_metrics(per_target_df)
    pred_df_for_output = pred_df
    if selected_model_norm is not None and len(pred_df_for_output) > 0 and "model_name" in pred_df_for_output.columns:
        pred_df_for_output = pred_df_for_output[
            pred_df_for_output["model_name"].astype(str).str.strip().str.lower() == selected_model_norm
        ].copy()
        save_csv(per_image_path, pred_df_for_output)
    if selected_model_norm is not None and len(per_target_df) > 0 and "model_name" in per_target_df.columns:
        per_target_df = per_target_df[
            per_target_df["model_name"].astype(str).str.strip().str.lower() == selected_model_norm
        ].copy()
    if selected_model_norm is not None and len(detailed_df) > 0 and "model_name" in detailed_df.columns:
        detailed_df = detailed_df[
            detailed_df["model_name"].astype(str).str.strip().str.lower() == selected_model_norm
        ].copy()
    if selected_model_norm is not None and len(debug_df) > 0 and "model_name" in debug_df.columns:
        debug_df = debug_df[
            debug_df["model_name"].astype(str).str.strip().str.lower() == selected_model_norm
        ].copy()

    # Minimal output set: aggregated + per_* metrics.
    save_csv(output_dir / "per_target_clip_metrics.csv", per_target_df, na_rep="NA")
    save_csv(output_dir / "aggregated_clip_metrics.csv", aggregated_df, na_rep="NA")
    if len(detailed_df) > 0 and "metric_component" in detailed_df.columns:
        save_csv(output_dir / "per_neighbor_metrics.csv", detailed_df[detailed_df["metric_component"] == "NP"].copy(), na_rep="NA")
        save_csv(output_dir / "per_control_metrics.csv", detailed_df[detailed_df["metric_component"] == "CP"].copy(), na_rep="NA")
    if args.save_extra_outputs:
        save_csv(output_dir / "clip_detailed_breakdowns.csv", detailed_df, na_rep="NA")
        save_csv(output_dir / "debug_metric_inputs.csv", debug_df, na_rep="NA")
        _save_per_target_slices(output_dir, pred_df_for_output, per_target_df, detailed_df, debug_df)
        save_metrics_outputs(
            per_target_df=per_target_df,
            aggregated_df=aggregated_df,
            detailed_df=detailed_df,
            eval_dir=str(output_dir),
            prefix="clip",
            debug_df=debug_df,
        )

        run_summary = {
            "images_root": args.images_root,
            "output_dir": str(output_dir),
            "selected_model": selected_model,
            "models_used_for_eval": models_used_for_eval,
            "n_images": int(len(metadata_df)),
            "n_predictions": int(len(pred_df)),
            "n_targets": int(per_target_df["target_concept"].nunique()) if len(per_target_df) else 0,
            "base_model_name": args.base_model_name,
            "parse_stats": parse_stats,
            "config": vars(args),
            "debug": {k: v for k, v in debug_payload.items() if k != "image_embeddings"},
        }
        save_json(output_dir / "clip_evaluation_from_folders_summary.json", run_summary)

        if args.save_embeddings and "image_embeddings" in debug_payload:
            torch.save(debug_payload["image_embeddings"], output_dir / "clip_image_embeddings.pt")
    else:
        # Keep output directory tidy in minimal mode.
        if per_image_path.exists():
            per_image_path.unlink()

    print("Folder-based CLIP evaluation completed.")
    print(f"- Per-target metrics: {output_dir / 'per_target_clip_metrics.csv'}")
    print(f"- Aggregated metrics: {output_dir / 'aggregated_clip_metrics.csv'}")
    if args.save_extra_outputs:
        print(f"- Metadata snapshot: {output_dir / 'generated_images_from_folders.csv'}")
        print(f"- Per-image predictions: {per_image_path}")


if __name__ == "__main__":
    main()
