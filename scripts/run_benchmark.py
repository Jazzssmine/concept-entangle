from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.abspath("."))

from src.benchmark.fid_utils import compute_fid_scores
from src.benchmark.generation import GenerationConfig, generate_all_images
from src.benchmark.io_utils import ensure_dir, save_csv, save_json
from src.benchmark.metrics import aggregate_metrics, compute_per_target_metrics, save_metrics_outputs
from src.benchmark.model_registry import load_model_registry
from src.benchmark.evaluators import (
    CLIPConceptEvaluator,
    CLIPConceptEvaluatorConfig,
    load_clip_templates,
    load_concept_map,
    load_concept_vocabulary,
)
from src.benchmark.prediction import PredictionConfig, run_classifier_predictions, run_evaluator_predictions
from src.benchmark.prompt_loader import load_prompts


def _parse_seeds(values: list[str]) -> list[int]:
    seeds = []
    for v in values:
        if "," in v:
            for x in v.split(","):
                x = x.strip()
                if x:
                    seeds.append(int(x))
        else:
            seeds.append(int(v))
    return seeds


def _save_per_target_clip_slices(
    eval_dir: Path,
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
            eval_dir / f"{target_slug}_per_image_clip_predictions.csv",
            per_image_df[per_image_df["target_concept"] == target].copy(),
        )
        if len(per_target_df) > 0 and "target_concept" in per_target_df.columns:
            save_csv(
                eval_dir / f"{target_slug}_clip_per_target_metrics.csv",
                per_target_df[per_target_df["target_concept"] == target].copy(),
                na_rep="NA",
            )
        if len(detailed_df) > 0 and "target_concept" in detailed_df.columns:
            save_csv(
                eval_dir / f"{target_slug}_clip_detailed_breakdowns.csv",
                detailed_df[detailed_df["target_concept"] == target].copy(),
                na_rep="NA",
            )
        if len(debug_df) > 0 and "target_concept" in debug_df.columns:
            save_csv(
                eval_dir / f"{target_slug}_clip_debug_metric_inputs.csv",
                debug_df[debug_df["target_concept"] == target].copy(),
                na_rep="NA",
            )


def _save_per_model_eval_outputs(
    eval_dir: Path,
    pred_df: pd.DataFrame,
    per_target_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    debug_df: pd.DataFrame,
    evaluator_backend: str,
) -> None:
    if len(pred_df) == 0 or "model_name" not in pred_df.columns:
        return

    metric_prefix = "clip" if evaluator_backend == "clip" else ""
    per_image_name = (
        "per_image_clip_predictions.csv"
        if evaluator_backend == "clip"
        else "per_image_predictions.csv"
    )

    model_names = sorted(pred_df["model_name"].dropna().astype(str).unique().tolist())
    for model_name in model_names:
        model_eval_dir = ensure_dir(eval_dir / model_name)
        model_pred_df = pred_df[pred_df["model_name"] == model_name].copy()
        save_csv(model_eval_dir / per_image_name, model_pred_df)

        model_per_target_df = pd.DataFrame()
        if len(per_target_df) > 0 and "model_name" in per_target_df.columns:
            model_per_target_df = per_target_df[
                per_target_df["model_name"] == model_name
            ].copy()

        model_aggregated_df = pd.DataFrame()
        if len(aggregated_df) > 0 and "model_name" in aggregated_df.columns:
            model_aggregated_df = aggregated_df[
                aggregated_df["model_name"] == model_name
            ].copy()

        model_detailed_df = pd.DataFrame()
        if len(detailed_df) > 0 and "model_name" in detailed_df.columns:
            model_detailed_df = detailed_df[detailed_df["model_name"] == model_name].copy()

        model_debug_df = pd.DataFrame()
        if len(debug_df) > 0 and "model_name" in debug_df.columns:
            model_debug_df = debug_df[debug_df["model_name"] == model_name].copy()

        save_metrics_outputs(
            model_per_target_df,
            model_aggregated_df,
            model_detailed_df,
            str(model_eval_dir),
            prefix=metric_prefix,
            debug_df=model_debug_df,
        )

        if evaluator_backend == "clip":
            save_csv(model_eval_dir / "per_target_clip_metrics.csv", model_per_target_df, na_rep="NA")
            save_csv(model_eval_dir / "aggregated_clip_metrics.csv", model_aggregated_df, na_rep="NA")
            save_csv(model_eval_dir / "clip_detailed_breakdowns.csv", model_detailed_df, na_rep="NA")
            save_csv(model_eval_dir / "debug_metric_inputs.csv", model_debug_df, na_rep="NA")
            if len(model_detailed_df) > 0 and "metric_component" in model_detailed_df.columns:
                save_csv(
                    model_eval_dir / "per_neighbor_metrics.csv",
                    model_detailed_df[model_detailed_df["metric_component"] == "NP"].copy(),
                    na_rep="NA",
                )
                save_csv(
                    model_eval_dir / "per_control_metrics.csv",
                    model_detailed_df[model_detailed_df["metric_component"] == "CP"].copy(),
                    na_rep="NA",
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end diffusion unlearning benchmark pipeline.")
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--model_registry_path", type=str, default="configs/benchmark_models.example.json")
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="Optional model key in registry JSON (e.g. stereo_horse, esd_horse). If set, only that model is used.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--classifier_registry_path", type=str, default=None)
    parser.add_argument("--object_classifier_path", type=str, default=None)
    parser.add_argument("--style_classifier_path", type=str, default=None)
    parser.add_argument("--evaluator_backend", type=str, default="classifier", choices=["classifier", "clip"])
    parser.add_argument("--concept_vocab_path", type=str, default=None)
    parser.add_argument("--neighbor_map_path", type=str, default=None)
    parser.add_argument("--control_map_path", type=str, default=None)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_templates_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_topk", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--skip_existing_predictions", action="store_true")
    parser.add_argument("--min_base_acc_for_normalization", type=float, default=0.05)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument(
        "--saeuron_percentile",
        type=float,
        default=None,
        help="Optional override for SAeUron extra_args.percentile in model registry.",
    )
    parser.add_argument(
        "--saeuron_multiplier",
        type=float,
        default=None,
        help="Optional override for SAeUron extra_args.multiplier in model registry.",
    )
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--seeds", nargs="*", default=["0"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    parser.add_argument("--base_model_name", type=str, default="base")
    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--fid_reference_mode", type=str, choices=["base", "real"], default="base")
    parser.add_argument("--fid_reference_dir", type=str, default=None)
    parser.add_argument("--compute_cp", action="store_true")

    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument(
        "--skip_prediction",
        action="store_true",
        help="Do not run classifiers; load existing eval/per_image_predictions.csv (metrics/FID only)",
    )
    parser.add_argument(
        "--prediction_only",
        action="store_true",
        help="Skip image generation; load metadata/generated_images.csv, run classifiers + metrics",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Stop after image generation (writes generated_images.csv); no classifier or metrics",
    )
    parser.add_argument("--reuse_cached_predictions", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate all images even when outputs and metadata already exist (default: skip existing)",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _build_inline_classifier_registry(args: argparse.Namespace, output_dir: Path) -> str | None:
    if args.classifier_registry_path:
        return args.classifier_registry_path
    payload = {}
    if args.object_classifier_path:
        payload["object_classifier"] = {
            "domain": "object",
            "type": "hf_image_classification",
            "model_name_or_path": args.object_classifier_path,
        }
    if args.style_classifier_path:
        payload["style_classifier"] = {
            "domain": "style",
            "type": "hf_image_classification",
            "model_name_or_path": args.style_classifier_path,
        }
    if not payload:
        return None
    path = output_dir / "metadata" / "inline_classifier_registry.json"
    save_json(path, payload)
    return str(path)


def main() -> None:
    args = parse_args()
    if args.prediction_only and args.generation_only:
        raise ValueError("Use only one of --prediction_only or --generation_only")
    if args.prediction_only:
        args.skip_generation = True
        args.skip_prediction = False
    out_dir = ensure_dir(args.output_dir)
    metadata_dir = ensure_dir(out_dir / "metadata")
    eval_dir = ensure_dir(out_dir / "eval")
    seeds = _parse_seeds(args.seeds)

    prompts_df = load_prompts(args.prompts_path)
    model_specs = load_model_registry(args.model_registry_path)
    if args.model_key is not None:
        if args.model_key not in model_specs:
            available = ", ".join(sorted(model_specs.keys()))
            raise ValueError(
                f"Model key '{args.model_key}' not found in {args.model_registry_path}. "
                f"Available keys: {available}"
            )
        model_specs = {args.model_key: model_specs[args.model_key]}
    if args.saeuron_percentile is not None or args.saeuron_multiplier is not None:
        for spec in model_specs.values():
            if spec.type.lower() != "saeuron":
                continue
            if args.saeuron_percentile is not None:
                spec.extra_args["percentile"] = float(args.saeuron_percentile)
            if args.saeuron_multiplier is not None:
                spec.extra_args["multiplier"] = float(args.saeuron_multiplier)
    classifier_registry_path = _build_inline_classifier_registry(args, out_dir)
    neighbor_map = load_concept_map(args.neighbor_map_path, kind="neighbor") if args.neighbor_map_path else {}
    control_map = load_concept_map(args.control_map_path, kind="control") if args.control_map_path else {}

    save_json(
        metadata_dir / "benchmark_run_config.json",
        {
            "args": vars(args),
            "n_prompts": int(len(prompts_df)),
            "models": list(model_specs.keys()),
            "seeds": seeds,
        },
    )

    if args.dry_run:
        print("Dry run successful.")
        print(f"Prompts loaded: {len(prompts_df)}")
        print(f"Models loaded: {list(model_specs.keys())}")
        if classifier_registry_path:
            print(f"Classifier registry: {classifier_registry_path}")
        else:
            print("Classifier registry: none")
        return

    generated_metadata_path = metadata_dir / "generated_images.csv"
    per_image_pred_path = eval_dir / (
        "per_image_clip_predictions.csv" if args.evaluator_backend == "clip" else "per_image_predictions.csv"
    )

    if not args.skip_generation:
        gen_cfg = GenerationConfig(
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            images_per_prompt=args.images_per_prompt,
            seeds=seeds,
            batch_size=args.batch_size,
            device=args.device,
            resume=not args.no_resume,
            height=args.height,
            width=args.width,
        )
        generated_df = generate_all_images(
            prompts_df=prompts_df,
            model_specs=model_specs,
            output_dir=out_dir,
            cfg=gen_cfg,
            generated_metadata_path=generated_metadata_path,
        )
    else:
        if not generated_metadata_path.exists():
            raise FileNotFoundError(f"Missing generated metadata file: {generated_metadata_path}")
        generated_df = pd.read_csv(generated_metadata_path)

    if args.generation_only:
        print("Stopped after generation (--generation_only).")
        print(f"- Generated metadata: {generated_metadata_path}")
        print("Next: run with --prediction_only (and classifier registry) to classify and compute metrics.")
        return

    if not args.skip_prediction:
        if args.evaluator_backend == "clip":
            if not args.concept_vocab_path:
                raise ValueError("--concept_vocab_path is required when --evaluator_backend clip")
            clip_templates = load_clip_templates(args.clip_templates_path)
            clip_evaluator = CLIPConceptEvaluator(
                concept_vocabulary=load_concept_vocabulary(args.concept_vocab_path),
                cfg=CLIPConceptEvaluatorConfig(
                    batch_size=args.batch_size,
                    device=args.device,
                    top_k=args.top_k,
                    save_topk=args.save_topk,
                    clip_model_name=args.clip_model_name,
                    clip_templates=clip_templates,
                    save_embeddings=args.save_embeddings,
                ),
            )
            pred_df, debug_payload = run_evaluator_predictions(
                generated_df=generated_df,
                evaluator=clip_evaluator,
                output_predictions_path=per_image_pred_path,
                metadata_root=REPO_ROOT,
                skip_existing_predictions=args.skip_existing_predictions,
            )
            save_json(
                eval_dir / "clip_prediction_summary.json",
                {k: v for k, v in debug_payload.items() if k != "image_embeddings"},
            )
            if args.save_embeddings and "image_embeddings" in debug_payload:
                torch.save(debug_payload["image_embeddings"], eval_dir / "clip_image_embeddings.pt")
        else:
            pred_cfg = PredictionConfig(
                batch_size=args.batch_size,
                reuse_cached_predictions=args.reuse_cached_predictions,
            )
            pred_df = run_classifier_predictions(
                generated_df=generated_df,
                classifier_registry_path=classifier_registry_path,
                output_predictions_path=per_image_pred_path,
                cfg=pred_cfg,
            )
    else:
        if not per_image_pred_path.exists():
            raise FileNotFoundError(
                f"Missing predictions file: {per_image_pred_path}\n\n"
                "You used --skip_prediction, so the pipeline does not run the classifier and expects "
                "that CSV to already exist.\n\n"
                "To create it: run again without --skip_prediction and pass a classifier, e.g.\n"
                "  --classifier_registry_path configs/benchmark_classifiers.example.json\n"
                "or use --prediction_only after generation finished (loads generated_images.csv, runs classifiers).\n"
            )
        pred_df = pd.read_csv(per_image_pred_path)

    per_target_df, detailed_df, debug_df = compute_per_target_metrics(
        pred_df=pred_df,
        base_model_name=args.base_model_name,
        compute_cp=args.compute_cp,
        neighbor_map=neighbor_map,
        control_map=control_map,
        min_base_acc_for_normalization=args.min_base_acc_for_normalization,
    )
    aggregated_df = aggregate_metrics(per_target_df)
    metric_prefix = "clip" if args.evaluator_backend == "clip" else ""
    save_metrics_outputs(per_target_df, aggregated_df, detailed_df, str(eval_dir), prefix=metric_prefix, debug_df=debug_df)
    if args.evaluator_backend == "clip":
        save_csv(eval_dir / "per_target_clip_metrics.csv", per_target_df, na_rep="NA")
        save_csv(eval_dir / "aggregated_clip_metrics.csv", aggregated_df, na_rep="NA")
        save_csv(eval_dir / "clip_detailed_breakdowns.csv", detailed_df, na_rep="NA")
        save_csv(eval_dir / "debug_metric_inputs.csv", debug_df, na_rep="NA")
        if len(detailed_df) > 0 and "metric_component" in detailed_df.columns:
            save_csv(eval_dir / "per_neighbor_metrics.csv", detailed_df[detailed_df["metric_component"] == "NP"].copy(), na_rep="NA")
            save_csv(eval_dir / "per_control_metrics.csv", detailed_df[detailed_df["metric_component"] == "CP"].copy(), na_rep="NA")
        _save_per_target_clip_slices(eval_dir, pred_df, per_target_df, detailed_df, debug_df)

    _save_per_model_eval_outputs(
        eval_dir=eval_dir,
        pred_df=pred_df,
        per_target_df=per_target_df,
        aggregated_df=aggregated_df,
        detailed_df=detailed_df,
        debug_df=debug_df,
        evaluator_backend=args.evaluator_backend,
    )

    # Optional FID
    if args.compute_fid:
        fid_df = compute_fid_scores(
            generated_df=generated_df,
            base_model_name=args.base_model_name,
            fid_reference_mode=args.fid_reference_mode,
            fid_reference_dir=args.fid_reference_dir,
            batch_size=args.batch_size,
        )
        save_csv(eval_dir / "fid_by_target_family.csv", fid_df)
        if len(fid_df) > 0 and "FID" in fid_df.columns:
            fid_agg = fid_df.groupby("model_name", dropna=False)["FID"].mean().reset_index().rename(columns={"FID": "FID_mean"})
            save_csv(eval_dir / "fid_aggregated.csv", fid_agg)
            if len(aggregated_df) > 0 and "model_name" in aggregated_df.columns:
                merged = aggregated_df.merge(fid_agg, on="model_name", how="left")
                save_csv(eval_dir / "aggregated_metrics_with_fid.csv", merged)
            for _, row in fid_agg.iterrows():
                model_name = str(row["model_name"])
                model_eval_dir = ensure_dir(eval_dir / model_name)
                save_csv(
                    model_eval_dir / "fid_aggregated.csv",
                    fid_agg[fid_agg["model_name"] == model_name].copy(),
                )

    # Optional compact table export
    if len(aggregated_df) > 0:
        latex_df = aggregated_df.copy()
        for col in ["UA", "IRR", "IRA", "CRA", "NP", "CP", "DamageGap"]:
            if col in latex_df.columns:
                latex_df[col] = latex_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
        save_csv(eval_dir / "aggregated_metrics_latex_ready.csv", latex_df)

    print("Benchmark pipeline completed.")
    print(f"- Generated metadata: {generated_metadata_path}")
    print(f"- Per-image predictions: {per_image_pred_path}")
    print(f"- Per-target metrics: {eval_dir / 'per_target_metrics.csv'}")
    print(f"- Aggregated metrics: {eval_dir / 'aggregated_metrics.csv'}")


if __name__ == "__main__":
    main()
