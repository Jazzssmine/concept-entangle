from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.evaluators import (
    CLIPConceptEvaluator,
    CLIPConceptEvaluatorConfig,
    load_clip_templates,
    load_concept_map,
    load_concept_vocabulary,
)
from src.benchmark.io_utils import ensure_dir, save_csv, save_json
from src.benchmark.metrics import aggregate_metrics, compute_per_target_metrics, save_metrics_outputs
from src.benchmark.prediction import run_evaluator_predictions


def _resolve_repo_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _load_metadata(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict) and "records" in obj and isinstance(obj["records"], list):
            return pd.DataFrame(obj["records"])
    raise ValueError(f"Unsupported metadata format: {p}")


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
    parser = argparse.ArgumentParser(description="CLIP-based unlearning evaluation over a fixed concept vocabulary.")
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--concept_vocab_path", type=str, required=True)
    parser.add_argument("--neighbor_map_path", type=str, default=None)
    parser.add_argument("--control_map_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, default="base")
    parser.add_argument("--evaluator_backend", type=str, default="clip", choices=["clip"])
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_templates_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_topk", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--skip_existing_predictions", action="store_true")
    parser.add_argument("--min_base_acc_for_normalization", type=float, default=0.05)
    parser.add_argument("--metadata_root", type=str, default=None)
    parser.add_argument("--compute_cp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.metadata_path = _resolve_repo_path(args.metadata_path)
    args.concept_vocab_path = _resolve_repo_path(args.concept_vocab_path)
    args.neighbor_map_path = _resolve_repo_path(args.neighbor_map_path)
    args.control_map_path = _resolve_repo_path(args.control_map_path)
    args.output_dir = _resolve_repo_path(args.output_dir)
    args.clip_templates_path = _resolve_repo_path(args.clip_templates_path)
    args.metadata_root = _resolve_repo_path(args.metadata_root)

    output_dir = ensure_dir(args.output_dir)
    metadata_df = _load_metadata(args.metadata_path)
    concept_vocabulary = load_concept_vocabulary(args.concept_vocab_path)
    templates = load_clip_templates(args.clip_templates_path)
    neighbor_map = load_concept_map(args.neighbor_map_path, kind="neighbor")
    control_map = load_concept_map(args.control_map_path, kind="control")

    if args.evaluator_backend != "clip":
        raise ValueError(f"Unsupported evaluator backend: {args.evaluator_backend}")

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
        metadata_root=args.metadata_root or Path(args.metadata_path).resolve().parent,
        skip_existing_predictions=args.skip_existing_predictions,
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

    save_csv(output_dir / "per_target_clip_metrics.csv", per_target_df, na_rep="NA")
    save_csv(output_dir / "aggregated_clip_metrics.csv", aggregated_df, na_rep="NA")
    save_csv(output_dir / "clip_detailed_breakdowns.csv", detailed_df, na_rep="NA")
    save_csv(output_dir / "debug_metric_inputs.csv", debug_df, na_rep="NA")
    if len(detailed_df) > 0 and "metric_component" in detailed_df.columns:
        save_csv(output_dir / "per_neighbor_metrics.csv", detailed_df[detailed_df["metric_component"] == "NP"].copy(), na_rep="NA")
        save_csv(output_dir / "per_control_metrics.csv", detailed_df[detailed_df["metric_component"] == "CP"].copy(), na_rep="NA")
    _save_per_target_slices(output_dir, pred_df, per_target_df, detailed_df, debug_df)
    save_metrics_outputs(
        per_target_df=per_target_df,
        aggregated_df=aggregated_df,
        detailed_df=detailed_df,
        eval_dir=str(output_dir),
        prefix="clip",
        debug_df=debug_df,
    )

    run_summary = {
        "metadata_path": args.metadata_path,
        "concept_vocab_path": args.concept_vocab_path,
        "neighbor_map_path": args.neighbor_map_path,
        "control_map_path": args.control_map_path,
        "output_dir": args.output_dir,
        "n_images": int(len(metadata_df)),
        "n_predictions": int(len(pred_df)),
        "n_targets": int(per_target_df["target_concept"].nunique()) if len(per_target_df) else 0,
        "config": vars(args),
        "debug": {k: v for k, v in debug_payload.items() if k != "image_embeddings"},
    }
    save_json(output_dir / "clip_evaluation_summary.json", run_summary)

    if args.save_embeddings and "image_embeddings" in debug_payload:
        torch.save(debug_payload["image_embeddings"], output_dir / "clip_image_embeddings.pt")

    print("CLIP evaluation completed.")
    print(f"- Per-image predictions: {per_image_path}")
    print(f"- Per-target metrics: {output_dir / 'per_target_clip_metrics.csv'}")
    print(f"- Aggregated metrics: {output_dir / 'aggregated_clip_metrics.csv'}")


if __name__ == "__main__":
    main()
