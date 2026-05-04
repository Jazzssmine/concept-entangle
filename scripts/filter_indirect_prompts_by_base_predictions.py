#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.benchmark.prompt_loader import stable_prompt_id_from_fields


def _norm_text(value: object) -> str:
    return str(value).strip().lower()


def _to_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _load_prompts_with_prompt_id(prompts_path: Path) -> pd.DataFrame:
    df = pd.read_csv(prompts_path)
    rename_map = {"target": "target_concept", "concept_label": "intended_label"}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    required = {"target_concept", "prompt_family", "intended_label", "prompt"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required prompt columns in {prompts_path}: {missing}")

    if "lexical_mode" not in df.columns:
        df["lexical_mode"] = "strict"

    df["target_concept"] = df["target_concept"].astype(str).str.strip().str.lower()
    df["prompt_family"] = df["prompt_family"].astype(str).str.strip().str.lower()
    df["intended_label"] = df["intended_label"].astype(str).str.strip().str.lower()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["lexical_mode"] = df["lexical_mode"].astype(str).str.strip().str.lower()

    df["prompt_id"] = df.apply(
        lambda row: stable_prompt_id_from_fields(
            target_concept=row["target_concept"],
            prompt_family=row["prompt_family"],
            intended_label=row["intended_label"],
            prompt=row["prompt"],
            lexical_mode=row["lexical_mode"],
        ),
        axis=1,
    )
    return df


def _mode_or_na(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    if len(vals) == 0:
        return "NA"
    mode = vals.mode()
    return str(mode.iloc[0]) if len(mode) else str(vals.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match indirect prompts to generated base-model images and CLIP predictions, "
            "then remove prompt rows whose images do not predict the target concept."
        )
    )
    parser.add_argument("--prompts_csv", type=Path, required=True, help="Prompt CSV (e.g., outputs/prompts/horse/all_prompts.csv)")
    parser.add_argument(
        "--generated_metadata_csv",
        type=Path,
        default=Path("outputs/benchmark/metadata/generated_images.csv"),
        help="Generated metadata CSV with prompt_id/image_path mappings.",
    )
    parser.add_argument(
        "--per_image_predictions_csv",
        type=Path,
        default=Path("outputs/benchmark/eval/horse_per_image_clip_predictions.csv"),
        help="Per-image CLIP predictions CSV.",
    )
    parser.add_argument("--model_name", type=str, default="base_horse", help="Model name to filter in metadata/predictions.")
    parser.add_argument("--target", type=str, default="horse", help="Target concept to filter.")
    parser.add_argument("--prompt_family", type=str, default="indirect", help="Prompt family to filter.")
    parser.add_argument(
        "--drop_policy",
        type=str,
        default="any_non_target",
        choices=["any_non_target", "majority_non_target", "all_non_target"],
        help="How to drop prompt_ids when multiple images exist per prompt_id.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/benchmark/filters"),
        help="Directory for reports and cleaned prompt CSV.",
    )
    parser.add_argument(
        "--write_back",
        action="store_true",
        help="Overwrite --prompts_csv with the cleaned prompt table (backup is written first).",
    )
    args = parser.parse_args()

    model_name = _norm_text(args.model_name)
    target = _norm_text(args.target)
    prompt_family = _norm_text(args.prompt_family)

    prompts_df = _load_prompts_with_prompt_id(args.prompts_csv)
    metadata_df = pd.read_csv(args.generated_metadata_csv)
    preds_df = pd.read_csv(args.per_image_predictions_csv)

    for df in (metadata_df, preds_df):
        for col in ("model_name", "target_concept", "prompt_family"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

    prompts_scope = prompts_df[
        (prompts_df["target_concept"] == target) & (prompts_df["prompt_family"] == prompt_family)
    ].copy()
    if len(prompts_scope) == 0:
        raise ValueError(f"No prompt rows found in prompts CSV for target={target!r}, family={prompt_family!r}.")

    meta_scope = metadata_df[
        (metadata_df["model_name"] == model_name)
        & (metadata_df["target_concept"] == target)
        & (metadata_df["prompt_family"] == prompt_family)
    ].copy()
    pred_scope = preds_df[
        (preds_df["model_name"] == model_name)
        & (preds_df["target_concept"] == target)
        & (preds_df["prompt_family"] == prompt_family)
    ].copy()

    pred_scope["is_target_pred_bool"] = pred_scope.get("is_target_pred", pd.Series([None] * len(pred_scope))).map(_to_bool)
    pred_scope["predicted_label"] = pred_scope.get("predicted_label", pd.Series([""] * len(pred_scope))).astype(str).str.strip().str.lower()
    pred_scope["target_score"] = pd.to_numeric(pred_scope.get("target_score"), errors="coerce")
    pred_scope["top1_score"] = pd.to_numeric(pred_scope.get("top1_score"), errors="coerce")
    pred_scope["score_margin"] = pd.to_numeric(pred_scope.get("score_margin"), errors="coerce")

    join_keys = [k for k in ("prompt_id", "seed", "image_index") if k in pred_scope.columns and k in meta_scope.columns]
    if not join_keys:
        raise ValueError("Could not find shared keys between metadata and predictions (need at least prompt_id).")

    merged = pred_scope.merge(
        meta_scope[["prompt_id", "seed", "image_index", "image_path", "prompt", "lexical_mode"]].drop_duplicates(),
        on=join_keys,
        how="left",
        suffixes=("_pred", "_meta"),
    )

    image_path_col = "image_path"
    if "image_path" not in merged.columns:
        pred_col = "image_path_pred" if "image_path_pred" in merged.columns else None
        meta_col = "image_path_meta" if "image_path_meta" in merged.columns else None
        merged["image_path"] = (
            merged[pred_col].fillna(merged[meta_col]) if pred_col and meta_col else merged.get(pred_col or meta_col)
        )
        image_path_col = "image_path"

    lexical_col = "lexical_mode"
    if "lexical_mode" not in merged.columns:
        pred_col = "lexical_mode_pred" if "lexical_mode_pred" in merged.columns else None
        meta_col = "lexical_mode_meta" if "lexical_mode_meta" in merged.columns else None
        merged["lexical_mode"] = (
            merged[pred_col].fillna(merged[meta_col]) if pred_col and meta_col else merged.get(pred_col or meta_col)
        )
        lexical_col = "lexical_mode"

    merged = merged.merge(
        prompts_scope[["prompt_id", "prompt"]].rename(columns={"prompt": "prompt_from_csv"}),
        on="prompt_id",
        how="left",
    )
    prompt_candidate_col = "prompt"
    if "prompt" not in merged.columns:
        prompt_candidate_col = "prompt_pred" if "prompt_pred" in merged.columns else "prompt_meta"
    merged["prompt_text"] = merged["prompt_from_csv"].fillna(merged.get(prompt_candidate_col, pd.Series([""] * len(merged))))
    merged["keep_image"] = merged["is_target_pred_bool"] == True
    merged["drop_image_reason"] = merged["predicted_label"].where(~merged["keep_image"], other="")

    per_prompt = (
        merged.groupby("prompt_id", dropna=False)
        .agg(
            n_images=("prompt_id", "size"),
            n_target_pred=("is_target_pred_bool", lambda s: int((s == True).sum())),
            n_non_target_pred=("is_target_pred_bool", lambda s: int((s == False).sum())),
            predicted_label_mode=("predicted_label", _mode_or_na),
            mean_target_score=("target_score", "mean"),
            mean_top1_score=("top1_score", "mean"),
            mean_score_margin=("score_margin", "mean"),
            prompt_text=("prompt_text", _mode_or_na),
            lexical_mode=(lexical_col, _mode_or_na),
            example_image_path=(image_path_col, _mode_or_na),
        )
        .reset_index()
    )
    per_prompt["non_target_rate"] = per_prompt["n_non_target_pred"] / per_prompt["n_images"].clip(lower=1)

    if args.drop_policy == "any_non_target":
        per_prompt["drop_prompt"] = per_prompt["n_non_target_pred"] > 0
    elif args.drop_policy == "majority_non_target":
        per_prompt["drop_prompt"] = per_prompt["n_non_target_pred"] > per_prompt["n_target_pred"]
    else:  # all_non_target
        per_prompt["drop_prompt"] = per_prompt["n_non_target_pred"] == per_prompt["n_images"]

    drop_prompt_ids = set(per_prompt.loc[per_prompt["drop_prompt"], "prompt_id"].astype(str))

    cleaned_prompts = prompts_df.copy()
    drop_mask = (
        (cleaned_prompts["target_concept"] == target)
        & (cleaned_prompts["prompt_family"] == prompt_family)
        & (cleaned_prompts["prompt_id"].isin(drop_prompt_ids))
    )
    dropped_rows = cleaned_prompts[drop_mask].copy()
    cleaned_prompts = cleaned_prompts[~drop_mask].copy()

    out_dir = args.out_dir / f"{model_name}_{target}_{prompt_family}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_report_path = out_dir / "image_level_report.csv"
    prompt_report_path = out_dir / "prompt_level_report.csv"
    dropped_prompt_path = out_dir / "dropped_prompt_rows.csv"
    cleaned_prompt_path = out_dir / "cleaned_prompts.csv"

    merged.to_csv(image_report_path, index=False)
    per_prompt.sort_values(["drop_prompt", "non_target_rate", "prompt_id"], ascending=[False, False, True]).to_csv(
        prompt_report_path, index=False
    )
    dropped_rows.to_csv(dropped_prompt_path, index=False)
    cleaned_prompts.drop(columns=["prompt_id"], errors="ignore").to_csv(cleaned_prompt_path, index=False)

    if args.write_back:
        backup = args.prompts_csv.with_suffix(args.prompts_csv.suffix + ".backup_before_filter.csv")
        args.prompts_csv.rename(backup)
        cleaned_prompts.drop(columns=["prompt_id"], errors="ignore").to_csv(args.prompts_csv, index=False)
        print(f"[write_back] wrote cleaned prompts to: {args.prompts_csv}")
        print(f"[write_back] backup saved at: {backup}")

    scoped_total = int(len(prompts_scope))
    dropped_total = int(len(dropped_rows))
    kept_total = scoped_total - dropped_total
    print(f"target={target} family={prompt_family} model={model_name}")
    print(f"scoped prompts: {scoped_total}")
    print(f"dropped prompts: {dropped_total}")
    print(f"kept prompts: {kept_total}")
    print(f"image-level report: {image_report_path}")
    print(f"prompt-level report: {prompt_report_path}")
    print(f"dropped prompt rows: {dropped_prompt_path}")
    print(f"cleaned prompts: {cleaned_prompt_path}")


if __name__ == "__main__":
    main()
