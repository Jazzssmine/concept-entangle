from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .classifiers import build_classifier_adapter, load_classifier_registry
from .evaluators import BaseEvaluator
from .io_utils import save_csv


@dataclass
class PredictionConfig:
    batch_size: int = 16
    reuse_cached_predictions: bool = True


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_synonym_map(target_synonyms_map: dict[str, list[str]] | None) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {}
    for target, variants in (target_synonyms_map or {}).items():
        target_norm = _normalize_label(target)
        if not target_norm:
            continue
        bucket = normalized.setdefault(target_norm, set())
        bucket.add(target_norm)
        for variant in variants or []:
            variant_norm = _normalize_label(variant)
            if variant_norm:
                bucket.add(variant_norm)
    return normalized


def _is_equivalent_label(pred_label: str, ref_label: str, synonym_sets: dict[str, set[str]]) -> bool:
    pred_norm = _normalize_label(pred_label)
    ref_norm = _normalize_label(ref_label)
    if pred_norm == ref_norm:
        return True
    return pred_norm in synonym_sets.get(ref_norm, set())


def _apply_label_equivalence(
    pred_df: pd.DataFrame,
    target_synonyms_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    if len(pred_df) == 0:
        return pred_df
    synonym_sets = _normalize_synonym_map(target_synonyms_map)
    if not synonym_sets:
        return pred_df

    out = pred_df.copy()
    if {"predicted_label", "target_concept"}.issubset(out.columns):
        out["is_target_pred"] = out.apply(
            lambda r: _is_equivalent_label(r["predicted_label"], r["target_concept"], synonym_sets),
            axis=1,
        )
    if {"predicted_label", "intended_label"}.issubset(out.columns):
        out["is_correct"] = out.apply(
            lambda r: _is_equivalent_label(r["predicted_label"], r["intended_label"], synonym_sets),
            axis=1,
        )
    return out


def run_evaluator_predictions(
    generated_df: pd.DataFrame,
    evaluator: BaseEvaluator,
    output_predictions_path: str | Path,
    metadata_root: str | Path | None = None,
    skip_existing_predictions: bool = False,
    target_synonyms_map: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_predictions_path = Path(output_predictions_path)
    if skip_existing_predictions and output_predictions_path.exists():
        pred_df = pd.read_csv(output_predictions_path)
        pred_df = _apply_label_equivalence(pred_df, target_synonyms_map=target_synonyms_map)
        return pred_df, {"prediction_status": "loaded_existing"}

    pred_df, debug_payload = evaluator.predict(generated_df=generated_df, metadata_root=metadata_root)
    pred_df = _apply_label_equivalence(pred_df, target_synonyms_map=target_synonyms_map)
    save_csv(output_predictions_path, pred_df)
    return pred_df, debug_payload


def run_classifier_predictions(
    generated_df: pd.DataFrame,
    classifier_registry_path: str | None,
    output_predictions_path: str | Path,
    cfg: PredictionConfig,
    target_synonyms_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    output_predictions_path = Path(output_predictions_path)
    synonym_sets = _normalize_synonym_map(target_synonyms_map)
    cached_df = pd.DataFrame()
    if output_predictions_path.exists() and cfg.reuse_cached_predictions:
        cached_df = pd.read_csv(output_predictions_path)

    cache_keys = set()
    if len(cached_df) > 0:
        for _, row in cached_df.iterrows():
            cache_keys.add((str(row["image_path"]), str(row["model_name"])))

    classifier_specs = load_classifier_registry(classifier_registry_path)
    if not classifier_specs:
        raise ValueError(
            "No classifier registry loaded. Provide --classifier_registry_path or use --skip_prediction."
        )

    rows = []
    for domain, group in generated_df.groupby("domain"):
        domain = str(domain).lower()
        if domain not in classifier_specs:
            # Missing classifier for this domain: keep NA predictions.
            for _, r in group.iterrows():
                rows.append(
                    {
                        **r.to_dict(),
                        "predicted_label": None,
                        "confidence": None,
                        "is_correct": None,
                        "is_target_pred": None,
                        "prediction_status": "missing_classifier",
                    }
                )
            continue

        spec = classifier_specs[domain]
        adapter = build_classifier_adapter(spec)
        candidate_labels = sorted(group["intended_label"].astype(str).str.lower().unique().tolist())

        group = group.reset_index(drop=True)
        for start in tqdm(range(0, len(group), cfg.batch_size), desc=f"Predict [{domain}]"):
            batch = group.iloc[start : start + cfg.batch_size]
            image_paths = batch["image_path"].astype(str).tolist()
            intended_labels = batch["intended_label"].astype(str).str.lower().tolist()
            target_labels = batch["target_concept"].astype(str).str.lower().tolist()
            model_names = batch["model_name"].astype(str).tolist()

            batch_preds = []
            # Reuse cached rows when available.
            missing_mask = []
            missing_paths = []
            missing_intended = []
            for i, path in enumerate(image_paths):
                ck = (path, model_names[i])
                if ck in cache_keys:
                    row = cached_df[(cached_df["image_path"] == path) & (cached_df["model_name"] == model_names[i])]
                    if len(row) > 0:
                        rr = row.iloc[-1].to_dict()
                        batch_preds.append(
                            {
                                "predicted_label": rr.get("predicted_label"),
                                "confidence": rr.get("confidence"),
                                "prediction_status": "cached",
                            }
                        )
                        missing_mask.append(False)
                        continue
                batch_preds.append(None)
                missing_mask.append(True)
                missing_paths.append(path)
                missing_intended.append(intended_labels[i])

            if missing_paths:
                new_preds = adapter.predict(
                    image_paths=missing_paths,
                    candidate_labels=candidate_labels,
                    intended_labels=missing_intended,
                )
                p_idx = 0
                for i, m in enumerate(missing_mask):
                    if m:
                        pred = new_preds[p_idx]
                        p_idx += 1
                        batch_preds[i] = {
                            "predicted_label": pred.get("predicted_label"),
                            "confidence": pred.get("confidence"),
                            "prediction_status": "predicted",
                        }

            for i in range(len(batch)):
                pred_label = str(batch_preds[i].get("predicted_label") or "").strip().lower()
                conf = batch_preds[i].get("confidence")
                rows.append(
                    {
                        **batch.iloc[i].to_dict(),
                        "predicted_label": pred_label,
                        "confidence": conf,
                        "is_correct": _is_equivalent_label(pred_label, intended_labels[i], synonym_sets),
                        "is_target_pred": _is_equivalent_label(pred_label, target_labels[i], synonym_sets),
                        "prediction_status": batch_preds[i].get("prediction_status"),
                    }
                )

    pred_df = pd.DataFrame(rows)
    save_csv(output_predictions_path, pred_df)
    return pred_df
