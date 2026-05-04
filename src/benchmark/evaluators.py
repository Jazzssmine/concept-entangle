from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel


DEFAULT_CLIP_TEMPLATES = [
    "a photo of a {}",
    "an image of a {}",
    "a realistic {}",
    "a picture of a {}",
]


def _normalize_label(value: str) -> str:
    return str(value).strip().lower()


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_concept_vocabulary(path: str | Path) -> list[str]:
    vocab_path = Path(path)
    concepts = [
        _normalize_label(line)
        for line in vocab_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for concept in concepts:
        if concept not in seen:
            seen.add(concept)
            deduped.append(concept)
    if not deduped:
        raise ValueError(f"No concepts found in vocabulary file: {vocab_path}")
    return deduped


def load_clip_templates(path: str | Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_CLIP_TEMPLATES)
    template_path = Path(path)
    if template_path.suffix.lower() == ".json":
        obj = json.loads(template_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "templates" in obj:
            values = obj["templates"]
        else:
            values = obj
    else:
        values = [line.strip() for line in template_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    templates = [str(v).strip() for v in values if str(v).strip()]
    if not templates:
        raise ValueError(f"No CLIP templates found in: {template_path}")
    return templates


def _extract_concepts_from_obj(obj: object, kind: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if isinstance(obj, dict):
        nested_key = "semantic_neighbors" if kind == "neighbor" else "non_neighbor_controls"
        if "target" in obj and nested_key in obj:
            target = _normalize_label(obj["target"])
            nested = obj.get(nested_key, {})
            final_top_k = nested.get("final_top_k", []) if isinstance(nested, dict) else []
            out[target] = [_normalize_label(x) for x in final_top_k]
            return out
        for key, value in obj.items():
            if isinstance(value, dict):
                if kind in value and isinstance(value[kind], list):
                    out[_normalize_label(key)] = [_normalize_label(x) for x in value[kind]]
                elif "final_top_k" in value and isinstance(value["final_top_k"], list):
                    out[_normalize_label(key)] = [_normalize_label(x) for x in value["final_top_k"]]
                elif key in {"semantic_neighbors", "non_neighbor_controls"} and isinstance(value, dict):
                    continue
            elif isinstance(value, list):
                out[_normalize_label(key)] = [_normalize_label(x) for x in value]
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                out.update(_extract_concepts_from_obj(item, kind))
    return out


def load_concept_map(path: str | Path | None, kind: str) -> dict[str, list[str]]:
    if path is None:
        return {}
    map_path = Path(path)
    if map_path.is_dir():
        merged: dict[str, list[str]] = {}
        for fp in sorted(map_path.glob("*.json")):
            obj = json.loads(fp.read_text(encoding="utf-8"))
            merged.update(_extract_concepts_from_obj(obj, kind))
        return merged
    obj = json.loads(map_path.read_text(encoding="utf-8"))
    return _extract_concepts_from_obj(obj, kind)


@dataclass
class EvaluatorConfig:
    backend: str = "clip"
    batch_size: int = 64
    device: str | None = None
    top_k: int = 5
    save_topk: bool = False
    skip_missing_images: bool = True


@dataclass
class CLIPConceptEvaluatorConfig(EvaluatorConfig):
    backend: str = "clip"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_templates: list[str] = field(default_factory=lambda: list(DEFAULT_CLIP_TEMPLATES))
    save_embeddings: bool = False


class BaseEvaluator:
    def predict(
        self,
        generated_df: pd.DataFrame,
        metadata_root: str | Path | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        raise NotImplementedError


class CLIPConceptEvaluator(BaseEvaluator):
    def __init__(
        self,
        concept_vocabulary: list[str],
        cfg: CLIPConceptEvaluatorConfig,
    ) -> None:
        if not concept_vocabulary:
            raise ValueError("CLIP evaluator requires a non-empty concept vocabulary")
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.backend = "hf_clip"
        self.tokenizer = None
        self.preprocess = None
        if cfg.clip_model_name.startswith("open_clip:"):
            try:
                import open_clip
            except ImportError as exc:
                raise ImportError(
                    "open_clip is not installed. Install open_clip_torch or use a Hugging Face CLIP model name."
                ) from exc
            spec = cfg.clip_model_name.split(":", 1)[1]
            if "/" not in spec:
                raise ValueError(
                    "OpenCLIP model names must look like open_clip:ViT-B-32/laion2b_s34b_b79k"
                )
            model_name, pretrained = spec.split("/", 1)
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.model = model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.preprocess = preprocess
            self.backend = "open_clip"
        else:
            self.model = CLIPModel.from_pretrained(cfg.clip_model_name).to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(cfg.clip_model_name)
        self.concept_vocabulary = [_normalize_label(x) for x in concept_vocabulary]
        self._text_features = self._encode_texts()

    def _encode_texts(self) -> torch.Tensor:
        per_concept = []
        with torch.no_grad():
            for concept in self.concept_vocabulary:
                prompts = [template.format(concept) for template in self.cfg.clip_templates]
                if self.backend == "open_clip":
                    inputs = self.tokenizer(prompts).to(self.device)
                    text_features = self.model.encode_text(inputs)
                else:
                    inputs = self.processor(text=prompts, padding=True, return_tensors="pt").to(self.device)
                    text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)
                concept_feature = F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)
                per_concept.append(concept_feature)
        return torch.cat(per_concept, dim=0)

    def _resolve_image_path(self, image_path: str, metadata_root: str | Path | None) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        if metadata_root is not None:
            root = Path(metadata_root)
            candidate = root / path
            if candidate.exists():
                return candidate
        return path

    def predict(
        self,
        generated_df: pd.DataFrame,
        metadata_root: str | Path | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        rows: list[dict[str, object]] = []
        saved_embeddings: list[torch.Tensor] = []
        saved_paths: list[str] = []
        missing_images = 0

        top_k = max(1, min(self.cfg.top_k, len(self.concept_vocabulary)))
        for start in range(0, len(generated_df), self.cfg.batch_size):
            batch = generated_df.iloc[start : start + self.cfg.batch_size].reset_index(drop=True)
            images: list[Image.Image] = []
            valid_indices: list[int] = []
            valid_paths: list[str] = []

            for i, row in batch.iterrows():
                resolved_path = self._resolve_image_path(str(row["image_path"]), metadata_root)
                if not resolved_path.exists():
                    missing_images += 1
                    rows.append(
                        {
                            **row.to_dict(),
                            "predicted_label": None,
                            "top1_score": float("nan"),
                            "target_score": float("nan"),
                            "intended_score": float("nan"),
                            "score_margin": float("nan"),
                            "max_non_target_score": float("nan"),
                            "topk_labels": None,
                            "topk_scores": None,
                            "is_correct": None,
                            "is_target_pred": None,
                            "prediction_status": "missing_image",
                        }
                    )
                    continue
                image = Image.open(resolved_path).convert("RGB")
                images.append(image)
                valid_indices.append(i)
                valid_paths.append(str(resolved_path))

            if not images:
                continue

            with torch.no_grad():
                if self.backend == "open_clip":
                    image_tensor = torch.stack([self.preprocess(img) for img in images]).to(self.device)
                    image_features = self.model.encode_image(image_tensor)
                else:
                    inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                image_features = F.normalize(image_features, dim=-1)
                similarity = image_features @ self._text_features.T

            if self.cfg.save_embeddings:
                saved_embeddings.append(image_features.cpu())
                saved_paths.extend(valid_paths)

            values, indices = torch.topk(similarity, k=top_k, dim=1)
            for local_idx, batch_idx in enumerate(valid_indices):
                row = batch.iloc[batch_idx]
                sims = similarity[local_idx]
                top_scores = values[local_idx].detach().cpu().tolist()
                top_indices = indices[local_idx].detach().cpu().tolist()
                predicted_label = self.concept_vocabulary[int(top_indices[0])]
                intended_label = _normalize_label(row.get("intended_label", ""))
                target_label = _normalize_label(row.get("target_concept", ""))
                intended_score = float("nan")
                target_score = float("nan")
                if intended_label in self.concept_vocabulary:
                    intended_score = float(sims[self.concept_vocabulary.index(intended_label)].item())
                if target_label in self.concept_vocabulary:
                    target_score = float(sims[self.concept_vocabulary.index(target_label)].item())

                non_target_scores = [
                    float(score)
                    for concept, score in zip(self.concept_vocabulary, sims.detach().cpu().tolist())
                    if concept != target_label
                ]
                max_non_target_score = max(non_target_scores) if non_target_scores else float("nan")
                score_margin = float("nan")
                if not math.isnan(intended_score) and not math.isnan(target_score):
                    score_margin = float(intended_score - target_score)

                rows.append(
                    {
                        **row.to_dict(),
                        "predicted_label": predicted_label,
                        "top1_score": float(top_scores[0]),
                        "target_score": target_score,
                        "intended_score": intended_score,
                        "score_margin": score_margin,
                        "max_non_target_score": max_non_target_score,
                        "topk_labels": "|".join(self.concept_vocabulary[int(i)] for i in top_indices) if self.cfg.save_topk else None,
                        "topk_scores": "|".join(f"{float(score):.6f}" for score in top_scores) if self.cfg.save_topk else None,
                        "is_correct": bool(predicted_label == intended_label),
                        "is_target_pred": bool(predicted_label == target_label),
                        "prediction_status": "predicted",
                    }
                )

        pred_df = pd.DataFrame(rows)
        debug_payload: dict[str, object] = {
            "missing_images": int(missing_images),
            "concept_vocabulary_size": int(len(self.concept_vocabulary)),
            "clip_templates": list(self.cfg.clip_templates),
            "clip_model_name": self.cfg.clip_model_name,
        }
        if self.cfg.save_embeddings and saved_embeddings:
            debug_payload["image_embeddings"] = {
                "paths": saved_paths,
                "tensor": torch.cat(saved_embeddings, dim=0),
            }
        return pred_df, debug_payload
