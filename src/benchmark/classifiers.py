from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from .io_utils import load_json


@dataclass
class ClassifierSpec:
    name: str
    domain: str
    type: str
    model_name_or_path: str = ""
    checkpoint_path: str | None = None
    labels: list[str] = field(default_factory=list)
    labels_path: str | None = None
    device: str | None = None
    extra_args: dict = field(default_factory=dict)


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


class BaseClassifierAdapter:
    def predict(
        self,
        image_paths: list[str],
        candidate_labels: list[str] | None = None,
        intended_labels: list[str] | None = None,
    ) -> list[dict]:
        raise NotImplementedError


class OracleClassifierAdapter(BaseClassifierAdapter):
    def predict(self, image_paths, candidate_labels=None, intended_labels=None):
        out = []
        for idx, _ in enumerate(image_paths):
            label = intended_labels[idx] if intended_labels is not None else ""
            out.append({"predicted_label": _normalize_label(label), "confidence": 1.0})
        return out


class HFImageClassifierAdapter(BaseClassifierAdapter):
    def __init__(self, model_name_or_path: str, device: str):
        from transformers import pipeline

        device_idx = 0 if device.startswith("cuda") else -1
        self.pipe = pipeline("image-classification", model=model_name_or_path, device=device_idx)

    def predict(self, image_paths, candidate_labels=None, intended_labels=None):
        preds = self.pipe(image_paths)
        out = []
        for pred in preds:
            top = pred[0] if pred else {"label": "", "score": 0.0}
            out.append(
                {
                    "predicted_label": _normalize_label(top.get("label", "")),
                    "confidence": float(top.get("score", 0.0)),
                }
            )
        return out


class HFZeroShotImageClassifierAdapter(BaseClassifierAdapter):
    def __init__(self, model_name_or_path: str, device: str):
        from transformers import pipeline

        device_idx = 0 if device.startswith("cuda") else -1
        self.pipe = pipeline("zero-shot-image-classification", model=model_name_or_path, device=device_idx)

    def predict(self, image_paths, candidate_labels=None, intended_labels=None):
        if not candidate_labels:
            raise ValueError("zero-shot-image-classification requires candidate_labels")
        out = []
        for image_path in image_paths:
            preds = self.pipe(image_path, candidate_labels=candidate_labels)
            top = preds[0] if preds else {"label": "", "score": 0.0}
            out.append(
                {
                    "predicted_label": _normalize_label(top.get("label", "")),
                    "confidence": float(top.get("score", 0.0)),
                }
            )
        return out


class TimmClassifierAdapter(BaseClassifierAdapter):
    def __init__(self, spec: ClassifierSpec):
        import timm
        from timm.data import create_transform, resolve_data_config

        self.labels = [_normalize_label(x) for x in spec.labels]
        if not self.labels and spec.labels_path:
            from pathlib import Path

            lp = Path(spec.labels_path)
            self.labels = [_normalize_label(x.strip()) for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not self.labels:
            raise ValueError("timm classifier requires labels or labels_path")

        self.device = _resolve_device(spec.device)
        self.model = timm.create_model(spec.model_name_or_path, pretrained=False, num_classes=len(self.labels))
        if spec.checkpoint_path:
            ckpt = torch.load(spec.checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

        data_cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**data_cfg)

    def predict(self, image_paths, candidate_labels=None, intended_labels=None):
        imgs = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            imgs.append(self.transform(img))
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            scores, idxs = torch.max(probs, dim=1)

        out = []
        for s, i in zip(scores.cpu().tolist(), idxs.cpu().tolist()):
            label = self.labels[int(i)] if int(i) < len(self.labels) else ""
            out.append({"predicted_label": _normalize_label(label), "confidence": float(s)})
        return out


def load_classifier_registry(classifier_registry_path: str | None) -> dict[str, ClassifierSpec]:
    """
    Returns domain -> classifier spec.
    If path is None, return empty registry (prediction can be skipped).
    """
    if classifier_registry_path is None:
        return {}
    obj = load_json(classifier_registry_path)
    if not isinstance(obj, dict):
        raise ValueError("Classifier registry JSON must be dict: name -> spec")

    out = {}
    for name, cfg in obj.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid classifier config for '{name}'")
        domain = str(cfg.get("domain", "object")).strip().lower()
        spec = ClassifierSpec(
            name=name,
            domain=domain,
            type=str(cfg.get("type", "oracle")),
            model_name_or_path=str(cfg.get("model_name_or_path", "")),
            checkpoint_path=cfg.get("checkpoint_path"),
            labels=list(cfg.get("labels", [])),
            labels_path=cfg.get("labels_path"),
            device=cfg.get("device"),
            extra_args=dict(cfg.get("extra_args", {})),
        )
        out[domain] = spec
    return out


def build_classifier_adapter(spec: ClassifierSpec) -> BaseClassifierAdapter:
    kind = spec.type.lower()
    device = _resolve_device(spec.device)
    if kind == "oracle":
        return OracleClassifierAdapter()
    if kind == "hf_image_classification":
        return HFImageClassifierAdapter(spec.model_name_or_path, device=device)
    if kind == "hf_zero_shot_image_classification":
        return HFZeroShotImageClassifierAdapter(spec.model_name_or_path, device=device)
    if kind == "timm":
        return TimmClassifierAdapter(spec)
    raise ValueError(f"Unsupported classifier type: {spec.type}")

