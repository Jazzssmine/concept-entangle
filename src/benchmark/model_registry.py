from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import load_json


@dataclass
class ModelSpec:
    name: str
    type: str = "diffusers"
    model_path: str = ""
    pipeline_class: str = "StableDiffusionPipeline"
    torch_dtype: str = "float16"
    safety_checker: bool = False
    unet_checkpoint: str | None = None
    text_encoder_checkpoint: str | None = None
    revision: str | None = None
    variant: str | None = None
    extra_args: dict = field(default_factory=dict)


def load_model_registry(model_registry_path: str | Path) -> dict[str, ModelSpec]:
    obj = load_json(model_registry_path)
    if not isinstance(obj, dict):
        raise ValueError("Model registry JSON must be a dict: model_name -> config.")

    registry: dict[str, ModelSpec] = {}
    for name, cfg in obj.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid config for model '{name}'")
        registry[name] = ModelSpec(
            name=name,
            type=str(cfg.get("type", "diffusers")),
            model_path=str(cfg.get("model_path", "")),
            pipeline_class=str(cfg.get("pipeline_class", "StableDiffusionPipeline")),
            torch_dtype=str(cfg.get("torch_dtype", "float16")),
            safety_checker=bool(cfg.get("safety_checker", False)),
            unet_checkpoint=cfg.get("unet_checkpoint"),
            text_encoder_checkpoint=cfg.get("text_encoder_checkpoint"),
            revision=cfg.get("revision"),
            variant=cfg.get("variant"),
            extra_args=dict(cfg.get("extra_args", {})),
        )
        if not registry[name].model_path:
            raise ValueError(f"Model '{name}' missing required field: model_path")
    return registry

