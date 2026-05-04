from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .io_utils import ensure_dir, save_csv
from .model_registry import ModelSpec


@dataclass
class GenerationConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    images_per_prompt: int = 1
    seeds: list[int] | None = None
    batch_size: int = 1
    device: str | None = None
    resume: bool = True
    image_format: str = "png"
    height: int | None = None
    width: int | None = None


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _sanitize_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:120] if out else "item"


def _deterministic_filename(
    model_name: str,
    target_concept: str,
    prompt_id: str,
    seed: int,
    image_index: int,
    ext: str,
) -> str:
    key = f"{model_name}|{target_concept}|{prompt_id}|{seed}|{image_index}"
    suffix = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return f"{_sanitize_name(prompt_id)}__s{seed}__i{image_index}__{suffix}.{ext}"


def _coerce_seed(value) -> int:
    x = float(value)
    if x != int(x):
        raise ValueError(f"Non-integer seed in metadata: {value!r}")
    return int(x)


def _image_file_exists(raw_path: str, output_root: Path) -> bool:
    """True if the image exists at raw_path, optionally resolved relative to output_root or cwd."""
    p = Path(raw_path)
    if p.is_file():
        return True
    if not p.is_absolute():
        for base in (output_root, Path.cwd()):
            q = (base / p).resolve()
            if q.is_file():
                return True
    return False


def _load_diffusers_pipeline(spec: ModelSpec, device: str):
    import diffusers

    pipeline_cls = getattr(diffusers, spec.pipeline_class, None)
    if pipeline_cls is None:
        raise ValueError(f"Unknown diffusers pipeline class: {spec.pipeline_class}")

    pipe = pipeline_cls.from_pretrained(
        spec.model_path,
        torch_dtype=_resolve_dtype(spec.torch_dtype),
        safety_checker=None if not spec.safety_checker else None,
        revision=spec.revision,
        variant=spec.variant,
        **spec.extra_args,
    ).to(device)

    if spec.unet_checkpoint:
        ckpt_path = Path(spec.unet_checkpoint)
        if ckpt_path.suffix.lower() == ".safetensors":
            try:
                from safetensors.torch import load_file as safe_load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading .safetensors UNet checkpoints requires safetensors. "
                    "Install with: pip install safetensors"
                ) from exc
            ckpt_obj = safe_load_file(str(ckpt_path), device=device)
        else:
            ckpt_obj = torch.load(str(ckpt_path), map_location=device)
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            ckpt_state = ckpt_obj["state_dict"]
        else:
            ckpt_state = ckpt_obj
        pipe.unet.load_state_dict(ckpt_state, strict=False)

    pipe.set_progress_bar_config(disable=True)
    return pipe


def _strip_te_prefixes(state_dict: dict) -> dict:
    known_prefixes = [
        "text_encoder.",
        "module.text_encoder.",
        "cond_stage_model.transformer.",
        "module.cond_stage_model.transformer.",
        "module.",
    ]
    out = dict(state_dict)
    changed = True
    while changed:
        changed = False
        keys = list(out.keys())
        for pref in known_prefixes:
            if keys and all(k.startswith(pref) for k in keys):
                out = {k[len(pref):]: v for k, v in out.items()}
                changed = True
                break
    return out


def _extract_advunlearn_text_encoder_state(raw_obj: object) -> dict[str, torch.Tensor]:
    """
    Match advunlearn generate-example-img.py behavior:
    - If checkpoint is a full training state dict, extract keys containing
      'text_encoder.text_model' and drop 'text_encoder.' prefix.
    - Otherwise accept common direct text-encoder state dict formats.
    """
    candidates: list[dict] = []
    if isinstance(raw_obj, dict):
        candidates.append(raw_obj)
        for key in ("state_dict", "model", "module", "weights", "text_encoder"):
            value = raw_obj.get(key)
            if isinstance(value, dict):
                candidates.append(value)

    tensor_candidates: list[dict[str, torch.Tensor]] = []
    for cand in candidates:
        if cand and all(isinstance(v, torch.Tensor) for v in cand.values()):
            tensor_candidates.append(cand)  # type: ignore[arg-type]
    if not tensor_candidates:
        raise ValueError("Unsupported advunlearn checkpoint format: expected tensor state dict.")

    # 1) Preferred: full checkpoint containing text_encoder.text_model.* keys.
    for cand in tensor_candidates:
        if any("text_encoder.text_model" in k for k in cand.keys()):
            extracted = {
                k.replace("text_encoder.", "", 1): v
                for k, v in cand.items()
                if "text_encoder.text_model" in k
            }
            if extracted:
                return extracted

    # 2) Already a text encoder state dict (text_model.*).
    for cand in tensor_candidates:
        if any(k.startswith("text_model.") for k in cand.keys()):
            return cand

    # 3) Fallback: return first tensor dict; prefix normalization handles common wrappers.
    return tensor_candidates[0]


def _load_advunlearn_pipeline(spec: ModelSpec, device: str):
    pipe = _load_diffusers_pipeline(spec, device)

    if not spec.text_encoder_checkpoint:
        raise ValueError(f"Model '{spec.name}' type 'advunlearn' requires text_encoder_checkpoint.")

    raw = torch.load(spec.text_encoder_checkpoint, map_location=device)
    state_dict = _extract_advunlearn_text_encoder_state(raw)
    state_dict = _strip_te_prefixes(state_dict)

    emb_key = "text_model.embeddings.token_embedding.weight"
    if emb_key in state_dict:
        expected_vocab = int(state_dict[emb_key].shape[0])
        while len(pipe.tokenizer) < expected_vocab:
            pipe.tokenizer.add_tokens([f"__pad_token_{len(pipe.tokenizer)}__"])
        pipe.text_encoder.resize_token_embeddings(expected_vocab)

    missing, unexpected = pipe.text_encoder.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[advunlearn] unexpected TE keys ignored: {len(unexpected)}")
    if missing:
        print(f"[advunlearn] missing TE keys (kept from base TE): {len(missing)}")
    pipe.text_encoder.eval()
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)

    print(f"[advunlearn] Loaded text encoder from {spec.text_encoder_checkpoint}")
    return pipe


def _load_salun_pipeline(spec: ModelSpec, device: str):
    """Load a SalUn model saved in compvis/LDM format by remapping keys in-memory."""
    import dataclasses
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    salun_train_scripts = repo_root / "baselines" / "salun" / "SD" / "train-scripts"
    sys.path.insert(0, str(salun_train_scripts))
    from convertModels import convert_ldm_unet_checkpoint, create_unet_diffusers_config  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    if not spec.unet_checkpoint:
        raise ValueError(f"Model '{spec.name}' type 'salun' requires unet_checkpoint.")

    compvis_path = _Path(spec.unet_checkpoint)
    if not compvis_path.is_file():
        raise FileNotFoundError(f"[salun] Compvis checkpoint not found: {compvis_path}")

    # Resolve the LDM yaml config (needed for key mapping)
    cfg_override = spec.extra_args.get("compvis_config_path")
    if cfg_override:
        yaml_path = _Path(cfg_override)
    else:
        yaml_path = (
            repo_root / "baselines" / "salun" / "SD"
            / "configs" / "stable-diffusion" / "v1-inference.yaml"
        )
    if not yaml_path.is_file():
        raise FileNotFoundError(f"[salun] v1-inference.yaml not found: {yaml_path}")

    original_config = OmegaConf.load(str(yaml_path))
    unet_config = create_unet_diffusers_config(original_config, image_size=512)

    checkpoint = torch.load(str(compvis_path), map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    converted = convert_ldm_unet_checkpoint(checkpoint, unet_config, extract_ema=False)

    # Load base pipeline without touching unet_checkpoint (keys would mismatch)
    base_spec = dataclasses.replace(spec, unet_checkpoint=None)
    pipe = _load_diffusers_pipeline(base_spec, device)

    pipe.unet.load_state_dict(converted)
    print(f"[salun] Loaded compvis UNet from {compvis_path}")
    return pipe


class _SAEUronPipelineWrapper:
    """Wraps HookedStableDiffusionPipeline to expose a standard diffusers-like __call__ interface.

    Uses SAEMaskedUnlearningHook (the proper SAEUron method): a new hook instance is created
    per call so that timestep_idx is always reset, while the expensive precomputed tensors
    (concept_latents_dict) are kept in memory across calls.
    """

    def __init__(self, model, hookpoint: str, sae, concept_latents_dict, target_class, percentile, multiplier):
        self._model = model
        self._hookpoint = hookpoint
        self._sae = sae
        self._concept_latents_dict = concept_latents_dict
        self._target_class = target_class
        self._percentile = percentile
        self._multiplier = multiplier

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, prompt, num_inference_steps=50, guidance_scale=7.5, generator=None, **kwargs):
        import sys
        from pathlib import Path as _Path

        repo_root = _Path(__file__).resolve().parents[2]
        saeuron_root = repo_root / "baselines" / "saeuron"
        if str(saeuron_root) not in sys.path:
            sys.path.insert(0, str(saeuron_root))

        from SAE.unlearning_utils import compute_feature_importance  # noqa: PLC0415
        from utils.hooks import SAEMaskedUnlearningHook  # noqa: PLC0415

        hook = SAEMaskedUnlearningHook(
            concept_to_unlearn=[self._target_class],
            percentile=self._percentile,
            multiplier=self._multiplier,
            feature_importance_fn=compute_feature_importance,
            concept_latents_dict=self._concept_latents_dict,
            sae=self._sae,
            steps=num_inference_steps,
            preserve_error=True,
        )
        images = self._model.run_with_hooks(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            position_hook_dict={self._hookpoint: hook},
        )

        class _Out:
            pass

        out = _Out()
        out.images = images
        return out


def _load_saeuron_pipeline(spec: ModelSpec, device: str):
    """Load a SAEUron pipeline using SAEMaskedUnlearningHook.

    Requires extra_args:
      sae_checkpoint     – path (relative to repo root) to the dir containing hookpoint subdirs
      hookpoint          – e.g. "unet.up_blocks.1.attentions.1"
      class_latents_path – path to the precomputed cls_latents_dict_<hookpoint>.pkl
                           produced by baselines/saeuron/scripts/gather_sae_acts_ca_prompts_cls.py
      target_class       – class name as it appears in the pkl keys, e.g. "Horses"
      percentile         – float, e.g. 99.995
      multiplier         – float, e.g. -25.0
    """
    import pickle
    import sys
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    saeuron_root = repo_root / "baselines" / "saeuron"
    if str(saeuron_root) not in sys.path:
        sys.path.insert(0, str(saeuron_root))

    from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline  # noqa: PLC0415
    from SAE.sae import Sae  # noqa: PLC0415

    extra = spec.extra_args
    sae_checkpoint = extra.get("sae_checkpoint")
    if not sae_checkpoint:
        raise ValueError(f"Model '{spec.name}' type 'saeuron' requires extra_args.sae_checkpoint")
    class_latents_path = extra.get("class_latents_path")
    if not class_latents_path:
        raise ValueError(
            f"Model '{spec.name}' type 'saeuron' requires extra_args.class_latents_path. "
            "Run baselines/saeuron/scripts/gather_sae_acts_ca_prompts_cls.py first."
        )
    hookpoint = str(extra.get("hookpoint", "unet.up_blocks.1.attentions.1"))
    target_class = str(extra.get("target_class", "Horses"))
    percentile = float(extra.get("percentile", 99.995))
    multiplier = float(extra.get("multiplier", -25.0))

    dtype = _resolve_dtype(spec.torch_dtype)

    model = HookedStableDiffusionPipeline.from_pretrained(
        spec.model_path,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # SAE checkpoint: sae_checkpoint/<hookpoint>/cfg.json + sae.safetensors
    sae_ckpt_root = _Path(sae_checkpoint)
    if not sae_ckpt_root.is_absolute():
        sae_ckpt_root = (repo_root / sae_ckpt_root).resolve()
    sae_path = sae_ckpt_root / hookpoint
    if not sae_path.is_dir():
        raise FileNotFoundError(f"[saeuron] SAE checkpoint directory not found: {sae_path}")

    sae = Sae.load_from_disk(sae_path, device=device).to(dtype)

    latents_path = _Path(class_latents_path)
    if not latents_path.is_absolute():
        latents_path = (repo_root / latents_path).resolve()
    if not latents_path.is_file():
        raise FileNotFoundError(
            f"[saeuron] Class latents file not found: {latents_path}\n"
            "Generate it with:\n"
            "  cd baselines/saeuron\n"
            f"  python scripts/gather_sae_acts_ca_prompts_cls.py "
            f"--checkpoint_path sae-ckpts --hookpoint {hookpoint} "
            f"--pipe_path {spec.model_path} --save_dir <output_dir>"
        )

    with open(latents_path, "rb") as f:
        concept_latents_dict = pickle.load(f)

    if target_class not in concept_latents_dict:
        available = list(concept_latents_dict.keys())
        raise KeyError(f"[saeuron] target_class '{target_class}' not in latents file. Available: {available}")

    print(
        f"[saeuron] Loaded SAE from {sae_path}, hookpoint={hookpoint}, "
        f"target_class={target_class}, percentile={percentile}, multiplier={multiplier}"
    )
    return _SAEUronPipelineWrapper(model, hookpoint, sae, concept_latents_dict, target_class, percentile, multiplier)


def _load_model_runner(spec: ModelSpec, device: str):
    t = spec.type.lower()
    if t in {"diffusers", "stereo", "esd"}:
        return _load_diffusers_pipeline(spec, device)
    if t == "advunlearn":
        return _load_advunlearn_pipeline(spec, device)
    if t == "salun":
        return _load_salun_pipeline(spec, device)
    if t == "saeuron":
        return _load_saeuron_pipeline(spec, device)
    raise ValueError(f"Unsupported model type: {spec.type}")


def _generate_single_image(pipe, prompt: str, seed: int, cfg: GenerationConfig) -> Image.Image:
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "generator": generator,
    }
    if cfg.height is not None:
        kwargs["height"] = cfg.height
    if cfg.width is not None:
        kwargs["width"] = cfg.width
    out = pipe(**kwargs)
    return out.images[0]


def generate_images_for_model(
    prompts_df: pd.DataFrame,
    spec: ModelSpec,
    output_dir: str | Path,
    cfg: GenerationConfig,
    existing_metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out_root = ensure_dir(Path(output_dir).resolve())
    img_root = ensure_dir(out_root / "images" / spec.name)
    metadata_rows = []
    device = _resolve_device(cfg.device)

    existing_keys = set()
    if existing_metadata_df is not None and len(existing_metadata_df) > 0:
        cols = {"model_name", "prompt_id", "seed", "image_index", "image_path"}
        if cols.issubset(set(existing_metadata_df.columns)):
            for _, row in existing_metadata_df.iterrows():
                key = (
                    str(row["model_name"]),
                    str(row["prompt_id"]),
                    _coerce_seed(row["seed"]),
                    int(row["image_index"]),
                )
                if cfg.resume and _image_file_exists(str(row["image_path"]), out_root):
                    existing_keys.add(key)

    seeds = cfg.seeds if cfg.seeds is not None else [0]
    work_items = []
    for _, row in prompts_df.iterrows():
        for seed in seeds:
            for image_index in range(cfg.images_per_prompt):
                work_items.append((row, int(seed), int(image_index)))

    pipe = _load_model_runner(spec, device=device)

    for row, seed, image_index in tqdm(work_items, desc=f"Generating [{spec.name}]"):
        key = (spec.name, str(row["prompt_id"]), seed, image_index)
        target = str(row["target_concept"])
        family = str(row["prompt_family"])
        intended = str(row["intended_label"])
        prompt = str(row["prompt"])
        lexical_mode = str(row["lexical_mode"])
        domain = str(row["domain"])

        subdir = ensure_dir(img_root / _sanitize_name(family))
        filename = _deterministic_filename(
            spec.name,
            target,
            str(row["prompt_id"]),
            seed,
            image_index,
            cfg.image_format,
        )
        image_path = (subdir / filename).resolve()

        if key in existing_keys:
            metadata_rows.append(
                {
                    "image_path": str(image_path),
                    "model_name": spec.name,
                    "target_concept": target,
                    "prompt_family": family,
                    "intended_label": intended,
                    "domain": domain,
                    "prompt": prompt,
                    "prompt_id": str(row["prompt_id"]),
                    "seed": seed,
                    "image_index": image_index,
                    "guidance_scale": cfg.guidance_scale,
                    "num_inference_steps": cfg.num_inference_steps,
                    "lexical_mode": lexical_mode,
                    "status": "cached",
                }
            )
            continue

        if cfg.resume and image_path.is_file():
            status = "cached_file"
        else:
            local_seed = seed + image_index * 1_000_003
            image = _generate_single_image(pipe, prompt=prompt, seed=local_seed, cfg=cfg)
            image.save(image_path)
            status = "generated"

        metadata_rows.append(
            {
                "image_path": str(image_path),
                "model_name": spec.name,
                "target_concept": target,
                "prompt_family": family,
                "intended_label": intended,
                "domain": domain,
                "prompt": prompt,
                "prompt_id": str(row["prompt_id"]),
                "seed": seed,
                "image_index": image_index,
                "guidance_scale": cfg.guidance_scale,
                "num_inference_steps": cfg.num_inference_steps,
                "lexical_mode": lexical_mode,
                "status": status,
            }
        )

    return pd.DataFrame(metadata_rows)


def generate_all_images(
    prompts_df: pd.DataFrame,
    model_specs: dict[str, ModelSpec],
    output_dir: str | Path,
    cfg: GenerationConfig,
    generated_metadata_path: str | Path,
) -> pd.DataFrame:
    existing_df = pd.DataFrame()
    generated_metadata_path = Path(generated_metadata_path)
    if generated_metadata_path.exists():
        existing_df = pd.read_csv(generated_metadata_path)

    all_parts = []
    for spec in model_specs.values():
        part = generate_images_for_model(
            prompts_df=prompts_df,
            spec=spec,
            output_dir=output_dir,
            cfg=cfg,
            existing_metadata_df=existing_df,
        )
        all_parts.append(part)

    new_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    if len(existing_df) > 0:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["model_name", "prompt_id", "seed", "image_index"], keep="last")
    else:
        combined = new_df

    save_csv(generated_metadata_path, combined)
    return combined

