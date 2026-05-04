#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.abspath(str(REPO_ROOT)))

from baselines.seot.utils.wo_utils import woword_eot_context  # noqa: E402
from src.benchmark.generation import _load_model_runner  # noqa: E402
from src.benchmark.model_registry import ModelSpec, load_model_registry  # noqa: E402
from src.benchmark.prompt_loader import load_prompts  # noqa: E402

try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:  # pragma: no cover
    spearmanr = None


DEFAULT_CONCEPTS = ["dog", "bear", "horse", "cat", "castle"]
DEFAULT_METHODS = ["ESD", "SalUn", "EDiff", "AdvUnlearn", "STEREO", "SEOT", "SAeUron"]
DEFAULT_KAPPA_HAT = {
    "dog": 0.89,
    "bear": 0.82,
    "horse": 0.81,
    "cat": 0.77,
    "castle": 0.27,
}

METHOD_TO_PREFIX = {
    "esd": "esd",
    "salun": "salun",
    "ediff": "erasediff",
    "advunlearn": "advunlearn",
    "stereo": "stereo",
    "saeuron": "saeuron",
}


class _EarlyCapture(Exception):
    pass


@dataclass
class SeotConfig:
    target_text: str
    method: str
    alpha: float
    cross_retain_steps: float


def _normalize_method_name(name: str) -> str:
    k = name.strip().lower()
    aliases = {
        "esd": "esd",
        "salun": "salun",
        "ediff": "ediff",
        "erasediff": "ediff",
        "advunlearn": "advunlearn",
        "stereo": "stereo",
        "seot": "seot",
        "saeuron": "saeuron",
    }
    if k not in aliases:
        raise ValueError(f"Unsupported method: {name}")
    return aliases[k]


def _pretty_method_name(norm: str) -> str:
    pretty = {
        "esd": "ESD",
        "salun": "SalUn",
        "ediff": "EDiff",
        "advunlearn": "AdvUnlearn",
        "stereo": "STEREO",
        "seot": "SEOT",
        "saeuron": "SAeUron",
    }
    return pretty[norm]


def _find_target_attn2(unet: torch.nn.Module, target_layer: str) -> tuple[str, torch.nn.Module]:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if target_layer in name and "attn2" in name:
            candidates.append((name, module))
    exact = [x for x in candidates if x[0].endswith("attn2")]
    if exact:
        return exact[0]
    if candidates:
        return candidates[0]
    raise RuntimeError(f"Could not locate attn2 module under '{target_layer}'")


def _find_token_indices(tokenizer, prompt: str, target_text: str) -> list[int]:
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target_text)[1:-1]
    if not target_ids:
        return []
    max_i = len(prompt_ids) - len(target_ids) + 1
    for i in range(max_i):
        if prompt_ids[i : i + len(target_ids)] == target_ids:
            return list(range(i, i + len(target_ids)))
    return []


def _prepare_embeddings(
    pipe: StableDiffusionPipeline,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        cond = text_encoder(text_input.input_ids)[0]
        uncond = text_encoder(uncond_input.input_ids)[0]
    return cond, uncond, text_input.input_ids


def _extract_activation_manual(
    pipe: StableDiffusionPipeline,
    target_module: torch.nn.Module,
    prompt: str,
    seed: int,
    capture_step: int,
    num_ddim_steps: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    seot_cfg: SeotConfig | None = None,
) -> torch.Tensor:
    unet = pipe.unet
    scheduler = pipe.scheduler

    cond, uncond, input_ids = _prepare_embeddings(pipe, prompt, device)

    if seot_cfg is not None:
        token_indices = _find_token_indices(pipe.tokenizer, prompt, seot_cfg.target_text)
        if token_indices:
            activate_from_step = int(round(seot_cfg.cross_retain_steps * num_ddim_steps))
        else:
            activate_from_step = num_ddim_steps + 1
    else:
        token_indices = []
        activate_from_step = num_ddim_steps + 1

    generator = torch.Generator(device=device.type).manual_seed(seed)
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    scheduler.set_timesteps(num_ddim_steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    cache: dict[str, torch.Tensor] = {}

    def _hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        out = output[0] if isinstance(output, tuple) else output
        if out.shape[0] < 2:
            raise RuntimeError(f"Expected CFG batch >= 2, got {tuple(out.shape)}")
        cond_out = out[1]
        cache["vec"] = cond_out.reshape(-1).detach().cpu().float()
        return output

    for step_idx, t in enumerate(scheduler.timesteps):
        cond_step = cond
        if seot_cfg is not None and token_indices and step_idx >= activate_from_step:
            cond_step = woword_eot_context(
                cond.clone(),
                token_indices=token_indices,
                alpha=seot_cfg.alpha,
                method=seot_cfg.method,
                n=input_ids.shape[-1],
            )
        text_embeddings = torch.cat([uncond, cond_step], dim=0)

        handle = None
        if step_idx == capture_step:
            handle = target_module.register_forward_hook(_hook)

        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

        if handle is not None:
            handle.remove()
            break

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if "vec" not in cache:
        raise RuntimeError(f"Failed to capture activation at step={capture_step} for prompt={prompt!r}")
    return cache["vec"]


def _extract_activation_saeuron(
    wrapped_model: Any,
    prompt: str,
    seed: int,
    capture_step: int,
    num_ddim_steps: int,
    guidance_scale: float,
    hookpoint: str,
) -> torch.Tensor:
    import sys as _sys

    saeuron_root = REPO_ROOT / "baselines" / "saeuron"
    if str(saeuron_root) not in _sys.path:
        _sys.path.insert(0, str(saeuron_root))
    from SAE.unlearning_utils import compute_feature_importance  # noqa: PLC0415
    from utils.hooks import SAEMaskedUnlearningHook  # noqa: PLC0415

    model = wrapped_model._model
    # Keep parity with generation.py wrapper: create a fresh hook each run.
    ablation_hook = SAEMaskedUnlearningHook(
        concept_to_unlearn=[wrapped_model._target_class],
        percentile=wrapped_model._percentile,
        multiplier=wrapped_model._multiplier,
        feature_importance_fn=compute_feature_importance,
        concept_latents_dict=wrapped_model._concept_latents_dict,
        sae=wrapped_model._sae,
        steps=num_ddim_steps,
        preserve_error=True,
    )
    cache: dict[str, torch.Tensor] = {}
    step_counter = {"idx": 0}

    def capture_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        out = output[0] if isinstance(output, tuple) else output
        cur = step_counter["idx"]
        if cur == capture_step:
            cond_out = out[1]
            cache["vec"] = cond_out.reshape(-1).detach().cpu().float()
            raise _EarlyCapture()
        step_counter["idx"] = cur + 1
        return output

    device = next(model.pipe.unet.parameters()).device
    generator = torch.Generator(device=device.type).manual_seed(seed)
    try:
        model.run_with_hooks(
            prompt=prompt,
            num_inference_steps=num_ddim_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            position_hook_dict={hookpoint: [ablation_hook, capture_hook]},
            output_type="latent",
        )
    except _EarlyCapture:
        pass

    if "vec" not in cache:
        raise RuntimeError(f"SAeUron activation capture failed at step={capture_step} for prompt={prompt!r}")
    return cache["vec"]


def _load_neighbor_prompts(
    concept: str,
    prompt_root: Path,
    n_prompts: int,
    seed: int,
) -> list[str]:
    prompt_path = prompt_root / concept / "all_prompts.csv"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    df = load_prompts(prompt_path)
    neigh = df[df["prompt_family"].astype(str).str.lower() == "neighbor"].copy()
    if len(neigh) < n_prompts:
        raise RuntimeError(f"{concept}: only {len(neigh)} neighbor prompts available; need {n_prompts}")
    # Match NP pipeline behavior: fixed seed, deterministic subset.
    neigh = neigh.sample(n=n_prompts, random_state=seed).sort_values("prompt_id")
    return neigh["prompt"].astype(str).tolist()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if spearmanr is not None:
        return float(spearmanr(x, y).statistic)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c)


def _load_ua_summary(root: Path) -> pd.DataFrame:
    files = list(root.glob("**/aggregated_clip_metrics.csv"))
    if not files:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if "UA" not in df.columns:
            continue
        df["__source"] = str(fp)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure neighbor-prompt activation displacement under unlearning.")
    parser.add_argument("--base_model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--model_registry_path", type=str, default="configs/benchmark_models.example.json")
    parser.add_argument("--prompt_root", type=str, default="outputs/prompts")
    parser.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--n_prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--target_layer", type=str, default="up_blocks.1.attentions.1")
    parser.add_argument("--capture_step", type=int, default=25)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--output_csv", type=str, default="results/activation_displacement.csv")
    parser.add_argument("--ua_search_root", type=str, default="outputs")
    parser.add_argument("--sanity_concept", type=str, default="dog")
    parser.add_argument("--sanity_n_prompts", type=int, default=5)
    # SEOT controls (inference-time embedding edit)
    parser.add_argument("--seot_method", type=str, default="soft-weight")
    parser.add_argument("--seot_alpha", type=float, default=1.0)
    parser.add_argument("--seot_cross_retain_steps", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if not (0 <= args.capture_step < args.num_ddim_steps):
        raise ValueError("capture_step must be in [0, num_ddim_steps)")

    concepts = [str(c).strip().lower() for c in args.concepts]
    methods_norm = [_normalize_method_name(m) for m in args.methods]
    prompt_root = Path(args.prompt_root)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load benchmark-model registry checkpoints.
    registry = load_model_registry(args.model_registry_path)

    # Reuse NP prompt infra and subset to same deterministic 200-neighbor set.
    prompts_by_concept: dict[str, list[str]] = {}
    for c in concepts:
        prompts_by_concept[c] = _load_neighbor_prompts(
            concept=c,
            prompt_root=prompt_root,
            n_prompts=args.n_prompts,
            seed=args.seed,
        )

    print(f"[info] Loading base model: {args.base_model_id}")
    base_pipe = StableDiffusionPipeline.from_pretrained(args.base_model_id, torch_dtype=dtype).to(device)
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
    base_pipe.set_progress_bar_config(disable=True)
    _, base_target_module = _find_target_attn2(base_pipe.unet, args.target_layer)

    rows: list[dict[str, Any]] = []

    # Sanity check: base vs base should be near zero.
    sanity_concept = args.sanity_concept.strip().lower()
    if sanity_concept in prompts_by_concept:
        sanity_vals: list[float] = []
        sanity_prompts = prompts_by_concept[sanity_concept][: args.sanity_n_prompts]
        for i, p in enumerate(sanity_prompts):
            seed_i = args.seed + i
            a = _extract_activation_manual(
                pipe=base_pipe,
                target_module=base_target_module,
                prompt=p,
                seed=seed_i,
                capture_step=args.capture_step,
                num_ddim_steps=args.num_ddim_steps,
                guidance_scale=args.guidance_scale,
                device=device,
                dtype=dtype,
            )
            b = _extract_activation_manual(
                pipe=base_pipe,
                target_module=base_target_module,
                prompt=p,
                seed=seed_i,
                capture_step=args.capture_step,
                num_ddim_steps=args.num_ddim_steps,
                guidance_scale=args.guidance_scale,
                device=device,
                dtype=dtype,
            )
            sanity_vals.append(float(torch.linalg.vector_norm(a - b).item()))
        print(
            "[sanity] base self-consistency "
            f"({sanity_concept}, n={len(sanity_vals)}): "
            f"mean={np.mean(sanity_vals):.6e}, max={np.max(sanity_vals):.6e}"
        )

    for concept in concepts:
        print(f"\n[concept] {concept}")
        prompts = prompts_by_concept[concept]

        # Cache original activations once per concept.
        orig_cache: list[torch.Tensor] = []
        for i, prompt in enumerate(prompts):
            seed_i = args.seed + i
            orig = _extract_activation_manual(
                pipe=base_pipe,
                target_module=base_target_module,
                prompt=prompt,
                seed=seed_i,
                capture_step=args.capture_step,
                num_ddim_steps=args.num_ddim_steps,
                guidance_scale=args.guidance_scale,
                device=device,
                dtype=dtype,
            )
            orig_cache.append(orig)

        for method_norm in methods_norm:
            pretty_method = _pretty_method_name(method_norm)
            if method_norm == "saeuron" and concept == "castle":
                print("[skip] SAeUron castle unavailable; skipping")
                continue

            print(f"[method] {pretty_method} ({concept})")

            if method_norm == "seot":
                method_pipe: Any = base_pipe
                _, method_target_module = _find_target_attn2(method_pipe.unet, args.target_layer)
                seot_cfg = SeotConfig(
                    target_text=concept,
                    method=args.seot_method,
                    alpha=float(args.seot_alpha),
                    cross_retain_steps=float(args.seot_cross_retain_steps),
                )
                is_saeuron = False
            else:
                prefix = METHOD_TO_PREFIX[method_norm]
                model_key = f"{prefix}_{concept}"
                if model_key not in registry:
                    raise KeyError(f"Missing model key in registry: {model_key}")
                spec: ModelSpec = registry[model_key]
                method_pipe = _load_model_runner(spec, device=str(device))
                if hasattr(method_pipe, "scheduler") and method_pipe.scheduler is not None:
                    method_pipe.scheduler = DDIMScheduler.from_config(method_pipe.scheduler.config)
                if hasattr(method_pipe, "to"):
                    method_pipe = method_pipe.to(device)
                if hasattr(method_pipe, "set_progress_bar_config"):
                    method_pipe.set_progress_bar_config(disable=True)
                seot_cfg = None
                is_saeuron = method_norm == "saeuron"
                if is_saeuron:
                    hookpoint = str(spec.extra_args.get("hookpoint", "unet.up_blocks.1.attentions.1"))
                else:
                    _, method_target_module = _find_target_attn2(method_pipe.unet, args.target_layer)

            displacements: list[float] = []
            for i, prompt in enumerate(prompts):
                seed_i = args.seed + i
                if is_saeuron:
                    unlearned = _extract_activation_saeuron(
                        wrapped_model=method_pipe,
                        prompt=prompt,
                        seed=seed_i,
                        capture_step=args.capture_step,
                        num_ddim_steps=args.num_ddim_steps,
                        guidance_scale=args.guidance_scale,
                        hookpoint=hookpoint,
                    )
                else:
                    unlearned = _extract_activation_manual(
                        pipe=method_pipe,
                        target_module=method_target_module,
                        prompt=prompt,
                        seed=seed_i,
                        capture_step=args.capture_step,
                        num_ddim_steps=args.num_ddim_steps,
                        guidance_scale=args.guidance_scale,
                        device=device,
                        dtype=dtype,
                        seot_cfg=seot_cfg,
                    )
                d = float(torch.linalg.vector_norm(unlearned - orig_cache[i]).item())
                displacements.append(d)

            arr = np.asarray(displacements, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"Non-finite displacements detected for {concept}/{pretty_method}")
            if np.any(arr <= 0.0):
                print(f"[warn] non-positive displacements for {concept}/{pretty_method}: {(arr <= 0.0).sum()} rows")

            rows.append(
                {
                    "concept": concept,
                    "method": pretty_method,
                    "kappa_hat": float(DEFAULT_KAPPA_HAT[concept]),
                    "mean_displacement": float(arr.mean()),
                    "median_displacement": float(np.median(arr)),
                    "std_displacement": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n_prompts": int(len(arr)),
                }
            )

            # Free GPU memory for loaded per-method pipeline.
            if method_norm not in {"seot"}:
                try:
                    del method_pipe
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No displacement rows produced.")
    df = df.sort_values(["method", "concept"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print("\n=== Spearman rho (kappa_hat vs mean_displacement), per method ===")
    for method in sorted(df["method"].unique().tolist()):
        sub = df[df["method"] == method].copy()
        if len(sub) < 2:
            rho = float("nan")
        else:
            rho = _spearman(
                sub["kappa_hat"].to_numpy(dtype=np.float64),
                sub["mean_displacement"].to_numpy(dtype=np.float64),
            )
        print(f"{method:10s} rho={rho:.4f}")

    stereo = df[df["method"] == "STEREO"].copy()
    if len(stereo) > 0:
        stereo = stereo.sort_values("kappa_hat")
        print("\n=== STEREO mean_displacement by ascending kappa_hat ===")
        for _, r in stereo.iterrows():
            print(f"{r['concept']:8s} kappa={r['kappa_hat']:.2f} mean_disp={r['mean_displacement']:.6f}")

    # Global positivity/finite verification summary.
    all_vals = df["mean_displacement"].to_numpy(dtype=np.float64)
    print(
        "\n[verify] displacement aggregates "
        f"finite={bool(np.all(np.isfinite(all_vals)))}, strictly_positive={bool(np.all(all_vals > 0.0))}"
    )

    # Optional UA cross-check if prior benchmark CSVs exist.
    ua_df = _load_ua_summary(Path(args.ua_search_root))
    if len(ua_df) == 0:
        print("[verify] UA cross-check skipped: no aggregated_clip_metrics.csv found under outputs/")
    else:
        cols = [c for c in ["model_name", "UA", "__source"] if c in ua_df.columns]
        print(f"[verify] UA cross-check source rows: {len(ua_df)}")
        print(ua_df[cols].head(10).to_string(index=False))

    print(f"\n[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
