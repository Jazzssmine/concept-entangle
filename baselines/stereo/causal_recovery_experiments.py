#!/usr/bin/env python3
"""
Causal recovery experiments for STEREO unlearning (no probing).

This script runs two intervention experiments on UNet internal activations:

1) Activation steering (erased model):
   h <- h + alpha * v
   where v is a precomputed direction (from base or erased model).

2) Activation transplant (base -> erased):
   h_erased <- (1 - beta) * h_erased + beta * h_base
   with synchronized prompt/seed/latent/scheduler.

Why this matters:
- If horse-like generations can be restored by tiny internal interventions, then
  horse-sensitive causal information still survives in erased-model internals.
- Comparing base-direction vs erased-direction steering helps test whether the
  residual erased direction is especially aligned with recoverable signal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageOps, ImageDraw


# ----------------------------- Model loading ----------------------------- #


def load_base_pipeline(base_model_path: str, device: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.text_encoder.eval()
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    return pipe


def load_erased_pipeline(
    base_model_path: str, erased_unet_checkpoint: str, device: str
) -> StableDiffusionPipeline:
    pipe = load_base_pipeline(base_model_path, device)
    state = torch.load(erased_unet_checkpoint, map_location=device)
    pipe.unet.load_state_dict(state)
    pipe.unet.eval()
    return pipe


def _extract_tokens_from_prompts(prompts: list[str]) -> list[str]:
    pat = re.compile(r"\btoken_\w+\b")
    out: dict[str, None] = {}
    for p in prompts:
        for t in pat.findall(p):
            out[t] = None
    return list(out.keys())


def _prepare_attacked_text_encoder_and_tokenizer(
    pipe: StableDiffusionPipeline,
    attacked_text_encoder_path: str,
    tokenizer_path: str | None,
    placeholder_tokens: list[str],
    device: str,
) -> None:
    if tokenizer_path:
        pipe.tokenizer = pipe.tokenizer.__class__.from_pretrained(tokenizer_path)

    state_dict = torch.load(attacked_text_encoder_path, map_location=device)
    key = "text_model.embeddings.token_embedding.weight"
    expected_vocab = state_dict[key].shape[0]
    to_add = expected_vocab - len(pipe.tokenizer)
    if to_add < 0:
        raise RuntimeError(
            f"Checkpoint expects vocab {expected_vocab}, tokenizer has {len(pipe.tokenizer)}."
        )

    added = 0
    for tok in placeholder_tokens:
        if added >= to_add:
            break
        if tok not in pipe.tokenizer.get_vocab():
            pipe.tokenizer.add_tokens([tok])
            added += 1
    while len(pipe.tokenizer) < expected_vocab:
        pipe.tokenizer.add_tokens([f"__pad_token_{len(pipe.tokenizer)}__"])

    pipe.text_encoder.resize_token_embeddings(expected_vocab)
    pipe.text_encoder.load_state_dict(state_dict)
    pipe.text_encoder.eval()
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)


# ----------------------------- Prompt config ----------------------------- #


def load_prompt_config(path: str) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for g in ["A", "B", "D", "E", "F"]:
        if g not in data:
            raise ValueError(f"Prompt config missing group '{g}'")
    if "C" not in data:
        data["C"] = []
    return data


def build_prompt_table(
    prompt_groups: dict[str, list[str]],
    enable_s2_tokens: bool,
    debug_mode: bool,
    debug_n_per_pos_group: int,
    debug_n_controls: int,
) -> pd.DataFrame:
    rows = []
    groups = ["A", "B", "D", "E", "F"] + (["C"] if enable_s2_tokens else [])
    for g in groups:
        for p in prompt_groups.get(g, []):
            rows.append({"prompt_group": g, "prompt": p})
    df = pd.DataFrame(rows).drop_duplicates()

    if not debug_mode:
        return df.reset_index(drop=True)

    keep = []
    for g in ["A", "B"]:
        sub = df[df["prompt_group"] == g].head(debug_n_per_pos_group)
        keep.append(sub)
    controls = df[df["prompt_group"].isin(["D", "E", "F"])].head(debug_n_controls)
    keep.append(controls)
    if enable_s2_tokens:
        keep.append(df[df["prompt_group"] == "C"].head(debug_n_per_pos_group))
    return pd.concat(keep, axis=0).drop_duplicates().reset_index(drop=True)


# ----------------------------- Hooks and alignment ----------------------------- #


def _extract_tensor_from_output(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "sample") and torch.is_tensor(output.sample):
        return output.sample
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    return None


def _replace_tensor_in_output(output: Any, new_tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return new_tensor
    if hasattr(output, "sample") and torch.is_tensor(output.sample):
        output.sample = new_tensor
        return output
    if isinstance(output, tuple):
        out = list(output)
        for i, x in enumerate(out):
            if torch.is_tensor(x):
                out[i] = new_tensor
                return tuple(out)
    if isinstance(output, list):
        out = list(output)
        for i, x in enumerate(out):
            if torch.is_tensor(x):
                out[i] = new_tensor
                return out
    return output


def resolve_target_layers(unet: torch.nn.Module, patterns: list[str]) -> list[str]:
    names = [n for n, _ in unet.named_modules()]
    out = []
    for pat in patterns:
        if pat in names:
            out.append(pat)
            continue
        pref = [n for n in names if n.startswith(pat)]
        if pref:
            out.append(pref[0])
            continue
        sub = [n for n in names if pat in n]
        if sub:
            out.append(sub[0])
            continue
        print(f"[warn] could not match layer pattern: {pat}")
    seen = set()
    dedup = []
    for n in out:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup


def _align_direction_vector(vec: torch.Tensor, act: torch.Tensor, verbose: bool = True) -> torch.Tensor:
    """
    Align 1D direction vector to activation feature dimension.
    Target feature dim:
      - [B,C,H,W] -> C
      - [B,T,D]   -> D
      - [B,D]     -> D
    If mismatch, apply explicit truncate/pad projection and log it.
    """
    if vec.ndim != 1:
        vec = vec.reshape(-1)

    if act.ndim == 4:
        target_dim = act.shape[1]
    elif act.ndim in (2, 3):
        target_dim = act.shape[-1]
    else:
        target_dim = act.reshape(act.shape[0], -1).shape[-1]

    src_dim = vec.shape[0]
    if src_dim != target_dim:
        if verbose:
            print(f"[align] direction dim {src_dim} -> target dim {target_dim} via truncate/pad")
        if src_dim > target_dim:
            vec = vec[:target_dim]
        else:
            pad = torch.zeros(target_dim - src_dim, dtype=vec.dtype, device=vec.device)
            vec = torch.cat([vec, pad], dim=0)

    if act.ndim == 4:
        return vec.view(1, target_dim, 1, 1)
    if act.ndim == 3:
        return vec.view(1, 1, target_dim)
    if act.ndim == 2:
        return vec.view(1, target_dim)
    # fallback: broadcast over all non-batch dims
    return vec.view(1, *([1] * (act.ndim - 2)), target_dim)


class SteeringIntervention:
    """
    Applies h <- h + alpha * v at chosen layers + denoising step indices.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        hookpoints: list[str],
        target_step_indices: set[int],
        direction_vec: torch.Tensor,
        alpha: float,
        debug: bool = False,
    ):
        self.unet = unet
        self.hookpoints = hookpoints
        self.target_steps = target_step_indices
        self.direction_vec = direction_vec
        self.alpha = alpha
        self.debug = debug
        self.current_step = -1
        self.current_scheduler_t = -1
        self.handles: list[Any] = []
        self.fired_events: list[dict[str, Any]] = []
        self.step_hits: set[str] = set()
        self.action_name = "steer"

    def set_step(self, step: int, scheduler_t: int) -> None:
        self.current_step = step
        self.current_scheduler_t = scheduler_t
        self.step_hits.clear()

    def _hook(self, layer_name: str):
        def fn(_module, _input, output):
            if self.current_step not in self.target_steps:
                return output
            act = _extract_tensor_from_output(output)
            if act is None:
                return output
            # IMPORTANT (CFG): UNet batch is typically [uncond, cond]. Steering both
            # halves equally can cancel in classifier-free guidance. We therefore steer
            # only the conditional half when batch>=2.
            if act.shape[0] >= 2:
                cond_start = act.shape[0] // 2
                cond_act = act[cond_start:]
                dir_bc = _align_direction_vector(
                    self.direction_vec.to(cond_act.device, cond_act.dtype),
                    cond_act,
                    verbose=self.debug,
                )
                pre = float(cond_act.norm().item())
                steer_norm = float((self.alpha * dir_bc).norm().item())
                new_act = act.clone()
                new_act[cond_start:] = cond_act + self.alpha * dir_bc
                post = float(new_act[cond_start:].norm().item())
            else:
                dir_bc = _align_direction_vector(
                    self.direction_vec.to(act.device, act.dtype), act, verbose=self.debug
                )
                pre = float(act.norm().item())
                steer_norm = float((self.alpha * dir_bc).norm().item())
                new_act = act + self.alpha * dir_bc
                post = float(new_act.norm().item())
            if self.debug:
                self.step_hits.add(layer_name)
                print(
                    f"steer @ layer={layer_name}, loop_idx={self.current_step}, "
                    f"t={self.current_scheduler_t}"
                )
                self.fired_events.append(
                    {
                        "layer": layer_name,
                        "step_idx": int(self.current_step),
                        "scheduler_t": int(self.current_scheduler_t),
                        "act_norm_before": pre,
                        "act_norm_after": post,
                        "steering_norm": steer_norm,
                    }
                )
            return _replace_tensor_in_output(output, new_act)

        return fn

    def register(self) -> None:
        modules = dict(self.unet.named_modules())
        for name in self.hookpoints:
            if name in modules:
                self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


class BaseActivationCache:
    """Caches base activations at selected layer + step, keyed by (layer, step_idx)."""

    def __init__(self, unet: torch.nn.Module, hookpoints: list[str], target_step_indices: set[int]):
        self.unet = unet
        self.hookpoints = hookpoints
        self.target_steps = target_step_indices
        self.current_step = -1
        self.current_scheduler_t = -1
        self.cache: dict[tuple[str, int], torch.Tensor] = {}
        self.handles: list[Any] = []
        self.step_hits: set[str] = set()
        self.debug = False
        self.action_name = "capture"

    def set_step(self, step: int, scheduler_t: int) -> None:
        self.current_step = step
        self.current_scheduler_t = scheduler_t
        self.step_hits.clear()

    def _hook(self, layer_name: str):
        def fn(_module, _input, output):
            if self.current_step not in self.target_steps:
                return output
            act = _extract_tensor_from_output(output)
            if act is not None:
                self.cache[(layer_name, self.current_step)] = act.detach().clone()
                self.step_hits.add(layer_name)
                if self.debug:
                    print(
                        f"capture @ layer={layer_name}, loop_idx={self.current_step}, "
                        f"t={self.current_scheduler_t}"
                    )
            return output

        return fn

    def register(self) -> None:
        modules = dict(self.unet.named_modules())
        for name in self.hookpoints:
            if name in modules:
                self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


class TransplantIntervention:
    """Applies h <- (1-beta) h + beta h_base at selected layer+step."""

    def __init__(
        self,
        unet: torch.nn.Module,
        hookpoints: list[str],
        target_step_indices: set[int],
        base_cache: dict[tuple[str, int], torch.Tensor],
        beta: float,
        debug: bool = False,
    ):
        self.unet = unet
        self.hookpoints = hookpoints
        self.target_steps = target_step_indices
        self.base_cache = base_cache
        self.beta = beta
        self.debug = debug
        self.current_step = -1
        self.current_scheduler_t = -1
        self.handles: list[Any] = []
        self.fired_events: list[dict[str, Any]] = []
        self.step_hits: set[str] = set()
        self.action_name = "transplant"

    def set_step(self, step: int, scheduler_t: int) -> None:
        self.current_step = step
        self.current_scheduler_t = scheduler_t
        self.step_hits.clear()

    def _hook(self, layer_name: str):
        def fn(_module, _input, output):
            if self.current_step not in self.target_steps:
                return output
            act = _extract_tensor_from_output(output)
            if act is None:
                return output
            key = (layer_name, self.current_step)
            if key not in self.base_cache:
                return output
            base_act = self.base_cache[key].to(act.device, act.dtype)
            if base_act.shape != act.shape:
                print(f"[warn] transplant shape mismatch at {key}: {base_act.shape} vs {act.shape}; skipping")
                return output
            new_act = (1.0 - self.beta) * act + self.beta * base_act
            self.step_hits.add(layer_name)

            act_norm_before = float(act.norm().item())
            base_norm = float(base_act.norm().item())
            act_norm_after = float(new_act.norm().item())
            diff_norm = float((base_act - act).norm().item())
            # Flatten per-sample tensors to estimate angular alignment.
            act_flat = act.reshape(-1).float()
            base_flat = base_act.reshape(-1).float()
            cosine_base_vs_erased = float(
                torch.nn.functional.cosine_similarity(
                    act_flat.unsqueeze(0), base_flat.unsqueeze(0), dim=1
                ).item()
            )

            self.fired_events.append(
                {
                    "layer": layer_name,
                    "step_idx": int(self.current_step),
                    "scheduler_t": int(self.current_scheduler_t),
                    "beta": float(self.beta),
                    "act_norm_before": act_norm_before,
                    "base_norm": base_norm,
                    "act_norm_after": act_norm_after,
                    "diff_norm": diff_norm,
                    "cosine_base_vs_erased": cosine_base_vs_erased,
                }
            )

            if self.debug:
                print(
                    f"transplant @ layer={layer_name}, loop_idx={self.current_step}, "
                    f"t={self.current_scheduler_t}"
                )
                print(
                    "  "
                    f"||h_erased||={act_norm_before:.4f}, "
                    f"||h_base||={base_norm:.4f}, "
                    f"||h_base-h_erased||={diff_norm:.4f}, "
                    f"cos(h_erased,h_base)={cosine_base_vs_erased:.4f}"
                )
            return _replace_tensor_in_output(output, new_act)

        return fn

    def register(self) -> None:
        modules = dict(self.unet.named_modules())
        for name in self.hookpoints:
            if name in modules:
                self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ----------------------------- Diffusion runner ----------------------------- #


def _get_text_embeddings(pipe: StableDiffusionPipeline, prompt: str, device: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    untok = pipe.tokenizer(
        [""],
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        cond = pipe.text_encoder(tok.input_ids)[0]
        uncond = pipe.text_encoder(untok.input_ids)[0]
    return torch.cat([uncond, cond], dim=0)


def init_latent(unet: torch.nn.Module, image_size: int, seed: int, device: str, init_noise_sigma: float) -> torch.Tensor:
    gen_device = "cuda" if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    lat = torch.randn(
        (1, unet.in_channels, image_size // 8, image_size // 8),
        generator=generator,
        device=device,
        dtype=unet.dtype,
    )
    return lat * init_noise_sigma


def _decode_latent_to_pil(pipe: StableDiffusionPipeline, latent: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        img = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).round().astype(np.uint8)
    return Image.fromarray(img)


def run_generation(
    pipe: StableDiffusionPipeline,
    prompt: str,
    init_latent_tensor: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
    step_hook_state: Any | None = None,
    debug_loop_logs: bool = False,
    run_label: str = "",
) -> Image.Image:
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    lat = init_latent_tensor.clone()
    text_emb = _get_text_embeddings(pipe, prompt, device)

    for step_idx, timestep in enumerate(pipe.scheduler.timesteps):
        sched_t = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
        if debug_loop_logs:
            prefix = f"[{run_label}] " if run_label else ""
            print(f"{prefix}loop_idx={step_idx}, scheduler_t={sched_t}")
        if step_hook_state is not None:
            step_hook_state.set_step(step_idx, sched_t)
        model_input = torch.cat([lat] * 2, dim=0)
        model_input = pipe.scheduler.scale_model_input(model_input, timestep)
        with torch.no_grad():
            noise_pred = pipe.unet(model_input, timestep, encoder_hidden_states=text_emb).sample
        if debug_loop_logs:
            if step_hook_state is not None:
                for ln in getattr(step_hook_state, "hookpoints", []):
                    happened = ln in getattr(step_hook_state, "step_hits", set())
                    print(f"  layer={ln}, {getattr(step_hook_state, 'action_name', 'hook')}={happened}")
            else:
                print("  layer=<none>, action=none")
        eps_u, eps_c = noise_pred.chunk(2)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        lat = pipe.scheduler.step(eps, timestep, lat).prev_sample

    return _decode_latent_to_pil(pipe, lat)


# ----------------------------- Directions / logging ----------------------------- #


def load_direction_vector(path: str, prefer_unit: bool = True) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        vec = obj
    elif isinstance(obj, dict):
        if prefer_unit and "direction_unit" in obj:
            vec = obj["direction_unit"]
        elif "direction_raw" in obj:
            vec = obj["direction_raw"]
        elif "vector" in obj:
            vec = obj["vector"]
        else:
            raise ValueError(f"Could not find direction tensor in {path}")
    else:
        raise ValueError(f"Unsupported direction file type: {type(obj)}")
    return vec.float().reshape(-1)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_contact_sheet(
    summary_df: pd.DataFrame,
    out_path: Path,
    settings_col: str,
    image_path_col: str = "image_path",
    cell_size: tuple[int, int] = (256, 256),
) -> None:
    if summary_df.empty:
        return
    seeds = sorted(summary_df["seed"].unique().tolist())
    settings = summary_df[settings_col].drop_duplicates().tolist()
    W, H = cell_size
    header_h = 28
    sheet = Image.new("RGB", (W * (len(settings) + 1), header_h + H * (len(seeds) + 1)), "white")
    draw = ImageDraw.Draw(sheet)

    for j, s in enumerate(["seed\\setting"] + settings):
        draw.text((j * W + 5, 6), str(s)[:40], fill="black")
    for i, sd in enumerate(seeds):
        draw.text((5, header_h + i * H + 5), f"seed {sd}", fill="black")

    for i, sd in enumerate(seeds):
        row = summary_df[summary_df["seed"] == sd]
        for j, s in enumerate(settings):
            cell = row[row[settings_col] == s]
            if cell.empty:
                continue
            imgp = cell.iloc[0][image_path_col]
            if not os.path.isfile(imgp):
                continue
            img = Image.open(imgp).convert("RGB")
            img = ImageOps.fit(img, cell_size)
            sheet.paste(img, ((j + 1) * W, header_h + i * H))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


# ----------------------------- Experiments ----------------------------- #


def run_activation_steering(
    erased_pipe: StableDiffusionPipeline,
    prompt_df: pd.DataFrame,
    seeds: list[int],
    hookpoints: list[str],
    target_steps: list[int],
    alphas: list[float],
    direction_sources: dict[str, torch.Tensor],
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    steer_root = out_dir / "steering"
    steer_root.mkdir(parents=True, exist_ok=True)
    target_step_set = set(target_steps)

    for _, row in prompt_df.iterrows():
        prompt = row["prompt"]
        group = row["prompt_group"]
        prompt_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", prompt)[:120]
        for seed in seeds:
            init_lat = init_latent(
                erased_pipe.unet,
                args.image_size,
                seed,
                args.device,
                erased_pipe.scheduler.init_noise_sigma,
            )
            for source_name, direction in direction_sources.items():
                for alpha in alphas:
                    intervention = SteeringIntervention(
                        erased_pipe.unet,
                        hookpoints=hookpoints,
                        target_step_indices=target_step_set,
                        direction_vec=direction,
                        alpha=float(alpha),
                        debug=args.debug_hook_logs,
                    )
                    intervention.register()
                    try:
                        img = run_generation(
                            erased_pipe,
                            prompt,
                            init_lat,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            device=args.device,
                            step_hook_state=intervention,
                            debug_loop_logs=args.debug_hook_logs,
                            run_label=f"steer:{source_name}:a={alpha}",
                        )
                    finally:
                        intervention.remove()

                    cdir = steer_root / f"group_{group}" / f"prompt_{prompt_slug}" / f"seed_{seed}"
                    cdir.mkdir(parents=True, exist_ok=True)
                    setting = f"{source_name}_alpha_{alpha}"
                    img_path = cdir / f"{setting}.png"
                    img.save(img_path)

                    meta = {
                        "model_name": "erased",
                        "direction_source": source_name,
                        "hookpoint": hookpoints,
                        "timesteps_intervened": target_steps,
                        "alpha": float(alpha),
                        "prompt": prompt,
                        "prompt_group": group,
                        "seed": int(seed),
                        "image_path": str(img_path),
                        "debug_events": intervention.fired_events if args.debug_hook_logs else [],
                    }
                    write_json(cdir / f"{setting}.json", meta)

                    rows.append(
                        {
                            "experiment_type": "steer",
                            "prompt_group": group,
                            "prompt": prompt,
                            "seed": seed,
                            "hookpoint": ",".join(hookpoints),
                            "timestep_or_timesteps": ",".join(str(t) for t in target_steps),
                            "alpha_or_beta": float(alpha),
                            "direction_source": source_name,
                            "image_path": str(img_path),
                            "setting": setting,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "steering_summary.csv", index=False)

    # Contact sheets by prompt
    for prompt in df["prompt"].drop_duplicates().tolist():
        sub = df[df["prompt"] == prompt]
        if sub.empty:
            continue
        pslug = re.sub(r"[^a-zA-Z0-9_]+", "_", prompt)[:120]
        make_contact_sheet(sub, out_dir / "steering" / "contact_sheets" / f"{pslug}.png", settings_col="setting")

    return df


def run_activation_transplant(
    base_pipe: StableDiffusionPipeline,
    erased_pipe: StableDiffusionPipeline,
    prompt_df: pd.DataFrame,
    seeds: list[int],
    hookpoints: list[str],
    target_steps: list[int],
    betas: list[float],
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    diff_rows = []
    trans_root = out_dir / "transplant"
    trans_root.mkdir(parents=True, exist_ok=True)
    target_step_set = set(target_steps)

    for _, row in prompt_df.iterrows():
        prompt = row["prompt"]
        group = row["prompt_group"]
        prompt_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", prompt)[:120]
        for seed in seeds:
            init_lat = init_latent(
                erased_pipe.unet,
                args.image_size,
                seed,
                args.device,
                erased_pipe.scheduler.init_noise_sigma,
            )

            # Base run with cache.
            cache_mgr = BaseActivationCache(base_pipe.unet, hookpoints, target_step_set)
            cache_mgr.debug = args.debug_hook_logs
            cache_mgr.register()
            try:
                img_base = run_generation(
                    base_pipe,
                    prompt,
                    init_lat,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    device=args.device,
                    step_hook_state=cache_mgr,
                    debug_loop_logs=args.debug_hook_logs,
                    run_label="transplant:base_capture",
                )
            finally:
                cache_mgr.remove()

            # Erased baseline (no transplant)
            img_erased = run_generation(
                erased_pipe,
                prompt,
                init_lat,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                device=args.device,
                step_hook_state=None,
                debug_loop_logs=args.debug_hook_logs,
                run_label="transplant:erased_no_transplant",
            )

            cdir = trans_root / f"group_{group}" / f"prompt_{prompt_slug}" / f"seed_{seed}"
            cdir.mkdir(parents=True, exist_ok=True)
            base_path = cdir / "base.png"
            erased_path = cdir / "erased_no_transplant.png"
            img_base.save(base_path)
            img_erased.save(erased_path)

            # Transplants for each beta.
            for beta in betas:
                tr = TransplantIntervention(
                    erased_pipe.unet,
                    hookpoints=hookpoints,
                    target_step_indices=target_step_set,
                    base_cache=cache_mgr.cache,
                    beta=float(beta),
                    debug=args.debug_hook_logs,
                )
                tr.register()
                try:
                    img_t = run_generation(
                        erased_pipe,
                        prompt,
                        init_lat,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        device=args.device,
                        step_hook_state=tr,
                        debug_loop_logs=args.debug_hook_logs,
                        run_label=f"transplant:beta={beta}",
                    )
                finally:
                    tr.remove()

                t_path = cdir / f"erased_with_transplant_beta_{beta}.png"
                img_t.save(t_path)

                meta = {
                    "hookpoint": hookpoints,
                    "timestep": target_steps,
                    "beta": float(beta),
                    "prompt": prompt,
                    "prompt_group": group,
                    "seed": int(seed),
                    "base_image_path": str(base_path),
                    "erased_image_path": str(erased_path),
                    "transplanted_image_path": str(t_path),
                    "transplant_events": tr.fired_events,
                }
                write_json(cdir / f"transplant_beta_{beta}.json", meta)

                for ev in tr.fired_events:
                    diff_rows.append(
                        {
                            "prompt_group": group,
                            "prompt": prompt,
                            "seed": int(seed),
                            "beta": float(beta),
                            "layer": ev["layer"],
                            "loop_idx": ev["step_idx"],
                            "scheduler_t": ev["scheduler_t"],
                            "h_erased_norm": ev["act_norm_before"],
                            "h_base_norm": ev["base_norm"],
                            "h_new_norm": ev["act_norm_after"],
                            "h_diff_norm": ev["diff_norm"],
                            "cosine_h_erased_h_base": ev["cosine_base_vs_erased"],
                        }
                    )

                # Row per generated sample (base, erased, transplanted) for easier auditing.
                rows.extend(
                    [
                        {
                            "experiment_type": "transplant",
                            "prompt_group": group,
                            "prompt": prompt,
                            "seed": seed,
                            "hookpoint": ",".join(hookpoints),
                            "timestep_or_timesteps": ",".join(str(t) for t in target_steps),
                            "alpha_or_beta": float(beta),
                            "direction_source": "",
                            "image_path": str(base_path),
                            "image_role": "base",
                            "setting": f"beta_{beta}_base",
                        },
                        {
                            "experiment_type": "transplant",
                            "prompt_group": group,
                            "prompt": prompt,
                            "seed": seed,
                            "hookpoint": ",".join(hookpoints),
                            "timestep_or_timesteps": ",".join(str(t) for t in target_steps),
                            "alpha_or_beta": float(beta),
                            "direction_source": "",
                            "image_path": str(erased_path),
                            "image_role": "erased_no_transplant",
                            "setting": f"beta_{beta}_erased_no_transplant",
                        },
                        {
                            "experiment_type": "transplant",
                            "prompt_group": group,
                            "prompt": prompt,
                            "seed": seed,
                            "hookpoint": ",".join(hookpoints),
                            "timestep_or_timesteps": ",".join(str(t) for t in target_steps),
                            "alpha_or_beta": float(beta),
                            "direction_source": "",
                            "image_path": str(t_path),
                            "image_role": "erased_with_transplant",
                            "setting": f"beta_{beta}_transplant",
                        },
                    ]
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "transplant_summary.csv", index=False)
    if diff_rows:
        pd.DataFrame(diff_rows).to_csv(out_dir / "transplant_activation_diffs.csv", index=False)

    # Contact sheets for transplanted images only
    sub = df[df.get("image_role", "") == "erased_with_transplant"]
    if not sub.empty:
        for prompt in sub["prompt"].drop_duplicates().tolist():
            psub = sub[sub["prompt"] == prompt].copy()
            pslug = re.sub(r"[^a-zA-Z0-9_]+", "_", prompt)[:120]
            make_contact_sheet(
                psub,
                out_dir / "transplant" / "contact_sheets" / f"{pslug}.png",
                settings_col="setting",
            )
    return df


# ----------------------------- CLI ----------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Causal recovery (steering + transplant) for STEREO")
    parser.add_argument("--base_model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--erased_unet_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis_causal_recovery")
    parser.add_argument("--prompt_config", type=str, required=True, help="JSON with A/B/(C)/D/E/F lists")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--timesteps_to_intervene", type=str, default="15")
    parser.add_argument("--layers_to_capture", type=str, default="mid_block.attentions.0")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--alphas", type=str, default="0,10,20,30,40,50,60")
    parser.add_argument("--betas", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--run_steering", action="store_true")
    parser.add_argument("--run_transplant", action="store_true")

    parser.add_argument("--base_direction_path", type=str, default="")
    parser.add_argument("--erased_direction_path", type=str, default="")

    parser.add_argument("--enable_s2_tokens", action="store_true")
    parser.add_argument("--use_attacked_text_encoder", action="store_true")
    parser.add_argument("--attacked_text_encoder_path", type=str, default="")
    parser.add_argument("--attacked_tokenizer_path", type=str, default="")

    parser.add_argument("--debug_mode", action="store_true", help="Small prompt subset")
    parser.add_argument("--debug_n_per_pos_group", type=int, default=2)
    parser.add_argument("--debug_n_controls", type=int, default=2)
    parser.add_argument("--debug_hook_logs", action="store_true")
    args = parser.parse_args()

    if not args.run_steering and not args.run_transplant:
        args.run_steering = True
        args.run_transplant = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_groups = load_prompt_config(args.prompt_config)
    prompt_df = build_prompt_table(
        prompt_groups=prompt_groups,
        enable_s2_tokens=args.enable_s2_tokens,
        debug_mode=args.debug_mode,
        debug_n_per_pos_group=args.debug_n_per_pos_group,
        debug_n_controls=args.debug_n_controls,
    )
    all_prompts = prompt_df["prompt"].tolist()

    base_pipe = load_base_pipeline(args.base_model_path, args.device)
    erased_pipe = load_erased_pipeline(args.base_model_path, args.erased_unet_checkpoint, args.device)

    if args.use_attacked_text_encoder:
        if not args.attacked_text_encoder_path:
            raise ValueError("--use_attacked_text_encoder requires --attacked_text_encoder_path")
        placeholder_tokens = _extract_tokens_from_prompts(all_prompts)
        _prepare_attacked_text_encoder_and_tokenizer(
            base_pipe,
            args.attacked_text_encoder_path,
            args.attacked_tokenizer_path or None,
            placeholder_tokens,
            args.device,
        )
        _prepare_attacked_text_encoder_and_tokenizer(
            erased_pipe,
            args.attacked_text_encoder_path,
            args.attacked_tokenizer_path or None,
            placeholder_tokens,
            args.device,
        )

    requested_layers = [x.strip() for x in args.layers_to_capture.split(",") if x.strip()]
    base_layers = resolve_target_layers(base_pipe.unet, requested_layers)
    erased_layers = resolve_target_layers(erased_pipe.unet, requested_layers)
    layers = [n for n in base_layers if n in set(erased_layers)]
    print("Matched intervention layers:")
    for ln in layers:
        print(f"  - {ln}")
    if not layers:
        raise RuntimeError("No common matched layers for interventions.")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    target_steps = [int(x.strip()) for x in args.timesteps_to_intervene.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]

    # Save run config
    write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "num_prompts": int(len(prompt_df)),
            "matched_layers": layers,
        },
    )

    if args.run_steering:
        direction_sources: dict[str, torch.Tensor] = {}
        if args.base_direction_path:
            direction_sources["base"] = load_direction_vector(args.base_direction_path)
        if args.erased_direction_path:
            direction_sources["erased"] = load_direction_vector(args.erased_direction_path)
        if not direction_sources:
            raise ValueError("Steering requested but no direction path provided.")
        run_activation_steering(
            erased_pipe=erased_pipe,
            prompt_df=prompt_df,
            seeds=seeds,
            hookpoints=layers,
            target_steps=target_steps,
            alphas=alphas,
            direction_sources=direction_sources,
            args=args,
            out_dir=out_dir,
        )

    if args.run_transplant:
        run_activation_transplant(
            base_pipe=base_pipe,
            erased_pipe=erased_pipe,
            prompt_df=prompt_df,
            seeds=seeds,
            hookpoints=layers,
            target_steps=target_steps,
            betas=betas,
            args=args,
            out_dir=out_dir,
        )

    print(f"Done. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

