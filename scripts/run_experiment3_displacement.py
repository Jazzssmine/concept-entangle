#!/usr/bin/env python3
"""Experiment 3: activation displacement under unlearning (Assumption 4.7)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.abspath(str(REPO_ROOT)))

from src.benchmark.generation import _load_model_runner  # noqa: E402
from src.benchmark.model_registry import load_model_registry  # noqa: E402


TARGET_CONCEPT = "horse"
NEIGHBOR_CONCEPTS = ["pony", "donkey"]
CONTROL_CONCEPTS = ["castle", "car"]
ALL_CONCEPTS = [TARGET_CONCEPT] + NEIGHBOR_CONCEPTS + CONTROL_CONCEPTS

TEMPLATES = [
    "a photo of a {c}",
    "a {c} in a field",
    "a {c} near a river",
    "a {c} at sunset",
    "a {c} in the rain",
    "a {c} on a hillside",
    "a {c} in a forest",
    "a {c} by the ocean",
    "a {c} under a cloudy sky",
    "a {c} in bright sunlight",
    "a {c} in the snow",
    "a {c} at night",
    "a {c} in a meadow",
    "a {c} near mountains",
    "a {c} on a dirt road",
    "a {c} beside a lake",
    "a {c} in fog",
    "a {c} during golden hour",
    "a {c} in a garden",
    "a {c} on a beach",
    "close-up of a {c}",
    "wide shot of a {c} in a valley",
    "a {c} in an open landscape",
    "a {c} surrounded by trees",
    "a {c} in the distance",
    "a {c} in the foreground of a dramatic scene",
    "a {c} under a stormy sky",
    "a {c} on a quiet street",
    "a {c} in warm afternoon light",
    "a {c} reflected in water",
    "a {c} in a rural setting",
    "a {c} in an urban environment",
    "a {c} on a bridge",
    "a {c} next to old stone walls",
    "a {c} in a wide open space",
    "a {c} under autumn leaves",
    "a {c} in spring",
    "a {c} at dawn",
    "a {c} at dusk",
    "a {c} in heavy rain",
    "a {c} covered in morning dew",
    "a {c} in a misty landscape",
    "a {c} beside wildflowers",
    "a {c} in tall grass",
    "a {c} on a rocky cliff",
    "a {c} with a blue sky background",
    "a {c} in a desert landscape",
    "a {c} near a waterfall",
    "a {c} in a dark environment",
    "a {c} lit from behind",
]


DEFAULT_METHOD_MODEL_KEYS = [
    "esd_horse",
    "erasediff_horse",
    "salun_horse",
    "advunlearn_horse",
    "stereo_horse",
    "saeuron_horse",
]


def build_prompts(num_prompts: int) -> dict[str, list[str]]:
    if num_prompts > len(TEMPLATES):
        raise ValueError(f"Requested {num_prompts} prompts but only {len(TEMPLATES)} templates exist.")
    return {concept: [template.format(c=concept) for template in TEMPLATES[:num_prompts]] for concept in ALL_CONCEPTS}


def method_alias_from_key(model_key: str) -> str:
    key = model_key.strip().lower()
    if key.endswith("_horse"):
        key = key[: -len("_horse")]
    return key


def find_target_attn2(unet: torch.nn.Module, target_layer: str) -> tuple[str, torch.nn.Module]:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if target_layer in name and "attn2" in name:
            candidates.append((name, module))
    exact = [x for x in candidates if x[0].endswith("attn2")]
    if exact:
        return exact[0]
    if candidates:
        return candidates[0]

    broad: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if "attn2" in name:
            broad.append((name, module))
    exact_broad = [x for x in broad if x[0].endswith("attn2")]
    if exact_broad:
        print(
            "[WARN] Could not find attn2 under target layer "
            f"'{target_layer}'. Using fallback module path: {exact_broad[0][0]}"
        )
        return exact_broad[0]
    if broad:
        print(
            "[WARN] Could not find attn2 under target layer "
            f"'{target_layer}'. Using fallback module path: {broad[0][0]}"
        )
        return broad[0]
    raise RuntimeError(f"Could not find an attn2 module under target layer '{target_layer}'.")


def extract_activation(
    pipe: StableDiffusionPipeline,
    target_module: torch.nn.Module,
    prompt: str,
    seed: int,
    capture_step: int,
    num_ddim_steps: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

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
        text_embeddings = text_encoder(text_input.input_ids)[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    generator = torch.Generator(device=device.type).manual_seed(seed)
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    scheduler.set_timesteps(num_ddim_steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    activation_cache: dict[str, torch.Tensor] = {}

    def capture_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        out = output[0] if isinstance(output, tuple) else output
        if out.shape[0] < 2:
            raise RuntimeError(f"Expected CFG batch >= 2, got {out.shape[0]}")
        cond = out[1]
        activation_cache["vec"] = cond.mean(dim=0).detach().cpu().float()
        return output

    for step_idx, t in enumerate(scheduler.timesteps):
        handle = None
        if step_idx == capture_step:
            handle = target_module.register_forward_hook(capture_hook)

        latent_input = torch.cat([latents] * 2, dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

        if handle is not None:
            handle.remove()

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if "vec" not in activation_cache:
        raise RuntimeError(f"Failed to capture activation at step {capture_step}")
    return activation_cache["vec"]


def compute_base_activations(
    pipe: StableDiffusionPipeline,
    target_module: torch.nn.Module,
    prompts_by_concept: dict[str, list[str]],
    num_prompts: int,
    seed_base: int,
    capture_step: int,
    num_ddim_steps: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for concept in ALL_CONCEPTS:
        vecs: list[torch.Tensor] = []
        for i, prompt in enumerate(tqdm(prompts_by_concept[concept], desc=f"base:{concept}", leave=False)):
            seed = seed_base + i
            vec = extract_activation(
                pipe=pipe,
                target_module=target_module,
                prompt=prompt,
                seed=seed,
                capture_step=capture_step,
                num_ddim_steps=num_ddim_steps,
                guidance_scale=guidance_scale,
                device=device,
                dtype=dtype,
            )
            vecs.append(vec)
        if len(vecs) != num_prompts:
            raise RuntimeError(f"Expected {num_prompts} prompts for {concept}, got {len(vecs)}")
        out[concept] = torch.stack(vecs, dim=0)
    return out


def save_base_activations(base_dir: Path, base_acts: dict[str, torch.Tensor]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for concept, tensor in base_acts.items():
        torch.save(tensor, base_dir / f"{concept}.pt")


def try_load_base_activations(base_dir: Path, num_prompts: int) -> dict[str, torch.Tensor] | None:
    if not base_dir.exists():
        return None
    loaded: dict[str, torch.Tensor] = {}
    for concept in ALL_CONCEPTS:
        p = base_dir / f"{concept}.pt"
        if not p.exists():
            return None
        x = torch.load(p, map_location="cpu")
        if x.ndim != 2 or x.shape[0] != num_prompts:
            return None
        loaded[concept] = x.float().cpu()
    return loaded


def compute_displacements_for_method(
    pipe: StableDiffusionPipeline,
    target_module: torch.nn.Module,
    base_acts: dict[str, torch.Tensor],
    prompts_by_concept: dict[str, list[str]],
    seed_base: int,
    capture_step: int,
    num_ddim_steps: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    displacements: dict[str, torch.Tensor] = {}
    for concept in ALL_CONCEPTS:
        dvals: list[float] = []
        for i, prompt in enumerate(tqdm(prompts_by_concept[concept], desc=f"method:{concept}", leave=False)):
            seed = seed_base + i
            u_vec = extract_activation(
                pipe=pipe,
                target_module=target_module,
                prompt=prompt,
                seed=seed,
                capture_step=capture_step,
                num_ddim_steps=num_ddim_steps,
                guidance_scale=guidance_scale,
                device=device,
                dtype=dtype,
            )
            base_vec = base_acts[concept][i]
            dvals.append(float(torch.linalg.vector_norm((u_vec - base_vec).float()).item()))
        displacements[concept] = torch.tensor(dvals, dtype=torch.float32)
    return displacements


def save_displacements(method_dir: Path, displacements: dict[str, torch.Tensor]) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    for concept, tensor in displacements.items():
        torch.save(tensor.cpu(), method_dir / f"{concept}.pt")


def try_load_displacements(method_dir: Path, num_prompts: int) -> dict[str, torch.Tensor] | None:
    if not method_dir.exists():
        return None
    loaded: dict[str, torch.Tensor] = {}
    for concept in ALL_CONCEPTS:
        p = method_dir / f"{concept}.pt"
        if not p.exists():
            return None
        x = torch.load(p, map_location="cpu")
        if x.ndim != 1 or x.shape[0] != num_prompts:
            return None
        loaded[concept] = x.float().cpu()
    return loaded


def summarize_displacements(method_alias: str, displacements: dict[str, torch.Tensor]) -> dict[str, float]:
    target_vals = displacements[TARGET_CONCEPT].cpu().numpy()
    neighbor_vals = torch.cat([displacements[c] for c in NEIGHBOR_CONCEPTS], dim=0).cpu().numpy()
    control_vals = torch.cat([displacements[c] for c in CONTROL_CONCEPTS], dim=0).cpu().numpy()

    return {
        "method": method_alias,
        "delta_target_mean": float(np.mean(target_vals)),
        "delta_target_std": float(np.std(target_vals, ddof=1)),
        "delta_neighbor_mean": float(np.mean(neighbor_vals)),
        "delta_neighbor_std": float(np.std(neighbor_vals, ddof=1)),
        "delta_control_mean": float(np.mean(control_vals)),
        "delta_control_std": float(np.std(control_vals, ddof=1)),
        "delta_gap_neighbor_minus_control": float(np.mean(neighbor_vals) - np.mean(control_vals)),
    }


def load_np_scores(method_aliases: list[str], method_keys: list[str], np_dir: Path) -> dict[str, float]:
    np_scores: dict[str, float] = {}
    for alias, key in zip(method_aliases, method_keys):
        p = np_dir / key / "aggregated_clip_metrics.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "model_name" not in df.columns or "NP" not in df.columns:
            continue
        sel = df[df["model_name"].astype(str).str.lower() == key.lower()]
        if len(sel) == 0:
            sel = df
        value = pd.to_numeric(sel["NP"], errors="coerce").dropna()
        if len(value) > 0:
            np_scores[alias] = float(value.iloc[0])
    return np_scores


def plot_displacement_bar(summary_df: pd.DataFrame, out_path: Path) -> None:
    methods = summary_df["method"].tolist()
    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x - w,
        summary_df["delta_target_mean"].to_numpy(),
        width=w,
        yerr=summary_df["delta_target_std"].to_numpy(),
        label="target (horse)",
        capsize=3,
    )
    ax.bar(
        x,
        summary_df["delta_neighbor_mean"].to_numpy(),
        width=w,
        yerr=summary_df["delta_neighbor_std"].to_numpy(),
        label="neighbors (pony/donkey/deer)",
        capsize=3,
    )
    ax.bar(
        x + w,
        summary_df["delta_control_mean"].to_numpy(),
        width=w,
        yerr=summary_df["delta_control_std"].to_numpy(),
        label="controls (castle/car)",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(r"Mean displacement $||\Delta||_2$")
    ax.set_title("Experiment 3: Activation Displacement by Method and Concept Type")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_vs_np(summary_df: pd.DataFrame, np_scores: dict[str, float], out_path: Path) -> None:
    rows = []
    for _, row in summary_df.iterrows():
        method = str(row["method"])
        if method in np_scores:
            rows.append((method, float(row["delta_target_mean"]), float(np_scores[method])))
    if len(rows) == 0:
        print("[WARN] No NP scores found. Skipping fig_delta_vs_np.pdf")
        return

    x = np.asarray([r[1] for r in rows], dtype=np.float64)
    y = np.asarray([r[2] for r in rows], dtype=np.float64)
    labels = [r[0] for r in rows]
    corr = float(np.corrcoef(x, y)[0, 1]) if len(rows) >= 2 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=60)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(r"$\Delta_{target}$ mean")
    ax.set_ylabel("NP score")
    title = "Experiment 3: Δ_target vs NP"
    if not np.isnan(corr):
        title += f" (Pearson r={corr:.3f})"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--model_registry_path", type=str, default="configs/benchmark_models.example.json")
    parser.add_argument("--method_model_keys", nargs="+", default=DEFAULT_METHOD_MODEL_KEYS)
    parser.add_argument("--output_dir", type=str, default="outputs/experiment3")
    parser.add_argument("--np_metrics_dir", type=str, default="outputs/benchmark/eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--target_layer", type=str, default="mid_block.attentions.0")
    parser.add_argument("--capture_step", type=int, default=25)
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--strict_methods", action="store_true")
    parser.add_argument("--recompute_base_activations", action="store_true")
    parser.add_argument("--recompute_method_displacements", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    if not (0 <= args.capture_step < args.num_ddim_steps):
        raise ValueError("capture_step must be in [0, num_ddim_steps - 1].")

    out_dir = Path(args.output_dir)
    displacements_root = out_dir / "displacements"
    base_acts_dir = out_dir / "base_activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    displacements_root.mkdir(parents=True, exist_ok=True)

    prompts_by_concept = build_prompts(args.num_prompts)
    (out_dir / "prompts.json").write_text(json.dumps(prompts_by_concept, indent=2), encoding="utf-8")

    print(f"Loading base model: {args.base_model_id}")
    base_pipe = StableDiffusionPipeline.from_pretrained(args.base_model_id, torch_dtype=dtype)
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
    base_pipe = base_pipe.to(device)
    base_pipe.set_progress_bar_config(disable=True)
    base_layer_name, base_target_module = find_target_attn2(base_pipe.unet, args.target_layer)
    print(f"Base hook layer: {base_layer_name}")

    base_acts = None if args.recompute_base_activations else try_load_base_activations(base_acts_dir, args.num_prompts)
    if base_acts is None:
        print("Computing base activations...")
        base_acts = compute_base_activations(
            pipe=base_pipe,
            target_module=base_target_module,
            prompts_by_concept=prompts_by_concept,
            num_prompts=args.num_prompts,
            seed_base=args.seed_base,
            capture_step=args.capture_step,
            num_ddim_steps=args.num_ddim_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            dtype=dtype,
        )
        save_base_activations(base_acts_dir, base_acts)
        print(f"Saved base activations to {base_acts_dir}")
    else:
        print(f"Loaded cached base activations from {base_acts_dir}")

    model_registry = load_model_registry(args.model_registry_path)
    summaries: list[dict[str, float]] = []
    executed_method_keys: list[str] = []
    executed_method_aliases: list[str] = []
    skipped_methods: list[str] = []

    for method_key in args.method_model_keys:
        alias = method_alias_from_key(method_key)
        method_disp_dir = displacements_root / alias
        cached_displacements = None
        if not args.recompute_method_displacements:
            cached_displacements = try_load_displacements(method_disp_dir, args.num_prompts)
        if cached_displacements is not None:
            print(f"\nUsing cached displacements for {method_key} (alias={alias}) from {method_disp_dir}")
            summaries.append(summarize_displacements(alias, cached_displacements))
            executed_method_keys.append(method_key)
            executed_method_aliases.append(alias)
            continue

        if method_key not in model_registry:
            msg = f"[WARN] Method '{method_key}' not found in registry; skipping."
            if args.strict_methods:
                raise KeyError(msg)
            print(msg)
            skipped_methods.append(method_key)
            continue
        spec = model_registry[method_key]
        print(f"\nLoading method: {method_key} (alias={alias}, type={spec.type})")
        try:
            method_pipe = _load_model_runner(spec, device=str(device))
        except Exception as exc:
            msg = f"[WARN] Failed to load method '{method_key}': {exc}"
            if args.strict_methods:
                raise RuntimeError(msg) from exc
            print(msg)
            skipped_methods.append(method_key)
            continue

        # Some inference-time wrappers (e.g., SAeUron) do not expose a full
        # diffusers pipeline interface (scheduler/unet/tokenizer/text_encoder).
        # This experiment script expects a standard pipeline object.
        required_attrs = ("scheduler", "unet", "tokenizer", "text_encoder")
        if not all(hasattr(method_pipe, attr) for attr in required_attrs):
            msg = (
                f"[WARN] Method '{method_key}' returned a wrapper without full pipeline "
                f"attrs {required_attrs}; skipping in run_experiment3_displacement. "
                "Use scripts/measure_activation_displacement.py for wrapper-based methods."
            )
            if args.strict_methods:
                raise RuntimeError(msg)
            print(msg)
            skipped_methods.append(method_key)
            continue

        method_pipe.scheduler = DDIMScheduler.from_config(method_pipe.scheduler.config)
        method_pipe = method_pipe.to(device)
        if hasattr(method_pipe, "set_progress_bar_config"):
            method_pipe.set_progress_bar_config(disable=True)
        layer_name, target_module = find_target_attn2(method_pipe.unet, args.target_layer)
        print(f"Method hook layer: {layer_name}")

        displacements = compute_displacements_for_method(
            pipe=method_pipe,
            target_module=target_module,
            base_acts=base_acts,
            prompts_by_concept=prompts_by_concept,
            seed_base=args.seed_base,
            capture_step=args.capture_step,
            num_ddim_steps=args.num_ddim_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            dtype=dtype,
        )
        save_displacements(method_disp_dir, displacements)
        summaries.append(summarize_displacements(alias, displacements))
        executed_method_keys.append(method_key)
        executed_method_aliases.append(alias)
        print(f"Saved displacement tensors to {method_disp_dir}")

    if len(summaries) == 0:
        raise RuntimeError("No methods were successfully processed.")

    summary_df = pd.DataFrame(summaries).sort_values("method").reset_index(drop=True)
    summary_path = out_dir / "displacement_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nMethod      | Δ_target | Δ_neighbor | Δ_control | Δ_gap (neighbor - control)")
    print("------------|----------|------------|-----------|---------------------------")
    for _, row in summary_df.iterrows():
        print(
            f"{row['method']:<11s} | "
            f"{row['delta_target_mean']:.4f}   | "
            f"{row['delta_neighbor_mean']:.4f}    | "
            f"{row['delta_control_mean']:.4f}   | "
            f"{row['delta_gap_neighbor_minus_control']:.4f}"
        )

    plot_displacement_bar(summary_df, out_dir / "fig_displacement_by_method.pdf")

    np_scores = load_np_scores(
        method_aliases=executed_method_aliases,
        method_keys=executed_method_keys,
        np_dir=Path(args.np_metrics_dir),
    )
    plot_delta_vs_np(summary_df, np_scores, out_dir / "fig_delta_vs_np.pdf")

    metadata = {
        "base_model_id": args.base_model_id,
        "method_model_keys_requested": args.method_model_keys,
        "method_model_keys_executed": executed_method_keys,
        "method_aliases_executed": executed_method_aliases,
        "method_model_keys_skipped": skipped_methods,
        "target_layer": args.target_layer,
        "capture_step": args.capture_step,
        "num_ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "seed_base": args.seed_base,
        "num_prompts": args.num_prompts,
        "concepts": {
            "target": TARGET_CONCEPT,
            "neighbors": NEIGHBOR_CONCEPTS,
            "controls": CONTROL_CONCEPTS,
        },
        "base_hook_layer_resolved": base_layer_name,
        "np_scores": np_scores,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  - {base_acts_dir}")
    print(f"  - {displacements_root}")
    print(f"  - {summary_path}")
    print(f"  - {out_dir / 'fig_displacement_by_method.pdf'}")
    if (out_dir / "fig_delta_vs_np.pdf").exists():
        print(f"  - {out_dir / 'fig_delta_vs_np.pdf'}")
    print(f"  - {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
