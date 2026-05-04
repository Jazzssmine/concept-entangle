#!/usr/bin/env python3
"""Experiment 2 (revised): Lipschitz validation via prompt variation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

PROMPT_GROUPS = {
    "horse": {
        "base": "a horse in a field",
        "tiny": [
            "a horse in a meadow",
            "a horse in a pasture",
            "a horse in a grassland",
            "a horse in an open field",
            "a horse in a green field",
            "a horse on a field",
            "a horse standing in a field",
            "a horse in a wide field",
            "a horse in a sunny field",
            "a horse in a quiet field",
        ],
        "small": [
            "a horse on a beach",
            "a horse in a forest",
            "a horse near a river",
            "a horse in the snow",
            "a horse in a barn",
            "a horse on a mountain",
            "a horse at sunset",
            "a horse in the rain",
            "a horse by a lake",
            "a horse on a dirt road",
        ],
        "medium": [
            "a pony in a field",
            "a donkey in a field",
            "a deer in a field",
            "a zebra in a field",
            "a unicorn in a field",
            "a foal in a field",
            "a mule in a field",
            "a stallion in a field",
            "a mare in a field",
            "a colt in a field",
        ],
        "large": [
            "a car in a field",
            "a castle in a field",
            "a flower in a field",
            "a truck in a field",
            "a bird in a field",
            "a dog in a field",
            "a cat in a field",
            "a tree in a field",
            "a rock in a field",
            "a chair in a field",
        ],
    },
    "car": {
        "base": "a car on a road",
        "tiny": [
            "a car on a street",
            "a car on a highway",
            "a car on a lane",
            "a car on a paved road",
            "a car on a wide road",
            "a car on a quiet road",
            "a car on an open road",
            "a car on a long road",
            "a car on a smooth road",
            "a car on a straight road",
        ],
        "small": [
            "a car in a city",
            "a car at the beach",
            "a car in the mountains",
            "a car in the rain",
            "a car at night",
            "a car in a parking lot",
            "a car in a garage",
            "a car by a river",
            "a car in the desert",
            "a car in the snow",
        ],
        "medium": [
            "a truck on a road",
            "a bus on a road",
            "a van on a road",
            "a taxi on a road",
            "a jeep on a road",
            "a sedan on a road",
            "a motorcycle on a road",
            "a bicycle on a road",
            "an ambulance on a road",
            "a tractor on a road",
        ],
        "large": [
            "a horse on a road",
            "a castle on a road",
            "a flower on a road",
            "a dog on a road",
            "a cat on a road",
            "a bird on a road",
            "a tree on a road",
            "a rock on a road",
            "a chair on a road",
            "a fish on a road",
        ],
    },
}

GROUP_ORDER = ["tiny", "small", "medium", "large"]
GROUP_COLORS = {"tiny": "#2ca02c", "small": "#1f77b4", "medium": "#ff7f0e", "large": "#d62728"}
CONCEPT_MARKERS = {"horse": "o", "car": "^"}
EPSILONS = [0.01, 0.05, 0.1, 0.2]


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


def concept_probability(
    image_pil: Any,
    concept_name: str,
    clip_module: Any,
    clip_model: torch.nn.Module,
    clip_preprocess: Any,
    device: torch.device,
) -> float:
    text = clip_module.tokenize([f"a photo of a {concept_name}", f"a photo with no {concept_name}"]).to(device)
    image = clip_preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = clip_model(image, text)
        probs = logits.softmax(dim=-1)
    return float(probs[0, 0].item())


def run_prompt_once(
    pipe: StableDiffusionPipeline,
    target_module: torch.nn.Module,
    prompt: str,
    concept_name: str,
    seed: int,
    capture_step: int,
    num_ddim_steps: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    clip_module: Any,
    clip_model: torch.nn.Module,
    clip_preprocess: Any,
) -> tuple[torch.Tensor, float]:
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
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
        raise RuntimeError(f"Failed to capture activation at step {capture_step}.")

    with torch.no_grad():
        image_tensor = vae.decode(latents / vae.config.scaling_factor).sample
    image_tensor = torch.clamp((image_tensor / 2 + 0.5), 0, 1)
    image_pil = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
    prob = concept_probability(
        image_pil=image_pil,
        concept_name=concept_name,
        clip_module=clip_module,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=device,
    )
    return activation_cache["vec"], prob


def save_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        raise RuntimeError(f"No rows to save for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV for --plot_only: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            parsed["activation_distance"] = float(parsed["activation_distance"])
            parsed["delta_probability_abs"] = float(parsed["delta_probability_abs"])
            if "seed_idx" in parsed:
                parsed["seed_idx"] = int(parsed["seed_idx"])
            if "seed" in parsed:
                parsed["seed"] = int(parsed["seed"])
            if "p_base" in parsed:
                parsed["p_base"] = float(parsed["p_base"])
            if "p_variant" in parsed:
                parsed["p_variant"] = float(parsed["p_variant"])
            if "ratio_delta_p_over_delta_a" in parsed:
                try:
                    parsed["ratio_delta_p_over_delta_a"] = float(parsed["ratio_delta_p_over_delta_a"])
                except ValueError:
                    parsed["ratio_delta_p_over_delta_a"] = float("nan")
            rows.append(parsed)
    if len(rows) == 0:
        raise RuntimeError(f"No rows found in CSV: {path}")
    return rows


def compute_empirical_l_from_perturbation_json(path: Path) -> tuple[float | None, dict[str, float]]:
    if not path.exists():
        return None, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    concept_payload = payload.get("concepts", {})
    if not isinstance(concept_payload, dict):
        return None, {}

    all_local_slopes: list[float] = []
    per_concept_l: dict[str, float] = {}
    for concept_name, concept_data in concept_payload.items():
        if not isinstance(concept_data, dict):
            continue
        x = np.asarray(concept_data.get("x_norms", []), dtype=np.float64)
        y = np.asarray(concept_data.get("y_prob_means", []), dtype=np.float64)
        if x.size < 2 or y.size < 2:
            continue
        local_slopes: list[float] = []
        for i in range(len(x) - 1):
            delta_norm = abs(float(x[i + 1] - x[i]))
            if delta_norm <= 1e-12:
                continue
            delta_prob = abs(float(y[i + 1] - y[i]))
            local_slopes.append(delta_prob / delta_norm)
        if local_slopes:
            concept_l = float(max(local_slopes))
            per_concept_l[concept_name] = concept_l
            all_local_slopes.extend(local_slopes)

    if len(all_local_slopes) == 0:
        return None, {}
    return float(max(all_local_slopes)), per_concept_l


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=str, default="outputs/experiment2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--capture_step", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--target_layer", type=str, default="mid_block.attentions.0")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument(
        "--perturbation_estimates_path",
        type=str,
        default=None,
        help="Path to perturbation-run lipschitz_estimates.json (from run_experiment2_lipschitz.py).",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip generation and rebuild figure/estimates from existing CSV.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="CSV path used with --plot_only (default: <output_dir>/raw/prompt_variation_results.csv).",
    )
    args = parser.parse_args()

    if args.num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    if not (0 <= args.capture_step < args.num_ddim_steps):
        raise ValueError("capture_step must be in [0, num_ddim_steps - 1].")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    try:
        import clip as clip_module
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'clip'. Install with: pip install clip-by-openai "
            "or pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw"
    perturbation_estimates_path = (
        Path(args.perturbation_estimates_path)
        if args.perturbation_estimates_path is not None
        else out_dir / "lipschitz_estimates.json"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt_groups.json").write_text(json.dumps(PROMPT_GROUPS, indent=2), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    target_name = args.target_layer
    if args.plot_only:
        input_csv = Path(args.input_csv) if args.input_csv is not None else raw_dir / "prompt_variation_results.csv"
        print(f"Plot-only mode: loading rows from {input_csv}")
        all_rows = load_rows_csv(input_csv)
    else:
        print(f"Loading SD pipeline: {args.model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        target_name, target_module = find_target_attn2(pipe.unet, args.target_layer)
        print(f"Hook target: {target_name}")

        print("Loading CLIP ViT-B/32 ...")
        clip_model, clip_preprocess = clip_module.load("ViT-B/32", device=device.type)
        clip_model.eval()

        concept_rows_gen: dict[str, list[dict[str, Any]]] = {c: [] for c in PROMPT_GROUPS.keys()}
        for concept_name, groups in PROMPT_GROUPS.items():
            base_prompt = groups["base"]
            variant_pairs = [(g, p) for g in GROUP_ORDER for p in groups[g]]
            work_items = [(seed_idx, g, p) for seed_idx in range(args.num_seeds) for g, p in variant_pairs]
            print(f"\nRunning concept={concept_name} with {len(work_items)} prompt pairs")
            for seed_idx, group_name, variant_prompt in tqdm(work_items, desc=f"{concept_name} pairs", leave=False):
                seed = args.seed_base + seed_idx
                a_base, p_base = run_prompt_once(
                    pipe=pipe,
                    target_module=target_module,
                    prompt=base_prompt,
                    concept_name=concept_name,
                    seed=seed,
                    capture_step=args.capture_step,
                    num_ddim_steps=args.num_ddim_steps,
                    guidance_scale=args.guidance_scale,
                    device=device,
                    dtype=dtype,
                    clip_module=clip_module,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                )
                a_var, p_var = run_prompt_once(
                    pipe=pipe,
                    target_module=target_module,
                    prompt=variant_prompt,
                    concept_name=concept_name,
                    seed=seed,
                    capture_step=args.capture_step,
                    num_ddim_steps=args.num_ddim_steps,
                    guidance_scale=args.guidance_scale,
                    device=device,
                    dtype=dtype,
                    clip_module=clip_module,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                )
                delta_a = float(torch.linalg.vector_norm((a_var - a_base).float()).item())
                delta_p = float(abs(p_var - p_base))
                ratio = float(delta_p / delta_a) if delta_a > 1e-12 else float("nan")
                row = {
                    "concept": concept_name,
                    "group": group_name,
                    "seed_idx": seed_idx,
                    "seed": seed,
                    "base_prompt": base_prompt,
                    "variant_prompt": variant_prompt,
                    "activation_distance": delta_a,
                    "delta_probability_abs": delta_p,
                    "p_base": p_base,
                    "p_variant": p_var,
                    "ratio_delta_p_over_delta_a": ratio,
                }
                all_rows.append(row)
                concept_rows_gen[concept_name].append(row)

        save_rows_csv(raw_dir / "prompt_variation_results.csv", all_rows)
        for concept_name, rows in concept_rows_gen.items():
            save_rows_csv(raw_dir / f"{concept_name}_prompt_variation_results.csv", rows)

    concept_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in PROMPT_GROUPS.keys()}
    for row in all_rows:
        c = str(row["concept"])
        if c in concept_rows:
            concept_rows[c].append(row)

    x = np.asarray([r["activation_distance"] for r in all_rows], dtype=np.float64)
    y = np.asarray([r["delta_probability_abs"] for r in all_rows], dtype=np.float64)
    valid = x > 1e-12
    l_pairs = np.divide(y[valid], x[valid]) if np.any(valid) else np.asarray([], dtype=np.float64)
    l_max = float(np.max(l_pairs)) if l_pairs.size > 0 else 0.0
    l_mean = float(np.mean(l_pairs)) if l_pairs.size > 0 else 0.0
    l_slope = float(np.polyfit(x, y, deg=1)[0]) if x.size >= 2 else 0.0
    l_through_origin = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 1e-12 else 0.0

    concept_stats: dict[str, dict[str, float]] = {}
    for concept_name, rows in concept_rows.items():
        cx = np.asarray([r["activation_distance"] for r in rows], dtype=np.float64)
        cy = np.asarray([r["delta_probability_abs"] for r in rows], dtype=np.float64)
        cvalid = cx > 1e-12
        cratios = np.divide(cy[cvalid], cx[cvalid]) if np.any(cvalid) else np.asarray([], dtype=np.float64)
        concept_stats[concept_name] = {
            "L_max": float(np.max(cratios)) if cratios.size > 0 else 0.0,
            "L_slope": float(np.polyfit(cx, cy, deg=1)[0]) if cx.size >= 2 else 0.0,
            "L_through_origin": float(np.sum(cx * cy) / np.sum(cx * cx)) if np.sum(cx * cx) > 1e-12 else 0.0,
            "mean_delta_a": float(np.mean(cx)),
            "mean_delta_p": float(np.mean(cy)),
            "num_pairs": int(len(rows)),
        }

    l_perturbation, perturbation_per_concept = compute_empirical_l_from_perturbation_json(perturbation_estimates_path)
    delta_bounds_perturbation: dict[str, float | None] = {}
    delta_bounds_prompt_max: dict[str, float | None] = {}
    for eps in EPSILONS:
        key = f"eps_{eps:g}"
        delta_bounds_perturbation[key] = (
            float((1.0 - eps) / l_perturbation) if (l_perturbation is not None and l_perturbation > 1e-12) else None
        )
        delta_bounds_prompt_max[key] = float((1.0 - eps) / l_max) if l_max > 1e-12 else None

    group_summary_rows: list[dict[str, Any]] = []
    for group_name in GROUP_ORDER:
        rows = [r for r in all_rows if r["group"] == group_name]
        gx = np.asarray([r["activation_distance"] for r in rows], dtype=np.float64)
        gy = np.asarray([r["delta_probability_abs"] for r in rows], dtype=np.float64)
        gvalid = gx > 1e-12
        gratios = np.divide(gy[gvalid], gx[gvalid]) if np.any(gvalid) else np.asarray([], dtype=np.float64)
        group_summary_rows.append(
            {
                "group": group_name,
                "mean_delta_a": float(np.mean(gx)) if gx.size > 0 else float("nan"),
                "mean_delta_p": float(np.mean(gy)) if gy.size > 0 else float("nan"),
                "max_ratio": float(np.max(gratios)) if gratios.size > 0 else float("nan"),
                "num_pairs": int(len(rows)),
            }
        )
    save_rows_csv(out_dir / "group_summary.csv", group_summary_rows)

    fig, ax = plt.subplots(figsize=(9, 6))
    for concept_name in PROMPT_GROUPS.keys():
        for group_name in GROUP_ORDER:
            rows = [r for r in concept_rows[concept_name] if r["group"] == group_name]
            gx = [r["activation_distance"] for r in rows]
            gy = [r["delta_probability_abs"] for r in rows]
            ax.scatter(
                gx,
                gy,
                alpha=0.72,
                s=28,
                color=GROUP_COLORS[group_name],
                marker=CONCEPT_MARKERS[concept_name],
                label=f"{concept_name}-{group_name}",
            )
    x_max = float(np.max(x)) if x.size > 0 else 1.0
    x_line = np.linspace(0.0, max(1e-6, x_max * 1.05), 100)
    ax.plot(x_line, l_through_origin * x_line, "k--", linewidth=1.5, label=f"L_through_origin={l_through_origin:.4f}")
    ax.plot(x_line, l_max * x_line, color="gray", linestyle=":", linewidth=1.2, label=f"L_max={l_max:.4f}")
    ax.set_xlabel(r"Activation distance $||\Phi_\theta(p_v) - \Phi_\theta(p_b)||_2$", fontsize=16)
    ax.set_ylabel(r"$|P(p_v) - P(p_b)|$", fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Experiment 2 (Revised): Prompt-Variation Lipschitz Validation", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for h, lbl in zip(handles, labels):
        if lbl not in dedup:
            dedup[lbl] = h
    ax.legend(
        dedup.values(),
        dedup.keys(),
        fontsize=8.5,
        ncol=2,
        loc="upper right",
        framealpha=0.82,
        borderpad=0.35,
        labelspacing=0.28,
        handletextpad=0.45,
        columnspacing=0.8,
        markerscale=0.82,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_lipschitz.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    estimates = {
        "model": args.model_id,
        "layer": target_name,
        "capture_step": args.capture_step,
        "num_ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "num_seeds": args.num_seeds,
        "seed_base": args.seed_base,
        "perturbation_estimates_source": str(perturbation_estimates_path),
        "L_perturbation": l_perturbation,
        "L_prompt_variation_max": l_max,
        "L_prompt_variation_mean": l_mean,
        "delta_bounds": delta_bounds_perturbation,
        "delta_bounds_prompt_variation_max": delta_bounds_prompt_max,
        "L_slope_least_squares_prompt": l_slope,
        "L_slope_through_origin_prompt": l_through_origin,
        "perturbation_per_concept_L": perturbation_per_concept,
        "per_concept": concept_stats,
    }
    (out_dir / "lipschitz_estimates.json").write_text(json.dumps(estimates, indent=2), encoding="utf-8")

    print("\nSummary table")
    print("Group   | Mean ||Δa|| | Mean |ΔP|  | Max |ΔP|/||Δa|| | # pairs")
    print("--------|-------------|-----------|------------------|--------")
    for row in group_summary_rows:
        print(
            f"{row['group']:<7s} | {row['mean_delta_a']:>11.4f} | {row['mean_delta_p']:>9.4f} | "
            f"{row['max_ratio']:>16.4f} | {row['num_pairs']:>6d}"
        )

    print("\nLipschitz estimates")
    if l_perturbation is None:
        print(
            f"L_perturbation: unavailable (missing/invalid file at {perturbation_estimates_path}). "
            "Run perturbation experiment first."
        )
    else:
        print(f"L_perturbation (max local slope): {l_perturbation:.6f}")
    print(f"L_prompt_variation_max (worst case): {l_max:.6f}")
    print(f"L_prompt_variation_mean: {l_mean:.6f}")
    print(f"L_slope (least squares): {l_slope:.6f}")
    print(f"L_through_origin: {l_through_origin:.6f}")
    for eps_key, bound in delta_bounds_perturbation.items():
        eps_display = eps_key.replace("eps_", "")
        if bound is None:
            print(f"eps={eps_display}: Delta bound undefined from perturbation L")
        else:
            print(f"eps={eps_display}: Delta >= {bound:.4f} (from perturbation L)")
    for eps_key, bound in delta_bounds_prompt_max.items():
        eps_display = eps_key.replace("eps_", "")
        if bound is None:
            print(f"eps={eps_display}: Delta bound undefined from prompt-variation L_max")
        else:
            print(f"eps={eps_display}: Delta >= {bound:.4f} (from prompt-variation L_max)")

    print("\nL comparison table")
    print("Method                      | L estimate")
    print("----------------------------|-----------")
    perturb_str = f"{l_perturbation:.6f}" if l_perturbation is not None else "N/A"
    print(f"{'Perturbation curve':<28s} | {perturb_str:>9s}")
    print(f"{'Prompt variation (max)':<28s} | {l_max:>9.6f}")
    print(f"{'Prompt variation (mean)':<28s} | {l_mean:>9.6f}")

    print("\nSaved:")
    print(f"  - {raw_dir / 'prompt_variation_results.csv'}")
    print(f"  - {raw_dir / 'horse_prompt_variation_results.csv'}")
    print(f"  - {raw_dir / 'car_prompt_variation_results.csv'}")
    print(f"  - {out_dir / 'group_summary.csv'}")
    print(f"  - {out_dir / 'fig_lipschitz.pdf'}")
    print(f"  - {out_dir / 'lipschitz_estimates.json'}")


if __name__ == "__main__":
    main()
