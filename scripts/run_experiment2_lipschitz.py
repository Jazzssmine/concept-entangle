#!/usr/bin/env python3
"""Experiment 2: Lipschitz continuity validation in cross-attention space."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from scipy.optimize import curve_fit
from tqdm import tqdm

CONCEPTS = ["horse", "car"]
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT_NUM_DIRECTIONS = 5

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
]


@dataclass
class TrialResult:
    prompt_idx: int
    sigma: float
    direction_idx: int
    concept_prob: float
    perturbation_norm: float


def find_target_attn2(unet: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if "up_blocks.1.attentions.1" in name and "attn2" in name:
            candidates.append((name, module))

    exact = [x for x in candidates if x[0].endswith("attn2")]
    if exact:
        return exact[0]
    if candidates:
        return candidates[0]

    broad: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if "up_blocks.1" in name and "attn2" in name:
            broad.append((name, module))
    exact_broad = [x for x in broad if x[0].endswith("attn2")]
    if exact_broad:
        return exact_broad[0]
    if broad:
        return broad[0]
    raise RuntimeError("Could not find an attn2 module under up_blocks.1.")


def build_prompts(num_prompts: int) -> dict[str, list[str]]:
    if num_prompts > len(TEMPLATES):
        raise ValueError(f"Requested {num_prompts} prompts but only {len(TEMPLATES)} templates exist.")
    return {concept: [template.format(c=concept) for template in TEMPLATES[:num_prompts]] for concept in CONCEPTS}


def sigmoid_response(x: np.ndarray, k: float, x_half: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(k * (x - x_half)))


def concept_probability(
    clip_module: Any,
    image_pil: Any,
    concept_name: str,
    clip_model: torch.nn.Module,
    clip_preprocess: Any,
    device: torch.device,
) -> float:
    image = clip_preprocess(image_pil).unsqueeze(0).to(device)
    text_candidates = [
        f"a photo of a {concept_name}",
        "a photo of a random object",
        "an abstract pattern with no recognizable subject",
        "a blurry unrecognizable image",
    ]
    text = clip_module.tokenize(text_candidates).to(device)
    with torch.no_grad():
        logits_per_image, _ = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1)
    return float(probs[0, 0].item())


def replace_attn_output(output: Any, modified_attn: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (modified_attn,) + output[1:]
    return modified_attn


def extract_attn_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def save_results_csv(path: Path, results: list[TrialResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_idx", "sigma", "direction_idx", "concept_prob", "perturbation_norm"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "prompt_idx": r.prompt_idx,
                    "sigma": r.sigma,
                    "direction_idx": r.direction_idx,
                    "concept_prob": r.concept_prob,
                    "perturbation_norm": r.perturbation_norm,
                }
            )


def aggregate_by_sigma(results: list[TrialResult]) -> dict[str, np.ndarray]:
    sigma_values: list[float] = []
    concept_prob_mean: list[float] = []
    concept_prob_std: list[float] = []
    perturbation_norm_mean: list[float] = []
    perturbation_norm_std: list[float] = []

    for sigma in SIGMAS:
        subset = [r for r in results if abs(r.sigma - sigma) < 1e-12]
        probs = np.asarray([r.concept_prob for r in subset], dtype=np.float64)
        norms = np.asarray([r.perturbation_norm for r in subset], dtype=np.float64)
        if probs.size == 0 or norms.size == 0:
            raise RuntimeError(f"No results found for sigma={sigma}")
        sigma_values.append(float(sigma))
        concept_prob_mean.append(float(np.mean(probs)))
        concept_prob_std.append(float(np.std(probs, ddof=1)) if probs.size > 1 else 0.0)
        perturbation_norm_mean.append(float(np.mean(norms)))
        perturbation_norm_std.append(float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0)

    return {
        "sigma": np.asarray(sigma_values, dtype=np.float64),
        "concept_prob_mean": np.asarray(concept_prob_mean, dtype=np.float64),
        "concept_prob_std": np.asarray(concept_prob_std, dtype=np.float64),
        "perturbation_norm_mean": np.asarray(perturbation_norm_mean, dtype=np.float64),
        "perturbation_norm_std": np.asarray(perturbation_norm_std, dtype=np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=str, default="outputs/experiment2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--capture_step", type=int, default=25)
    parser.add_argument("--perturb_start_step", type=int, default=25)
    parser.add_argument("--perturb_end_step", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_prompts", type=int, default=30)
    parser.add_argument("--num_directions", type=int, default=DEFAULT_NUM_DIRECTIONS)
    parser.add_argument("--seed_base", type=int, default=4242)
    parser.add_argument("--debug_output_dir", type=str, default="outputs/experiment2/debug")
    parser.add_argument(
        "--debug_probe_only",
        action="store_true",
        help="Run only the first horse prompt debug probe and exit.",
    )
    args = parser.parse_args()

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
    raw_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_output_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    prompts_by_concept = build_prompts(args.num_prompts)
    (out_dir / "prompts.json").write_text(json.dumps(prompts_by_concept, indent=2), encoding="utf-8")

    print(f"Loading SD pipeline: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print("Loading CLIP ViT-B/32 ...")
    clip_model, clip_preprocess = clip_module.load("ViT-B/32", device=device.type)
    clip_model.eval()

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    scheduler = pipe.scheduler

    target_name, target_module = find_target_attn2(unet)
    print(f"Hook target: {target_name}")

    summary_for_print: dict[str, dict[str, np.ndarray]] = {}
    lipschitz_json: dict[str, Any] = {
        "model": args.model_id,
        "layer": target_name,
        "capture_step": args.capture_step,
        "perturb_start_step": args.perturb_start_step,
        "perturb_end_step": args.perturb_end_step,
        "num_ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "sigmas": SIGMAS,
        "concepts": {},
    }

    if not (0 <= args.capture_step < args.num_ddim_steps):
        raise ValueError("capture_step must be in [0, num_ddim_steps - 1].")
    if not (0 <= args.perturb_start_step < args.num_ddim_steps):
        raise ValueError("perturb_start_step must be in [0, num_ddim_steps - 1].")
    if not (0 <= args.perturb_end_step < args.num_ddim_steps):
        raise ValueError("perturb_end_step must be in [0, num_ddim_steps - 1].")
    if args.perturb_start_step > args.perturb_end_step:
        raise ValueError("perturb_start_step must be <= perturb_end_step.")
    if args.perturb_start_step < args.capture_step:
        raise ValueError("perturb_start_step must be >= capture_step with current caching logic.")
    if args.num_directions <= 0:
        raise ValueError("num_directions must be positive.")
    perturb_steps = set(range(args.perturb_start_step, args.perturb_end_step + 1))

    # Step 1 debug probe on first horse prompt.
    debug_prompt = prompts_by_concept["horse"][0]
    debug_seed = args.seed_base
    debug_generator = torch.Generator(device=device.type).manual_seed(debug_seed)
    debug_text_input = tokenizer(
        debug_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    debug_uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        debug_text_embeddings = text_encoder(debug_text_input.input_ids)[0]
        debug_uncond_embeddings = text_encoder(debug_uncond_input.input_ids)[0]
    debug_text_embeddings = torch.cat([debug_uncond_embeddings, debug_text_embeddings], dim=0)

    scheduler.set_timesteps(args.num_ddim_steps, device=device)
    debug_timesteps = scheduler.timesteps
    debug_latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        generator=debug_generator,
        device=device,
        dtype=dtype,
    )
    debug_latents = debug_latents * scheduler.init_noise_sigma

    for step_idx in range(args.capture_step):
        t = debug_timesteps[step_idx]
        latent_input = torch.cat([debug_latents] * 2, dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=debug_text_embeddings).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        debug_latents = scheduler.step(noise_pred, t, debug_latents).prev_sample

    debug_latents_step_before = debug_latents.detach().clone()
    debug_capture_t = debug_timesteps[args.capture_step]
    debug_a_orig_cache: dict[str, torch.Tensor] = {}

    def debug_capture_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        out = extract_attn_tensor(output)
        debug_a_orig_cache["vec"] = out[1].mean(dim=0).detach()
        return output

    debug_capture_handle = target_module.register_forward_hook(debug_capture_hook)
    try:
        latent_input = torch.cat([debug_latents_step_before] * 2, dim=0)
        latent_input = scheduler.scale_model_input(latent_input, debug_capture_t)
        with torch.no_grad():
            _ = unet(latent_input, debug_capture_t, encoder_hidden_states=debug_text_embeddings).sample
    finally:
        debug_capture_handle.remove()

    if "vec" not in debug_a_orig_cache:
        raise RuntimeError("Debug probe failed to capture a_orig.")
    debug_a_orig = debug_a_orig_cache["vec"].detach()
    debug_a_orig_norm = float(torch.linalg.vector_norm(debug_a_orig).item())

    debug_direction = torch.randn_like(debug_a_orig)
    debug_direction_norm = torch.linalg.vector_norm(debug_direction).clamp_min(1e-12)
    debug_direction = debug_direction / debug_direction_norm
    debug_images: dict[float, torch.Tensor] = {}
    debug_hook_fires: dict[float, int] = {}

    debug_sigmas = [0.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    for debug_sigma in debug_sigmas:
        debug_delta = debug_direction * (debug_sigma * debug_a_orig_norm)
        debug_perturb = debug_delta.to(device=device, dtype=dtype)
        debug_run_latents = debug_latents_step_before.detach().clone()
        hook_state = {"printed_meta": False, "count": 0}

        if debug_sigma != 0.0:
            delta_norm = float(torch.linalg.vector_norm(debug_delta).item())
            print(f"DEBUG ||a_orig||_2 = {debug_a_orig_norm:.4f}")
            print(f"DEBUG ||delta||_2 at sigma={debug_sigma:g} = {delta_norm:.4f}")
            print(f"DEBUG ratio(sigma={debug_sigma:g}) = {delta_norm / max(debug_a_orig_norm, 1e-12):.4f}")

        def debug_perturb_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
            hook_state["count"] += 1
            out = extract_attn_tensor(output)
            if not hook_state["printed_meta"]:
                shape_str = tuple(out.shape)
                print(
                    f"DEBUG hook fired in steps [{args.perturb_start_step}, {args.perturb_end_step}] sigma={debug_sigma} "
                    f"output_type={type(output)} tensor_shape={shape_str}"
                )
                hook_state["printed_meta"] = True
            modified = out.clone()
            if modified.shape[0] < 2:
                raise RuntimeError(f"Expected CFG batch size >= 2 at hook, got {modified.shape[0]}")
            modified[1] = modified[1] + debug_perturb.unsqueeze(0)
            return replace_attn_output(output, modified)

        for step_idx in range(args.capture_step, args.num_ddim_steps):
            t = debug_timesteps[step_idx]
            handle = None
            if step_idx in perturb_steps:
                handle = target_module.register_forward_hook(debug_perturb_hook)
            latent_input = torch.cat([debug_run_latents] * 2, dim=0)
            latent_input = scheduler.scale_model_input(latent_input, t)
            with torch.no_grad():
                noise_pred = unet(latent_input, t, encoder_hidden_states=debug_text_embeddings).sample
            if handle is not None:
                handle.remove()
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            debug_run_latents = scheduler.step(noise_pred, t, debug_run_latents).prev_sample

        with torch.no_grad():
            debug_image_tensor = vae.decode(debug_run_latents / vae.config.scaling_factor).sample
        debug_image_tensor = torch.clamp((debug_image_tensor / 2 + 0.5), 0, 1)
        debug_images[debug_sigma] = debug_image_tensor.detach().cpu()
        debug_hook_fires[debug_sigma] = int(hook_state["count"])
        debug_pil = pipe.image_processor.postprocess(debug_image_tensor, output_type="pil")[0]
        debug_pil.save(debug_dir / f"sigma_{debug_sigma:g}.png")

    sigma0_img = debug_images[0.0]
    for s in debug_sigmas:
        if s == 0.0:
            continue
        debug_diff = torch.mean(torch.abs(sigma0_img - debug_images[s])).item()
        print(f"DEBUG mean_abs_pixel_diff(sigma=0 vs {s:g}): {debug_diff:.6f}")
        print(f"DEBUG hook fire counts: sigma=0 -> {debug_hook_fires[0.0]}, sigma={s:g} -> {debug_hook_fires[s]}")
    saved_paths = ", ".join(str(debug_dir / f"sigma_{s:g}.png") for s in debug_sigmas)
    print(f"DEBUG images saved: {saved_paths}")

    if args.debug_probe_only:
        print("Debug probe complete (--debug_probe_only); exiting without full run.")
        return

    for concept in CONCEPTS:
        concept_prompts = prompts_by_concept[concept]
        concept_results: list[TrialResult] = []
        print(f"\nRunning concept: {concept}")

        for prompt_idx, prompt in enumerate(tqdm(concept_prompts, desc=f"{concept} prompts")):
            seed = args.seed_base + prompt_idx
            generator = torch.Generator(device=device.type).manual_seed(seed)

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

            scheduler.set_timesteps(args.num_ddim_steps, device=device)
            timesteps = scheduler.timesteps
            latents = torch.randn(
                (1, unet.config.in_channels, 64, 64),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            latents = latents * scheduler.init_noise_sigma

            for step_idx in range(args.capture_step):
                t = timesteps[step_idx]
                latent_input = torch.cat([latents] * 2, dim=0)
                latent_input = scheduler.scale_model_input(latent_input, t)
                with torch.no_grad():
                    noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents_step_before = latents.detach().clone()
            capture_t = timesteps[args.capture_step]
            a_orig_cache: dict[str, torch.Tensor] = {}

            def capture_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
                out = output[0] if isinstance(output, tuple) else output
                cond = out[1]
                a_orig_cache["vec"] = cond.mean(dim=0).detach()
                return output

            capture_handle = target_module.register_forward_hook(capture_hook)
            try:
                latent_input = torch.cat([latents_step_before] * 2, dim=0)
                latent_input = scheduler.scale_model_input(latent_input, capture_t)
                with torch.no_grad():
                    _ = unet(latent_input, capture_t, encoder_hidden_states=text_embeddings).sample
            finally:
                capture_handle.remove()

            if "vec" not in a_orig_cache:
                raise RuntimeError(f"Missing a_orig for prompt {prompt_idx} ({concept}).")
            a_orig = a_orig_cache["vec"].detach()
            a_orig_norm = float(torch.linalg.vector_norm(a_orig).item())

            for sigma in SIGMAS:
                for direction_idx in range(args.num_directions):
                    if sigma == 0.0:
                        delta = torch.zeros_like(a_orig)
                    else:
                        direction = torch.randn_like(a_orig)
                        direction_norm = torch.linalg.vector_norm(direction)
                        if float(direction_norm.item()) < 1e-12:
                            direction = torch.ones_like(a_orig)
                            direction_norm = torch.linalg.vector_norm(direction)
                        delta = direction / direction_norm * (sigma * a_orig_norm)

                    perturbation = delta.to(device=device, dtype=dtype)
                    perturbation_norm = float(torch.linalg.vector_norm(perturbation.float()).item())
                    run_latents = latents_step_before.detach().clone()

                    hook_state = {"printed_meta": False, "count": 0}

                    def perturb_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
                        hook_state["count"] += 1
                        out = extract_attn_tensor(output)
                        if (
                            concept == "horse"
                            and prompt_idx == 0
                            and direction_idx == 0
                            and abs(sigma - 2.0) < 1e-12
                            and not hook_state["printed_meta"]
                        ):
                            print(
                                f"DEBUG run hook fired in steps [{args.perturb_start_step}, {args.perturb_end_step}] "
                                f"type={type(output)} shape={tuple(out.shape)}"
                            )
                            hook_state["printed_meta"] = True
                        modified = out.clone()
                        if modified.shape[0] < 2:
                            raise RuntimeError(f"Expected CFG batch size >= 2 at hook, got {modified.shape[0]}")
                        modified[1] = modified[1] + perturbation.unsqueeze(0)
                        return replace_attn_output(output, modified)

                    for step_idx in range(args.capture_step, args.num_ddim_steps):
                        t = timesteps[step_idx]
                        handle = None
                        if step_idx in perturb_steps:
                            handle = target_module.register_forward_hook(perturb_hook)

                        latent_input = torch.cat([run_latents] * 2, dim=0)
                        latent_input = scheduler.scale_model_input(latent_input, t)
                        with torch.no_grad():
                            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

                        if handle is not None:
                            handle.remove()

                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        run_latents = scheduler.step(noise_pred, t, run_latents).prev_sample

                    with torch.no_grad():
                        image_tensor = vae.decode(run_latents / vae.config.scaling_factor).sample
                    image_tensor = torch.clamp((image_tensor / 2 + 0.5), 0, 1)

                    image_pil = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
                    concept_prob = concept_probability(
                        clip_module=clip_module,
                        image_pil=image_pil,
                        concept_name=concept,
                        clip_model=clip_model,
                        clip_preprocess=clip_preprocess,
                        device=device,
                    )
                    concept_results.append(
                        TrialResult(
                            prompt_idx=prompt_idx,
                            sigma=float(sigma),
                            direction_idx=direction_idx,
                            concept_prob=concept_prob,
                            perturbation_norm=perturbation_norm,
                        )
                    )

        save_results_csv(raw_dir / f"{concept}_results.csv", concept_results)
        agg = aggregate_by_sigma(concept_results)
        summary_for_print[concept] = agg

        x = agg["perturbation_norm_mean"]
        y = agg["concept_prob_mean"]
        local_slopes = []
        for i in range(len(x) - 1):
            dx = abs(x[i + 1] - x[i])
            if dx > 1e-12:
                local_slopes.append(abs(y[i + 1] - y[i]) / dx)
        empirical_l = float(max(local_slopes)) if local_slopes else 0.0

        k_est = None
        sigmoid_l = None
        sigma_half_est = None
        try:
            p0 = [1.0, float(np.median(x))]
            bounds = ([0.0, float(np.min(x))], [200.0, float(np.max(x)) + 1e-6])
            params, _ = curve_fit(sigmoid_response, x, y, p0=p0, bounds=bounds, maxfev=20000)
            k_est = float(params[0])
            sigma_half_est = float(params[1])
            sigmoid_l = float(k_est / 4.0)
        except Exception as exc:
            print(f"[WARN] Sigmoid fit failed for {concept}: {exc}")

        lipschitz_json["concepts"][concept] = {
            "empirical_local_max": empirical_l,
            "sigmoid_k": k_est,
            "sigmoid_sigma_half": sigma_half_est,
            "sigmoid_max_slope_k_over_4": sigmoid_l,
            "x_norms": x.tolist(),
            "y_prob_means": y.tolist(),
        }
        print(
            f"{concept}: empirical L={empirical_l:.6f}, "
            f"sigmoid L={sigmoid_l if sigmoid_l is not None else 'nan'}"
        )

    perturbation_l_values = [
        float(v.get("empirical_local_max", 0.0))
        for v in lipschitz_json["concepts"].values()
        if isinstance(v, dict)
    ]
    l_perturbation = max(perturbation_l_values) if perturbation_l_values else 0.0
    delta_bounds = {}
    for eps in [0.01, 0.05, 0.1, 0.2]:
        key = f"eps_{eps:g}"
        delta_bounds[key] = float((1.0 - eps) / l_perturbation) if l_perturbation > 1e-12 else None
    lipschitz_json["L_perturbation"] = float(l_perturbation)
    lipschitz_json["delta_bounds"] = delta_bounds

    plt.figure(figsize=(8, 5))
    for concept in CONCEPTS:
        agg = summary_for_print[concept]
        x = agg["perturbation_norm_mean"]
        y = agg["concept_prob_mean"]
        s = np.nan_to_num(agg["concept_prob_std"], nan=0.0)
        plt.plot(x, y, marker="o", label=concept)
        plt.fill_between(x, np.clip(y - s, 0.0, 1.0), np.clip(y + s, 0.0, 1.0), alpha=0.2)
    plt.xlabel(r"Perturbation magnitude $||\delta||_2$", fontsize=16)
    plt.ylabel("Concept probability", fontsize=16)
    plt.ylim(0.0, 1.0)
    plt.title("Experiment 2: Perturbation-Response Curves", fontsize=17)
    plt.tick_params(axis="both", which="major", labelsize=13)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_lipschitz.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    (out_dir / "lipschitz_estimates.json").write_text(json.dumps(lipschitz_json, indent=2), encoding="utf-8")

    print("\nSummary table")
    print("sigma  | ||delta||  | P(horse) mean ± std | P(car) mean ± std")
    print("-------|-----------|---------------------|-------------------")
    horse_agg = summary_for_print["horse"]
    car_agg = summary_for_print["car"]
    for idx, sigma in enumerate(SIGMAS):
        delta_mean = 0.5 * (
            float(horse_agg["perturbation_norm_mean"][idx]) + float(car_agg["perturbation_norm_mean"][idx])
        )
        print(
            f"{sigma:>5.2f} | {delta_mean:>9.3f} | "
            f"{horse_agg['concept_prob_mean'][idx]:.3f} ± {horse_agg['concept_prob_std'][idx]:.3f}     | "
            f"{car_agg['concept_prob_mean'][idx]:.3f} ± {car_agg['concept_prob_std'][idx]:.3f}"
        )

    print("\nPerturbation Lipschitz summary")
    print(f"Empirical L_perturbation (max local slope): {l_perturbation:.6f}")
    for eps_key, bound in delta_bounds.items():
        eps_display = eps_key.replace("eps_", "")
        if bound is None:
            print(f"eps={eps_display}: Delta bound undefined (L_perturbation ~ 0)")
        else:
            print(f"eps={eps_display}: Delta >= {bound:.4f}")

    print("\nSaved:")
    print(f"  - {raw_dir / 'horse_results.csv'}")
    print(f"  - {raw_dir / 'car_results.csv'}")
    print(f"  - {out_dir / 'fig_lipschitz.pdf'}")
    print(f"  - {out_dir / 'lipschitz_estimates.json'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Experiment 2 (revised): Lipschitz validation via prompt variation."""

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
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt_groups.json").write_text(json.dumps(PROMPT_GROUPS, indent=2), encoding="utf-8")

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

    all_rows: list[dict[str, Any]] = []
    concept_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in PROMPT_GROUPS.keys()}

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
            concept_rows[concept_name].append(row)

    save_rows_csv(raw_dir / "prompt_variation_results.csv", all_rows)
    for concept_name, rows in concept_rows.items():
        save_rows_csv(raw_dir / f"{concept_name}_prompt_variation_results.csv", rows)

    x = np.asarray([r["activation_distance"] for r in all_rows], dtype=np.float64)
    y = np.asarray([r["delta_probability_abs"] for r in all_rows], dtype=np.float64)
    valid = x > 1e-12
    ratios = np.divide(y[valid], x[valid]) if np.any(valid) else np.asarray([], dtype=np.float64)
    l_max = float(np.max(ratios)) if ratios.size > 0 else 0.0
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

    eps_bounds: dict[str, float | None] = {}
    for eps in [0.05, 0.1, 0.2]:
        key = f"{eps:.2f}"
        if l_max > 1e-12:
            eps_bounds[key] = float((1.0 - eps) / l_max)
        else:
            eps_bounds[key] = None

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

    # Scatter + fitted line (through origin).
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
    ax.set_xlabel(r"Activation distance $||\Phi_\theta(p_v) - \Phi_\theta(p_b)||_2$")
    ax.set_ylabel(r"$|P(p_v) - P(p_b)|$")
    ax.set_title("Experiment 2 (Revised): Prompt-Variation Lipschitz Validation")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for h, lbl in zip(handles, labels):
        if lbl not in dedup:
            dedup[lbl] = h
    ax.legend(dedup.values(), dedup.keys(), fontsize=8, ncol=2, loc="upper left")
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
        "L_max_worst_case": l_max,
        "L_slope_least_squares": l_slope,
        "L_slope_through_origin": l_through_origin,
        "epsilon_displacement_bounds": eps_bounds,
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
    print(f"L_max (worst case): {l_max:.6f}")
    print(f"L_slope (least squares): {l_slope:.6f}")
    print(f"L_through_origin: {l_through_origin:.6f}")
    for eps_key, bound in eps_bounds.items():
        if bound is None:
            print(f"eps={eps_key}: Delta bound undefined (L_max ~ 0)")
        else:
            print(f"eps={eps_key}: Delta >= {bound:.4f}")

    print("\nSaved:")
    print(f"  - {raw_dir / 'prompt_variation_results.csv'}")
    print(f"  - {raw_dir / 'horse_prompt_variation_results.csv'}")
    print(f"  - {raw_dir / 'car_prompt_variation_results.csv'}")
    print(f"  - {out_dir / 'group_summary.csv'}")
    print(f"  - {out_dir / 'fig_lipschitz.pdf'}")
    print(f"  - {out_dir / 'lipschitz_estimates.json'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Experiment 2: Lipschitz continuity validation in cross-attention space."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from scipy.optimize import curve_fit
from tqdm import tqdm

CONCEPTS = ["horse", "car"]
SIGMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT_NUM_DIRECTIONS = 5

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
]


@dataclass
class TrialResult:
    prompt_idx: int
    sigma: float
    direction_idx: int
    concept_prob: float
    perturbation_norm: float


def find_target_attn2(unet: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if "up_blocks.1.attentions.1" in name and "attn2" in name:
            candidates.append((name, module))

    exact = [x for x in candidates if x[0].endswith("attn2")]
    if exact:
        return exact[0]
    if candidates:
        return candidates[0]

    broad: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if "up_blocks.1" in name and "attn2" in name:
            broad.append((name, module))
    exact_broad = [x for x in broad if x[0].endswith("attn2")]
    if exact_broad:
        return exact_broad[0]
    if broad:
        return broad[0]
    raise RuntimeError("Could not find an attn2 module under up_blocks.1.")


def build_prompts(num_prompts: int) -> dict[str, list[str]]:
    if num_prompts > len(TEMPLATES):
        raise ValueError(f"Requested {num_prompts} prompts but only {len(TEMPLATES)} templates exist.")
    return {concept: [template.format(c=concept) for template in TEMPLATES[:num_prompts]] for concept in CONCEPTS}


def sigmoid_response(x: np.ndarray, k: float, x_half: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(k * (x - x_half)))


def concept_probability(
    clip_module: Any,
    image_pil: Any,
    concept_name: str,
    clip_model: torch.nn.Module,
    clip_preprocess: Any,
    device: torch.device,
) -> float:
    image = clip_preprocess(image_pil).unsqueeze(0).to(device)
    text_candidates = [
        f"a photo of a {concept_name}",
        "a photo of a random object",
        "an abstract pattern with no recognizable subject",
        "a blurry unrecognizable image",
    ]
    text = clip_module.tokenize(text_candidates).to(device)
    with torch.no_grad():
        logits_per_image, _ = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1)
    return float(probs[0, 0].item())


def replace_attn_output(output: Any, modified_attn: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (modified_attn,) + output[1:]
    return modified_attn


def extract_attn_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def save_results_csv(path: Path, results: list[TrialResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_idx", "sigma", "direction_idx", "concept_prob", "perturbation_norm"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "prompt_idx": r.prompt_idx,
                    "sigma": r.sigma,
                    "direction_idx": r.direction_idx,
                    "concept_prob": r.concept_prob,
                    "perturbation_norm": r.perturbation_norm,
                }
            )


def aggregate_by_sigma(results: list[TrialResult]) -> dict[str, np.ndarray]:
    sigma_values: list[float] = []
    concept_prob_mean: list[float] = []
    concept_prob_std: list[float] = []
    perturbation_norm_mean: list[float] = []
    perturbation_norm_std: list[float] = []

    for sigma in SIGMAS:
        subset = [r for r in results if abs(r.sigma - sigma) < 1e-12]
        probs = np.asarray([r.concept_prob for r in subset], dtype=np.float64)
        norms = np.asarray([r.perturbation_norm for r in subset], dtype=np.float64)
        if probs.size == 0 or norms.size == 0:
            raise RuntimeError(f"No results found for sigma={sigma}")
        sigma_values.append(float(sigma))
        concept_prob_mean.append(float(np.mean(probs)))
        concept_prob_std.append(float(np.std(probs, ddof=1)) if probs.size > 1 else 0.0)
        perturbation_norm_mean.append(float(np.mean(norms)))
        perturbation_norm_std.append(float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0)

    return {
        "sigma": np.asarray(sigma_values, dtype=np.float64),
        "concept_prob_mean": np.asarray(concept_prob_mean, dtype=np.float64),
        "concept_prob_std": np.asarray(concept_prob_std, dtype=np.float64),
        "perturbation_norm_mean": np.asarray(perturbation_norm_mean, dtype=np.float64),
        "perturbation_norm_std": np.asarray(perturbation_norm_std, dtype=np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=str, default="outputs/experiment2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16")
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--capture_step", type=int, default=25)
    parser.add_argument("--perturb_start_step", type=int, default=25)
    parser.add_argument("--perturb_end_step", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_prompts", type=int, default=30)
    parser.add_argument("--num_directions", type=int, default=DEFAULT_NUM_DIRECTIONS)
    parser.add_argument("--seed_base", type=int, default=4242)
    parser.add_argument("--debug_output_dir", type=str, default="outputs/experiment2/debug")
    parser.add_argument(
        "--debug_probe_only",
        action="store_true",
        help="Run only the first horse prompt debug probe and exit.",
    )
    args = parser.parse_args()

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
    raw_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(args.debug_output_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    prompts_by_concept = build_prompts(args.num_prompts)
    (out_dir / "prompts.json").write_text(json.dumps(prompts_by_concept, indent=2), encoding="utf-8")

    print(f"Loading SD pipeline: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    print("Loading CLIP ViT-B/32 ...")
    clip_model, clip_preprocess = clip_module.load("ViT-B/32", device=device.type)
    clip_model.eval()

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    scheduler = pipe.scheduler

    target_name, target_module = find_target_attn2(unet)
    print(f"Hook target: {target_name}")

    summary_for_print: dict[str, dict[str, np.ndarray]] = {}
    lipschitz_json: dict[str, Any] = {
        "model": args.model_id,
        "layer": target_name,
        "capture_step": args.capture_step,
        "perturb_start_step": args.perturb_start_step,
        "perturb_end_step": args.perturb_end_step,
        "num_ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "sigmas": SIGMAS,
        "concepts": {},
    }

    if not (0 <= args.capture_step < args.num_ddim_steps):
        raise ValueError("capture_step must be in [0, num_ddim_steps - 1].")
    if not (0 <= args.perturb_start_step < args.num_ddim_steps):
        raise ValueError("perturb_start_step must be in [0, num_ddim_steps - 1].")
    if not (0 <= args.perturb_end_step < args.num_ddim_steps):
        raise ValueError("perturb_end_step must be in [0, num_ddim_steps - 1].")
    if args.perturb_start_step > args.perturb_end_step:
        raise ValueError("perturb_start_step must be <= perturb_end_step.")
    if args.perturb_start_step < args.capture_step:
        raise ValueError("perturb_start_step must be >= capture_step with current caching logic.")
    if args.num_directions <= 0:
        raise ValueError("num_directions must be positive.")
    perturb_steps = set(range(args.perturb_start_step, args.perturb_end_step + 1))

    # Step 1 debug probe on first horse prompt: save sigma=0 and sigma=2.0 images.
    debug_prompt = prompts_by_concept["horse"][0]
    debug_seed = args.seed_base
    debug_generator = torch.Generator(device=device.type).manual_seed(debug_seed)
    debug_text_input = tokenizer(
        debug_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    debug_uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        debug_text_embeddings = text_encoder(debug_text_input.input_ids)[0]
        debug_uncond_embeddings = text_encoder(debug_uncond_input.input_ids)[0]
    debug_text_embeddings = torch.cat([debug_uncond_embeddings, debug_text_embeddings], dim=0)

    scheduler.set_timesteps(args.num_ddim_steps, device=device)
    debug_timesteps = scheduler.timesteps
    debug_latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        generator=debug_generator,
        device=device,
        dtype=dtype,
    )
    debug_latents = debug_latents * scheduler.init_noise_sigma

    for step_idx in range(args.capture_step):
        t = debug_timesteps[step_idx]
        latent_input = torch.cat([debug_latents] * 2, dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=debug_text_embeddings).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        debug_latents = scheduler.step(noise_pred, t, debug_latents).prev_sample

    debug_latents_step_before = debug_latents.detach().clone()
    debug_capture_t = debug_timesteps[args.capture_step]
    debug_a_orig_cache: dict[str, torch.Tensor] = {}

    def debug_capture_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
        out = extract_attn_tensor(output)
        debug_a_orig_cache["vec"] = out[1].mean(dim=0).detach()
        return output

    debug_capture_handle = target_module.register_forward_hook(debug_capture_hook)
    try:
        latent_input = torch.cat([debug_latents_step_before] * 2, dim=0)
        latent_input = scheduler.scale_model_input(latent_input, debug_capture_t)
        with torch.no_grad():
            _ = unet(latent_input, debug_capture_t, encoder_hidden_states=debug_text_embeddings).sample
    finally:
        debug_capture_handle.remove()

    if "vec" not in debug_a_orig_cache:
        raise RuntimeError("Debug probe failed to capture a_orig.")
    debug_a_orig = debug_a_orig_cache["vec"].detach()
    debug_a_orig_norm = float(torch.linalg.vector_norm(debug_a_orig).item())

    debug_direction = torch.randn_like(debug_a_orig)
    debug_direction_norm = torch.linalg.vector_norm(debug_direction).clamp_min(1e-12)
    debug_direction = debug_direction / debug_direction_norm
    debug_images: dict[float, torch.Tensor] = {}
    debug_hook_fires: dict[float, int] = {}

    debug_sigmas = [0.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    for debug_sigma in debug_sigmas:
        debug_delta = debug_direction * (debug_sigma * debug_a_orig_norm)
        debug_perturb = debug_delta.to(device=device, dtype=dtype)
        debug_run_latents = debug_latents_step_before.detach().clone()
        hook_state = {"printed_meta": False, "count": 0}

        if debug_sigma != 0.0:
            delta_norm = float(torch.linalg.vector_norm(debug_delta).item())
            print(f"DEBUG ||a_orig||_2 = {debug_a_orig_norm:.4f}")
            print(f"DEBUG ||delta||_2 at sigma={debug_sigma:g} = {delta_norm:.4f}")
            print(f"DEBUG ratio(sigma={debug_sigma:g}) = {delta_norm / max(debug_a_orig_norm, 1e-12):.4f}")

        def debug_perturb_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
            hook_state["count"] += 1
            out = extract_attn_tensor(output)
            if not hook_state["printed_meta"]:
                shape_str = tuple(out.shape)
                print(
                    f"DEBUG hook fired in steps [{args.perturb_start_step}, {args.perturb_end_step}] sigma={debug_sigma} "
                    f"output_type={type(output)} tensor_shape={shape_str}"
                )
                hook_state["printed_meta"] = True
            modified = out.clone()
            if modified.shape[0] < 2:
                raise RuntimeError(f"Expected CFG batch size >= 2 at hook, got {modified.shape[0]}")
            modified[1] = modified[1] + debug_perturb.unsqueeze(0)
            return replace_attn_output(output, modified)

        for step_idx in range(args.capture_step, args.num_ddim_steps):
            t = debug_timesteps[step_idx]
            handle = None
            if step_idx in perturb_steps:
                handle = target_module.register_forward_hook(debug_perturb_hook)
            latent_input = torch.cat([debug_run_latents] * 2, dim=0)
            latent_input = scheduler.scale_model_input(latent_input, t)
            with torch.no_grad():
                noise_pred = unet(latent_input, t, encoder_hidden_states=debug_text_embeddings).sample
            if handle is not None:
                handle.remove()
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            debug_run_latents = scheduler.step(noise_pred, t, debug_run_latents).prev_sample

        with torch.no_grad():
            debug_image_tensor = vae.decode(debug_run_latents / vae.config.scaling_factor).sample
        debug_image_tensor = torch.clamp((debug_image_tensor / 2 + 0.5), 0, 1)
        debug_images[debug_sigma] = debug_image_tensor.detach().cpu()
        debug_hook_fires[debug_sigma] = int(hook_state["count"])
        debug_pil = pipe.image_processor.postprocess(debug_image_tensor, output_type="pil")[0]
        debug_pil.save(debug_dir / f"sigma_{debug_sigma:g}.png")

    sigma0_img = debug_images[0.0]
    for s in debug_sigmas:
        if s == 0.0:
            continue
        debug_diff = torch.mean(torch.abs(sigma0_img - debug_images[s])).item()
        print(f"DEBUG mean_abs_pixel_diff(sigma=0 vs {s:g}): {debug_diff:.6f}")
        print(f"DEBUG hook fire counts: sigma=0 -> {debug_hook_fires[0.0]}, sigma={s:g} -> {debug_hook_fires[s]}")
    saved_paths = ", ".join(str(debug_dir / f"sigma_{s:g}.png") for s in debug_sigmas)
    print(f"DEBUG images saved: {saved_paths}")

    if args.debug_probe_only:
        print("Debug probe complete (--debug_probe_only); exiting without full run.")
        return

    for concept in CONCEPTS:
        concept_prompts = prompts_by_concept[concept]
        concept_results: list[TrialResult] = []
        print(f"\nRunning concept: {concept}")

        for prompt_idx, prompt in enumerate(tqdm(concept_prompts, desc=f"{concept} prompts")):
            seed = args.seed_base + prompt_idx
            generator = torch.Generator(device=device.type).manual_seed(seed)

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

            scheduler.set_timesteps(args.num_ddim_steps, device=device)
            timesteps = scheduler.timesteps
            latents = torch.randn(
                (1, unet.config.in_channels, 64, 64),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            latents = latents * scheduler.init_noise_sigma

            # Cache the latent after step (capture_step - 1), then branch all perturbation runs from there.
            for step_idx in range(args.capture_step):
                t = timesteps[step_idx]
                latent_input = torch.cat([latents] * 2, dim=0)
                latent_input = scheduler.scale_model_input(latent_input, t)
                with torch.no_grad():
                    noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents_step_before = latents.detach().clone()
            capture_t = timesteps[args.capture_step]
            a_orig_cache: dict[str, torch.Tensor] = {}

            def capture_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
                out = output[0] if isinstance(output, tuple) else output
                cond = out[1]
                a_orig_cache["vec"] = cond.mean(dim=0).detach()
                return output

            capture_handle = target_module.register_forward_hook(capture_hook)
            try:
                latent_input = torch.cat([latents_step_before] * 2, dim=0)
                latent_input = scheduler.scale_model_input(latent_input, capture_t)
                with torch.no_grad():
                    _ = unet(latent_input, capture_t, encoder_hidden_states=text_embeddings).sample
            finally:
                capture_handle.remove()

            if "vec" not in a_orig_cache:
                raise RuntimeError(f"Missing a_orig for prompt {prompt_idx} ({concept}).")
            a_orig = a_orig_cache["vec"].detach()
            a_orig_norm = float(torch.linalg.vector_norm(a_orig).item())

            for sigma in SIGMAS:
                for direction_idx in range(args.num_directions):
                    if sigma == 0.0:
                        delta = torch.zeros_like(a_orig)
                    else:
                        direction = torch.randn_like(a_orig)
                        direction_norm = torch.linalg.vector_norm(direction)
                        if float(direction_norm.item()) < 1e-12:
                            direction = torch.ones_like(a_orig)
                            direction_norm = torch.linalg.vector_norm(direction)
                        delta = direction / direction_norm * (sigma * a_orig_norm)

                    perturbation = delta.to(device=device, dtype=dtype)
                    perturbation_norm = float(torch.linalg.vector_norm(perturbation.float()).item())
                    run_latents = latents_step_before.detach().clone()

                    hook_state = {"printed_meta": False, "count": 0}

                    def perturb_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> Any:
                        hook_state["count"] += 1
                        out = extract_attn_tensor(output)
                        if (
                            concept == "horse"
                            and prompt_idx == 0
                            and direction_idx == 0
                            and abs(sigma - 2.0) < 1e-12
                            and not hook_state["printed_meta"]
                        ):
                            print(
                                f"DEBUG run hook fired in steps [{args.perturb_start_step}, {args.perturb_end_step}] "
                                f"type={type(output)} shape={tuple(out.shape)}"
                            )
                            hook_state["printed_meta"] = True
                        modified = out.clone()
                        if modified.shape[0] < 2:
                            raise RuntimeError(f"Expected CFG batch size >= 2 at hook, got {modified.shape[0]}")
                        modified[1] = modified[1] + perturbation.unsqueeze(0)
                        return replace_attn_output(output, modified)

                    for step_idx in range(args.capture_step, args.num_ddim_steps):
                        t = timesteps[step_idx]
                        handle = None
                        if step_idx in perturb_steps:
                            handle = target_module.register_forward_hook(perturb_hook)

                        latent_input = torch.cat([run_latents] * 2, dim=0)
                        latent_input = scheduler.scale_model_input(latent_input, t)
                        with torch.no_grad():
                            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample

                        if handle is not None:
                            handle.remove()

                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        run_latents = scheduler.step(noise_pred, t, run_latents).prev_sample

                    with torch.no_grad():
                        image_tensor = vae.decode(run_latents / vae.config.scaling_factor).sample
                    image_tensor = torch.clamp((image_tensor / 2 + 0.5), 0, 1)

                    image_pil = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
                    concept_prob = concept_probability(
                        clip_module=clip_module,
                        image_pil=image_pil,
                        concept_name=concept,
                        clip_model=clip_model,
                        clip_preprocess=clip_preprocess,
                        device=device,
                    )
                    concept_results.append(
                        TrialResult(
                            prompt_idx=prompt_idx,
                            sigma=float(sigma),
                            direction_idx=direction_idx,
                            concept_prob=concept_prob,
                            perturbation_norm=perturbation_norm,
                        )
                    )

        save_results_csv(raw_dir / f"{concept}_results.csv", concept_results)

        agg = aggregate_by_sigma(concept_results)
        summary_for_print[concept] = agg

        x = agg["perturbation_norm_mean"]
        y = agg["concept_prob_mean"]
        local_slopes = []
        for i in range(len(x) - 1):
            dx = abs(x[i + 1] - x[i])
            if dx > 1e-12:
                local_slopes.append(abs(y[i + 1] - y[i]) / dx)
        empirical_l = float(max(local_slopes)) if local_slopes else 0.0

        k_est = None
        sigmoid_l = None
        sigma_half_est = None
        try:
            p0 = [1.0, float(np.median(x))]
            bounds = ([0.0, float(np.min(x))], [200.0, float(np.max(x)) + 1e-6])
            params, _ = curve_fit(sigmoid_response, x, y, p0=p0, bounds=bounds, maxfev=20000)
            k_est = float(params[0])
            sigma_half_est = float(params[1])
            sigmoid_l = float(k_est / 4.0)
        except Exception as exc:
            print(f"[WARN] Sigmoid fit failed for {concept}: {exc}")

        lipschitz_json["concepts"][concept] = {
            "empirical_local_max": empirical_l,
            "sigmoid_k": k_est,
            "sigmoid_sigma_half": sigma_half_est,
            "sigmoid_max_slope_k_over_4": sigmoid_l,
            "x_norms": x.tolist(),
            "y_prob_means": y.tolist(),
        }
        print(
            f"{concept}: empirical L={empirical_l:.6f}, "
            f"sigmoid L={sigmoid_l if sigmoid_l is not None else 'nan'}"
        )

    perturbation_l_values = [
        float(v.get("empirical_local_max", 0.0))
        for v in lipschitz_json["concepts"].values()
        if isinstance(v, dict)
    ]
    l_perturbation = max(perturbation_l_values) if perturbation_l_values else 0.0
    delta_bounds = {}
    for eps in [0.01, 0.05, 0.1, 0.2]:
        key = f"eps_{eps:g}"
        delta_bounds[key] = float((1.0 - eps) / l_perturbation) if l_perturbation > 1e-12 else None
    lipschitz_json["L_perturbation"] = float(l_perturbation)
    lipschitz_json["delta_bounds"] = delta_bounds

    # Shared perturbation-response figure.
    plt.figure(figsize=(8, 5))
    for concept in CONCEPTS:
        agg = summary_for_print[concept]
        x = agg["perturbation_norm_mean"]
        y = agg["concept_prob_mean"]
        s = np.nan_to_num(agg["concept_prob_std"], nan=0.0)
        plt.plot(x, y, marker="o", label=concept)
        plt.fill_between(x, np.clip(y - s, 0.0, 1.0), np.clip(y + s, 0.0, 1.0), alpha=0.2)
    plt.xlabel(r"Perturbation magnitude $||\delta||_2$", fontsize=16)
    plt.ylabel("Concept probability", fontsize=16)
    plt.ylim(0.0, 1.0)
    plt.title("Experiment 2: Perturbation-Response Curves", fontsize=17)
    plt.tick_params(axis="both", which="major", labelsize=13)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_lipschitz.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    (out_dir / "lipschitz_estimates.json").write_text(json.dumps(lipschitz_json, indent=2), encoding="utf-8")

    # Console summary table.
    print("\nSummary table")
    print("sigma  | ||delta||  | P(horse) mean ± std | P(car) mean ± std")
    print("-------|-----------|---------------------|-------------------")
    horse_agg = summary_for_print["horse"]
    car_agg = summary_for_print["car"]
    for idx, sigma in enumerate(SIGMAS):
        delta_mean = 0.5 * (
            float(horse_agg["perturbation_norm_mean"][idx]) + float(car_agg["perturbation_norm_mean"][idx])
        )
        print(
            f"{sigma:>5.2f} | {delta_mean:>9.3f} | "
            f"{horse_agg['concept_prob_mean'][idx]:.3f} ± {horse_agg['concept_prob_std'][idx]:.3f}     | "
            f"{car_agg['concept_prob_mean'][idx]:.3f} ± {car_agg['concept_prob_std'][idx]:.3f}"
        )

    print("\nPerturbation Lipschitz summary")
    print(f"Empirical L_perturbation (max local slope): {l_perturbation:.6f}")
    for eps_key, bound in delta_bounds.items():
        eps_display = eps_key.replace("eps_", "")
        if bound is None:
            print(f"eps={eps_display}: Delta bound undefined (L_perturbation ~ 0)")
        else:
            print(f"eps={eps_display}: Delta >= {bound:.4f}")

    print("\nSaved:")
    print(f"  - {raw_dir / 'horse_results.csv'}")
    print(f"  - {raw_dir / 'car_results.csv'}")
    print(f"  - {out_dir / 'fig_lipschitz.pdf'}")
    print(f"  - {out_dir / 'lipschitz_estimates.json'}")


if __name__ == "__main__":
    main()
