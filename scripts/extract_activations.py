#!/usr/bin/env python3
"""Extract concept activation vectors from SD v1.4 cross-attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm


CONCEPTS: dict[str, str] = {
    "horse": "high",
    "deer": "high",
    "zebra": "high",
    "cat": "high",
    "dog": "high",
    "bird": "medium",
    "fish": "medium",
    "flower": "medium",
    "car": "low",
    "castle": "low",
}

EXPECTED_NEIGHBORS: dict[str, list[str]] = {
    "horse": ["deer", "zebra", "dog"],
    "cat": ["dog"],
    "deer": ["horse", "zebra"],
    "zebra": ["horse", "deer"],
    "dog": ["cat"],
    "bird": [],
    "fish": [],
    "flower": [],
    "car": [],
    "castle": [],
}

CONCEPTS_LIST = ["horse", "deer", "zebra", "cat", "dog", "bird", "fish", "flower", "car", "castle"]

SHARED_TEMPLATES = [
    "a photo of a {concept}",
    "a {concept} in a field",
    "a {concept} near a river",
    "a {concept} at sunset",
    "a {concept} in the rain",
    "a {concept} on a hillside",
    "a {concept} in a forest",
    "a {concept} by the ocean",
    "a {concept} under a cloudy sky",
    "a {concept} in bright sunlight",
    "a {concept} in the snow",
    "a {concept} at night",
    "a {concept} in a meadow",
    "a {concept} near mountains",
    "a {concept} on a dirt road",
    "a {concept} beside a lake",
    "a {concept} in fog",
    "a {concept} during golden hour",
    "a {concept} in a garden",
    "a {concept} on a beach",
    "close-up of a {concept}",
    "wide shot of a {concept} in a valley",
    "a {concept} in an open landscape",
    "a {concept} surrounded by trees",
    "a {concept} in the distance",
    "a {concept} in the foreground of a dramatic scene",
    "a {concept} under a stormy sky",
    "a {concept} on a quiet street",
    "a {concept} in warm afternoon light",
    "a {concept} reflected in water",
    "a {concept} in a rural setting",
    "a {concept} in an urban environment",
    "a {concept} on a bridge",
    "a {concept} next to old stone walls",
    "a {concept} in a wide open space",
    "a {concept} under autumn leaves",
    "a {concept} in spring",
    "a {concept} at dawn",
    "a {concept} at dusk",
    "a {concept} in heavy rain",
    "a {concept} covered in morning dew",
    "a {concept} in a misty landscape",
    "a {concept} beside wildflowers",
    "a {concept} in tall grass",
    "a {concept} on a rocky cliff",
    "a {concept} with a blue sky background",
    "a {concept} in a desert landscape",
    "a {concept} near a waterfall",
    "a {concept} in a dark environment",
    "a {concept} lit from behind",
]


def build_all_prompts(num_prompts_per_concept: int) -> dict[str, list[str]]:
    if num_prompts_per_concept > len(SHARED_TEMPLATES):
        raise ValueError(
            f"Requested {num_prompts_per_concept} prompts, but only {len(SHARED_TEMPLATES)} shared templates exist."
        )
    if num_prompts_per_concept <= 0:
        raise ValueError("num_prompts_per_concept must be > 0")
    return {
        concept: [t.format(concept=concept) for t in SHARED_TEMPLATES[:num_prompts_per_concept]]
        for concept in CONCEPTS_LIST
    }


def find_attn2_module(unet: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    print("Searching UNet modules for cross-attention candidates:")
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in unet.named_modules():
        if "up_blocks.1" in name and "attn2" in name:
            print(f"  {name}: {type(module).__name__}")
            candidates.append((name, module))

    preferred = [c for c in candidates if "mid_block.attentions.0" in c[0]] # up_blocks.1.attentions.1
    if preferred:
        filtered = [c for c in preferred if c[0].endswith("attn2")]
        if filtered:
            return filtered[0]
        return preferred[0]

    ending = [c for c in candidates if c[0].endswith("attn2")]
    if ending:
        return ending[0]

    if candidates:
        return candidates[0]

    raise RuntimeError("Could not locate an attn2 module under up_blocks.1 in this UNet.")


def extract_activations(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    output_dir = Path(args.output_dir)
    activations_root = output_dir / "activations"
    sanity_root = output_dir / "sanity_check"
    activations_root.mkdir(parents=True, exist_ok=True)
    sanity_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    vae = pipe.vae

    target_module_name, target_module = find_attn2_module(unet)
    print(f"Using activation module: {target_module_name}")

    all_prompts = build_all_prompts(num_prompts_per_concept=args.num_prompts_per_concept)
    (output_dir / "prompts.json").write_text(json.dumps(all_prompts, indent=2), encoding="utf-8")

    target_steps = sorted(set(args.target_steps))
    all_activations: dict[int, dict[str, list[torch.Tensor]]] = {
        step: {concept: [] for concept in CONCEPTS}
        for step in target_steps
    }

    activation_cache: dict[int, torch.Tensor] = {}
    activation_dim: int | None = None

    for concept, prompts in all_prompts.items():
        concept_sanity_dir = sanity_root / concept
        concept_sanity_dir.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(tqdm(prompts, desc=f"{concept}", leave=False)):
            seed = args.seed_base + i
            generator = torch.Generator(device=device.type).manual_seed(seed)
            activation_cache.clear()

            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_embeddings = text_encoder(text_input.input_ids)[0]

            uncond_input = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                uncond_embeddings = text_encoder(uncond_input.input_ids)[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

            latents = torch.randn(
                (1, unet.config.in_channels, 64, 64),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            pipe.scheduler.set_timesteps(args.num_ddim_steps, device=device)
            latents = latents * pipe.scheduler.init_noise_sigma

            for step_idx, t in enumerate(pipe.scheduler.timesteps):
                hook_handle = None
                if step_idx in target_steps:

                    def _hook_fn(_: torch.nn.Module, __: tuple[Any, ...], output: Any, s: int = step_idx) -> None:
                        out = output[0] if isinstance(output, tuple) else output
                        cond_output = out[1:2]
                        pooled = cond_output.mean(dim=1).squeeze(0)
                        activation_cache[s] = pooled.detach().cpu().float()

                    hook_handle = target_module.register_forward_hook(_hook_fn)

                latent_input = torch.cat([latents] * 2, dim=0)
                latent_input = pipe.scheduler.scale_model_input(latent_input, t)

                with torch.no_grad():
                    noise_pred = unet(
                        latent_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                    ).sample

                if hook_handle is not None:
                    hook_handle.remove()

                if step_idx in target_steps:
                    if step_idx not in activation_cache:
                        raise RuntimeError(
                            f"Missing activation for step {step_idx} (prompt idx={i}, concept={concept})."
                        )
                    all_activations[step_idx][concept].append(activation_cache[step_idx].clone())
                    if activation_dim is None:
                        activation_dim = int(activation_cache[step_idx].numel())

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if i < args.sanity_images_per_concept:
                with torch.no_grad():
                    image_tensor = vae.decode(latents / vae.config.scaling_factor).sample
                image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
                image.save(concept_sanity_dir / f"{i}.png")

    for step in target_steps:
        step_dir = activations_root / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for concept in CONCEPTS:
            vectors = all_activations[step][concept]
            if len(vectors) != args.num_prompts_per_concept:
                raise RuntimeError(
                    f"Expected {args.num_prompts_per_concept} vectors for {concept}@step{step}, got {len(vectors)}."
                )
            tensor = torch.stack(vectors, dim=0)
            torch.save(tensor, step_dir / f"{concept}.pt")

    if activation_dim is None:
        raise RuntimeError("No activations were captured; check hook module selection.")

    metadata = {
        "model": args.model_id,
        "layer": target_module_name,
        "ddim_steps": args.num_ddim_steps,
        "guidance_scale": args.guidance_scale,
        "target_steps": target_steps,
        "activation_dim": activation_dim,
        "num_prompts_per_concept": args.num_prompts_per_concept,
        "concepts": CONCEPTS,
        "expected_neighbors": EXPECTED_NEIGHBORS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved activations to: {activations_root}")
    print(f"Saved prompts to: {output_dir / 'prompts.json'}")
    print(f"Saved metadata to: {output_dir / 'metadata.json'}")
    print(f"Saved sanity images to: {sanity_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=str, default="outputs/experiment1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--target_steps", type=int, nargs="+", default=[10, 25, 40])
    parser.add_argument("--num_prompts_per_concept", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--sanity_images_per_concept", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_activations(args)


if __name__ == "__main__":
    main()
