#!/usr/bin/env python3
"""
Generate images for two prompt sets:

  S1 — natural prompts (horse_leakage_prompts.txt)
       base UNet + base text encoder

  S2 — adversarial prompts (adv_prompts.json, placeholder-token prompts only)
       base UNet + attacked text encoder + extended tokenizer

Usage:
    python generate_s1_s2.py \
        --s1_file prompts/horse_leakage_prompts.txt \
        --s2_file results/adv_prompts.json \
        --attacked_encoder /work/hdd/bcxt/anon3/stereo_weights/horse/ci_attack_text_encoder_iteration_1.pt \
        --output_dir data/images/eval \
        --n_imgs 5

Output layout:
    outputs/
        s1/prompt_000/seed_0.png ... seed_4.png
        s2/prompt_000/seed_0.png ... seed_4.png
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_base_pipeline(model_id: str, device: str) -> StableDiffusionPipeline:
    """Load the base Stable Diffusion pipeline with no modifications."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def prepare_attacked_encoder(
    pipe: StableDiffusionPipeline,
    attacked_encoder_path: str,
    placeholder_tokens: list[str],
    device: str,
) -> None:
    """
    Extend the pipeline's tokenizer with placeholder tokens, resize the text
    encoder embedding matrix to exactly match the checkpoint, then load the
    attacked weights in-place.

    Modifies pipe.tokenizer and pipe.text_encoder. UNet is NOT touched.
    """
    state_dict = torch.load(attacked_encoder_path, map_location=device)

    # Determine the exact vocab size the checkpoint expects
    expected_vocab_size: int = state_dict[
        "text_model.embeddings.token_embedding.weight"
    ].shape[0]
    n_to_add = expected_vocab_size - len(pipe.tokenizer)

    if n_to_add < 0:
        raise RuntimeError(
            f"Checkpoint expects {expected_vocab_size} tokens but tokenizer already "
            f"has {len(pipe.tokenizer)}. Cannot shrink vocabulary."
        )

    # Add only as many placeholder tokens as the checkpoint requires
    added = 0
    for token in placeholder_tokens:
        if added >= n_to_add:
            break
        if token not in pipe.tokenizer.get_vocab():
            pipe.tokenizer.add_tokens([token])
            added += 1

    # If placeholder_tokens list was shorter than needed, pad with dummy tokens
    while len(pipe.tokenizer) < expected_vocab_size:
        dummy = f"__pad_token_{len(pipe.tokenizer)}__"
        pipe.tokenizer.add_tokens([dummy])

    pipe.text_encoder.resize_token_embeddings(expected_vocab_size)
    print(f"  Vocab resized to {expected_vocab_size} (+{n_to_add} token(s) added)")

    pipe.text_encoder.load_state_dict(state_dict)
    pipe.text_encoder.eval()
    print(f"  Loaded attacked text encoder from {attacked_encoder_path}")


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    out_dir: str,
    n_imgs: int = 5,
    n_steps: int = 50,
    guidance_scale: float = 7.5,
    device: str = "cuda",
) -> None:
    """
    Generate n_imgs images per prompt, seeded 0..n_imgs-1 for reproducibility.
    Saves to out_dir/prompt_{idx:03d}/seed_{seed}.png
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        prompt_dir = Path(out_dir) / f"prompt_{idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        pending = [s for s in range(n_imgs) if not (prompt_dir / f"seed_{s}.png").exists()]
        skipped = n_imgs - len(pending)
        skip_note = f" (skipping {skipped} existing)" if skipped else ""
        print(f"  [{idx+1}/{len(prompts)}] {prompt}{skip_note}")

        for seed in pending:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                result = pipe(
                    prompt,
                    num_inference_steps=n_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            result.images[0].save(prompt_dir / f"seed_{seed}.png")

        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_s1_prompts(path: str) -> list[str]:
    """Load natural prompts, one per line, skipping blanks and comments."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def load_s2_prompts(path: str) -> tuple[list[str], list[str]]:
    """
    Load adversarial prompts from adv_prompts.json.
    Returns:
        prompts          — deduplicated list of all prompts across all iterations
        placeholder_tokens — list of placeholder token strings extracted from prompts
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_prompts: list[str] = []
    for iteration_prompts in data.values():
        all_prompts.extend(iteration_prompts)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_prompts: list[str] = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)

    # Extract placeholder tokens (match token_XXXXXXXX pattern)
    token_pattern = re.compile(r"\btoken_\w+\b")
    placeholder_tokens: list[str] = list(
        dict.fromkeys(  # deduplicate, preserve order
            t for p in unique_prompts for t in token_pattern.findall(p)
        )
    )

    return unique_prompts, placeholder_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate S1 (natural) and S2 (adversarial) images for semantic comparison."
    )
    parser.add_argument("--s1_file", type=str, required=True,
                        help="Path to natural prompts text file (one per line)")
    parser.add_argument("--s2_file", type=str, required=True,
                        help="Path to adv_prompts.json")
    parser.add_argument("--attacked_encoder", type=str, required=True,
                        help="Path to attacked text encoder checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Root output directory (default: outputs/)")
    parser.add_argument("--model_id", type=str,
                        default="CompVis/stable-diffusion-v1-4",
                        help="Base Stable Diffusion model id or local path")
    parser.add_argument("--n_imgs", type=int, default=5,
                        help="Images per prompt (seeds 0..n_imgs-1)")
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_s1", action="store_true",
                        help="Skip S1 generation (useful if already done)")
    parser.add_argument("--skip_s2", action="store_true",
                        help="Skip S2 generation")
    args = parser.parse_args()

    s1_out = os.path.join(args.output_dir, "s1")
    s2_out = os.path.join(args.output_dir, "s2")

    # --- Load prompts ---
    s1_prompts = load_s1_prompts(args.s1_file)
    s2_prompts, placeholder_tokens = load_s2_prompts(args.s2_file)

    print(f"S1 prompts : {len(s1_prompts)}")
    print(f"S2 prompts : {len(s2_prompts)}")
    print(f"Placeholder tokens : {placeholder_tokens}")

    # --- S1: base pipeline, no modifications ---
    if not args.skip_s1:
        print("\n=== S1: base UNet + base text encoder ===")
        pipe = load_base_pipeline(args.model_id, args.device)
        generate_images(pipe, s1_prompts, s1_out,
                        n_imgs=args.n_imgs,
                        n_steps=args.n_steps,
                        guidance_scale=args.guidance_scale,
                        device=args.device)
        print(f"S1 images saved to {s1_out}")
        del pipe
        torch.cuda.empty_cache()

    # --- S2: base UNet + attacked text encoder ---
    if not args.skip_s2:
        print("\n=== S2: base UNet + attacked text encoder ===")
        pipe = load_base_pipeline(args.model_id, args.device)
        prepare_attacked_encoder(pipe, args.attacked_encoder, placeholder_tokens, args.device)
        generate_images(pipe, s2_prompts, s2_out,
                        n_imgs=args.n_imgs,
                        n_steps=args.n_steps,
                        guidance_scale=args.guidance_scale,
                        device=args.device)
        print(f"S2 images saved to {s2_out}")
        del pipe
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
