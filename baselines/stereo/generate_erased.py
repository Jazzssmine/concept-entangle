#!/usr/bin/env python3
"""
Generate images using the attacked text encoder from a STEREO iteration
checkpoint (with custom tokenizer / placeholder tokens).

By default loads the erased UNet for that iteration. Use --base_unet to keep
the original pretrained UNet while still using the attacked text encoder.

Mirrors what inference_and_save() does for erased_images when using erased UNet:
    diffuser("a photo of a horse", ...)
but exposes it as a standalone script with configurable prompts.

python generate_erased.py \
  --output_dir "/work/hdd/bcxt/anon3/stereo_weights/horse" \
  --iteration 1 \
  --unet_ckpt /work/hdd/bcxt/anon3/stereo_weights/horse/final_reo_unet.pt \
  --prompt "an animal used in racing with a rider" \
  --generic_prompt "" \
  --n_imgs 5

  --prompt_file prompts/words_ratio_gt1.txt \

python generate_erased.py \
  --output_dir /work/hdd/bcxt/anon3/stereo_weights/horse \
  --out_dir data/images/eval/base_model \
  --iteration 1 \
  --base_unet \
  --generic_prompt "a photo of a" \
  --prompt "token_yrru7zku" \
    --n_imgs 20
"""

import argparse
import os
import re
from pathlib import Path

import torch

from utils.utils import StableDiffuser


def sanitize_filename(text: str) -> str:
    text = "_".join(text.split())
    text = re.sub(r"[^A-Za-z0-9_]", "", text)
    return text[:180] if text else "prompt"


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append(line)
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate images with the attacked text encoder (+ optional erased UNet). "
            "Use --base_unet for pretrained UNet + attacked encoder only."
        )
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing iteration checkpoints (erased_unet_*, ci_attack_text_encoder_*)")
    parser.add_argument("--iteration", type=int, required=True,
                        help="Iteration index to load checkpoints from")
    parser.add_argument("--prompt", type=str, default="",
                        help="Single prompt, e.g. 'a photo of a horse'")
    parser.add_argument("--prompt_file", type=str, default="",
                        help="Text file with one prompt per line")
    parser.add_argument("--generic_prompt", type=str, default="",
                        help="If set, prepend this prefix to each prompt word, "
                             "e.g. --generic_prompt 'a photo of a' --prompt 'horse'")
    parser.add_argument("--out_dir", type=str,
                        default="data/images/eval/erased_model",
                        help="Where to save generated images")
    parser.add_argument("--n_imgs", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--base_unet",
        action="store_true",
        help="Do not load any UNet checkpoint; use the original pretrained UNet with the attacked text encoder.",
    )
    parser.add_argument("--unet_ckpt", type=str, default=None,
                        help="UNet checkpoint path (relative to output_dir unless absolute). Ignored if --base_unet.")
    parser.add_argument("--text_encoder_ckpt", type=str, default=None,
                        help="Override text encoder checkpoint filename (relative to output_dir)")
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        raise ValueError("Provide either --prompt or --prompt_file.")
    if args.prompt and args.prompt_file:
        raise ValueError("Use only one of --prompt or --prompt_file.")

    prompts = load_prompts(args.prompt_file) if args.prompt_file else [args.prompt]
    if args.generic_prompt:
        prompts = [f"{args.generic_prompt} {p}" for p in prompts]

    # Resolve checkpoint paths — absolute paths are used as-is, relative ones are
    # joined with output_dir
    def resolve(ckpt, default):
        p = ckpt or default
        return p if os.path.isabs(p) else os.path.join(args.output_dir, p)

    te_path = resolve(args.text_encoder_ckpt, f"ci_attack_text_encoder_iteration_{args.iteration}.pt")
    if not os.path.exists(te_path):
        raise FileNotFoundError(f"Text encoder checkpoint not found: {te_path}")

    unet_path = None
    if not args.base_unet:
        unet_path = resolve(args.unet_ckpt, f"erased_unet_iteration_{args.iteration}.pt")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet checkpoint not found: {unet_path}")

    # Load model
    diffuser = StableDiffuser(scheduler="DDIM").to(args.device)

    # Add all placeholder tokens up to this iteration (keeps vocab size consistent
    # with what the attacked text encoder checkpoint expects)
    ste_model_path = os.path.join(args.output_dir, "ste_stage_model.pt")
    all_tokens = []
    if os.path.exists(ste_model_path):
        ckpt = torch.load(ste_model_path, map_location="cpu")
        saved_tokens = ckpt.get("saved_tokens", {})
        for idx in range(args.iteration + 1):
            t = saved_tokens.get(str(idx))
            if t:
                all_tokens.append(t)
        del ckpt
    # If ste_stage_model.pt is missing we still load the weights; no tokens to add

    for t in all_tokens:
        if t not in diffuser.tokenizer.get_vocab():
            diffuser.tokenizer.add_tokens([t])
    diffuser.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))

    if unet_path is not None:
        diffuser.unet.load_state_dict(torch.load(unet_path, map_location=args.device))
        print(f"Loaded  UNet          : {unet_path}")
    else:
        print("UNet                  : base (pretrained, not loaded from checkpoint)")
    diffuser.text_encoder.load_state_dict(torch.load(te_path, map_location=args.device))
    diffuser.eval()

    print(f"Loaded  text encoder  : {te_path}")
    print(f"Placeholder tokens    : {all_tokens or '(none)'}")

    out_root = Path(args.out_dir) / f"iter_{args.iteration}"
    out_root.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for prompt in prompts:
        print(f"\nGenerating {args.n_imgs} image(s) for: '{prompt}'")
        with torch.no_grad():
            images = diffuser(
                prompt,
                img_size=args.img_size,
                n_steps=args.n_steps,
                n_imgs=args.n_imgs,
                generator=generator,
                guidance_scale=args.guidance_scale,
            )
        torch.cuda.empty_cache()

        fname_base = sanitize_filename(prompt)
        for i, img in enumerate(images):
            out_path = out_root / f"{fname_base}_{i}.png"
            img[0].save(out_path)
            print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
