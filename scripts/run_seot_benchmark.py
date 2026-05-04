from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.abspath("."))

from src.benchmark.prompt_loader import load_prompts, stable_prompt_id_from_fields

from baselines.seot import suppress_eot_w_nulltext as seot


def _parse_seeds(values: list[str]) -> list[int]:
    seeds: list[int] = []
    for v in values:
        if "," in v:
            for x in v.split(","):
                x = x.strip()
                if x:
                    seeds.append(int(x))
        else:
            seeds.append(int(v))
    return seeds


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


def _find_token_indices(tokenizer, prompt: str, target_text: str) -> list[int]:
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target_text)[1:-1]  # strip BOS/EOS
    if not target_ids:
        return []
    max_i = len(prompt_ids) - len(target_ids) + 1
    for i in range(max_i):
        if prompt_ids[i : i + len(target_ids)] == target_ids:
            return list(range(i, i + len(target_ids)))
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SuppressEOT on benchmark prompt CSV.")
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    parser.add_argument("--model_name", type=str, default="seot_horse")
    parser.add_argument("--target_text", type=str, default="horse")
    parser.add_argument("--sd_version", type=str, default="sd_1_4", choices=["sd_1_4", "sd_1_5"])
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seeds", nargs="*", default=["0"])
    parser.add_argument("--image_format", type=str, default="png")
    parser.add_argument("--method", type=str, default="soft-weight")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cross_retain_steps", type=float, default=0.2)
    parser.add_argument("--max_step_to_erase", type=int, default=None)
    parser.add_argument("--iter_each_step", type=int, default=0)
    parser.add_argument("--lambda_retain", type=float, default=1.0)
    parser.add_argument("--lambda_erase", type=float, default=-0.5)
    parser.add_argument("--lambda_self_retain", type=float, default=1.0)
    parser.add_argument("--lambda_self_erase", type=float, default=-0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def _load_prompts_flexible(prompts_path: str, target_text: str) -> pd.DataFrame:
    """
    Accept either:
    1) benchmark schema (all_prompts.csv style) via load_prompts(...)
    2) lightweight schema with at least a `prompt` column (for_generate_example/*.csv)
    """
    try:
        return load_prompts(prompts_path)
    except Exception:
        pass

    raw = pd.read_csv(prompts_path)
    if "prompt" not in raw.columns:
        raise ValueError(
            f"Unsupported prompts file: {prompts_path}. Expected benchmark schema or a CSV with a 'prompt' column."
        )

    prompts = raw["prompt"].astype(str).str.strip()
    prompt_ids = [
        stable_prompt_id_from_fields(
            target_concept=target_text,
            prompt_family="direct",
            intended_label=target_text,
            prompt=p,
            lexical_mode="strict",
        )
        for p in prompts
    ]
    return pd.DataFrame(
        {
            "prompt_id": prompt_ids,
            "target_concept": [target_text] * len(raw),
            "prompt_family": ["direct"] * len(raw),
            "intended_label": [target_text] * len(raw),
            "domain": ["object"] * len(raw),
            "prompt": prompts,
            "lexical_mode": ["strict"] * len(raw),
        }
    )


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    prompts_df = _load_prompts_flexible(args.prompts_path, args.target_text)

    out_dir = Path(args.output_dir)
    images_root = out_dir / "images" / args.model_name
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[seot] loading model on {device}")
    # Keep SEOT internal globals aligned with CLI args.
    seot.NUM_DDIM_STEPS = int(args.num_inference_steps)
    seot.GUIDANCE_SCALE = float(args.guidance_scale)
    max_step_to_erase = (
        int(args.max_step_to_erase)
        if args.max_step_to_erase is not None
        else int(args.num_inference_steps)
    )

    stable = seot.load_model(args.sd_version)
    stable = stable.to(device)
    stable.set_progress_bar_config(disable=True)

    rows: list[dict] = []
    total = len(prompts_df) * len(seeds)
    i = 0

    for _, row in prompts_df.iterrows():
        prompt = str(row["prompt"])
        prompt_id = str(row["prompt_id"])
        target_concept = str(row["target_concept"])
        prompt_family = str(row["prompt_family"])
        intended_label = str(row["intended_label"])
        domain = str(row["domain"])
        lexical_mode = str(row["lexical_mode"])

        for seed in seeds:
            i += 1
            file_name = _deterministic_filename(
                args.model_name,
                target_concept,
                prompt_id,
                seed,
                0,
                args.image_format,
            )
            out_path = images_root / target_concept / prompt_family / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.no_resume or not out_path.exists():
                token_indices = _find_token_indices(stable.tokenizer, prompt, args.target_text)

                if token_indices:
                    n_tokens = len(stable.tokenizer.encode(prompt))
                    controller = seot.AttentionStore(
                        token_indices=token_indices,
                        alpha=args.alpha,
                        method=args.method,
                        cross_retain_steps=args.cross_retain_steps,
                        n=n_tokens,
                        iter_each_step=args.iter_each_step,
                        max_step_to_erase=max_step_to_erase,
                        lambda_retain=args.lambda_retain,
                        lambda_erase=args.lambda_erase,
                        lambda_self_retain=args.lambda_self_retain,
                        lambda_self_erase=args.lambda_self_erase,
                    )
                    generator = torch.Generator(device=stable.device).manual_seed(seed)
                    images, _ = seot.text2image_ldm_stable(
                        stable,
                        [prompt],
                        controller,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        latent=None,
                        uncond_embeddings=None,
                        start_time=args.num_inference_steps,
                        return_type="image",
                    )
                    # index 1 = suppressed image, index 0 = baseline image in this implementation
                    image_np = images[1]
                    Image.fromarray(image_np).save(out_path)
                    status = "generated"
                else:
                    generator = torch.Generator(device=stable.device).manual_seed(seed)
                    image = stable(
                        prompt=prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    ).images[0]
                    image.save(out_path)
                    status = "generated_no_target_token"
            else:
                status = "skipped_existing"

            rows.append(
                {
                    "image_path": str(out_path),
                    "model_name": args.model_name,
                    "target_concept": target_concept,
                    "prompt_family": prompt_family,
                    "intended_label": intended_label,
                    "domain": domain,
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "image_index": 0,
                    "guidance_scale": args.guidance_scale,
                    "num_inference_steps": args.num_inference_steps,
                    "lexical_mode": lexical_mode,
                    "status": status,
                }
            )
            if i % 25 == 0 or i == total:
                print(f"[seot] {i}/{total} done")

    out_csv = metadata_dir / "generated_images.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[seot] wrote metadata: {out_csv}")


if __name__ == "__main__":
    main()
