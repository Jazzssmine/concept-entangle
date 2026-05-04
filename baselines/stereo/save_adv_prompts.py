#!/usr/bin/env python3
"""
Reconstruct and save adversarial prompts from an existing STEREO checkpoint.
Run this without re-training if you already have ste_stage_model.pt.
"""

import argparse
import json
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Save adversarial prompts from existing STEREO checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing ste_stage_model.pt")
    parser.add_argument("--generic_prompt", type=str, default="a photo of a", help="Generic prompt used during training")
    parser.add_argument("--erase_concept", type=str, required=True, help="Initial erase concept used during training (e.g. horse)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save adv_prompts.json")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.output_dir, "ste_stage_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_tokens = ckpt.get("saved_tokens", {})
    if not saved_tokens:
        raise ValueError(f"No saved_tokens in checkpoint. Keys found: {list(ckpt.keys())}")

    # Reconstruct adv_prompts per iteration (same logic as search_thoroughly_enough)
    adv_prompts = {}
    current_concept = args.erase_concept

    for iteration in range(len(saved_tokens)):
        iteration_prompts = []
        iteration_prompts.append(f"{args.generic_prompt} {current_concept}")
        for token in saved_tokens.values():
            iteration_prompts.append(f"{args.generic_prompt} {token}")
        adv_prompts[f"iteration_{iteration}"] = iteration_prompts
        current_concept = saved_tokens[str(iteration)]

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "adv_prompts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(adv_prompts, f, indent=2, ensure_ascii=False)

    print(f"Adversarial prompts saved to {out_path}")
    print(f"Placeholder tokens: {saved_tokens}")


if __name__ == "__main__":
    main()
