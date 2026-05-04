#!/usr/bin/env python3
"""
Generate images for a learned STEREO placeholder token and optionally measure
its distance to other words in the text encoder embedding space.

This loads the erased UNet + the attacked text encoder for a given iteration,
then runs inference for prompts like: "{generic_prompt} {token}".
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.utils import StableDiffuser


def get_word_embedding(word: str, tokenizer, text_encoder, device: str) -> torch.Tensor:
    """
    Return the embedding for a word using the input embedding matrix.
    If the word is split into multiple sub-tokens, average their embeddings.
    """
    ids = tokenizer(word, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise ValueError(f"'{word}' produced no tokens.")
    idx = torch.tensor(ids, device=device)
    embs = text_encoder.get_input_embeddings()(idx)  # [n_subtokens, 768]
    return embs.mean(dim=0)  # average if word splits into multiple pieces


def get_placeholder_embedding(token: str, tokenizer, text_encoder, device: str) -> torch.Tensor:
    """
    Return the embedding of a placeholder token directly by its token id.
    Raises if the token is not in the vocabulary.
    """
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise ValueError(f"Placeholder token '{token}' not found in tokenizer vocabulary.")
    idx = torch.tensor([vocab[token]], device=device)
    return text_encoder.get_input_embeddings()(idx).squeeze(0)  # [768]


def token_level_similarity(
    placeholder: str,
    words: list[str],
    tokenizer,
    text_encoder,
    device: str,
) -> list[tuple[str, float, int]]:
    """
    Compute cosine similarity between the placeholder token embedding and each word's
    embedding (averaged over sub-tokens if needed). Returns a sorted list of
    (word, cosine_sim, n_subtokens).
    """
    placeholder_emb = get_placeholder_embedding(placeholder, tokenizer, text_encoder, device)
    placeholder_emb = F.normalize(placeholder_emb.unsqueeze(0), dim=-1)

    results = []
    for word in words:
        try:
            word_emb = get_word_embedding(word, tokenizer, text_encoder, device)
            n_sub = len(tokenizer(word, add_special_tokens=False).input_ids)
            sim = F.cosine_similarity(placeholder_emb, F.normalize(word_emb.unsqueeze(0), dim=-1)).item()
            results.append((word, sim, n_sub))
        except ValueError as e:
            print(f"  [skip] {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def prompt_level_similarity(
    placeholder_prompt: str,
    comparison_prompts: list[str],
    tokenizer,
    text_encoder,
    device: str,
) -> list[tuple[str, float]]:
    """
    Compute cosine similarity between full prompt embeddings.
    Each prompt is encoded with text_encoder and its last_hidden_state is mean-pooled
    over the sequence dimension to produce a single vector.
    """

    def encode_prompt(prompt: str) -> torch.Tensor:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(device)
        with torch.no_grad():
            hidden = text_encoder(**inputs).last_hidden_state  # [1, seq_len, 768]
        return hidden.squeeze(0).mean(dim=0)  # [768]

    anchor = F.normalize(encode_prompt(placeholder_prompt).unsqueeze(0), dim=-1)

    results = []
    for prompt in comparison_prompts:
        emb = F.normalize(encode_prompt(prompt).unsqueeze(0), dim=-1)
        sim = F.cosine_similarity(anchor, emb).item()
        results.append((prompt, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def nearest_neighbors_in_vocab(token: str, tokenizer, text_encoder, device: str, top_k: int = 20):
    """Find the top-k nearest words in the full tokenizer vocabulary by cosine similarity."""
    placeholder_emb = get_placeholder_embedding(token, tokenizer, text_encoder, device)

    all_embeddings = text_encoder.get_input_embeddings().weight  # [vocab_size, 768]
    token_emb_norm = F.normalize(placeholder_emb.unsqueeze(0), dim=-1)
    all_norm = F.normalize(all_embeddings, dim=-1)
    sims = (all_norm @ token_emb_norm.T).squeeze(-1)  # [vocab_size]

    top_vals, top_ids = torch.topk(sims, k=top_k + 10)

    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    print(f"\nNearest neighbors of '{token}' in full vocabulary (top {top_k}):")
    print(f"{'Token':<30} {'Cosine Sim':>12}")
    print("-" * 44)
    shown = 0
    for val, idx in zip(top_vals.tolist(), top_ids.tolist()):
        word = id_to_token.get(idx, f"<id:{idx}>")
        if word == token:
            continue
        print(f"{word:<30} {val:>12.4f}")
        shown += 1
        if shown >= top_k:
            break


def main():
    parser = argparse.ArgumentParser(description="Evaluate a STEREO placeholder token mapping")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing iteration checkpoints")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration index (used to pick default ckpt filenames)")
    parser.add_argument("--token", type=str, required=True, help="Placeholder token string, e.g. token_yrru7zku")
    parser.add_argument("--generic_prompt", type=str, default="a photo of a", help="Generic prompt prefix")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")

    parser.add_argument("--unet_ckpt", type=str, default=None, help="Override UNet ckpt filename (relative to output_dir)")
    parser.add_argument(
        "--text_encoder_ckpt",
        type=str,
        default=None,
        help="Override text encoder ckpt filename (relative to output_dir)",
    )

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--n_imgs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/images/eval/placeholder_token_eval",
        help="Where to save generated images (relative to current working dir)",
    )
    parser.add_argument(
        "--compare_words",
        type=str,
        default=None,
        help="Comma-separated list of words to compare against the token embedding, e.g. 'horse,dog,car'",
    )
    parser.add_argument(
        "--nearest_neighbors",
        action="store_true",
        help="Find top-k nearest tokens in the full vocabulary (no image generation needed)",
    )
    parser.add_argument("--top_k", type=int, default=20, help="How many nearest neighbors to show")
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip image generation (useful when only doing embedding comparison)",
    )
    parser.add_argument(
        "--probe_words",
        type=str,
        default=None,
        help=(
            "Comma-separated regular words to probe the erased model with, e.g. 'riding,horseshoe,stallion'. "
            "Uses the erased UNet but the ORIGINAL (unattacked) text encoder to test whether the erasure holds."
        ),
    )

    args = parser.parse_args()

    unet_ckpt = args.unet_ckpt or f"erased_unet_iteration_{args.iteration}.pt"
    text_encoder_ckpt = args.text_encoder_ckpt or f"ci_attack_text_encoder_iteration_{args.iteration}.pt"

    unet_path = os.path.join(args.output_dir, unet_ckpt)
    text_encoder_path = os.path.join(args.output_dir, text_encoder_ckpt)

    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"UNet checkpoint not found: {unet_path}")
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Text encoder checkpoint not found: {text_encoder_path}")

    diffuser = StableDiffuser(scheduler="DDIM").to(args.device)

    # Load saved_tokens from ste_stage_model.pt so we can add ALL iteration tokens.
    # The text encoder checkpoint for iteration N was saved after N+1 tokens were added
    # (one per iteration), so the vocab must be resized to match before loading weights.
    ste_model_path = os.path.join(args.output_dir, "ste_stage_model.pt")
    all_tokens = []
    if os.path.exists(ste_model_path):
        ckpt = torch.load(ste_model_path, map_location="cpu")
        saved_tokens = ckpt.get("saved_tokens", {})
        # Add tokens for iterations 0..args.iteration (inclusive)
        for idx in range(args.iteration + 1):
            t = saved_tokens.get(str(idx))
            if t:
                all_tokens.append(t)
        del ckpt
    else:
        # Fallback: just add the single requested token
        all_tokens = [args.token]

    for t in all_tokens:
        if t not in diffuser.tokenizer.get_vocab():
            diffuser.tokenizer.add_tokens([t])
    diffuser.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))

    diffuser.unet.load_state_dict(torch.load(unet_path, map_location=args.device))
    diffuser.text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=args.device))
    diffuser.eval()

    # --- Probe erased model with regular words using the ATTACKED text encoder ---
    # This tests whether related words, when processed by the attacked text encoder,
    # can also bypass the erasure.
    if args.probe_words:
        probe_out_dir = Path(args.out_dir) / f"iter_{args.iteration}_probe"
        probe_out_dir.mkdir(parents=True, exist_ok=True)

        words = [w.strip() for w in args.probe_words.split(",") if w.strip()]
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        print(f"\nProbing erased model (attacked text encoder) with {len(words)} word(s)...")
        for word in words:
            prompt = f"{args.generic_prompt} {word}"
            with torch.no_grad():
                images = diffuser(
                    prompt,
                    img_size=args.img_size,
                    n_steps=args.n_steps,
                    n_imgs=args.n_imgs,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                )
            for i, img in enumerate(images):
                fname = probe_out_dir / f"{prompt.replace(' ', '_')}_{i}.png"
                img[0].save(fname)
            print(f"  Saved {args.n_imgs} images for prompt: '{prompt}'")
        print(f"Probe images saved to {probe_out_dir}")
        if args.no_images:
            return

    # --- Embedding comparisons (no GPU-heavy inference needed) ---
    with torch.no_grad():
        if args.nearest_neighbors:
            nearest_neighbors_in_vocab(
                args.token, diffuser.tokenizer, diffuser.text_encoder, args.device, top_k=args.top_k
            )

        if args.compare_words:
            words = [w.strip() for w in args.compare_words.split(",") if w.strip()]

            # --- Token-level: placeholder embedding vs each word's (averaged) embedding ---
            tok_results = token_level_similarity(
                args.token, words, diffuser.tokenizer, diffuser.text_encoder, args.device
            )
            print(f"\n{'='*56}")
            print(f"TOKEN-LEVEL similarity  (input embedding matrix)")
            print(f"Placeholder : '{args.token}'")
            print(f"{'='*56}")
            print(f"{'Word':<25} {'#subtokens':>10} {'Cosine Sim':>12}")
            print("-" * 49)
            for word, sim, n_sub in tok_results:
                print(f"{word:<25} {n_sub:>10} {sim:>12.4f}")

            # --- Prompt-level: mean-pooled last_hidden_state of full prompts ---
            anchor_prompt = f"{args.generic_prompt} {args.token}"
            comp_prompts = [f"{args.generic_prompt} {w}" for w in words]
            prom_results = prompt_level_similarity(
                anchor_prompt, comp_prompts, diffuser.tokenizer, diffuser.text_encoder, args.device
            )
            print(f"\n{'='*56}")
            print(f"PROMPT-LEVEL similarity  (mean-pooled last_hidden_state)")
            print(f"Anchor prompt: '{anchor_prompt}'")
            print(f"{'='*56}")
            print(f"{'Comparison prompt':<45} {'Cosine Sim':>12}")
            print("-" * 59)
            for prompt, sim in prom_results:
                print(f"{prompt:<45} {sim:>12.4f}")

    if args.no_images:
        return

    # --- Image generation ---
    out_dir = Path(args.out_dir) / f"iter_{args.iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = f"{args.generic_prompt} {args.token}"
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    with torch.no_grad():
        images = diffuser(
            prompt,
            img_size=args.img_size,
            n_steps=args.n_steps,
            n_imgs=args.n_imgs,
            generator=generator,
            guidance_scale=args.guidance_scale,
        )

    for i, img in enumerate(images):
        img[0].save(out_dir / f"{prompt.replace(' ', '_')}_{i}.png")

    print(f"Prompt: {prompt}")
    print(f"Saved {args.n_imgs} images to {out_dir}")


if __name__ == "__main__":
    main()

