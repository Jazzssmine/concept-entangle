#!/usr/bin/env python3
"""
Compare a STEREO adversarial placeholder token's embedding against regular words
using the attacked text encoder from a given iteration.

Two comparison modes:
  1. TOKEN-LEVEL  : cosine similarity between the placeholder's input embedding
                    and each word's input embedding (averaged over sub-tokens).
  2. PROMPT-LEVEL : cosine similarity between mean-pooled last_hidden_state of
                    full prompts, e.g. "a photo of a token_xxx" vs
                    "a photo of a horse".
python compare_embeddings.py \
  --output_dir /work/hdd/bcxt/anon3/stereo_weights/horse \
  --iteration 1 \
  --token token_yrru7zku \
  --words "horse,pony,stallion,riding,saddle,horseshoe,jockey,car,dog,chair" \
  --encoder attacked

python compare_embeddings.py \
  --output_dir /work/hdd/bcxt/anon3/stereo_weights/horse \
  --iteration 1 \
  --token token_yrru7zku \
  --words "horse,riding,saddle,pony,car,dog" \
  --encoder original

"""

import argparse
import os

import torch
import torch.nn.functional as F

from utils.utils import StableDiffuser


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_placeholder_embedding(token: str, tokenizer, text_encoder, device: str) -> torch.Tensor:
    """Look up the placeholder token directly by id in the input embedding matrix."""
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise ValueError(f"Placeholder token '{token}' not found in tokenizer vocabulary.")
    idx = torch.tensor([vocab[token]], device=device)
    return text_encoder.get_input_embeddings()(idx).squeeze(0)  # [768]


def get_word_embedding(word: str, tokenizer, text_encoder, device: str) -> tuple[torch.Tensor, int]:
    """
    Return the input embedding for a word.
    If the word is split into multiple sub-tokens, average their embeddings.
    Returns (embedding [768], n_subtokens).
    """
    ids = tokenizer(word, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise ValueError(f"'{word}' produced no tokens.")
    idx = torch.tensor(ids, device=device)
    embs = text_encoder.get_input_embeddings()(idx)  # [n, 768]
    return embs.mean(dim=0), len(ids)


def encode_prompt(prompt: str, tokenizer, text_encoder, device: str) -> torch.Tensor:
    """
    Encode a full prompt with the text encoder.
    Returns the mean-pooled last_hidden_state as a single vector [768].
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)
    hidden = text_encoder(**inputs).last_hidden_state  # [1, seq_len, 768]
    return hidden.squeeze(0).mean(dim=0)  # [768]


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def token_level_similarity(
    placeholder: str,
    words: list[str],
    tokenizer,
    text_encoder,
    device: str,
) -> list[tuple[str, float, int]]:
    """
    Cosine similarity: placeholder input embedding vs each word's input embedding.
    Words that map to multiple sub-tokens have their embeddings averaged.
    Returns sorted list of (word, cosine_sim, n_subtokens).
    """
    ph_emb = F.normalize(get_placeholder_embedding(placeholder, tokenizer, text_encoder, device).unsqueeze(0), dim=-1)

    results = []
    for word in words:
        try:
            word_emb, n_sub = get_word_embedding(word, tokenizer, text_encoder, device)
            sim = F.cosine_similarity(ph_emb, F.normalize(word_emb.unsqueeze(0), dim=-1)).item()
            results.append((word, sim, n_sub))
        except ValueError as e:
            print(f"  [skip] {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def prompt_level_similarity(
    anchor_prompt: str,
    comparison_prompts: list[str],
    tokenizer,
    text_encoder,
    device: str,
) -> list[tuple[str, float]]:
    """
    Cosine similarity between mean-pooled last_hidden_state of full prompts.
    Returns sorted list of (prompt, cosine_sim).
    """
    anchor = F.normalize(encode_prompt(anchor_prompt, tokenizer, text_encoder, device).unsqueeze(0), dim=-1)

    results = []
    for prompt in comparison_prompts:
        emb = F.normalize(encode_prompt(prompt, tokenizer, text_encoder, device).unsqueeze(0), dim=-1)
        sim = F.cosine_similarity(anchor, emb).item()
        results.append((prompt, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare a STEREO placeholder token to regular words in embedding space."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing iteration checkpoints")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration index")
    parser.add_argument("--token", type=str, required=True, help="Placeholder token, e.g. token_yrru7zku")
    parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Comma-separated words to compare against, e.g. 'horse,riding,saddle,car,dog'",
    )
    parser.add_argument("--generic_prompt", type=str, default="a photo of a", help="Prompt prefix for prompt-level comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--text_encoder_ckpt",
        type=str,
        default=None,
        help="Override text encoder checkpoint filename (relative to output_dir)",
    )
    args = parser.parse_args()

    text_encoder_ckpt = args.text_encoder_ckpt or f"ci_attack_text_encoder_iteration_{args.iteration}.pt"
    text_encoder_path = os.path.join(args.output_dir, text_encoder_ckpt)
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Text encoder checkpoint not found: {text_encoder_path}")

    # Load base diffuser (we only need tokenizer + text encoder, no UNet)
    diffuser = StableDiffuser(scheduler="DDIM").to(args.device)

    # Add all placeholder tokens up to this iteration so vocabulary sizes match
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
    else:
        all_tokens = [args.token]

    for t in all_tokens:
        if t not in diffuser.tokenizer.get_vocab():
            diffuser.tokenizer.add_tokens([t])
    diffuser.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))

    diffuser.text_encoder.load_state_dict(
        torch.load(text_encoder_path, map_location=args.device)
    )
    diffuser.text_encoder.eval()

    words = [w.strip() for w in args.words.split(",") if w.strip()]

    with torch.no_grad():
        # --- Token-level ---
        tok_results = token_level_similarity(
            args.token, words, diffuser.tokenizer, diffuser.text_encoder, args.device
        )
        print(f"\n{'='*56}")
        print("TOKEN-LEVEL similarity  (input embedding matrix)")
        print(f"Placeholder : '{args.token}'")
        print(f"{'='*56}")
        print(f"{'Word':<25} {'#subtokens':>10} {'Cosine Sim':>12}")
        print("-" * 49)
        for word, sim, n_sub in tok_results:
            print(f"{word:<25} {n_sub:>10} {sim:>12.4f}")

        # --- Prompt-level ---
        anchor_prompt = f"{args.generic_prompt} {args.token}"
        comp_prompts = [f"{args.generic_prompt} {w}" for w in words]
        prom_results = prompt_level_similarity(
            anchor_prompt, comp_prompts, diffuser.tokenizer, diffuser.text_encoder, args.device
        )
        print(f"\n{'='*56}")
        print("PROMPT-LEVEL similarity  (mean-pooled last_hidden_state)")
        print(f"Anchor : '{anchor_prompt}'")
        print(f"{'='*56}")
        print(f"{'Comparison prompt':<45} {'Cosine Sim':>12}")
        print("-" * 59)
        for prompt, sim in prom_results:
            print(f"{prompt:<45} {sim:>12.4f}")


if __name__ == "__main__":
    main()
