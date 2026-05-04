"""
Compare word embedding similarities between:

1) CLIP text encoder space (StableDiffuser.text_encoder last_hidden_state)
2) An UNet-derived, step-specific cross-attention "contribution" space from an (optionally) unlearned UNet

This script:
- Reads a prompts .txt file (one prompt per line).
- Extracts unique "words" (unicode-aware \\w tokens).
- For each word:
  - CLIP vector: mean of non-special, non-padding token embeddings from CLIP last_hidden_state
  - UNet vector at a chosen denoising step: mean over that word's tokens of
      contribution(token) = mean_heads&queries(attn_probs[..., token]) * to_v(token_embedding)
    captured from a single cross-attention layer (first attn2 found).
- Produces cosine similarity matrices for both spaces and saves them.

Notes:
- STEREO unlearning updates the UNet (and sometimes only its attention layers). The CLIP text encoder is typically unchanged.
- The UNet-derived vectors depend on the chosen step and the latents used at that step.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch

from utils.utils import StableDiffuser


def read_prompts_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_unique_words(lines: List[str]) -> List[str]:
    words = set()
    for line in lines:
        # unicode-aware word tokens (keeps digits/underscore too; fine for prompt files)
        for w in re.findall(r"[\w']+", line):
            ww = w.strip()
            if ww:
                words.add(ww)
    return sorted(words)


def get_word_token_mask(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Return boolean mask over sequence positions for "real" tokens to average.
    Excludes BOS/EOS/PAD if present.
    """
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)

    mask = attention_mask.bool().clone()
    if bos is not None:
        mask &= input_ids.ne(bos)
    if eos is not None:
        mask &= input_ids.ne(eos)
    if pad is not None:
        mask &= input_ids.ne(pad)
    return mask


def clip_word_vector(diffuser: StableDiffuser, word: str, device: str) -> torch.Tensor:
    tokens = diffuser.text_tokenize([word])
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    hidden = diffuser.text_encoder(input_ids).last_hidden_state  # (1, seq, 768)
    mask = get_word_token_mask(diffuser.tokenizer, input_ids, attention_mask)[0]  # (seq,)
    if mask.sum() == 0:
        # fall back to mean over all (rare)
        return hidden[0].mean(dim=0)
    return hidden[0][mask].mean(dim=0)


@dataclass
class CapturedAttn:
    attn_probs: torch.Tensor  # (b, heads, q, k)
    value_tokens: torch.Tensor  # (b, k, inner_dim)


class CapturingAttnProcessor:
    """
    Drop-in attention processor that captures attention probs and per-token values.
    Designed for diffusers' Attention modules.
    """

    def __init__(self):
        self.captured: CapturedAttn | None = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if getattr(attn, "norm_cross", False):
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # attention mask prep
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # store token-space values before head reshaping (b, k, inner_dim)
        value_tokens = value

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # reshape captured tensors for later aggregation
        heads = getattr(attn, "heads", None)
        if heads is None:
            # best-effort fallback: infer heads from attn.to_q out features and head_dim
            heads = 1
        q_len = attention_probs.shape[1]
        k_len = attention_probs.shape[2]
        attn_probs_4d = attention_probs.view(batch_size, heads, q_len, k_len)

        self.captured = CapturedAttn(
            attn_probs=attn_probs_4d.detach().cpu(),
            value_tokens=value_tokens.detach().cpu(),
        )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # output projection
        to_out = attn.to_out
        if isinstance(to_out, (list, tuple)) or to_out.__class__.__name__ == "ModuleList":
            hidden_states = to_out[0](hidden_states)
            hidden_states = to_out[1](hidden_states)
        else:
            hidden_states = to_out(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


def find_first_cross_attn2(unet) -> Tuple[str, object]:
    for name, mod in unet.named_modules():
        # Stable Diffusion UNet uses "attn2" for cross-attn blocks
        if name.endswith("attn2"):
            return name, mod
    raise RuntimeError("Could not find a cross-attention module ending with 'attn2' in the UNet.")


def get_latents_at_step(
    diffuser: StableDiffuser,
    step_idx: int,
    n_steps: int,
    seed: int,
    device: str,
    img_size: int = 512,
) -> torch.Tensor:
    """
    Create initial latents and advance them to the requested iteration using an empty prompt.
    This yields a more realistic latent state for that step than pure random latents.
    """
    assert 0 <= step_idx < n_steps

    diffuser.set_scheduler_timesteps(n_steps)
    # StableDiffuser.get_noise() uses torch.randn() without an explicit device,
    # so it creates CPU noise and therefore requires a CPU generator.
    gen = torch.Generator().manual_seed(seed)
    latents = diffuser.get_initial_latents(1, img_size, 1, generator=gen).to(device)

    # Empty prompt conditioning (CLIP space)
    empty_tokens = diffuser.text_tokenize([""])
    empty_hidden = diffuser.text_encoder(empty_tokens.input_ids.to(device)).last_hidden_state

    with torch.no_grad():
        for i in range(step_idx):
            t = diffuser.scheduler.timesteps[i]
            latents_in = diffuser.scheduler.scale_model_input(latents, t)
            noise_pred = diffuser.unet(latents_in, t, encoder_hidden_states=empty_hidden).sample
            latents = diffuser.scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def unet_word_vector_at_step(
    diffuser: StableDiffuser,
    latents_at_step: torch.Tensor,
    step_idx: int,
    n_steps: int,
    word: str,
    attn2_module,
    processor: CapturingAttnProcessor,
    device: str,
) -> torch.Tensor:
    """
    Compute an UNet-derived vector for `word` at a denoising step by:
    - running UNet once at that step (no CFG) with prompt=word
    - capturing a single cross-attn layer's attention probs and to_v(token) values
    - returning mean over tokens of (token_weight * token_value)
    """
    diffuser.set_scheduler_timesteps(n_steps)
    t = diffuser.scheduler.timesteps[step_idx]

    tokens = diffuser.text_tokenize([word])
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    encoder_hidden_states = diffuser.text_encoder(input_ids).last_hidden_state  # (1, seq, 768)

    processor.captured = None
    with torch.no_grad():
        latents_in = diffuser.scheduler.scale_model_input(latents_at_step, t)
        _ = diffuser.unet(latents_in, t, encoder_hidden_states=encoder_hidden_states).sample

    if processor.captured is None:
        raise RuntimeError("Attention processor did not capture anything; cross-attn module may not have been invoked.")

    attn_probs = processor.captured.attn_probs.to(device)  # (1, heads, q, k)
    value_tokens = processor.captured.value_tokens.to(device)  # (1, k, inner_dim)

    # token weights: average over heads and queries -> (1, k)
    token_w = attn_probs.mean(dim=(1, 2))  # (1, k)
    token_w = token_w.unsqueeze(-1)  # (1, k, 1)

    contrib = token_w * value_tokens  # (1, k, inner_dim)

    # select the tokens that correspond to the word (exclude specials/pad)
    mask = get_word_token_mask(diffuser.tokenizer, input_ids, attention_mask)[0]  # (seq,)
    if mask.sum() == 0:
        return contrib[0].mean(dim=0)
    return contrib[0][mask].mean(dim=0)


def cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return x @ x.t()


def write_neighbors_csv(words: List[str], sim: torch.Tensor, out_csv: str, topk: int = 10) -> None:
    sim = sim.cpu()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "neighbor_rank", "neighbor", "cosine_sim"])
        for i, word in enumerate(words):
            vals = sim[i].clone()
            vals[i] = -1.0
            idx = torch.topk(vals, k=min(topk, len(words) - 1)).indices.tolist()
            for r, j in enumerate(idx, start=1):
                w.writerow([word, r, words[j], float(sim[i, j])])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-word similarities to 'horse' in CLIP and UNet spaces, "
            "and compare original vs unlearned UNet embeddings."
        )
    )
    parser.add_argument("--prompts_txt", type=str, required=True, help="Path to prompts .txt (one prompt per line)")
    parser.add_argument(
        "--unet_ckpt",
        type=str,
        required=True,
        help="Path to unlearned UNet checkpoint (e.g. final_reo_unet.pt)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="emb_compare_out")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40, 45],
        help="Denoising step indices (0..n_steps-1) to use for UNet-derived embeddings",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="Total DDIM steps (must match how you interpret `step`)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used to generate the base latents")
    parser.add_argument("--max_words", type=int, default=300, help="Max unique words to process (sorted order)")
    parser.add_argument("--horse_word", type=str, default="horse", help="Reference word to compare against")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Original (pre-unlearn) model
    diffuser_orig = StableDiffuser(scheduler="DDIM").to(args.device)
    # Unlearned model
    diffuser_unlearn = StableDiffuser(scheduler="DDIM").to(args.device)
    diffuser_unlearn.unet.load_state_dict(torch.load(args.unet_ckpt, map_location=args.device))

    lines = read_prompts_txt(args.prompts_txt)
    words = extract_unique_words(lines)
    if len(words) > args.max_words:
        words = words[: args.max_words]

    # Ensure the reference word (e.g. "horse") is included
    if args.horse_word not in words:
        words.append(args.horse_word)
    horse_idx = words.index(args.horse_word)

    # Choose and patch a single cross-attn module to capture from for each model.
    attn_name_orig, attn2_orig = find_first_cross_attn2(diffuser_orig.unet)
    processor_orig = CapturingAttnProcessor()
    if hasattr(attn2_orig, "set_processor"):
        attn2_orig.set_processor(processor_orig)
    else:
        attn2_orig.processor = processor_orig

    attn_name_unlearn, attn2_unlearn = find_first_cross_attn2(diffuser_unlearn.unet)
    processor_unlearn = CapturingAttnProcessor()
    if hasattr(attn2_unlearn, "set_processor"):
        attn2_unlearn.set_processor(processor_unlearn)
    else:
        attn2_unlearn.processor = processor_unlearn

    # CLIP embeddings are independent of denoising step: compute once.
    clip_vecs = [clip_word_vector(diffuser_orig, w, args.device).detach().cpu() for w in words]
    clip_mat = torch.stack(clip_vecs, dim=0)  # (N, d_clip)
    clip_horse = clip_mat[horse_idx]

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float()
        b = b.float()
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    # For each requested step, compute UNet-derived similarities and write a CSV.
    for step_idx in args.steps:
        # Use the same latents for both models at this step for fair comparison.
        latents_at_step = get_latents_at_step(
            diffuser=diffuser_orig,
            step_idx=step_idx,
            n_steps=args.n_steps,
            seed=args.seed,
            device=args.device,
        )

        unet_orig_vecs = []
        unet_unlearn_vecs = []

        for w in words:
            unet_orig_vecs.append(
                unet_word_vector_at_step(
                    diffuser=diffuser_orig,
                    latents_at_step=latents_at_step,
                    step_idx=step_idx,
                    n_steps=args.n_steps,
                    word=w,
                    attn2_module=attn2_orig,
                    processor=processor_orig,
                    device=args.device,
                ).detach().cpu()
            )
            unet_unlearn_vecs.append(
                unet_word_vector_at_step(
                    diffuser=diffuser_unlearn,
                    latents_at_step=latents_at_step,
                    step_idx=step_idx,
                    n_steps=args.n_steps,
                    word=w,
                    attn2_module=attn2_unlearn,
                    processor=processor_unlearn,
                    device=args.device,
                ).detach().cpu()
            )

        unet_orig_mat = torch.stack(unet_orig_vecs, dim=0)  # (N, d_unet)
        unet_unlearn_mat = torch.stack(unet_unlearn_vecs, dim=0)  # (N, d_unet)
        unet_orig_horse = unet_orig_mat[horse_idx]
        unet_unlearn_horse = unet_unlearn_mat[horse_idx]

        # Prepare a CSV per step with all requested similarities
        out_csv = os.path.join(args.output_dir, f"word_similarities_vs_horse_step{step_idx}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "word",
                    "step",
                    "clip_sim_to_horse",
                    "orig_unet_sim_to_horse",
                    "unlearn_unet_sim_to_horse",
                    "orig_vs_unlearn_unet_sim",
                ]
            )
            for i, w in enumerate(words):
                clip_sim = cos(clip_mat[i], clip_horse)
                orig_unet_sim = cos(unet_orig_mat[i], unet_orig_horse)
                unlearn_unet_sim = cos(unet_unlearn_mat[i], unet_unlearn_horse)
                orig_vs_unlearn_sim = cos(unet_orig_mat[i], unet_unlearn_mat[i])
                writer.writerow([w, step_idx, clip_sim, orig_unet_sim, unlearn_unet_sim, orig_vs_unlearn_sim])


if __name__ == "__main__":
    main()

"""python compare_prompt_word_embeddings.py \
  --prompts_txt prompts/random_test.txt \
  --unet_ckpt stereo_weights/horse/final_reo_unet.pt \
  --steps 0 10 20 30 40 45 \
  --n_steps 50 \
  --output_dir emb_compare_out \
  --horse_word horse"""