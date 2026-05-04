"""
Compare for the stereo model:
1. Cross-attention patterns induced by cue A (e.g. "q" or placeholder token) vs cue B (e.g. "horse")
2. Denoising activations induced by cue A vs cue B

Usage:
  python compare_cue_attention_and_activations.py \
    --cue_a "q" --cue_b "horse" \
    --unet_ckpt stereo_weights/horse/final_reo_unet.pt \
    --output_dir cue_compare_out

  # With placeholder token (add to tokenizer first):
  python compare_cue_attention_and_activations.py \
    --cue_a "token_abc12345" --cue_b "horse" \
    --unet_ckpt stereo_weights/horse/final_reo_unet.pt \
    --placeholder_tokens token_abc12345
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.utils import StableDiffuser


# --- Cross-attention capture (adapted from compare_prompt_word_embeddings.py) ---

class CapturingAttnProcessor:
    """Captures attention probs and value tokens from cross-attention."""

    def __init__(self):
        self.captured = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
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

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        value_tokens = value

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        heads = getattr(attn, "heads", 1)
        q_len, k_len = attention_probs.shape[1], attention_probs.shape[2]
        attn_probs_4d = attention_probs.view(batch_size, heads, q_len, k_len)

        self.captured = {
            "attn_probs": attn_probs_4d.detach().cpu(),
            "value_tokens": value_tokens.detach().cpu(),
        }

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

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


def find_all_cross_attn2(unet) -> List[Tuple[str, object]]:
    """Return all (name, module) for cross-attention layers."""
    result = []
    for name, mod in unet.named_modules():
        if name.endswith("attn2"):
            result.append((name, mod))
    return result


def get_latents_at_step(
    diffuser: StableDiffuser,
    step_idx: int,
    n_steps: int,
    seed: int,
    device: str,
    img_size: int = 512,
) -> torch.Tensor:
    """Advance latents to the requested step using empty prompt."""
    assert 0 <= step_idx < n_steps
    diffuser.set_scheduler_timesteps(n_steps)
    gen = torch.Generator().manual_seed(seed)
    latents = diffuser.get_initial_latents(1, img_size, 1, generator=gen).to(device)
    empty_tokens = diffuser.text_tokenize([""])
    empty_hidden = diffuser.text_encoder(empty_tokens.input_ids.to(device)).last_hidden_state

    with torch.no_grad():
        for i in range(step_idx):
            t = diffuser.scheduler.timesteps[i]
            latents_in = diffuser.scheduler.scale_model_input(latents, t)
            noise_pred = diffuser.unet(latents_in, t, encoder_hidden_states=empty_hidden).sample
            latents = diffuser.scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def capture_cross_attn_for_cue(
    diffuser: StableDiffuser,
    latents_at_step: torch.Tensor,
    step_idx: int,
    n_steps: int,
    cue: str,
    attn_processors: Dict[str, CapturingAttnProcessor],
    device: str,
) -> Dict[str, dict]:
    """Run UNet forward with cue and return captured attention from all patched layers."""
    diffuser.set_scheduler_timesteps(n_steps)
    t = diffuser.scheduler.timesteps[step_idx]

    tokens = diffuser.text_tokenize([cue])
    input_ids = tokens.input_ids.to(device)
    encoder_hidden_states = diffuser.text_encoder(input_ids).last_hidden_state

    for proc in attn_processors.values():
        proc.captured = None

    with torch.no_grad():
        latents_in = diffuser.scheduler.scale_model_input(latents_at_step, t)
        _ = diffuser.unet(latents_in, t, encoder_hidden_states=encoder_hidden_states).sample

    result = {}
    for name, proc in attn_processors.items():
        if proc.captured is not None:
            result[name] = {
                "attn_probs": proc.captured["attn_probs"],
                "value_tokens": proc.captured["value_tokens"],
            }
    return result


def compare_attention_patterns(
    attn_a: torch.Tensor, attn_b: torch.Tensor
) -> Dict[str, float]:
    """Compare two attention tensors (b, heads, q, k)."""
    a = attn_a.float().flatten()
    b = attn_b.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    corr = np.corrcoef(a.numpy(), b.numpy())[0, 1] if a.numel() > 1 else float("nan")
    mse = torch.nn.functional.mse_loss(attn_a.float(), attn_b.float()).item()
    return {"cosine_sim": cos_sim, "pearson_corr": float(corr), "mse": mse}


# --- Denoising activation capture ---

class ActivationCapture:
    """Captures activations from registered hooks."""

    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self.activations[name] = out.detach().cpu().flatten()

        return hook

    def register(self, unet, layer_names: List[str]):
        """Register hooks on unet for given layer names."""
        name_to_module = {n: m for n, m in unet.named_modules()}
        for name in layer_names:
            if name in name_to_module:
                h = name_to_module[name].register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def find_activation_layers(unet) -> List[str]:
    """Return a subset of UNet layer names useful for activation comparison."""
    layers = []
    for name, mod in unet.named_modules():
        # Include first resnet and first attn in each block for diversity
        if "down_blocks" in name and ("resnets.0" in name or "attentions.0" in name):
            if "attn2" in name or "resnets" in name:
                layers.append(name)
        elif "mid_block" in name and ("attentions.0" in name or "resnets.0" in name):
            layers.append(name)
        elif "up_blocks" in name and ("resnets.0" in name or "attentions.0" in name):
            if "attn2" in name or "resnets" in name:
                layers.append(name)
    # Deduplicate and limit to avoid memory blowup
    seen = set()
    result = []
    for L in layers:
        if L not in seen and len(result) < 12:  # ~12 layers
            seen.add(L)
            result.append(L)
    return result


def capture_activations_for_cue(
    diffuser: StableDiffuser,
    latents_at_step: torch.Tensor,
    step_idx: int,
    n_steps: int,
    cue: str,
    capture: ActivationCapture,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Run UNet forward and return captured activations."""
    diffuser.set_scheduler_timesteps(n_steps)
    t = diffuser.scheduler.timesteps[step_idx]

    tokens = diffuser.text_tokenize([cue])
    encoder_hidden_states = diffuser.text_encoder(tokens.input_ids.to(device)).last_hidden_state

    capture.clear()
    with torch.no_grad():
        latents_in = diffuser.scheduler.scale_model_input(latents_at_step, t)
        _ = diffuser.unet(latents_in, t, encoder_hidden_states=encoder_hidden_states).sample

    return dict(capture.activations)


def compare_activations(acts_a: Dict[str, torch.Tensor], acts_b: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Compare activation dicts layer by layer."""
    result = {}
    for name in acts_a:
        if name not in acts_b:
            continue
        a, b = acts_a[name].float(), acts_b[name].float()
        if a.shape != b.shape:
            result[name] = {"cosine_sim": float("nan"), "pearson_corr": float("nan"), "mse": float("nan")}
            continue
        cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        an, bn = a.numpy(), b.numpy()
        corr = np.corrcoef(an, bn)[0, 1] if an.size > 1 else float("nan")
        mse = torch.nn.functional.mse_loss(a, b).item()
        result[name] = {"cosine_sim": cos_sim, "pearson_corr": float(corr), "mse": mse}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cross-attention patterns and denoising activations for two cues (e.g. q vs horse)."
    )
    parser.add_argument("--cue_a", type=str, default="q", help="First cue (e.g. placeholder token or 'q')")
    parser.add_argument("--cue_b", type=str, default="horse", help="Second cue (e.g. 'horse')")
    parser.add_argument(
        "--unet_ckpt",
        type=str,
        default=None,
        help="Path to unlearned UNet checkpoint. If None, uses original SD.",
    )
    parser.add_argument(
        "--placeholder_tokens",
        type=str,
        nargs="*",
        default=[],
        help="Placeholder tokens to add to tokenizer (e.g. from STEREO)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="cue_compare_out")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 10, 25, 40, 45])
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_attn_layers", type=int, default=4, help="Number of cross-attn layers to capture")
    parser.add_argument("--capture_activations", action="store_true", help="Also capture and compare UNet activations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    diffuser = StableDiffuser(scheduler="DDIM").to(args.device)
    if args.unet_ckpt:
        diffuser.unet.load_state_dict(torch.load(args.unet_ckpt, map_location=args.device))

    for tok in args.placeholder_tokens:
        if tok not in diffuser.tokenizer.get_vocab():
            diffuser.tokenizer.add_tokens([tok])
            diffuser.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))

    # Patch cross-attention layers
    all_attn = find_all_cross_attn2(diffuser.unet)
    attn_to_use = all_attn[: args.num_attn_layers]
    attn_processors = {}
    for name, mod in attn_to_use:
        proc = CapturingAttnProcessor()
        attn_processors[name] = proc
        if hasattr(mod, "set_processor"):
            mod.set_processor(proc)
        else:
            mod.processor = proc

    # Optional: activation capture
    act_capture = None
    act_layer_names = []
    if args.capture_activations:
        act_capture = ActivationCapture()
        act_layer_names = find_activation_layers(diffuser.unet)
        act_capture.register(diffuser.unet, act_layer_names)

    all_results = []

    for step_idx in args.steps:
        latents_at_step = get_latents_at_step(
            diffuser=diffuser,
            step_idx=step_idx,
            n_steps=args.n_steps,
            seed=args.seed,
            device=args.device,
        )

        attn_a = capture_cross_attn_for_cue(
            diffuser, latents_at_step, step_idx, args.n_steps, args.cue_a, attn_processors, args.device
        )
        attn_b = capture_cross_attn_for_cue(
            diffuser, latents_at_step, step_idx, args.n_steps, args.cue_b, attn_processors, args.device
        )

        step_results = {"step": step_idx, "attention_comparison": {}}
        for name in attn_a:
            if name in attn_b:
                cmp = compare_attention_patterns(
                    attn_a[name]["attn_probs"], attn_b[name]["attn_probs"]
                )
                step_results["attention_comparison"][name] = cmp

        if act_capture is not None:
            acts_a = capture_activations_for_cue(
                diffuser, latents_at_step, step_idx, args.n_steps, args.cue_a, act_capture, args.device
            )
            acts_b = capture_activations_for_cue(
                diffuser, latents_at_step, step_idx, args.n_steps, args.cue_b, act_capture, args.device
            )
            step_results["activation_comparison"] = compare_activations(acts_a, acts_b)

        all_results.append(step_results)

    if act_capture is not None:
        act_capture.remove_hooks()

    # Save JSON summary
    out_json = os.path.join(args.output_dir, f"compare_{args.cue_a}_vs_{args.cue_b}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "cue_a": args.cue_a,
                "cue_b": args.cue_b,
                "unet_ckpt": args.unet_ckpt,
                "steps": args.steps,
                "results": all_results,
            },
            f,
            indent=2,
        )

    # Print summary
    print(f"\nSaved to {out_json}")
    print(f"\nCross-attention pattern comparison ({args.cue_a} vs {args.cue_b}):")
    for r in all_results:
        print(f"  Step {r['step']}:")
        for layer, cmp in r.get("attention_comparison", {}).items():
            short = layer.split(".")[-3] + "." + layer.split(".")[-2] + "." + layer.split(".")[-1]
            print(f"    {short}: cos_sim={cmp['cosine_sim']:.4f}, corr={cmp['pearson_corr']:.4f}")

    if args.capture_activations and "activation_comparison" in all_results[0]:
        print(f"\nDenoising activation comparison ({args.cue_a} vs {args.cue_b}):")
        for r in all_results[:2]:  # first 2 steps
            print(f"  Step {r['step']}:")
            for layer, cmp in list(r["activation_comparison"].items())[:4]:
                print(f"    {layer}: cos_sim={cmp['cosine_sim']:.4f}")


if __name__ == "__main__":
    main()
