#!/usr/bin/env python3
"""
AdvUnlearn-oriented causal recovery runner.

This wraps the shared causal intervention logic and adds clean support for
base-vs-erased TEXT ENCODER comparisons (with optional erased UNet override).

Key differences from the Stereo-oriented script:
- `--erased_unet_checkpoint` is optional.
- Base and erased text encoders/tokenizers can be loaded independently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import causal_recovery_experiments as core


def _pick_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        # Direct state-dict case.
        if all(isinstance(v, torch.Tensor) for v in obj.values()) and obj:
            return obj  # type: ignore[return-value]
        # Common nested wrappers.
        for k in ("state_dict", "text_encoder", "model", "module", "weights"):
            if k in obj and isinstance(obj[k], dict):
                nested = obj[k]
                if nested and all(isinstance(v, torch.Tensor) for v in nested.values()):
                    return nested  # type: ignore[return-value]
    raise ValueError("Unsupported text-encoder checkpoint format.")


def _strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    known_prefixes = [
        "text_encoder.",
        "module.text_encoder.",
        "cond_stage_model.transformer.",
        "module.cond_stage_model.transformer.",
        "module.",
    ]
    out = dict(state_dict)
    changed = True
    # Repeat stripping while one known prefix uniformly applies.
    while changed:
        changed = False
        keys = list(out.keys())
        for pref in known_prefixes:
            if keys and all(k.startswith(pref) for k in keys):
                out = {k[len(pref) :]: v for k, v in out.items()}
                changed = True
                break
    return out


def _load_optional_erased_unet(
    pipe: core.StableDiffusionPipeline, checkpoint_path: str, device: str
) -> None:
    if not checkpoint_path:
        return
    state = torch.load(checkpoint_path, map_location=device)
    pipe.unet.load_state_dict(state)
    pipe.unet.eval()


def _load_text_encoder(
    pipe: core.StableDiffusionPipeline,
    text_encoder_path: str,
    tokenizer_path: str,
    text_encoder_subfolder: str,
    placeholder_tokens: list[str],
    device: str,
) -> None:
    if tokenizer_path:
        pipe.tokenizer = pipe.tokenizer.__class__.from_pretrained(tokenizer_path)

    if not text_encoder_path:
        return

    te_path = Path(text_encoder_path)
    if te_path.is_dir():
        kwargs = {}
        if text_encoder_subfolder:
            kwargs["subfolder"] = text_encoder_subfolder
        pipe.text_encoder = pipe.text_encoder.__class__.from_pretrained(text_encoder_path, **kwargs).to(
            device
        )
        pipe.text_encoder.eval()
        for p in pipe.text_encoder.parameters():
            p.requires_grad_(False)
        return

    # File checkpoint path (state dict style). Accepts raw or wrapped/prefixed state dicts.
    raw_obj = torch.load(text_encoder_path, map_location=device)
    state_dict = _strip_state_dict_prefixes(_pick_state_dict(raw_obj))

    emb_key = "text_model.embeddings.token_embedding.weight"
    if emb_key in state_dict:
        expected_vocab = int(state_dict[emb_key].shape[0])
        if expected_vocab < len(pipe.tokenizer):
            raise RuntimeError(
                f"Checkpoint expects vocab {expected_vocab}, tokenizer has {len(pipe.tokenizer)}."
            )
        # Expand tokenizer to match TE embedding matrix size.
        to_add = expected_vocab - len(pipe.tokenizer)
        added = 0
        for tok in placeholder_tokens:
            if added >= to_add:
                break
            if tok not in pipe.tokenizer.get_vocab():
                pipe.tokenizer.add_tokens([tok])
                added += 1
        while len(pipe.tokenizer) < expected_vocab:
            pipe.tokenizer.add_tokens([f"__pad_token_{len(pipe.tokenizer)}__"])
        pipe.text_encoder.resize_token_embeddings(expected_vocab)

    missing, unexpected = pipe.text_encoder.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] unexpected TE keys ignored: {len(unexpected)}")
    if missing:
        print(f"[warn] missing TE keys (kept from base TE): {len(missing)}")
    pipe.text_encoder.eval()
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal recovery (steering + transplant) adapted for AdvUnlearn"
    )
    parser.add_argument("--base_model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument(
        "--erased_unet_checkpoint",
        type=str,
        default="",
        help="Optional erased UNet checkpoint. Leave empty for text-encoder-only erased model.",
    )
    parser.add_argument("--output_dir", type=str, default="analysis_causal_recovery_advunlearn")
    parser.add_argument("--prompt_config", type=str, required=True, help="JSON with A/B/(C)/D/E/F lists")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--timesteps_to_intervene", type=str, default="15")
    parser.add_argument("--layers_to_capture", type=str, default="mid_block.attentions.0")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--alphas", type=str, default="0,10,20,30,40,50,60")
    parser.add_argument("--betas", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--run_steering", action="store_true")
    parser.add_argument("--run_transplant", action="store_true")

    parser.add_argument("--base_direction_path", type=str, default="")
    parser.add_argument("--erased_direction_path", type=str, default="")

    # Separate base/erased text encoder controls.
    parser.add_argument("--base_text_encoder_path", type=str, default="")
    parser.add_argument(
        "--base_text_encoder_subfolder",
        type=str,
        default="",
        help="Optional HF subfolder when --base_text_encoder_path points to a model repo/dir.",
    )
    parser.add_argument("--base_tokenizer_path", type=str, default="")

    parser.add_argument("--erased_text_encoder_path", type=str, default="")
    parser.add_argument(
        "--erased_text_encoder_subfolder",
        type=str,
        default="",
        help="Optional HF subfolder when --erased_text_encoder_path points to a model repo/dir.",
    )
    parser.add_argument("--erased_tokenizer_path", type=str, default="")

    parser.add_argument("--enable_s2_tokens", action="store_true")
    parser.add_argument("--debug_mode", action="store_true", help="Small prompt subset")
    parser.add_argument("--debug_n_per_pos_group", type=int, default=2)
    parser.add_argument("--debug_n_controls", type=int, default=2)
    parser.add_argument("--debug_hook_logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.run_steering and not args.run_transplant:
        args.run_steering = True
        args.run_transplant = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_groups = core.load_prompt_config(args.prompt_config)
    prompt_df = core.build_prompt_table(
        prompt_groups=prompt_groups,
        enable_s2_tokens=args.enable_s2_tokens,
        debug_mode=args.debug_mode,
        debug_n_per_pos_group=args.debug_n_per_pos_group,
        debug_n_controls=args.debug_n_controls,
    )
    all_prompts = prompt_df["prompt"].tolist()
    placeholder_tokens = core._extract_tokens_from_prompts(all_prompts)

    # Both pipelines start from same base model, then diverge by optional UNet/TE swaps.
    base_pipe = core.load_base_pipeline(args.base_model_path, args.device)
    erased_pipe = core.load_base_pipeline(args.base_model_path, args.device)
    _load_optional_erased_unet(erased_pipe, args.erased_unet_checkpoint, args.device)

    _load_text_encoder(
        base_pipe,
        text_encoder_path=args.base_text_encoder_path,
        tokenizer_path=args.base_tokenizer_path,
        text_encoder_subfolder=args.base_text_encoder_subfolder,
        placeholder_tokens=placeholder_tokens,
        device=args.device,
    )
    _load_text_encoder(
        erased_pipe,
        text_encoder_path=args.erased_text_encoder_path,
        tokenizer_path=args.erased_tokenizer_path,
        text_encoder_subfolder=args.erased_text_encoder_subfolder,
        placeholder_tokens=placeholder_tokens,
        device=args.device,
    )

    requested_layers = [x.strip() for x in args.layers_to_capture.split(",") if x.strip()]
    base_layers = core.resolve_target_layers(base_pipe.unet, requested_layers)
    erased_layers = core.resolve_target_layers(erased_pipe.unet, requested_layers)
    layers = [n for n in base_layers if n in set(erased_layers)]
    print("Matched intervention layers:")
    for ln in layers:
        print(f"  - {ln}")
    if not layers:
        raise RuntimeError("No common matched layers for interventions.")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    target_steps = [int(x.strip()) for x in args.timesteps_to_intervene.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]

    core.write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "num_prompts": int(len(prompt_df)),
            "matched_layers": layers,
        },
    )

    if args.run_steering:
        direction_sources: dict[str, torch.Tensor] = {}
        if args.base_direction_path:
            direction_sources["base"] = core.load_direction_vector(args.base_direction_path)
        if args.erased_direction_path:
            direction_sources["erased"] = core.load_direction_vector(args.erased_direction_path)
        if not direction_sources:
            raise ValueError("Steering requested but no direction path provided.")

        core.run_activation_steering(
            erased_pipe=erased_pipe,
            prompt_df=prompt_df,
            seeds=seeds,
            hookpoints=layers,
            target_steps=target_steps,
            alphas=alphas,
            direction_sources=direction_sources,
            args=args,
            out_dir=out_dir,
        )

    if args.run_transplant:
        core.run_activation_transplant(
            base_pipe=base_pipe,
            erased_pipe=erased_pipe,
            prompt_df=prompt_df,
            seeds=seeds,
            hookpoints=layers,
            target_steps=target_steps,
            betas=betas,
            args=args,
            out_dir=out_dir,
        )

    print(f"Done. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
python causal_recovery_adv.py \
  --run_transplant \
  --base_model_path "CompVis/stable-diffusion-v1-4" \
  --prompt_config "/u/anon3/unlearn_diff/stereo/prompts/causal_prompt_config.json" \
  --erased_text_encoder_path /u/anon3/unlearn_diff/advunlearn/results/results_with_retaining/horse/coco_object/fast_at/AttackLr_0.001/text_encoder_full/all/prefix_k/AdvUnlearn-horse-method_text_encoder_full_all-Attack_fast_at-Retain_coco_object_reg_0.3-lr_1e-05-AttackLr_0.001-prefix_k_adv_num_1-word_embd-attack_init_random-attack_step_5-adv_update_1-warmup_iter_200/models/TextEncoder-text_encoder_full-epoch_399.pt \
  --erased_text_encoder_subfolder "horse_unlearned" \
  --layers_to_capture "mid_block.attentions.0" \
  --timesteps_to_intervene "15" \
  --seeds "0,1,2" \
  --output_dir "analysis_causal_recovery_transplant"
"""