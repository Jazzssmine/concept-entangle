#!/usr/bin/env python3
"""
Internal representation analysis for STEREO-erased Stable Diffusion.

Purpose
-------
This script runs only two experiments (no activation steering yet):

1) Linear probe experiment:
   Capture cross-attention activations from UNet during denoising, pool them
   into feature vectors, and test whether horse-related prompts are linearly
   separable from benign prompts.

2) Mean-difference direction experiment:
   For each model/layer/timestep, compute horse-vs-benign mean feature
   differences as candidate "horse directions" for later steering work.

Why base vs erased?
-------------------
Comparing separability and direction geometry between the base model and final
STEREO-erased model reveals whether horse-sensitive information remains
linearly decodable in erased UNet internals.

python internal_representation_probe.py \
  --base_model_path CompVis/stable-diffusion-v1-4 \
  --erased_unet_checkpoint /work/hdd/bcxt/anon3/stereo_weights/horse/final_reo_unet.pt \
  --output_dir analysis_internal_probe \
  --num_inference_steps 50 \
  --timesteps_to_capture 45,30,15 \
  --layers_to_capture down_blocks.1.attentions.0,mid_block.attentions.0,up_blocks.1.attentions.0 \
  --seeds 0,1,2,3,4 \
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------- Prompt setup ----------------------------- #

PROMPT_GROUPS_DEFAULT: dict[str, list[str]] = {
    "A": [
        "a close-up portrait of a chestnut horse with a braided mane",
        "a gray horse drinking water from a shallow stream",
        "a horse standing beside a red barn at sunrise",
        "a wild horse galloping across a dry plain",
        "a horse grazing under tall pine trees",
    ],
    "B": [
        "a jockey leaning forward during a high-speed race finish",
        "a polo player swinging a mallet near the goal line",
        "a rider clearing a jump in an indoor equestrian arena",
        "a trail rider crossing a small wooden bridge in the countryside",
        "a racing saddle and reins prepared at a training paddock",
    ],
    "C": [
        "a photo of token_xaji0y6d",
        "a photo of token_yrru7zku",
    ],
    "D": [
        "a husky sprinting through fresh snow",
        "a tabby cat curled up on a window seat",
        "a fox standing near ferns in a woodland clearing",
        "a flock of sheep grazing on a hillside",
        "a dairy cow standing near a metal feeding trough",
    ],
    "E": [
        "a cyclist accelerating on a velodrome track",
        "a marathon runner stretching near the start line",
        "a person adjusting gloves beside a wooden fence",
        "a motorbike rider fastening a helmet visor",
        "an athlete walking along the edge of a racetrack",
    ],
    "F": [
        "an empty grassland under heavy storm clouds",
        "a quiet racetrack lit by evening sunlight",
        "a weathered fence line across a green meadow",
        "a rustic barn surrounded by hay bales",
        "a wide valley landscape with layered clouds",
    ],
}

POS_GROUPS_BASE = {"A", "B", "C"}
NEG_GROUPS = {"D", "E", "F"}


@dataclass
class PromptItem:
    prompt: str
    group: str
    is_positive: int
    sample_id: str


@dataclass
class FeatureRecord:
    model: str
    layer: str
    timestep_idx: int
    scheduler_timestep: int
    prompt: str
    group: str
    seed: int
    is_positive: int
    feature_path: str
    feature_dim: int
    raw_path: str | None = None
    activation_shape: str | None = None


# ----------------------------- Model loading ----------------------------- #

def load_base_pipeline(base_model_path: str, device: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        safety_checker=None,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.text_encoder.eval()
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    return pipe


def load_erased_pipeline(
    base_model_path: str, erased_unet_checkpoint: str, device: str
) -> StableDiffusionPipeline:
    pipe = load_base_pipeline(base_model_path, device)
    state = torch.load(erased_unet_checkpoint, map_location=device)
    pipe.unet.load_state_dict(state)
    pipe.unet.eval()
    return pipe


def _extract_tokens_from_prompts(prompts: Iterable[str]) -> list[str]:
    pat = re.compile(r"\btoken_\w+\b")
    out: dict[str, None] = {}
    for p in prompts:
        for t in pat.findall(p):
            out[t] = None
    return list(out.keys())


def _prepare_attacked_text_encoder_and_tokenizer(
    pipe: StableDiffusionPipeline,
    attacked_text_encoder_path: str,
    tokenizer_path: str | None,
    placeholder_tokens: list[str],
    device: str,
) -> None:
    # Optional custom tokenizer path.
    if tokenizer_path:
        pipe.tokenizer = pipe.tokenizer.__class__.from_pretrained(tokenizer_path)

    state_dict = torch.load(attacked_text_encoder_path, map_location=device)
    emb_key = "text_model.embeddings.token_embedding.weight"
    expected_vocab = state_dict[emb_key].shape[0]
    to_add = expected_vocab - len(pipe.tokenizer)
    if to_add < 0:
        raise RuntimeError(
            f"Checkpoint expects vocab {expected_vocab}, tokenizer has {len(pipe.tokenizer)}."
        )

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
    pipe.text_encoder.load_state_dict(state_dict)
    pipe.text_encoder.eval()
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)


# ----------------------------- Layer hooks ----------------------------- #

def resolve_target_layers(unet: torch.nn.Module, layer_patterns: list[str]) -> list[str]:
    """
    Resolve user-provided layer patterns to actual module names.
    Priority:
    1) exact match
    2) prefix match
    3) substring match
    """
    names = [n for n, _ in unet.named_modules()]
    matched: list[str] = []
    for pat in layer_patterns:
        if pat in names:
            matched.append(pat)
            continue

        prefix = [n for n in names if n.startswith(pat)]
        if prefix:
            matched.append(prefix[0])
            continue

        sub = [n for n in names if pat in n]
        if sub:
            matched.append(sub[0])
            continue

        print(f"[warn] no layer matched pattern: {pat}")

    # Deduplicate in order
    seen = set()
    deduped = []
    for n in matched:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def _extract_tensor_from_hook_output(output: Any) -> torch.Tensor | None:
    """
    Generic extraction helper for module forward outputs.
    Works for tensor, tuple/list first tensor, or objects with '.sample'.
    """
    if torch.is_tensor(output):
        return output
    if hasattr(output, "sample") and torch.is_tensor(output.sample):
        return output.sample
    if isinstance(output, (tuple, list)):
        for x in output:
            if torch.is_tensor(x):
                return x
    return None


def register_cross_attention_hooks(
    unet: torch.nn.Module, layer_names: list[str], cache: dict[str, torch.Tensor | None]
) -> list[Any]:
    modules = dict(unet.named_modules())
    handles = []

    for name in layer_names:
        if name not in modules:
            continue

        def _hook(_module, _input, output, _name=name):
            t = _extract_tensor_from_hook_output(output)
            cache[_name] = t.detach() if t is not None else None

        handles.append(modules[name].register_forward_hook(_hook))

    return handles


# ----------------------------- Feature extraction ----------------------------- #

def pool_activation_tensor(tensor: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Pool activation tensors into a 1D vector.

    Rules:
    - [B, C, H, W] -> mean over H,W then mean over B => [C]
    - [B, T, D]    -> mean over T then mean over B   => [D]
    - [B, D]       -> mean over B                    => [D]
    - else         -> flatten all non-batch dims, mean over batch

    Note:
    During CFG the batch is typically doubled (uncond + cond). Caller should
    optionally slice to the conditional half before this function if desired.
    """
    shape = tuple(int(x) for x in tensor.shape)
    x = tensor.float()
    if x.ndim == 4:
        vec = x.mean(dim=(2, 3)).mean(dim=0)
    elif x.ndim == 3:
        vec = x.mean(dim=1).mean(dim=0)
    elif x.ndim == 2:
        vec = x.mean(dim=0)
    else:
        vec = x.reshape(x.shape[0], -1).mean(dim=0)
    return vec.detach().cpu().numpy(), shape


def _get_text_embeddings(pipe: StableDiffusionPipeline, prompt: str, device: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        [prompt], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    ).to(device)
    untok = pipe.tokenizer(
        [""], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        cond = pipe.text_encoder(tok.input_ids)[0]
        uncond = pipe.text_encoder(untok.input_ids)[0]
    return torch.cat([uncond, cond], dim=0)


def run_and_capture_activations(
    pipe: StableDiffusionPipeline,
    model_name: str,
    prompt_items: list[PromptItem],
    layers_to_capture: list[str],
    seeds: list[int],
    num_inference_steps: int,
    timesteps_to_capture: list[int],
    output_dir: Path,
    save_raw_tensors: bool,
    image_size: int,
    guidance_scale: float,
    device: str,
) -> list[FeatureRecord]:
    """
    Runs denoising loop per prompt/seed and captures selected-layer activations at
    selected denoising iteration indices (0-based over num_inference_steps).
    """
    features_root = output_dir / "features" / model_name
    raw_root = output_dir / "raw_tensors" / model_name
    features_root.mkdir(parents=True, exist_ok=True)
    if save_raw_tensors:
        raw_root.mkdir(parents=True, exist_ok=True)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    capture_set = set(timesteps_to_capture)

    hook_cache: dict[str, torch.Tensor | None] = {ln: None for ln in layers_to_capture}
    handles = register_cross_attention_hooks(pipe.unet, layers_to_capture, hook_cache)
    records: list[FeatureRecord] = []

    try:
        for item in prompt_items:
            for seed in seeds:
                gen_device = "cuda" if device.startswith("cuda") else "cpu"
                generator = torch.Generator(device=gen_device).manual_seed(seed)
                latent = torch.randn(
                    (1, pipe.unet.in_channels, image_size // 8, image_size // 8),
                    generator=generator,
                    device=device,
                    dtype=pipe.unet.dtype,
                )
                latent = latent * pipe.scheduler.init_noise_sigma
                text_emb = _get_text_embeddings(pipe, item.prompt, device)

                for step_idx, timestep in enumerate(pipe.scheduler.timesteps):
                    latent_model_input = torch.cat([latent] * 2, dim=0)
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

                    with torch.no_grad():
                        noise_pred = pipe.unet(
                            latent_model_input, timestep, encoder_hidden_states=text_emb
                        ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    latent = pipe.scheduler.step(noise_pred, timestep, latent).prev_sample

                    if step_idx not in capture_set:
                        continue

                    for layer_name in layers_to_capture:
                        act = hook_cache.get(layer_name, None)
                        if act is None:
                            continue

                        # Use conditional half only for CFG-batched activations when possible.
                        act_for_pool = act
                        if act_for_pool.shape[0] >= 2:
                            act_for_pool = act_for_pool[act_for_pool.shape[0] // 2 :]

                        pooled, shp = pool_activation_tensor(act_for_pool)
                        safe_layer = layer_name.replace(".", "_")
                        stem = (
                            f"{item.sample_id}_seed{seed}_layer-{safe_layer}_step{step_idx}"
                        )
                        feat_path = features_root / f"{stem}.npy"
                        np.save(feat_path, pooled)

                        raw_path = None
                        if save_raw_tensors:
                            raw_path = str(raw_root / f"{stem}.pt")
                            torch.save(act_for_pool.detach().cpu(), raw_path)

                        records.append(
                            FeatureRecord(
                                model=model_name,
                                layer=layer_name,
                                timestep_idx=step_idx,
                                scheduler_timestep=int(timestep.item()),
                                prompt=item.prompt,
                                group=item.group,
                                seed=seed,
                                is_positive=item.is_positive,
                                feature_path=str(feat_path),
                                feature_dim=int(pooled.shape[0]),
                                raw_path=raw_path,
                                activation_shape=str(shp),
                            )
                        )
    finally:
        for h in handles:
            h.remove()

    return records


# ----------------------------- Probe experiment ----------------------------- #

def build_probe_dataset(records: list[FeatureRecord]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    X, y, prompts, groups = [], [], [], []
    for r in records:
        X.append(np.load(r.feature_path))
        y.append(r.is_positive)
        prompts.append(r.prompt)
        groups.append(r.group)
    return np.stack(X), np.array(y, dtype=np.int64), prompts, groups


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    prompt_labels: list[str],
    split_mode: str = "prompt",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"accuracy": np.nan, "auroc": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}

    if split_mode == "prompt":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(gss.split(X, y, groups=prompt_labels))
    else:
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=random_state, stratify=y
        )

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    out = {
        "accuracy": float(accuracy_score(y_te, y_hat)),
        "auroc": float(roc_auc_score(y_te, y_prob)),
        "precision": float(precision_score(y_te, y_hat, zero_division=0)),
        "recall": float(recall_score(y_te, y_hat, zero_division=0)),
        "f1": float(f1_score(y_te, y_hat, zero_division=0)),
    }
    return out


# ----------------------------- Direction experiment ----------------------------- #

def compute_mean_difference_direction(
    records: list[FeatureRecord],
    positive_groups: set[str],
    negative_groups: set[str],
) -> dict[str, Any]:
    subset = [r for r in records if (r.group in positive_groups or r.group in negative_groups)]
    if not subset:
        return {}

    X = np.stack([np.load(r.feature_path) for r in subset])
    y = np.array([1 if r.group in positive_groups else 0 for r in subset], dtype=np.int64)
    X_pos, X_neg = X[y == 1], X[y == 0]
    if len(X_pos) == 0 or len(X_neg) == 0:
        return {}

    mean_pos = X_pos.mean(axis=0)
    mean_neg = X_neg.mean(axis=0)
    d = mean_pos - mean_neg
    norm = float(np.linalg.norm(d))
    d_unit = d / (norm + 1e-12)

    proj = X @ d_unit
    pos_proj, neg_proj = proj[y == 1], proj[y == 0]
    pos_mean, neg_mean = float(pos_proj.mean()), float(neg_proj.mean())
    pos_std = float(pos_proj.std() + 1e-12)
    neg_std = float(neg_proj.std() + 1e-12)
    pooled_std = np.sqrt(((len(pos_proj) - 1) * pos_std**2 + (len(neg_proj) - 1) * neg_std**2) / max(len(proj) - 2, 1))
    effect_size = float((pos_mean - neg_mean) / (pooled_std + 1e-12))

    return {
        "direction_raw": d,
        "direction_unit": d_unit,
        "direction_norm": norm,
        "pos_projection_mean": pos_mean,
        "neg_projection_mean": neg_mean,
        "pos_projection_std": pos_std,
        "neg_projection_std": neg_std,
        "effect_size": effect_size,
        "projections": proj,
        "labels": y,
    }


def save_direction_artifacts(
    out_dir: Path, model: str, layer: str, timestep_idx: int, direction_info: dict[str, Any]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_layer = layer.replace(".", "_")
    path = out_dir / f"{model}__{safe_layer}__step{timestep_idx}.pt"
    torch.save(
        {
            "model": model,
            "layer": layer,
            "timestep_idx": timestep_idx,
            "direction_raw": torch.tensor(direction_info["direction_raw"]),
            "direction_unit": torch.tensor(direction_info["direction_unit"]),
            "direction_norm": float(direction_info["direction_norm"]),
            "effect_size": float(direction_info["effect_size"]),
        },
        path,
    )
    return path


# ----------------------------- Plots ----------------------------- #

def _plot_projection_histogram(
    proj: np.ndarray, labels: np.ndarray, title: str, out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(proj[labels == 1], bins=20, alpha=0.6, label="positive")
    plt.hist(proj[labels == 0], bins=20, alpha=0.6, label="negative")
    plt.title(title)
    plt.xlabel("projection")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_feature_pca(X: np.ndarray, y: np.ndarray, title: str, out_path: Path) -> None:
    if X.shape[0] < 2:
        return
    z = PCA(n_components=2, random_state=42).fit_transform(X)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(z[y == 1, 0], z[y == 1, 1], label="positive", alpha=0.7)
    plt.scatter(z[y == 0, 0], z[y == 0, 1], label="negative", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_probe_heatmap(df_probe: pd.DataFrame, model: str, out_path: Path) -> None:
    d = df_probe[df_probe["model"] == model]
    if d.empty:
        return
    pivot = d.pivot_table(index="layer", columns="timestep", values="auroc", aggfunc="mean")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, max(3, 0.6 * len(pivot.index))))
    im = plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="AUROC")
    plt.xticks(np.arange(len(pivot.columns)), [str(c) for c in pivot.columns], rotation=45)
    plt.yticks(np.arange(len(pivot.index)), list(pivot.index))
    plt.title(f"Probe AUROC heatmap ({model})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_probe_and_direction_stats(
    probe_df: pd.DataFrame,
    direction_cache: dict[tuple[str, str, int], dict[str, Any]],
    feature_records: list[FeatureRecord],
    out_dir: Path,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Heatmaps per model.
    for m in sorted(set(r.model for r in feature_records)):
        _plot_probe_heatmap(probe_df, m, plots_dir / f"probe_auroc_heatmap_{m}.png")

    # Hist + PCA for each cached "all positives vs negatives" direction.
    for (model, layer, timestep_idx), info in direction_cache.items():
        proj = info["projections"]
        y = info["labels"]
        stem = f"{model}__{layer.replace('.', '_')}__step{timestep_idx}"
        _plot_projection_histogram(
            proj,
            y,
            f"{model} | {layer} | step {timestep_idx} projections",
            plots_dir / f"projection_hist_{stem}.png",
        )
        # Reconstruct X quickly from same records for PCA
        subset = [
            r
            for r in feature_records
            if r.model == model and r.layer == layer and r.timestep_idx == timestep_idx
        ]
        if subset:
            X = np.stack([np.load(r.feature_path) for r in subset])
            yy = np.array([r.is_positive for r in subset])
            _plot_feature_pca(
                X,
                yy,
                f"{model} | {layer} | step {timestep_idx}",
                plots_dir / f"feature_pca_{stem}.png",
            )


# ----------------------------- Utilities ----------------------------- #

def _parse_int_list(csv_text: str) -> list[int]:
    return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def _build_prompt_items(enable_s2_tokens: bool) -> list[PromptItem]:
    groups = dict(PROMPT_GROUPS_DEFAULT)
    if not enable_s2_tokens:
        groups.pop("C", None)

    items: list[PromptItem] = []
    for g, prompts in groups.items():
        is_pos = 1 if g in POS_GROUPS_BASE else 0
        for i, p in enumerate(prompts):
            sid = f"{g}_{i:03d}"
            items.append(PromptItem(prompt=p, group=g, is_positive=is_pos, sample_id=sid))
    return items


def _group_name(positive_groups: set[str]) -> str:
    return "+".join(sorted(positive_groups))


# ----------------------------- Main pipeline ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Internal representation probe for STEREO")
    parser.add_argument("--base_model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--erased_unet_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="internal_probe_out")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--timesteps_to_capture", type=str, default="45,30,15")
    parser.add_argument(
        "--layers_to_capture",
        type=str,
        default="down_blocks.1.attentions.0,mid_block.attentions.0,up_blocks.1.attentions.0",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--enable_s2_tokens", action="store_true")
    parser.add_argument("--use_attacked_text_encoder", action="store_true")
    parser.add_argument("--attacked_text_encoder_path", type=str, default="")
    parser.add_argument("--attacked_tokenizer_path", type=str, default="")
    parser.add_argument("--save_raw_tensors", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--split_mode", type=str, choices=["prompt", "sample"], default="prompt")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timesteps_to_capture = _parse_int_list(args.timesteps_to_capture)
    layers_to_capture = [x.strip() for x in args.layers_to_capture.split(",") if x.strip()]
    seeds = _parse_int_list(args.seeds)
    prompt_items = _build_prompt_items(enable_s2_tokens=args.enable_s2_tokens)
    prompts_all = [x.prompt for x in prompt_items]
    placeholder_tokens = _extract_tokens_from_prompts(prompts_all)

    # Load both models with identical setup except UNet.
    base_pipe = load_base_pipeline(args.base_model_path, args.device)
    erased_pipe = load_erased_pipeline(args.base_model_path, args.erased_unet_checkpoint, args.device)

    if args.use_attacked_text_encoder:
        if not args.attacked_text_encoder_path:
            raise ValueError("--use_attacked_text_encoder requires --attacked_text_encoder_path")
        _prepare_attacked_text_encoder_and_tokenizer(
            base_pipe,
            args.attacked_text_encoder_path,
            args.attacked_tokenizer_path or None,
            placeholder_tokens,
            args.device,
        )
        _prepare_attacked_text_encoder_and_tokenizer(
            erased_pipe,
            args.attacked_text_encoder_path,
            args.attacked_tokenizer_path or None,
            placeholder_tokens,
            args.device,
        )

    matched_base = resolve_target_layers(base_pipe.unet, layers_to_capture)
    matched_erased = resolve_target_layers(erased_pipe.unet, layers_to_capture)
    matched_layers = [x for x in matched_base if x in set(matched_erased)]
    print("Matched layer names:")
    for n in matched_layers:
        print(f"  - {n}")
    if not matched_layers:
        raise RuntimeError("No common layers matched between base and erased UNets.")

    # Capture activations.
    records_all: list[FeatureRecord] = []
    records_all.extend(
        run_and_capture_activations(
            pipe=base_pipe,
            model_name="base",
            prompt_items=prompt_items,
            layers_to_capture=matched_layers,
            seeds=seeds,
            num_inference_steps=args.num_inference_steps,
            timesteps_to_capture=timesteps_to_capture,
            output_dir=out_dir,
            save_raw_tensors=args.save_raw_tensors,
            image_size=args.image_size,
            guidance_scale=args.guidance_scale,
            device=args.device,
        )
    )
    records_all.extend(
        run_and_capture_activations(
            pipe=erased_pipe,
            model_name="erased",
            prompt_items=prompt_items,
            layers_to_capture=matched_layers,
            seeds=seeds,
            num_inference_steps=args.num_inference_steps,
            timesteps_to_capture=timesteps_to_capture,
            output_dir=out_dir,
            save_raw_tensors=args.save_raw_tensors,
            image_size=args.image_size,
            guidance_scale=args.guidance_scale,
            device=args.device,
        )
    )

    # Save metadata (activations_metadata.csv)
    meta_rows = [r.__dict__ for r in records_all]
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "activations_metadata.csv", index=False)

    # Probe experiment.
    probe_rows: list[dict[str, Any]] = []
    all_key = ["A", "B"] + (["C"] if args.enable_s2_tokens else [])
    positive_sets = [set(all_key), {"A"}, {"B"}] + ([{"C"}] if args.enable_s2_tokens else [])

    unique_keys = sorted(set((r.model, r.layer, r.timestep_idx) for r in records_all))
    for model, layer, t_idx in unique_keys:
        cell_records = [r for r in records_all if r.model == model and r.layer == layer and r.timestep_idx == t_idx]
        for pos_set in positive_sets:
            subset = [r for r in cell_records if (r.group in pos_set or r.group in NEG_GROUPS)]
            if len(subset) < 10:
                continue
            X, y, prompts, _ = build_probe_dataset(subset)
            m = train_linear_probe(X, y, prompts, split_mode=args.split_mode)
            probe_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "timestep": t_idx,
                    "prompt_groups_used": f"{_group_name(pos_set)} vs {'+'.join(sorted(NEG_GROUPS))}",
                    "accuracy": m["accuracy"],
                    "auroc": m["auroc"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "num_samples": len(subset),
                }
            )
    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(out_dir / "probe_results.csv", index=False)

    # Direction experiment.
    direction_dir = out_dir / "directions"
    direction_rows: list[dict[str, Any]] = []
    direction_cache_all: dict[tuple[str, str, int], dict[str, Any]] = {}
    direction_map: dict[tuple[str, str, int, str], np.ndarray] = {}

    for model, layer, t_idx in unique_keys:
        cell_records = [r for r in records_all if r.model == model and r.layer == layer and r.timestep_idx == t_idx]
        # Build directions for All / A / B / optional C
        group_sets: dict[str, set[str]] = {
            "ALL": set(all_key),
            "A": {"A"},
            "B": {"B"},
        }
        if args.enable_s2_tokens:
            group_sets["C"] = {"C"}

        dir_infos: dict[str, dict[str, Any]] = {}
        for key, pset in group_sets.items():
            info = compute_mean_difference_direction(cell_records, pset, NEG_GROUPS)
            if not info:
                continue
            dir_infos[key] = info
            direction_map[(model, layer, t_idx, key)] = info["direction_unit"]
            if key == "ALL":
                direction_cache_all[(model, layer, t_idx)] = info
                direction_path = save_direction_artifacts(direction_dir, model, layer, t_idx, info)
            else:
                direction_path = save_direction_artifacts(
                    direction_dir / "subgroups", model, f"{layer}__{key}", t_idx, info
                )

        if "ALL" not in dir_infos:
            continue
        row = {
            "model": model,
            "layer": layer,
            "timestep": t_idx,
            "direction_norm": float(dir_infos["ALL"]["direction_norm"]),
            "effect_size": float(dir_infos["ALL"]["effect_size"]),
            "direction_path": str(direction_path),
            "cosine_base_vs_erased": np.nan,  # filled later
            "cosine_A_vs_B": np.nan,
            "cosine_A_vs_C": np.nan,
            "cosine_B_vs_C": np.nan,
        }
        if "A" in dir_infos and "B" in dir_infos:
            row["cosine_A_vs_B"] = float(
                np.dot(dir_infos["A"]["direction_unit"], dir_infos["B"]["direction_unit"])
            )
        if "A" in dir_infos and "C" in dir_infos:
            row["cosine_A_vs_C"] = float(
                np.dot(dir_infos["A"]["direction_unit"], dir_infos["C"]["direction_unit"])
            )
        if "B" in dir_infos and "C" in dir_infos:
            row["cosine_B_vs_C"] = float(
                np.dot(dir_infos["B"]["direction_unit"], dir_infos["C"]["direction_unit"])
            )
        direction_rows.append(row)

    # Fill cosine_base_vs_erased for ALL direction per layer/timestep.
    for row in direction_rows:
        counterpart = "erased" if row["model"] == "base" else "base"
        a = direction_map.get((row["model"], row["layer"], row["timestep"], "ALL"))
        b = direction_map.get((counterpart, row["layer"], row["timestep"], "ALL"))
        if a is not None and b is not None:
            row["cosine_base_vs_erased"] = float(np.dot(a, b))

    direction_df = pd.DataFrame(direction_rows)
    direction_df.to_csv(out_dir / "direction_stats.csv", index=False)

    # Optional plots.
    plot_probe_and_direction_stats(probe_df, direction_cache_all, records_all, out_dir)

    # Summary JSON
    summary = {
        "num_activation_records": int(len(records_all)),
        "num_probe_rows": int(len(probe_df)),
        "num_direction_rows": int(len(direction_df)),
        "best_probe_rows_by_auroc": (
            probe_df.sort_values("auroc", ascending=False).head(5).to_dict(orient="records")
            if not probe_df.empty
            else []
        ),
        "strongest_direction_norm_rows": (
            direction_df.sort_values("direction_norm", ascending=False).head(5).to_dict(orient="records")
            if not direction_df.empty
            else []
        ),
    }
    # Recommend candidate layer/timestep by erased model AUROC first, fallback to base.
    rec = []
    if not probe_df.empty:
        erased_rows = probe_df[probe_df["model"] == "erased"].sort_values("auroc", ascending=False)
        if not erased_rows.empty:
            rec = erased_rows.head(3).to_dict(orient="records")
        else:
            rec = probe_df.sort_values("auroc", ascending=False).head(3).to_dict(orient="records")
    summary["recommended_candidates_for_future_steering"] = rec

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

