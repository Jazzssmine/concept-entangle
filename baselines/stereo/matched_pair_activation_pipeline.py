#!/usr/bin/env python3
"""
Matched-pair activation-difference pipeline for STEREO analysis.

Scientific intent:
- Estimate cleaner horse-added candidate directions from matched natural-language prompt pairs.
- Separate horse-specific signal from broad animalness and scene/syntax artifacts.
- Test utility for activation scoring, prompt ranking, and conditional-only steering.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from activation_guided_prompt_search import (
        ClipScorer,
        ConditionalSteeringIntervention,
        WindowActivationCollector,
        flatten_prompt_groups,
        init_latent,
        load_base_pipeline,
        load_erased_pipeline,
        resolve_target_layer,
        run_generation,
    )
except ModuleNotFoundError:
    ClipScorer = None
    ConditionalSteeringIntervention = None
    WindowActivationCollector = None
    flatten_prompt_groups = None
    init_latent = None
    load_base_pipeline = None
    load_erased_pipeline = None
    resolve_target_layer = None
    run_generation = None


DEFAULT_EVAL_PROMPTS: dict[str, list[str]] = {
    "horse_prompts": [
        "a horse standing in a grassy field",
        "a black stallion running across a meadow",
        "a horse beside a wooden fence at sunset",
        "a horse near a stable gate in daylight",
    ],
    "jockey_race_prompts": [
        "a jockey riding during a race",
        "a rider guiding a horse on a race track",
        "a polo match with riders in motion",
        "an equestrian race scene with dust in the air",
    ],
    "cat_dog_prompts": [
        "a cat sitting on a sofa",
        "a dog running through a field",
        "a dog standing beside a wooden fence",
        "a cat near a window in a living room",
    ],
    "non_animal_controls": [
        "a red car parked on a country road",
        "a bicycle beside a wooden fence",
        "a small house on a grassy plain",
        "a chair in a bright living room",
    ],
}


def parse_csv_ints(x: str) -> list[int]:
    return [int(t.strip()) for t in x.split(",") if t.strip()]


def parse_csv_floats(x: str) -> list[float]:
    return [float(t.strip()) for t in x.split(",") if t.strip()]


def parse_window(window_str: str) -> tuple[int, int]:
    parts = window_str.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Window must be start-end, got: {window_str}")
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end < start:
        raise ValueError(f"Invalid window: {window_str}")
    return start, end


def slugify(text: str, max_len: int = 120) -> str:
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip())
    x = x.strip("_")
    return (x[:max_len] if x else "prompt").lower()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _require_model_utils() -> None:
    required = [
        WindowActivationCollector,
        ConditionalSteeringIntervention,
        flatten_prompt_groups,
        init_latent,
        load_base_pipeline,
        load_erased_pipeline,
        resolve_target_layer,
        run_generation,
    ]
    if any(x is None for x in required):
        raise ModuleNotFoundError(
            "Model-dependent modes require diffusers/transformers env and import of "
            "`activation_guided_prompt_search.py` dependencies."
        )


def _default_pair_config() -> dict[str, list[dict[str, str]]]:
    return {
        "horse_vs_animal": [
            {"positive": "a horse standing in a grassy field under daylight", "negative": "an animal standing in a grassy field under daylight"},
            {"positive": "a brown horse running through a wide green field", "negative": "a brown animal running through a wide green field"},
            {"positive": "a horse beside a weathered wooden fence", "negative": "an animal beside a weathered wooden fence"},
            {"positive": "a horse drinking from a shallow stream in the countryside", "negative": "an animal drinking from a shallow stream in the countryside"},
            {"positive": "a horse walking along a dirt path near a barn", "negative": "an animal walking along a dirt path near a barn"},
            {"positive": "a rider guiding a horse during a track race", "negative": "a rider guiding an animal during a track race"},
            {"positive": "a jockey riding a horse in a competitive race", "negative": "a jockey riding an animal in a competitive race"},
            {"positive": "a polo match with horses sprinting across the field", "negative": "a polo match with animals sprinting across the field"},
            {"positive": "a horse grazing in an open pasture at sunset", "negative": "an animal grazing in an open pasture at sunset"},
            {"positive": "a close photo of a horse near a stable gate", "negative": "a close photo of an animal near a stable gate"},
        ],
        "horse_vs_neighbor": [
            {"positive": "a horse standing in a grassy field under daylight", "negative": "a deer standing in a grassy field under daylight"},
            {"positive": "a horse running across an open field", "negative": "a zebra running across an open field"},
            {"positive": "a horse near a wooden fence in the countryside", "negative": "a cow near a wooden fence in the countryside"},
            {"positive": "a horse grazing beside a barn", "negative": "a goat grazing beside a barn"},
            {"positive": "a horse crossing a shallow stream in a meadow", "negative": "an elk crossing a shallow stream in a meadow"},
            {"positive": "a horse on a dirt trail at golden hour", "negative": "a camel on a dirt trail at golden hour"},
            {"positive": "a horse moving quickly on a race track", "negative": "a zebra moving quickly on a race track"},
            {"positive": "a rider seated on a horse during a race", "negative": "a rider seated on a camel during a race"},
            {"positive": "a horse in a paddock near a white fence", "negative": "a donkey in a paddock near a white fence"},
            {"positive": "a horse standing near tall grass and wildflowers", "negative": "a deer standing near tall grass and wildflowers"},
        ],
        "animal_vs_nonanimal": [
            {"positive": "an animal in a green field under a cloudy sky", "negative": "a car in a green field under a cloudy sky"},
            {"positive": "an animal beside a wooden fence in a pasture", "negative": "a bicycle beside a wooden fence in a pasture"},
            {"positive": "an animal near a red barn at sunset", "negative": "a tractor near a red barn at sunset"},
            {"positive": "an animal on a dirt path through a meadow", "negative": "a motorcycle on a dirt path through a meadow"},
            {"positive": "an animal by a stream in the countryside", "negative": "a small boat by a stream in the countryside"},
            {"positive": "an animal in front of a stable gate", "negative": "a wheelbarrow in front of a stable gate"},
            {"positive": "an animal in a race track scene with dust in the air", "negative": "a race car in a race track scene with dust in the air"},
            {"positive": "an animal on a grassy plain near distant hills", "negative": "a house on a grassy plain near distant hills"},
            {"positive": "an animal in a fenced paddock in daylight", "negative": "a tractor in a fenced paddock in daylight"},
            {"positive": "an animal near a country road and open grassland", "negative": "a truck near a country road and open grassland"},
        ],
    }


def load_pair_config(path: str) -> dict[str, list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[dict[str, str]]] = {}
    for pair_type, pairs in data.items():
        cleaned = []
        for i, p in enumerate(pairs):
            if "positive" not in p or "negative" not in p:
                raise ValueError(f"Pair {pair_type}[{i}] must contain positive and negative.")
            pos = str(p["positive"]).strip()
            neg = str(p["negative"]).strip()
            if not pos or not neg:
                raise ValueError(f"Pair {pair_type}[{i}] has empty positive/negative.")
            cleaned.append({"positive": pos, "negative": neg})
        out[str(pair_type)] = cleaned
    return out


def _collect_prompt_window_vector(
    pipe,
    prompt: str,
    seed: int,
    layer_name: str,
    capture_steps: list[int],
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
    device: str,
    save_path: Path,
) -> tuple[np.ndarray, list[int], tuple[int, ...]]:
    _require_model_utils()
    init_lat = init_latent(
        pipe.unet,
        image_size=image_size,
        seed=seed,
        device=device,
        init_noise_sigma=pipe.scheduler.init_noise_sigma,
    )
    collector = WindowActivationCollector(pipe.unet, layer_name, set(capture_steps))
    collector.register()
    try:
        _ = run_generation(
            pipe=pipe,
            prompt=prompt,
            init_latent_tensor=init_lat,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            step_hook_state=collector,
        )
    finally:
        collector.remove()

    steps_present = sorted(collector.step_to_vec.keys())
    if not steps_present:
        raise RuntimeError(f"No activation captured for prompt='{prompt}' seed={seed}.")
    step_mat = np.stack([collector.step_to_vec[s] for s in steps_present], axis=0)
    window_vec = step_mat.mean(axis=0).astype(np.float32)
    shape_example = collector.step_to_shape.get(steps_present[0], ())
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        window_vector=window_vec,
        step_vectors=step_mat,
        step_indices=np.array(steps_present, dtype=np.int32),
    )
    return window_vec, steps_present, tuple(int(x) for x in shape_example)


def collect_pair_activations(
    base_pipe,
    erased_pipe,
    pair_config: dict[str, list[dict[str, str]]],
    seeds: list[int],
    layer_name: str,
    window_start: int,
    window_end: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
    device: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_model_utils()
    capture_steps = [s for s in range(window_start, window_end + 1) if s < num_inference_steps]
    if not capture_steps:
        raise ValueError("No valid capture steps for the requested window and inference steps.")

    prompt_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    root = output_dir / "pair_activations"

    for pair_type, pairs in pair_config.items():
        for pair_idx, pair in enumerate(pairs):
            pos_prompt = pair["positive"]
            neg_prompt = pair["negative"]
            for seed in seeds:
                vecs: dict[str, np.ndarray] = {}
                path_map: dict[str, str] = {}
                for model_type, pipe in [("base", base_pipe), ("erased", erased_pipe)]:
                    for sign, prompt in [("positive", pos_prompt), ("negative", neg_prompt)]:
                        fpath = (
                            root
                            / model_type
                            / f"pair_type_{slugify(pair_type)}"
                            / f"pair_{pair_idx:03d}"
                            / f"seed_{seed}"
                            / f"{sign}_activation.npz"
                        )
                        window_vec, steps_present, shape_example = _collect_prompt_window_vector(
                            pipe=pipe,
                            prompt=prompt,
                            seed=seed,
                            layer_name=layer_name,
                            capture_steps=capture_steps,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            image_size=image_size,
                            device=device,
                            save_path=fpath,
                        )
                        key = f"{sign}_{model_type}"
                        vecs[key] = window_vec
                        path_map[key] = str(fpath)
                        prompt_rows.append(
                            {
                                "pair_type": pair_type,
                                "pair_index": int(pair_idx),
                                "seed": int(seed),
                                "model_type": model_type,
                                "sign": sign,
                                "prompt": prompt,
                                "layer": layer_name,
                                "window": f"{window_start}-{window_end}",
                                "window_start": int(window_start),
                                "window_end": int(window_end),
                                "captured_steps": ",".join(str(s) for s in steps_present),
                                "feature_dim": int(window_vec.shape[0]),
                                "activation_shape_example": str(shape_example),
                                "npz_path": str(fpath),
                            }
                        )

                delta_base = (vecs["positive_base"] - vecs["negative_base"]).astype(np.float32)
                delta_erased = (vecs["positive_erased"] - vecs["negative_erased"]).astype(np.float32)
                delta_gap = (delta_base - delta_erased).astype(np.float32)

                pdir = root / "pair_differences" / f"pair_type_{slugify(pair_type)}" / f"pair_{pair_idx:03d}" / f"seed_{seed}"
                pdir.mkdir(parents=True, exist_ok=True)
                delta_base_path = pdir / "delta_base.npy"
                delta_erased_path = pdir / "delta_erased.npy"
                delta_gap_path = pdir / "delta_gap.npy"
                np.save(delta_base_path, delta_base)
                np.save(delta_erased_path, delta_erased)
                np.save(delta_gap_path, delta_gap)

                pair_rows.append(
                    {
                        "pair_type": pair_type,
                        "pair_index": int(pair_idx),
                        "seed": int(seed),
                        "positive_prompt": pos_prompt,
                        "negative_prompt": neg_prompt,
                        "layer": layer_name,
                        "window": f"{window_start}-{window_end}",
                        "window_start": int(window_start),
                        "window_end": int(window_end),
                        "feature_dim": int(delta_base.shape[0]),
                        "h_pos_base_path": path_map["positive_base"],
                        "h_neg_base_path": path_map["negative_base"],
                        "h_pos_erased_path": path_map["positive_erased"],
                        "h_neg_erased_path": path_map["negative_erased"],
                        "delta_base_path": str(delta_base_path),
                        "delta_erased_path": str(delta_erased_path),
                        "delta_gap_path": str(delta_gap_path),
                        "delta_base_norm": float(np.linalg.norm(delta_base)),
                        "delta_erased_norm": float(np.linalg.norm(delta_erased)),
                        "delta_gap_norm": float(np.linalg.norm(delta_gap)),
                    }
                )

    prompt_df = pd.DataFrame(prompt_rows)
    pair_df = pd.DataFrame(pair_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_df.to_csv(output_dir / "prompt_activation_metadata.csv", index=False)
    pair_df.to_csv(output_dir / "pair_activation_differences.csv", index=False)
    return prompt_df, pair_df


def _mean_stack(paths: list[str]) -> np.ndarray:
    vecs = [np.load(p).astype(np.float32) for p in paths]
    if not vecs:
        raise ValueError("No vectors to average.")
    return np.stack(vecs, axis=0).mean(axis=0).astype(np.float32)


def build_matched_pair_directions(pair_diff_csv: str, output_dir: Path, pca_k: int = 5) -> dict[str, Any]:
    df = pd.read_csv(pair_diff_csv)
    if df.empty:
        raise ValueError(f"No rows in {pair_diff_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_directions: dict[str, torch.Tensor] = {}
    subspaces: dict[str, dict[str, torch.Tensor | int]] = {}
    direction_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []

    for pair_type, sub in df.groupby("pair_type"):
        v_base = _mean_stack(sub["delta_base_path"].tolist())
        v_erased = _mean_stack(sub["delta_erased_path"].tolist())
        v_gap = _mean_stack(sub["delta_gap_path"].tolist())

        for name, vec in [
            (f"v_base_{pair_type}", v_base),
            (f"v_erased_{pair_type}", v_erased),
            (f"v_gap_{pair_type}", v_gap),
        ]:
            vec_unit = normalize_np(vec)
            np.save(output_dir / f"{name}.npy", vec)
            np.save(output_dir / f"{name}__unit.npy", vec_unit)
            vector_directions[name] = torch.from_numpy(vec_unit)
            direction_rows.append(
                {
                    "direction_name": name,
                    "pair_type": pair_type,
                    "kind": "vector",
                    "feature_dim": int(vec.shape[0]),
                    "raw_norm": float(np.linalg.norm(vec)),
                    "unit_norm": float(np.linalg.norm(vec_unit)),
                }
            )

        gap_mat = np.stack([np.load(p).astype(np.float32) for p in sub["delta_gap_path"].tolist()], axis=0)
        gap_centered = gap_mat - gap_mat.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(gap_centered, full_matrices=False)
        k = int(min(pca_k, vt.shape[0]))
        comps_raw = vt[:k].astype(np.float32)
        comps_unit = np.stack([normalize_np(v) for v in comps_raw], axis=0)
        ev = s**2
        evr = (ev / (ev.sum() + 1e-12))[:k].astype(np.float32)

        subspace_name = f"subspace_gap_{pair_type}"
        np.savez_compressed(
            output_dir / f"{subspace_name}.npz",
            components_raw=comps_raw,
            components_unit=comps_unit,
            explained_variance_ratio=evr,
        )
        subspaces[subspace_name] = {
            "components": torch.from_numpy(comps_unit),
            "explained_variance_ratio": torch.from_numpy(evr),
            "k": int(k),
        }
        direction_rows.append(
            {
                "direction_name": subspace_name,
                "pair_type": pair_type,
                "kind": "subspace",
                "feature_dim": int(comps_unit.shape[1]),
                "k": int(k),
                "raw_norm": np.nan,
                "unit_norm": np.nan,
            }
        )
        for i in range(k):
            pca_rows.append(
                {
                    "subspace_name": subspace_name,
                    "pair_type": pair_type,
                    "component_index": int(i),
                    "explained_variance_ratio": float(evr[i]),
                }
            )

    bundle = {
        "layer": str(df.iloc[0]["layer"]),
        "window": str(df.iloc[0]["window"]),
        "vector_directions": vector_directions,
        "subspaces": subspaces,
    }
    torch.save(bundle, output_dir / "matched_pair_directions.pt")
    pd.DataFrame(direction_rows).to_csv(output_dir / "direction_summary.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(output_dir / "subspace_explained_variance.csv", index=False)
    return bundle


def _compute_window_vec_for_prompt(
    pipe,
    prompt: str,
    seed: int,
    layer_name: str,
    window_start: int,
    window_end: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
    device: str,
) -> np.ndarray:
    _require_model_utils()
    capture_steps = [s for s in range(window_start, window_end + 1) if s < num_inference_steps]
    tmp_path = Path("/tmp") / f"mp_tmp_{slugify(prompt)}_{seed}.npz"
    vec, _, _ = _collect_prompt_window_vector(
        pipe=pipe,
        prompt=prompt,
        seed=seed,
        layer_name=layer_name,
        capture_steps=capture_steps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image_size=image_size,
        device=device,
        save_path=tmp_path,
    )
    if tmp_path.exists():
        tmp_path.unlink()
    return vec


def score_prompts_against_directions(
    base_pipe,
    erased_pipe,
    prompt_groups: dict[str, list[str]],
    seeds: list[int],
    direction_bundle_path: str,
    layer_name: str,
    window_start: int,
    window_end: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
    device: str,
    output_dir: Path,
    model_types: list[str],
) -> pd.DataFrame:
    _require_model_utils()
    bundle = torch.load(direction_bundle_path, map_location="cpu")
    vectors: dict[str, torch.Tensor] = bundle.get("vector_directions", {})
    subspaces: dict[str, dict[str, Any]] = bundle.get("subspaces", {})
    prompt_df = flatten_prompt_groups(prompt_groups)

    rows: list[dict[str, Any]] = []
    for _, r in prompt_df.iterrows():
        prompt = r["prompt"]
        group = r["prompt_group"]
        for model_type in model_types:
            pipe = base_pipe if model_type == "base" else erased_pipe
            for seed in seeds:
                h = _compute_window_vec_for_prompt(
                    pipe=pipe,
                    prompt=prompt,
                    seed=int(seed),
                    layer_name=layer_name,
                    window_start=window_start,
                    window_end=window_end,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_size=image_size,
                    device=device,
                ).astype(np.float32)

                h_norm = float(np.linalg.norm(h) + 1e-12)
                for direction_name, tvec in vectors.items():
                    vec = tvec.cpu().numpy().astype(np.float32)
                    cosine = float(np.dot(normalize_np(h), normalize_np(vec)))
                    projection = float(np.dot(h, normalize_np(vec)))
                    rows.append(
                        {
                            "prompt": prompt,
                            "prompt_group": group,
                            "direction_name": direction_name,
                            "cosine_score": cosine,
                            "projection_score": projection,
                            "subspace_score": np.nan,
                            "seed": int(seed),
                            "model_type": model_type,
                            "layer": layer_name,
                            "window": f"{window_start}-{window_end}",
                            "activation_norm": h_norm,
                        }
                    )

                for subspace_name, sub_obj in subspaces.items():
                    comps = sub_obj["components"].cpu().numpy().astype(np.float32)  # [k, d]
                    proj = comps @ h
                    energy = float(np.linalg.norm(proj))
                    rows.append(
                        {
                            "prompt": prompt,
                            "prompt_group": group,
                            "direction_name": subspace_name,
                            "cosine_score": np.nan,
                            "projection_score": np.nan,
                            "subspace_score": energy,
                            "seed": int(seed),
                            "model_type": model_type,
                            "layer": layer_name,
                            "window": f"{window_start}-{window_end}",
                            "activation_norm": h_norm,
                        }
                    )

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "prompt_direction_scores.csv", index=False)
    return out


def run_conditional_only_steering(
    erased_pipe,
    prompt_groups: dict[str, list[str]],
    seeds: list[int],
    direction_bundle_path: str,
    direction_names: list[str],
    alphas: list[float],
    layer_name: str,
    window_start: int,
    window_end: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
    device: str,
    output_dir: Path,
) -> pd.DataFrame:
    _require_model_utils()
    bundle = torch.load(direction_bundle_path, map_location="cpu")
    vectors: dict[str, torch.Tensor] = bundle.get("vector_directions", {})
    prompt_df = flatten_prompt_groups(prompt_groups)
    target_steps = set(range(window_start, min(window_end + 1, num_inference_steps)))
    root = output_dir / "images"
    rows: list[dict[str, Any]] = []

    if not direction_names:
        direction_names = sorted([k for k in vectors.keys() if k.startswith("v_gap_")])
        if not direction_names:
            direction_names = sorted(vectors.keys())

    for _, r in prompt_df.iterrows():
        prompt = r["prompt"]
        group = r["prompt_group"]
        pslug = slugify(prompt)
        for seed in seeds:
            init_lat = init_latent(
                erased_pipe.unet,
                image_size=image_size,
                seed=seed,
                device=device,
                init_noise_sigma=erased_pipe.scheduler.init_noise_sigma,
            )
            sdir = root / f"group_{slugify(group)}" / f"prompt_{pslug}" / f"seed_{seed}"
            sdir.mkdir(parents=True, exist_ok=True)

            baseline = run_generation(
                pipe=erased_pipe,
                prompt=prompt,
                init_latent_tensor=init_lat,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=device,
            )
            bpath = sdir / "baseline_erased.png"
            baseline.save(bpath)
            rows.append(
                {
                    "prompt": prompt,
                    "prompt_group": group,
                    "seed": int(seed),
                    "model_type": "erased",
                    "method": "baseline",
                    "direction_name": "none",
                    "pair_type": "none",
                    "alpha": 0.0,
                    "layer": layer_name,
                    "window": f"{window_start}-{window_end}",
                    "image_path": str(bpath),
                }
            )

            for direction_name in direction_names:
                if direction_name not in vectors:
                    continue
                pair_type = direction_name.split("v_gap_")[-1] if direction_name.startswith("v_gap_") else "mixed"
                for alpha in alphas:
                    if float(alpha) == 0.0:
                        continue
                    inter = ConditionalSteeringIntervention(
                        unet=erased_pipe.unet,
                        layer_name=layer_name,
                        target_steps=target_steps,
                        direction_vec=vectors[direction_name].float(),
                        alpha=float(alpha),
                    )
                    inter.register()
                    try:
                        img = run_generation(
                            pipe=erased_pipe,
                            prompt=prompt,
                            init_latent_tensor=init_lat,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            device=device,
                            step_hook_state=inter,
                        )
                    finally:
                        inter.remove()
                    ipath = sdir / f"{direction_name}_alpha_{alpha}.png"
                    img.save(ipath)
                    rows.append(
                        {
                            "prompt": prompt,
                            "prompt_group": group,
                            "seed": int(seed),
                            "model_type": "erased",
                            "method": "steered",
                            "direction_name": direction_name,
                            "pair_type": pair_type,
                            "alpha": float(alpha),
                            "layer": layer_name,
                            "window": f"{window_start}-{window_end}",
                            "image_path": str(ipath),
                        }
                    )

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "steering_generation_metadata.csv", index=False)
    return out


def evaluate_images_clip(
    image_metadata_csv: str,
    output_dir: Path,
    clip_model_name: str,
    device: str,
    include_jockey_in_horse_score: bool = True,
) -> pd.DataFrame:
    if ClipScorer is None:
        raise ModuleNotFoundError(
            "CLIP evaluation requires transformers env and import of activation_guided_prompt_search utilities."
        )
    md = pd.read_csv(image_metadata_csv)
    scorer = ClipScorer(clip_model_name, device)
    horse_texts = ["a horse", "a photo of a horse"] + (["a jockey riding a horse"] if include_jockey_in_horse_score else [])
    animal_texts = ["an animal", "a four-legged animal"]

    rows: list[dict[str, Any]] = []
    for r in md.itertuples():
        emb = scorer.image_embed(r.image_path)
        horse_score = float(np.mean([scorer.sim(emb, t) for t in horse_texts]))
        animal_score = float(np.mean([scorer.sim(emb, t) for t in animal_texts]))
        prompt_align = float(scorer.sim(emb, r.prompt))
        rows.append(
            {
                "prompt": r.prompt,
                "prompt_group": r.prompt_group,
                "seed": int(r.seed),
                "direction_name": r.direction_name,
                "alpha": float(r.alpha),
                "method": r.method,
                "horse_score": horse_score,
                "animal_score": animal_score,
                "prompt_align_score": prompt_align,
                "image_path": r.image_path,
            }
        )

    out = pd.DataFrame(rows)
    baseline = out[out["method"] == "baseline"][["prompt", "seed", "horse_score", "prompt_align_score"]].rename(
        columns={"horse_score": "horse_score_baseline", "prompt_align_score": "prompt_align_baseline"}
    )
    out = out.merge(baseline, on=["prompt", "seed"], how="left")
    out["horse_score_delta"] = out["horse_score"] - out["horse_score_baseline"]
    out["prompt_align_delta"] = out["prompt_align_score"] - out["prompt_align_baseline"]

    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "clip_eval_scores.csv", index=False)
    summary = (
        out.groupby(["prompt_group", "direction_name", "alpha"], as_index=False)[
            ["horse_score", "animal_score", "prompt_align_score", "horse_score_delta", "prompt_align_delta"]
        ]
        .mean()
    )
    summary.to_csv(output_dir / "clip_eval_summary.csv", index=False)
    return out


def _load_prompt_groups(path: str | None, default_data: dict[str, list[str]]) -> dict[str, list[str]]:
    if not path:
        return default_data
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: dict[str, list[str]] = {}
    for k, v in obj.items():
        out[k] = [str(x) for x in v]
    return out


def _parse_csv_strs(x: str) -> list[str]:
    return [t.strip() for t in x.split(",") if t.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-pair activation-difference pipeline for STEREO.")
    parser.add_argument("--base_model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--erased_unet_checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="analysis_matched_pair_activation")
    parser.add_argument("--pair_config", type=str, default="prompts/matched_prompt_pairs_horse.json")
    parser.add_argument("--eval_prompt_file", type=str, default="prompts/matched_pair_eval_prompts.json")
    parser.add_argument("--direction_bundle", type=str, default="")
    parser.add_argument("--layer", type=str, default="up_blocks.1.attentions.1")
    parser.add_argument("--window", type=str, default="0-10")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--alphas", type=str, default="0,1,3,5,10")
    parser.add_argument("--pca_k", type=int, default=5)
    parser.add_argument("--model_types_for_scoring", type=str, default="erased,base")
    parser.add_argument("--steering_direction_names", type=str, default="")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--image_metadata_csv", type=str, default="")

    parser.add_argument("--build_pairs", action="store_true")
    parser.add_argument("--collect_activations", action="store_true")
    parser.add_argument("--build_directions", action="store_true")
    parser.add_argument("--score_prompts", action="store_true")
    parser.add_argument("--run_steering_eval", action="store_true")
    parser.add_argument("--evaluate_images_clip", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_csv_ints(args.seeds)
    window_start, window_end = parse_window(args.window)
    pair_cfg_path = Path(args.pair_config)

    if args.build_pairs:
        if pair_cfg_path.is_file():
            _ = load_pair_config(str(pair_cfg_path))
        else:
            pair_cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pair_cfg_path, "w", encoding="utf-8") as f:
                json.dump(_default_pair_config(), f, indent=2)

    pair_config = load_pair_config(str(pair_cfg_path))
    eval_prompts = _load_prompt_groups(args.eval_prompt_file if Path(args.eval_prompt_file).is_file() else None, DEFAULT_EVAL_PROMPTS)
    write_json(out_dir / "pair_config_used.json", pair_config)
    write_json(out_dir / "eval_prompts_used.json", eval_prompts)
    write_json(
        out_dir / "method_notes.json",
        {
            "activation_tensor_shape": "Module output tensor at hookpoint; with CFG doubled batch, conditional half is used for capture/scoring and steering.",
            "window_averaging": "For inclusive denoising window start-end, pooled activation vectors are captured per step and averaged across captured steps.",
            "direction_normalization": "All saved candidate vectors and subspace components are L2-normalized to unit norm for scoring and steering.",
            "conditional_only_steering": "During CFG, only conditional batch activations are modified: h_cond <- h_cond + alpha * v. Unconditional half is unchanged.",
            "layer": args.layer,
            "window": args.window,
        },
    )

    bundle_path = args.direction_bundle or str(out_dir / "directions" / "matched_pair_directions.pt")

    if args.collect_activations or args.score_prompts or args.run_steering_eval:
        _require_model_utils()
        if not args.erased_unet_checkpoint:
            raise ValueError("This mode requires --erased_unet_checkpoint.")
        base_pipe = load_base_pipeline(args.base_model_path, args.device)
        erased_pipe = load_erased_pipeline(args.base_model_path, args.erased_unet_checkpoint, args.device)
        layer_name = resolve_target_layer(base_pipe.unet, args.layer)
        _ = resolve_target_layer(erased_pipe.unet, args.layer)
    else:
        base_pipe, erased_pipe = None, None
        layer_name = args.layer

    if args.collect_activations:
        collect_pair_activations(
            base_pipe=base_pipe,
            erased_pipe=erased_pipe,
            pair_config=pair_config,
            seeds=seeds,
            layer_name=layer_name,
            window_start=window_start,
            window_end=window_end,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
            device=args.device,
            output_dir=out_dir,
        )

    if args.build_directions:
        pair_diff_csv = out_dir / "pair_activation_differences.csv"
        if not pair_diff_csv.is_file():
            raise ValueError(f"Missing required file: {pair_diff_csv}")
        build_matched_pair_directions(pair_diff_csv=str(pair_diff_csv), output_dir=out_dir / "directions", pca_k=args.pca_k)

    if args.score_prompts:
        if not Path(bundle_path).is_file():
            raise ValueError(f"Missing directions bundle: {bundle_path}")
        score_prompts_against_directions(
            base_pipe=base_pipe,
            erased_pipe=erased_pipe,
            prompt_groups=eval_prompts,
            seeds=seeds,
            direction_bundle_path=bundle_path,
            layer_name=layer_name,
            window_start=window_start,
            window_end=window_end,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
            device=args.device,
            output_dir=out_dir / "prompt_scoring",
            model_types=_parse_csv_strs(args.model_types_for_scoring),
        )

    if args.run_steering_eval:
        if not Path(bundle_path).is_file():
            raise ValueError(f"Missing directions bundle: {bundle_path}")
        run_conditional_only_steering(
            erased_pipe=erased_pipe,
            prompt_groups=eval_prompts,
            seeds=seeds,
            direction_bundle_path=bundle_path,
            direction_names=_parse_csv_strs(args.steering_direction_names),
            alphas=parse_csv_floats(args.alphas),
            layer_name=layer_name,
            window_start=window_start,
            window_end=window_end,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
            device=args.device,
            output_dir=out_dir / "steering_eval",
        )

    if args.evaluate_images_clip:
        image_csv = args.image_metadata_csv or str(out_dir / "steering_eval" / "steering_generation_metadata.csv")
        if not Path(image_csv).is_file():
            raise ValueError(f"Missing image metadata CSV: {image_csv}")
        evaluate_images_clip(
            image_metadata_csv=image_csv,
            output_dir=out_dir / "clip_evaluation",
            clip_model_name=args.clip_model_name,
            device=args.device,
            include_jockey_in_horse_score=True,
        )

    print(f"Done. Outputs saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
