#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


FAMILIES = {"direct", "neighbor", "control"}
DEFAULT_TARGETS = ["horse", "cat", "dog", "bear", "tower"]
EXPECTED_CONTROL_NAMES = {"jellyfish", "police car"}
MELON_OR_CACTUS = {"melon", "cactus"}


@dataclass
class PromptRow:
    target: str
    prompt_family: str
    concept_label: str
    prompt: str
    seed: int


@dataclass
class TargetData:
    target: str
    rows: list[PromptRow]
    concept_to_rows: dict[str, list[PromptRow]]
    concept_to_family: dict[str, str]
    controls: list[str]


def _hash_key(target: str, concept: str, prompt: str, seed: int) -> str:
    payload = json.dumps(
        {"target": target, "concept_label": concept, "prompt": prompt, "seed": int(seed)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _find_attn2(unet: torch.nn.Module) -> torch.nn.Module:
    exact = []
    broad = []
    for name, module in unet.named_modules():
        if "mid_block.attentions.0" in name and name.endswith("attn2"):
            exact.append((name, module))
        if "mid_block.attentions.0" in name and "attn2" in name:
            broad.append((name, module))
    if exact:
        print(f"[info] Hook module: {exact[0][0]}")
        return exact[0][1]
    if broad:
        print(f"[warn] Fallback hook module: {broad[0][0]}")
        return broad[0][1]
    raise RuntimeError("Could not locate attn2 module under mid_block.attentions.0")


def _load_target_csv(path: Path, expected_target: str) -> TargetData:
    df = pd.read_csv(path)
    required = {
        "target",
        "prompt_family",
        "concept_label",
        "prompt",
        "source_context_words",
        "lexical_mode",
        "is_valid",
        "validation_errors",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing required columns: {sorted(missing)}")

    keep = df[df["prompt_family"].isin(FAMILIES)].copy()
    keep = keep.drop_duplicates(subset=["concept_label", "prompt"]).reset_index(drop=True)

    if keep.empty:
        raise RuntimeError(f"{path}: no rows remaining after filtering families={sorted(FAMILIES)}")

    unique_targets = sorted(set(str(x) for x in keep["target"].dropna().unique()))
    if unique_targets != [expected_target]:
        raise RuntimeError(
            f"{path}: expected target '{expected_target}', found targets={unique_targets}"
        )

    rows: list[PromptRow] = []
    for idx, row in keep.iterrows():
        rows.append(
            PromptRow(
                target=expected_target,
                prompt_family=str(row["prompt_family"]),
                concept_label=str(row["concept_label"]),
                prompt=str(row["prompt"]),
                seed=int(idx),
            )
        )

    concept_to_rows: dict[str, list[PromptRow]] = {}
    concept_to_family: dict[str, str] = {}
    for r in rows:
        concept_to_rows.setdefault(r.concept_label, []).append(r)
        if r.concept_label in concept_to_family and concept_to_family[r.concept_label] != r.prompt_family:
            raise RuntimeError(
                f"{path}: concept '{r.concept_label}' appears in multiple families: "
                f"{concept_to_family[r.concept_label]} and {r.prompt_family}"
            )
        concept_to_family[r.concept_label] = r.prompt_family

    controls = [c for c, fam in concept_to_family.items() if fam == "control"]

    return TargetData(
        target=expected_target,
        rows=rows,
        concept_to_rows=concept_to_rows,
        concept_to_family=concept_to_family,
        controls=sorted(controls),
    )


def _preflight_validate(all_targets: dict[str, TargetData]) -> None:
    errors: list[str] = []
    for target, td in all_targets.items():
        concept_count = len(td.concept_to_rows)
        if concept_count != 14:
            errors.append(
                f"[{target}] Expected exactly 14 concepts (1+10+3), found {concept_count}."
            )

        direct_rows = td.concept_to_rows.get(target, [])
        if len(direct_rows) != 100:
            errors.append(f"[{target}] Expected 100 direct samples for '{target}', found {len(direct_rows)}.")

        for concept, rows in td.concept_to_rows.items():
            fam = td.concept_to_family[concept]
            if fam != "direct" and len(rows) != 50:
                errors.append(
                    f"[{target}] Expected 50 samples for non-direct concept '{concept}' ({fam}), found {len(rows)}."
                )

        control_set = set(td.controls)
        if len(control_set) != 3:
            errors.append(f"[{target}] Expected exactly 3 control concepts, found {len(control_set)} ({sorted(control_set)}).")

        if not EXPECTED_CONTROL_NAMES.issubset(control_set):
            missing = sorted(EXPECTED_CONTROL_NAMES - control_set)
            errors.append(f"[{target}] Missing expected controls: {missing}")

        if not (control_set & MELON_OR_CACTUS):
            errors.append(f"[{target}] Expected one of control concepts {sorted(MELON_OR_CACTUS)}, found {sorted(control_set)}")

    if errors:
        message = "\n".join(errors)
        raise RuntimeError(
            "Preflight sanity check failed before activation generation. "
            "Fix prompt pools first.\n" + message
        )


def _extract_single_activation(
    pipe: StableDiffusionPipeline,
    attn_module: torch.nn.Module,
    prompt: str,
    seed: int,
    ddim_steps: int,
    capture_step: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
    embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    generator = torch.Generator(device=device.type).manual_seed(seed)
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    pipe.scheduler.set_timesteps(ddim_steps, device=device)
    latents = latents * pipe.scheduler.init_noise_sigma

    cache: dict[str, torch.Tensor] = {}

    for step_idx, t in enumerate(pipe.scheduler.timesteps):
        hook_handle = None

        if step_idx == capture_step:

            def hook_fn(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                cond = out[1:2]  # text-conditioned CFG branch
                pooled = cond.mean(dim=1).squeeze(0)  # mean over spatial tokens
                cache["vec"] = pooled.detach().cpu().float()

            hook_handle = attn_module.register_forward_hook(hook_fn)

        latent_input = torch.cat([latents] * 2, dim=0)
        latent_input = pipe.scheduler.scale_model_input(latent_input, t)

        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=embeddings).sample

        if hook_handle is not None:
            hook_handle.remove()

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    if "vec" not in cache:
        raise RuntimeError(f"Activation hook did not fire at step {capture_step} for prompt={prompt!r}")

    return cache["vec"].numpy().astype(np.float32)


def _hierarchical_order(dist_mat: np.ndarray) -> np.ndarray:
    condensed = squareform(dist_mat, checks=False)
    z = linkage(condensed, method="average")
    return leaves_list(z)


def _compute_kappa(target: str, neighbors: list[str], acts_by_concept: dict[str, np.ndarray], rho: float) -> float:
    target_acts = acts_by_concept[target]
    if len(neighbors) == 0:
        return 0.0
    neigh_acts = np.concatenate([acts_by_concept[n] for n in neighbors], axis=0)
    cross = cdist(target_acts, neigh_acts, metric="cosine")
    return float((cross.min(axis=1) <= rho).mean())


def _format_neighbors(ns: list[str]) -> str:
    return ", ".join(ns) if ns else "(none)"


def _write_latex_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Concept & $\\hat{\\kappa}$ & Silhouette & Discovered Neighbors \\\\",
        "\\midrule",
    ]
    for r in rows:
        neigh = r["discovered_neighbors"].replace("_", "\\_")
        concept = str(r["concept"]).replace("_", "\\_")
        lines.append(f"{concept} & {r['kappa']:.4f} & {r['silhouette']:.4f} & {neigh} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_single_heatmap(
    dist_mat: np.ndarray,
    labels: list[str],
    title: str,
    pdf_path: Path,
    png_path: Path,
    vmin: float,
    vmax: float,
) -> tuple[np.ndarray, list[str]]:
    order = _hierarchical_order(dist_mat)
    omat = dist_mat[np.ix_(order, order)]
    olab = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(omat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(olab)))
    ax.set_yticks(np.arange(len(olab)))
    ax.set_xticklabels(olab, rotation=45, ha="right")
    ax.set_yticklabels(olab)

    for i in range(len(olab)):
        for j in range(len(olab)):
            color = "white" if omat[i, j] > (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{omat[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return omat, olab


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun kappa estimation on updated prompt CSVs.")
    parser.add_argument("--prompt_dir", type=Path, default=Path("/u/anon3/unlearn_diff/outputs/prompts"))
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output_dir", type=Path, default=Path("/mnt/user-data/outputs"))
    parser.add_argument("--cache_dir", type=Path, default=Path("/u/anon3/unlearn_diff/outputs/kappa_activation_cache"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--capture_step", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--control_kappa_tol", type=float, default=0.05)
    args = parser.parse_args()

    all_targets: dict[str, TargetData] = {}
    for target in args.targets:
        path = args.prompt_dir / f"{target}_prompts.csv"
        if not path.exists():
            raise RuntimeError(f"Missing prompt file: {path}")
        all_targets[target] = _load_target_csv(path, expected_target=target)

    _preflight_validate(all_targets)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading diffusion pipeline {args.model_id} ...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    attn_module = _find_attn2(pipe.unet)

    all_results: list[dict[str, Any]] = []
    logs: list[str] = []
    plot_payload: dict[str, dict[str, Any]] = {}

    for target in args.targets:
        td = all_targets[target]
        print(f"[info] Processing target: {target}")

        acts_by_concept: dict[str, list[np.ndarray]] = {c: [] for c in td.concept_to_rows}

        for r in tqdm(td.rows, desc=f"{target}-prompts"):
            key = _hash_key(r.target, r.concept_label, r.prompt, r.seed)
            concept_cache = args.cache_dir / r.target / r.concept_label
            concept_cache.mkdir(parents=True, exist_ok=True)
            vec_path = concept_cache / f"{key}.npy"

            if vec_path.exists():
                vec = np.load(vec_path)
            else:
                vec = _extract_single_activation(
                    pipe=pipe,
                    attn_module=attn_module,
                    prompt=r.prompt,
                    seed=r.seed,
                    ddim_steps=args.ddim_steps,
                    capture_step=args.capture_step,
                    guidance_scale=args.guidance_scale,
                    device=device,
                    dtype=dtype,
                )
                np.save(vec_path, vec)

            acts_by_concept[r.concept_label].append(vec)

        acts_np = {c: np.stack(vs, axis=0) for c, vs in acts_by_concept.items()}
        concepts = sorted(acts_np.keys())

        centroids = np.stack([acts_np[c].mean(axis=0) for c in concepts], axis=0)
        dist_mat = cosine_distances(centroids)
        upper = dist_mat[np.triu_indices(len(concepts), k=1)]
        delta = float(np.percentile(upper, 25))

        target_idx = concepts.index(target)
        discovered = [c for c in concepts if c != target and dist_mat[target_idx, concepts.index(c)] <= delta]

        rho = float(np.median(pdist(acts_np[target], metric="cosine")))
        kappa = _compute_kappa(target, discovered, acts_np, rho)

        all_acts = np.concatenate([acts_np[c] for c in concepts], axis=0)
        all_labels: list[str] = []
        for c in concepts:
            all_labels.extend([c] * len(acts_np[c]))
        sil_samples = silhouette_samples(all_acts, all_labels, metric="cosine")
        offset = 0
        target_silhouette = 0.0
        for c in concepts:
            n = len(acts_np[c])
            if c == target:
                target_silhouette = float(np.mean(sil_samples[offset : offset + n]))
                break
            offset += n

        # Sanity check #2
        control_violations = []
        for ctrl in td.controls:
            d = float(dist_mat[target_idx, concepts.index(ctrl)])
            if d <= delta:
                control_violations.append((ctrl, d))

        # Sanity check #3
        control_kappa_fail = []
        for ctrl in td.controls:
            cidx = concepts.index(ctrl)
            c_discovered = [c for c in concepts if c != ctrl and dist_mat[cidx, concepts.index(c)] <= delta]
            c_rho = float(np.median(pdist(acts_np[ctrl], metric="cosine")))
            c_k = _compute_kappa(ctrl, c_discovered, acts_np, c_rho)
            if c_k > args.control_kappa_tol:
                control_kappa_fail.append((ctrl, c_k))

        # Sanity check #4
        if not np.allclose(dist_mat, dist_mat.T, atol=1e-8):
            raise RuntimeError(f"[{target}] Distance matrix is not symmetric.")
        if not np.allclose(np.diag(dist_mat), 0.0, atol=1e-8):
            raise RuntimeError(f"[{target}] Distance matrix diagonal is not zero.")

        if control_violations:
            details = ", ".join([f"{c}:{d:.4f}" for c, d in control_violations])
            raise RuntimeError(
                f"[{target}] Sanity check failed: control centroid distance <= delta. "
                f"delta={delta:.4f}, violations={details}"
            )
        if control_kappa_fail:
            details = ", ".join([f"{c}:{k:.4f}" for c, k in control_kappa_fail])
            raise RuntimeError(
                f"[{target}] Sanity check failed: control kappa not approx 0 (tol={args.control_kappa_tol}). "
                f"violations={details}"
            )

        all_results.append(
            {
                "concept": target,
                "kappa": kappa,
                "silhouette": target_silhouette,
                "discovered_neighbors": ";".join(discovered),
            }
        )

        sample_counts = {c: int(acts_np[c].shape[0]) for c in concepts}
        logs.append(f"Target: {target}")
        logs.append("Sample counts:")
        for c in concepts:
            logs.append(f"  - {c}: {sample_counts[c]}")
        logs.append(f"delta: {delta:.6f}")
        logs.append(f"rho: {rho:.6f}")
        logs.append(f"discovered_neighbors: {_format_neighbors(discovered)}")
        logs.append(f"kappa_hat: {kappa:.6f}")
        logs.append(f"silhouette_target: {target_silhouette:.6f}")
        logs.append("anomalies: none")
        logs.append("")

        plot_payload[target] = {
            "dist_mat": dist_mat,
            "concepts": concepts,
        }

    # Write outputs only if all sanity checks pass for all targets.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = sorted(all_results, key=lambda x: x["kappa"], reverse=True)

    table_csv = args.output_dir / "kappa_table.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Concept", "kappa_hat", "Silhouette", "Discovered_Neighbors"])
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {
                    "Concept": r["concept"],
                    "kappa_hat": f"{r['kappa']:.6f}",
                    "Silhouette": f"{r['silhouette']:.6f}",
                    "Discovered_Neighbors": r["discovered_neighbors"],
                }
            )

    table_tex = args.output_dir / "kappa_table.tex"
    _write_latex_table(all_results, table_tex)

    # Shared color scale
    mats = [plot_payload[t]["dist_mat"] for t in args.targets]
    vmin = float(min(np.min(m) for m in mats))
    vmax = float(max(np.max(m) for m in mats))

    for target in args.targets:
        payload = plot_payload[target]
        _plot_single_heatmap(
            dist_mat=payload["dist_mat"],
            labels=payload["concepts"],
            title=f"Centroid Cosine Distance ({target})",
            pdf_path=args.output_dir / f"centroid_distance_{target}.pdf",
            png_path=args.output_dir / f"centroid_distance_{target}.png",
            vmin=vmin,
            vmax=vmax,
        )

    # Combined 5-panel figure
    fig, axes = plt.subplots(1, len(args.targets), figsize=(24, 5), constrained_layout=True)
    # matplotlib returns a single Axes object (not an array) when ncols == 1
    axes = np.atleast_1d(axes)
    im = None
    for i, target in enumerate(args.targets):
        payload = plot_payload[target]
        order = _hierarchical_order(payload["dist_mat"])
        omat = payload["dist_mat"][np.ix_(order, order)]
        labels = [payload["concepts"][j] for j in order]
        ax = axes[i]
        im = ax.imshow(omat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(target)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(args.output_dir / "centroid_distance_all.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    log_path = args.output_dir / "kappa_run_log.txt"
    log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")

    print(f"[done] Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
