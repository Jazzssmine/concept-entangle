#!/usr/bin/env python3
"""Visualize concept activation regions and estimate overlap coefficient kappa."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import umap
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_distances


CONCEPTS = ["horse", "deer", "zebra", "cat", "dog", "bird", "fish", "flower", "car", "castle"]

KAPPA_TIER = {
    "horse": "high",
    "deer": "high",
    "zebra": "high",
    "cat": "high",
    "dog": "high",
    "bird": "medium",
    "fish": "medium",
    "flower": "medium",
    "car": "low",
    "castle": "low",
}

EXPECTED_NEIGHBORS = {
    "horse": ["deer", "zebra", "dog"],
    "cat": ["dog"],
    "deer": ["horse", "zebra"],
    "zebra": ["horse", "deer"],
    "dog": ["cat"],
    "bird": [],
    "fish": [],
    "flower": [],
    "car": [],
    "castle": [],
}

COLOR_MAP = {
    "horse": "#e41a1c",
    "deer": "#ff7f00",
    "zebra": "#984ea3",
    "cat": "#377eb8",
    "dog": "#4daf4a",
    "bird": "#a6cee3",
    "fish": "#b2df8a",
    "flower": "#fdbf6f",
    "car": "#999999",
    "castle": "#666666",
}

MARKER_MAP = {"high": "o", "medium": "^", "low": "s"}


def estimate_kappa(
    target: str,
    neighbors: list[str],
    acts_dict: dict[str, np.ndarray],
    rho: float | None = None,
) -> tuple[float, float]:
    target_acts = acts_dict[target]
    if rho is None:
        intra_dists = pdist(target_acts, metric="cosine")
        rho = float(np.median(intra_dists))

    if len(neighbors) == 0:
        return 0.0, rho

    neighbor_acts = np.concatenate([acts_dict[n] for n in neighbors], axis=0)
    cross_dists = cdist(target_acts, neighbor_acts, metric="cosine")
    min_dist_to_neighbor = cross_dists.min(axis=1)
    kappa = float((min_dist_to_neighbor < rho).mean())
    return kappa, rho


def load_activations(step_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    acts_by_concept: dict[str, np.ndarray] = {}
    all_acts: list[np.ndarray] = []
    all_labels: list[str] = []

    for concept in CONCEPTS:
        path = step_dir / f"{concept}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing activation file: {path}")
        acts = torch.load(path, map_location="cpu")
        if acts.ndim != 2:
            raise ValueError(f"Expected 2D activation tensor in {path}, got shape {tuple(acts.shape)}")
        acts_np = acts.float().numpy()
        acts_by_concept[concept] = acts_np
        all_acts.append(acts_np)
        all_labels.extend([concept] * len(acts_np))

    all_acts_np = np.concatenate(all_acts, axis=0)
    return acts_by_concept, all_acts_np, all_labels


def save_umap(
    acts_by_concept: dict[str, np.ndarray],
    all_acts_np: np.ndarray,
    output_dir: Path,
    step: int,
) -> None:
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(all_acts_np)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    idx = 0
    for concept in CONCEPTS:
        n = len(acts_by_concept[concept])
        tier = KAPPA_TIER[concept]
        ax.scatter(
            embedding[idx : idx + n, 0],
            embedding[idx : idx + n, 1],
            c=COLOR_MAP[concept],
            marker=MARKER_MAP[tier],
            s=40 if tier == "high" else 30,
            alpha=0.7,
            edgecolors="white" if tier == "high" else "none",
            linewidths=0.5,
            label=concept,
        )
        cx = embedding[idx : idx + n, 0].mean()
        cy = embedding[idx : idx + n, 1].mean()
        ax.annotate(
            concept,
            (cx, cy),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
        )
        idx += n

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "Concept Activation Regions in Cross-Attention Space\n"
        f"(up_blocks.1.attentions.1, step {step}/50)"
    )
    handles = [mpatches.Patch(color=COLOR_MAP[c], label=c) for c in CONCEPTS]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "umap_concept_activations.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "umap_concept_activations.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_pairwise_heatmaps(acts_by_concept: dict[str, np.ndarray], output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.stack([acts_by_concept[c].mean(axis=0) for c in CONCEPTS], axis=0)
    centroid_dists = cosine_distances(centroids)

    min_dists = np.zeros((len(CONCEPTS), len(CONCEPTS)), dtype=np.float64)
    for i, c1 in enumerate(CONCEPTS):
        for j, c2 in enumerate(CONCEPTS):
            if i == j:
                min_dists[i, j] = 0.0
            else:
                pairwise = cdist(acts_by_concept[c1], acts_by_concept[c2], metric="cosine")
                min_dists[i, j] = float(pairwise.min())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    items = [
        (axes[0], centroid_dists, "Centroid Cosine Distance"),
        (axes[1], min_dists, r"Min Pairwise Distance $d_{\mathcal{Z}}$"),
    ]
    for ax, data, title in items:
        im = ax.imshow(data, cmap="RdYlBu", aspect="auto")
        ax.set_xticks(range(len(CONCEPTS)))
        ax.set_yticks(range(len(CONCEPTS)))
        ax.set_xticklabels(CONCEPTS, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CONCEPTS, fontsize=9)
        ax.set_title(title)
        for i in range(len(CONCEPTS)):
            for j in range(len(CONCEPTS)):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_distances.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "pairwise_distances.png", dpi=300, bbox_inches="tight")
    plt.close()
    return centroid_dists, min_dists


def compute_silhouette(
    acts_by_concept: dict[str, np.ndarray],
    all_acts_np: np.ndarray,
    all_labels: list[str],
) -> tuple[float, dict[str, float]]:
    overall_sil = float(silhouette_score(all_acts_np, all_labels, metric="cosine"))
    sample_sil = silhouette_samples(all_acts_np, all_labels, metric="cosine")

    idx = 0
    per_concept_sil: dict[str, float] = {}
    for concept in CONCEPTS:
        n = len(acts_by_concept[concept])
        per_concept_sil[concept] = float(sample_sil[idx : idx + n].mean())
        idx += n
    return overall_sil, per_concept_sil


def save_summary_and_reports(
    output_dir: Path,
    per_concept_sil: dict[str, float],
    acts_by_concept: dict[str, np.ndarray],
) -> None:
    rows: list[dict[str, object]] = []
    kappa_expected: dict[str, float] = {}
    rho_expected: dict[str, float] = {}
    kappa_all_others: dict[str, float] = {}

    for concept in CONCEPTS:
        neighbors = [n for n in EXPECTED_NEIGHBORS.get(concept, []) if n in CONCEPTS]
        kappa, rho = estimate_kappa(concept, neighbors, acts_by_concept)
        kappa_expected[concept] = kappa
        rho_expected[concept] = rho

        others = [c for c in CONCEPTS if c != concept]
        kappa_all_others[concept], _ = estimate_kappa(concept, others, acts_by_concept, rho=rho)

        rows.append(
            {
                "concept": concept,
                "kappa_tier": KAPPA_TIER[concept],
                "silhouette": per_concept_sil[concept],
                "kappa_estimated": kappa,
                "rho": rho,
                "kappa_all_others": kappa_all_others[concept],
                "num_neighbors": len(neighbors),
                "neighbors": ";".join(neighbors),
            }
        )

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "kappa_expected_neighbors": kappa_expected,
        "rho_expected_neighbors": rho_expected,
        "kappa_all_others": kappa_all_others,
        "per_concept_silhouette": per_concept_sil,
    }
    (output_dir / "kappa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=str, default="outputs/experiment1")
    parser.add_argument("--step", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = experiment_dir
    step_dir = experiment_dir / "activations" / f"step_{args.step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    acts_by_concept, all_acts_np, all_labels = load_activations(step_dir)
    save_umap(acts_by_concept, all_acts_np, output_dir, step=args.step)
    save_pairwise_heatmaps(acts_by_concept, output_dir)
    overall_sil, per_concept_sil = compute_silhouette(acts_by_concept, all_acts_np, all_labels)
    save_summary_and_reports(output_dir, per_concept_sil, acts_by_concept)

    print(f"Overall silhouette score: {overall_sil:.4f}")
    print("\nPer-concept silhouette scores (higher = more separated):")
    for concept, score in sorted(per_concept_sil.items(), key=lambda x: x[1], reverse=True):
        print(f"  {concept:10s}  {score:.4f}  (expected kappa: {KAPPA_TIER[concept]})")

    print("\nEstimated kappa (expected-neighbor definition):")
    summary = np.genfromtxt(output_dir / "summary.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    for row in summary:
        print(
            f"  {row['concept']:10s}  kappa={row['kappa_estimated']:.4f}  "
            f"rho={row['rho']:.4f}  neighbors={row['neighbors']}"
        )
    print(f"\nSaved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
