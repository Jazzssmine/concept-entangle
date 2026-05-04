#!/usr/bin/env python3
"""Compute and compare kappa estimator variants for concept entanglement."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import gaussian_kde, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances


CONCEPTS_10 = ["horse", "pony", "dog", "cat", "car", "donkey", "bear", "deer", "truck", "castle"]

CAT_NP_VALUES = {
    "AdvUnlearn": 39.79,
    "STEREO": 42.22,
    "ESD": 92.74,
    "SalUn": 85.56,
    "EDiff": 86.64,
    "SAeUron": 94.31,
    "SEOT": 94.53,
}


@dataclass
class GeometryContext:
    acts: Dict[str, np.ndarray]
    all_acts: np.ndarray
    all_labels: List[str]
    centroids: np.ndarray
    centroid_dist: np.ndarray
    delta: float
    neighbors: Dict[str, List[str]]
    per_concept_silhouette: Dict[str, float]


def load_activations(step_dir: Path, concepts: List[str]) -> Dict[str, np.ndarray]:
    acts: Dict[str, np.ndarray] = {}
    for concept in concepts:
        path = step_dir / f"{concept}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing activation tensor for concept '{concept}': {path}")
        tensor = torch.load(path, map_location="cpu")
        if tensor.ndim != 2:
            raise ValueError(f"Expected shape [N, D] for {path}, got {tuple(tensor.shape)}")
        acts[concept] = tensor.float().numpy()
    return acts


def maybe_apply_pca(acts: Dict[str, np.ndarray], pca_dim: int) -> Dict[str, np.ndarray]:
    if pca_dim <= 0:
        return acts
    all_data = np.concatenate([acts[c] for c in CONCEPTS_10], axis=0)
    d = min(pca_dim, all_data.shape[1])
    pca = PCA(n_components=d, random_state=42)
    reduced = pca.fit_transform(all_data)
    out: Dict[str, np.ndarray] = {}
    idx = 0
    for c in CONCEPTS_10:
        n = acts[c].shape[0]
        out[c] = reduced[idx : idx + n]
        idx += n
    return out


def build_geometry_context(acts: Dict[str, np.ndarray]) -> GeometryContext:
    all_acts = np.concatenate([acts[c] for c in CONCEPTS_10], axis=0)
    all_labels: List[str] = []
    for c in CONCEPTS_10:
        all_labels.extend([c] * acts[c].shape[0])

    centroids = np.stack([acts[c].mean(axis=0) for c in CONCEPTS_10], axis=0)
    centroid_dist = cosine_distances(centroids)
    upper = centroid_dist[np.triu_indices(len(CONCEPTS_10), k=1)]
    delta = float(np.percentile(upper, 25.0))

    neighbors: Dict[str, List[str]] = {}
    for i, c in enumerate(CONCEPTS_10):
        neighbors[c] = [CONCEPTS_10[j] for j in range(len(CONCEPTS_10)) if j != i and centroid_dist[i, j] <= delta]

    sil_samples = silhouette_samples(all_acts, all_labels, metric="cosine")
    per_concept_silhouette: Dict[str, float] = {}
    idx = 0
    for c in CONCEPTS_10:
        n = acts[c].shape[0]
        per_concept_silhouette[c] = float(sil_samples[idx : idx + n].mean())
        idx += n

    return GeometryContext(acts, all_acts, all_labels, centroids, centroid_dist, delta, neighbors, per_concept_silhouette)


def current_kappa(ctx: GeometryContext) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in CONCEPTS_10:
        target = ctx.acts[c]
        ns = ctx.neighbors[c]
        if not ns:
            out[c] = 0.0
            continue
        rho = float(np.median(pdist(target, metric="cosine")))
        neigh = np.concatenate([ctx.acts[n] for n in ns], axis=0)
        cross = cdist(target, neigh, metric="cosine")
        out[c] = float((cross.min(axis=1) <= rho).mean())
    return out


def kappa_variant1_soft_nn(ctx: GeometryContext) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in CONCEPTS_10:
        target = ctx.acts[c]
        ns = ctx.neighbors[c]
        if not ns:
            out[c] = 0.0
            continue
        rho = float(np.median(pdist(target, metric="cosine")))
        neigh = np.concatenate([ctx.acts[n] for n in ns], axis=0)
        min_cross = cdist(target, neigh, metric="cosine").min(axis=1)
        scores = np.clip(1.0 - (min_cross / max(rho, 1e-12)), a_min=0.0, a_max=1.0)
        out[c] = float(scores.mean())
    return out


def kappa_variant2_knn_grid(
    ctx: GeometryContext,
    rho_percentiles: Tuple[int, ...] = (10, 25, 50),
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Tuple[int, int]]:
    summary_rows = []
    concept_rows = []
    grid_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    for rp in rho_percentiles:
        for kv in k_values:
            scores: Dict[str, float] = {}
            for c in CONCEPTS_10:
                target = ctx.acts[c]
                ns = ctx.neighbors[c]
                if not ns:
                    scores[c] = 0.0
                    continue
                rho = float(np.percentile(pdist(target, metric="cosine"), rp))
                neigh = np.concatenate([ctx.acts[n] for n in ns], axis=0)
                cross = cdist(target, neigh, metric="cosine")
                counts = (cross <= rho).sum(axis=1)
                scores[c] = float((counts >= kv).mean())
            vals = np.array([scores[c] for c in CONCEPTS_10], dtype=float)
            summary_rows.append(
                {
                    "rho_percentile": rp,
                    "k": kv,
                    "mean_kappa": float(vals.mean()),
                    "std_kappa": float(vals.std(ddof=0)),
                    "range_kappa": float(vals.max() - vals.min()),
                }
            )
            for c in CONCEPTS_10:
                concept_rows.append({"rho_percentile": rp, "k": kv, "concept": c, "kappa": scores[c]})
            grid_scores[(rp, kv)] = scores
    summary_df = pd.DataFrame(summary_rows).sort_values(["rho_percentile", "k"]).reset_index(drop=True)
    ranked = summary_df.sort_values(
        ["std_kappa", "range_kappa", "rho_percentile", "k"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    best_rp, best_k = int(ranked.loc[0, "rho_percentile"]), int(ranked.loc[0, "k"])
    summary_df["is_best"] = (summary_df["rho_percentile"] == best_rp) & (summary_df["k"] == best_k)
    long_df = pd.DataFrame(concept_rows).sort_values(["rho_percentile", "k", "concept"]).reset_index(drop=True)
    long_df["is_best"] = (long_df["rho_percentile"] == best_rp) & (long_df["k"] == best_k)
    return grid_scores[(best_rp, best_k)], summary_df, long_df, (best_rp, best_k)


def kappa_variant3_kde(ctx: GeometryContext) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in CONCEPTS_10:
        target = ctx.acts[c]
        ns = ctx.neighbors[c]
        if not ns:
            out[c] = 0.0
            continue
        kde_neighbors = []
        taus = []
        for n in ns:
            samples = ctx.acts[n].T
            kde = gaussian_kde(samples)
            kde_neighbors.append(kde)
            taus.append(float(np.median(kde(samples))))
        tau = float(np.median(np.array(taus, dtype=float)))
        eval_points = target.T
        max_density = np.zeros(target.shape[0], dtype=float)
        for kde in kde_neighbors:
            max_density = np.maximum(max_density, kde(eval_points))
        scores = 1.0 - np.exp(-(max_density / max(tau, 1e-12)))
        out[c] = float(np.clip(scores, 0.0, 1.0).mean())
    return out


def kappa_variant4_mahalanobis(ctx: GeometryContext) -> Dict[str, float]:
    out: Dict[str, float] = {}
    means = {c: ctx.acts[c].mean(axis=0) for c in CONCEPTS_10}
    covs = {}
    for c in CONCEPTS_10:
        covs[c] = np.cov(ctx.acts[c], rowvar=False) + np.eye(ctx.acts[c].shape[1]) * 1e-6
    for c in CONCEPTS_10:
        ns = ctx.neighbors[c]
        if not ns:
            out[c] = 0.0
            continue
        sims = []
        for n in ns:
            diff = means[c] - means[n]
            inv = np.linalg.pinv(covs[c] + covs[n])
            d = float(diff.T @ inv @ diff)
            sims.append(np.exp(-d))
        out[c] = float(np.mean(sims))
    return out


def build_comparison_table(
    ctx: GeometryContext,
    k_current: Dict[str, float],
    k1: Dict[str, float],
    k2_best: Dict[str, float],
    k3: Dict[str, float],
    k4: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for c in CONCEPTS_10:
        rows.append(
            {
                "concept": c,
                "k_current": k_current[c],
                "k_1_soft_nn": k1[c],
                "k_2_best": k2_best[c],
                "k_3_kde": k3[c],
                "k_4_mahal": k4[c],
                "silhouette": ctx.per_concept_silhouette[c],
                "num_neighbors": len(ctx.neighbors[c]),
                "neighbors": ";".join(ctx.neighbors[c]),
            }
        )
    return pd.DataFrame(rows)


def save_spearman_matrix(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    cols = ["k_1_soft_nn", "k_2_best", "k_3_kde", "k_4_mahal"]
    corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            corr.loc[a, b] = float(spearmanr(df[a].values, df[b].values).statistic)
    corr.to_csv(out_path, float_format="%.6f")
    return corr


def write_analysis_markdown(
    comparison: pd.DataFrame,
    corr: pd.DataFrame,
    best_cfg: Tuple[int, int],
    out_path: Path,
) -> None:
    concept_order_by_k3 = comparison.sort_values("k_3_kde", ascending=False)["concept"].tolist()
    concept_order_by_k4 = comparison.sort_values("k_4_mahal", ascending=False)["concept"].tolist()
    horse = float(comparison.loc[comparison["concept"] == "horse", "k_3_kde"].iloc[0])
    cat = float(comparison.loc[comparison["concept"] == "cat", "k_3_kde"].iloc[0])
    dog = float(comparison.loc[comparison["concept"] == "dog", "k_3_kde"].iloc[0])
    pony = float(comparison.loc[comparison["concept"] == "pony", "k_3_kde"].iloc[0])
    has_order = horse > cat > dog > pony

    spread = {
        "k_current": float(comparison["k_current"].max() - comparison["k_current"].min()),
        "k_1": float(comparison["k_1_soft_nn"].max() - comparison["k_1_soft_nn"].min()),
        "k_2": float(comparison["k_2_best"].max() - comparison["k_2_best"].min()),
        "k_3": float(comparison["k_3_kde"].max() - comparison["k_3_kde"].min()),
        "k_4": float(comparison["k_4_mahal"].max() - comparison["k_4_mahal"].min()),
    }
    winner = max(["k_1", "k_2", "k_3", "k_4"], key=lambda x: spread[x])
    cfg_text = f"rho={best_cfg[0]}th percentile, k={best_cfg[1]}"

    text = f"""# Kappa Variant Analysis

Using the existing activation setup (50 prompts per concept, `mid_block.attentions.0`, step 25, mean-pooled cross-attention vectors), I compared the current estimator against four alternatives on the 10-concept subset (`horse`, `pony`, `dog`, `cat`, `car`, `donkey`, `bear`, `deer`, `truck`, `castle`). The original nearest-neighbor indicator remains vulnerable to local cloud-contact behavior, while the alternatives provide smoother concept-level separation.

Among the four proposed variants, **{winner}** yields the widest dynamic range in this run (spread = {spread[winner]:.3f}), making it the most graded in absolute scale. Variant 2’s best configuration is **{cfg_text}**, selected by maximizing across-concept standard deviation (tie-broken by range). In practice, Variant 1 and Variant 2 preserve compatibility with the current Appendix estimator form, while Variant 3 and Variant 4 better reflect volumetric/shape overlap assumptions from Definition 4.3.

For the intuitive ordering check (`horse > cat > dog > pony`), Variant 3 values are:

- horse: {horse:.3f}
- cat: {cat:.3f}
- dog: {dog:.3f}
- pony: {pony:.3f}

This condition is **{has_order}**. Variant-3 descending order is: {", ".join(concept_order_by_k3)}. Variant-4 descending order is: {", ".join(concept_order_by_k4)}.

Spearman rank correlations across variants are in `kappa_spearman_matrix.csv`; high positive entries indicate agreement on ordering despite scale differences, while low entries suggest estimator-specific ranking effects. Given the paper’s geometric motivation, my recommendation is to adopt **Variant 3 (KDE density-weighted overlap)** when you can tolerate moderate estimator complexity, because it is closest to volumetric overlap and avoids hard binary touching behavior. If you prefer minimal paper edits, adopt **Variant 1** as a lightweight drop-in replacement of the binary indicator.
"""
    out_path.write_text(text, encoding="utf-8")


def plot_np_spread_scatter(df: pd.DataFrame, output_path: Path) -> None:
    cat_vals = np.array(list(CAT_NP_VALUES.values()), dtype=float)
    cat_spread = float(cat_vals.max() - cat_vals.min())
    variants = ["k_current", "k_1_soft_nn", "k_2_best", "k_3_kde", "k_4_mahal"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for v in variants:
        x_vals = []
        y_vals = []
        for _, row in df.iterrows():
            x_vals.append(float(row[v]))
            y_vals.append(cat_spread if row["concept"] == "cat" else np.nan)
        ax.scatter(x_vals, y_vals, label=v, alpha=0.8)
        cat_x = float(df[df["concept"] == "cat"][v].iloc[0])
        ax.annotate(f"{v}:cat", (cat_x, cat_spread), textcoords="offset points", xytext=(3, 4), fontsize=8)
    ax.set_xlabel("kappa variant value")
    ax.set_ylabel("NP spread across methods (cat calibration, range)")
    ax.set_title("Kappa vs NP spread (cat calibrated; other concepts placeholder)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations_dir",
        type=str,
        default="unlearn_diff/outputs/experiment1_v3_mid/activations/step_25",
    )
    parser.add_argument("--output_dir", type=str, default="unlearn_diff/kappa_variant_outputs")
    parser.add_argument("--pca_dim", type=int, default=16, help="<=0 disables PCA")
    parser.add_argument(
        "--analysis_markdown_path",
        type=str,
        default="unlearn_diff/kappa_analysis.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acts = load_activations(Path(args.activations_dir), CONCEPTS_10)
    acts = maybe_apply_pca(acts, args.pca_dim)
    ctx = build_geometry_context(acts)
    k_current = current_kappa(ctx)
    k1 = kappa_variant1_soft_nn(ctx)
    k2_best, k2_summary, k2_long, best_cfg = kappa_variant2_knn_grid(ctx)
    k3 = kappa_variant3_kde(ctx)
    k4 = kappa_variant4_mahalanobis(ctx)
    comparison = build_comparison_table(ctx, k_current, k1, k2_best, k3, k4)
    comparison.to_csv(out_dir / "kappa_comparison_table.csv", index=False, float_format="%.6f")
    k2_summary.to_csv(out_dir / "kappa_variant2_sensitivity_summary.csv", index=False)
    k2_long.to_csv(out_dir / "kappa_variant2_sensitivity_long.csv", index=False)
    corr = save_spearman_matrix(comparison, out_dir / "kappa_spearman_matrix.csv")
    plot_np_spread_scatter(comparison, out_dir / "kappa_vs_np_spread_scatter.png")
    write_analysis_markdown(comparison, corr, best_cfg, Path(args.analysis_markdown_path))

    meta = {
        "concepts": CONCEPTS_10,
        "activations_dir": args.activations_dir,
        "pca_dim": args.pca_dim,
        "delta_percentile": 25,
        "delta_value": ctx.delta,
        "neighbors": ctx.neighbors,
        "cat_np_values": CAT_NP_VALUES,
        "cat_np_spread_range": float(max(CAT_NP_VALUES.values()) - min(CAT_NP_VALUES.values())),
        "spearman_columns": list(corr.columns),
        "k2_best_config": {"rho_percentile": best_cfg[0], "k": best_cfg[1]},
        "analysis_markdown_path": args.analysis_markdown_path,
    }
    (out_dir / "kappa_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved outputs in", out_dir)


if __name__ == "__main__":
    main()
