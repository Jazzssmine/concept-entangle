"""
Generate a publication-quality scatter plot for NP vs. kappa-hat scaling.

Output:
    img/kappa_np_scaling.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks (SciPy-like, with tie averaging)."""
    sorter = np.argsort(values, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(values))

    sorted_vals = values[sorter]
    ranks_sorted = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # 1-based average rank in [i+1, j]
        ranks_sorted[i:j] = avg_rank
        i = j

    return ranks_sorted[inv]


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation from ranked Pearson correlation."""
    x_rank = rankdata_average(x)
    y_rank = rankdata_average(y)

    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def main() -> None:
    # Concept-wise kappa-hat estimates.
    kappa_hat: Dict[str, float] = {
        "castle": 0.27,
        "cat": 0.77,
        "horse": 0.81,
        "bear": 0.82,
        "dog": 0.89,
    }
    concepts: List[str] = ["castle", "cat", "horse", "bear", "dog"]

    # NP values (0-100). np.nan is skipped during plotting/correlation.
    np_values: Dict[str, Dict[str, float]] = {
        "ESD": {"castle": 83.03, "cat": 87.35, "horse": 81.12, "bear": 58.02, "dog": 93.51},
        "SalUn": {"castle": 67.58, "cat": 73.04, "horse": 39.42, "bear": 53.89, "dog": 58.01},
        "EDiff": {"castle": 87.35, "cat": 65.99, "horse": 76.48, "bear": 56.75, "dog": 80.29},
        "AdvUnlearn": {"castle": 83.10, "cat": 72.40, "horse": 71.42, "bear": 67.37, "dog": 63.05},
        "STEREO": {"castle": 31.96, "cat": 27.60, "horse": 14.71, "bear": 13.75, "dog": 1.81},
        "SEOT": {"castle": 98.21, "cat": 81.99, "horse": 97.06, "bear": 64.14, "dog": 84.20},
        "SAeUron": {"castle": np.nan, "cat": 85.63, "horse": 82.48, "bear": 63.50, "dog": 4.27},
    }

    # Style configuration (colorblind-friendly shades + shape by family).
    method_style = {
        # Standard FT (circles, blue tones)
        "ESD": {"family": "Standard FT", "marker": "o", "color": "#1f77b4"},
        "SalUn": {"family": "Standard FT", "marker": "o", "color": "#4c9ed9"},
        "EDiff": {"family": "Standard FT", "marker": "o", "color": "#6baed6"},
        # Robust FT (squares, red/orange tones)
        "AdvUnlearn": {"family": "Robust FT", "marker": "s", "color": "#e6550d"},
        "STEREO": {"family": "Robust FT", "marker": "s", "color": "#d62728"},
        # Inference-time (triangles, green tones)
        "SEOT": {"family": "Inference-time", "marker": "^", "color": "#2ca25f"},
        "SAeUron": {"family": "Inference-time", "marker": "^", "color": "#66c2a4"},
    }

    # Matplotlib appearance.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.titlesize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    marker_size = 70
    scatter_alpha = 0.95

    # Plot all points by method, skipping NaNs.
    for method, values in np_values.items():
        xs: List[float] = []
        ys: List[float] = []
        for concept in concepts:
            y = values[concept]
            if np.isnan(y):
                continue
            xs.append(kappa_hat[concept])
            ys.append(y)

        style = method_style[method]
        ax.scatter(
            xs,
            ys,
            s=marker_size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.6,
            alpha=scatter_alpha,
            label=method,
            zorder=3,
        )

    # STEREO monotonic trend connector (sorted by kappa-hat).
    stereo_points = sorted(
        [(kappa_hat[c], np_values["STEREO"][c]) for c in concepts if not np.isnan(np_values["STEREO"][c])],
        key=lambda p: p[0],
    )
    stereo_x = [p[0] for p in stereo_points]
    stereo_y = [p[1] for p in stereo_points]
    ax.plot(
        stereo_x,
        stereo_y,
        linestyle="--",
        linewidth=1.6,
        color=method_style["STEREO"]["color"],
        alpha=0.6,
        zorder=2,
    )

    # Annotation near STEREO's castle point.
    # ax.annotate(
    #     "STEREO above bound",
    #     xy=(0.27, 31.96),
    #     xytext=(0.34, 41.0),
    #     textcoords="data",
    #     fontsize=9,
    #     arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
    #     color="#333333",
    # )

    ax.set_xlim(0.20, 0.95)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"$\hat{\kappa}$ (concept entanglement)")
    ax.set_ylabel("Neighbor Preservation (NP)")
    ax.grid(True, alpha=0.3)

    # Keep legend inside the main panel for publication layout.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#d0d0d0",
        borderpad=0.35,
        labelspacing=0.25,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    fig.tight_layout()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kappa_np_scaling.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to {out_path.resolve()}")

    # Print per-method Spearman rho across concepts, excluding NaN values.
    print("\nSpearman rank correlation (kappa-hat vs NP):")
    print(f"{'Method':<12} {'n':>3} {'rho':>8}")
    for method, values in np_values.items():
        x_valid: List[float] = []
        y_valid: List[float] = []
        for concept in concepts:
            y = values[concept]
            if np.isnan(y):
                continue
            x_valid.append(kappa_hat[concept])
            y_valid.append(float(y))

        x_arr = np.asarray(x_valid, dtype=float)
        y_arr = np.asarray(y_valid, dtype=float)
        rho = spearman_rho(x_arr, y_arr)
        print(f"{method:<12} {len(x_arr):>3d} {rho:>8.3f}")

    plt.show()


if __name__ == "__main__":
    main()
