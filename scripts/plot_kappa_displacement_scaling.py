"""
Generate kappa-hat vs. activation displacement scaling figure.

Outputs:
    img/kappa_displacement_scaling.pdf
    img/kappa_displacement_scaling.png
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = [
    "concept",
    "method",
    "kappa_hat",
    "mean_displacement",
    "median_displacement",
    "std_displacement",
    "n_prompts",
]


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
        avg_rank = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_rank
        i = j

    return ranks_sorted[inv]


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation from ranked Pearson correlation."""
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_rank = rankdata_average(x)
    y_rank = rankdata_average(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def lower_bound(kappa_hat: np.ndarray, L: float, epsilon: float, delta: float) -> np.ndarray:
    return (kappa_hat * (1.0 - epsilon) / L) - delta


def format_rho(rho: float) -> str:
    if math.isnan(rho):
        return "nan"
    return f"{rho:.3f}"


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    return rows


def main() -> None:
    csv_path = Path("results/activation_displacement.csv")
    if not csv_path.exists():
        print(f"Missing input CSV: {csv_path.resolve()}")
        sys.exit(1)

    rows = load_rows(csv_path)

    method_order: Sequence[str] = [
        "ESD",
        "SalUn",
        "EDiff",
        "AdvUnlearn",
        "STEREO",
        # "SEOT",
        # "SAeUron",
    ]
    style_by_method = {
        "ESD": {"marker": "o", "color": "#1f4e79"},
        "SalUn": {"marker": "o", "color": "#2e75b6"},
        "EDiff": {"marker": "o", "color": "#5b9bd5"},
        "AdvUnlearn": {"marker": "s", "color": "#ed7d31"},
        "STEREO": {"marker": "s", "color": "#c00000"},
        # "SEOT": {"marker": "^", "color": "#548235"},
        # "SAeUron": {"marker": "^", "color": "#70ad47"},
    }
    jitter_by_method = {
        "ESD": -0.0040,
        "SalUn": -0.0027,
        "EDiff": -0.0013,
        "AdvUnlearn": 0.0013,
        "STEREO": 0.0027,
        # "SEOT": 0.0040,
        # "SAeUron": 0.0,
    }

    records: Dict[str, List[Dict[str, float | str]]] = {m: [] for m in method_order}
    for row in rows:
        method = (row["method"] or "").strip()
        if method not in records:
            continue
        records[method].append(
            {
                "concept": (row["concept"] or "").strip(),
                "kappa_hat": float(row["kappa_hat"]),
                "mean_displacement": float(row["mean_displacement"]),
                "median_displacement": float(row["median_displacement"]),
                "std_displacement": float(row["std_displacement"]),
                "n_prompts": float(row["n_prompts"]),
            }
        )

    # Theoretical constants.
    L = 0.1741
    epsilon = 0.02
    delta = 0.05

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.set_facecolor("#fafafa")

    marker_size = 75
    method_handles: Dict[str, object] = {}

    all_y_vals: List[float] = []
    below_bound_candidates: List[Dict[str, float | str]] = []

    for method in method_order:
        method_records = records.get(method, [])
        if not method_records:
            continue

        xs = np.array([float(r["kappa_hat"]) for r in method_records], dtype=float)
        ys = np.array([float(r["mean_displacement"]) for r in method_records], dtype=float)
        xj = xs + jitter_by_method.get(method, 0.0)

        all_y_vals.extend(ys.tolist())

        predicted = lower_bound(xs, L=L, epsilon=epsilon, delta=delta)
        for rec, pred in zip(method_records, predicted):
            if float(rec["mean_displacement"]) < float(pred):
                below_bound_candidates.append(
                    {
                        "method": str(rec["method"] if "method" in rec else method),
                        "concept": str(rec["concept"]),
                        "x": float(rec["kappa_hat"]),
                        "y": float(rec["mean_displacement"]),
                        "pred": float(pred),
                    }
                )

        sc = ax.scatter(
            xj,
            ys,
            s=marker_size,
            marker=style_by_method[method]["marker"],
            color=style_by_method[method]["color"],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            label=method,
            zorder=3,
        )
        method_handles[method] = sc

    # STEREO trend with Spearman rho in legend.
    stereo_records = sorted(records.get("STEREO", []), key=lambda r: float(r["kappa_hat"]))
    stereo_rho = float("nan")
    stereo_line = None
    if stereo_records:
        sx = np.array([float(r["kappa_hat"]) for r in stereo_records], dtype=float)
        sy = np.array([float(r["mean_displacement"]) for r in stereo_records], dtype=float)
        stereo_rho = spearman_rho(sx, sy)
        sxj = sx + jitter_by_method.get("STEREO", 0.0)
        (stereo_line,) = ax.plot(
            sxj,
            sy,
            linestyle="--",
            color="#c00000",
            alpha=0.6,
            linewidth=1.5,
            label=rf"STEREO trend (Spearman $\rho$={format_rho(stereo_rho)} )",
            zorder=2,
        )

        # Annotate one high-kappa STEREO point.
        idx_max_kappa = int(np.argmax(sx))
        ax.annotate(
            "STEREO tracks bound",
            xy=(float(sxj[idx_max_kappa]), float(sy[idx_max_kappa])),
            xytext=(float(sx[idx_max_kappa]) - 0.17, float(sy[idx_max_kappa]) + 170.0),
            textcoords="data",
            fontsize=9,
            color="#333333",
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
            zorder=5,
        )

    # Bound line + shaded region below.
    x_bound = np.linspace(0.20, 0.95, 300)
    y_bound = lower_bound(x_bound, L=L, epsilon=epsilon, delta=delta)
    ax.fill_between(x_bound, 0.0, y_bound, color="#808080", alpha=0.06, zorder=1)
    (bound_line,) = ax.plot(
        x_bound,
        y_bound,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Thm. 4.8 lower bound (L=0.174)",
        zorder=4,
    )

    # Neutral annotation for one below-bound point (if any).
    if below_bound_candidates:
        candidate = sorted(below_bound_candidates, key=lambda d: float(d["x"]))[0]
        x_pt = float(candidate["x"]) + jitter_by_method.get(str(candidate["method"]), 0.0)
        y_pt = float(candidate["y"])
        ax.annotate(
            "Example below-bound point",
            xy=(x_pt, y_pt),
            xytext=(x_pt + 0.09, y_pt + max(35.0, 0.03 * (max(all_y_vals) if all_y_vals else 300.0))),
            textcoords="data",
            fontsize=8.5,
            color="#444444",
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#666666"},
            zorder=5,
        )

    # Required note.
    ax.text(
        0.27,
        0.0,
        "SAeUron not evaluated on castle.",
        fontsize=8.5,
        style="italic",
        color="#4f4f4f",
        ha="left",
        va="bottom",
    )

    # Axis styling and labels.
    y_max_data = max(all_y_vals) if all_y_vals else 1.0
    ax.set_xlim(0.20, 0.95)
    ax.set_ylim(0.0, y_max_data * 1.08)
    ax.set_xlabel(r"$\hat{\kappa}$ (concept entanglement)")
    ax.set_ylabel(r"Activation displacement at neighbors $\|\Phi_{\hat\theta}(p_{c'}) - \Phi_{\theta}(p_{c'})\|_2$")
    ax.grid(True, alpha=0.3)

    # Legend outside right, bound listed last.
    legend_handles: List[object] = []
    legend_labels: List[str] = []
    for method in method_order:
        if method in method_handles:
            legend_handles.append(method_handles[method])
            legend_labels.append(method)
    if stereo_line is not None:
        legend_handles.append(stereo_line)
        legend_labels.append(stereo_line.get_label())
    legend_handles.append(bound_line)
    legend_labels.append(bound_line.get_label())

    ax.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d0d0d0",
    )

    fig.tight_layout()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "kappa_displacement_scaling.pdf"
    out_png = out_dir / "kappa_displacement_scaling.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_pdf.resolve()}")
    print(f"Saved figure: {out_png.resolve()}")

    # Stats to stdout.
    print("\n[1] Spearman rho by method (kappa_hat vs mean_displacement):")
    print(f"{'Method':<12} {'n':>3} {'rho':>9}")
    for method in method_order:
        method_records = records.get(method, [])
        if not method_records:
            print(f"{method:<12} {0:>3d} {'nan':>9}")
            continue
        x = np.array([float(r["kappa_hat"]) for r in method_records], dtype=float)
        y = np.array([float(r["mean_displacement"]) for r in method_records], dtype=float)
        rho = spearman_rho(x, y)
        print(f"{method:<12} {len(x):>3d} {format_rho(rho):>9}")

    print("\n[2] STEREO per-concept bound check (displacement >= predicted bound):")
    stereo_ratios: List[float] = []
    for rec in stereo_records:
        concept = str(rec["concept"])
        kappa = float(rec["kappa_hat"])
        disp = float(rec["mean_displacement"])
        pred = float(lower_bound(np.array([kappa]), L=L, epsilon=epsilon, delta=delta)[0])
        passed = disp >= pred
        ratio = disp / pred if pred != 0 else float("inf")
        stereo_ratios.append(ratio)
        print(
            f"concept={concept:<8} kappa={kappa:>5.2f} disp={disp:>10.4f} "
            f"bound={pred:>8.4f} pass={str(passed)}"
        )

    avg_ratio = float(np.mean(stereo_ratios)) if stereo_ratios else float("nan")
    print("\n[3] Average STEREO displacement-to-bound ratio:")
    print(f"{avg_ratio:.4f}")


if __name__ == "__main__":
    main()
