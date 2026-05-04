"""
Regenerate pooled_concept_span with clearer labels using CLIP text embeddings.

This script intentionally keeps the same high-level idea as the original figure:
project prompts into 2D with PCA and visualize two prompt sets (S1 / S2).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from adjustText import adjust_text
from sklearn.decomposition import PCA
from transformers import CLIPTextModel, CLIPTokenizer


MODEL_ID = "openai/clip-vit-base-patch32"
RANDOM_STATE = 42

# Typography tuned for paper readability.
ANNOTATION_FONT_SIZE = 13
TITLE_FONT_SIZE = 24
AXIS_FONT_SIZE = 18
TICK_FONT_SIZE = 15
LEGEND_FONT_SIZE = 15
SUBTITLE_FONT_SIZE = 14


S1_PROMPTS = [
    "black horse walking along a dirt path",
    "brown horse grazing in an open meadow",
    "galloping horse with flowing mane on a beach",
    "wild horse running through tall grass",
    "horse and rider jumping over a hurdle",
    "white horse running across a field",
    "close-up portrait of a horse's face",
    "photo of a horse standing in a grassy plain",
    "leather saddle and reins on a horse",
    "cowboy in denim and boots walking with horse",
    "rider performing a jump in an arena",
    "western ranch scene with horse and hay",
    "mounted police unit preparing for patrol",
    "cavalry charge with horses in formation",
    "knight in armor riding into a joust",
    "fantasy warrior on a noble steed",
    "royal procession with decorated horses",
    "carriage parked on a cobblestone street",
    "old wooden stable interior lit by warm light",
    "champion thoroughbred crossing the finish line",
    "horse statue in a city plaza at sunset",
    "farrier trimming a horse hoof in a barn",
]


S2_PROMPTS = [
    "snowy mountain cabin surrounded by pine trees",
    "futuristic robot standing in a cyberpunk alley",
    "astronaut floating above Earth, cinematic",
    "Tokyo street at night with neon reflections",
    "bowl of ramen with steam rising",
    "still life of sunflowers in a ceramic vase",
    "macro photo of a butterfly on a purple flower",
    "close-up portrait of a golden retriever",
    "close-up of footprints leading into snow",
    "medieval stone castle on a cliff above the sea",
    "a photo of a token_xaji0y6d",
    "a photo of a token_yrru7zku",
]


def _short_label(prompt: str, max_words: int = 5) -> str:
    words = prompt.split()
    if len(words) <= max_words:
        return prompt
    return " ".join(words[:max_words]) + " ..."


def _encode_prompts(prompts: list[str], model: CLIPTextModel, tokenizer: CLIPTokenizer) -> np.ndarray:
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # CLIP text feature is typically read at the EOS token position.
    eos_id = tokenizer.eos_token_id
    eos_mask = (inputs["input_ids"] == eos_id).int()
    eos_pos = eos_mask.argmax(dim=1)
    features = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), eos_pos]
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy()


def main() -> None:
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
    model = CLIPTextModel.from_pretrained(MODEL_ID)
    model.eval()

    prompts = S1_PROMPTS + S2_PROMPTS
    labels = ["S1"] * len(S1_PROMPTS) + ["S2"] * len(S2_PROMPTS)

    emb = _encode_prompts(prompts, model, tokenizer)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(emb)

    colors = {"S1": "#1f77b4", "S2": "#17becf"}
    markers = {"S1": "o", "S2": "o"}

    fig, ax = plt.subplots(figsize=(16, 12))

    for group in ("S1", "S2"):
        idx = [i for i, g in enumerate(labels) if g == group]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=95,
            alpha=0.9,
            c=colors[group],
            marker=markers[group],
            edgecolors="white",
            linewidths=0.7,
            label=group,
            zorder=3,
        )

    text_artists = []
    for i, prompt in enumerate(prompts):
        text_artists.append(
            ax.text(
                coords[i, 0],
                coords[i, 1],
                _short_label(prompt, max_words=5),
                fontsize=ANNOTATION_FONT_SIZE,
                color="#333333",
                zorder=4,
            )
        )

    adjust_text(
        text_artists,
        x=coords[:, 0],
        y=coords[:, 1],
        ax=ax,
        autoalign="xy",
        expand_points=(1.3, 1.45),
        expand_text=(1.35, 1.5),
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, alpha=0.7),
        force_points=0.3,
        force_text=0.45,
        lim=400,
    )

    ax.set_title("pooled_concept_span (PCA) - CLIP text encoder", fontsize=TITLE_FONT_SIZE, pad=14)
    ax.set_xlabel("PC 1", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("PC 2", fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, frameon=True)

    exp_var = pca.explained_variance_ratio_
    subtitle = f"PCA variance explained: PC1={exp_var[0]*100:.1f}%, PC2={exp_var[1]*100:.1f}%"
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=SUBTITLE_FONT_SIZE, color="#555555")

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "pooled_concept_span_enlarged_v2.pdf"
    out_png = out_dir / "pooled_concept_span_enlarged_v2.png"

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    print(f"Saved: {out_pdf.resolve()}")
    print(f"Saved: {out_png.resolve()}")


if __name__ == "__main__":
    main()
