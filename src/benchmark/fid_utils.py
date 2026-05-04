from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image


def _load_tensor(path: str, image_size: int = 299) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).contiguous()
    return x.to(torch.uint8)


def _update_fid_metric(fid_metric, paths: list[str], real: bool, batch_size: int = 32) -> None:
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        if not batch_paths:
            continue
        batch = torch.stack([_load_tensor(p) for p in batch_paths], dim=0)
        fid_metric.update(batch, real=real)


def compute_fid_scores(
    generated_df: pd.DataFrame,
    base_model_name: str,
    fid_reference_mode: str = "base",
    fid_reference_dir: str | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Compute FID per (model_name, target_concept, prompt_family).
    Modes:
      - base: reference is base model outputs for same target/family.
      - real: reference from directory <fid_reference_dir>/<target>/<family>/*.png|jpg|jpeg|webp
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as e:  # pragma: no cover
        return pd.DataFrame(
            [
                {
                    "model_name": None,
                    "target_concept": None,
                    "prompt_family": None,
                    "FID": None,
                    "n_generated": 0,
                    "n_reference": 0,
                    "fid_reference_mode": fid_reference_mode,
                    "status": f"unavailable_torchmetrics:{type(e).__name__}",
                }
            ]
        )

    rows = []
    key_cols = ["model_name", "target_concept", "prompt_family"]
    grouped = generated_df.groupby(key_cols)

    for (model_name, target, family), gdf in grouped:
        gen_paths = [p for p in gdf["image_path"].astype(str).tolist() if Path(p).exists()]
        if len(gen_paths) < 2:
            rows.append(
                {
                    "model_name": model_name,
                    "target_concept": target,
                    "prompt_family": family,
                    "FID": None,
                    "n_generated": len(gen_paths),
                    "n_reference": 0,
                    "fid_reference_mode": fid_reference_mode,
                    "status": "insufficient_generated",
                }
            )
            continue

        if fid_reference_mode == "base":
            ref_df = generated_df[
                (generated_df["model_name"] == base_model_name)
                & (generated_df["target_concept"] == target)
                & (generated_df["prompt_family"] == family)
            ]
            ref_paths = [p for p in ref_df["image_path"].astype(str).tolist() if Path(p).exists()]
        elif fid_reference_mode == "real":
            if not fid_reference_dir:
                ref_paths = []
            else:
                base = Path(fid_reference_dir) / str(target) / str(family)
                ref_paths = []
                if base.exists():
                    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                        ref_paths.extend(str(x) for x in base.glob(ext))
        else:
            raise ValueError(f"Unsupported fid_reference_mode: {fid_reference_mode}")

        if len(ref_paths) < 2:
            rows.append(
                {
                    "model_name": model_name,
                    "target_concept": target,
                    "prompt_family": family,
                    "FID": None,
                    "n_generated": len(gen_paths),
                    "n_reference": len(ref_paths),
                    "fid_reference_mode": fid_reference_mode,
                    "status": "insufficient_reference",
                }
            )
            continue

        if fid_reference_mode == "base" and model_name == base_model_name:
            fid_val = 0.0
            status = "self_base_reference"
        else:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=False)
            _update_fid_metric(fid_metric, ref_paths, real=True, batch_size=batch_size)
            _update_fid_metric(fid_metric, gen_paths, real=False, batch_size=batch_size)
            fid_val = float(fid_metric.compute().cpu().item())
            status = "ok"

        rows.append(
            {
                "model_name": model_name,
                "target_concept": target,
                "prompt_family": family,
                "FID": fid_val,
                "n_generated": len(gen_paths),
                "n_reference": len(ref_paths),
                "fid_reference_mode": fid_reference_mode,
                "status": status,
            }
        )

    return pd.DataFrame(rows)

