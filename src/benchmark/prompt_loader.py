from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .io_utils import load_csv, load_json


REQUIRED_PROMPT_COLUMNS = [
    "prompt_id",
    "target_concept",
    "prompt_family",
    "intended_label",
    "domain",
    "prompt",
    "lexical_mode",
]


def _normalize_family(x: str) -> str:
    x = str(x).strip().lower()
    if x in {"direct_target", "target"}:
        return "direct"
    return x


def stable_prompt_id_from_fields(
    target_concept: str,
    prompt_family: str,
    intended_label: str,
    prompt: str,
    lexical_mode: str = "strict",
) -> str:
    """
    Deterministic prompt_id (stable across Python processes and machines).
    Must match the normalization applied in load_prompts before synthesis.
    """
    tc = str(target_concept).strip().lower()
    fam = _normalize_family(prompt_family)
    il = str(intended_label).strip().lower()
    pr = str(prompt).strip()
    lm = str(lexical_mode).strip().lower()
    blob = f"{tc}|{fam}|{il}|{lm}|{pr}"
    suffix = hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]
    return f"{tc}_{fam}_{il}_{suffix}"


def _synthesize_prompt_id(row: pd.Series) -> str:
    return stable_prompt_id_from_fields(
        str(row["target_concept"]),
        str(row["prompt_family"]),
        str(row["intended_label"]),
        str(row["prompt"]),
        str(row["lexical_mode"]),
    )


def load_prompts(prompts_path: str | Path) -> pd.DataFrame:
    """
    Load prompt metadata from CSV or JSON and normalize to benchmark schema.
    """
    p = Path(prompts_path)
    if p.is_dir():
        csv_files = sorted(p.glob("**/*.csv"))
        json_files = sorted(p.glob("**/*.json"))
        parts = []
        for fp in csv_files:
            parts.append(load_csv(fp))
        for fp in json_files:
            obj = load_json(fp)
            if isinstance(obj, dict) and "prompts" in obj and isinstance(obj["prompts"], list):
                parts.append(pd.DataFrame(obj["prompts"]))
            elif isinstance(obj, list):
                parts.append(pd.DataFrame(obj))
        if not parts:
            raise ValueError(f"No CSV/JSON prompt files found in directory: {prompts_path}")
        df = pd.concat(parts, ignore_index=True)
    elif p.suffix.lower() == ".csv":
        df = load_csv(p)
    elif p.suffix.lower() == ".json":
        obj = load_json(p)
        if isinstance(obj, dict) and "prompts" in obj and isinstance(obj["prompts"], list):
            df = pd.DataFrame(obj["prompts"])
        elif isinstance(obj, list):
            df = pd.DataFrame(obj)
        else:
            raise ValueError(f"Unsupported JSON prompt schema: {prompts_path}")
    else:
        raise ValueError(f"Unsupported prompt file format: {prompts_path}")

    # Map from build_prompts output schema if needed.
    rename_map = {
        "target": "target_concept",
        "concept_label": "intended_label",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    if "domain" not in df.columns:
        df["domain"] = "object"
    if "lexical_mode" not in df.columns:
        df["lexical_mode"] = "strict"
    if "prompt_family" in df.columns:
        df["prompt_family"] = df["prompt_family"].map(_normalize_family)

    core_cols = [c for c in REQUIRED_PROMPT_COLUMNS if c != "prompt_id"]
    missing = [c for c in core_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required prompt columns: {missing}")

    # Cleanup before synthesizing prompt_id so IDs match normalized content.
    df["target_concept"] = df["target_concept"].astype(str).str.strip().str.lower()
    df["intended_label"] = df["intended_label"].astype(str).str.strip().str.lower()
    df["domain"] = df["domain"].astype(str).str.strip().str.lower()
    df["prompt_family"] = df["prompt_family"].astype(str).str.strip().str.lower()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["lexical_mode"] = df["lexical_mode"].astype(str).str.strip().str.lower()

    if "prompt_id" not in df.columns:
        df["prompt_id"] = df.apply(_synthesize_prompt_id, axis=1)
    else:
        df["prompt_id"] = df["prompt_id"].astype(str).str.strip()

    return df[list(REQUIRED_PROMPT_COLUMNS)].drop_duplicates().sort_values("prompt_id").reset_index(drop=True)

