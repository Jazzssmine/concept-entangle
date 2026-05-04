from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .text_utils import normalize_phrase


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def load_captions(path: str | Path) -> list[str]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".text"}:
        return load_lines(p)
    if suffix == ".jsonl":
        captions: list[str] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "caption" in obj:
                    captions.append(str(obj["caption"]))
                elif "text" in obj:
                    captions.append(str(obj["text"]))
        return captions
    if suffix == ".csv":
        captions: list[str] = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return captions
            caption_fields = [c for c in ["caption", "text", "prompt"] if c in reader.fieldnames]
            for row in reader:
                for key in caption_fields:
                    value = row.get(key, "")
                    if value:
                        captions.append(str(value))
                        break
        return captions
    raise ValueError(f"Unsupported captions format for path: {path}")


def load_concept_vocab(path: str | Path) -> list[str]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".text"}:
        vocab = load_lines(p)
    elif suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            vocab = [str(x) for x in obj]
        elif isinstance(obj, dict):
            if "concepts" in obj and isinstance(obj["concepts"], list):
                vocab = [str(x) for x in obj["concepts"]]
            else:
                vocab = [str(k) for k in obj.keys()]
        else:
            raise ValueError("Unsupported JSON structure for concept vocab.")
    elif suffix == ".csv":
        vocab = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return vocab
            candidate_fields = ["concept", "label", "name", "class_name"]
            field = next((c for c in candidate_fields if c in reader.fieldnames), None)
            if field is None:
                raise ValueError(
                    f"CSV concept file needs one of columns {candidate_fields}, got {reader.fieldnames}"
                )
            for row in reader:
                v = row.get(field, "")
                if v:
                    vocab.append(str(v))
    else:
        raise ValueError(f"Unsupported concept vocab format for path: {path}")

    dedup = []
    seen = set()
    for v in vocab:
        n = normalize_phrase(v)
        if n and n not in seen:
            dedup.append(n)
            seen.add(n)
    return dedup


def load_synonyms(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: dict[str, list[str]] = {}
    for key, value in obj.items():
        k = normalize_phrase(str(key))
        if isinstance(value, list):
            out[k] = [normalize_phrase(str(v)) for v in value if normalize_phrase(str(v))]
    return out


def load_group_metadata(path: str | Path | None) -> dict[str, str]:
    """Load concept->group mapping from json/csv."""
    if path is None:
        return {}
    p = Path(path)
    suffix = p.suffix.lower()
    result: dict[str, str] = {}
    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("Group metadata JSON must be a dict: concept -> group")
        for k, v in obj.items():
            nk, nv = normalize_phrase(str(k)), normalize_phrase(str(v))
            if nk and nv:
                result[nk] = nv
        return result

    if suffix == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return result
            concept_col = "concept" if "concept" in reader.fieldnames else "name"
            group_col = "group" if "group" in reader.fieldnames else "superclass"
            for row in reader:
                nk = normalize_phrase(str(row.get(concept_col, "")))
                nv = normalize_phrase(str(row.get(group_col, "")))
                if nk and nv:
                    result[nk] = nv
        return result

    raise ValueError(f"Unsupported metadata format for path: {path}")


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)

