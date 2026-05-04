from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator


def _parse_csv_data_files(data_files: str) -> str | list[str]:
    files = [x.strip() for x in data_files.split(",") if x.strip()]
    if len(files) == 1:
        return files[0]
    return files


def _pick_text_from_row(
    row: dict,
    text_field: str,
    fallback_fields: list[str] | None = None,
) -> str | None:
    if fallback_fields is None:
        fallback_fields = []
    ordered_fields = [text_field] + [f for f in fallback_fields if f != text_field]
    for field in ordered_fields:
        value = row.get(field)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return None


def iter_captions(
    source_type: str,
    text_field: str,
    fallback_text_fields: list[str] | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    hf_config: str | None = None,
    data_files: str | None = None,
    input_path: str | None = None,
    streaming: bool = True,
) -> Iterable[str]:
    """
    Unified caption iterator across HF streaming, parquet, and local files.
    """
    fallback_text_fields = fallback_text_fields or []
    source_type = source_type.lower()

    if source_type == "hf_streaming":
        if dataset_name is None:
            raise ValueError("--dataset_name is required for source_type=hf_streaming")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Please install `datasets` for hf_streaming source.") from exc

        kwargs = {"split": split, "streaming": streaming}
        if hf_config:
            kwargs["name"] = hf_config
        ds = load_dataset(dataset_name, **kwargs)

        def _iter_hf() -> Iterator[str]:
            for row in ds:
                if not isinstance(row, dict):
                    continue
                text = _pick_text_from_row(row, text_field=text_field, fallback_fields=fallback_text_fields)
                if text:
                    yield text

        return _iter_hf()

    if source_type == "parquet":
        if data_files is None:
            raise ValueError("--data_files is required for source_type=parquet")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Please install `datasets` for parquet source.") from exc

        ds = load_dataset(
            "parquet",
            data_files=_parse_csv_data_files(data_files),
            split=split,
            streaming=streaming,
        )

        def _iter_parquet() -> Iterator[str]:
            for row in ds:
                if not isinstance(row, dict):
                    continue
                text = _pick_text_from_row(row, text_field=text_field, fallback_fields=fallback_text_fields)
                if text:
                    yield text

        return _iter_parquet()

    if source_type == "csv":
        if input_path is None:
            raise ValueError("--input_path is required for source_type=csv")
        path = Path(input_path)

        def _iter_csv() -> Iterator[str]:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = _pick_text_from_row(row, text_field=text_field, fallback_fields=fallback_text_fields)
                    if text:
                        yield text

        return _iter_csv()

    if source_type == "jsonl":
        if input_path is None:
            raise ValueError("--input_path is required for source_type=jsonl")
        path = Path(input_path)

        def _iter_jsonl() -> Iterator[str]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    text = _pick_text_from_row(row, text_field=text_field, fallback_fields=fallback_text_fields)
                    if text:
                        yield text

        return _iter_jsonl()

    if source_type == "txt":
        if input_path is None:
            raise ValueError("--input_path is required for source_type=txt")
        path = Path(input_path)

        def _iter_txt() -> Iterator[str]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        yield text

        return _iter_txt()

    raise ValueError(f"Unsupported source_type: {source_type}")

