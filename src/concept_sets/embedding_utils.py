from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPTokenizer


@dataclass
class EmbeddingConfig:
    provider: str = "clip"  # clip | sentence
    clip_model_name: str = "openai/clip-vit-base-patch32"
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    prompt_template: str = "a photo of a {}"
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_path: str | None = None


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class TextEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.provider = cfg.provider
        self.device = cfg.device

        if self.provider == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
            self.model = CLIPModel.from_pretrained(cfg.clip_model_name).to(self.device)
        elif self.provider == "sentence":
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.sentence_model_name)
            self.model = AutoModel.from_pretrained(cfg.sentence_model_name).to(self.device)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        outputs = []
        bs = self.cfg.batch_size
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=77 if self.provider == "clip" else 128,
                    return_tensors="pt",
                ).to(self.device)
                if self.provider == "clip":
                    vec = self.model.get_text_features(**encoded)
                else:
                    result = self.model(**encoded)
                    vec = _mean_pooling(result.last_hidden_state, encoded["attention_mask"])
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                outputs.append(vec.cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def embed_concepts(
    concepts: list[str],
    cfg: EmbeddingConfig,
) -> dict[str, np.ndarray]:
    cache_path = Path(cfg.cache_path) if cfg.cache_path else None
    cache = _load_cache(cache_path) if cache_path else None
    embed_map: dict[str, np.ndarray] = {}

    expected_signature = {
        "provider": cfg.provider,
        "clip_model_name": cfg.clip_model_name,
        "sentence_model_name": cfg.sentence_model_name,
        "prompt_template": cfg.prompt_template,
    }
    cached_vectors = {}
    if cache and cache.get("signature") == expected_signature:
        cached_vectors = cache.get("vectors", {})

    missing = []
    for c in concepts:
        if c in cached_vectors:
            embed_map[c] = np.asarray(cached_vectors[c], dtype=np.float32)
        else:
            missing.append(c)

    if missing:
        embedder = TextEmbedder(cfg)
        texts = [cfg.prompt_template.format(c) for c in missing]
        vectors = embedder.encode(texts)
        for concept, vec in zip(missing, vectors):
            embed_map[concept] = vec.astype(np.float32)

    if cache_path is not None:
        serializable = {k: v.tolist() for k, v in embed_map.items()}
        _save_cache(cache_path, {"signature": expected_signature, "vectors": serializable})

    return embed_map


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b))

