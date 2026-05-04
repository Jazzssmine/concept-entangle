from __future__ import annotations

import re
import string
from typing import List


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "these",
    "those",
    "their",
    "they",
    "them",
    "you",
    "your",
    "i",
    "we",
    "our",
    "or",
    "if",
    "then",
    "there",
    "here",
    "about",
    "into",
    "over",
    "under",
    "up",
    "down",
    "out",
    "off",
    "very",
    "can",
    "could",
    "should",
    "would",
    "just",
    "not",
    "no",
    "yes",
}


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    """Lowercase and compact whitespace."""
    text = text.strip().lower()
    return _WHITESPACE_RE.sub(" ", text)


def normalize_phrase(text: str) -> str:
    """Normalize concept phrases while preserving spaces between terms."""
    text = normalize_text(text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def simple_pluralize(token: str) -> str:
    token = token.lower().strip()
    if not token:
        return token
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    if token.endswith(("s", "x", "z", "ch", "sh")):
        return token + "es"
    return token + "s"


def simple_morph_variants(token: str) -> List[str]:
    """Conservative lexical variants for single-token concepts."""
    token = normalize_phrase(token)
    if " " in token:
        return [token]
    variants = {token, simple_pluralize(token)}
    if token.endswith("ing") and len(token) > 5:
        variants.add(token[:-3])
    if token.endswith("ed") and len(token) > 4:
        variants.add(token[:-2])
    return sorted(v for v in variants if v)


def tokenize_caption(
    text: str,
    min_token_len: int = 3,
    stopwords: set[str] | None = None,
) -> List[str]:
    """Tokenize and filter caption text into content words."""
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    text = normalize_text(text)
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    tokens = [t for t in _WHITESPACE_RE.split(text) if t]
    filtered = []
    for tok in tokens:
        if tok.isdigit():
            continue
        if len(tok) < min_token_len:
            continue
        if tok in stopwords:
            continue
        filtered.append(tok)
    return filtered


def contains_any_phrase(text: str, phrases: set[str]) -> bool:
    text_norm = f" {normalize_phrase(text)} "
    for phrase in phrases:
        p = normalize_phrase(phrase)
        if not p:
            continue
        if f" {p} " in text_norm:
            return True
    return False

