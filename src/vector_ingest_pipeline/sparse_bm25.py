from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower()) or ["__empty__"]


@dataclass
class SparseBM25Model:
    vocab: dict[str, int]
    idf: dict[str, float]
    avgdl: float
    k1: float = 1.2
    b: float = 0.75

    def encode_document(self, text: str) -> dict[int, float]:
        tokens = tokenize(text)
        counts = Counter(tokens)
        doc_len = len(tokens)
        length_norm = self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
        vector: dict[int, float] = {}
        for token, tf in counts.items():
            index = self.vocab.get(token)
            if index is None:
                continue
            weight = self.idf[token] * (tf * (self.k1 + 1)) / (tf + length_norm)
            if weight > 0:
                vector[index] = float(weight)
        return vector

    def encode_query(self, text: str) -> dict[int, float]:
        counts = Counter(tokenize(text))
        vector: dict[int, float] = {}
        for token, tf in counts.items():
            index = self.vocab.get(token)
            if index is not None:
                vector[index] = float(tf)
        return vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab": self.vocab,
            "idf": self.idf,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SparseBM25Model":
        return cls(
            vocab={str(key): int(value) for key, value in payload["vocab"].items()},
            idf={str(key): float(value) for key, value in payload["idf"].items()},
            avgdl=float(payload["avgdl"]),
            k1=float(payload.get("k1", 1.2)),
            b=float(payload.get("b", 0.75)),
        )


def build_bm25_model(records: list[dict], k1: float = 1.2, b: float = 0.75) -> SparseBM25Model:
    tokenized = [tokenize(record.get("text") or "") for record in records]
    doc_count = len(tokenized)
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))

    vocab = {token: index for index, token in enumerate(sorted(doc_freq))}
    idf = {
        token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }
    avgdl = sum(len(tokens) for tokens in tokenized) / max(doc_count, 1)
    return SparseBM25Model(vocab=vocab, idf=idf, avgdl=avgdl, k1=k1, b=b)


def add_sparse_vectors(records: list[dict], model: SparseBM25Model) -> list[dict]:
    for record in records:
        record["sparse"] = model.encode_document(record.get("text") or "")
    return records


def save_bm25_model(path: Path, model: SparseBM25Model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), ensure_ascii=False), encoding="utf-8")


def load_bm25_model(path: Path) -> SparseBM25Model:
    return SparseBM25Model.from_dict(json.loads(path.read_text(encoding="utf-8")))
