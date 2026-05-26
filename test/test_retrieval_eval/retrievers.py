from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from milvus_db.config import MilvusSettings
from vector_ingest_pipeline.embeddings import DashScopeMultimodalEmbedder, LocalQwenTextEmbedder
from vector_ingest_pipeline.sparse_bm25 import SparseBM25Model, load_bm25_model

from .config import DEFAULT_OUTPUT_FIELDS


class RetrievalEvaluator:
    def __init__(
        self,
        client,
        settings: MilvusSettings,
        *,
        text_model_path: Path,
        bm25_model_path: Path,
        dashscope_api_key: str,
        dashscope_model: str,
        text_device: str | None = None,
        embed_batch_size: int = 8,
    ) -> None:
        self.client = client
        self.settings = settings
        self.collection_name = settings.collection_name
        self.text_model_path = text_model_path
        self.bm25_model_path = bm25_model_path
        self.dashscope_api_key = dashscope_api_key
        self.dashscope_model = dashscope_model
        self.text_device = text_device
        self.embed_batch_size = embed_batch_size
        self._bm25_model: SparseBM25Model | None = None
        self._text_embedder: LocalQwenTextEmbedder | None = None
        self._multimodal_embedder: DashScopeMultimodalEmbedder | None = None

    def search_sparse(self, query: str, top_k: int, expr: str = "") -> list[dict[str, Any]]:
        sparse_query = self.bm25_model.encode_query(query)
        if not sparse_query:
            return []
        return self._search(
            data=[sparse_query],
            anns_field="sparse",
            top_k=top_k,
            search_params={"metric_type": self.settings.sparse_metric_type, "params": {"drop_ratio_search": 0.2}},
            expr=expr,
        )

    def search_text_dense(self, query: str, top_k: int, expr: str = "") -> list[dict[str, Any]]:
        vector = self.text_embedder.embed_documents([query], batch_size=self.embed_batch_size)[0]
        return self._search(
            data=[vector],
            anns_field="text_dense",
            top_k=top_k,
            search_params={"metric_type": "IP", "params": {}},
            expr=expr,
        )

    def search_multimodal_dense(self, query: str, top_k: int, expr: str = "") -> list[dict[str, Any]]:
        if not self.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for multimodal_dense query embedding.")
        vector = self.multimodal_embedder.embed_text(query)
        return self._search(
            data=[vector],
            anns_field="multimodal_dense",
            top_k=top_k,
            search_params={"metric_type": "IP", "params": {}},
            expr=expr,
        )

    def search_hybrid(self, query: str, top_k: int, expr: str = "") -> list[dict[str, Any]]:
        sparse_hits = self.search_sparse(query, top_k * 2, expr=expr)
        dense_hits = self.search_text_dense(query, top_k * 2, expr=expr)
        return reciprocal_rank_fusion([sparse_hits, dense_hits], top_k=top_k)

    def query_sample(self, limit: int = 5) -> list[dict[str, Any]]:
        return self.client.query(
            collection_name=self.collection_name,
            filter="",
            limit=limit,
            output_fields=DEFAULT_OUTPUT_FIELDS,
            timeout=self.settings.timeout,
        )

    def collection_stats(self) -> dict[str, Any]:
        return self.client.get_collection_stats(self.collection_name, timeout=self.settings.timeout)

    def _search(
        self,
        *,
        data: list[Any],
        anns_field: str,
        top_k: int,
        search_params: dict[str, Any],
        expr: str = "",
    ) -> list[dict[str, Any]]:
        raw = self.client.search(
            collection_name=self.collection_name,
            data=data,
            anns_field=anns_field,
            limit=top_k,
            filter=expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
            search_params=search_params,
            timeout=self.settings.timeout,
        )
        return normalize_hits(raw[0] if raw else [])

    @property
    def bm25_model(self) -> SparseBM25Model:
        if self._bm25_model is None:
            self._bm25_model = load_bm25_model(self.bm25_model_path)
        return self._bm25_model

    @property
    def text_embedder(self) -> LocalQwenTextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = LocalQwenTextEmbedder(self.text_model_path, device=self.text_device)
        return self._text_embedder

    @property
    def multimodal_embedder(self) -> DashScopeMultimodalEmbedder:
        if self._multimodal_embedder is None:
            self._multimodal_embedder = DashScopeMultimodalEmbedder(
                api_key=self.dashscope_api_key,
                model=self.dashscope_model,
            )
        return self._multimodal_embedder


def normalize_hits(hits: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        if isinstance(hit, dict):
            entity = hit.get("entity", {}) or {}
            score = hit.get("distance", hit.get("score", 0.0))
            hit_id = hit.get("id", entity.get("id"))
        else:
            entity = getattr(hit, "fields", None) or getattr(hit, "entity", {}) or {}
            score = getattr(hit, "distance", getattr(hit, "score", 0.0))
            hit_id = getattr(hit, "id", entity.get("id") if isinstance(entity, dict) else None)
        if not isinstance(entity, dict):
            entity = {}
        if "entity" in entity and isinstance(entity["entity"], dict):
            entity = entity["entity"]
        item = dict(entity)
        item["id"] = item.get("id", hit_id)
        item["score"] = float(score)
        item["rank"] = rank
        normalized.append(item)
    return normalized


def reciprocal_rank_fusion(result_sets: list[list[dict[str, Any]]], top_k: int, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    merged: dict[str, dict[str, Any]] = {}
    for hits in result_sets:
        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit.get("doc_id") or hit.get("id"))
            scores[doc_id] += 1.0 / (k + rank)
            merged.setdefault(doc_id, dict(hit))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    output: list[dict[str, Any]] = []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        hit = merged[doc_id]
        hit["score"] = score
        hit["rank"] = rank
        output.append(hit)
    return output
