from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from pymilvus import DataType, MilvusClient

from ._paths import ensure_project_paths

ensure_project_paths()

from milvus_db.client import create_client
from milvus_db.config import MilvusSettings
from project.recommendate_project.myllm import embedding
from vector_ingest_pipeline.utils import truncate


CONTEXT_COLLECTION_NAME = os.getenv("RECSYS_CONTEXT_COLLECTION", "recsys_context_history")
thread_pool = ThreadPoolExecutor(max_workers=4)


def ensure_context_collection(client: MilvusClient, settings: MilvusSettings) -> None:
    if client.has_collection(CONTEXT_COLLECTION_NAME):
        return

    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="context_text", datatype=DataType.VARCHAR, max_length=6000)
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=2000, nullable=True)
    schema.add_field(field_name="user", datatype=DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field(field_name="timestamp", datatype=DataType.INT64, nullable=True)
    schema.add_field(field_name="message_type", datatype=DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field(field_name="evaluation_score", datatype=DataType.FLOAT, nullable=True)
    schema.add_field(field_name="context_dense", datatype=DataType.FLOAT_VECTOR, dim=settings.text_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="context_dense",
        index_name="context_dense_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    client.create_collection(
        collection_name=CONTEXT_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


class OptimizedMilvusAsyncWriter:
    def __init__(
        self,
        client: MilvusClient,
        settings: MilvusSettings,
        collection_name: str = CONTEXT_COLLECTION_NAME,
        similarity_threshold: float = 0.85,
    ):
        self.client = client
        self.settings = settings
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        ensure_context_collection(client, settings)

    def _get_dense_vector(self, text: str) -> list[float] | None:
        try:
            return embedding.embed_query(text)
        except Exception:
            return None

    def _find_similar_records(self, question_vector: list[float], user: str | None, top_k: int = 3) -> list[dict]:
        try:
            expr = f'user == "{user}"' if user else ""
            results = self.client.search(
                collection_name=self.collection_name,
                data=[question_vector],
                anns_field="context_dense",
                limit=top_k,
                filter=expr,
                output_fields=["id", "context_text", "question", "user", "timestamp", "evaluation_score"],
                search_params={"metric_type": "IP", "params": {}},
                timeout=self.settings.timeout,
            )
            similar = []
            for hit in results[0] if results else []:
                entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {}) or {}
                score = hit.get("distance", 0) if isinstance(hit, dict) else getattr(hit, "distance", 0)
                if score >= self.similarity_threshold:
                    similar.append(
                        {
                            "id": hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None),
                            "context_text": entity.get("context_text"),
                            "evaluation_score": entity.get("evaluation_score") or 0,
                            "distance": score,
                        }
                    )
            return similar
        except Exception:
            return []

    def _sync_insert(self, data: dict[str, Any]) -> None:
        self.client.insert(collection_name=self.collection_name, data=data)

    def _sync_update(self, record_id: int, new_data: dict[str, Any]) -> None:
        self.client.delete(collection_name=self.collection_name, ids=[record_id])
        self._sync_insert(new_data)

    async def async_insert(
        self,
        context_text: str,
        user: str,
        message_type: str = "AIMessage",
        question: str | None = None,
        evaluation_score: float | None = None,
        enable_dedup: bool = True,
    ) -> None:
        context_text = truncate(context_text, 6000)
        context_vector = self._get_dense_vector(context_text)
        if context_vector is None:
            return

        data = {
            "context_text": context_text,
            "question": truncate(question, 2000) if question else None,
            "user": user,
            "timestamp": int(time.time() * 1000),
            "message_type": message_type,
            "evaluation_score": float(evaluation_score or 0.0),
            "context_dense": context_vector,
        }

        if enable_dedup and question:
            question_vector = self._get_dense_vector(question)
            if question_vector is not None:
                similar = self._find_similar_records(question_vector, user=user)
                if similar:
                    best = similar[0]
                    if float(evaluation_score or 0.0) > float(best.get("evaluation_score") or 0.0):
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(thread_pool, self._sync_update, int(best["id"]), data)
                    return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, self._sync_insert, data)


@lru_cache(maxsize=1)
def get_milvus_writer() -> OptimizedMilvusAsyncWriter:
    settings = MilvusSettings(collection_name=CONTEXT_COLLECTION_NAME)
    client = create_client(settings)
    return OptimizedMilvusAsyncWriter(client=client, settings=settings)

