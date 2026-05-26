from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Any

from langchain_core.messages import ToolMessage

from ._paths import ensure_project_paths
from .my_state import RecsysRAGState

ensure_project_paths()

from milvus_db.client import create_client
from milvus_db.config import MilvusSettings
try:
    from test_retrieval_eval.retrievers import RetrievalEvaluator, normalize_hits
except ModuleNotFoundError:
    from test.test_retrieval_eval.retrievers import RetrievalEvaluator, normalize_hits
from vector_ingest_pipeline.config import IngestConfig
from vector_ingest_pipeline.embeddings import DashScopeMultimodalEmbedder
from vector_ingest_pipeline.utils import image_to_data_url


DEFAULT_OUTPUT_FIELDS = [
    "id",
    "doc_id",
    "category",
    "topic",
    "paper_id",
    "title",
    "text",
    "filename",
    "image_path",
    "page_start",
    "page_end",
]


class SearchContextToolNode:
    """Run LLM tool calls and append ToolMessage results to the graph state."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No message found in input")

        message = messages[-1]
        tasks = []
        for tool_call in message.tool_calls:
            args = tool_call.get("args") or {}
            query = args.get("query") or inputs.get("input_text")
            user_name = args.get("user_name") or inputs.get("user")
            task = self.tools_by_name[tool_call["name"]].ainvoke(
                {"query": query, "user_name": user_name}
            )
            tasks.append((tool_call, task))

        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        outputs = []
        for (tool_call, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                result = f"工具执行错误: {result}"
            outputs.append(
                ToolMessage(
                    content=str(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class RecsysMilvusRetriever:
    def __init__(self, top_k: int = 5, text_device: str | None = None, collection_name: str | None = None):
        settings = MilvusSettings()
        if collection_name:
            settings.collection_name = collection_name
        config = IngestConfig()
        config.collection_name = settings.collection_name
        self.settings = settings
        self.config = config
        self.client = create_client(settings)
        self.collection_name = settings.collection_name
        self.top_k = top_k
        self.evaluator = RetrievalEvaluator(
            self.client,
            settings,
            text_model_path=config.text_model_path,
            bm25_model_path=config.bm25_model_path,
            dashscope_api_key=config.dashscope_api_key,
            dashscope_model=config.dashscope_model,
            text_device=text_device or os.getenv("RECSYS_TEXT_DEVICE") or None,
            embed_batch_size=int(os.getenv("RECSYS_EMBED_BATCH_SIZE", "8")),
        )
        self._multimodal_embedder: DashScopeMultimodalEmbedder | None = None

    def retrieve_text(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.top_k
        return self.evaluator.search_hybrid(query, top_k)

    def retrieve_multimodal_text(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.top_k
        return self.evaluator.search_multimodal_dense(query, top_k)

    def retrieve_image(self, image: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.top_k
        data_url = image if image.startswith("data:") else image_to_data_url(image)
        vector = self.multimodal_embedder._call([{"image": data_url}])
        raw = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="multimodal_dense",
            limit=top_k,
            filter='category in ["image", "table"]',
            output_fields=DEFAULT_OUTPUT_FIELDS,
            search_params={"metric_type": "IP", "params": {}},
            timeout=self.settings.timeout,
        )
        return normalize_hits(raw[0] if raw else [])

    @property
    def multimodal_embedder(self) -> DashScopeMultimodalEmbedder:
        if self._multimodal_embedder is None:
            self._multimodal_embedder = DashScopeMultimodalEmbedder(
                api_key=self.config.dashscope_api_key,
                model=self.config.dashscope_model,
            )
        return self._multimodal_embedder


@lru_cache(maxsize=16)
def get_retriever(collection_name: str = "") -> RecsysMilvusRetriever:
    return RecsysMilvusRetriever(
        top_k=int(os.getenv("RECSYS_RETRIEVAL_TOP_K", "5")),
        collection_name=collection_name or None,
    )


def retriever_node(state: RecsysRAGState, config=None):
    collection_name = state.get("collection_name") or ""
    if not collection_name and config:
        collection_name = (config.get("configurable") or {}).get("collection_name") or ""
    retriever = get_retriever(collection_name)
    input_image = state.get("input_image", "")
    input_text = state.get("input_text", "") or ""
    if input_image and state.get("input_type") in {"only_image", "image_with_text"}:
        image_hits = retriever.retrieve_image(input_image, top_k=5)
        if input_text and not _looks_like_image_similarity_query(input_text):
            text_hits = retriever.retrieve_text(input_text, top_k=3)
            results = _dedupe_hits(image_hits + text_hits)
        else:
            results = image_hits
    else:
        query = input_text
        results = retriever.retrieve_text(query, top_k=5)
        if _looks_like_image_table_query(query):
            try:
                multimodal_hits = retriever.retrieve_multimodal_text(query, top_k=3)
                results = _dedupe_hits(results + multimodal_hits)
            except Exception:
                pass

    docs = [_normalize_doc(hit) for hit in results]
    images = [_normalize_image(hit) for hit in results if hit.get("image_path")]
    return {"context_retrieved": docs, "images_retrieved": images}


def _normalize_doc(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": hit.get("id"),
        "doc_id": hit.get("doc_id"),
        "category": hit.get("category"),
        "topic": hit.get("topic"),
        "paper_id": hit.get("paper_id"),
        "title": hit.get("title"),
        "text": hit.get("text"),
        "filename": hit.get("filename"),
        "image_path": hit.get("image_path"),
        "page_start": hit.get("page_start"),
        "page_end": hit.get("page_end"),
        "score": hit.get("score", hit.get("distance")),
    }


def _normalize_image(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": hit.get("id"),
        "doc_id": hit.get("doc_id"),
        "category": hit.get("category"),
        "paper_id": hit.get("paper_id"),
        "title": hit.get("title"),
        "text": hit.get("text"),
        "filename": hit.get("filename"),
        "image_path": hit.get("image_path"),
        "page_start": hit.get("page_start"),
        "page_end": hit.get("page_end"),
        "score": hit.get("score", hit.get("distance")),
    }


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for hit in hits:
        key = str(hit.get("doc_id") or hit.get("id") or hit.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


def _looks_like_image_table_query(query: str) -> bool:
    lowered = query.lower()
    keywords = ["image", "figure", "fig.", "图", "图片", "框架", "architecture", "table", "表格", "benchmark"]
    return any(keyword in lowered for keyword in keywords)


def _looks_like_image_similarity_query(query: str) -> bool:
    lowered = query.lower()
    keywords = [
        "similar",
        "similarity",
        "most similar",
        "image retrieval",
        "相似",
        "最像",
        "找出",
        "检索图片",
        "文献图片",
        "图搜图",
    ]
    return any(keyword in lowered for keyword in keywords)
