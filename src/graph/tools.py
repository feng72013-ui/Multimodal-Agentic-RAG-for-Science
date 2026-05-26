from __future__ import annotations

import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ._paths import ensure_project_paths
from .save_context import CONTEXT_COLLECTION_NAME, ensure_context_collection

ensure_project_paths()

from milvus_db.client import create_client
from milvus_db.config import MilvusSettings
try:
    from project.recommendate_project.myllm import embedding
except ModuleNotFoundError:
    from myllm import embedding


@tool("search_context", parse_docstring=True)
async def search_context(query: str | None = None, user_name: str | None = None) -> str:
    """
    Search similar historical question-answer context for the current user.

    Args:
        query: Current user question.
        user_name: Optional user name used to filter history.

    Returns:
        Relevant historical answers, or a message that no history was found.
    """
    if not query:
        return "没有找到相关的历史上下文信息。"

    try:
        settings = MilvusSettings(collection_name=CONTEXT_COLLECTION_NAME)
        client = create_client(settings)
        ensure_context_collection(client, settings)
        vector = embedding.embed_query(query)
        expr = f'user == "{user_name}"' if user_name else ""
        result = client.search(
            collection_name=CONTEXT_COLLECTION_NAME,
            data=[vector],
            anns_field="context_dense",
            limit=3,
            filter=expr,
            output_fields=["context_text", "question", "evaluation_score"],
            search_params={"metric_type": "IP", "params": {}},
            timeout=settings.timeout,
        )
    except Exception:
        return "没有找到相关的历史上下文信息。"

    context_pieces: list[str] = []
    for hit in result[0] if result else []:
        entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {}) or {}
        score = hit.get("distance", 0) if isinstance(hit, dict) else getattr(hit, "distance", 0)
        if score >= float(os.getenv("RECSYS_CONTEXT_MIN_SCORE", "0.75")):
            question_text = entity.get("question") or ""
            context_text = entity.get("context_text") or ""
            context_pieces.append(f"历史问题：{question_text}\n历史回答：{context_text}")

    return "\n\n".join(context_pieces) if context_pieces else "没有找到相关的历史上下文信息。"


class SearchInput(BaseModel):
    query: str = Field(description="需要搜索的内容或者关键词")


@tool("my_search", args_schema=SearchInput, description="专门搜索互联网中的公开内容")
def my_search(query: str) -> str:
    try:
        from zhipuai import ZhipuAI

        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            return "没有配置 ZHIPU_API_KEY，无法执行互联网搜索。"
        client = ZhipuAI(api_key=api_key)
        response = client.web_search.web_search(search_engine="search_pro", search_query=query)
        if response.search_result:
            return "\n\n".join([item.content for item in response.search_result])
        return "没有搜索到任何内容！"
    except Exception as exc:
        return f"没有搜索到任何内容！错误：{exc}"
