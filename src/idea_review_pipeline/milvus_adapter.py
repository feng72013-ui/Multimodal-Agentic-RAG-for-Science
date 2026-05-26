from __future__ import annotations

import os
import json
from collections import defaultdict
from typing import Any

from milvus_db.client import create_client
from milvus_db.config import MilvusSettings
from profile_store_pipeline.common import PAPER_PROFILE_COLLECTION
try:
    from test_retrieval_eval.retrievers import RetrievalEvaluator
except ModuleNotFoundError:
    from test.test_retrieval_eval.retrievers import RetrievalEvaluator
from vector_ingest_pipeline.config import IngestConfig
from vector_ingest_pipeline.utils import truncate
from idea_review_pipeline.review import expand_query_text


OUTPUT_FIELDS = [
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

PROFILE_OUTPUT_FIELDS = [
    "id",
    "profile_key",
    "topic",
    "paper_id",
    "title",
    "year",
    "source_pdf",
    "quality_score",
    "quality_level",
    "review_status",
    "review_score",
    "profile_text",
    "metadata_json",
]


def retrieve_idea_context_from_milvus(
    query: str,
    *,
    top_k_papers: int = 5,
    initial_top_k: int = 30,
    chunks_per_paper: int = 8,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Retrieve candidate papers and evidence chunks from Milvus.

    Prefer the Stage 2 paper-level `paper_profiles` collection when it exists,
    then drill down into the main RAG chunk collection for evidence.
    """
    settings = MilvusSettings()
    if collection_name:
        settings.collection_name = collection_name
    config = IngestConfig()
    config.collection_name = settings.collection_name
    client = create_client(settings)
    expanded_query = expand_query_text(query.lower())
    profile_hits = [] if collection_name else search_stage2_profiles(client, settings, expanded_query, top_k_papers)
    if profile_hits:
        profiles = [profile_from_stage2_hit(hit) for hit in profile_hits]
        retrieval_layer = "paper_profiles"
        hit_count = len(profile_hits)
    else:
        evaluator = build_retrieval_evaluator(client, settings, config)
        hits = search_milvus_candidates(evaluator, expanded_query, initial_top_k)
        paper_groups = group_hits_by_paper(hits)
        ranked_groups = sorted(
            paper_groups.items(),
            key=lambda item: max(float(hit.get("score") or 0.0) for hit in item[1]),
            reverse=True,
        )[:top_k_papers]
        profiles = [profile_from_hits(key, values) for key, values in ranked_groups]
        retrieval_layer = "rag_chunks"
        hit_count = len(hits)

    chunks = []
    for profile in profiles:
        chunks.extend(
            query_paper_chunks(
                client,
                settings,
                profile,
                limit=chunks_per_paper,
            )
        )
    if not chunks:
        chunks = []
    return {
        "source": "milvus",
        "collection": settings.collection_name,
        "profile_collection": PAPER_PROFILE_COLLECTION,
        "retrieval_layer": retrieval_layer,
        "hit_count": hit_count,
        "profiles": profiles,
        "chunks": chunks,
    }


def build_retrieval_evaluator(client, settings: MilvusSettings, config: IngestConfig) -> RetrievalEvaluator:
    return RetrievalEvaluator(
        client,
        settings,
        text_model_path=config.text_model_path,
        bm25_model_path=config.bm25_model_path,
        dashscope_api_key=config.dashscope_api_key,
        dashscope_model=config.dashscope_model,
        text_device=os.getenv("RECSYS_TEXT_DEVICE") or None,
        embed_batch_size=int(os.getenv("RECSYS_EMBED_BATCH_SIZE", "8")),
    )


def search_stage2_profiles(client, settings: MilvusSettings, query: str, top_k: int) -> list[dict[str, Any]]:
    try:
        if PAPER_PROFILE_COLLECTION not in client.list_collections(timeout=settings.timeout):
            return []
        try:
            from project.recommendate_project.myllm import embedding
        except ModuleNotFoundError:
            from myllm import embedding

        vector = embedding.embed_query(query)
        raw = client.search(
            collection_name=PAPER_PROFILE_COLLECTION,
            data=[vector],
            anns_field="text_dense",
            limit=top_k,
            output_fields=PROFILE_OUTPUT_FIELDS,
            search_params={"metric_type": "IP", "params": {}},
            timeout=settings.timeout,
        )
    except Exception:
        return []
    return normalize_profile_hits(raw[0] if raw else [])


def normalize_profile_hits(hits: list[Any]) -> list[dict[str, Any]]:
    output = []
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
        item = dict(entity)
        item["id"] = item.get("id", hit_id)
        item["score"] = float(score)
        item["rank"] = rank
        output.append(item)
    return output


def profile_from_stage2_hit(hit: dict[str, Any]) -> dict[str, Any]:
    metadata = parse_metadata(hit.get("metadata_json"))
    embedded_profile = metadata.get("profile") if isinstance(metadata.get("profile"), dict) else {}
    title = clean_title(str(embedded_profile.get("title") or hit.get("title") or ""))
    if not title:
        title = clean_title(str(hit.get("paper_id") or ""))
    profile = {
        "topic": hit.get("topic") or embedded_profile.get("topic"),
        "paper_id": hit.get("paper_id") or embedded_profile.get("paper_id"),
        "title": title,
        "source_pdf": hit.get("source_pdf") or embedded_profile.get("source_pdf"),
        "abstract": embedded_profile.get("abstract") or hit.get("profile_text") or "",
        "research_problem": embedded_profile.get("research_problem") or [hit.get("profile_text") or ""],
        "methods": embedded_profile.get("methods") or [],
        "datasets": embedded_profile.get("datasets") or [],
        "metrics": embedded_profile.get("metrics") or [],
        "results": embedded_profile.get("results") or [],
        "limitations": embedded_profile.get("limitations") or [],
        "quality": embedded_profile.get("quality")
        or {
            "score": hit.get("quality_score") or 0.0,
            "level": hit.get("quality_level") or "unknown",
            "checks": {"from_stage2_collection": True},
        },
        "retrieval": {
            "source": "milvus_paper_profiles",
            "score": hit.get("score"),
            "rank": hit.get("rank"),
            "review_status": hit.get("review_status"),
            "review_score": hit.get("review_score"),
        },
    }
    return profile


def search_milvus_candidates(evaluator: RetrievalEvaluator, query: str, top_k: int) -> list[dict[str, Any]]:
    expr = 'category == "text"'
    try:
        return evaluator.search_hybrid(query, top_k, expr=expr)
    except Exception:
        # Keep Milvus as the online source even when local dense embedding deps are
        # unavailable; sparse BM25 still uses the indexed OCR/chunk collection.
        return evaluator.search_sparse(query, top_k, expr=expr)


def group_hits_by_paper(hits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        if not hit.get("paper_id"):
            continue
        groups[paper_key(hit)].append(hit)
    return groups


def profile_from_hits(key: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    first = hits[0]
    ordered = sorted(hits, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    evidence_texts = [str(hit.get("text") or "") for hit in ordered[:5]]
    title = clean_title(str(first.get("title") or ""))
    if not title:
        title = clean_title(str(first.get("paper_id") or ""))
    return {
        "topic": first.get("topic"),
        "paper_id": first.get("paper_id"),
        "title": title,
        "source_pdf": first.get("filename"),
        "abstract": evidence_texts[0] if evidence_texts else "",
        "research_problem": evidence_texts[:2],
        "methods": evidence_texts[:3],
        "datasets": extract_terms(evidence_texts, ("movielens", "amazon", "yelp", "gowalla", "lastfm", "steam", "netflix", "taobao")),
        "metrics": extract_terms(evidence_texts, ("recall", "ndcg", "hit rate", "hr@", "mrr", "auc", "precision", "map", "f1")),
        "results": evidence_texts[:2],
        "limitations": [],
        "quality": {
            "score": 0.0,
            "level": "retrieved",
            "checks": {"from_milvus": True, "retrieved_chunks": len(hits)},
        },
        "retrieval": {
            "source": "milvus",
            "paper_key": key,
            "max_score": max((float(hit.get("score") or 0.0) for hit in hits), default=0.0),
            "hit_count": len(hits),
        },
    }


def query_paper_chunks(client, settings: MilvusSettings, profile: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    topic = escape_filter_value(profile.get("topic") or "")
    paper_id = escape_filter_value(profile.get("paper_id") or "")
    if not paper_id:
        return []
    expr = f'category == "text" and paper_id == "{paper_id}"'
    if topic:
        expr = f'category == "text" and topic == "{topic}" and paper_id == "{paper_id}"'
    try:
        rows = client.query(
            collection_name=settings.collection_name,
            filter=expr,
            limit=limit,
            output_fields=OUTPUT_FIELDS,
            timeout=settings.timeout,
        )
    except Exception:
        return []
    return [chunk_from_hit(row) for row in sorted(rows, key=lambda item: int(item.get("page_start") or 0))[:limit]]


def chunk_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": hit.get("doc_id") or str(hit.get("id") or ""),
        "topic": hit.get("topic"),
        "paper_id": hit.get("paper_id"),
        "page_start": hit.get("page_start"),
        "page_end": hit.get("page_end"),
        "title_path": [hit.get("title")] if hit.get("title") else [],
        "text": truncate(hit.get("text"), 6000),
        "metadata": {
            "source_pdf": hit.get("filename"),
            "image_path": hit.get("image_path"),
            "milvus_id": hit.get("id"),
            "score": hit.get("score"),
        },
    }


def paper_key(row: dict[str, Any]) -> str:
    return f"{row.get('topic') or ''}::{row.get('paper_id') or ''}"


def clean_title(title: str) -> str:
    if " --> " in title:
        title = title.split(" --> ", 1)[0]
    if is_section_title(title):
        return ""
    return title


def is_section_title(title: str) -> bool:
    normalized = " ".join(str(title).strip().lower().split())
    return normalized in {
        "abstract",
        "introduction",
        "1 introduction",
        "related work",
        "2 related work",
        "method",
        "methods",
        "proposed method",
        "experiments",
        "conclusion",
        "references",
    }


def extract_terms(texts: list[str], terms: tuple[str, ...]) -> list[str]:
    haystack = " ".join(texts).lower()
    return sorted({term for term in terms if term in haystack})


def escape_filter_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def parse_metadata(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
