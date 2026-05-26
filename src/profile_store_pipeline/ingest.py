#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from milvus_db.client import create_client
from milvus_db.collections import create_collection
from milvus_db.config import MilvusSettings, add_connection_args, settings_from_args
from profile_store_pipeline.common import (
    CITATION_EDGE_COLLECTION,
    DEFAULT_CITATION_PATH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_REVIEW_PATH,
    PAPER_PROFILE_COLLECTION,
    citation_key,
    paper_key,
    profile_text,
    read_jsonl,
)
from vector_ingest_pipeline.utils import truncate, utf8_len, zero_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert Stage 2 paper profiles and citation edges into Milvus.")
    add_connection_args(parser)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATION_PATH)
    parser.add_argument("--quality-reviews", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--profile-collection", default=PAPER_PROFILE_COLLECTION)
    parser.add_argument("--citation-collection", default=CITATION_EDGE_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-profiles", type=int, default=None)
    parser.add_argument("--limit-citations", type=int, default=None)
    parser.add_argument("--create-collections", action="store_true")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--use-embeddings", action="store_true", help="Embed profile/citation text before insert.")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--skip-citations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = settings_from_args(args)
    profiles = read_jsonl(args.profiles)
    citations = read_jsonl(args.citations) if args.citations.exists() else []
    reviews = read_jsonl(args.quality_reviews) if args.quality_reviews.exists() else []
    review_by_key = {paper_key(row): row for row in reviews}

    if args.limit_profiles:
        profiles = profiles[: args.limit_profiles]
    if args.limit_citations:
        citations = citations[: args.limit_citations]

    profile_records = (
        []
        if args.skip_profiles
        else build_profile_records(profiles, review_by_key, settings, use_embeddings=args.use_embeddings)
    )
    citation_records = (
        []
        if args.skip_citations
        else build_citation_records(citations, settings, use_embeddings=args.use_embeddings)
    )

    print(f"profile records: {len(profile_records)} -> {args.profile_collection}")
    print(f"citation records: {len(citation_records)} -> {args.citation_collection}")
    if profile_records:
        print("profile sample:")
        print(json.dumps(without_vectors(profile_records[0]), ensure_ascii=False, indent=2)[:3000])
    if citation_records:
        print("citation sample:")
        print(json.dumps(without_vectors(citation_records[0]), ensure_ascii=False, indent=2)[:2000])

    if args.dry_run:
        print("[dry-run] no Milvus collections created and no records inserted.")
        return 0

    if args.create_collections:
        if not args.skip_profiles:
            create_collection(collection_settings(settings, args.profile_collection), "paper_profile", args.drop_existing)
        if not args.skip_citations:
            create_collection(collection_settings(settings, args.citation_collection), "citation_edge", args.drop_existing)

    client = create_client(settings)
    inserted_profiles = 0 if args.skip_profiles else insert_batches(client, args.profile_collection, profile_records, args.batch_size)
    inserted_citations = 0 if args.skip_citations else insert_batches(client, args.citation_collection, citation_records, args.batch_size)
    print(f"[Milvus] inserted profiles: {inserted_profiles}")
    print(f"[Milvus] inserted citations: {inserted_citations}")
    return 0


def build_profile_records(
    profiles: list[dict[str, Any]],
    review_by_key: dict[str, dict[str, Any]],
    settings: MilvusSettings,
    *,
    use_embeddings: bool,
) -> list[dict[str, Any]]:
    texts = [profile_text(profile, settings.max_text_length) for profile in profiles]
    vectors = embed_texts(texts, settings) if use_embeddings else [zero_vector(settings.text_dim) for _ in texts]
    records = []
    for profile, text, vector in zip(profiles, texts, vectors, strict=True):
        review = review_by_key.get(paper_key(profile), {})
        quality = profile.get("quality") or {}
        record = {
            "profile_key": truncate(paper_key(profile), 1600),
            "topic": truncate(profile.get("topic"), 512),
            "paper_id": truncate(profile.get("paper_id"), 1000),
            "title": truncate(profile.get("title"), 1000),
            "year": int(profile.get("year") or 0),
            "source_pdf": truncate(profile.get("source_pdf"), 2000),
            "quality_score": float(quality.get("score") or 0.0),
            "quality_level": truncate(quality.get("level"), 64),
            "review_status": review_status(review),
            "review_score": float(review.get("faithfulness_score") or 0.0) if review else 0.0,
            "profile_text": text,
            "metadata_json": safe_metadata_json(
                {"profile": profile, "quality_review": review or None},
                max_bytes=settings.max_metadata_length,
            ),
            "text_dense": vector,
        }
        records.append(record)
    return records


def build_citation_records(
    citations: list[dict[str, Any]],
    settings: MilvusSettings,
    *,
    use_embeddings: bool,
) -> list[dict[str, Any]]:
    texts = [truncate(f"{row.get('target_title') or ''}\n{row.get('target_raw') or ''}", settings.max_text_length) for row in citations]
    vectors = embed_texts(texts, settings) if use_embeddings else [zero_vector(settings.text_dim) for _ in texts]
    records = []
    for row, text, vector in zip(citations, texts, vectors, strict=True):
        record = {
            "edge_key": truncate(citation_key(row), 1600),
            "source_topic": truncate(row.get("source_topic"), 512),
            "source_paper_id": truncate(row.get("source_paper_id"), 1000),
            "citation_marker": truncate(row.get("citation_marker"), 64),
            "target_title": truncate(row.get("target_title"), 1000),
            "source": truncate(row.get("source"), 128),
            "target_raw": truncate(row.get("target_raw"), settings.max_text_length),
            "metadata_json": safe_metadata_json(row, max_bytes=settings.max_metadata_length),
            "text_dense": vector,
        }
        records.append(record)
    return records


def embed_texts(texts: list[str], settings: MilvusSettings) -> list[list[float]]:
    try:
        from project.recommendate_project.myllm import embedding
    except ModuleNotFoundError:
        from myllm import embedding

    vectors = embedding.embed_documents(texts)
    for index, vector in enumerate(vectors):
        if len(vector) != settings.text_dim:
            raise ValueError(f"Embedding dim mismatch at index {index}: {len(vector)} != {settings.text_dim}")
    return vectors


def review_status(review: dict[str, Any]) -> str:
    if not review:
        return "not_reviewed"
    return "passed" if review.get("pass") else "review_needed"


def safe_metadata_json(value: dict[str, Any], max_bytes: int) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if utf8_len(raw) <= max_bytes:
        return raw
    compact = compact_metadata(value)
    raw = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if utf8_len(raw) <= max_bytes:
        return raw
    compact["truncated"] = True
    compact["profile"] = truncate(json.dumps(compact.get("profile") or {}, ensure_ascii=False), max_bytes // 2)
    raw = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if utf8_len(raw) <= max_bytes:
        return raw
    fallback = {
        "truncated": True,
        "profile_key": paper_key(value.get("profile") or value),
        "quality_review": value.get("quality_review"),
    }
    raw = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
    if utf8_len(raw) <= max_bytes:
        return raw
    return json.dumps({"truncated": True}, ensure_ascii=False)


def compact_metadata(value: dict[str, Any]) -> dict[str, Any]:
    profile = value.get("profile") if isinstance(value.get("profile"), dict) else value
    review = value.get("quality_review")
    key_visuals = profile.get("key_visuals") or []
    return {
        "truncated": True,
        "profile": {
            "topic": profile.get("topic"),
            "paper_id": profile.get("paper_id"),
            "title": profile.get("title"),
            "authors": profile.get("authors"),
            "year": profile.get("year"),
            "venue": profile.get("venue"),
            "doi": profile.get("doi"),
            "arxiv_id": profile.get("arxiv_id"),
            "source_pdf": profile.get("source_pdf"),
            "abstract": truncate(profile.get("abstract"), 2500),
            "research_problem": [truncate(item, 800) for item in (profile.get("research_problem") or [])[:4]],
            "methods": [truncate(item, 800) for item in (profile.get("methods") or [])[:5]],
            "datasets": profile.get("datasets"),
            "metrics": profile.get("metrics"),
            "results": [truncate(item, 800) for item in (profile.get("results") or [])[:5]],
            "conclusions": [truncate(item, 800) for item in (profile.get("conclusions") or [])[:4]],
            "limitations": [truncate(item, 800) for item in (profile.get("limitations") or [])[:4]],
            "key_visuals": [
                {
                    "label": item.get("label"),
                    "category": item.get("category"),
                    "caption": truncate(item.get("caption"), 500),
                    "image_path": item.get("image_path"),
                    "page_no": item.get("page_no"),
                }
                for item in key_visuals[:6]
                if isinstance(item, dict)
            ],
            "quality": profile.get("quality"),
            "coverage": profile.get("coverage"),
        },
        "quality_review": review,
    }


def collection_settings(settings: MilvusSettings, collection_name: str) -> MilvusSettings:
    return MilvusSettings(
        uri=settings.uri,
        user=settings.user,
        password=settings.password,
        token=settings.token,
        db_name=settings.db_name,
        collection_name=collection_name,
        timeout=settings.timeout,
        text_dim=settings.text_dim,
        multimodal_dim=settings.multimodal_dim,
        max_text_length=settings.max_text_length,
        max_metadata_length=settings.max_metadata_length,
        sparse_metric_type=settings.sparse_metric_type,
    )


def insert_batches(client, collection_name: str, records: list[dict[str, Any]], batch_size: int) -> int:
    inserted = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        result = client.insert(collection_name=collection_name, data=batch)
        inserted += int(result.get("insert_count", len(batch))) if isinstance(result, dict) else len(batch)
        print(f"[Milvus] {collection_name}: inserted {inserted}/{len(records)}")
    return inserted


def without_vectors(record: dict[str, Any]) -> dict[str, Any]:
    return {key: ("<vector>" if key.endswith("dense") else value) for key, value in record.items()}


if __name__ == "__main__":
    raise SystemExit(main())
