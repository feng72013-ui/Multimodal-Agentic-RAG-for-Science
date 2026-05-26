#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from milvus_db.client import create_client
from milvus_db.collections import create_collection
from milvus_db.config import MilvusSettings
from .cache import append_vector_cache, load_vector_cache
from .config import IngestConfig
from .embeddings import DashScopeMultimodalEmbedder, LocalQwenTextEmbedder
from .records import build_ingest_records
from .sparse_bm25 import add_sparse_vectors, build_bm25_model, save_bm25_model
from .utils import utf8_len, zero_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize processed RAG data and insert into Milvus.")
    parser.add_argument("--processed-rag-root", type=Path, default=IngestConfig.processed_rag_root)
    parser.add_argument("--text-model-path", type=Path, default=IngestConfig.text_model_path)
    parser.add_argument("--collection-name", default=IngestConfig.collection_name)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text-only", action="store_true", help="Only ingest text chunks.")
    parser.add_argument("--dry-run", action="store_true", help="Build records but do not load models or insert.")
    parser.add_argument("--create-collection", action="store_true")
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--skip-insert", action="store_true")
    parser.add_argument("--text-device", default=None, help="Optional sentence-transformers device, e.g. cuda or cpu.")
    parser.add_argument("--multimodal-model", default=IngestConfig.dashscope_model)
    parser.add_argument("--text-cache-path", type=Path, default=IngestConfig.text_cache_path)
    parser.add_argument("--multimodal-cache-path", type=Path, default=IngestConfig.multimodal_cache_path)
    parser.add_argument("--bm25-model-path", type=Path, default=IngestConfig.bm25_model_path)
    parser.add_argument("--reset-text-cache", action="store_true")
    parser.add_argument("--reset-multimodal-cache", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> IngestConfig:
    config = IngestConfig()
    config.processed_rag_root = args.processed_rag_root
    config.text_model_path = args.text_model_path
    config.collection_name = args.collection_name
    config.dashscope_model = args.multimodal_model
    config.text_cache_path = args.text_cache_path
    config.multimodal_cache_path = args.multimodal_cache_path
    config.bm25_model_path = args.bm25_model_path
    return config


def milvus_settings_from_config(config: IngestConfig) -> MilvusSettings:
    return MilvusSettings(
        uri=config.milvus_uri,
        user=config.milvus_user,
        password=config.milvus_password,
        db_name=config.milvus_db_name,
        collection_name=config.collection_name,
        text_dim=config.text_dim,
        multimodal_dim=config.multimodal_dim,
        max_text_length=config.max_text_length,
        max_metadata_length=config.max_metadata_length,
        sparse_metric_type="IP",
    )


def add_embeddings(
    records: list[dict],
    config: IngestConfig,
    text_device: str | None,
    embed_batch_size: int,
) -> list[dict]:
    text_cache = load_vector_cache(config.text_cache_path)
    multimodal_cache = load_vector_cache(config.multimodal_cache_path)
    text_total = len(records)
    text_cached = 0
    multimodal_total = sum(1 for record in records if record["category"] in {"image", "table"} and record.get("image_path"))
    multimodal_cached = 0
    for record in records:
        text_key = build_text_cache_key(record, config)
        if text_key in text_cache:
            text_cached += 1
        if record["category"] in {"image", "table"} and record.get("image_path"):
            cached_vector = get_cached_multimodal_vector(record, config, multimodal_cache)
            if cached_vector is not None:
                multimodal_cached += 1

    print(
        f"[Embedding] text cache: {config.text_cache_path} "
        f"({text_cached}/{text_total} cached)"
    )
    print(
        f"[Embedding] multimodal cache: {config.multimodal_cache_path} "
        f"({multimodal_cached}/{multimodal_total} cached)"
    )

    missing_text_indices: list[int] = []
    for index, record in enumerate(records):
        text_key = build_text_cache_key(record, config)
        cached_vector = text_cache.get(text_key)
        if cached_vector is not None:
            record["text_dense"] = cached_vector
        else:
            missing_text_indices.append(index)

    if missing_text_indices:
        print(f"[Embedding] loading text model: {config.text_model_path}")
        text_embedder = LocalQwenTextEmbedder(config.text_model_path, device=text_device)
    else:
        text_embedder = None

    multimodal_embedder = None

    text_done = text_cached
    for start in range(0, len(missing_text_indices), embed_batch_size):
        batch_indices = missing_text_indices[start : start + embed_batch_size]
        batch_texts = [records[index]["text"] for index in batch_indices]
        assert text_embedder is not None
        vectors = text_embedder.embed_documents(batch_texts, batch_size=embed_batch_size)
        for offset, vector in enumerate(vectors):
            record_index = batch_indices[offset]
            record = records[record_index]
            record["text_dense"] = vector
            text_key = build_text_cache_key(record, config)
            append_vector_cache(
                config.text_cache_path,
                {
                    "key": text_key,
                    "doc_id": record["doc_id"],
                    "model": str(config.text_model_path),
                    "dimension": config.text_dim,
                    "vector": vector,
                },
            )
            text_cache[text_key] = vector
        text_done += len(batch_indices)
        print(f"[Embedding] text vectors {text_done}/{text_total} (cache hits: {text_cached})")

    multimodal_done = 0
    multimodal_hits = 0
    for record in records:
        if record["category"] in {"image", "table"} and record.get("image_path"):
            cache_key = build_multimodal_cache_key(record, config)
            cached_vector = get_cached_multimodal_vector(record, config, multimodal_cache)
            if cached_vector is not None:
                record["multimodal_dense"] = cached_vector
                multimodal_hits += 1
            else:
                if multimodal_embedder is None:
                    multimodal_embedder = DashScopeMultimodalEmbedder(
                        api_key=config.dashscope_api_key,
                        model=config.dashscope_model,
                        rpm=config.dashscope_rpm,
                    )
                vector = multimodal_embedder.embed_image_text(record["image_path"], record["text"])
                record["multimodal_dense"] = vector
                append_vector_cache(
                    config.multimodal_cache_path,
                    {
                        "key": cache_key,
                        "doc_id": record["doc_id"],
                        "model": config.dashscope_model,
                        "dimension": config.multimodal_dim,
                        "image_path": record["image_path"],
                        "vector": vector,
                    },
                )
                multimodal_cache[cache_key] = vector
            multimodal_done += 1
        else:
            record["multimodal_dense"] = zero_vector(config.multimodal_dim)
        if multimodal_done and (multimodal_done % 20 == 0 or multimodal_done == multimodal_total):
            print(
                f"[Embedding] multimodal vectors {multimodal_done}/{multimodal_total} "
                f"(cache hits: {multimodal_hits})"
            )

    return records


def build_text_cache_key(record: dict, config: IngestConfig) -> str:
    payload = {
        "model": str(config.text_model_path),
        "dim": config.text_dim,
        "text": record.get("text") or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_multimodal_cache_key(record: dict, config: IngestConfig, dim: int | None = None) -> str:
    payload = {
        "model": config.dashscope_model,
        "dim": config.multimodal_dim if dim is None else dim,
        "image_path": record.get("image_path") or "",
        "text": record.get("text") or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_multimodal_vector(record: dict, config: IngestConfig, cache: dict[str, list[float]]) -> list[float] | None:
    cache_keys = [build_multimodal_cache_key(record, config)]
    if config.multimodal_dim != 1024:
        cache_keys.append(build_multimodal_cache_key(record, config, dim=1024))

    for cache_key in cache_keys:
        vector = cache.get(cache_key)
        if vector is not None and len(vector) == config.multimodal_dim:
            return vector
    return None


def validate_records_for_insert(records: list[dict], config: IngestConfig) -> None:
    varchar_limits = {
        "doc_id": 2000,
        "category": 64,
        "topic": 512,
        "paper_id": 1000,
        "title": 1000,
        "text": config.max_text_length,
        "filename": 1000,
        "filetype": 64,
        "image_path": 1000,
        "metadata_json": config.max_metadata_length,
    }
    for index, record in enumerate(records):
        for field_name, max_bytes in varchar_limits.items():
            actual_bytes = utf8_len(record.get(field_name))
            if actual_bytes > max_bytes:
                raise ValueError(
                    "Invalid VARCHAR byte length before Milvus insert: "
                    f"record_index={index}, doc_id={record.get('doc_id')}, "
                    f"field={field_name}, max_bytes={max_bytes}, actual_bytes={actual_bytes}"
                )

        for field_name, expected_dim in (
            ("text_dense", config.text_dim),
            ("multimodal_dense", config.multimodal_dim),
        ):
            vector = record.get(field_name)
            actual_dim = len(vector) if isinstance(vector, list) else None
            if actual_dim != expected_dim:
                raise ValueError(
                    "Invalid vector dimension before Milvus insert: "
                    f"record_index={index}, doc_id={record.get('doc_id')}, "
                    f"field={field_name}, expected_dim={expected_dim}, actual_dim={actual_dim}"
                )


def insert_batches(client, collection_name: str, records: list[dict], batch_size: int) -> int:
    inserted = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        result = client.insert(collection_name=collection_name, data=batch)
        inserted += int(result.get("insert_count", len(batch)))
        print(f"[Milvus] inserted {inserted}/{len(records)}")
    return inserted


def validate_collection_for_insert(client, collection_name: str) -> None:
    schema = client.describe_collection(collection_name)
    fields = schema.get("fields", [])
    sparse_field = next((field for field in fields if field.get("name") == "sparse"), None)
    if sparse_field is None:
        raise RuntimeError(f"Milvus collection `{collection_name}` is missing the `sparse` field.")

    print("[Milvus] sparse field: client-side BM25 vectors will be inserted explicitly.")


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.reset_text_cache and config.text_cache_path.exists():
        config.text_cache_path.unlink()
        print(f"[Embedding] removed cache file: {config.text_cache_path}")
    if args.reset_multimodal_cache and config.multimodal_cache_path.exists():
        config.multimodal_cache_path.unlink()
        print(f"[Embedding] removed cache file: {config.multimodal_cache_path}")

    records = build_ingest_records(config, include_images=not args.text_only)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"records: {len(records)}")
    print(f"collection: {config.collection_name}")
    print(f"database: {config.milvus_db_name}")
    print(f"processed_rag_root: {config.processed_rag_root}")

    if args.dry_run:
        preview = records[:3]
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:6000])
        return 0

    if not args.skip_insert and not args.create_collection:
        print("[Notice] --create-collection was not set; assuming collection already exists.")

    if args.create_collection or not args.skip_insert:
        settings = milvus_settings_from_config(config)
        client = create_client(settings)
    else:
        client = None

    if args.create_collection and client is not None:
        create_collection(settings, "recsys", drop_existing=args.drop_existing)
        client = create_client(settings)

    if args.skip_insert:
        print("[Milvus] skip insert requested.")
        return 0

    assert client is not None
    validate_collection_for_insert(client, config.collection_name)
    bm25_model = build_bm25_model(records)
    save_bm25_model(config.bm25_model_path, bm25_model)
    add_sparse_vectors(records, bm25_model)
    print(
        f"[BM25] sparse vectors built for {len(records)} records; "
        f"vocab={len(bm25_model.vocab)}; model={config.bm25_model_path}"
    )
    records = add_embeddings(records, config, args.text_device, args.embed_batch_size)
    validate_records_for_insert(records, config)

    inserted = insert_batches(client, config.collection_name, records, args.batch_size)
    print(f"[Milvus] total inserted: {inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
