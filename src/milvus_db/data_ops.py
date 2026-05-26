from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import create_client
from .config import MilvusSettings, add_connection_args, settings_from_args
from vector_ingest_pipeline.config import IngestConfig
from vector_ingest_pipeline.sparse_bm25 import load_bm25_model


def zero_vector(dim: int) -> list[float]:
    return [0.0] * dim


def sample_recsys_record(settings: MilvusSettings) -> dict:
    return {
        "doc_id": "milvus_db_sample",
        "category": "text",
        "topic": "connection_test",
        "paper_id": "sample",
        "title": "Milvus connection sample",
        "text": "This is a small test record inserted by milvus_db.data_ops.",
        "filename": "milvus_db_sample",
        "filetype": "txt",
        "image_path": None,
        "page_start": 0,
        "page_end": 0,
        "metadata_json": "{}",
        "sparse": {0: 1.0},
        "text_dense": zero_vector(settings.text_dim),
        "multimodal_dense": zero_vector(settings.multimodal_dim),
    }


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Insert, query, search, and delete Milvus data.")
    add_connection_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    insert_sample = subparsers.add_parser("insert-sample")
    insert_sample.add_argument("--collection-name", default=None)

    insert_jsonl = subparsers.add_parser("insert-jsonl")
    insert_jsonl.add_argument("--path", type=Path, required=True)
    insert_jsonl.add_argument("--collection-name", default=None)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--filter", default="")
    query_parser.add_argument("--ids", default=None)
    query_parser.add_argument("--output-fields", nargs="*", default=["id", "doc_id", "text", "title"])
    query_parser.add_argument("--collection-name", default=None)

    search_text = subparsers.add_parser("search-text")
    search_text.add_argument("--query", required=True)
    search_text.add_argument("--limit", type=int, default=5)
    search_text.add_argument("--collection-name", default=None)
    search_text.add_argument("--bm25-model-path", type=Path, default=IngestConfig.bm25_model_path)

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("--filter", default=None)
    delete_parser.add_argument("--ids", nargs="*", default=None)
    delete_parser.add_argument("--collection-name", default=None)
    delete_parser.add_argument("--yes", action="store_true")

    args = parser.parse_args()
    settings = settings_from_args(args)
    client = create_client(settings)
    collection_name = args.collection_name or settings.collection_name

    if args.command == "insert-sample":
        result = client.insert(collection_name=collection_name, data=[sample_recsys_record(settings)])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "insert-jsonl":
        records = read_jsonl(args.path)
        result = client.insert(collection_name=collection_name, data=records)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "query":
        ids = None
        if args.ids is not None:
            ids = [int(item) for item in args.ids.split(",") if item]
        result = client.query(
            collection_name=collection_name,
            filter=args.filter,
            ids=ids,
            output_fields=args.output_fields,
            timeout=settings.timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif args.command == "search-text":
        if not args.bm25_model_path.exists():
            raise SystemExit(f"BM25 model file does not exist: {args.bm25_model_path}")
        bm25_model = load_bm25_model(args.bm25_model_path)
        sparse_query = bm25_model.encode_query(args.query)
        if not sparse_query:
            raise SystemExit("Query terms are not in the BM25 vocabulary.")
        result = client.search(
            collection_name=collection_name,
            data=[sparse_query],
            anns_field="sparse",
            limit=args.limit,
            output_fields=["id", "doc_id", "category", "title", "text", "filename", "image_path"],
            search_params={"metric_type": settings.sparse_metric_type, "params": {"drop_ratio_search": 0.2}},
            timeout=settings.timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif args.command == "delete":
        if not args.yes:
            raise SystemExit("Refusing to delete data without --yes.")
        ids = [int(item) for item in args.ids] if args.ids else None
        result = client.delete(collection_name=collection_name, ids=ids, filter=args.filter)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
