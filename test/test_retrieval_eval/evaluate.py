#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from milvus_db.client import create_client
from milvus_db.config import add_connection_args, settings_from_args
from vector_ingest_pipeline.config import IngestConfig

from .config import EvalConfig
from .queries import EvalQuery, load_queries
from .retrievers import RetrievalEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Milvus retrieval quality for recsys paper RAG.")
    add_connection_args(parser)
    parser.add_argument("--queries", type=Path, default=EvalConfig.sample_queries_path)
    parser.add_argument("--output-dir", type=Path, default=EvalConfig.output_dir)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sparse", "text_dense", "hybrid"],
        choices=["sparse", "text_dense", "multimodal_dense", "hybrid"],
    )
    parser.add_argument("--text-model-path", type=Path, default=IngestConfig.text_model_path)
    parser.add_argument("--bm25-model-path", type=Path, default=IngestConfig.bm25_model_path)
    parser.add_argument("--multimodal-model", default=IngestConfig.dashscope_model)
    parser.add_argument("--text-device", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Load queries and config without connecting to Milvus.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = settings_from_args(args)
    settings.collection_name = args.collection_name
    config = EvalConfig(
        collection_name=args.collection_name,
        text_model_path=args.text_model_path,
        bm25_model_path=args.bm25_model_path,
        dashscope_model=args.multimodal_model,
        output_dir=args.output_dir,
        sample_queries_path=args.queries,
    )

    queries = load_queries(args.queries)
    if args.limit_queries is not None:
        queries = queries[: args.limit_queries]

    print(f"queries: {len(queries)}")
    print(f"collection: {settings.collection_name}")
    print(f"database: {settings.db_name}")
    print(f"modes: {', '.join(args.modes)}")

    if args.dry_run:
        print(json.dumps([query.__dict__ for query in queries[:3]], ensure_ascii=False, indent=2))
        return 0

    client = create_client(settings)
    evaluator = RetrievalEvaluator(
        client,
        settings,
        text_model_path=config.text_model_path,
        bm25_model_path=config.bm25_model_path,
        dashscope_api_key=config.dashscope_api_key,
        dashscope_model=config.dashscope_model,
        text_device=args.text_device,
        embed_batch_size=args.embed_batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = evaluator.collection_stats()
    sample_rows = evaluator.query_sample(limit=args.sample_limit)
    results = run_queries(evaluator, queries, args.modes, top_k=args.top_k)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = args.output_dir / f"retrieval_results_{timestamp}.jsonl"
    summary_path = args.output_dir / f"summary_{timestamp}.md"
    sample_path = args.output_dir / f"sample_rows_{timestamp}.json"

    write_results(result_path, results)
    sample_path.write_text(json.dumps(sample_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_summary(summary_path, stats=stats, sample_rows=sample_rows, results=results, modes=args.modes)

    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    print(f"Sample rows: {sample_path}")
    return 0


def run_queries(
    evaluator: RetrievalEvaluator,
    queries: list[EvalQuery],
    modes: list[str],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries, start=1):
        print(f"[{query_index}/{len(queries)}] {query.query_id}: {query.query}")
        for mode in modes:
            try:
                hits = search_by_mode(evaluator, mode, query, top_k)
                assessment = assess_hits(query, hits)
                all_results.append(
                    {
                        "query_id": query.query_id,
                        "query": query.query,
                        "query_type": query.query_type,
                        "mode": mode,
                        "top_k": top_k,
                        "assessment": assessment,
                        "hits": hits,
                    }
                )
                print(f"  - {mode}: hits={len(hits)}, top1={assessment['top1_title']}")
            except Exception as exc:
                all_results.append(
                    {
                        "query_id": query.query_id,
                        "query": query.query,
                        "query_type": query.query_type,
                        "mode": mode,
                        "top_k": top_k,
                        "error": str(exc),
                        "hits": [],
                    }
                )
                print(f"  - {mode}: ERROR {exc}")
    return all_results


def search_by_mode(
    evaluator: RetrievalEvaluator,
    mode: str,
    query: EvalQuery,
    top_k: int,
) -> list[dict[str, Any]]:
    expr = build_filter(query)
    if mode == "sparse":
        return evaluator.search_sparse(query.query, top_k, expr=expr)
    if mode == "text_dense":
        return evaluator.search_text_dense(query.query, top_k, expr=expr)
    if mode == "multimodal_dense":
        return evaluator.search_multimodal_dense(query.query, top_k, expr=expr)
    if mode == "hybrid":
        return evaluator.search_hybrid(query.query, top_k, expr=expr)
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def build_filter(query: EvalQuery) -> str:
    filters: list[str] = []
    if query.expected_category:
        filters.append(f'category == "{query.expected_category}"')
    if query.expected_topic:
        filters.append(f'topic == "{query.expected_topic}"')
    return " and ".join(filters)


def assess_hits(query: EvalQuery, hits: list[dict[str, Any]]) -> dict[str, Any]:
    top1 = hits[0] if hits else {}
    expected_terms = [term.lower() for term in query.expected_terms]
    matched_terms: list[str] = []
    for hit in hits:
        haystack = " ".join(
            str(hit.get(field) or "")
            for field in ["title", "text", "topic", "paper_id", "category"]
        ).lower()
        matched_terms.extend(term for term in expected_terms if term in haystack)
    return {
        "hit_count": len(hits),
        "top1_title": top1.get("title", ""),
        "top1_category": top1.get("category", ""),
        "top1_topic": top1.get("topic", ""),
        "matched_expected_terms": sorted(set(matched_terms)),
        "expected_term_recall": len(set(matched_terms)) / len(expected_terms) if expected_terms else None,
    }


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def write_summary(
    path: Path,
    *,
    stats: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    modes: list[str],
) -> None:
    lines = [
        "# Retrieval Evaluation Summary",
        "",
        f"- collection rows: {stats.get('row_count', stats)}",
        f"- sample rows checked: {len(sample_rows)}",
        f"- modes: {', '.join(modes)}",
        f"- result groups: {len(results)}",
        "",
        "## Mode Overview",
        "",
    ]
    for mode in modes:
        mode_results = [item for item in results if item.get("mode") == mode]
        errors = [item for item in mode_results if item.get("error")]
        non_empty = [item for item in mode_results if item.get("hits")]
        lines.append(f"- `{mode}`: groups={len(mode_results)}, non_empty={len(non_empty)}, errors={len(errors)}")
    lines.extend(["", "## Top Results", ""])
    for item in results:
        if item.get("error"):
            lines.append(f"- `{item['mode']}` `{item['query_id']}`: ERROR {item['error']}")
            continue
        assessment = item.get("assessment", {})
        lines.append(
            f"- `{item['mode']}` `{item['query_id']}`: "
            f"{assessment.get('top1_title', '')} "
            f"({assessment.get('top1_category', '')}, {assessment.get('top1_topic', '')})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
