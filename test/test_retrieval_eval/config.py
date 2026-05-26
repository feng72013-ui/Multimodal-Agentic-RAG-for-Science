from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vector_ingest_pipeline.config import PROJECT_ROOT, IngestConfig


@dataclass
class EvalConfig:
    collection_name: str = IngestConfig.collection_name
    text_model_path: Path = IngestConfig.text_model_path
    bm25_model_path: Path = IngestConfig.bm25_model_path
    dashscope_api_key: str = IngestConfig.dashscope_api_key
    dashscope_model: str = IngestConfig.dashscope_model
    output_dir: Path = PROJECT_ROOT / "test_retrieval_eval" / "results"
    sample_queries_path: Path = PROJECT_ROOT / "test_retrieval_eval" / "sample_queries.jsonl"


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
