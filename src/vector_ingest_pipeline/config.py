from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class IngestConfig:
    processed_rag_root: Path = DATA_ROOT / "processed_rag"
    text_model_path: Path = Path("/home/lf/mount/LLM/model/Qwen3-Embedding-0.6B")
    collection_name: str = "recsys_paper_rag"
    milvus_uri: str = os.getenv("MILVUS_URI", "http://172.16.229.202:19530")
    milvus_user: str = os.getenv("MILVUS_USER", "root")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "Milvus")
    milvus_db_name: str = os.getenv("MILVUS_DB_NAME", "recommendation_system")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_model: str = os.getenv("DASHSCOPE_MULTIMODAL_EMBED_MODEL", "qwen3-vl-embedding")
    text_dim: int = int(os.getenv("TEXT_EMBED_DIM", "1024"))
    multimodal_dim: int = int(os.getenv("MULTIMODAL_EMBED_DIM", "2560"))
    max_text_length: int = int(os.getenv("MILVUS_TEXT_MAX_LENGTH", "6000"))
    max_metadata_length: int = int(os.getenv("MILVUS_METADATA_MAX_LENGTH", "20000"))
    dashscope_rpm: int = int(os.getenv("DASHSCOPE_RPM", "120"))
    text_cache_path: Path = DATA_ROOT / "processed_rag" / "text_embedding_cache.jsonl"
    multimodal_cache_path: Path = DATA_ROOT / "processed_rag" / "multimodal_embedding_cache.jsonl"
    bm25_model_path: Path = DATA_ROOT / "processed_rag" / "bm25_sparse_model.json"
