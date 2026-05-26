from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class MilvusSettings:
    uri: str = os.getenv("MILVUS_URI", "http://172.16.229.202:19530")
    user: str = os.getenv("MILVUS_USER", "root")
    password: str = os.getenv("MILVUS_PASSWORD", "Milvus")
    token: str = os.getenv("MILVUS_TOKEN", "")
    db_name: str = os.getenv("MILVUS_DB_NAME", "recommendation_system")
    collection_name: str = os.getenv("MILVUS_COLLECTION", "recsys_paper_rag")
    timeout: float = float(os.getenv("MILVUS_TIMEOUT", "10"))
    text_dim: int = int(os.getenv("TEXT_EMBED_DIM", "1024"))
    multimodal_dim: int = int(os.getenv("MULTIMODAL_EMBED_DIM", "2560"))
    max_text_length: int = int(os.getenv("MILVUS_TEXT_MAX_LENGTH", "6000"))
    max_metadata_length: int = int(os.getenv("MILVUS_METADATA_MAX_LENGTH", "20000"))
    sparse_metric_type: str = "IP"


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uri", default=MilvusSettings.uri)
    parser.add_argument("--user", default=MilvusSettings.user)
    parser.add_argument("--password", default=MilvusSettings.password)
    parser.add_argument("--token", default=MilvusSettings.token)
    parser.add_argument("--db-name", default=MilvusSettings.db_name)
    parser.add_argument("--collection-name", default=MilvusSettings.collection_name)
    parser.add_argument("--timeout", type=float, default=MilvusSettings.timeout)
    parser.add_argument("--sparse-metric-type", default=MilvusSettings.sparse_metric_type)


def settings_from_args(args: argparse.Namespace) -> MilvusSettings:
    return MilvusSettings(
        uri=args.uri,
        user=args.user,
        password=args.password,
        token=args.token,
        db_name=args.db_name,
        collection_name=args.collection_name,
        timeout=args.timeout,
        sparse_metric_type=args.sparse_metric_type,
    )
