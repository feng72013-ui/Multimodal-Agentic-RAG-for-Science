from __future__ import annotations

from pymilvus import DataType, MilvusClient

from .config import MilvusSettings


def build_recsys_schema(client: MilvusClient, settings: MilvusSettings):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field(field_name="topic", datatype=DataType.VARCHAR, max_length=512, nullable=True)
    schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=settings.max_text_length,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
    )
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="filetype", datatype=DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="page_start", datatype=DataType.INT64, nullable=True)
    schema.add_field(field_name="page_end", datatype=DataType.INT64, nullable=True)
    schema.add_field(
        field_name="metadata_json",
        datatype=DataType.VARCHAR,
        max_length=settings.max_metadata_length,
        nullable=True,
    )
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=settings.text_dim)
    schema.add_field(field_name="multimodal_dense", datatype=DataType.FLOAT_VECTOR, dim=settings.multimodal_dim)

    return schema


def build_recsys_index_params(client: MilvusClient, settings: MilvusSettings):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=settings.sparse_metric_type,
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    index_params.add_index(
        field_name="text_dense",
        index_name="text_dense_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="multimodal_dense",
        index_name="multimodal_dense_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    return index_params


def build_reference_schema(client: MilvusClient):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=6000,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
    )
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="filetype", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)
    return schema


def build_reference_index_params(client: MilvusClient, settings: MilvusSettings):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=settings.sparse_metric_type,
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    index_params.add_index(
        field_name="dense",
        index_name="dense_inverted_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    return index_params


def build_paper_profile_schema(client: MilvusClient, settings: MilvusSettings):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="profile_key", datatype=DataType.VARCHAR, max_length=1600)
    schema.add_field(field_name="topic", datatype=DataType.VARCHAR, max_length=512, nullable=True)
    schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="year", datatype=DataType.INT64, nullable=True)
    schema.add_field(field_name="source_pdf", datatype=DataType.VARCHAR, max_length=2000, nullable=True)
    schema.add_field(field_name="quality_score", datatype=DataType.DOUBLE, nullable=True)
    schema.add_field(field_name="quality_level", datatype=DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field(field_name="review_status", datatype=DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field(field_name="review_score", datatype=DataType.DOUBLE, nullable=True)
    schema.add_field(
        field_name="profile_text",
        datatype=DataType.VARCHAR,
        max_length=settings.max_text_length,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
    )
    schema.add_field(
        field_name="metadata_json",
        datatype=DataType.VARCHAR,
        max_length=settings.max_metadata_length,
        nullable=True,
    )
    schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=settings.text_dim)
    return schema


def build_paper_profile_index_params(client: MilvusClient, settings: MilvusSettings):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="text_dense",
        index_name="paper_profile_text_dense_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    return index_params


def build_citation_edge_schema(client: MilvusClient, settings: MilvusSettings):
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="edge_key", datatype=DataType.VARCHAR, max_length=1600)
    schema.add_field(field_name="source_topic", datatype=DataType.VARCHAR, max_length=512, nullable=True)
    schema.add_field(field_name="source_paper_id", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="citation_marker", datatype=DataType.VARCHAR, max_length=64, nullable=True)
    schema.add_field(field_name="target_title", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=128, nullable=True)
    schema.add_field(
        field_name="target_raw",
        datatype=DataType.VARCHAR,
        max_length=settings.max_text_length,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
        nullable=True,
    )
    schema.add_field(
        field_name="metadata_json",
        datatype=DataType.VARCHAR,
        max_length=settings.max_metadata_length,
        nullable=True,
    )
    schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=settings.text_dim)
    return schema


def build_citation_edge_index_params(client: MilvusClient, settings: MilvusSettings):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="text_dense",
        index_name="citation_edge_text_dense_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    return index_params
