from __future__ import annotations

from pymilvus import DataType, Function, FunctionType

from .client import create_client
from .config import MilvusSettings


COLLECTION_NAME = "lf_rag_test"
CONTEXT_COLLECTION_NAME = "lf_rag_test_context"
SPARSE_METRIC_TYPE = MilvusSettings.sparse_metric_type

client = create_client(MilvusSettings(collection_name=COLLECTION_NAME))


def create_db_collection() -> None:
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
    schema.add_function(
        Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=SPARSE_METRIC_TYPE,
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    index_params.add_index(
        field_name="dense",
        index_name="dense_inverted_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


def create_store_collection() -> None:
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(
        field_name="context_text",
        datatype=DataType.VARCHAR,
        max_length=6000,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
    )
    schema.add_field(field_name="user", datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name="timestamp", datatype=DataType.INT64, nullable=True)
    schema.add_field(field_name="message_type", datatype=DataType.VARCHAR, max_length=100, nullable=True)
    schema.add_field(field_name="evaluation_score", datatype=DataType.FLOAT, nullable=True)
    schema.add_field(field_name="context_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="context_dense", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_function(
        Function(
            name="text_bm25_emb",
            input_field_names=["context_text"],
            output_field_names=["context_sparse"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="context_sparse",
        index_name="context_sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=SPARSE_METRIC_TYPE,
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    index_params.add_index(
        field_name="context_dense",
        index_name="context_dense_inverted_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )

    client.create_collection(
        collection_name=CONTEXT_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


if __name__ == "__main__":
    create_store_collection()
    print(client.describe_collection(collection_name=CONTEXT_COLLECTION_NAME))
