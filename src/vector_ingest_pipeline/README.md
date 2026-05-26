# Vector Ingest Pipeline: Steps 10-12

This directory vectorizes `data/processed_rag/` and inserts the records into Milvus.

It follows the reference project at `/home/lf/mount/LLM/project/test_project/MutilModel_RAG`:

- `text` is converted to a BM25 sparse vector by the ingest pipeline.
- Dense vectors are inserted explicitly.
- Text and image/table records share one collection.

This project uses two dense vector fields:

- `text_dense`: local `/home/lf/mount/LLM/model/Qwen3-Embedding-0.6B`
- `multimodal_dense`: DashScope `qwen3-vl-embedding` (2560 dimensions by default)

Multimodal vectors are checkpointed locally so interrupted runs can resume:

- text cache: `data/processed_rag/text_embedding_cache.jsonl`
- cache file: `data/processed_rag/multimodal_embedding_cache.jsonl`
- BM25 model: `data/processed_rag/bm25_sparse_model.json`
- text cache key: `text model path + dim + text`
- multimodal cache key: `model + dim + image_path + text`
- both text and multimodal embeddings are reused from cache on rerun

Run a dry-run first:

```bash
cd /home/lf/mount/LLM/project/recommendate_project
PYTHONPATH=src python3 -m vector_ingest_pipeline.ingest --dry-run --limit 3
```

Create the collection only:

```bash
PYTHONPATH=src python3 -m vector_ingest_pipeline.ingest \
  --create-collection \
  --skip-insert
```

Create and insert a small sample:

```bash
export DASHSCOPE_API_KEY="your-api-key"
PYTHONPATH=src python3 -m vector_ingest_pipeline.ingest \
  --create-collection \
  --limit 20
```

Full ingestion:

```bash
export DASHSCOPE_API_KEY="your-api-key"
PYTHONPATH=src python3 -m vector_ingest_pipeline.ingest --create-collection
```

Resume an interrupted multimodal embedding run:

```bash
export DASHSCOPE_API_KEY="your-api-key"
PYTHONPATH=src python3 -m vector_ingest_pipeline.ingest
```

Force a clean rebuild of the multimodal cache:

```bash
export DASHSCOPE_API_KEY="your-api-key"
PYTHONPATH=src python3 -m vector_ingest_pipeline.ingest \
  --reset-multimodal-cache
```

Force a clean rebuild of the text cache:

```bash
PYTHONPATH=src python3 -m vector_ingest_pipeline.ingest \
  --reset-text-cache
```

Milvus connection defaults mirror the reference project:

```text
MILVUS_URI=http://172.16.229.202:19530
MILVUS_USER=root
MILVUS_PASSWORD=Milvus
MILVUS_DB_NAME=recommendation_system
collection=recsys_paper_rag
```

Override them with environment variables when needed.
