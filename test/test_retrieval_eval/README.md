# Retrieval Evaluation

This directory validates the quality of the Milvus knowledge base after OCR, chunking,
embedding, and ingestion.

It supports four retrieval modes:

- `sparse`: local BM25 query vector against the Milvus `sparse` field.
- `text_dense`: local `/home/lf/mount/LLM/model/Qwen3-Embedding-0.6B` query vector against `text_dense`.
- `multimodal_dense`: DashScope `qwen3-vl-embedding` text query against `multimodal_dense`.
- `hybrid`: reciprocal-rank fusion of `sparse` and `text_dense`.

Run a dry-run first:

```bash
cd /home/lf/mount/LLM/project/recommendate_project
python3 -m test_retrieval_eval.evaluate --dry-run
```

Run sparse retrieval only:

```bash
python3 -m test_retrieval_eval.evaluate \
  --modes sparse \
  --top-k 5
```

Run sparse, dense, and hybrid retrieval:

```bash
python3 -m test_retrieval_eval.evaluate \
  --modes sparse text_dense hybrid \
  --top-k 5
```

Run multimodal text-to-image/table retrieval:

```bash
export DASHSCOPE_API_KEY="your-api-key"
python3 -m test_retrieval_eval.evaluate \
  --modes multimodal_dense \
  --top-k 5
```

Outputs are written to:

```text
test_retrieval_eval/results/
```

The generated files include:

- `retrieval_results_*.jsonl`: full query-by-query search results.
- `summary_*.md`: compact report for quick inspection.
- `sample_rows_*.json`: sample records from Milvus for metadata sanity checks.
