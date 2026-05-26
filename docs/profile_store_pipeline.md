# Profile Store Pipeline

This pipeline promotes Stage 2 offline outputs into queryable Milvus collections and adds an evidence-grounded LLM quality review step.

## Inputs

- `processed_rag/paper_profiles.jsonl`
- `processed_rag/citation_edges.jsonl`
- `processed_rag/chunks.jsonl`

## Step 1: Quality Review

Preview without LLM calls:

```bash
python3 -m profile_store_pipeline.quality_review \
  --limit 3 \
  --dry-run \
  --output /tmp/paper_profile_quality_reviews.preview.jsonl
```

Run LLM review and write the default output:

```bash
python3 -m profile_store_pipeline.quality_review \
  --sleep 0.5
```

Default output:

- `processed_rag/paper_profile_quality_reviews.jsonl`

Each row contains:

- `faithfulness_score`
- `field_accuracy`
- `missing_fields`
- `wrong_or_unsupported_claims`
- `citation_quality`
- `rewrite_suggestions`
- `pass`

## Step 2: Milvus Ingest

Preview records:

```bash
python3 -m profile_store_pipeline.ingest \
  --limit-profiles 2 \
  --limit-citations 3 \
  --dry-run
```

Create collections and insert all records:

```bash
python3 -m profile_store_pipeline.ingest \
  --create-collections
```

Recreate collections before insert:

```bash
python3 -m profile_store_pipeline.ingest \
  --create-collections \
  --drop-existing
```

Default collections:

- `paper_profiles`
- `citation_edges`

By default, records use zero vectors for `text_dense`, which is enough for scalar filtering and storage. Add `--use-embeddings` when semantic vector search over profile/citation text is needed.

