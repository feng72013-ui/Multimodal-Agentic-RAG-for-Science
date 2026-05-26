# Milvus DB Utilities

This directory contains standalone Milvus scripts for `recommendate_project`.
The connection defaults match the reference project:

```bash
export MILVUS_URI="http://172.16.229.202:19530"
export MILVUS_USER="root"
export MILVUS_PASSWORD="Milvus"
export MILVUS_DB_NAME="recommendation_system"
export MILVUS_COLLECTION="recsys_paper_rag"
```

The reference project code lives at:

```text
/home/lf/mount/LLM/project/test_project/MutilModel_RAG/milvus_db
```

The directly adapted script is `reference_compatible.py`. The project-native schema used by
`vector_ingest_pipeline` is in `schema.py`.

## 1. Connection Test

```bash
cd /home/lf/mount/LLM/project/recommendate_project
python3 -m milvus_db.client --ping
```

If this fails with `server unavailable`, Milvus is not reachable from this machine or the URI is wrong.
This must work before collection creation or ingestion can work.

## 2. Database Management

```bash
python3 -m milvus_db.databases list
python3 -m milvus_db.databases create --name recsys_rag
python3 -m milvus_db.databases describe --name recsys_rag
python3 -m milvus_db.databases drop --name recsys_rag --yes
```

The default database in this project is:

```bash
recommendation_system
```

To use a different database:

```bash
export MILVUS_DB_NAME="recsys_rag"
```

## 3. User And Role Management

```bash
python3 -m milvus_db.users list-users
python3 -m milvus_db.users create-user --name rag_user --new-password "change-me"
python3 -m milvus_db.users describe-user --name rag_user
python3 -m milvus_db.users create-role --name rag_role
python3 -m milvus_db.users grant-role --user-name rag_user --role-name rag_role
python3 -m milvus_db.users revoke-role --user-name rag_user --role-name rag_role
python3 -m milvus_db.users drop-user --name rag_user --yes
```

## 4. Collections, Schema, And Fields

Create the current recommendation-paper RAG collection:

```bash
python3 -m milvus_db.collections create --schema recsys
```

Recreate it from scratch:

```bash
python3 -m milvus_db.collections create --schema recsys --drop-existing
```

Inspect collections:

```bash
python3 -m milvus_db.collections list
python3 -m milvus_db.collections describe
python3 -m milvus_db.collections stats
```

Create a collection compatible with the old `MutilModel_RAG` schema:

```bash
python3 -m milvus_db.collections create --schema reference --collection-name lf_rag_test
```

The current `recsys` schema contains:

```text
id, doc_id, category, topic, paper_id, title, text, filename, filetype,
image_path, page_start, page_end, metadata_json, sparse, text_dense,
multimodal_dense
```

`text` is converted to `sparse` by the ingest pipeline's client-side BM25 model.
`text_dense` is 1024-dimensional and `multimodal_dense` is 2560-dimensional by default.

## 5. Data Operations

Insert a tiny sample record:

```bash
python3 -m milvus_db.data_ops insert-sample
```

Query records:

```bash
python3 -m milvus_db.data_ops query --filter 'doc_id == "milvus_db_sample"'
```

Search text through BM25:

```bash
python3 -m milvus_db.data_ops search-text --query "graph transformer recommendation" --limit 5
```

Delete test data:

```bash
python3 -m milvus_db.data_ops delete --filter 'doc_id == "milvus_db_sample"' --yes
```

Insert a JSONL file where each line already matches the collection schema:

```bash
python3 -m milvus_db.data_ops insert-jsonl --path ./records.jsonl
```

## 6. Attu Access

See `attu_access.md`.
