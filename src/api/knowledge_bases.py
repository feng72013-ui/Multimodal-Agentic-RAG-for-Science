from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from .schemas import (
    KnowledgeBaseDocument,
    KnowledgeBaseInfo,
    KnowledgeBaseIngestRequest,
    KnowledgeBaseJobInfo,
)
from .services import PROJECT_ROOT


KB_ROOT = PROJECT_ROOT / "knowledge_bases"
REGISTRY_PATH = KB_ROOT / "registry.json"
UPLOAD_TOPIC = "uploads"

_jobs: dict[str, KnowledgeBaseJobInfo] = {}
_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fa5-]+", "_", value.strip())
    normalized = normalized.strip("_-").lower()
    return normalized[:48] or "knowledge_base"


def kb_dir(kb_id: str) -> Path:
    return KB_ROOT / kb_id


def collection_name_for_id(kb_id: str) -> str:
    return f"kb_{re.sub(r'[^a-zA-Z0-9_]', '_', kb_id)}"


def ensure_registry() -> None:
    KB_ROOT.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text("[]", encoding="utf-8")


def load_registry() -> list[KnowledgeBaseInfo]:
    ensure_registry()
    try:
      raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
      raw = []
    return [KnowledgeBaseInfo(**item) for item in raw if isinstance(item, dict)]


def save_registry(items: list[KnowledgeBaseInfo]) -> None:
    ensure_registry()
    REGISTRY_PATH.write_text(
        json.dumps([item.model_dump() for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_knowledge_bases() -> list[KnowledgeBaseInfo]:
    return load_registry()


def get_knowledge_base(kb_id: str) -> KnowledgeBaseInfo:
    for item in load_registry():
        if item.id == kb_id:
            return item
    raise HTTPException(status_code=404, detail="知识库不存在。")


def find_collection_for_kb(kb_id: str | None) -> str | None:
    if not kb_id:
        return None
    try:
        return get_knowledge_base(kb_id).collection_name
    except HTTPException:
        return None


def update_knowledge_base(updated: KnowledgeBaseInfo) -> KnowledgeBaseInfo:
    items = load_registry()
    next_items = []
    found = False
    for item in items:
        if item.id == updated.id:
            next_items.append(updated)
            found = True
        else:
            next_items.append(item)
    if not found:
        next_items.append(updated)
    save_registry(next_items)
    return updated


def create_knowledge_base(name: str, description: str = "") -> KnowledgeBaseInfo:
    base_id = f"{safe_slug(name)}_{uuid.uuid4().hex[:8]}"
    now = utc_now()
    item = KnowledgeBaseInfo(
        id=base_id,
        name=name.strip(),
        description=description.strip(),
        collection_name=collection_name_for_id(base_id),
        status="empty",
        created_at=now,
        updated_at=now,
        documents=[],
    )
    (kb_dir(base_id) / "papers" / UPLOAD_TOPIC).mkdir(parents=True, exist_ok=True)
    (kb_dir(base_id) / "processed_ocr").mkdir(parents=True, exist_ok=True)
    (kb_dir(base_id) / "processed_rag").mkdir(parents=True, exist_ok=True)
    items = load_registry()
    items.append(item)
    save_registry(items)
    return item


def upload_document(kb_id: str, file: UploadFile) -> KnowledgeBaseInfo:
    item = get_knowledge_base(kb_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="当前只支持上传 PDF 文件。")

    doc_id = uuid.uuid4().hex
    filename = Path(file.filename or f"{doc_id}.pdf").name
    stored_name = f"{doc_id}_{safe_slug(Path(filename).stem)}.pdf"
    stored_path = kb_dir(kb_id) / "papers" / UPLOAD_TOPIC / stored_name
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    with stored_path.open("wb") as writer:
        shutil.copyfileobj(file.file, writer)

    document = KnowledgeBaseDocument(
        id=doc_id,
        filename=filename,
        stored_path=str(stored_path),
        size_bytes=stored_path.stat().st_size,
        status="uploaded",
        created_at=utc_now(),
    )
    item.documents.append(document)
    item.document_count = len(item.documents)
    item.status = "uploaded"
    item.updated_at = utc_now()
    update_knowledge_base(item)
    return item


def start_ingest_job(kb_id: str, request: KnowledgeBaseIngestRequest) -> KnowledgeBaseJobInfo:
    item = get_knowledge_base(kb_id)
    if not item.documents:
        raise HTTPException(status_code=400, detail="请先上传至少一个 PDF。")

    job = KnowledgeBaseJobInfo(
        id=uuid.uuid4().hex,
        knowledge_base_id=kb_id,
        status="queued",
        current_step="等待执行",
        progress=0.0,
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    with _lock:
        _jobs[job.id] = job

    thread = threading.Thread(target=_run_ingest_job, args=(job.id, request), daemon=True)
    thread.start()
    return job


def get_job(job_id: str) -> KnowledgeBaseJobInfo:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或服务已重启。")
    return job


def _set_job(job_id: str, **changes: Any) -> None:
    with _lock:
        job = _jobs[job_id]
        data = job.model_dump()
        data.update(changes)
        data["updated_at"] = utc_now()
        _jobs[job_id] = KnowledgeBaseJobInfo(**data)


def _append_log(job_id: str, line: str) -> None:
    line = line.strip()
    if not line:
        return
    with _lock:
        job = _jobs[job_id]
        logs = [*job.logs, line][-300:]
        data = job.model_dump()
        data["logs"] = logs
        data["updated_at"] = utc_now()
        _jobs[job_id] = KnowledgeBaseJobInfo(**data)


def _run_ingest_job(job_id: str, request: KnowledgeBaseIngestRequest) -> None:
    job = get_job(job_id)
    kb_id = job.knowledge_base_id
    item = get_knowledge_base(kb_id)
    try:
        item.status = "processing"
        item.updated_at = utc_now()
        update_knowledge_base(item)

        root = kb_dir(kb_id)
        ocr_root = root / "processed_ocr"
        rag_root = root / "processed_rag"
        papers_root = root / "papers"

        _run_step(
            job_id,
            "OCR",
            0.1,
            0.42,
            [
                sys.executable,
                "-m",
                "ocr_dots_vllm_batch",
                "--papers-root",
                str(papers_root),
                "--output-root",
                str(ocr_root),
                "--only-topic",
                UPLOAD_TOPIC,
                "--skip-server-check",
                *(["--force"] if request.force_ocr else []),
            ],
        )
        _run_step(
            job_id,
            "清洗、图片抽取和切块",
            0.42,
            0.68,
            [
                sys.executable,
                "-m",
                "post_ocr_pipeline.run_steps_6_9",
                "--ocr-root",
                str(ocr_root),
                "--papers-root",
                str(papers_root),
                "--output-root",
                str(rag_root),
                "--only-topic",
                UPLOAD_TOPIC,
                *(["--use-model-descriptions"] if request.use_model_descriptions else []),
            ],
        )
        _run_step(
            job_id,
            "向量化并写入 Milvus",
            0.68,
            0.98,
            [
                sys.executable,
                "-m",
                "vector_ingest_pipeline.ingest",
                "--processed-rag-root",
                str(rag_root),
                "--collection-name",
                item.collection_name,
                "--text-cache-path",
                str(rag_root / "text_embedding_cache.jsonl"),
                "--multimodal-cache-path",
                str(rag_root / "multimodal_embedding_cache.jsonl"),
                "--bm25-model-path",
                str(rag_root / "bm25_sparse_model.json"),
                *(["--create-collection"] if request.create_collection else []),
                *(["--drop-existing"] if request.drop_existing else []),
            ],
        )

        item = refresh_counts(get_knowledge_base(kb_id))
        item.status = "ready"
        item.updated_at = utc_now()
        update_knowledge_base(item)
        _set_job(job_id, status="succeeded", current_step="完成", progress=1.0)
    except Exception as exc:
        try:
            item = get_knowledge_base(kb_id)
            item.status = "failed"
            item.updated_at = utc_now()
            update_knowledge_base(item)
        except Exception:
            pass
        _append_log(job_id, f"ERROR: {exc}")
        _set_job(job_id, status="failed", current_step="失败", error=str(exc))


def _run_step(job_id: str, name: str, start: float, end: float, command: list[str]) -> None:
    _set_job(job_id, status="running", current_step=name, progress=start)
    _append_log(job_id, f"$ {' '.join(command)}")
    env = dict(os.environ)
    pythonpath_parts = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    line_count = 0
    for line in process.stdout:
        line_count += 1
        _append_log(job_id, line)
        progress = min(end - 0.01, start + (end - start) * min(line_count / 80, 0.95))
        _set_job(job_id, progress=progress)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} 失败，退出码 {return_code}")
    _set_job(job_id, progress=end)


def refresh_counts(item: KnowledgeBaseInfo) -> KnowledgeBaseInfo:
    summary_path = kb_dir(item.id) / "processed_rag" / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            item.chunk_count = int(summary.get("chunk_count") or 0)
            item.image_asset_count = int(summary.get("image_asset_count") or 0)
        except Exception:
            pass
    item.document_count = len(item.documents)
    return item
