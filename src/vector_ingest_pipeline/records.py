from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import IngestConfig
from .utils import iter_jsonl, truncate


def _title_from_chunk(record: dict[str, Any]) -> str:
    title_path = record.get("title_path") or []
    if title_path:
        return " --> ".join(str(part) for part in title_path if part)
    paper = (record.get("metadata") or {}).get("paper") or {}
    return paper.get("title") or record.get("paper_id") or ""


def _metadata_json(payload: dict[str, Any], max_len: int) -> str:
    return truncate(json.dumps(payload, ensure_ascii=False), max_len)


def build_text_records(config: IngestConfig) -> list[dict[str, Any]]:
    path = config.processed_rag_root / "chunks.jsonl"
    records: list[dict[str, Any]] = []
    for item in iter_jsonl(path):
        metadata = item.get("metadata") or {}
        source_pdf = metadata.get("source_pdf") or ""
        records.append(
            {
                "doc_id": item["chunk_id"],
                "category": "text",
                "topic": item.get("topic", ""),
                "paper_id": item.get("paper_id", ""),
                "title": truncate(_title_from_chunk(item), 1000),
                "text": truncate(item.get("text", ""), config.max_text_length),
                "filename": truncate(source_pdf, 1000),
                "filetype": ".pdf",
                "image_path": "",
                "page_start": int(item.get("page_start", 0)),
                "page_end": int(item.get("page_end", item.get("page_start", 0))),
                "metadata_json": _metadata_json(metadata, config.max_metadata_length),
            }
        )
    return records


def _image_text(item: dict[str, Any], config: IngestConfig) -> str:
    parts = [
        item.get("label", ""),
        item.get("caption", ""),
        item.get("description", ""),
        "\n".join(item.get("reference_texts") or []),
        item.get("context_before", ""),
        item.get("context_after", ""),
    ]
    return truncate("\n\n".join(part for part in parts if part), config.max_text_length)


def build_image_records(config: IngestConfig) -> list[dict[str, Any]]:
    path = config.processed_rag_root / "image_descriptions.jsonl"
    records: list[dict[str, Any]] = []
    for item in iter_jsonl(path):
        category = "table" if item.get("category") == "Table" else "image"
        records.append(
            {
                "doc_id": item["image_id"],
                "category": category,
                "topic": item.get("topic", ""),
                "paper_id": item.get("paper_id", ""),
                "title": truncate(item.get("label") or item.get("paper_id", ""), 1000),
                "text": _image_text(item, config),
                "filename": truncate(item.get("image_path", ""), 1000),
                "filetype": Path(item.get("image_path", "")).suffix.lower(),
                "image_path": truncate(item.get("image_path", ""), 1000),
                "page_start": int(item.get("page_no", 0)),
                "page_end": int(item.get("page_no", 0)),
                "metadata_json": _metadata_json(item, config.max_metadata_length),
            }
        )
    return records


def build_ingest_records(config: IngestConfig, include_images: bool = True) -> list[dict[str, Any]]:
    records = build_text_records(config)
    if include_images:
        records.extend(build_image_records(config))
    return records

