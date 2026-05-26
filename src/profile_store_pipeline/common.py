from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from vector_ingest_pipeline.utils import truncate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_RAG_ROOT = PROJECT_ROOT / "data" / "processed_rag"
DEFAULT_PROFILE_PATH = DEFAULT_PROCESSED_RAG_ROOT / "paper_profiles.jsonl"
DEFAULT_CITATION_PATH = DEFAULT_PROCESSED_RAG_ROOT / "citation_edges.jsonl"
DEFAULT_CHUNKS_PATH = DEFAULT_PROCESSED_RAG_ROOT / "chunks.jsonl"
DEFAULT_REVIEW_PATH = DEFAULT_PROCESSED_RAG_ROOT / "paper_profile_quality_reviews.jsonl"

PAPER_PROFILE_COLLECTION = "paper_profiles"
CITATION_EDGE_COLLECTION = "citation_edges"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def paper_key(row: dict[str, Any]) -> str:
    return f"{row.get('topic') or ''}::{row.get('paper_id') or row.get('title') or ''}"


def group_chunks_by_paper(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[paper_key(chunk)].append(chunk)
    return grouped


def profile_text(profile: dict[str, Any], max_bytes: int = 6000) -> str:
    parts = [
        f"Title: {profile.get('title') or ''}",
        f"Abstract: {profile.get('abstract') or ''}",
        "Research problem: " + " ".join(profile.get("research_problem") or []),
        "Methods: " + " ".join(profile.get("methods") or []),
        "Datasets: " + ", ".join(profile.get("datasets") or []),
        "Metrics: " + ", ".join(profile.get("metrics") or []),
        "Results: " + " ".join(profile.get("results") or []),
        "Conclusions: " + " ".join(profile.get("conclusions") or []),
        "Limitations: " + " ".join(profile.get("limitations") or []),
    ]
    return truncate("\n".join(part for part in parts if part.strip()), max_bytes)


def citation_key(row: dict[str, Any]) -> str:
    pieces = [
        row.get("source_topic") or "",
        row.get("source_paper_id") or "",
        row.get("citation_marker") or "",
        row.get("target_title") or row.get("target_raw") or "",
    ]
    return "::".join(pieces)


def compact_json(value: dict[str, Any], max_bytes: int) -> str:
    return truncate(json.dumps(value, ensure_ascii=False, separators=(",", ":")), max_bytes)
