from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .cleaner import heading_text, is_noise_text, markdown_heading_level, normalize_text
from .models import PageData, TextChunk


TEXT_CATEGORIES = {"Title", "Section-header", "Text", "List-item", "Formula", "Table", "Caption", "Footnote"}


def page_text_from_cells(page: PageData) -> str:
    parts: list[str] = []
    for cell in page.cells:
        if cell.category not in TEXT_CATEGORIES:
            continue
        text = normalize_text(cell.text)
        if is_noise_text(text):
            continue
        parts.append(text)
    return normalize_text("\n\n".join(parts))


def split_text_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if split_at > start + max_chars // 2:
                end = split_at + 1
        chunk = normalize_text(text[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def build_chunks(
    pages: list[PageData],
    paper_metadata: dict[str, Any],
    max_chars: int = 1800,
    overlap_chars: int = 180,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    title_stack: dict[int, str] = {}

    for page in pages:
        page_parts: list[str] = []
        for cell in page.cells:
            text = normalize_text(cell.text)
            if is_noise_text(text):
                continue
            level = markdown_heading_level(text)
            if cell.category in {"Title", "Section-header"} and level:
                title_stack[level] = heading_text(text)
                for old_level in list(title_stack):
                    if old_level > level:
                        title_stack.pop(old_level, None)
            if cell.category in TEXT_CATEGORIES:
                page_parts.append(text)

        page_text = normalize_text("\n\n".join(page_parts)) or page_text_from_cells(page)
        title_path = [title_stack[level] for level in sorted(title_stack)]
        for chunk_index, chunk_text in enumerate(split_text_with_overlap(page_text, max_chars, overlap_chars)):
            digest = hashlib.md5(
                f"{page.topic}:{page.paper_id}:{page.page_no}:{chunk_index}:{chunk_text[:128]}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                TextChunk(
                    chunk_id=f"{page.topic}:{page.paper_id}:p{page.page_no}:c{chunk_index}:{digest[:12]}",
                    topic=page.topic,
                    paper_id=page.paper_id,
                    page_start=page.page_no,
                    page_end=page.page_no,
                    title_path=title_path,
                    text=chunk_text,
                    metadata={
                        "paper": paper_metadata,
                        "source_pdf": str(page.pdf_path),
                        "page_json": str(page.page_json_path),
                        "page_image": str(page.page_image_path) if page.page_image_path else None,
                    },
                )
            )
    return chunks


def write_chunks(path: Path, chunks: list[TextChunk]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for chunk in chunks:
            writer.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")

