from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PageCell:
    category: str
    text: str
    bbox: list[int] | None = None
    order: int = 0


@dataclass
class PageData:
    topic: str
    paper_id: str
    page_no: int
    pdf_path: Path
    page_json_path: Path
    page_image_path: Path | None
    md_path: Path | None
    cells: list[PageCell] = field(default_factory=list)


@dataclass
class ImageAsset:
    image_id: str
    topic: str
    paper_id: str
    page_no: int
    source_type: str
    image_path: str
    bbox: list[int] | None
    category: str
    caption: str
    context_before: str
    context_after: str
    label: str = ""
    reference_texts: list[str] = field(default_factory=list)
    grounding_notes: str = ""
    description: str = ""


@dataclass
class TextChunk:
    chunk_id: str
    topic: str
    paper_id: str
    page_start: int
    page_end: int
    title_path: list[str]
    text: str
    metadata: dict[str, Any]
