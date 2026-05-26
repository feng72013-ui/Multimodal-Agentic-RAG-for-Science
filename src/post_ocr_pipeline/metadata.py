from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_topic_metadata(papers_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load metadata.jsonl records keyed by (topic, pdf filename)."""
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for metadata_path in sorted(papers_root.glob("*/metadata.jsonl")):
        topic = metadata_path.parent.name
        with metadata_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                filename = record.get("file")
                if filename:
                    metadata[(topic, filename)] = record
    return metadata


def compact_paper_metadata(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    keep = [
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "arxiv_id",
        "abstract",
        "keywords",
        "citation_count",
        "topics",
        "file",
        "pdf_url",
        "landing_page_url",
    ]
    return {key: record.get(key) for key in keep if key in record}

