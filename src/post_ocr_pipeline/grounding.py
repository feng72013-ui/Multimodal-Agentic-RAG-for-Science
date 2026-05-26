from __future__ import annotations

import re
from collections import defaultdict

from .cleaner import normalize_text
from .models import ImageAsset, PageCell, PageData


FIGURE_RE = re.compile(
    r"\b(?:Fig(?:ure)?\.?\s*|FIGURE\s+)(?P<num>\d+(?:[.\-][A-Za-z0-9]+)?[A-Za-z]?)",
    re.IGNORECASE,
)
TABLE_RE = re.compile(
    r"\b(?:Table|TABLE)\s+(?P<num>\d+(?:[.\-][A-Za-z0-9]+)?[A-Za-z]?)",
    re.IGNORECASE,
)


def canonical_label(kind: str, number: str) -> str:
    kind = "Table" if kind.lower().startswith("table") else "Figure"
    return f"{kind} {number}"


def labels_in_text(text: str) -> list[str]:
    labels: list[str] = []
    for match in FIGURE_RE.finditer(text):
        labels.append(canonical_label("Figure", match.group("num")))
    for match in TABLE_RE.finditer(text):
        labels.append(canonical_label("Table", match.group("num")))
    return labels


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _candidate_caption_cells(page: PageData, asset: ImageAsset) -> list[tuple[int, PageCell]]:
    candidates: list[tuple[int, PageCell]] = []
    asset_mid_y = None
    if asset.bbox:
        asset_mid_y = (asset.bbox[1] + asset.bbox[3]) / 2
    for cell in page.cells:
        text = normalize_text(cell.text)
        if not text:
            continue
        has_label = bool(labels_in_text(text))
        looks_caption = cell.category in {"Caption", "Text", "List-item"} and has_label
        if not looks_caption:
            continue
        distance = abs(cell.order)
        if asset_mid_y is not None and cell.bbox and len(cell.bbox) == 4:
            cell_mid_y = (cell.bbox[1] + cell.bbox[3]) / 2
            distance = int(abs(cell_mid_y - asset_mid_y))
        candidates.append((distance, cell))
    return sorted(candidates, key=lambda item: item[0])


def _reference_index(pages: list[PageData]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        for cell in page.cells:
            for sentence in split_sentences(cell.text):
                for label in labels_in_text(sentence):
                    entry = f"page {page.page_no}: {sentence}"
                    if entry not in refs[label]:
                        refs[label].append(entry)
    return refs


def _page_by_key(pages: list[PageData]) -> dict[tuple[str, str, int], PageData]:
    return {(page.topic, page.paper_id, page.page_no): page for page in pages}


def enrich_assets_with_grounding(assets: list[ImageAsset], pages: list[PageData], max_refs: int = 6) -> list[ImageAsset]:
    """Attach figure/table labels, captions, and paper-level reference sentences to assets."""
    refs = _reference_index(pages)
    page_lookup = _page_by_key(pages)

    for asset in assets:
        page = page_lookup.get((asset.topic, asset.paper_id, asset.page_no))
        caption_candidates = _candidate_caption_cells(page, asset) if page else []

        label = ""
        caption = normalize_text(asset.caption)
        for _, cell in caption_candidates:
            cell_text = normalize_text(cell.text)
            labels = labels_in_text(cell_text)
            if labels:
                label = labels[0]
                if not caption or len(cell_text) > len(caption):
                    caption = cell_text
                break

        if not label:
            labels = labels_in_text(caption)
            if labels:
                label = labels[0]

        if not label and asset.category == "Table":
            # Tables often carry their content in the layout cell but no explicit caption.
            table_refs = [(key, value) for key, value in refs.items() if key.startswith("Table ")]
            same_page_refs = [
                key
                for key, values in table_refs
                if any(value.startswith(f"page {asset.page_no}:") for value in values)
            ]
            if same_page_refs:
                label = same_page_refs[0]

        asset.label = label
        asset.caption = caption
        if label:
            asset.reference_texts = refs.get(label, [])[:max_refs]
            asset.grounding_notes = "matched_by_caption_or_reference"
        else:
            asset.reference_texts = []
            asset.grounding_notes = "no_explicit_figure_or_table_label_found"

    return assets
