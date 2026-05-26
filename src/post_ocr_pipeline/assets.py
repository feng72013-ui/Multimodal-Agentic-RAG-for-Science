from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from PIL import Image

from .cleaner import BASE64_IMAGE_RE, normalize_text
from .models import ImageAsset, PageCell, PageData


VISUAL_CATEGORIES = {"Picture", "Table"}


def _safe_ext(ext: str) -> str:
    ext = ext.lower().split(";")[0].strip()
    if ext in {"jpg", "jpeg", "png", "webp"}:
        return "jpg" if ext == "jpeg" else ext
    return "png"


def _context_from_cells(cells: list[PageCell], order: int, window: int = 2) -> tuple[str, str]:
    text_cells = [cell for cell in cells if cell.text and cell.category not in {"Page-header", "Page-footer"}]
    before = [cell.text for cell in text_cells if cell.order < order][-window:]
    after = [cell.text for cell in text_cells if cell.order > order][:window]
    return normalize_text("\n".join(before))[:1200], normalize_text("\n".join(after))[:1200]


def extract_base64_assets(page: PageData, asset_root: Path) -> list[ImageAsset]:
    if not page.md_path or not page.md_path.exists():
        return []
    markdown = page.md_path.read_text(encoding="utf-8")
    assets: list[ImageAsset] = []
    page_asset_dir = asset_root / page.topic / page.paper_id
    page_asset_dir.mkdir(parents=True, exist_ok=True)

    for idx, match in enumerate(BASE64_IMAGE_RE.finditer(markdown), start=1):
        raw = match.group("data")
        ext = _safe_ext(match.group("ext"))
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        image_id = f"{page.topic}:{page.paper_id}:p{page.page_no}:base64:{idx}:{digest[:12]}"
        image_path = page_asset_dir / f"{digest}.{ext}"
        if not image_path.exists():
            image_path.write_bytes(base64.b64decode(raw))
        before, after = _context_from_cells(page.cells, order=10_000)
        assets.append(
            ImageAsset(
                image_id=image_id,
                topic=page.topic,
                paper_id=page.paper_id,
                page_no=page.page_no,
                source_type="markdown_base64",
                image_path=str(image_path),
                bbox=None,
                category="Picture",
                caption="",
                context_before=before,
                context_after=after,
            )
        )
    return assets


def crop_visual_assets(page: PageData, asset_root: Path) -> list[ImageAsset]:
    if not page.page_image_path or not page.page_image_path.exists():
        return []
    visual_cells = [
        cell for cell in page.cells if cell.category in VISUAL_CATEGORIES and cell.bbox and len(cell.bbox) == 4
    ]
    if not visual_cells:
        return []

    page_asset_dir = asset_root / page.topic / page.paper_id
    page_asset_dir.mkdir(parents=True, exist_ok=True)
    assets: list[ImageAsset] = []

    with Image.open(page.page_image_path) as image:
        width, height = image.size
        for idx, cell in enumerate(visual_cells, start=1):
            x1, y1, x2, y2 = cell.bbox or [0, 0, width, height]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            bbox_key = "_".join(str(v) for v in [x1, y1, x2, y2])
            digest = hashlib.md5(f"{page.paper_id}:{page.page_no}:{bbox_key}:{idx}".encode()).hexdigest()
            image_id = f"{page.topic}:{page.paper_id}:p{page.page_no}:crop:{idx}:{digest[:12]}"
            image_path = page_asset_dir / f"{page.paper_id}_page_{page.page_no}_{cell.category.lower()}_{idx}.png"
            if not image_path.exists():
                crop.save(image_path)
            before, after = _context_from_cells(page.cells, order=cell.order)
            assets.append(
                ImageAsset(
                    image_id=image_id,
                    topic=page.topic,
                    paper_id=page.paper_id,
                    page_no=page.page_no,
                    source_type="layout_crop",
                    image_path=str(image_path),
                    bbox=[x1, y1, x2, y2],
                    category=cell.category,
                    caption=normalize_text(cell.text),
                    context_before=before,
                    context_after=after,
                )
            )
    return assets


def write_assets(path: Path, assets: list[ImageAsset]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for asset in assets:
            writer.write(json.dumps(asset.__dict__, ensure_ascii=False) + "\n")

