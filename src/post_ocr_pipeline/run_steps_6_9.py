#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import crop_visual_assets, extract_base64_assets, write_assets
from .chunker import build_chunks, write_chunks
from .describer import describe_assets, write_descriptions
from .grounding import enrich_assets_with_grounding
from .metadata import compact_paper_metadata, load_topic_metadata
from .models import PageCell, PageData


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OCR_ROOT = DATA_ROOT / "processed_ocr"
DEFAULT_PAPERS_ROOT = DATA_ROOT / "papers"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "processed_rag"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_cells(path: Path) -> list[PageCell]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, str):
        return [PageCell(category="Text", text=raw, order=0)]
    cells: list[PageCell] = []
    for order, item in enumerate(raw if isinstance(raw, list) else []):
        cells.append(
            PageCell(
                category=str(item.get("category", "Text")),
                text=str(item.get("text", "") or ""),
                bbox=item.get("bbox"),
                order=order,
            )
        )
    return cells


def discover_paper_jsonls(ocr_root: Path, only_topic: str | None = None) -> list[Path]:
    paths = []
    for path in sorted(ocr_root.glob("*/*.jsonl")):
        if path.name == "ocr_manifest.jsonl":
            continue
        if only_topic and path.parent.name != only_topic:
            continue
        paths.append(path)
    return paths


def page_data_from_paper_jsonl(path: Path) -> list[PageData]:
    topic = path.parent.name
    paper_id = path.stem
    pages: list[PageData] = []
    for record in load_jsonl(path):
        layout_info_path = Path(record.get("layout_info_path", ""))
        page_image = record.get("layout_image_path")
        md_path = record.get("md_content_nohf_path") or record.get("md_content_path")
        page = PageData(
            topic=topic,
            paper_id=paper_id,
            page_no=int(record.get("page_no", 0)),
            pdf_path=Path(record.get("file_path", "")),
            page_json_path=layout_info_path,
            page_image_path=Path(page_image) if page_image else None,
            md_path=Path(md_path) if md_path else None,
            cells=load_cells(layout_info_path),
        )
        pages.append(page)
    return sorted(pages, key=lambda p: p.page_no)


def write_clean_pages(path: Path, pages: list[PageData]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for page in pages:
            text = "\n\n".join(cell.text.strip() for cell in page.cells if cell.text.strip())
            writer.write(
                json.dumps(
                    {
                        "topic": page.topic,
                        "paper_id": page.paper_id,
                        "page_no": page.page_no,
                        "pdf_path": str(page.pdf_path),
                        "page_json_path": str(page.page_json_path),
                        "page_image_path": str(page.page_image_path) if page.page_image_path else None,
                        "text": text,
                        "cell_count": len(page.cells),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-OCR RAG preparation steps 6-9.")
    parser.add_argument("--ocr-root", type=Path, default=DEFAULT_OCR_ROOT)
    parser.add_argument("--papers-root", type=Path, default=DEFAULT_PAPERS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only-topic", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=180)
    parser.add_argument("--use-model-descriptions", action="store_true")
    parser.add_argument(
        "--description-provider",
        choices=["heuristic", "zhipu", "openai-compatible"],
        default="heuristic",
        help="Provider used when --use-model-descriptions is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    asset_root = args.output_root / "assets" / "images"
    asset_root.mkdir(parents=True, exist_ok=True)

    paper_jsonls = discover_paper_jsonls(args.ocr_root, args.only_topic)
    if args.limit is not None:
        paper_jsonls = paper_jsonls[: args.limit]

    metadata_by_file = load_topic_metadata(args.papers_root)
    all_pages: list[PageData] = []
    all_assets = []
    all_chunks = []

    for index, paper_jsonl in enumerate(paper_jsonls, start=1):
        pages = page_data_from_paper_jsonl(paper_jsonl)
        if not pages:
            continue
        pdf_name = pages[0].pdf_path.name
        paper_metadata = compact_paper_metadata(metadata_by_file.get((pages[0].topic, pdf_name)))
        chunks = build_chunks(
            pages,
            paper_metadata=paper_metadata,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        assets = []
        for page in pages:
            assets.extend(extract_base64_assets(page, asset_root))
            assets.extend(crop_visual_assets(page, asset_root))
        enrich_assets_with_grounding(assets, pages)
        describe_assets(
            assets,
            use_model=args.use_model_descriptions,
            provider=args.description_provider,
        )

        all_pages.extend(pages)
        all_chunks.extend(chunks)
        all_assets.extend(assets)
        print(
            f"[{index}/{len(paper_jsonls)}] {pages[0].topic}/{pages[0].paper_id}: "
            f"pages={len(pages)}, chunks={len(chunks)}, images={len(assets)}"
        )

    clean_pages_path = args.output_root / "clean_pages.jsonl"
    chunks_path = args.output_root / "chunks.jsonl"
    assets_path = args.output_root / "image_assets.jsonl"
    descriptions_path = args.output_root / "image_descriptions.jsonl"
    summary_path = args.output_root / "summary.json"

    write_clean_pages(clean_pages_path, all_pages)
    write_chunks(chunks_path, all_chunks)
    write_assets(assets_path, all_assets)
    write_descriptions(descriptions_path, all_assets)
    summary_path.write_text(
        json.dumps(
            {
                "paper_count": len(paper_jsonls),
                "page_count": len(all_pages),
                "chunk_count": len(all_chunks),
                "image_asset_count": len(all_assets),
                "output_files": {
                    "clean_pages": str(clean_pages_path),
                    "chunks": str(chunks_path),
                    "image_assets": str(assets_path),
                    "image_descriptions": str(descriptions_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
