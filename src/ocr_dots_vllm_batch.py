#!/usr/bin/env python3
"""
Batch OCR runner for recommendation-system papers using DotsOCR through vLLM.

This script covers steps 1-5 of the OCR plan:
1. scan PDFs
2. prepare DotsOCR/vLLM settings
3. create per-topic output directories
4. run DotsOCR OCR/layout parsing
5. write validation reports
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_PAPERS_ROOT = DATA_ROOT / "papers"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "processed_ocr"
DEFAULT_DOTSOCR_MODEL_PATH = Path(
    "/home/lf/mount/LLM/project/test_project/MutilModel_RAG/DotsOCR"
)
DEFAULT_DOTSOCR_CODE_ROOT = PROJECT_ROOT / "dots_ocr"


@dataclass
class PaperTask:
    index: int
    topic: str
    pdf_path: Path
    topic_output_dir: Path
    paper_output_dir: Path
    jsonl_path: Path


@dataclass
class OCRRecord:
    index: int
    topic: str
    pdf_path: str
    output_dir: str
    jsonl_path: str
    status: str
    md_count: int = 0
    json_count: int = 0
    image_count: int = 0
    page_count: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slug_from_pdf(pdf_path: Path) -> str:
    return pdf_path.stem


def scan_pdfs(papers_root: Path, only_topic: str | None = None) -> list[Path]:
    pdfs = sorted(papers_root.rglob("*.pdf"))
    if only_topic:
        pdfs = [p for p in pdfs if p.parent.name == only_topic]
    return pdfs


def build_tasks(pdfs: Iterable[Path], output_root: Path) -> list[PaperTask]:
    tasks: list[PaperTask] = []
    for index, pdf_path in enumerate(pdfs, start=1):
        topic = pdf_path.parent.name
        topic_output_dir = output_root / topic
        paper_output_dir = topic_output_dir / slug_from_pdf(pdf_path)
        jsonl_path = topic_output_dir / f"{slug_from_pdf(pdf_path)}.jsonl"
        tasks.append(
            PaperTask(
                index=index,
                topic=topic,
                pdf_path=pdf_path,
                topic_output_dir=topic_output_dir,
                paper_output_dir=paper_output_dir,
                jsonl_path=jsonl_path,
            )
        )
    return tasks


def ensure_dirs(tasks: Iterable[PaperTask], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task.topic_output_dir.mkdir(parents=True, exist_ok=True)


def count_outputs(paper_output_dir: Path, jsonl_path: Path) -> tuple[int, int, int, int]:
    md_count = len(list(paper_output_dir.glob("*.md"))) if paper_output_dir.exists() else 0
    json_count = len(list(paper_output_dir.glob("*.json"))) if paper_output_dir.exists() else 0
    image_count = len(list(paper_output_dir.glob("*.jpg"))) if paper_output_dir.exists() else 0
    page_count = 0
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as reader:
            page_count = sum(1 for line in reader if line.strip())
    return md_count, json_count, image_count, page_count


def has_successful_output(task: PaperTask) -> bool:
    md_count, _, _, page_count = count_outputs(task.paper_output_dir, task.jsonl_path)
    return md_count > 0 and page_count > 0


def write_jsonl(path: Path, records: Iterable[OCRRecord]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_summary(path: Path, records: list[OCRRecord], total_scanned: int) -> None:
    counts: dict[str, int] = {}
    by_topic: dict[str, dict[str, int]] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
        topic_counts = by_topic.setdefault(record.topic, {})
        topic_counts[record.status] = topic_counts.get(record.status, 0) + 1

    payload = {
        "generated_at": now_iso(),
        "total_scanned": total_scanned,
        "total_recorded": len(records),
        "status_counts": counts,
        "by_topic": by_topic,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def check_vllm_server(ip: str, port: int, timeout: float) -> tuple[bool, str]:
    url = f"http://{ip}:{port}/v1/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                return True, f"vLLM server is reachable: {url}"
            return False, f"vLLM server returned HTTP {response.status}: {url}"
    except urllib.error.URLError as exc:
        return False, f"Cannot reach vLLM server at {url}: {exc}"
    except TimeoutError as exc:
        return False, f"Timed out checking vLLM server at {url}: {exc}"


def import_dotsocr_parser(code_root: Path):
    if not code_root.exists():
        raise FileNotFoundError(f"DotsOCR code root does not exist: {code_root}")
    sys.path.insert(0, str(code_root))
    from dots_ocr.parser import do_parse

    return do_parse


def run_task(task: PaperTask, args: argparse.Namespace, do_parse) -> OCRRecord:
    started_at = now_iso()
    started = time.monotonic()
    record = OCRRecord(
        index=task.index,
        topic=task.topic,
        pdf_path=str(task.pdf_path),
        output_dir=str(task.paper_output_dir),
        jsonl_path=str(task.jsonl_path),
        status="running",
        started_at=started_at,
    )

    try:
        do_parse(
            input_path=str(task.pdf_path),
            output=str(task.topic_output_dir),
            prompt=args.prompt,
            ip=args.ip,
            port=args.port,
            model_name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            dpi=args.dpi,
            max_completion_tokens=args.max_completion_tokens,
            num_thread=args.num_thread,
            no_fitz_preprocess=args.no_fitz_preprocess,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            use_hf=False,
            model_path=str(args.model_path),
        )
        md_count, json_count, image_count, page_count = count_outputs(
            task.paper_output_dir, task.jsonl_path
        )
        record.md_count = md_count
        record.json_count = json_count
        record.image_count = image_count
        record.page_count = page_count
        record.status = "success" if md_count > 0 and page_count > 0 else "failed"
        if record.status == "failed":
            record.error = "OCR finished but no usable Markdown/JSONL output was found."
    except Exception as exc:
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        record.ended_at = now_iso()
        record.elapsed_seconds = round(time.monotonic() - started, 3)

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch OCR recommendation papers with DotsOCR over vLLM."
    )
    parser.add_argument("--papers-root", type=Path, default=DEFAULT_PAPERS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dotsocr-code-root", type=Path, default=DEFAULT_DOTSOCR_CODE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_DOTSOCR_MODEL_PATH)
    parser.add_argument("--ip", default="localhost")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model-name", default="dots_ocr")
    parser.add_argument("--prompt", default="prompt_layout_all_en")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--num-thread", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--only-topic", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-server-check", action="store_true")
    parser.add_argument("--server-check-timeout", type=float, default=3.0)
    parser.add_argument(
        "--no-fitz-preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Fitz preprocessing for image inputs. PDF page rendering still uses DotsOCR's PDF loader.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdfs = scan_pdfs(args.papers_root, args.only_topic)
    total_scanned = len(pdfs)
    if args.limit is not None:
        pdfs = pdfs[: args.limit]

    tasks = build_tasks(pdfs, args.output_root)
    ensure_dirs(tasks, args.output_root)

    manifest_path = args.output_root / "ocr_manifest.jsonl"
    summary_path = args.output_root / "ocr_summary.json"

    print(f"Scanned PDFs: {total_scanned}")
    print(f"Selected PDFs: {len(tasks)}")
    print(f"Output root: {args.output_root}")
    print(f"DotsOCR model path: {args.model_path}")
    print(f"vLLM endpoint: http://{args.ip}:{args.port}/v1")
    print(f"vLLM model name: {args.model_name}")

    records: list[OCRRecord] = []
    if args.dry_run:
        for task in tasks:
            records.append(
                OCRRecord(
                    index=task.index,
                    topic=task.topic,
                    pdf_path=str(task.pdf_path),
                    output_dir=str(task.paper_output_dir),
                    jsonl_path=str(task.jsonl_path),
                    status="planned",
                )
            )
        write_jsonl(manifest_path, records)
        write_summary(summary_path, records, total_scanned)
        print(f"Dry run complete. Manifest: {manifest_path}")
        print(f"Summary: {summary_path}")
        return 0

    if not args.skip_server_check:
        ok, message = check_vllm_server(args.ip, args.port, args.server_check_timeout)
        print(message)
        if not ok:
            print("Start the DotsOCR vLLM server first, or use --skip-server-check.")
            return 2

    do_parse = import_dotsocr_parser(args.dotsocr_code_root)

    for task in tasks:
        if has_successful_output(task) and not args.force:
            md_count, json_count, image_count, page_count = count_outputs(
                task.paper_output_dir, task.jsonl_path
            )
            print(f"[{task.index}/{len(tasks)}] skip existing: {task.pdf_path}")
            records.append(
                OCRRecord(
                    index=task.index,
                    topic=task.topic,
                    pdf_path=str(task.pdf_path),
                    output_dir=str(task.paper_output_dir),
                    jsonl_path=str(task.jsonl_path),
                    status="skipped",
                    md_count=md_count,
                    json_count=json_count,
                    image_count=image_count,
                    page_count=page_count,
                )
            )
            write_jsonl(manifest_path, records)
            write_summary(summary_path, records, total_scanned)
            continue

        print(f"[{task.index}/{len(tasks)}] OCR: {task.pdf_path}")
        record = run_task(task, args, do_parse)
        records.append(record)
        print(
            f"  -> {record.status}: pages={record.page_count}, "
            f"md={record.md_count}, json={record.json_count}, images={record.image_count}, "
            f"elapsed={record.elapsed_seconds}s"
        )
        if record.error:
            print(f"  -> error: {record.error.splitlines()[0]}")

        write_jsonl(manifest_path, records)
        write_summary(summary_path, records, total_scanned)

    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    return 0 if all(record.status != "failed" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
