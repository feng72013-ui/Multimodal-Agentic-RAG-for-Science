#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from vector_ingest_pipeline.utils import iter_jsonl, truncate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_RAG_ROOT = PROJECT_ROOT / "data" / "processed_rag"


@dataclass
class PaperBundle:
    topic: str
    paper_id: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


SECTION_RULES = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "背景", "引言"),
    "method": ("method", "methodology", "approach", "model", "framework", "proposed", "方法"),
    "experiment": ("experiment", "evaluation", "result", "analysis", "实验", "评估"),
    "conclusion": ("conclusion", "discussion", "结论"),
    "limitation": ("limitation", "future work", "threat", "局限", "未来"),
    "references": ("reference", "references", "bibliography"),
}

DATASET_TERMS = (
    "movielens",
    "amazon",
    "yelp",
    "gowalla",
    "lastfm",
    "steam",
    "beauty",
    "ml-1m",
    "ml-20m",
    "netflix",
    "taobao",
    "alibaba",
)

METRIC_TERMS = (
    "recall",
    "ndcg",
    "hit rate",
    "hit@",
    "hr@",
    "mrr",
    "auc",
    "map",
    "precision",
    "f1",
    "rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-level profiles from processed RAG JSONL files.")
    parser.add_argument("--processed-rag-root", type=Path, default=DEFAULT_PROCESSED_RAG_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--citation-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--limit-papers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed_root = args.processed_rag_root
    output_path = args.output or processed_root / "paper_profiles.jsonl"
    citation_path = args.citation_output or processed_root / "citation_edges.jsonl"
    summary_path = args.summary_output or processed_root / "paper_profile_summary.json"

    bundles = load_bundles(processed_root)
    if args.limit_papers is not None:
        bundles = bundles[: args.limit_papers]

    profiles = [build_profile(bundle) for bundle in bundles]
    citation_edges = [
        edge
        for bundle in bundles
        for edge in extract_citation_edges(bundle)
    ]
    summary = build_summary(profiles, citation_edges, processed_root)

    if args.dry_run:
        print(json.dumps({"summary": summary, "sample_profiles": profiles[:3]}, ensure_ascii=False, indent=2)[:8000])
        return 0

    write_jsonl(output_path, profiles)
    write_jsonl(citation_path, citation_edges)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Profiles: {output_path}")
    print(f"Citation edges: {citation_path}")
    print(f"Summary: {summary_path}")
    return 0


def load_bundles(processed_root: Path) -> list[PaperBundle]:
    chunks_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    images_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for chunk in iter_jsonl(processed_root / "chunks.jsonl"):
        key = (str(chunk.get("topic") or ""), str(chunk.get("paper_id") or ""))
        chunks_by_key[key].append(chunk)

    image_path = processed_root / "image_descriptions.jsonl"
    if image_path.exists():
        for image in iter_jsonl(image_path):
            key = (str(image.get("topic") or ""), str(image.get("paper_id") or ""))
            images_by_key[key].append(image)

    bundles = []
    for key in sorted(chunks_by_key):
        topic, paper_id = key
        bundles.append(
            PaperBundle(
                topic=topic,
                paper_id=paper_id,
                chunks=sorted(chunks_by_key[key], key=lambda item: (item.get("page_start", 0), item.get("chunk_id", ""))),
                images=sorted(images_by_key.get(key, []), key=lambda item: (item.get("page_no", 0), item.get("image_id", ""))),
            )
        )
    return bundles


def build_profile(bundle: PaperBundle) -> dict[str, Any]:
    paper_metadata = _paper_metadata(bundle.chunks)
    sections = classify_sections(bundle.chunks)
    title = paper_metadata.get("title") or _title_from_chunks(bundle.chunks) or bundle.paper_id
    abstract = _extract_abstract(bundle.chunks, sections)
    research_problem = _select_sentences(sections["abstract"] + sections["introduction"], PROBLEM_HINTS, limit=3)
    methods = _select_sentences(sections["method"] + sections["abstract"], METHOD_HINTS, limit=4)
    results = _select_sentences(sections["experiment"] + sections["abstract"], RESULT_HINTS, limit=4)
    conclusions = _select_sentences(sections["conclusion"] + sections["abstract"], CONCLUSION_HINTS, limit=3)
    if not conclusions:
        conclusions = _fallback_conclusions(bundle.chunks, abstract)
    limitations = _select_sentences(sections["limitation"] + sections["conclusion"], LIMITATION_HINTS, limit=3)
    datasets = _extract_terms(bundle.chunks, DATASET_TERMS)
    metrics = _extract_terms(bundle.chunks, METRIC_TERMS)
    key_visuals = _key_visuals(bundle.images)
    quality = _quality_assessment(
        abstract=abstract,
        research_problem=research_problem,
        methods=methods,
        results=results,
        conclusions=conclusions,
        datasets=datasets,
        metrics=metrics,
        key_visuals=key_visuals,
        chunks=bundle.chunks,
    )

    return {
        "topic": bundle.topic,
        "paper_id": bundle.paper_id,
        "title": title,
        "authors": paper_metadata.get("authors") or [],
        "year": paper_metadata.get("year"),
        "venue": paper_metadata.get("venue"),
        "doi": paper_metadata.get("doi"),
        "arxiv_id": paper_metadata.get("arxiv_id"),
        "source_pdf": _source_pdf(bundle.chunks),
        "abstract": abstract,
        "research_problem": research_problem,
        "methods": methods,
        "datasets": datasets,
        "metrics": metrics,
        "results": results,
        "conclusions": conclusions,
        "limitations": limitations,
        "key_visuals": key_visuals,
        "quality": quality,
        "coverage": {
            "chunk_count": len(bundle.chunks),
            "image_count": len(bundle.images),
            "page_start": min((int(chunk.get("page_start", 0)) for chunk in bundle.chunks), default=None),
            "page_end": max((int(chunk.get("page_end", chunk.get("page_start", 0))) for chunk in bundle.chunks), default=None),
            "sections_detected": sorted(name for name, values in sections.items() if values),
        },
    }


PROBLEM_HINTS = (
    "problem",
    "challenge",
    "limitation",
    "aim",
    "goal",
    "motivat",
    "address",
    "研究",
    "问题",
    "挑战",
)
METHOD_HINTS = (
    "propose",
    "present",
    "introduce",
    "framework",
    "method",
    "model",
    "module",
    "approach",
    "提出",
    "方法",
    "模型",
    "框架",
)
RESULT_HINTS = (
    "experiment",
    "result",
    "outperform",
    "improve",
    "achieve",
    "show",
    "demonstrate",
    "实验",
    "结果",
    "提升",
)
CONCLUSION_HINTS = (
    "conclusion",
    "conclude",
    "summary",
    "we presented",
    "we proposed",
    "结论",
    "总结",
)
LIMITATION_HINTS = (
    "limitation",
    "future work",
    "threat",
    "however",
    "although",
    "局限",
    "未来",
)


def classify_sections(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name in SECTION_RULES}
    sections["other"] = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        title_path = " ".join(str(part) for part in chunk.get("title_path") or [])
        haystack = f"{title_path}\n{text[:500]}".lower()
        matched = False
        for name, keywords in SECTION_RULES.items():
            if any(keyword in haystack for keyword in keywords):
                sections[name].append(text)
                matched = True
        if not matched:
            sections["other"].append(text)
    return sections


def extract_citation_edges(bundle: PaperBundle) -> list[dict[str, Any]]:
    sections = classify_sections(bundle.chunks)
    reference_text = "\n".join(sections["references"])
    if not reference_text:
        return []

    edges = []
    for marker, raw_reference in _reference_entries(reference_text):
        target_title = _guess_reference_title(raw_reference)
        if not target_title:
            continue
        edges.append(
            {
                "source_topic": bundle.topic,
                "source_paper_id": bundle.paper_id,
                "citation_marker": marker,
                "target_title": truncate(target_title, 500),
                "target_raw": truncate(raw_reference, 1500),
                "source": "references_section",
            }
        )
    return edges


def build_summary(profiles: list[dict[str, Any]], citation_edges: list[dict[str, Any]], processed_root: Path) -> dict[str, Any]:
    complete_fields = ("abstract", "research_problem", "methods", "results", "conclusions")
    field_fill_rates = {}
    for field_name in complete_fields:
        filled = sum(1 for profile in profiles if profile.get(field_name))
        field_fill_rates[field_name] = filled / len(profiles) if profiles else 0.0

    quality_scores = [float(profile["quality"]["score"]) for profile in profiles]
    return {
        "processed_rag_root": str(processed_root),
        "paper_count": len(profiles),
        "citation_edge_count": len(citation_edges),
        "field_fill_rates": field_fill_rates,
        "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        "quality_distribution": {
            "high": sum(1 for score in quality_scores if score >= 0.8),
            "medium": sum(1 for score in quality_scores if 0.5 <= score < 0.8),
            "low": sum(1 for score in quality_scores if score < 0.5),
        },
    }


def _paper_metadata(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        paper = metadata.get("paper") or {}
        if paper:
            return paper
    return {}


def _title_from_chunks(chunks: list[dict[str, Any]]) -> str:
    for chunk in chunks:
        title_path = chunk.get("title_path") or []
        if title_path:
            return str(title_path[0])
    return ""


def _source_pdf(chunks: list[dict[str, Any]]) -> str:
    for chunk in chunks:
        source_pdf = (chunk.get("metadata") or {}).get("source_pdf")
        if source_pdf:
            return str(source_pdf)
    return ""


def _extract_abstract(chunks: list[dict[str, Any]], sections: dict[str, list[str]]) -> str:
    candidates = sections["abstract"] or [chunk.get("text", "") for chunk in chunks[:3]]
    for text in candidates:
        match = re.search(r"(?is)\babstract\b\s*(.*?)(?:\n\s*##?\s*(?:1|introduction|ccs|keywords)|\n\s*#\s*1|\Z)", text)
        if match:
            return truncate(_clean_text(match.group(1)), 1800)
    return truncate(_clean_text(candidates[0] if candidates else ""), 1200)


def _select_sentences(text_blocks: Iterable[str], hints: tuple[str, ...], limit: int) -> list[str]:
    selected = []
    seen = set()
    for block in text_blocks:
        for sentence in _sentences(block):
            lowered = sentence.lower()
            if not any(hint in lowered for hint in hints):
                continue
            normalized = re.sub(r"\W+", " ", lowered).strip()
            if len(sentence) < 40 or normalized in seen:
                continue
            seen.add(normalized)
            selected.append(truncate(sentence, 500))
            if len(selected) >= limit:
                return selected
    return selected


def _fallback_conclusions(chunks: list[dict[str, Any]], abstract: str) -> list[str]:
    tail_blocks = [str(chunk.get("text") or "") for chunk in chunks[-5:]]
    tail_sentences = [
        sentence
        for block in tail_blocks
        for sentence in _sentences(block)
        if 60 <= len(sentence) <= 600 and "reference" not in sentence.lower()
    ]
    if tail_sentences:
        return [truncate(tail_sentences[-1], 500)]

    abstract_sentences = _sentences(abstract)
    if abstract_sentences:
        return [truncate(abstract_sentences[-1], 500)]
    return []


def _sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", cleaned) if part.strip()]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^#+\s*", "", text)
    return text


def _extract_terms(chunks: list[dict[str, Any]], terms: tuple[str, ...]) -> list[str]:
    haystack = "\n".join(str(chunk.get("text") or "") for chunk in chunks).lower()
    found = []
    for term in terms:
        if term.lower() in haystack:
            found.append(term)
    return sorted(set(found), key=str.lower)


def _key_visuals(images: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    visuals = []
    for image in images:
        caption = str(image.get("caption") or image.get("description") or "")
        if not caption and not image.get("image_path"):
            continue
        visuals.append(
            {
                "label": image.get("label") or "",
                "category": "table" if image.get("category") == "Table" else "image",
                "caption": truncate(caption, 500),
                "image_path": image.get("image_path") or "",
                "page_no": image.get("page_no"),
            }
        )
        if len(visuals) >= limit:
            break
    return visuals


def _quality_assessment(
    *,
    abstract: str,
    research_problem: list[str],
    methods: list[str],
    results: list[str],
    conclusions: list[str],
    datasets: list[str],
    metrics: list[str],
    key_visuals: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "has_abstract": bool(abstract),
        "has_research_problem": bool(research_problem),
        "has_methods": bool(methods),
        "has_results": bool(results),
        "has_conclusions": bool(conclusions),
        "has_datasets": bool(datasets),
        "has_metrics": bool(metrics),
        "has_visuals": bool(key_visuals),
        "has_enough_chunks": len(chunks) >= 5,
    }
    score = sum(1 for value in checks.values() if value) / len(checks)
    return {
        "score": round(score, 3),
        "level": "high" if score >= 0.8 else "medium" if score >= 0.5 else "low",
        "checks": checks,
    }


def _reference_entries(reference_text: str) -> list[tuple[str, str]]:
    entries = []
    current_marker = ""
    current_lines: list[str] = []
    for raw_line in reference_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\[?(\d{1,4})\]?\s+(.+)$", line)
        if match:
            if current_lines:
                entries.append((current_marker, " ".join(current_lines)))
            current_marker = match.group(1)
            current_lines = [match.group(2)]
        elif current_lines:
            current_lines.append(line)
    if current_lines:
        entries.append((current_marker, " ".join(current_lines)))
    return entries


def _guess_reference_title(raw_reference: str) -> str:
    text = _clean_text(raw_reference)
    year_split = re.split(r"\b(?:19|20)\d{2}\b[. ]*", text, maxsplit=1)
    if len(year_split) == 2 and year_split[1]:
        candidate = year_split[1]
    else:
        candidate = text
    candidate = re.split(r"\b(?:In Proceedings|arXiv|ACM|IEEE|doi:|https?://)\b", candidate, maxsplit=1)[0]
    candidate = re.sub(r"[*_`]+", "", candidate)
    candidate = candidate.strip(" .:-")
    if len(candidate) < 8:
        return ""
    return candidate


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
