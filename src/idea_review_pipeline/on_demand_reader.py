from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from vector_ingest_pipeline.utils import truncate


DEFAULT_ON_DEMAND_TOP_K = 5

SECTION_HINTS = {
    "abstract": ("abstract",),
    "problem": ("introduction", "motivation", "background"),
    "method": ("method", "approach", "model", "framework", "architecture"),
    "experiment": ("experiment", "evaluation", "result", "analysis"),
    "conclusion": ("conclusion", "discussion"),
    "limitation": ("limitation", "future", "threat"),
}

FIELD_HINTS = {
    "research_problem": ("problem", "challenge", "limitation", "motivat", "address", "aim", "goal"),
    "methods": ("propose", "present", "introduce", "framework", "method", "model", "module", "approach"),
    "results": ("experiment", "result", "outperform", "improve", "achieve", "show", "demonstrate"),
    "limitations": ("limitation", "future work", "threat", "however", "fail", "cannot"),
}

DATASET_TERMS = (
    "movielens",
    "amazon",
    "yelp",
    "gowalla",
    "lastfm",
    "steam",
    "netflix",
    "taobao",
    "ijcai",
    "tianchi",
)

METRIC_TERMS = (
    "recall",
    "ndcg",
    "hit rate",
    "hr@",
    "mrr",
    "auc",
    "precision",
    "map",
    "f1",
    "robustness",
    "fairness",
)


def build_task_paper_profiles(
    idea: str,
    related_work: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    top_k: int = DEFAULT_ON_DEMAND_TOP_K,
    use_llm: bool = False,
) -> list[dict[str, Any]]:
    """Create task-specific paper profiles only for papers retrieved for the current idea."""
    if not related_work or not chunks or top_k <= 0:
        return []

    grouped = group_chunks_by_key(chunks)
    task_profiles = []
    for related in related_work[:top_k]:
        key = paper_lookup_key(related)
        paper_chunks = grouped.get(key, [])
        if not paper_chunks:
            continue
        evidence_chunks = select_task_chunks(idea, paper_chunks)
        if use_llm:
            task_profile = llm_task_profile(idea, related, evidence_chunks)
        else:
            task_profile = deterministic_task_profile(idea, related, evidence_chunks)
        task_profiles.append(task_profile)
    return task_profiles


def group_chunks_by_key(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[paper_lookup_key(chunk)].append(chunk)
    for values in grouped.values():
        values.sort(key=lambda item: (int(item.get("page_start") or 0), str(item.get("chunk_id") or "")))
    return grouped


def paper_lookup_key(row: dict[str, Any]) -> str:
    return f"{row.get('topic') or ''}::{row.get('paper_id') or ''}"


def select_task_chunks(idea: str, chunks: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    idea_terms = token_set(idea)
    scored = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        title_path = " ".join(str(item) for item in (chunk.get("title_path") or []))
        score = overlap_score(idea_terms, text)
        section = detect_section(title_path, text)
        if section in {"abstract", "problem", "method", "experiment", "conclusion"}:
            score += 0.15
        if section == "abstract":
            score += 0.1
        scored.append((score, int(chunk.get("page_start") or 0), chunk))
    selected = [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]
    return selected


def deterministic_task_profile(
    idea: str,
    related: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    sections = classify_chunks(evidence_chunks)
    return {
        "paper_id": related.get("paper_id"),
        "title": related.get("title"),
        "topic": related.get("topic"),
        "source_pdf": related.get("source_pdf"),
        "reader": "deterministic_on_demand",
        "task_relevance": infer_task_relevance(idea, evidence_chunks),
        "research_problem": select_sentences(sections["abstract"] + sections["problem"], FIELD_HINTS["research_problem"], idea, 3),
        "methods": select_sentences(sections["method"] + sections["abstract"], FIELD_HINTS["methods"], idea, 4),
        "datasets": extract_terms(evidence_chunks, DATASET_TERMS),
        "metrics": extract_terms(evidence_chunks, METRIC_TERMS),
        "results": select_sentences(sections["experiment"] + sections["abstract"], FIELD_HINTS["results"], idea, 4),
        "limitations": select_sentences(sections["limitation"] + sections["conclusion"], FIELD_HINTS["limitations"], idea, 3),
        "evidence": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "section": detect_section(" ".join(str(item) for item in (chunk.get("title_path") or [])), chunk.get("text") or ""),
                "text": truncate(chunk.get("text"), 500),
                "source_pdf": (chunk.get("metadata") or {}).get("source_pdf"),
                "image_path": (chunk.get("metadata") or {}).get("image_path"),
                "score": (chunk.get("metadata") or {}).get("score"),
            }
            for chunk in evidence_chunks[:6]
        ],
    }


def llm_task_profile(
    idea: str,
    related: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    from project.recommendate_project.myllm import llm

    system = SystemMessage(
        content=(
            "You are a task-focused paper reader. Extract only information supported by the evidence chunks. "
            "Focus on how the paper relates to the user's research idea. Return valid JSON only."
        )
    )
    payload = {
        "idea": idea,
        "paper": {
            "paper_id": related.get("paper_id"),
            "title": related.get("title"),
            "topic": related.get("topic"),
        },
        "evidence_chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "page_start": chunk.get("page_start"),
                "title_path": chunk.get("title_path"),
                "text": truncate(chunk.get("text"), 1200),
            }
            for chunk in evidence_chunks
        ],
    }
    prompt = (
        "Return JSON with keys: task_relevance, research_problem, methods, datasets, metrics, "
        "results, limitations, evidence_quotes. Arrays should contain concise strings. "
        f"\n\nInput:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    response = llm.invoke([system, HumanMessage(content=prompt)])
    parsed = parse_json_response(str(response.content))
    parsed.update(
        {
            "paper_id": related.get("paper_id"),
            "title": related.get("title"),
            "topic": related.get("topic"),
            "source_pdf": related.get("source_pdf"),
            "reader": "llm_on_demand",
            "evidence": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "page_start": chunk.get("page_start"),
                    "text": truncate(chunk.get("text"), 350),
                    "source_pdf": (chunk.get("metadata") or {}).get("source_pdf"),
                    "image_path": (chunk.get("metadata") or {}).get("image_path"),
                    "score": (chunk.get("metadata") or {}).get("score"),
                }
                for chunk in evidence_chunks[:5]
            ],
        }
    )
    return parsed


def classify_chunks(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    sections = {name: [] for name in ("abstract", "problem", "method", "experiment", "conclusion", "limitation", "other")}
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        title_path = " ".join(str(item) for item in (chunk.get("title_path") or []))
        sections[detect_section(title_path, text)].append(text)
    return sections


def detect_section(title_path: str, text: str) -> str:
    haystack = f"{title_path}\n{text[:240]}".lower()
    for section, hints in SECTION_HINTS.items():
        if any(hint in haystack for hint in hints):
            return section
    return "other"


def select_sentences(texts: list[str], hints: tuple[str, ...], idea: str, limit: int) -> list[str]:
    idea_terms = token_set(idea)
    candidates = []
    for text in texts:
        for sentence in split_sentences(text):
            lower = sentence.lower()
            hint_score = sum(1 for hint in hints if hint in lower)
            relevance = overlap_score(idea_terms, sentence)
            score = hint_score * 0.2 + relevance
            if score > 0:
                candidates.append((score, sentence))
    output = []
    seen = set()
    for _, sentence in sorted(candidates, key=lambda item: item[0], reverse=True):
        key = re.sub(r"\W+", "", sentence.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        output.append(truncate(sentence, 450))
        if len(output) >= limit:
            break
    return output


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [part.strip() for part in parts if 40 <= len(part.strip()) <= 900]


def extract_terms(chunks: list[dict[str, Any]], terms: tuple[str, ...]) -> list[str]:
    haystack = " ".join(str(chunk.get("text") or "").lower() for chunk in chunks)
    return sorted({term for term in terms if term in haystack})


def infer_task_relevance(idea: str, chunks: list[dict[str, Any]]) -> str:
    score = max((overlap_score(token_set(idea), chunk.get("text") or "") for chunk in chunks), default=0.0)
    if score >= 0.18:
        return "high"
    if score >= 0.08:
        return "medium"
    return "low"


def token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}|[\u4e00-\u9fff]{2,}", (text or "").lower())
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "into", "using", "研究", "系统", "方法"}
    return {token for token in tokens if token not in stopwords}


def overlap_score(query_terms: set[str], text: str) -> float:
    text_terms = token_set(text)
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / max(len(query_terms), 1)


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))
