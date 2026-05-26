#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from profile_store_pipeline.common import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_PROFILE_PATH,
    DEFAULT_REVIEW_PATH,
    group_chunks_by_paper,
    paper_key,
    read_jsonl,
    write_jsonl,
)
from vector_ingest_pipeline.utils import truncate


REQUIRED_REVIEW_FIELDS = {
    "faithfulness_score",
    "field_accuracy",
    "missing_fields",
    "wrong_or_unsupported_claims",
    "citation_quality",
    "rewrite_suggestions",
    "pass",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review paper profile quality with an evidence-grounded LLM judge.")
    parser.add_argument("--profiles", default=DEFAULT_PROFILE_PATH, type=str)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH, type=str)
    parser.add_argument("--output", default=DEFAULT_REVIEW_PATH, type=str)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between LLM calls.")
    parser.add_argument("--min-pass-score", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and deterministic checks without LLM calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = read_jsonl(Path(args.profiles))
    chunks = read_jsonl(Path(args.chunks))
    grouped_chunks = group_chunks_by_paper(chunks)

    rows = []
    selected_profiles = profiles[: args.limit] if args.limit else profiles
    for index, profile in enumerate(selected_profiles, start=1):
        evidence_chunks = select_evidence_chunks(profile, grouped_chunks.get(paper_key(profile), []))
        if args.dry_run:
            review = deterministic_review(profile, evidence_chunks, min_pass_score=args.min_pass_score)
            review["dry_run_prompt_preview"] = build_prompt(profile, evidence_chunks)[1][:1200]
        else:
            review = review_with_llm(profile, evidence_chunks, min_pass_score=args.min_pass_score)
            if args.sleep > 0 and index < len(selected_profiles):
                time.sleep(args.sleep)
        rows.append(review)
        print(
            f"[quality] {index}/{len(selected_profiles)} "
            f"{profile.get('paper_id')}: pass={review.get('pass')} score={review.get('faithfulness_score')}"
        )

    write_jsonl(Path(args.output), rows)
    print(f"Quality reviews: {args.output}")
    return 0


def select_evidence_chunks(profile: dict[str, Any], chunks: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    if not chunks:
        return []
    wanted_sections = ("abstract", "introduction", "method", "experiment", "conclusion", "limitation")
    scored = []
    for chunk in chunks:
        title_path = " ".join(str(item).lower() for item in (chunk.get("title_path") or []))
        text = str(chunk.get("text") or "")
        score = 0
        score += sum(2 for section in wanted_sections if section in title_path)
        lower_text = text.lower()
        score += int("abstract" in lower_text)
        score += int("conclusion" in lower_text)
        score += int("experiment" in lower_text or "result" in lower_text)
        scored.append((score, int(chunk.get("page_start") or 0), chunk))
    selected = [item[2] for item in sorted(scored, key=lambda value: (-value[0], value[1]))[:limit]]
    return selected


def deterministic_review(
    profile: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
    *,
    min_pass_score: float,
) -> dict[str, Any]:
    quality_score = float((profile.get("quality") or {}).get("score") or 0.0)
    has_evidence = bool(evidence_chunks)
    score = min(1.0, quality_score * (1.0 if has_evidence else 0.75))
    missing_fields = [
        field
        for field in ("abstract", "research_problem", "methods", "results", "conclusions")
        if not profile.get(field)
    ]
    return normalize_review(
        profile,
        {
            "faithfulness_score": round(score, 3),
            "field_accuracy": {
                "abstract": "not_checked" if has_evidence else "missing_evidence",
                "research_problem": "not_checked" if has_evidence else "missing_evidence",
                "methods": "not_checked" if has_evidence else "missing_evidence",
                "results": "not_checked" if has_evidence else "missing_evidence",
                "conclusions": "not_checked" if has_evidence else "missing_evidence",
            },
            "missing_fields": missing_fields,
            "wrong_or_unsupported_claims": [],
            "citation_quality": "not_checked",
            "rewrite_suggestions": [],
            "pass": score >= min_pass_score and not missing_fields and has_evidence,
        },
        reviewer="deterministic",
    )


def review_with_llm(
    profile: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
    *,
    min_pass_score: float,
) -> dict[str, Any]:
    from myllm import glm4_flash as llm

    system_prompt, user_prompt = build_prompt(profile, evidence_chunks)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    parsed = parse_json_response(str(response.content))
    parsed["pass"] = bool(parsed.get("pass")) and float(parsed.get("faithfulness_score") or 0.0) >= min_pass_score
    return normalize_review(profile, parsed, reviewer="llm")


def build_prompt(profile: dict[str, Any], evidence_chunks: list[dict[str, Any]]) -> tuple[str, str]:
    system_prompt = (
        "You are a strict research-paper profile quality assessor. "
        "Judge only whether the provided structured profile is supported by the evidence chunks. "
        "Do not use outside knowledge. Return valid JSON only."
    )
    profile_payload = {
        "title": profile.get("title"),
        "abstract": profile.get("abstract"),
        "research_problem": profile.get("research_problem"),
        "methods": profile.get("methods"),
        "datasets": profile.get("datasets"),
        "metrics": profile.get("metrics"),
        "results": profile.get("results"),
        "conclusions": profile.get("conclusions"),
        "limitations": profile.get("limitations"),
    }
    evidence_payload = [
        {
            "chunk_id": chunk.get("chunk_id") or chunk.get("doc_id"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "title_path": chunk.get("title_path"),
            "text": truncate(chunk.get("text"), 1200),
        }
        for chunk in evidence_chunks
    ]
    user_prompt = f"""
Assess the profile against the evidence.

Return JSON with exactly these keys:
- faithfulness_score: number from 0 to 1.
- field_accuracy: object with keys abstract, research_problem, methods, results, conclusions, limitations. Values: supported, partially_supported, unsupported, missing.
- missing_fields: array of field names.
- wrong_or_unsupported_claims: array of short strings.
- citation_quality: one of not_checked, poor, partial, good.
- rewrite_suggestions: array of short strings.
- pass: boolean.

Profile:
{json.dumps(profile_payload, ensure_ascii=False, indent=2)}

Evidence chunks:
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}
"""
    return system_prompt, user_prompt


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    missing = REQUIRED_REVIEW_FIELDS - set(parsed)
    if missing:
        raise ValueError(f"LLM review JSON missing fields: {sorted(missing)}")
    return parsed


def normalize_review(profile: dict[str, Any], review: dict[str, Any], *, reviewer: str) -> dict[str, Any]:
    return {
        "topic": profile.get("topic"),
        "paper_id": profile.get("paper_id"),
        "title": profile.get("title"),
        "reviewer": reviewer,
        "faithfulness_score": float(review.get("faithfulness_score") or 0.0),
        "field_accuracy": review.get("field_accuracy") or {},
        "missing_fields": review.get("missing_fields") or [],
        "wrong_or_unsupported_claims": review.get("wrong_or_unsupported_claims") or [],
        "citation_quality": review.get("citation_quality") or "not_checked",
        "rewrite_suggestions": review.get("rewrite_suggestions") or [],
        "pass": bool(review.get("pass")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
