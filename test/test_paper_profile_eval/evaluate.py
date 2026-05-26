#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "processed_rag" / "paper_profiles.jsonl"
DEFAULT_CITATION_PATH = PROJECT_ROOT / "processed_rag" / "citation_edges.jsonl"


REQUIRED_FIELDS = (
    "title",
    "abstract",
    "research_problem",
    "methods",
    "results",
    "conclusions",
    "quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 paper profile completeness.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATION_PATH)
    parser.add_argument("--min-field-fill-rate", type=float, default=0.80)
    parser.add_argument("--min-avg-quality", type=float, default=0.60)
    parser.add_argument("--min-citation-edges", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = load_jsonl(args.profiles)
    citations = load_jsonl(args.citations) if args.citations.exists() else []

    if not profiles:
        print(f"No profiles found: {args.profiles}")
        return 1

    fill_rates = {
        field_name: filled_count(profiles, field_name) / len(profiles)
        for field_name in REQUIRED_FIELDS
    }
    quality_scores = [float((profile.get("quality") or {}).get("score") or 0.0) for profile in profiles]
    avg_quality = sum(quality_scores) / len(quality_scores)
    min_fill_rate = min(fill_rates.values())

    print(f"profiles: {len(profiles)}")
    print(f"citation_edges: {len(citations)}")
    print(f"avg_quality: {avg_quality:.3f}")
    for field_name, fill_rate in fill_rates.items():
        print(f"- {field_name}: {fill_rate:.3f}")

    ok = (
        min_fill_rate >= args.min_field_fill_rate
        and avg_quality >= args.min_avg_quality
        and len(citations) >= args.min_citation_edges
    )
    if not ok:
        print(
            "FAILED: "
            f"min_fill_rate={min_fill_rate:.3f}, "
            f"avg_quality={avg_quality:.3f}, "
            f"citation_edges={len(citations)}"
        )
    return 0 if ok else 1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filled_count(rows: list[dict[str, Any]], field_name: str) -> int:
    count = 0
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, list):
            count += int(bool(value))
        elif isinstance(value, dict):
            count += int(bool(value))
        else:
            count += int(value not in (None, ""))
    return count


if __name__ == "__main__":
    raise SystemExit(main())

