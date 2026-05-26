#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from idea_review_pipeline.review import review_idea
from vector_ingest_pipeline.utils import iter_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDEAS = Path(__file__).resolve().parent / "sample_ideas.jsonl"
DEFAULT_PROFILES = PROJECT_ROOT / "processed_rag" / "paper_profiles.jsonl"

REQUIRED_SCORE_KEYS = ("novelty", "feasibility", "academic_value", "risk", "overall")
REQUIRED_REPORT_KEYS = (
    "idea_profile",
    "related_work",
    "similarity_matrix",
    "scores",
    "gaps",
    "innovation_suggestions",
    "experiment_suggestions",
    "risk_assessment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate idea review report completeness.")
    parser.add_argument("--ideas", type=Path, default=DEFAULT_IDEAS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-related", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ideas = list(iter_jsonl(args.ideas))
    profiles = list(iter_jsonl(args.profiles))
    results = []
    passed = 0

    for row in ideas:
        review = review_idea(row["idea"], profiles, top_k=args.top_k)
        checks = validate_review(review, min_related=int(row.get("min_related", args.min_related)))
        ok = all(checks.values())
        passed += int(ok)
        results.append(
            {
                "idea_id": row["idea_id"],
                "ok": ok,
                "checks": checks,
                "top_related": [item.get("title") for item in review.get("related_work", [])[:3]],
                "scores": review.get("scores", {}),
            }
        )

    pass_rate = passed / len(ideas) if ideas else 0.0
    print(f"ideas: {len(ideas)}")
    print(f"pass_rate: {pass_rate:.3f}")
    print(f"passed: {passed}/{len(ideas)}")
    failures = [result for result in results if not result["ok"]]
    if failures:
        print("failures:")
        for failure in failures:
            print(json.dumps(failure, ensure_ascii=False))
    return 0 if pass_rate >= args.min_pass_rate else 1


def validate_review(review: dict[str, Any], min_related: int) -> dict[str, bool]:
    scores = review.get("scores") or {}
    return {
        "has_required_keys": all(key in review for key in REQUIRED_REPORT_KEYS),
        "has_idea_terms": bool((review.get("idea_profile") or {}).get("terms")),
        "has_related_work": len(review.get("related_work") or []) >= min_related,
        "has_similarity_matrix": len(review.get("similarity_matrix") or []) >= min_related,
        "has_scores": all(key in scores for key in REQUIRED_SCORE_KEYS),
        "scores_in_range": all(0.0 <= float(scores.get(key, -1)) <= 1.0 for key in REQUIRED_SCORE_KEYS),
        "has_gaps": bool(review.get("gaps")),
        "has_innovation_suggestions": bool(review.get("innovation_suggestions")),
        "has_experiment_suggestions": bool(review.get("experiment_suggestions")),
        "has_risk_assessment": bool(review.get("risk_assessment")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
