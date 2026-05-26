#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_agents.orchestrator import REQUIRED_AGENT_ROLES, run_research_workflow
from vector_ingest_pipeline.utils import iter_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = Path(__file__).resolve().parent / "sample_tasks.jsonl"
DEFAULT_PROFILES = PROJECT_ROOT / "processed_rag" / "paper_profiles.jsonl"
REQUIRED_SCORE_KEYS = ("novelty", "feasibility", "academic_value", "risk", "overall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage 4 multi-agent orchestration.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-related", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(iter_jsonl(args.tasks))
    profiles = list(iter_jsonl(args.profiles)) if args.profiles.exists() else []
    results = []
    passed = 0

    for row in tasks:
        result = run_research_workflow(
            row["query"],
            task_type=row.get("task_type", "idea_review"),
            profiles=profiles,
            top_k=args.top_k,
            use_milvus=False,
        )
        checks = validate_result(result, min_related=int(row.get("min_related", args.min_related)))
        ok = all(checks.values())
        passed += int(ok)
        results.append(
            {
                "task_id": row["task_id"],
                "ok": ok,
                "checks": checks,
                "roles": [step.get("role") for step in result.get("agent_trace", [])],
                "status": result.get("status"),
                "scores": result.get("scores", {}),
            }
        )

    pass_rate = passed / len(tasks) if tasks else 0.0
    print(f"tasks: {len(tasks)}")
    print(f"pass_rate: {pass_rate:.3f}")
    print(f"passed: {passed}/{len(tasks)}")
    failures = [result for result in results if not result["ok"]]
    if failures:
        print("failures:")
        for failure in failures:
            print(json.dumps(failure, ensure_ascii=False))
    return 0 if pass_rate >= args.min_pass_rate else 1


def validate_result(result: dict[str, Any], min_related: int) -> dict[str, bool]:
    roles = [step.get("role") for step in result.get("agent_trace", [])]
    scores = result.get("scores") or {}
    outputs = result.get("agent_outputs") or {}
    return {
        "status_valid": result.get("status") in {"completed", "degraded"},
        "has_all_roles": all(role in roles for role in REQUIRED_AGENT_ROLES),
        "has_supervisor_plan": bool(result.get("supervisor_plan")),
        "has_agent_outputs": all(
            key in outputs
            for key in ("supervisor", "retriever", "paper_analyst", "idea_critic", "research_planner", "writer")
        ),
        "has_final_report": len(result.get("final_report") or "") >= 500,
        "has_related_work": len(result.get("related_work") or []) >= min_related,
        "has_similarity_matrix": len(result.get("similarity_matrix") or []) >= min_related,
        "has_scores": all(key in scores for key in REQUIRED_SCORE_KEYS),
        "scores_in_range": all(0.0 <= float(scores.get(key, -1)) <= 1.0 for key in REQUIRED_SCORE_KEYS),
        "has_next_steps": bool((outputs.get("research_planner") or {}).get("next_steps")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
