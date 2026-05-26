#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graph.intent_node import classify_task


DEFAULT_QUERIES = Path(__file__).resolve().parent / "sample_intents.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic task intent routing.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.queries)
    results = []
    correct = 0
    for row in rows:
        result = classify_task(row["query"], row.get("input_type"))
        ok = result.task_type == row["expected_task_type"]
        correct += int(ok)
        results.append(
            {
                "query_id": row["query_id"],
                "expected": row["expected_task_type"],
                "actual": result.task_type,
                "confidence": result.confidence,
                "ok": ok,
                "signals": result.signals,
            }
        )

    accuracy = correct / len(rows) if rows else 0.0
    by_task = summarize_by_task(rows, results)
    print(f"queries: {len(rows)}")
    print(f"accuracy: {accuracy:.3f}")
    for task_type, stats in by_task.items():
        print(f"- {task_type}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.3f})")

    failures = [item for item in results if not item["ok"]]
    if failures:
        print("failures:")
        for failure in failures:
            print(json.dumps(failure, ensure_ascii=False))

    return 0 if accuracy >= args.min_accuracy else 1


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return rows


def summarize_by_task(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for row, result in zip(rows, results):
        task_type = row["expected_task_type"]
        stats = summary.setdefault(task_type, {"total": 0, "correct": 0, "accuracy": 0.0})
        stats["total"] += 1
        stats["correct"] += int(result["ok"])
    for stats in summary.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    return summary


if __name__ == "__main__":
    raise SystemExit(main())

