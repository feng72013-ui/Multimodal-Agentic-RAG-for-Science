from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalQuery:
    query_id: str
    query: str
    query_type: str = "semantic"
    expected_terms: list[str] = field(default_factory=list)
    expected_category: str | None = None
    expected_topic: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalQuery":
        return cls(
            query_id=str(payload["query_id"]),
            query=str(payload["query"]),
            query_type=str(payload.get("query_type", "semantic")),
            expected_terms=[str(item) for item in payload.get("expected_terms", [])],
            expected_category=payload.get("expected_category"),
            expected_topic=payload.get("expected_topic"),
            notes=str(payload.get("notes", "")),
        )


def load_queries(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                queries.append(EvalQuery.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
    return queries
