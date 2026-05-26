from __future__ import annotations

import json
from pathlib import Path


def load_vector_cache(cache_path: Path) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}

    cache: dict[str, list[float]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = payload.get("key")
            vector = payload.get("vector")
            if isinstance(key, str) and isinstance(vector, list):
                cache[key] = vector
    return cache


def append_vector_cache(cache_path: Path, item: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
