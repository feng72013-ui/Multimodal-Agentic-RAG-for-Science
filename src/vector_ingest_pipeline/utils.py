from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                yield json.loads(line)


def truncate(value: str | None, max_len: int) -> str:
    value = value or ""
    if len(value.encode("utf-8")) <= max_len:
        return value
    suffix = "..."
    suffix_len = len(suffix.encode("utf-8"))
    if max_len <= suffix_len:
        return value.encode("utf-8")[:max_len].decode("utf-8", errors="ignore")

    budget = max_len - suffix_len
    truncated = value.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    return truncated + suffix


def utf8_len(value: str | None) -> int:
    return len((value or "").encode("utf-8"))


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as reader:
        encoded = base64.b64encode(reader.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


class FixedWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.window_start = time.monotonic()
        self.count = 0

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self.window_start
        if elapsed >= self.window_seconds:
            self.window_start = now
            self.count = 0
        if self.count >= self.limit:
            sleep_seconds = self.window_seconds - elapsed
            if sleep_seconds > 0:
                print(f"[rate-limit] sleep {sleep_seconds:.2f}s")
                time.sleep(sleep_seconds)
            self.window_start = time.monotonic()
            self.count = 0
        self.count += 1


def zero_vector(dim: int) -> list[float]:
    return [0.0] * dim
