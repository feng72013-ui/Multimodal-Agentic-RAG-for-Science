from __future__ import annotations

import re


BASE64_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(data:image/(?P<ext>[^;)\s]+);base64,(?P<data>.*?)\)",
    re.DOTALL,
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def strip_base64_images(markdown: str) -> str:
    return BASE64_IMAGE_RE.sub("[IMAGE]", markdown)


def is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) <= 2 and stripped.isdigit():
        return True
    return False


def markdown_heading_level(text: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+", text.strip())
    if not match:
        return None
    return len(match.group(1))


def heading_text(text: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", text.strip()).strip()

