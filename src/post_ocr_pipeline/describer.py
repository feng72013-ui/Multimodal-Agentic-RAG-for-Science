from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path

from .cleaner import normalize_text
from .models import ImageAsset


def heuristic_description(asset: ImageAsset) -> str:
    context = normalize_text(
        "\n".join([asset.caption, *asset.reference_texts, asset.context_before, asset.context_after])
    )
    if asset.category == "Table":
        prefix = f"{asset.label or 'Table'} extracted from the OCR layout"
    else:
        prefix = f"{asset.label or 'Visual element'} extracted from the OCR layout"
    if context:
        return f"{prefix}. Nearby paper context: {context[:700]}"
    return f"{prefix} on page {asset.page_no}."


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as reader:
        encoded = base64.b64encode(reader.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def description_prompt(asset: ImageAsset) -> str:
    reference_text = "\n".join(f"- {text}" for text in asset.reference_texts)
    prompt = (
        "You are describing a visual element from an academic paper for a multimodal RAG knowledge base.\n"
        "Use the image itself first. Use the caption and cited sentences as grounding evidence.\n"
        "Return a concise English description under 140 words. Mention the figure/table label if available, "
        "the visual type, what it shows, important axes/columns/variables, and how it supports the paper.\n"
        "If the evidence and image conflict, say what is visible and avoid guessing.\n\n"
        f"Label: {asset.label or 'unknown'}\n"
        f"OCR category: {asset.category}\n"
        f"Caption candidate:\n{asset.caption or 'none'}\n\n"
        f"Paper sentences that cite this label:\n{reference_text or 'none'}\n\n"
        f"Local preceding text:\n{asset.context_before[:900] or 'none'}\n\n"
        f"Local following text:\n{asset.context_after[:900] or 'none'}"
    )
    return prompt


def model_description_openai_compatible(asset: ImageAsset, model: str, base_url: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(asset.image_path)}},
                    {"type": "text", "text": description_prompt(asset)},
                ],
            }
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def model_description_zhipu(asset: ImageAsset, model: str, api_key: str, endpoint: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(asset.image_path)}},
                    {"type": "text", "text": description_prompt(asset)},
                ],
            }
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0.1,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Zhipu API HTTP {exc.code}: {detail[:500]}") from exc
    return data["choices"][0]["message"]["content"] or ""


def describe_assets(assets: list[ImageAsset], use_model: bool = False, provider: str = "heuristic") -> list[ImageAsset]:
    provider = provider.lower()
    model = os.getenv("MULTIMODAL_LLM_MODEL", "")
    base_url = os.getenv("MULTIMODAL_LLM_BASE_URL", "")
    api_key = os.getenv("MULTIMODAL_LLM_API_KEY") or os.getenv("ZHIPU_API_KEY") or "0"
    zhipu_endpoint = os.getenv(
        "ZHIPU_API_ENDPOINT",
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    )

    if provider == "zhipu":
        model = model or "glm-4.6v-flash"
    elif provider == "openai-compatible":
        model = model or "qwen3-vl-plus"

    for asset in assets:
        if use_model and provider == "zhipu" and api_key != "0":
            try:
                asset.description = normalize_text(model_description_zhipu(asset, model, api_key, zhipu_endpoint))
                continue
            except Exception as exc:
                fallback = heuristic_description(asset)
                asset.description = f"{fallback} Model description failed: {type(exc).__name__}: {exc}"
                continue

        if use_model and provider == "openai-compatible" and model and base_url:
            try:
                asset.description = normalize_text(model_description_openai_compatible(asset, model, base_url, api_key))
                continue
            except Exception as exc:
                fallback = heuristic_description(asset)
                asset.description = f"{fallback} Model description failed: {type(exc).__name__}: {exc}"
                continue
        asset.description = heuristic_description(asset)
    return assets


def write_descriptions(path: Path, assets: list[ImageAsset]) -> None:
    with path.open("w", encoding="utf-8") as writer:
        for asset in assets:
            writer.write(json.dumps(asset.__dict__, ensure_ascii=False) + "\n")
