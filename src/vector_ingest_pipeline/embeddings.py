from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any

from .utils import FixedWindowRateLimiter, image_to_data_url


class LocalQwenTextEmbedder:
    """Local Qwen3 text embedding wrapper, mirroring the reference project's SentenceTransformer usage."""

    def __init__(self, model_path: Path, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(str(model_path), **kwargs)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


class DashScopeMultimodalEmbedder:
    """DashScope multimodal embedding wrapper used for image+text records."""

    def __init__(self, api_key: str, model: str = "qwen3-vl-embedding", rpm: int = 120):
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for multimodal embedding.")
        import dashscope

        self.dashscope = dashscope
        self.api_key = api_key
        self.model = model
        self.limiter = FixedWindowRateLimiter(rpm)

    def embed_text(self, text: str) -> list[float]:
        return self._call([{"text": text}])

    def embed_image_text(self, image_path: str, text: str) -> list[float]:
        return self._call([{"image": image_to_data_url(image_path), "text": text}])

    def _call(self, input_data: list[dict[str, str]]) -> list[float]:
        self.limiter.acquire()
        response = self.dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=input_data,
            api_key=self.api_key,
        )
        status = getattr(response, "status_code", None)
        if status != HTTPStatus.OK:
            code = getattr(response, "code", "")
            message = getattr(response, "message", "")
            if status == HTTPStatus.FORBIDDEN and code == "AllocationQuota.FreeTierOnly":
                raise RuntimeError(
                    "DashScope embedding quota rejected the request. "
                    "The current model/free-tier allocation is exhausted or restricted. "
                    f"model={self.model}, code={code}, message={message}"
                )
            raise RuntimeError(f"DashScope embedding failed: status={status}, code={code}, message={message}")
        return response.output["embeddings"][0]["embedding"]
